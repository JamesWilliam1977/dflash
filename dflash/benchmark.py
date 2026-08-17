from __future__ import annotations

import argparse
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from types import SimpleNamespace
from typing import Any

import requests
from tqdm import tqdm

DATASETS = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```".format(**x),
    },
    "mbpp": {
        "load_args": ("google-research-datasets/mbpp", "sanitized"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: x["prompt"],
    },
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: x["prompt"],
        "multi_turn": True,
    },
}


def load_and_process_dataset(data_name: str) -> list[dict]:
    from datasets import load_dataset

    if data_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{data_name}'. Available: {list(DATASETS.keys())}")

    cfg = DATASETS[data_name]
    dataset = load_dataset(*cfg["load_args"], **cfg["load_kwargs"])
    return [
        {"turns": cfg["format"](row) if cfg.get("multi_turn") else [cfg["format"](row)]}
        for row in dataset
    ]


def _select_dataset(
    dataset: list[dict], count: int | None, *, repeat: bool = False,
) -> list[dict]:
    order = list(range(len(dataset)))
    random.Random(42).shuffle(order)
    count = len(order) if count is None else count
    if not repeat:
        count = min(count, len(order))
    return [dataset[order[i % len(order)]] for i in range(count)]


def _reasoning_kwargs(enable_thinking: bool, reasoning_level: str | None) -> dict:
    kwargs = {"enable_thinking": enable_thinking}
    if reasoning_level is not None:
        kwargs.update(reasoning_effort=reasoning_level, reasoning_strength=reasoning_level)
    return kwargs


def _apply_chat_template(
    tokenizer, messages: list[dict], enable_thinking: bool, reasoning_level: str | None,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **_reasoning_kwargs(enable_thinking, reasoning_level),
    )


def _make_decode_metrics(num_output_tokens: int, generation_tps: float, acceptance_lengths: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        num_output_tokens=num_output_tokens,
        time_per_output_token=1.0 / generation_tps if generation_tps > 0 else float("inf"),
        acceptance_lengths=acceptance_lengths,
    )


def _print_decode_summary(responses: list[dict[int, SimpleNamespace]], block_size: int) -> None:
    baseline_tpot = statistics.mean(r[1].time_per_output_token for r in responses)
    dflash_tpot = statistics.mean(r[block_size].time_per_output_token for r in responses)
    print(f"Baseline throughput: {1 / baseline_tpot:.2f} tok/s")
    print(f"DFlash throughput:  {1 / dflash_tpot:.2f} tok/s")
    print(f"Decoding speedup: {baseline_tpot / dflash_tpot:.2f}")

    per_request = [
        r[block_size].acceptance_lengths
        for r in responses
        if r[block_size].acceptance_lengths
    ]
    acceptance_lengths = list(chain.from_iterable(r[block_size].acceptance_lengths for r in responses))
    if not acceptance_lengths:
        print("Average Acceptance length: n/a")
        return
    mean_accept = statistics.mean(statistics.mean(x) for x in per_request)
    print(f"Average Acceptance length: {mean_accept:.2f}")

    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _dist_init(torch_dist) -> None:
    if "RANK" not in os.environ:
        return
    torch_dist.init_process_group(backend="nccl", init_method="env://")


def _dist_size() -> int:
    return _env_int("WORLD_SIZE", 1)


def _dist_rank() -> int:
    return _env_int("RANK", 0)


def _dist_local_rank() -> int:
    return _env_int("LOCAL_RANK", 0)


def _dist_is_main() -> bool:
    return _dist_rank() == 0


def _dist_gather(torch_dist, obj: Any, dst: int = 0):
    if not torch_dist.is_initialized():
        return [obj]
    if _dist_is_main():
        objs = [None] * _dist_size()
        torch_dist.gather_object(obj, objs, dst=dst)
        return objs
    torch_dist.gather_object(obj, dst=dst)
    return None


def _run_transformers(args: argparse.Namespace) -> None:
    import torch
    from torch import distributed as torch_dist
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    from .model import DFlash2DraftModel, DFlashDraftModel, dflash_generate

    torch.manual_seed(0)

    _dist_init(torch_dist)
    torch.cuda.set_device(_dist_local_rank())
    device = torch.device(f"cuda:{_dist_local_rank()}")
    target_kwargs = {"attn_implementation": "sdpa", "dtype": torch.bfloat16}
    try:
        target = AutoModelForCausalLM.from_pretrained(args.model, **target_kwargs)
    except ValueError:
        target = AutoModelForImageTextToText.from_pretrained(args.model, **target_kwargs)
    target = target.to(device).eval()

    draft_config = AutoConfig.from_pretrained(args.draft_model)
    architectures = draft_config.architectures or []
    draft_class = DFlash2DraftModel if "DFlash2DraftModel" in architectures else DFlashDraftModel
    draft_model = draft_class.from_pretrained(
        args.draft_model, attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_and_process_dataset(args.dataset)

    dataset = _select_dataset(dataset, args.max_samples)

    warmup_text = _apply_chat_template(
        tokenizer,
        [{"role": "user", "content": dataset[0]["turns"][0]}],
        args.enable_thinking,
        args.reasoning_level,
    )
    warmup = tokenizer.encode(warmup_text, return_tensors="pt").to(device)
    warmup_tokens = min(64, args.max_new_tokens)
    for bs in (1, block_size):
        dflash_generate(
            draft_model, target, warmup, warmup_tokens, None,
            args.temperature, args.top_p, args.top_k, block_size=bs,
        )

    responses = []
    indices = range(_dist_rank(), len(dataset), _dist_size())
    for idx in tqdm(indices, disable=not _dist_is_main()):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = _apply_chat_template(
                tokenizer, messages, args.enable_thinking, args.reasoning_level,
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            for bs in [1, block_size]:
                response[bs] = dflash_generate(
                    draft_model,
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    block_size=bs,
                    return_stats=True,
                )

            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if _dist_size() > 1:
        responses = _dist_gather(torch_dist, responses, dst=0)
        if not _dist_is_main():
            return
        responses = list(chain(*responses))

    _print_decode_summary(responses, block_size)


def _send_openai(
    base_url: str,
    messages: list[dict],
    *,
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout_s: int,
    enable_thinking: bool = False,
    reasoning_level: str | None = None,
) -> dict:
    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "chat_template_kwargs": _reasoning_kwargs(enable_thinking, reasoning_level),
        "return_meta_info": True,
    }
    if top_k > 0:
        body["top_k"] = top_k
    resp = requests.post(
        base_url + "/v1/chat/completions",
        json=body,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json()


def _run_mlx(args: argparse.Namespace) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import stream_generate as stream_generate_baseline

    from .model_mlx import load, load_draft, make_sampler, stream_generate

    mx.random.seed(0)
    sampler = make_sampler(args.temperature, args.top_p, args.top_k)

    print(f"Loading target: {args.model}")
    model, tokenizer = load(args.model)
    print(f"Loading draft: {args.draft_model}")
    draft = load_draft(args.draft_model)
    if args.draft_bits is not None:
        nn.quantize(draft, group_size=64, bits=args.draft_bits)
        mx.eval(draft.parameters())
    block_size = args.block_size if args.block_size is not None else int(draft.config.block_size)

    dataset = load_and_process_dataset(args.dataset)
    dataset = _select_dataset(dataset, args.max_samples)

    warmup_prompt = tokenizer.encode("Hi")
    list(stream_generate_baseline(model, tokenizer, warmup_prompt, 3, sampler=sampler))
    list(stream_generate(
        model, draft, tokenizer, warmup_prompt,
        block_size=block_size,
        max_tokens=3,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    ))

    responses = []
    for idx in tqdm(range(len(dataset))):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            prompt = _apply_chat_template(
                tokenizer, messages, args.enable_thinking, args.reasoning_level,
            )

            response = {}

            tokens_bl, tps_bl = [], 0
            for r in stream_generate_baseline(model, tokenizer, prompt, args.max_new_tokens, sampler=sampler):
                tokens_bl.append(r.token)
                tps_bl = r.generation_tps
            response[1] = _make_decode_metrics(len(tokens_bl), tps_bl, [1])

            tokens_df, accs, tps_df = [], [], 0
            for r in stream_generate(
                model, draft, tokenizer, prompt,
                block_size=block_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            ):
                tokens_df.extend(r.tokens)
                if r.accepted is not None:
                    accs.append(r.accepted)
                tps_df = r.generation_tps
            response[block_size] = _make_decode_metrics(len(tokens_df), tps_df, accs)

            output_text = tokenizer.decode(tokens_df)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    _print_decode_summary(responses, block_size)


def _run_openai(args: argparse.Namespace) -> None:
    bs = max(args.concurrency, 1)
    dataset = _select_dataset(
        load_and_process_dataset(args.dataset), args.num_prompts + bs, repeat=True,
    )
    prompts = [
        [{"role": "user", "content": item["turns"][0]}]
        for item in dataset[:args.num_prompts]
    ]
    warmup_prompts = [
        [{"role": "user", "content": item["turns"][0]}]
        for item in dataset[args.num_prompts:]
    ]

    def send_one(messages: list[dict], max_new_tokens=args.max_new_tokens) -> dict:
        return _send_openai(
            args.base_url,
            messages,
            model=args.model,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            timeout_s=args.timeout_s,
            enable_thinking=args.enable_thinking,
            reasoning_level=args.reasoning_level,
        )

    print(f"[warmup] {bs} requests ...")
    with ThreadPoolExecutor(max_workers=bs) as pool:
        list(pool.map(lambda p: send_one(p, min(64, args.max_new_tokens)), warmup_prompts))

    print(f"Running benchmark: {args.num_prompts} prompts, concurrency={bs} ...")
    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_accept_lengths: list[float] = []

    with ThreadPoolExecutor(max_workers=bs) as pool:
        futures = [pool.submit(send_one, p) for p in prompts]
        for fut in tqdm(as_completed(futures), total=len(prompts), desc="Benchmarking"):
            out = fut.result()
            usage = out.get("usage", {}) or {}
            total_tokens += int(usage.get("completion_tokens", 0))
            meta = out.get("meta_info", {}) or {}
            spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
            if "spec_accept_length" in meta:
                try:
                    spec_accept_lengths.append(float(meta["spec_accept_length"]))
                except (TypeError, ValueError):
                    pass

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    print(f"\n{'=' * 50}")
    print(f"Backend:          {args.backend}")
    print(f"Dataset:          {args.dataset}")
    print(f"Num prompts:      {args.num_prompts}")
    print(f"Concurrency:      {bs}")
    print(f"Latency:          {latency:.1f}s")
    print(f"Output tokens:    {total_tokens}")
    print(f"Throughput:       {toks_per_s:,.2f} tok/s")
    if spec_accept_lengths:
        print(f"Accept length:    {statistics.mean(spec_accept_lengths):.3f}")
    if spec_verify_ct_sum > 0:
        print(f"Spec verify ct:   {spec_verify_ct_sum}")
    print(f"{'=' * 50}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DFlash benchmark")
    parser.add_argument("--backend", choices=["transformers", "openai", "mlx"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--draft-bits", type=int, choices=[4, 8], default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--num-prompts", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--reasoning-level",
        type=str,
        default=None,
        help="Muse: low/medium/high/xhigh; Qwen3.8: low/medium/xhigh",
    )
    parser.add_argument("--timeout-s", type=int, default=3600)

    args = parser.parse_args()
    if args.reasoning_level is not None:
        args.enable_thinking = True

    if args.backend == "transformers":
        if args.draft_model is None:
            parser.error("--draft-model is required for transformers backend")
        _run_transformers(args)
    elif args.backend == "mlx":
        if args.draft_model is None:
            parser.error("--draft-model is required for mlx backend")
        _run_mlx(args)
    else:
        _run_openai(args)


if __name__ == "__main__":
    main()
