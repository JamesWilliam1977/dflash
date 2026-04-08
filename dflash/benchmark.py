from __future__ import annotations

import argparse
import os
import random
import re
import statistics
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from types import SimpleNamespace
from typing import Any, List, Optional

import numpy as np
import requests
import torch
from torch import distributed as torch_dist
from loguru import logger
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .model import DFlashDraftModel, sample, load_and_process_dataset, extract_context_feature


def _dist_init() -> None:
    if "RANK" not in os.environ:
        warnings.warn("RANK not set. Skipping distributed initialization.")
        return
    torch_dist.init_process_group(backend="nccl", init_method="env://")

def _dist_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))

def _dist_rank() -> int:
    return int(os.environ.get("RANK", 0))

def _dist_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))

def _dist_is_main() -> bool:
    return _dist_rank() == 0

def _dist_gather(obj: Any, dst: int = 0) -> Optional[List[Any]]:
    if not torch_dist.is_initialized():
        return [obj]
    if _dist_is_main():
        objs: List[Any] = [None for _ in range(_dist_size())]
        torch_dist.gather_object(obj, objs, dst=dst)
        return objs
    else:
        torch_dist.gather_object(obj, dst=dst)
        return None


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def _dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    prefill_start = _cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    time_to_first_token = _cuda_time() - prefill_start

    decode_start = _cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size+1:, :])
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)
            if draft_prefill:
                draft_prefill = False
                decode_start = _cuda_time()

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )

        posterior = sample(output.logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = _cuda_time() - decode_start
    time_per_output_token = total_decode_time / num_output_tokens

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
    )


_TRANSFORMERS_SUPPORTED_PATTERN = re.compile(r"qwen3(?!\.5)[\w-]*|llama.*3\.1.*8b.*instruct", re.IGNORECASE)


def _check_transformers_model(model_name: str) -> None:
    if not _TRANSFORMERS_SUPPORTED_PATTERN.search(model_name):
        raise ValueError(
            f"Transformers backend does not support '{model_name}'. "
            f"Only Qwen3 series and LLaMA-3.1-8B-Instruct are supported. "
            f"Use --backend sglang or --backend vllm for other models."
        )


def _run_transformers(args: argparse.Namespace) -> None:
    _check_transformers_model(args.model)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _dist_init()
    torch.cuda.set_device(_dist_local_rank())
    device = torch.device(f"cuda:{_dist_local_rank()}")

    def has_flash_attn():
        try:
            import flash_attn
            return True
        except ImportError:
            logger.warning(
                "flash_attn not installed. Falling back to torch.sdpa. Speedup will be lower. "
                "For optimal speedup in Transformers backend, please install: "
                "pip install flash-attn --no-build-isolation"
            )
            return False

    installed_flash_attn = has_flash_attn()
    attn_impl = "flash_attention_2" if installed_flash_attn else "sdpa"

    target = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation=attn_impl, dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_model, attn_implementation=attn_impl, dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        random.shuffle(dataset)
        dataset = dataset[:args.max_samples]

    responses = []
    indices = range(_dist_rank(), len(dataset), _dist_size())
    for idx in tqdm(indices, disable=not _dist_is_main()):
        instance = dataset[idx]
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            for bs in [1, block_size]:
                response[bs] = _dflash_generate(
                    model=draft_model, target=target, input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens, block_size=bs,
                    stop_token_ids=[tokenizer.eos_token_id], temperature=args.temperature,
                )

            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if _dist_size() > 1:
        responses = _dist_gather(responses, dst=0)
        if not _dist_is_main():
            return
        responses = list(chain(*responses))

    t1 = np.mean([r[1].time_per_output_token for r in responses])
    tb = np.mean([r[block_size].time_per_output_token for r in responses])
    print(f"Decoding speedup: {t1 / tb:.2f}")

    tau = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")


def _send_sglang(
    base_url: str, text: str, *, max_new_tokens: int,
    temperature: float, top_p: float, top_k: int, timeout_s: int,
) -> dict:
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": text,
            "sampling_params": {
                "temperature": temperature, "top_p": top_p,
                "top_k": top_k, "max_new_tokens": max_new_tokens,
            },
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    out = resp.json()
    return out if isinstance(out, dict) else out[0]


def _send_vllm(
    base_url: str, text: str, *, model: str, max_new_tokens: int,
    temperature: float, top_p: float, timeout_s: int,
    enable_thinking: bool = False,
) -> dict:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    resp = requests.post(
        base_url + "/v1/chat/completions",
        json=body,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return resp.json()


def _run_server(args: argparse.Namespace) -> None:
    is_vllm = args.backend == "vllm"
    dataset = load_and_process_dataset(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    num_prompts = args.num_prompts + args.concurrency
    prompts: list[str] = []
    for i in range(num_prompts):
        item = dataset[i % len(dataset)]
        user_content = item["turns"][0]
        if is_vllm:
            prompts.append(user_content)
        else:
            prompts.append(tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            ))

    def send_one(prompt: str) -> dict:
        if is_vllm:
            return _send_vllm(
                args.base_url, prompt, model=args.model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
                timeout_s=args.timeout_s,
                enable_thinking=args.enable_thinking,
            )
        else:
            return _send_sglang(
                args.base_url, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
                top_k=args.top_k, timeout_s=args.timeout_s,
            )

    if not is_vllm:
        try:
            requests.get(args.base_url + "/flush_cache", timeout=60).raise_for_status()
        except Exception:
            print("Warning: /flush_cache failed. Continuing.")

    bs = max(args.concurrency, 1)
    if len(prompts) > bs:
        print(f"[warmup] {bs} requests ...")
        with ThreadPoolExecutor(max_workers=bs) as pool:
            list(pool.map(send_one, prompts[:bs]))
        prompts = prompts[bs:]

    print(f"Running benchmark: {args.num_prompts} prompts, concurrency={args.concurrency} ...")
    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_accept_lengths: list[float] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(send_one, p): i for i, p in enumerate(prompts)}
        for fut in tqdm(as_completed(futures), total=len(prompts), desc="Benchmarking"):
            out = fut.result()
            if is_vllm:
                usage = out.get("usage", {})
                total_tokens += int(usage.get("completion_tokens", 0))
            else:
                meta = out.get("meta_info", {}) or {}
                total_tokens += int(meta.get("completion_tokens", 0))
                spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
                if "spec_accept_length" in meta:
                    try:
                        spec_accept_lengths.append(float(meta["spec_accept_length"]))
                    except (TypeError, ValueError):
                        pass

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    print(f"\n{'='*50}")
    print(f"Backend:          {args.backend}")
    print(f"Dataset:          {args.dataset}")
    print(f"Num prompts:      {args.num_prompts}")
    print(f"Concurrency:      {args.concurrency}")
    print(f"Latency:          {latency:.1f}s")
    print(f"Output tokens:    {total_tokens}")
    print(f"Throughput:       {toks_per_s:,.2f} tok/s")
    if spec_accept_lengths:
        print(f"Accept length:    {statistics.mean(spec_accept_lengths):.3f}")
    if spec_verify_ct_sum > 0:
        print(f"Spec verify ct:   {spec_verify_ct_sum}")
    print(f"{'='*50}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DFlash benchmark")
    parser.add_argument("--backend", choices=["transformers", "sglang", "vllm"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--num-prompts", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=3600)

    args = parser.parse_args()

    if args.backend == "transformers":
        if args.draft_model is None:
            parser.error("--draft-model is required for transformers backend")
        _run_transformers(args)
    else:
        _run_server(args)


if __name__ == "__main__":
    main()
