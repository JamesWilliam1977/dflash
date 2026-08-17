# DFlash: Block Diffusion for Flash Speculative Decoding
[**Paper**](https://arxiv.org/abs/2602.06036) | [**Blog**](https://z-lab.ai/projects/dflash/) | [**Models**](https://huggingface.co/z-lab/models?search=DFlash)

**DFlash** is a lightweight **block diffusion** model designed for speculative decoding. It enables efficient and high-quality parallel drafting.

<details open>
<summary><strong>DFlash2 architecture</strong></summary>

<p align="center"><img src="https://raw.githubusercontent.com/jianc99/jianc99.github.io/master/images/dflash2_system.png" alt="DFlash2 architecture"></p>
</details>

<details>
<summary><strong>DFlash architecture</strong></summary>

![DFlash architecture](https://raw.githubusercontent.com/jianc99/jianc99.github.io/master/images/dflash_system.png)

https://github.com/user-attachments/assets/5b29cabb-eb95-44c9-8ffe-367c0758de8c
</details>

## Supported Models

### DFlash2

Available checkpoints: [Muse-Glimmer-30B](https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2) and [Qwen3.8-27B](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2). See the [DFlash2 collection](https://huggingface.co/collections/z-lab/dflash2-6a82bfe5f57644038bc8714a) for updates.

### DFlash

Public checkpoints are available in the [DFlash collection](https://huggingface.co/collections/z-lab/dflash):

- **Qwen:** Qwen3.6 (27B, 35B-A3B), Qwen3.5 (4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B), Qwen3 (4B/8B non-thinking, Coder-Next, Coder-30B-A3B)
- **Gemma:** Gemma 4 (12B, 31B, 26B-A4B)
- **MiniMax:** M2.5, M2.7
- **Kimi:** K2.5, K2.6, K2.7-Code
- **Others:** GPT-OSS (20B, 120B), Llama-3.1-8B, GLM 5.1, Alpamayo 1.5/R1 10B

Use the Transformers or MLX backends below for their explicitly listed model families. Other checkpoints can be benchmarked through an OpenAI-compatible SGLang or vLLM server.

## 📦 Installation

Use a separate virtual environment for each to avoid conflict.

| Backend | Install command |
|---|---|
| **Transformers** | `uv pip install -e ".[transformers]"` |
| **MLX** (Apple Silicon) | `pip install -e ".[mlx]"` |

For serving benchmarks, install the latest [SGLang](https://github.com/sgl-project/sglang) or [vLLM](https://github.com/vllm-project/vllm) separately, launch an OpenAI-compatible server with DFlash, and use the `openai` backend below.

## 🚀 Quick Start

### Transformers

The Transformers backend supports DFlash2 for Muse-Glimmer-30B, and DFlash for
Qwen3 and LLaMA-3.1-8B. Muse uses `reasoning_strength`: `low`, `medium`, `high`
(default), or `xhigh`.

```python
from dflash.model import DFlash2DraftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

target = AutoModelForImageTextToText.from_pretrained("meta-models/Muse-Glimmer-30B").cuda().eval()
draft = DFlash2DraftModel.from_pretrained("z-lab/Muse-Glimmer-30B-DFlash2").cuda().eval()
tokenizer = AutoTokenizer.from_pretrained("meta-models/Muse-Glimmer-30B")

messages = [{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, reasoning_strength="high"
)
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(draft.device)

output = draft.spec_generate(input_ids=input_ids, max_new_tokens=2048, temperature=0.0, target=target, stop_token_ids=[tokenizer.eos_token_id])
print(tokenizer.decode(output[0], skip_special_tokens=False))
```

### MLX (Apple Silicon)

The MLX backend supports DFlash2 for Qwen3.8-27B, and DFlash for Qwen3,
Qwen3.5, Qwen3.6, and Gemma 4. Qwen3.8 uses `reasoning_effort`: `low`, `medium`,
or `xhigh` (default). For quantized targets or drafts, use `block_size <= 5`: MLX's current
quantized matmul kernel becomes less efficient at larger verify widths.
The example below runs both the target and draft with 4-bit weights.

```python
import mlx.core as mx
import mlx.nn as nn
from dflash.model_mlx import load, load_draft, stream_generate

model, tokenizer = load("mlx-community/Qwen3.8-27B-4bit")
draft = load_draft("z-lab/Qwen3.8-27B-DFlash2")
nn.quantize(draft, group_size=64, bits=4)
mx.eval(draft.parameters())

messages = [{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=True, reasoning_effort="xhigh",
)
tps = 0.0
for r in stream_generate(
    model, draft, tokenizer, prompt, block_size=5, max_tokens=2048,
    temperature=1.0, top_p=0.95, top_k=20,
):
    print(r.text, end="", flush=True)
    tps = r.generation_tps
print(f"\nThroughput: {tps:.2f} tok/s")
```

Both local backends use exact rejection sampling for sampled decoding and support
`temperature`, `top_p`, and `top_k`.

## 📊 Evaluation

All benchmarks share the same datasets (gsm8k, math500, humaneval, mbpp, mt-bench), downloaded and cached by Hugging Face Datasets.

**OpenAI-compatible server** (SGLang or vLLM):
```bash
python -m dflash.benchmark --backend openai \
    --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.5-27B \
    --dataset gsm8k --num-prompts 128 --concurrency 1 --enable-thinking
```

**Transformers** (Muse-Glimmer-30B DFlash2):
```bash
torchrun --nproc_per_node=8 -m dflash.benchmark --backend transformers \
    --model meta-models/Muse-Glimmer-30B --draft-model z-lab/Muse-Glimmer-30B-DFlash2 \
    --dataset gsm8k --max-samples 128 --reasoning-level high
```

**MLX** (Qwen3.8-27B 4-bit DFlash2):
```bash
python -m dflash.benchmark --backend mlx \
    --model mlx-community/Qwen3.8-27B-4bit --draft-model z-lab/Qwen3.8-27B-DFlash2 \
    --dataset gsm8k --max-samples 128 --reasoning-level xhigh --block-size 5 --draft-bits 4
```

## Acknowledgement

Huge thanks to [@dcw02](https://github.com/dcw02), [@gongy](https://github.com/gongy), and the team at [@modal-labs](https://github.com/modal-labs) for their fast, high-quality support in bringing DFlash to SGLang. And huge thanks as well to [@benchislett](https://github.com/benchislett) at NVIDIA for his work in bringing DFlash to vLLM and helping make it available to the broader serving community.

## Citation
If you find DFlash useful, please cite our work. To share feedback on DFlash or request new model support, please fill out this form: [DFlash Feedback](https://forms.gle/4YNwfqb4nJdqn6hq9).

```bibtex
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}

@misc{inco2026dflash2,
  title  = {DFlash 2: Keep Drafting Parallel},
  author = {{Inco AI}},
  year   = {2026},
  month  = {August},
  url    = {https://inco.ai/blog/dflash2/}
}
```
