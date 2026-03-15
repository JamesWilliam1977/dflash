mkdir -p logs

datasets=(gsm8k math500 humaneval mbpp mt-bench alpaca)

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

for i in "${!datasets[@]}"; do
  ds="${datasets[$i]}"

  CUDA_VISIBLE_DEVICES=$i python benchmark_sglang.py \
      --target-model Qwen/Qwen3.5-9B \
      --draft-model z-lab/Qwen3.5-9B-DFlash \
      --concurrencies 1,8,16,32 \
      --dataset-name "$ds" \
      --max-new-tokens 4096 \
      --max-questions-per-config 1024 \
      --attention-backends trtllm_mha \
      --speculative-draft-attention-backend fa4 \
      --page-size 64 \
      --tp-size 1 \
      --mem-fraction-static 0.9 \
      --max-running-requests 64 \
      --mamba-scheduler-strategy extra_buffer \
      --enable-thinking \
      > "logs/${ds}.log" 2>&1 &
done

wait
