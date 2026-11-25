# Qwen3 batch-invariant inference

Set `RTP_BATCH_INVARIANT=1` for Qwen3-8B Dense or Qwen3-30B-A3B with unquantized BF16
weights on one node with 1, 2 or 4 SM103 CUDA GPUs. The mode uses eager text
generation, a BF16 KV cache with 64-token pages, and the regular scheduler, including
dynamic batching and chunked prefill. Leave CUDA Graph, speculative
decoding and prefix reuse disabled. An unset switch or `0` selects the
regular execution path.

With tensor parallelism, each rank computes the existing BF16 projections on
its weight shards. Attention and MLP output projections use an all-gather and
a fixed rank-order FP32 sum, rounded once to BF16. Verify batch invariance
against a solo request at the same TP size. Changing TP size also changes the
weight shards and the rounding of their partial products.

For Qwen3-30B-A3B, leave `--moe_strategy` at `auto` or select `no_quant_cpp`.
The batch-invariant switch selects fixed-order projections in that BF16 backend.
Each TP rank owns a shard of all 128 experts. The gate selects and renormalizes
eight experts per token. Both expert GEMMs use fixed tiles, then the rank
accumulates the eight weighted outputs in routing order before the TP reduction.

Use a CUDA 13 wheel built for SM103 and run the following from outside the
checkout so Python imports the installed package. Set `repo` and `checkpoint`
to absolute paths before running the commands.

```bash
# TP2 example. For TP4, select four devices and set TP_SIZE=4.
# For a single-GPU regression, select one device and set TP_SIZE=1.
export CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2
export WORLD_SIZE="$TP_SIZE" LOCAL_WORLD_SIZE="$TP_SIZE" WORLD_RANK=0 LOCAL_RANK=0
export DP_SIZE=1 EP_SIZE=1 PP_SIZE=1 ROLE_TYPE=PDFUSION
export RTP_BATCH_INVARIANT=1
export RTP_LLM_ENABLE_BATCH_RECORD=1 RTP_LLM_RECORD_DECODE_SAMPLE_RATE=1
model_type=qwen_3_moe  # Use qwen_3 with the Qwen3-8B checkpoint.

python3 "$repo/rtp_llm/test/batch_invariant/attention_test.py"
python3 "$repo/rtp_llm/test/batch_invariant/moe_test.py"

serve() (
    LOG_PATH="$1" exec python3 -m rtp_llm.start_server \
        --checkpoint_path "$checkpoint" --tokenizer_path "$checkpoint" \
        --model_type "$model_type" --act_type bf16 --max_seq_len 1024 \
        --max_context_batch_size 4 --max_batch_tokens_size 8192 --concurrency_limit 12 \
        --kv_cache_mem_mb 4096 --start_port "$2" --prefill_chunk_size "$3"
)

run_dir=$(mktemp -d "${TMPDIR:-/tmp}/qwen3-bi-tp${TP_SIZE}.XXXXXX")
serve "$run_dir/unchunked" 19480 0 &
server=$!
python3 "$repo/rtp_llm/test/batch_invariant/service_test.py" \
    --checkpoint "$checkpoint" --log-dir "$run_dir/unchunked" --save "$run_dir/solo.npz"
kill "$server"
wait "$server" || true

serve "$run_dir/chunked" 19580 192 &
server=$!
python3 "$repo/rtp_llm/test/batch_invariant/service_test.py" \
    --checkpoint "$checkpoint" --address 127.0.0.1:19581 \
    --log-dir "$run_dir/chunked" --reference "$run_dir/solo.npz"
kill "$server"
wait "$server" || true
```

Repeat the two service runs for each TP size. For TP2 and TP4 check that
the worker startup logs show all expected ranks and the requested world size.
The test checks real prefill and decode co-batching,
target row placement and prefill grants. It compares the token IDs,
full-vocabulary FP32 logits, sampled-token softmax probabilities and token
logprobs returned by gRPC bit for bit. Regenerate older reference files with `--save`.

The Linear and Attention factories select the fixed-order implementations
using the same separation between model code and backend selection as vLLM.
[vLLM Linear](https://github.com/vllm-project/vllm/blob/acb87186fa011e273f2d77af4a0517664e2c35a2/vllm/model_executor/layers/linear.py#L226-L236)
· [vLLM Attention](https://github.com/vllm-project/vllm/blob/acb87186fa011e273f2d77af4a0517664e2c35a2/vllm/v1/attention/backend.py#L350-L361)
