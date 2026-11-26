# Batch-invariant inference

## Qwen4-Max M11

The native MXFP4 text path runs on one eight-GPU SM103 CUDA 13 host with
TP1, DP8, EP8, BF16 dense projections and KV, FP32 SSM state and 128-token
cache pages. Set `RTP_BATCH_INVARIANT=1` and use the native `mega_moe` backend
with `--enable_cuda_graph 0`.
The M11 service test sets these options, runs budgets 0 and 256 in sequence,
and records every decode step. It compares generated IDs, full-vocabulary
FP32 logits and sampled probabilities bit for bit across target row positions,
DP ranks, load on a second rank, and chunk boundaries. It also runs the M11
Chinese-answer and long-context retrieval checks.

```bash
bazel test --config=cuda13 //rtp_llm/test/batch_invariant:config_test
bazel test --config=cuda13 //rtp_llm/test/batch_invariant:m11_gdn_test //rtp_llm/test/batch_invariant:m11_ops_test
export QWEN4_M11_CHECKPOINT=/mnt/fuse/oss-wlcb-qwen4-m11
bazel test --config=cuda13 --test_env=QWEN4_M11_CHECKPOINT --test_output=errors \
    //rtp_llm/test/batch_invariant:qwen4_m11_mxfp4_sm10x
```

The two operator tests each use one SM103 GPU. The service test requires the
full native M11 checkpoint on the eight-GPU worker and writes its unchunked
reference to `budget-0/solo.npz` in the Bazel test outputs.

## Qwen3

Set `RTP_BATCH_INVARIANT=1` for Qwen3-8B Dense or Qwen3-30B-A3B with unquantized BF16
weights on one node with 1, 2 or 4 SM103 CUDA GPUs. The mode uses a BF16 KV cache
with 64-token pages and the regular scheduler, including dynamic batching and
chunked prefill. Leave speculative decoding and prefix reuse disabled.
An unset switch or `0` selects the regular execution path.

Add `--enable_cuda_graph 1` to capture pure decode batches. The runner pads
each batch to the smallest captured batch size that fits. Larger batches and
all prefill steps run eagerly. Use `--decode_capture_config` to select the
decode batch sizes and leave `--generation_prefill_capture_config` unset.

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
set -e
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

run_dir=$(mktemp -d "${TMPDIR:-/tmp}/qwen3-bi-tp${TP_SIZE}.XXXXXX")
check_service() (
    run="$run_dir/$1" port=$2 chunk=$3 graph=$4
    shift 4
    mkdir -p "$run/logs"
    cd "$run"
    if [ "$graph" = 1 ]; then export LOG_LEVEL=DEBUG; fi
    LOG_PATH="$run/logs" python3 -m rtp_llm.start_server \
        --checkpoint_path "$checkpoint" --tokenizer_path "$checkpoint" \
        --model_type "$model_type" --act_type bf16 --max_seq_len 1024 \
        --max_context_batch_size 4 --max_batch_tokens_size 8192 --concurrency_limit 12 \
        --kv_cache_mem_mb 4096 --start_port "$port" --prefill_chunk_size "$chunk" \
        --enable_cuda_graph "$graph" --decode_capture_config 1,4 >server.log 2>&1 &
    server=$!
    trap 'kill "$server" 2>/dev/null || true; wait "$server" || true' EXIT
    python3 "$repo/rtp_llm/test/batch_invariant/service_test.py" \
        --checkpoint "$checkpoint" --address "127.0.0.1:$((port + 1))" \
        --log-dir "$run/logs" "$@"
)

reference="$run_dir/solo.npz"
check_service eager          19480   0 0 --save "$reference"
check_service eager-chunked  19580 192 0 --reference "$reference"
check_service graph          19680   0 1 --reference "$reference" --decode-graph unchunked
check_service graph-chunked  19780 192 1 --reference "$reference" --decode-graph chunked
```

Repeat the service runs for each model and TP size. For TP2 and TP4 check that
the worker startup logs show all expected ranks and the requested world size.
The test checks real prefill and decode co-batching,
target row placement and prefill grants. It compares the token IDs,
full-vocabulary FP32 logits and sampled-token softmax probabilities returned
by gRPC bit for bit. With an RPC schema that exposes token logprobs, it also
compares those values. Graph runs match the target request's batch records to
completed decode replays, including batches of two or three requests replayed
in bucket four. Regenerate older reference files with `--save`.

The Linear and Attention factories select the fixed-order implementations
using the same separation between model code and backend selection as vLLM.
[vLLM Linear](https://github.com/vllm-project/vllm/blob/acb87186fa011e273f2d77af4a0517664e2c35a2/vllm/model_executor/layers/linear.py#L226-L236)
· [vLLM Attention](https://github.com/vllm-project/vllm/blob/acb87186fa011e273f2d77af4a0517664e2c35a2/vllm/v1/attention/backend.py#L350-L361)
