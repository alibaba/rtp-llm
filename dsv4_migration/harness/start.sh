#!/bin/bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
export PATH=/opt/conda310/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$repo/bazel-bin:/opt/conda310/lib/python3.10/site-packages/torch/lib:/opt/conda310/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/local/lib64:/usr/lib64"
for library_dir in /opt/conda310/lib/python3.10/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$library_dir"
done
export CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_IB_DISABLE=1
export RTP_LLM_STREAM_ASYNC=0 RTP_LLM_DEVICE_INPUT=0 RTP_BENCH_REAL_PREFILL=1
export DG_JIT_CPP_STANDARD=20 DSV4_CHUNK_TOKENS=8192
export MOE_STRATEGY=mega_moe_se
export TORCHINDUCTOR_COMPILE_THREADS=1 FRONTEND_SERVER_COUNT=1
export TRITON_CACHE_DIR=/tmp/dsv4-offload-triton
export DSV4_CSA_FETCH_CTAS=${DSV4_CSA_FETCH_CTAS:-256}
export DSV4_CSA_GPU_CACHE_MIB=${DSV4_CSA_GPU_CACHE_MIB:-6144}
if [[ "${1:-}" == --check ]]; then
    exec /opt/conda310/bin/python -c 'import rtp_llm.start_server; from rtp_llm.models.deepseek_v4 import DeepSeekV4; print("DSV4 main startup imports passed")'
fi
if [[ "${1:-}" == --python ]]; then
    shift
    exec /opt/conda310/bin/python "$repo/dsv4_migration/harness/numa_exec.py" /opt/conda310/bin/python "$@"
fi
scheme=${1:?vanilla or offload required}
case "$scheme" in
    vanilla) export DSV4_CSA_OFFLOAD=0 ;;
    offload) export DSV4_CSA_OFFLOAD=1 ;;
    *) exit 2 ;;
esac
batch=${DSV4_MAX_BATCH:-32}
capture=${DSV4_CAPTURE:-1,8,16,24,32}
model=${DSV4_MODEL:-/home/admin/model/DeepSeek-V4-Pro}
mkdir -p "${DSV4_RUN_ROOT:?run output directory required}"
cd "$DSV4_RUN_ROOT"
ulimit -c 0
exec /opt/conda310/bin/python "$repo/dsv4_migration/harness/numa_exec.py" /opt/conda310/bin/python -m rtp_llm.start_server \
    --model_type deepseek_v4 --checkpoint_path "$model" --tokenizer_path "$model" \
    --act_type bf16 --fp8_kv_cache 1 --load_method fastsafetensors \
    --tp_size 4 --dp_size 1 --ep_size 4 --world_size 4 --local_world_size 4 \
    --use_deepep_moe 1 --use_deepep_low_latency 0 \
    --max_seq_len "${DSV4_MAX_SEQ:-132096}" --concurrency_limit "$batch" \
    --kv_cache_mem_mb "${DSV4_KV_MIB:-12288}" \
    --seq_size_per_block 256 --kernel_seq_size_per_block 256 \
    --dsv4_fixed_pool_blocks "$((2 * batch + 2))" \
    --dsv4_hca_state_pool_blocks "$((batch + 2))" \
    --max_context_batch_size 1 --max_batch_tokens_size "${DSV4_MAX_SEQ:-132096}" \
    --use_batch_decode_scheduler true --batch_decode_scheduler_batch_size 1 \
    --batch_decode_scheduler_warmup_type 1 \
    --reuse_cache false --reserve_block_ratio 0 \
    --enable_cuda_graph true --decode_capture_config "$capture" \
    --enable_layer_micro_batch 0 --warm_up true --start_port "${DSV4_PORT:-18640}"
