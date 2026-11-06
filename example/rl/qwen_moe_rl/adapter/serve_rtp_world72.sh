#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
Q38_DEPLOY_ROOT="${Q38_DEPLOY_ROOT:?Set the current dedicated deployment root}"
export RTP_SOURCE="${RTP_SOURCE:-$Q38_DEPLOY_ROOT/rtp-runtime}"
export RTP_PYTHON="${RTP_PYTHON:-$Q38_DEPLOY_ROOT/runtime/env/bin/python}"
export RTP_MODEL_PATH="${RTP_MODEL_PATH:?The owner must bind the mounted BF16 model}"
# Never inherit platform tokenizer/adapters or draft-model object-store paths.
export CHECKPOINT_PATH="$RTP_MODEL_PATH" TOKENIZER_PATH="$RTP_MODEL_PATH"
export PTUNING_PATH= EXTRA_DATA_PATH= LOCAL_EXTRA_DATA_PATH= SP_CHECKPOINT_PATH= SP_TYPE= SP_MODEL_TYPE=
export LORA_INFO='{}' JSON_MODEL_OVERRIDE_ARGS='{}'
unset QUANTIZATION SP_QUANTIZATION SP_ACT_TYPE TREE_DECODE_CONFIG
export RTP_ARM_BF16_ROPE_BACKEND=torch_bf16_v1
export FLASHINFER_CUDA_ARCH_LIST=10.0 FLASHINFER_NO_DOWNLOAD=1
export FLASHINFER_CUBIN_DIR="$Q38_DEPLOY_ROOT/rtp-runtime-kernels/flashinfer-runtime/cubins"
export FLASHINFER_WORKSPACE_BASE="$Q38_DEPLOY_ROOT/rtp-runtime-kernels/flashinfer-jit"
export MAX_JOBS=4
unset FLASHINFER_DISABLE_JIT
export RTP_WEIGHT_MANIFEST_DIR="${RTP_WEIGHT_MANIFEST_DIR:?The owner must bind the current native incarnation manifest directory}"
export RTP_RL_NATIVE_LIBRARY_SHA256="${RTP_RL_NATIVE_LIBRARY_SHA256:?Pin the fresh SM100 native library artifact}"
export RTP_RL_BOOTSTRAP_POLICY_MANIFEST="${RTP_RL_BOOTSTRAP_POLICY_MANIFEST:?Pin the 1590-source canonical metadata manifest}"
export RTP_PHY2LOG_PATH="${RTP_PHY2LOG_PATH:?The owner must create the exact fixed 92-layer physical expert map}"
export RLLM_Q38_NCCL_LIBRARY_PATH="${RLLM_Q38_NCCL_LIBRARY_PATH:?Pin the actual communication-qualified native NCCL mapped SO}"
export RLLM_Q38_NCCL_LIBRARY_SHA256="${RLLM_Q38_NCCL_LIBRARY_SHA256:?Pin the actual communication-qualified NCCL SHA256}"
export RTP_LLM_RL_MODE=1 RTP_LLM_RL_WORLD72=1 RTP_LLM_RL_START_PAUSED=1 RTP_LLM_RL_METADATA_BOOTSTRAP=1
export WORLD_SIZE=72 LOCAL_WORLD_SIZE=4 TP_SIZE=8 DP_SIZE=9 EP_SIZE=72 FFN_SP_SIZE=8 PP_SIZE=1
export RTP_WORLD_SIZE=72 RTP_LOCAL_WORLD_SIZE=4 RTP_TP_SIZE=8 RTP_DP_SIZE=9 RTP_EP_SIZE=72 RTP_FFN_SP_SIZE=8
export RTP_NCCL_EP_REFERENCE_MAX_PADDED_TOKENS=73728
export RTP_FIXED_PHYSICAL_EP_LAYOUT=balanced_v1 RTP_RL_BORROW_SINGLETON_HF_SOURCES=1
export RTP_LLM_RL_RUN_ID="${RTP_LLM_RL_RUN_ID:?The owner must supply the current native incarnation ID}"
export RTP_WORLD_RANK="${RTP_WORLD_RANK:?The owner must supply the first world rank of this node}"
export WORLD_RANK="$RTP_WORLD_RANK"
export GANG_CONFIG_STRING="${GANG_CONFIG_STRING:?The complete 18-node gang is required}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?The owner must bind exactly four current physical GPU UUIDs}"
export RTP_START_PORT="${RTP_START_PORT:-24000}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=backend:native,expandable_segments:False
unset PYTORCH_CUDA_ALLOC_CONF
export OMP_NUM_THREADS="${RTP_OMP_NUM_THREADS:-4}"
export LD_PRELOAD="$RLLM_Q38_NCCL_LIBRARY_PATH${LD_PRELOAD:+:$LD_PRELOAD}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PATH="$(dirname "$RTP_PYTHON"):$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$RTP_SOURCE:$RTP_SOURCE/rtp_llm/libs${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$RTP_SOURCE/rtp_llm/libs:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LOG_PATH="${RLLM_RTP_ACCEPTANCE_DIR:?The current owned node audit directory is required}/native"
"$RTP_PYTHON" - <<'PY'
import os,re
from pathlib import Path
e=os.environ
if (e['WORLD_SIZE'],e['LOCAL_WORLD_SIZE'],e['TP_SIZE'],e['DP_SIZE'],e['EP_SIZE'],e['FFN_SP_SIZE']) != ('72','4','8','9','72','8'):
    raise ValueError('This launcher is exclusively one TP8/DP9/EP72/FFN-SP8/PP1 world')
rank=int(e['RTP_WORLD_RANK'])
if rank%4 or not 0<=rank<72 or e['WORLD_RANK']!=str(rank):raise ValueError('Invalid current four-rank node offset')
if len(e['GANG_CONFIG_STRING'].split(';'))!=18:raise ValueError('The current complete gang must contain 18 nodes')
uuids=e['CUDA_VISIBLE_DEVICES'].split(',')
if len(set(uuids))!=4 or any(not re.fullmatch(r'GPU-[a-fA-F0-9-]{36}',value) for value in uuids):
    raise ValueError('Exactly four bound physical GPU UUID selectors are required')
if not re.fullmatch(r'[a-f0-9]{64}',e['RTP_RL_NATIVE_LIBRARY_SHA256']):raise ValueError('Invalid accepted native SHA256')
if not Path(e['RTP_RL_BOOTSTRAP_POLICY_MANIFEST']).is_file():raise ValueError('Missing pinned canonical source manifest')
if not Path(e['RTP_PHY2LOG_PATH']).is_file():raise ValueError('The fixed 576-slot expert map is absent')
if not (Path(e['RTP_MODEL_PATH'])/'config.json').is_file():raise ValueError('Mounted model metadata is absent')
PY
args=("$RTP_PYTHON" -m qwen38_world72_entrypoint
  --model_type qwen38_moe_text --checkpoint_path "$RTP_MODEL_PATH" --tokenizer_path "$RTP_MODEL_PATH" --act_type BF16
  --ptuning_path "" --extra_data_path "" --local_extra_data_path ""
  --sp_checkpoint_path "" --sp_type "" --sp_model_type "" --lora_info '{}' --json_model_override_args '{}'
  --start_port "$RTP_START_PORT" --world_size 72 --local_world_size 4 --world_rank "$RTP_WORLD_RANK"
  --tp_size 8 --dp_size 9 --ep_size 72 --ffn_sp_size 8
  --moe_strategy nccl_bf16_reference --moe_pure_tp_preshard false
  --redundant_expert 64 --phy2log_path "$RTP_PHY2LOG_PATH" --eplb_mode NONE
  --frontend_server_count 0 --frontend_pre_stop_drain_seconds 0
  --dash_sc_grpc_pre_stop_drain_seconds 0 --backend_post_frontend_drain_seconds 0
  --batch_invariant false --enable_fp32_lm_head false
  --max_seq_len "${RTP_MAX_SEQ_LEN:-8192}" --kv_cache_mem_mb "${RTP_KV_CACHE_MEM_MB:-2048}"
  --ssm_state_dtype fp32 --seq_size_per_block 8192 --kernel_seq_size_per_block 64
  --max_context_batch_size "${RTP_MAX_CONTEXT_BATCH_SIZE:-8}"
  --max_batch_tokens_size "${RTP_MAX_BATCH_TOKENS_SIZE:-8192}"
  --concurrency_limit "${RTP_CONCURRENCY_LIMIT:-8}"
  --enable_mixed_continuous_batching false --pdfusion_coord_mode off
  --reuse_cache false --enable_device_cache false --enable_remote_cache false
  --warm_up false --enable_cuda_graph false --enable_native_cuda_graph false)
if [[ "${1:-}" == --print-command && "$#" == 1 ]]; then printf '%q ' "${args[@]}"; printf '\n'; exit 0; fi
if [[ "$#" != 0 ]]; then echo 'Use the pinned world72 profile; unexpected launcher arguments are forbidden.' >&2; exit 2; fi
cd "$RTP_SOURCE"
exec "${args[@]}"
