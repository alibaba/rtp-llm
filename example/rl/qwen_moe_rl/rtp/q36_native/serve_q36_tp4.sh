#!/usr/bin/env bash
# Q36 single-node TP4 independent RTP process; owner supplies topology and native paths.
set -euo pipefail
ulimit -c 0
OVERLAY="${Q36_NATIVE_OVERLAY:?The frozen Q36 native overlay directory is required}"
: "${RTP_SOURCE:?Supply RTP source}" "${RTP_PYTHON:?Supply native Python}" "${RLLM_RTP_DEPLOY_ROOT:?Supply deployment root}"
: "${Q36_NCCL_ROOT:?Supply pinned NCCL Python-only root}" "${Q36_NATIVE_NCCL_SO:?Supply pinned NCCL library}"
: "${Q36_NATIVE_LIBRARY_PATH:?Supply the native-only dynamic library path}" "${Q36_ARM_ROPE_DIR:?Supply native ARM RoPE source}"
: "${Q36_ARM_ROPE_SHA256:?Supply the qualified ARM RoPE hook SHA256}"
: "${Q36_FLASHINFER_CUBIN_DIR:?Supply FlashInfer cubins}" "${Q36_FLASHINFER_WORKSPACE_BASE:?Supply FlashInfer workspace}"
: "${RTP_MODEL_PATH:?}" "${RTP_WEIGHT_MANIFEST_DIR:?}" "${RTP_LLM_RL_RUN_ID:?}" "${CUDA_VISIBLE_DEVICES:?}" "${RLLM_RTP_ACCEPTANCE_DIR:?}"
[[ "${RTP_TP_SIZE:-}" == 4 && "${RTP_EP_SIZE:-}" == 1 && "${RTP_LOCAL_WORLD_SIZE:-}" == 4 && "${RTP_WORLD_RANK:-}" == 0 ]] || { echo 'Expected one TP4 EP1 node' >&2; exit 2; }
unset RTP_LLM_RL_WORLD72 RTP_LLM_RL_START_PAUSED RTP_LLM_RL_METADATA_BOOTSTRAP RTP_RL_SHARED_PROVIDER
unset RTP_FIXED_PHYSICAL_EP_LAYOUT RTP_PHY2LOG_PATH QUANTIZATION SP_QUANTIZATION SP_ACT_TYPE TREE_DECODE_CONFIG
export RTP_RL_SHARED_BACKING=0 RTP_LLM_RL_MODE=1 RTP_ARM_BF16_ROPE_BACKEND=torch_bf16_v1
export WORLD_SIZE=4 LOCAL_WORLD_SIZE=4 WORLD_RANK=0 TP_SIZE=4 DP_SIZE=1 EP_SIZE=1 FFN_SP_SIZE=1 PP_SIZE=1
export CHECKPOINT_PATH="$RTP_MODEL_PATH" TOKENIZER_PATH="$RTP_MODEL_PATH"
export PTUNING_PATH= EXTRA_DATA_PATH= LOCAL_EXTRA_DATA_PATH= SP_CHECKPOINT_PATH= SP_TYPE= SP_MODEL_TYPE=
export LORA_INFO='{}' JSON_MODEL_OVERRIDE_ARGS='{}' MERGE_LORA=false
export FLASHINFER_CUDA_ARCH_LIST=10.0 FLASHINFER_NO_DOWNLOAD=1
export FLASHINFER_CUBIN_DIR="$Q36_FLASHINFER_CUBIN_DIR"
export FLASHINFER_WORKSPACE_BASE="$Q36_FLASHINFER_WORKSPACE_BASE"
export USE_DEEPEP_MOE=false USE_DEEPEP_LOW_LATENCY=false USE_DEEPEP_INTERNODE=false USE_MORI_EP=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=backend:native,expandable_segments:False
unset PYTORCH_CUDA_ALLOC_CONF
# Drop training-only cuDNN and Transformer Engine settings before starting py310.
unset CUDNN_HOME CUDNN_PATH LD_PRELOAD
for name in ${!NVTE_@}; do unset "$name"; done
# Preserve the previously qualified pinned NCCL preload, never parent preloads.
[[ "$(realpath "${Q397_NCCL_SO:?}")" == "$(realpath "$Q36_NATIVE_NCCL_SO")" ]] || { echo "Unexpected NCCL preload path" >&2; exit 2; }
export LD_PRELOAD="$Q397_NCCL_SO"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}" OMP_NUM_THREADS="${RTP_OMP_NUM_THREADS:-4}"
export PATH="$(dirname "$RTP_PYTHON"):$CUDA_HOME/bin:$PATH"
# Native py310 must never import the training worker's py312 packages.
export PYTHONNOUSERSITE=1
export PYTHONPATH="$Q36_NCCL_ROOT:$OVERLAY:$Q36_ARM_ROPE_DIR:$RTP_SOURCE/example/rl/qwen_moe_rl/rtp:$RTP_SOURCE:$RTP_SOURCE/rtp_llm/libs"
export LD_LIBRARY_PATH="$RTP_SOURCE/rtp_llm/libs:$Q36_NATIVE_LIBRARY_PATH"
export LOG_PATH="$RLLM_RTP_ACCEPTANCE_DIR/native"
export RTP_RL_NATIVE_LIBRARY_SHA256=4e5cc7c48c6620230ab6d0ee3effc7e36a2410c70fd97d471e687847c142fa93
"$RTP_PYTHON" - <<'PY'
import hashlib,os
from pathlib import Path
e=os.environ
p=Path(e['RTP_SOURCE'])/'rtp_llm/libs/libth_transformer.so'
d=hashlib.sha256()
with p.open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024**2),b''):d.update(chunk)
if d.hexdigest()!=e['RTP_RL_NATIVE_LIBRARY_SHA256']:raise ValueError('native library SHA mismatch')
if len(e['CUDA_VISIBLE_DEVICES'].split(','))!=4:raise ValueError('exactly four physical GPU selectors required')
if not (Path(e['RTP_MODEL_PATH'])/'config.json').is_file():raise ValueError('missing checkpoint config')
if e.get('Q397_NCCL_SHA256')!='cc01f863461ea28f3ecc4390716a2d7a5aca9bc00e04eeb6fc7a5c2e666b3dfb':raise ValueError('native owner must provide pinned NCCL scope')
PY
if [[ "${1:-}" == --cpu-import-check && "$#" == 1 ]]; then
 CUDA_VISIBLE_DEVICES='' "$RTP_PYTHON" - <<'PY'
import importlib,importlib.util,json,os,sys
from pathlib import Path
import q36_native_entrypoint,torch,transformers,huggingface_hub
mods={}
for name in ('torch','transformers','huggingface_hub','pydantic_core._pydantic_core','jiter.jiter'):
 try:
  module=importlib.import_module(name);mods[name]=str(getattr(module,'__file__',None))
 except ModuleNotFoundError:mods[name]=None
if torch.cuda.is_initialized():raise RuntimeError('CPU preflight unexpectedly initialized CUDA')
if any(k in os.environ for k in ('CUDNN_HOME','CUDNN_PATH')) or any(k.startswith('NVTE_') for k in os.environ):raise RuntimeError('training backend variables leaked')
if os.environ.get('LD_PRELOAD') != os.environ.get('Q397_NCCL_SO'):raise RuntimeError('unexpected native preload')
paths=sys.path+list(filter(None,mods.values()))+os.environ['LD_LIBRARY_PATH'].split(':')
if any('venv-py312' in p or 'nonshared-python-deps' in p for p in paths):raise RuntimeError('training Python paths leaked into native process')
mapped=set()
for line in Path('/proc/self/maps').read_text().splitlines():
 parts=line.split(None,5)
 if len(parts)==6 and parts[-1].startswith('/') and any(s in parts[-1] for s in ('libnccl.so','libtorch','libth_transformer.so','librtp_compute_ops.so','libcudart.so','libcudnn')):
  mapped.add(str(Path(parts[-1]).resolve()))
if any('q36-cudnn-9.28' in p or 'pyoverlay' in p for p in mapped):raise RuntimeError('training cuDNN mapped in native process')
if not any(p.endswith('/rtp_llm/libs/libth_transformer.so') for p in mapped):raise RuntimeError('native transformer library not mapped')
if [p for p in mapped if Path(p).name.startswith('libnccl.so')]!=[str(Path(os.environ['Q36_NATIVE_NCCL_SO']).resolve())]:
 raise RuntimeError('unexpected mapped NCCL library')
print('Q36_NATIVE_ISOLATION '+json.dumps({'python':sys.version,'sys_path':sys.path,'module_paths':mods,
 'mapped_native_libraries':sorted(mapped),'ld_library_path':os.environ['LD_LIBRARY_PATH'],'ld_preload':os.environ['LD_PRELOAD'],'cudnn_version':torch.backends.cudnn.version(),'cuda_initialized':False}),flush=True)
PY
 exit 0
fi
args=("$RTP_PYTHON" -m q36_native_entrypoint
  --model_type qwen35_moe_text --checkpoint_path "$RTP_MODEL_PATH" --tokenizer_path "$RTP_MODEL_PATH" --act_type BF16
  --start_port "${RTP_START_PORT:-24000}" --world_size 4 --local_world_size 4 --world_rank 0
  --tp_size 4 --dp_size 1 --ep_size 1 --ffn_sp_size 1 --moe_pure_tp_preshard false
  --frontend_server_count 0 --frontend_pre_stop_drain_seconds 0 --dash_sc_grpc_pre_stop_drain_seconds 0
  --backend_post_frontend_drain_seconds 0 --batch_invariant false --enable_fp32_lm_head false
  --max_seq_len "${RTP_MAX_SEQ_LEN:-8192}" --kv_cache_mem_mb "${RTP_KV_CACHE_MEM_MB:-4096}"
  --ssm_state_dtype fp32 --seq_size_per_block 8192 --kernel_seq_size_per_block 64
  --max_context_batch_size "${RTP_MAX_CONTEXT_BATCH_SIZE:-8}" --max_batch_tokens_size "${RTP_MAX_BATCH_TOKENS_SIZE:-8192}"
  --concurrency_limit "${RTP_CONCURRENCY_LIMIT:-8}" --reuse_cache false --enable_device_cache false
  --enable_remote_cache false --warm_up false --enable_cuda_graph false --enable_native_cuda_graph false)
if [[ "${1:-}" == --print-command && "$#" == 1 ]]; then printf '%q ' "${args[@]}"; printf '\n'; exit 0; fi
[[ "$#" == 0 ]] || { echo 'Unexpected launcher arguments' >&2; exit 2; }
cd "$RTP_SOURCE"
exec "${args[@]}"
