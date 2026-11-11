#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
: "${RTP_SOURCE:?Pin the current RTP source checkout}" "${RTP_PYTHON:?Pin the native Python 3.10}"
: "${RTP_NATIVE_LIBRARY_DIR:?Pin the SM103/Torch2.13 native build}" "${RTP_NATIVE_DEPENDENCY_PYTHONPATH:?Pin the native Python dependencies}"
: "${RTP_NATIVE_DEPENDENCY_LIBRARY_PATH:?Pin the native ABI dependency libraries}" "${RTP_RL_NATIVE_LIBRARY_SHA256:?Pin the accepted native SO digest}"
: "${RTP_RL_BOOTSTRAP_POLICY_MANIFEST:?Supply the pinned 1590-source shape contract}" "${RTP_PHY2LOG_PATH:?Supply the per-layer 512-slot identity map}"
: "${RTP_NODE_IP:?Bind the current rollout container IP}" "${RTP_WORLD_RANK:?Bind its first global rank}"
: "${RTP_WEIGHT_MANIFEST_DIR:?Use a fresh rank manifest directory}" "${RTP_LLM_RL_RUN_ID:?Bind the current incarnation}"
: "${CUDA_VISIBLE_DEVICES:?Bind four current physical GPU UUIDs}" "${GANG_CONFIG_STRING:?Bind all eight rollout containers}"
: "${RLLM_Q38_NCCL_LIBRARY_PATH:?Pin the qualified NCCL library}" "${RTP_START_PORT:?Reserve native ports}"
: "${FLASHINFER_CUBIN_DIR:?Pin the SM103 FlashInfer artifacts}" "${FLASHINFER_WORKSPACE_BASE:?Use a private kernel workspace}"
: "${RTP_MODEL_PATH:?Supply the verified mounted BF16 model}" "${RLLM_RTP_ACCEPTANCE_DIR:?Use a fresh audit directory}"
export RTP_LLM_RL_MODE=1 RTP_LLM_RL_WORLD32=1 RTP_LLM_RL_START_PAUSED=1 RTP_LLM_RL_METADATA_BOOTSTRAP=1
unset RTP_LLM_RL_WORLD72 RTP_LLM_RL_Q36_EP4 PYTORCH_CUDA_ALLOC_CONF
export RTP_FIXED_PHYSICAL_EP_LAYOUT=balanced_v1 RTP_RL_BORROW_SINGLETON_HF_SOURCES=1
export RTP_NCCL_EP_REFERENCE_MAX_PADDED_TOKENS=32768 RTP_ARM_BF16_ROPE_BACKEND=torch_bf16_v1
export WORLD_SIZE=32 LOCAL_WORLD_SIZE=4 TP_SIZE=8 DP_SIZE=4 EP_SIZE=32 FFN_SP_SIZE=8 PP_SIZE=1
export RTP_WORLD_SIZE=32 RTP_LOCAL_WORLD_SIZE=4 RTP_TP_SIZE=8 RTP_DP_SIZE=4 RTP_EP_SIZE=32 RTP_FFN_SP_SIZE=8
export WORLD_RANK="$RTP_WORLD_RANK"
export CHECKPOINT_PATH="$RTP_MODEL_PATH" TOKENIZER_PATH="$RTP_MODEL_PATH"
export PTUNING_PATH= EXTRA_DATA_PATH= LOCAL_EXTRA_DATA_PATH= SP_CHECKPOINT_PATH= SP_TYPE= SP_MODEL_TYPE=
export LORA_INFO='{}' JSON_MODEL_OVERRIDE_ARGS='{}' MERGE_LORA=false
unset QUANTIZATION SP_QUANTIZATION SP_ACT_TYPE TREE_DECODE_CONFIG FLASHINFER_DISABLE_JIT
export USE_DEEPEP_MOE=false USE_DEEPEP_LOW_LATENCY=false USE_DEEPEP_INTERNODE=false USE_MORI_EP=false
export FLASHINFER_CUDA_ARCH_LIST=10.3 FLASHINFER_NO_DOWNLOAD=1 MAX_JOBS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF=backend:native,expandable_segments:False OMP_NUM_THREADS="${RTP_OMP_NUM_THREADS:-4}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PATH="$(dirname "$RTP_PYTHON"):$CUDA_HOME/bin:/opt/conda310/bin:$PATH"
export PYTHONPATH="$RTP_SOURCE/example/rl/qwen_moe_rl/rtp:$RTP_SOURCE:$RTP_NATIVE_LIBRARY_DIR:$RTP_NATIVE_DEPENDENCY_PYTHONPATH"
export LD_LIBRARY_PATH="$RTP_NATIVE_LIBRARY_DIR:$CUDA_HOME/lib64:$RTP_NATIVE_DEPENDENCY_LIBRARY_PATH"
export LD_PRELOAD="$RLLM_Q38_NCCL_LIBRARY_PATH"
export LOG_PATH="$RLLM_RTP_ACCEPTANCE_DIR/native"
"$RTP_PYTHON" - <<'PY'
import ipaddress, os, re
from pathlib import Path
from uuid import UUID
e=os.environ
rank, port = int(e['RTP_WORLD_RANK']), int(e['RTP_START_PORT'])
if rank%4 or not 0<=rank<32 or not 1035<=port<65488:raise ValueError('invalid rank offset or port block')
hosts=[]
for i, entry in enumerate(e['GANG_CONFIG_STRING'].split(';')):
    name,addr,p=entry.split(',')
    if name!=f'name:q38_world_part{i}' or not addr.startswith('ip:') or p!=f'port:{port}':
        raise ValueError('invalid ordered native gang')
    hosts.append(str(ipaddress.IPv4Address(addr[3:])))
if len(hosts)!=8 or len(set(hosts))!=8 or hosts[rank//4]!=e['RTP_NODE_IP']:
    raise ValueError('the current owner does not bind this eight-node gang')
uuids=[UUID(x.removeprefix('GPU-')) for x in e['CUDA_VISIBLE_DEVICES'].split(',')]
if len(uuids)!=4 or len(set(uuids))!=4:raise ValueError('exactly four physical GPU UUIDs required')
if not re.fullmatch(r'[a-f0-9]{64}',e['RTP_RL_NATIVE_LIBRARY_SHA256']):raise ValueError('invalid native SHA256')
if e.get('RTP_LLM_RL_WORLD32_LOCAL_EXPERTS') not in ('0','1'):raise ValueError('select the expert wire format explicitly')
for key in ('RTP_RL_BOOTSTRAP_POLICY_MANIFEST','RTP_PHY2LOG_PATH','RLLM_Q38_NCCL_LIBRARY_PATH'):
    if not Path(e[key]).is_file():raise ValueError(f'missing pinned input {key}')
if not (Path(e['RTP_MODEL_PATH'])/'config.json').is_file():raise ValueError('mounted model metadata is absent')
PY
args=("$RTP_PYTHON" -m qwen38_world72_entrypoint
  --model_type qwen38_moe_text --checkpoint_path "$RTP_MODEL_PATH" --tokenizer_path "$RTP_MODEL_PATH" --act_type BF16
  --ptuning_path "" --extra_data_path "" --local_extra_data_path ""
  --sp_checkpoint_path "" --sp_type "" --sp_model_type "" --lora_info '{}' --merge_lora false --json_model_override_args '{}'
  --start_port "$RTP_START_PORT" --world_size 32 --local_world_size 4 --world_rank "$RTP_WORLD_RANK"
  --tp_size 8 --dp_size 4 --ep_size 32 --ffn_sp_size 8
  --moe_strategy nccl_bf16_reference --moe_pure_tp_preshard false --redundant_expert 0 --phy2log_path "$RTP_PHY2LOG_PATH" --eplb_mode NONE
  --use_deepep_moe false --use_deepep_low_latency false --use_deepep_internode false --use_mori_ep false
  --frontend_server_count 0 --frontend_pre_stop_drain_seconds 0 --dash_sc_grpc_pre_stop_drain_seconds 0 --backend_post_frontend_drain_seconds 0
  --batch_invariant false --enable_fp32_lm_head false --max_seq_len 8192 --kv_cache_mem_mb "${RTP_KV_CACHE_MEM_MB:-4096}"
  --ssm_state_dtype fp32 --seq_size_per_block 8192 --kernel_seq_size_per_block 64
  --max_context_batch_size "${RTP_MAX_CONTEXT_BATCH_SIZE:-8}" --max_batch_tokens_size 8192
  --concurrency_limit "${RTP_CONCURRENCY_LIMIT:-8}" --enable_mixed_continuous_batching false --pdfusion_coord_mode off
  --reuse_cache false --enable_device_cache false --enable_remote_cache false
  --warm_up false --enable_cuda_graph false --enable_native_cuda_graph false)
if [[ "${1:-}" == --print-command && "$#" == 1 ]]; then printf '%q ' "${args[@]}"; printf '\n'; exit 0; fi
[[ "$#" == 0 ]]
cd "$RTP_SOURCE"
exec "${args[@]}"
