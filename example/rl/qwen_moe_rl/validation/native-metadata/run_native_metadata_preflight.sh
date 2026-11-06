#!/usr/bin/env bash
# Run only under the root's identity-bound cluster supervisor and peak samplers.
set -euo pipefail
ulimit -c 0
DEPLOY="${Q38_DEPLOY_ROOT:?Set this current deployment root}"
RUN="${Q38_RUN_ROOT:?The supervisor must supply a fresh run directory}"
PROJECT="${Q38_PROJECT_ROOT:-$DEPLOY/integration}"
PYTHON="${Q38_TRAIN_PYTHON:-$DEPLOY/runtime/env/bin/python}"
export Q38_DEPLOY_ROOT="$DEPLOY" Q38_RUN_ROOT="$RUN" Q38_PROJECT_ROOT="$PROJECT"
export Q38_RAY_ADDRESS="${Q38_RAY_ADDRESS:?Use only the current owned Ray allocation}"
export Q38_NODES_FILE="${Q38_NODES_FILE:?The fresh 18-node allocation manifest is required}"
export Q38_GATEWAY_HOST="${Q38_GATEWAY_HOST:?Bind the current head container IP}"
export Q38_MODEL_PATH="${Q38_MODEL_PATH:-/mnt/fuse/Qwen/Qwen3.8-2.4T-A95B}"
export Q38_ENABLE_BF16_STATE_RUNTIME=1 Q38_BF16_ROUNDING_SEED=20261106
export Q38_VERIFIED_CPU_LIMIT_BYTES_PER_NODE=858993459200
export Q38_CPU_MOMENT_BYTES_PER_NODE=549755813888 Q38_CPU_RESERVE_BYTES_PER_NODE=103079215104
export Q38_GPU_GRADIENT_CACHE_BYTES=85899345920 Q38_GPU_REFIT_MASTER_CACHE_BYTES=25769803776
export Q38_TRAINING_AUDIT_DIR="$RUN/training" Q38_POLICY_AUDIT_DIR="$RUN/policy-loss"
export RTP_PROJECT_ROOT="$PROJECT" QWEN36_RL_ROOT="$PROJECT"
export RTP_SOURCE="${RTP_SOURCE:-$DEPLOY/rtp-runtime}"
export RTP_PYTHON="$PYTHON" RTP_MODEL_PATH="$Q38_MODEL_PATH"
export RTP_SERVER_LAUNCHER="$PROJECT/adapter/serve_rtp_world72.sh"
export RTP_RL_BOOTSTRAP_POLICY_MANIFEST="$PROJECT/training/expected-canonical-policy-manifest.json"
export RTP_RL_NATIVE_LIBRARY_SHA256="${RTP_RL_NATIVE_LIBRARY_SHA256:?Pin the accepted freshly built native artifact}"
export RLLM_Q38_NCCL_LIBRARY_PATH="${RLLM_Q38_NCCL_LIBRARY_PATH:?Pin the actual all72 communication-tested NCCL mapped SO path}"
export RLLM_Q38_NCCL_LIBRARY_SHA256="${RLLM_Q38_NCCL_LIBRARY_SHA256:?Pin that actual NCCL mapped SO SHA256}"
export LD_PRELOAD="$RLLM_Q38_NCCL_LIBRARY_PATH${LD_PRELOAD:+:$LD_PRELOAD}"
export RTP_LLM_RL_MODE=1 RTP_LLM_RL_WORLD72=1 RTP_LLM_RL_START_PAUSED=1 RTP_LLM_RL_METADATA_BOOTSTRAP=1
export EP_SIZE=72 FFN_SP_SIZE=8 RTP_TP_SIZE=8 RTP_DP_SIZE=9 RTP_EP_SIZE=72 RTP_FFN_SP_SIZE=8
export RTP_MAX_SEQ_LEN=8192 RTP_KV_CACHE_MEM_MB=2048 RTP_MAX_CONTEXT_BATCH_SIZE=8
export RTP_MAX_BATCH_TOKENS_SIZE=8192 RTP_CONCURRENCY_LIMIT=8
export RTP_NCCL_EP_REFERENCE_MAX_PADDED_TOKENS=73728
export RTP_FIXED_PHYSICAL_EP_LAYOUT=balanced_v1 RTP_RL_BORROW_SINGLETON_HF_SOURCES=1
export RLLM_RTP_ACCEPTANCE_DIR="$RUN/rtp" RLLM_RTP_DEPLOY_ROOT="$PROJECT"
export RLLM_RTP_KEEP_KV_CACHE=1 RLLM_RTP_STARTUP_TIMEOUT=5400 RLLM_RTP_CONTROL_TIMEOUT=1800
export RLLM_RTP_TOKEN_TIMEOUT=1800 RLLM_RTP_MAX_ACTIVE_REQUESTS=8
export RLLM_PYTHON_TOOL_TRAIN_JSONL="$PROJECT/data/deepscaler-world72-18.jsonl"
export RLLM_PYTHON_TOOL_ENABLE_THINKING=1 RLLM_PYTHON_TOOL_REASONING_EFFORT=low
export RLLM_PYTHON_TOOL_MAX_TURNS=3 RLLM_PYTHON_TOOL_CALL_TOKENS=2048 RLLM_PYTHON_TOOL_TOTAL_TOKENS=4096
export RLLM_PYTHON_TOOL_HTTP_TIMEOUT=2100 RLLM_PYTHON_TOOL_HTTP_MAX_RETRIES=0
export RLLM_PYTHON_TOOL_TRACE_DIR="$RUN/tool-episodes" RLLM_EPISODE_LOG_DIR="$RUN/episodes"
export RLLM_QWEN36_TRACE_AUDIT_DIR="$RUN/trace-linkage" RLLM_REQUIRE_REAL_UPDATE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=backend:native,expandable_segments:False
unset PYTORCH_CUDA_ALLOC_CONF
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false RAY_DEDUP_LOGS=0 CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="$PROJECT/training:$PROJECT/adapter:$PROJECT/agent:$PROJECT/rtp:$RTP_SOURCE:$RTP_SOURCE/rtp_llm/libs${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN/training" "$RUN/policy-loss" "$RUN/rtp" "$RUN/episodes" "$RUN/tool-episodes" "$RUN/trace-linkage"
"$PYTHON" - "$RLLM_PYTHON_TOOL_TRAIN_JSONL" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);payload=p.read_bytes()
if hashlib.sha256(payload).hexdigest()!='a1030fb5e9cf9b29517d918e4135d69d50357de3c7f615857e0aa3fc45bfd16d':
    raise ValueError('The exact genuine 18-question training dataset differs')
rows=[json.loads(line) for line in payload.splitlines()]
if len(rows)!=18 or len({row['source_row_index'] for row in rows})!=18:raise ValueError('Expected 18 distinct source questions')
PY
args=("$PYTHON" -m run_qwen38_bf16_state_rtp_training
  rllm/backend=verl rllm.algorithm.adv_estimator=grpo ++rllm.disable_thinking=false
  rllm.algorithm.norm_adv_by_std_in_grpo=true rllm.algorithm.kl_beta=0.0
  rllm.algorithm.lr_schedule=constant rllm.algorithm.warmup_steps_ratio=0.0 rllm.algorithm.router_replay=disabled
  rllm.data.train_batch_size=18 rllm.data.max_prompt_length=4096 rllm.data.max_response_length=4096
  rllm.rollout.n=4 rllm.rollout.train.temperature=1.0 rllm.rollout.train.top_p=1.0
  +rllm.rollout.train.top_k=0 +rllm.rollout.val.top_k=0 '~rllm.rollout.train.max_tokens'
  rllm.workflow.n_parallel_tasks=18 rllm.workflow.retry_limit=1 rllm.workflow.raise_on_error=true
  "rllm.gateway.host=$Q38_GATEWAY_HOST" rllm.gateway.port=30135
  rllm.episode_logging.log_episodes=true "rllm.episode_logging.episode_log_dir=$RUN/episodes"
  rllm.trainer.total_epochs=2 rllm.trainer.total_batches=2 "rllm.trainer.logger=['console']"
  rllm.trainer.project_name=qwen38-bf16-rtp-rl rllm.trainer.experiment_name=world72-bf16-state-grpo-python-tool
  rllm.trainer.val_before_train=false rllm.trainer.test_freq=-1 rllm.trainer.save_freq=-1
  "+model.name=$Q38_MODEL_PATH" "actor_rollout_ref.model.path=$Q38_MODEL_PATH"
  actor_rollout_ref.model.external_lib=qwen38_bf16_training_runtime
  actor_rollout_ref.model.use_remove_padding=false actor_rollout_ref.model.use_fused_kernels=false
  actor_rollout_ref.model.use_liger=false actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.model.lora_rank=0 +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  actor_rollout_ref.model.mtp.enable=false actor_rollout_ref.model.mtp.enable_train=false actor_rollout_ref.model.mtp.enable_rollout=false
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.fsdp_config.fsdp_size=72
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 actor_rollout_ref.actor.fsdp_config.use_orig_params=true
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true
  actor_rollout_ref.actor.fsdp_config.param_offload=false actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
  actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.weight_decay=0.0
  actor_rollout_ref.actor.optim.optimizer_impl=qwen38_bf16_training_runtime actor_rollout_ref.actor.optim.optimizer=Bf16CpuAdamW
  actor_rollout_ref.actor.entropy_coeff=0.0 actor_rollout_ref.actor.use_torch_compile=false
  actor_rollout_ref.actor.ppo_mini_batch_size=18 actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.use_dynamic_bsz=false actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.name=rtp actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.tensor_model_parallel_size=8 actor_rollout_ref.rollout.data_parallel_size=9
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 actor_rollout_ref.rollout.checkpoint_engine.backend=naive
  actor_rollout_ref.rollout.free_cache_engine=true actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=0
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0
  actor_rollout_ref.rollout.val_kwargs.top_k=0 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.enforce_eager=true actor_rollout_ref.rollout.max_model_len=8192
  actor_rollout_ref.rollout.gpu_memory_utilization=0.15 actor_rollout_ref.rollout.max_num_seqs=72
  trainer.balance_batch=true trainer.n_gpus_per_node=4 trainer.nnodes=18
  "trainer.default_local_dir=$RUN/disabled-checkpoints" trainer.default_hdfs_dir=null trainer.resume_mode=disable
  "+ray_init.address=$Q38_RAY_ADDRESS")
if [[ "${1:-}" == --print-command && "$#" == 1 ]]; then printf '%q ' "${args[@]}"; printf '\n'; exit 0; fi
if [[ "$#" != 0 ]]; then echo 'Unexpected recipe arguments are forbidden; freeze a new recipe for changed settings.' >&2; exit 2; fi
exec "$PYTHON" "${Q38_NATIVE_METADATA_PREFLIGHT_DRIVER:?Bind the frozen metadata-only driver}"
