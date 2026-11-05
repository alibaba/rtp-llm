#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${QWEN36_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
TRAIN_ROOT="$PROJECT_ROOT/training"
ORIGINAL_MATH="${RLLM_MATH_ROOT:?Set RLLM_MATH_ROOT to the compatible rLLM/verl project}"
MODEL_PATH="${QWEN36_MODEL_PATH:?Set QWEN36_MODEL_PATH to a verified snapshot under your models directory}"
RUN_ROOT="${QWEN36_RUN_ROOT:?Use the supervised launcher to set a fresh run directory}"
mkdir -p "$RUN_ROOT" "$RUN_ROOT/optimizer" "$RUN_ROOT/model-load" "$RUN_ROOT/episodes" "$RUN_ROOT/rtp"
export PYTHONPATH="$PROJECT_ROOT/adapter:$TRAIN_ROOT:$TRAIN_ROOT/overrides:$PROJECT_ROOT/references/agent:$ORIGINAL_MATH/agent${PYTHONPATH:+:$PYTHONPATH}"
export RTP_SERVER_LAUNCHER="$PROJECT_ROOT/adapter/serve_rtp_distributed.sh"
export RTP_RUN_ROOT="$RUN_ROOT/rtp"
export RTP_PROJECT_ROOT="$PROJECT_ROOT"
export RLLM_RTP_ACCEPTANCE_DIR="$RUN_ROOT/rtp"
export RLLM_RTP_DEPLOY_ROOT="$PROJECT_ROOT"
export RLLM_RTP_KEEP_KV_CACHE=1
export RTP_SOURCE="${RTP_SOURCE:-$(cd "$PROJECT_ROOT/../../.." && pwd -P)}"
export RTP_NATIVE_LIBRARY_DIR="${RTP_NATIVE_LIBRARY_DIR:-$PROJECT_ROOT/build/rtp-native/rawlp-v1}"
export RTP_MAX_SEQ_LEN=8192
export RTP_KV_CACHE_MEM_MB="${RTP_KV_CACHE_MEM_MB:-4096}"
export RLLM_RTP_STARTUP_TIMEOUT="${RLLM_RTP_STARTUP_TIMEOUT:-3600}"
export RLLM_RTP_UPDATE_BUFFER_BYTES="${RLLM_RTP_UPDATE_BUFFER_BYTES:-8589934592}"
export QWEN36_PROJECT_ROOT="$PROJECT_ROOT"
export QWEN36_MODEL_PATH="$MODEL_PATH"
export QWEN36_RL_ROOT="$PROJECT_ROOT"
export RLLM_PYTHON_TOOL_TRAIN_JSONL="${RLLM_PYTHON_TOOL_TRAIN_JSONL:-$TRAIN_ROOT/data/eight-original-prompts.jsonl}"
export RLLM_OPTIMIZER_AUDIT_DIR="$RUN_ROOT/optimizer"
export RLLM_QWEN36_MODEL_AUDIT_DIR="$RUN_ROOT/model-load"
export RLLM_QWEN36_TRACE_AUDIT_DIR="$RUN_ROOT/trace-linkage"
export RLLM_PYTHON_TOOL_TRACE_DIR="$RUN_ROOT/tool-episodes"
export RLLM_EPISODE_LOG_DIR="$RUN_ROOT/episodes"
export RLLM_PYTHON_TOOL_ENABLE_THINKING=0
export RLLM_PYTHON_TOOL_MAX_TURNS=3
export RLLM_PYTHON_TOOL_CALL_TOKENS="${RLLM_PYTHON_TOOL_CALL_TOKENS:-1024}"
export RLLM_PYTHON_TOOL_TOTAL_TOKENS=3072
export RLLM_REQUIRE_REAL_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_CUMEM_ENABLE=0
PYTHON="${QWEN36_TRAIN_PYTHON:-$ORIGINAL_MATH/.venv/bin/python}"
if [[ ! -f "$MODEL_PATH/config.json" ]]; then echo "Model snapshot not ready: $MODEL_PATH" >&2; exit 2; fi
# Ray adds a long session/socket suffix. RUN_ROOT is too long for AF_UNIX.
RAY_TMP_DIR="$("$PYTHON" - "$RUN_ROOT" "${QWEN36_RAY_TMP_DIR:-}" <<'PY'
import hashlib
import os
from pathlib import Path
import sys

run_root, override = sys.argv[1:]
root = Path(override).absolute() if override else Path('/tmp')/('q35122ray-'+hashlib.sha256(run_root.encode()).hexdigest()[:12])
probe = root/'session_2000-01-01_00-00-00_000000_2147483647'/'sockets'/'plasma_store999'
limit = 103 if sys.platform.startswith('darwin') else 107
if len(os.fsencode(probe)) > limit:
    raise SystemExit(
        f'Ray socket path would exceed the {limit}-byte AF_UNIX limit: {probe}. '
        'Set QWEN36_RAY_TMP_DIR to a shorter absolute path, for example /tmp/q35122ray-custom.'
    )
print(root)
PY
)"
mkdir -p "$RAY_TMP_DIR"
printf '%s\n' "$RAY_TMP_DIR" > "$RUN_ROOT/ray-tmpdir.txt"
if [[ ! -f "$RLLM_PYTHON_TOOL_TRAIN_JSONL" ]]; then
  echo "Missing acceptance dataset: $RLLM_PYTHON_TOOL_TRAIN_JSONL" >&2
  exit 2
fi
exec "$PYTHON" -m run_qwen36_rtp_training \
  rllm/backend=verl rllm.algorithm.adv_estimator=grpo \
  ++rllm.disable_thinking=true \
  rllm.algorithm.norm_adv_by_std_in_grpo=true rllm.algorithm.kl_beta=0.0 \
  rllm.algorithm.lr_schedule=constant rllm.algorithm.warmup_steps_ratio=0.0 \
  rllm.algorithm.router_replay=disabled \
  rllm.data.train_batch_size=8 rllm.data.max_prompt_length=4096 rllm.data.max_response_length=4096 \
  rllm.rollout.n=4 rllm.rollout.train.temperature=1.0 rllm.rollout.train.top_p=1.0 \
  +rllm.rollout.train.top_k=0 +rllm.rollout.val.top_k=0 "rllm.rollout.train.max_tokens=$RLLM_PYTHON_TOOL_CALL_TOKENS" \
  rllm.workflow.n_parallel_tasks=8 rllm.workflow.retry_limit=1 rllm.workflow.raise_on_error=true \
  rllm.gateway.host=127.0.0.1 rllm.gateway.port=30135 \
  rllm.episode_logging.log_episodes=true "rllm.episode_logging.episode_log_dir=$RUN_ROOT/episodes" \
  rllm.trainer.total_epochs=2 rllm.trainer.total_batches=2 "rllm.trainer.logger=['console']" \
  rllm.trainer.project_name=qwen35-122b-rtp-rl rllm.trainer.experiment_name=8gpu-grpo-python-tool \
  rllm.trainer.val_before_train=false rllm.trainer.test_freq=-1 rllm.trainer.save_freq=-1 \
  "+model.name=$MODEL_PATH" "actor_rollout_ref.model.path=$MODEL_PATH" \
  actor_rollout_ref.model.external_lib=qwen36_training_overrides \
  actor_rollout_ref.model.use_remove_padding=false actor_rollout_ref.model.use_fused_kernels=false \
  actor_rollout_ref.model.use_liger=false actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.lora_rank=0 \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.mtp.enable=false actor_rollout_ref.model.mtp.enable_train=false \
  actor_rollout_ref.model.mtp.enable_rollout=false \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true \
  actor_rollout_ref.actor.fsdp_config.param_offload=false actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.weight_decay=0.0 \
  actor_rollout_ref.actor.optim.optimizer_impl=qwen36_optimizer_audit \
  actor_rollout_ref.actor.optim.optimizer=CpuMomentAuditedAdamW \
  actor_rollout_ref.actor.entropy_coeff=0.0 actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=false actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=rtp actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=8 actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 actor_rollout_ref.rollout.checkpoint_engine.backend=naive \
  actor_rollout_ref.rollout.free_cache_engine=true actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=0 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.enforce_eager=true actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.15 actor_rollout_ref.rollout.max_num_seqs=8 \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 "trainer.default_local_dir=$RUN_ROOT/checkpoints" \
  trainer.default_hdfs_dir=null trainer.resume_mode=disable \
  +ray_init.address=local +ray_init.include_dashboard=false +ray_init.object_store_memory=8589934592 \
  "+ray_init._temp_dir=$RAY_TMP_DIR" "$@"
