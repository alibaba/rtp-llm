#!/usr/bin/env bash
# Q36 TP4 copying-refit recipe. Paths and qualified hashes are explicit inputs.
set -euo pipefail
ulimit -c 0
MODEL="${Q36_MODEL_PATH:?Supply the qualified official snapshot}"
export Q36_MODEL_PATH="$MODEL"
export Q36_DATASET_SHA256="${Q36_DATASET_SHA256:?Supply the verified dataset SHA256}"
export Q36_CUDNN_MANIFEST="${Q36_CUDNN_MANIFEST:?Supply a pinned cuDNN library manifest}"
export Q36_CUDNN_MANIFEST_SHA256="${Q36_CUDNN_MANIFEST_SHA256:?Supply the manifest SHA256}"
export Q36_NATIVE_LAUNCHER_SHA256="${Q36_NATIVE_LAUNCHER_SHA256:?Supply the qualified launcher SHA256}"
: "${Q36_PYTHON:?Supply training Python}" "${Q36_BASE_PYTHONPATH:?Supply training source/dependency paths}"
: "${Q36_TRAIN_LIBRARY_PATH:?Supply a qualified training-only dynamic library path}"
: "${Q36_ADAPTER_DIR:?Supply the qualified RTP adapter source directory}"
: "${Q36_CUDNN_ROOT:?Supply the qualified cuDNN root}" "${RTP_SOURCE:?Supply RTP source}" "${RTP_PYTHON:?Supply native Python}"
: "${Q36_BASE_TORCH_DIR:?Supply the base Torch package directory for overlay identity checks}"
: "${Q36_BASE_TE_DIR:?Supply the base Transformer Engine package directory for mapping checks}"
: "${RLLM_RTP_DEPLOY_ROOT:?Supply the owned RTP deployment root}" "${RLLM_PYTHON_TOOL_TRAIN_JSONL:?Supply the training dataset}"
RUN="${Q36_RUN_ROOT:?Supply a fresh, supervised 35B run directory}"
export Q36_CODE_DIR="${Q36_CODE_DIR:?Supply the verified private 35B worker code directory}"
export Q36_OPT_SOURCE_DIR="${Q36_OPT_SOURCE_DIR:?Supply the verified nonshared optimizer source directory}"
export Q36_NATIVE_OVERLAY="${Q36_NATIVE_OVERLAY:?Supply the verified pure-TP native overlay}"
export Q36_SR_MANIFEST="${Q36_SR_MANIFEST:?Supply the audited 40-layer SR manifest}"
export Q36_SR_MANIFEST_SHA256="${Q36_SR_MANIFEST_SHA256:?Supply the audited manifest SHA256}"
export RTP_SERVER_LAUNCHER="${RTP_SERVER_LAUNCHER:?Supply the qualified pure-TP4 native launcher}"
export RTP_MODEL_PATH="$MODEL" RTP_TP_SIZE=4 RTP_DP_SIZE=1 RTP_EP_SIZE=1
export RTP_LLM_RL_MODE=1 RTP_RL_SHARED_BACKING=0
export Q36_RTP_PROFILE_REFIT=1
export Q36_REFIT_BOUNDED_DIGEST=1
unset RTP_LLM_RL_START_PAUSED RTP_LLM_RL_WORLD72 RTP_LLM_RL_METADATA_BOOTSTRAP RTP_RL_SHARED_PROVIDER
export WARM_UP=0 MODEL_WARM_UP=0 ENABLE_CUDA_GRAPH=0 ENABLE_NATIVE_CUDA_GRAPH=0 ENABLE_LAYER_MICRO_BATCH=0
export LOAD_PYTHON_MODEL=1 RLLM_RTP_GPUS_PER_NODE=4
mkdir -p "$RLLM_RTP_DEPLOY_ROOT/q36-runs"
mkdir "$RLLM_RTP_DEPLOY_ROOT/q36-runs/${RUN##*/}"
export RLLM_RTP_ACCEPTANCE_DIR="$RLLM_RTP_DEPLOY_ROOT/q36-runs/${RUN##*/}/rtp"
mkdir "$RLLM_RTP_ACCEPTANCE_DIR"
export RLLM_PYTHON_TOOL_ENABLE_THINKING=0
unset RLLM_PYTHON_TOOL_REASONING_EFFORT
export RLLM_PYTHON_TOOL_MAX_TURNS=2 RLLM_PYTHON_TOOL_CALL_TOKENS=1024 RLLM_PYTHON_TOOL_TOTAL_TOKENS=2048
export RLLM_PYTHON_TOOL_HTTP_TIMEOUT=1200 RLLM_PYTHON_TOOL_HTTP_MAX_RETRIES=0 RLLM_RTP_TOKEN_TIMEOUT=900
export RLLM_PYTHON_TOOL_TRACE_DIR="$RUN/tool-episodes" RLLM_EPISODE_LOG_DIR="$RUN/episodes"
export RLLM_QWEN36_TRACE_AUDIT_DIR="$RUN/trace-linkage"
export RLLM_REQUIRE_REAL_UPDATE="${RLLM_REQUIRE_REAL_UPDATE:-0}"
[[ "$RLLM_REQUIRE_REAL_UPDATE" == 0 || "$RLLM_REQUIRE_REAL_UPDATE" == 1 ]] || { echo 'RLLM_REQUIRE_REAL_UPDATE must be 0 or 1' >&2; exit 2; }
export Q36_ZERO_UPDATE_PROFILE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export HYDRA_FULL_ERROR=1
export CUDNN_HOME="$Q36_CUDNN_ROOT/nvidia/cudnn"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:$Q36_TRAIN_LIBRARY_PATH"
export LD_PRELOAD="$CUDNN_HOME/lib/libcudnn.so.9"
export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1 NVTE_UNFUSED_ATTN=0
export PYTHONPATH="$Q36_CODE_DIR:$Q36_OPT_SOURCE_DIR:$Q36_NATIVE_OVERLAY:$Q36_CUDNN_ROOT/pyoverlay:$Q36_ADAPTER_DIR:$Q36_BASE_PYTHONPATH"
exec "$Q36_PYTHON" -m run_q36_nonshared_megatron \
  rllm/backend=verl model_engine=megatron rllm.algorithm.adv_estimator=grpo \
  rllm.data.train_batch_size=2 rllm.data.max_response_length=1024 rllm.disable_thinking=true rllm.workflow.n_parallel_tasks=4 \
  rllm.workflow.retry_limit=1 rllm.rollout.n=4 rllm.trainer.total_batches=2 \
  rllm.trainer.total_epochs=2 rllm.trainer.save_freq=-1 \
  "actor_rollout_ref.model.path=$MODEL" actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.mtp.enable=false actor_rollout_ref.model.mtp.enable_train=false \
  actor_rollout_ref.model.mtp.enable_rollout=false \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=4 \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
  actor_rollout_ref.actor.megatron.context_parallel_size=1 \
  actor_rollout_ref.actor.megatron.use_megatron_fsdp=false \
  actor_rollout_ref.actor.megatron.param_offload=false actor_rollout_ref.actor.megatron.optimizer_offload=false \
  actor_rollout_ref.actor.megatron.use_distributed_optimizer=false \
  +actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=false \
  +actor_rollout_ref.actor.megatron.override_ddp_config.overlap_grad_reduce=false \
  +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=false \
  actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=fused \
  +actor_rollout_ref.actor.megatron.override_transformer_config.mtp_num_layers=0 \
  actor_rollout_ref.actor.megatron.dtype=bfloat16 \
  actor_rollout_ref.actor.optim.optimizer=adam actor_rollout_ref.actor.optim.use_precision_aware_optimizer=false \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=true \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=false \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=rtp actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 actor_rollout_ref.rollout.checkpoint_engine.backend=naive \
  trainer.nnodes=1 trainer.n_gpus_per_node=4 "hydra.run.dir=$RUN/hydra" "$@"
