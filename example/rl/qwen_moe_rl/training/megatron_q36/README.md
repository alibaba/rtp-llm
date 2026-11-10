# Q36 Megatron training with RTP copying refit

This opt-in recipe targets the qualified Qwen3.6-35B-A3B text model with one
node, TP4/EP1/ETP4, BF16 main gradients and weights, and CPU BF16 Adam m/v.
It uses separate native RTP weights and the full-source ACK/commit transaction.
It is not a general Megatron configuration or a shared-weight implementation.

The Python 3.12 training environment must provide Torch 2.13.0+cu130, Megatron
Core 0.19.2, Bridge 0.6.2, Transformer Engine 2.20.2, the pinned verl/rLLM
source versions and their runtime dependencies. The native process uses its
separately qualified Python 3.10 runtime, libraries and NCCL 2.29.7. The
training and native cuDNN libraries must remain isolated. The training PyTorch
overlay is a same-filesystem hardlink copy of the base Torch package, under
`$Q36_CUDNN_ROOT/pyoverlay/torch`, with the qualified cuDNN available under
`$Q36_CUDNN_ROOT/nvidia/cudnn/lib`. The launcher validates every non-bytecode
overlay file against the base inode, size and device. It also checks fixed
cuDNN SHA256 values, and each GPU-assigned worker checks its actual mappings.

Bridge's Transformers/FLA requirements and the pinned verl project's requested
versions differ; qualify the assembled versions together instead of assuming
either upstream lock covers the combined stack. The native runtime and model
weights are external artifacts and must not be committed.

Before launching, supply independently verified, locally accessible inputs:

```sh
export Q36_CODE_DIR="$REPOSITORY/example/rl/qwen_moe_rl/training/megatron_q36"
export Q36_OPT_SOURCE_DIR="$Q36_CODE_DIR"
export Q36_ADAPTER_DIR="$REPOSITORY/example/rl/qwen_moe_rl/adapter"
export Q36_NATIVE_OVERLAY="$REPOSITORY/example/rl/qwen_moe_rl/rtp/q36_native"
export Q36_MODEL_PATH="$VERIFIED_MODEL_SNAPSHOT"
export Q36_SR_MANIFEST="$PRIVATE_OUTPUT_DIR/q36-provider-ddp-sr-manifest.json"
export Q36_SR_MANIFEST_SHA256="$("$TRAINING_PYTHON" "$Q36_CODE_DIR/prepare_sr_manifest.py" "$Q36_SR_MANIFEST")"
export RLLM_PYTHON_TOOL_TRAIN_JSONL="$VERIFIED_DATASET"
export Q36_DATASET_SHA256="$VERIFIED_DATASET_SHA256"
export Q36_PYTHON="$TRAINING_PYTHON"
export Q36_BASE_PYTHONPATH="$VERIFIED_TRAINING_AND_GATEWAY_SOURCE_PATHS"
export Q36_TRAIN_LIBRARY_PATH="$VERIFIED_TRAINING_LIBRARY_PATHS"
export Q36_BASE_TORCH_DIR="$BASE_TORCH_PACKAGE_DIR"
export Q36_BASE_TE_DIR="$BASE_TRANSFORMER_ENGINE_PACKAGE_DIR"
export Q36_CUDNN_ROOT="$ISOLATED_CUDNN_AND_TORCH_OVERLAY_ROOT"
export Q36_CUDNN_MANIFEST="$PRIVATE_OUTPUT_DIR/cudnn-manifest.json"
export Q36_CUDNN_MANIFEST_SHA256="$("$Q36_PYTHON" "$Q36_CODE_DIR/write_cudnn_manifest.py" "$Q36_CUDNN_MANIFEST")"
export RTP_SOURCE="$RTP_CHECKOUT_WITH_QUALIFIED_NATIVE_LIBRARY"
export RTP_PYTHON="$QUALIFIED_NATIVE_PYTHON"
export RLLM_RTP_DEPLOY_ROOT="$PRIVATE_DEPLOYMENT_ROOT"
export Q36_NCCL_ROOT="$QUALIFIED_NCCL_PYTHON_ROOT"
export Q36_NATIVE_NCCL_SO="$QUALIFIED_NATIVE_NCCL_LIBRARY"
export Q36_NATIVE_LIBRARY_PATH="$QUALIFIED_NATIVE_ONLY_LIBRARY_PATHS"
export Q36_ARM_ROPE_DIR="$QUALIFIED_ARM_ROPE_SOURCE_DIR"
export Q36_ARM_ROPE_SHA256="$QUALIFIED_ARM_ROPE_HOOK_SHA256"
export Q36_FLASHINFER_CUBIN_DIR="$QUALIFIED_FLASHINFER_CUBINS"
export Q36_FLASHINFER_WORKSPACE_BASE="$PRIVATE_FLASHINFER_WORKSPACE"
export RTP_SERVER_LAUNCHER="$Q36_NATIVE_OVERLAY/serve_q36_tp4.sh"
export Q36_NATIVE_LAUNCHER_SHA256="$REVIEWED_LAUNCHER_SHA256"
export Q36_RUN_ROOT="$PRIVATE_OUTPUT_DIR/q36-rl35-megatron-$UNIQUE_TEN_HEX_ID"
mkdir "$Q36_RUN_ROOT"
bash "$Q36_CODE_DIR/run_q36_nonshared_megatron.sh"
```

Both manifest generators exclusively create their output; use fresh paths.
The SR schema is included as a compressed, encoded parameter/source metadata
artifact, without model weights. Its extractor verifies both the archive and
the full decoded SHA256 (`d824d3ac60b76ac7699dfe36f3935580fdf4033780e3ebb9f8b311e10ac0d4e0`)
before creating the file. The schema captures the qualified model/provider
layout; each worker still checks it against its actual live DDP parameters.
The metadata was derived from the official Qwen3.6-35B-A3B model configuration
and the qualified Megatron Bridge mapping; the official model is Apache-2.0.
`Q36_BASE_PYTHONPATH` must include the pinned Bridge, verl, rLLM, gateway,
complete RTP adapter and Python Tool sources and dependencies. Supply only
reviewed native library directories: including an alternate cuDNN/NCCL build
causes the native search-path guard to reject startup. A matching hash supplied
by the operator is not a substitute for independently verifying the source.

The real request guard rejects a response budget above 1024 tokens and any
prompt-plus-response budget above 8192. The recipe does not fabricate rewards
or require a nonzero update by default: an all-zero-advantage batch may be a
valid no-op. For a qualification run, set `RLLM_REQUIRE_REAL_UPDATE=1` to
require a changed source and fixed-token probe score after update. Other RTP
profiles keep their existing strict post-sync behavior. Full-source hashes,
per-source all-rank ACKs, versions and transaction commit remain mandatory in
both modes.

Before using a new deployment, perform CPU import/config/source-binding and
negative tests, then independently qualify four Ray worker UUIDs, actual
training/native library maps, complete native loading and transactional ACKs,
nonzero BF16 gradients and updates, updated-policy rollout, real exits and
owned-process cleanup. Static settings, a CPU test, or a reduced model alone do
not qualify a complete training run. This portable proposal remains subject to
those checks on its exact source hashes.
