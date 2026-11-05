# Qwen text-MoE RL with verl, rLLM and native RTP

This example preserves the previously developed Qwen3.6-35B-A3B,
Qwen3.5-122B-A10B and Qwen3.5-397B-A17B integration. It uses the real RTP token
RPC and versioned in-place weight updates, rLLM's AgentTrainer, verl FSDP2, GRPO,
DeepScaleR Math and the Python Tool. It contains source and CPU regressions;
model weights, datasets, captured trajectories, run reports and binaries are
not bundled.

The RTP fixes are normal modules in this repository. The historical
`rtp/qwen36_overlay.py` import name is retained as a compatibility bootstrap;
it no longer substitutes five full RTP modules. It verifies the canonical
checkout, registers the normal text aliases, and isolates this example's logs.
Use `qwen35_moe_text`; `qwen36_moe_text` remains an equivalent compatibility
alias. Ordinary multimodal models retain their existing behavior.

## Runtime and allocation

Use separate compatible interpreters for native RTP and training. The original
ARM deployment used native Python 3.10, training Python 3.11, Torch
2.11.0+cu130, Transformers 5.2 on native RTP and 5.22 on training, verl 0.8 with
its deployed rLLM 0.3 integration. Those are reproducibility dependencies,
not interchangeable with an arbitrary newer stack. Build the RTP native
libraries for the actual host architecture and CUDA target; do not reuse x86
binaries on ARM or mix the two interpreters' library directories.

`training/train_8gpu.sh` is the single-node eight-GPU 122B baseline.
`training/train_32gpu.sh` and `run_qwen397_rtp_training.py` explicitly require
eight allocated Ray nodes with four GPUs each, FSDP2 world size 32, and two
native TP16 replicas. RTP DP, PP and EP stay 1; TP is no longer forced to 1.
The entrypoint rejects a different live allocation instead of silently using
one GPU or joining an unrelated Ray cluster.

The 397B path keeps FP32 master parameters and gradients on GPU, uses BF16
forward computation and FP32 reduction, and stages bounded chunks of the two
FP32 AdamW moments from CPU. It is not CPU parameter offload. The streamed
checkpoint reader and exporter avoid materializing a full model on every rank.
Both BI and CUDA Graph are disabled in this preserved RL baseline.

Native processes are owned by PID plus creation time, process group, allocated
node and physical GPU UUID. Generation admission closes before training;
exports retain node-local CUDA IPC ownership, require every expected source and
native rank acknowledgement, then commit a version and run policy probes.
The request journal links actual delivered token IDs and raw FP32 logprobs to
complete tool trajectories. The agent enforces per-call and remaining episode
budgets independently of fixed generation defaults.

## Configure explicitly

Use a dedicated allocated Ray cluster and a fresh shared run directory. Do not
start these scripts on a four-GPU machine as a replacement for the 32-GPU job.
The caller owns cluster startup and stop; the scripts never run a global
`ray stop` or kill unrelated processes.

```bash
export RTP_SOURCE=/path/to/this/rtp-llm
export RLLM_MATH_ROOT=/path/to/compatible/rllm-math-project
export RTP_LEGACY_ROOT=/path/to/native-runtime-root
export RTP_PYTHON=/path/to/native-runtime/bin/python
export Q397_TRAIN_PYTHON=/path/to/training-runtime/bin/python
export Q397_PROJECT_ROOT="$RTP_SOURCE/example/rl/qwen_moe_rl"
export Q397_MODEL_PATH=/your/oss/mount/models/Qwen3.5-397B-A17B/REVISION/snapshot
export Q397_RUN_ROOT=/shared/fresh-run-directory
export Q397_RAY_ADDRESS=ALLOCATED_HEAD_IP:PORT
export Q397_GATEWAY_HOST=ALLOCATED_HEAD_IP
export Q397_NODES_FILE=/path/to/allocated-nodes.json
export RLLM_PYTHON_TOOL_TRAIN_JSONL=/path/to/deepscaler-selection.jsonl
bash "$Q397_PROJECT_ROOT/training/train_32gpu.sh"
```

The nodes file contains eight objects under `nodes`, each with `container_ip`
and `gpu_count: 4`. Every node must have matching paths, model snapshot, native
libraries and dependencies. Set `RTP_NATIVE_LIBRARY_DIR` and
`RTP_KERNEL_PYTHON_PATH` to the built engine/kernel artifacts. The native
launcher pins a verified NCCL 2.29.3 ARM distribution through
`adapter/q397_nccl_scope.py`; stage its documented bytes under the example's
`logs/dependencies` or explicitly provide the corresponding `Q397_NCCL_*`
settings. This helper does not download or replace system packages.

`training/prepare_training_dataset.py` requires explicit source and evaluation
dataset paths. Training does not generate or rewrite rewards to force a pass.
Preserve original model-download receipts during storage relocation; record the
old-to-new path mapping separately rather than editing historical evidence.

## Optional ARM BF16 backend

Set `RTP_ARM_BF16_ROPE_BACKEND=torch_bf16_v1` and
`RTP_ENTRYPOINT="$Q397_PROJECT_ROOT/arm-rope/entrypoint.py"` only for the
supported explicit BF16 fallback. The four source files in `arm-rope/` install
in every spawn child, keep fused QK normalization dispatch verifiable, and
record the selected backend. This is an opt-in compatibility path, not a
global replacement for `rtp_kernel`, FP8 kernels or CUDA Graph implementations.

## Validation scope

The reusable CPU tests check topology and process ownership, transaction/source
coverage, trace linkage, budgets, bounded CPU AdamW and generic padding with
its upstream reference license. The 397B recorded-evidence validator is
model- and allocation-specific; it must not be used as a generic new-model pass.
For example:

```bash
python -m unittest discover -s "$Q397_PROJECT_ROOT/adapter" -p test_multinode_contract.py
python "$Q397_PROJECT_ROOT/validation/test_rtp_request_journal_std_cpu.py" --project-root "$Q397_PROJECT_ROOT"
python -m unittest discover -s "$Q397_PROJECT_ROOT/validation" -p test_32gpu_contracts.py
```

Historical 397B execution completed two real updates on all 32 ranks and
refit/probes at versions 0, 1 and 2. Its original final exit was 1 because of a
second cleanup attempt after Ray actor ownership ended; 17 of 19 acceptance
checks passed. That failure remains a failure, not a new successful run.

Raw FP32 score transport and short independent HF agreement do not establish
full-trajectory training/inference agreement. Actual trajectory probability
difference maxima were approximately 0.561 and 0.504. FP32 GDN source inputs
also do not prove that native descriptors or FSDP forward preserve their
original FP32 values. These precision boundaries remain explicit.

This port has not rerun the complete model after packaging. It does not add or
claim Qwen3.8-2.4T training support; that model requires its own architecture,
weight mapping, distributed execution and end-to-end numerical validation.
`SOURCE_PROVENANCE.json` records the frozen input-source fingerprints before
portable path/bootstrap changes.
