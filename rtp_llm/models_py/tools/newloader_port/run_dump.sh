#!/usr/bin/env bash
# Container-side driver: load ONE model with the old loader and the new loader,
# dumping weight fingerprints from each, so compare_dumps.py can diff them.
#
# RUN THIS INSIDE THE GPU CONTAINER, from the repo root:
#   /data1/hengcang.wyd/RTP-LLM/github-opensource/
#
# Usage:
#   bash rtp_llm/models_py/tools/newloader_port/run_dump.sh <model_type> <ckpt_path> [tp_size]
#
# Single-process TP=1: needs ONE free GPU big enough for the whole model
# (e.g. ~60GB for a 30B bf16). Pick a free card and prefix CUDA_VISIBLE_DEVICES,
# else it lands on busy GPU 0 and OOMs:
#   nvidia-smi --query-gpu=index,memory.free --format=csv
#   CUDA_VISIBLE_DEVICES=3 bash run_dump.sh qwen_3_moe <ckpt>   # bazel run inherits it
#
# Output:
#   /tmp/newloader_cmp/<model_type>/old/rank*.json
#   /tmp/newloader_cmp/<model_type>/new/rank*.json
# Both dirs sync back to the Mac via mutagen for Claude to compare.
set -euo pipefail

MODEL_TYPE="${1:?model_type required, e.g. llama}"
CKPT_PATH="${2:?ckpt_path required}"
TP_SIZE="${3:-1}"

# This is a bazel monorepo: builds need the arch config matching THIS container.
# x86 H20 -> cuda12_9 ; arm -> cuda12_9_arm ; AMD -> rocm. Override via env:
#   BAZEL_CONFIG=cuda12_9_arm bash run_dump.sh ...
# This repo is driven by bazelisk; fall back to bazel. Override with BAZEL=...
if [ -n "${BAZEL:-}" ]; then :; elif command -v bazelisk >/dev/null 2>&1; then BAZEL=bazelisk; else BAZEL=bazel; fi

BAZEL_CONFIG="${BAZEL_CONFIG:-cuda12_9}"
CFG="--config=${BAZEL_CONFIG}"
echo "Using ${BAZEL} ${CFG}  (override with BAZEL=... / BAZEL_CONFIG=...)"

# .bazelrc sets this LD_LIBRARY_PATH only for build *actions* (--action_env),
# not for `bazel run` of the produced binary -> GPU torch then can't find
# libcupti.so.12 etc. Re-export the same path (from .bazelrc:50) for the run.
CUDA_LD="/lib64:/opt/conda310/lib/:/usr/local/cuda/compat/:/usr/local/nvidia/lib64:/usr/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs/:/usr/local/cuda/extras/CUPTI/lib64/"
export LD_LIBRARY_PATH="${CUDA_LD}:${LD_LIBRARY_PATH:-}"

# This container ships a stale 0-byte libnvidia-ml.so (550) that /lib/...so.1
# points at, while the real driver lib (matching nvidia-smi) sits in /usr/lib64.
# Preload the largest (= real, non-empty) libnvidia-ml so the compiled ops don't
# dlopen the broken stub -> "libnvidia-ml.so.1: file too short". Auto-detects the
# driver version so it survives driver upgrades.
NVML="$(ls -S /usr/lib64/libnvidia-ml.so.* 2>/dev/null | head -1 || true)"
if [ -n "${NVML}" ] && [ -s "${NVML}" ]; then
  export LD_PRELOAD="${NVML}:${LD_PRELOAD:-}"
  echo "Preloading NVML: ${NVML}"
fi

# Write dumps INSIDE the repo tree so they sync back to the Mac via mutagen
# (Claude can then read/compare them). /tmp does NOT sync. Override with OUT=...
OUT="${OUT:-rtp_llm/models_py/tools/newloader_port/_dumps/${MODEL_TYPE}}"
OLD_DIR="${OUT}/old"
NEW_DIR="${OUT}/new"
rm -rf "${OUT}"
mkdir -p "${OLD_DIR}" "${NEW_DIR}"

DRIVER="//rtp_llm/models_py/tools/newloader_port:load_once"

# The driver takes NO argv (rtp_llm parses sys.argv itself); pass via env.
export MODEL_TYPE CHECKPOINT_PATH TOKENIZER_PATH TP_SIZE
CHECKPOINT_PATH="${CKPT_PATH}"
TOKENIZER_PATH="${CKPT_PATH}"

# Force scratch loading: the fastsafetensors path needs pinned memory / GDS which
# fails in this container ("Failed to allocate pinned memory: invalid argument").
# scratch yields identical weights, just a different load mechanism. Both loaders
# honor LOAD_METHOD (old via FakeModelLoader, new via _resolve_load_method).
# Override with LOAD_METHOD=fastsafetensors if you ever want to test that path.
export LOAD_METHOD="${LOAD_METHOD:-scratch}"
echo "LOAD_METHOD=${LOAD_METHOD}"

# Build once so the two runs don't each re-trigger a build banner.
"${BAZEL}" build "${CFG}" "${DRIVER}"

echo "==== [1/2] OLD loader dump -> ${OLD_DIR} ===="
USE_NEW_LOADER=0 DUMP_WEIGHTS="${OLD_DIR}" \
  "${BAZEL}" run "${CFG}" "${DRIVER}"

echo "==== [2/2] NEW loader dump -> ${NEW_DIR} ===="
USE_NEW_LOADER=1 DUMP_WEIGHTS="${NEW_DIR}" \
  "${BAZEL}" run "${CFG}" "${DRIVER}"

echo
echo "Done. Now compare (works on Mac or container, pure stdlib):"
echo "  python rtp_llm/models_py/tools/newloader_port/compare_dumps.py \\"
echo "      --old-dir ${OLD_DIR} --new-dir ${NEW_DIR}"
