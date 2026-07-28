#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LAUNCHER="${REPO_ROOT}/start_rtp_pd_cuda13_sleep.sh"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda310/bin/python3}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the DeepSeek-V4-Flash checkpoint}"
DECODE_GPUS="${DECODE_GPUS:-4,5}"
PREFILL_GPUS="${PREFILL_GPUS:-6,7}"
DECODE_PORT="${DECODE_PORT:-21000}"
PREFILL_PORT="${PREFILL_PORT:-22000}"
MAX_IDLE_GPU_MIB="${MAX_IDLE_GPU_MIB:-64}"
WORLD_SIZE=2
PORTS_PER_RANK=9

failures=0

pass() {
    printf '[PASS] %s\n' "$*"
}

fail() {
    printf '[FAIL] %s\n' "$*" >&2
    failures=$((failures + 1))
}

require_file() {
    local path="$1"
    if [[ -f "${path}" ]]; then
        pass "file exists: ${path}"
    else
        fail "missing file: ${path}"
    fi
}

require_text() {
    local path="$1"
    local text="$2"
    if rg -F --quiet -- "${text}" "${path}"; then
        pass "$(basename "${path}") contains: ${text}"
    else
        fail "$(basename "${path}") is missing: ${text}"
    fi
}

check_port_range() {
    local role="$1"
    local base="$2"
    local end=$((base + WORLD_SIZE * PORTS_PER_RANK - 1))
    local listeners
    listeners="$(ss -ltnH 2>/dev/null | awk '{split($4, a, ":"); print a[length(a)]}' | sort -nu)"
    local occupied
    occupied="$(awk -v lo="${base}" -v hi="${end}" '$1 >= lo && $1 <= hi' <<<"${listeners}")"
    if [[ -n "${occupied}" ]]; then
        fail "${role} port range ${base}-${end} is occupied: $(tr '\n' ',' <<<"${occupied}" | sed 's/,$//')"
    else
        pass "${role} port range ${base}-${end} is free"
    fi
}

check_gpu() {
    local gpu="$1"
    local row
    row="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits \
        | awk -F, -v wanted="${gpu}" '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); if ($1 == wanted) print}')"
    if [[ -z "${row}" ]]; then
        fail "GPU ${gpu} is unavailable"
        return
    fi
    local used
    used="$(awk -F, '{gsub(/[[:space:]]/, "", $4); print $4}' <<<"${row}")"
    if [[ "${used}" -le "${MAX_IDLE_GPU_MIB}" ]]; then
        pass "GPU ${gpu} is idle enough: ${used} MiB used"
    else
        fail "GPU ${gpu} is busy: ${used} MiB used (limit ${MAX_IDLE_GPU_MIB} MiB)"
    fi
}

printf 'Level3 PD CP2 preflight\n'
printf '  model:   %s\n' "${MODEL_DIR}"
printf '  decode:  GPUs %s, port %s\n' "${DECODE_GPUS}" "${DECODE_PORT}"
printf '  prefill: GPUs %s, port %s\n' "${PREFILL_GPUS}" "${PREFILL_PORT}"

command -v rg >/dev/null && pass 'rg is available' || fail 'rg is unavailable'
command -v ss >/dev/null && pass 'ss is available' || fail 'ss is unavailable'
command -v nvidia-smi >/dev/null && pass 'nvidia-smi is available' || fail 'nvidia-smi is unavailable'
[[ -x "${PYTHON_BIN}" ]] && pass "python is executable: ${PYTHON_BIN}" || fail "python is not executable: ${PYTHON_BIN}"
[[ -x "${LAUNCHER}" ]] && pass "launcher is executable: ${LAUNCHER}" || fail "launcher is not executable: ${LAUNCHER}"

BAZEL_BIN="$(command -v bazelisk || command -v bazel || true)"
if [[ -n "${BAZEL_BIN}" ]]; then
    BAZEL_OUTPUT_BASE="$(USE_BAZEL_VERSION=6.4.0 "${BAZEL_BIN}" info output_base 2>/dev/null || true)"
else
    BAZEL_OUTPUT_BASE=""
fi
if [[ -n "${BAZEL_OUTPUT_BASE}" ]]; then
    PIP_REPOS=("${BAZEL_OUTPUT_BASE}"/external/pip_gpu_cuda13_torch_*/site-packages)
    PIP_PATH="$(IFS=:; echo "${PIP_REPOS[*]}")"
    PREFLIGHT_PYTHONPATH="${REPO_ROOT}/bazel-bin:${PIP_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
    pass "Bazel CUDA13 Python repositories are available"
else
    PREFLIGHT_PYTHONPATH="${PYTHONPATH:-}"
    fail "Bazel output_base is unavailable"
fi

if [[ -d "${MODEL_DIR}" ]]; then
    pass "model directory exists: ${MODEL_DIR}"
else
    fail "model directory is missing: ${MODEL_DIR}"
fi
require_file "${MODEL_DIR}/config.json"
require_file "${MODEL_DIR}/tokenizer_config.json"

require_text "${LAUNCHER}" '--tp_size 2 --ep_size 2 --world_size 2'
require_text "${LAUNCHER}" '--cp_rotate_method ALL_GATHER'
require_text "${LAUNCHER}" '--tp_size 1 --dp_size 2 --ep_size 2 --world_size 2'
require_text "${LAUNCHER}" '--cp_rotate_method PREFILL_CP'
require_text "${LAUNCHER}" '--enable_cuda_graph 1'

if PYTHONPATH="${PREFLIGHT_PYTHONPATH}" "${PYTHON_BIN}" - <<'PY'
import grpc
import torch
from torch_memory_saver.utils import get_binary_path_from_package
import rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2

shim = get_binary_path_from_package("torch_memory_saver_hook_mode_preload")
print(f"python dependencies: torch={torch.__version__} grpc={grpc.__version__} tms={shim}")
PY
then
    pass 'Python, CUDA13 Torch, gRPC proto, and torch_memory_saver import successfully'
else
    fail 'Python dependency import failed'
fi

keeper_runtime="${REPO_ROOT}/rtp_llm/utils/multicast_keeper.py"
require_file "${keeper_runtime}"
require_text "${LAUNCHER}" 'RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER=1'
require_text "${keeper_runtime}" 'child_env.setdefault("NCCL_NVLS_ENABLE", "1")'
require_text "${keeper_runtime}" 'child_env.setdefault("TORCH_SYMM_MEM_DISABLE_MULTICAST", "0")'

if command -v ibv_devices >/dev/null && [[ $(ibv_devices 2>/dev/null | awk 'NR > 2 {count++} END {print count+0}') -gt 0 ]]; then
    pass 'RDMA devices are present'
else
    fail 'no RDMA device was detected'
fi

memlock_soft="$(ulimit -S -l)"
if [[ "${memlock_soft}" == 'unlimited' ]]; then
    pass 'memlock is unlimited'
elif command -v sudo >/dev/null && sudo -n true >/dev/null 2>&1; then
    pass "memlock is ${memlock_soft} KiB; launcher can raise it with passwordless sudo"
else
    fail "memlock is ${memlock_soft} KiB and launcher cannot raise it non-interactively"
fi

check_port_range decode "${DECODE_PORT}"
check_port_range prefill "${PREFILL_PORT}"

IFS=',' read -r -a requested_gpus <<<"${DECODE_GPUS},${PREFILL_GPUS}"
for gpu in "${requested_gpus[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    [[ -n "${gpu}" ]] && check_gpu "${gpu}"
done

printf '\nGPU compute applications (informational):\n'
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits 2>&1 || true

if ((failures > 0)); then
    printf '\nPreflight failed: %d check(s) failed. No service was started or stopped.\n' "${failures}" >&2
    exit 1
fi

printf '\nPreflight passed. No service was started or stopped.\n'
