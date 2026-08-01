#!/usr/bin/env bash
set -euo pipefail

# Keep the one-command path usable under `docker exec -u`, which can preserve
# root's HOME while switching the effective user.
if [[ -z "${HOME:-}" || ! -d "${HOME}" || ! -w "${HOME}" ]]; then
  resolved_home="$(
    getent passwd "$(id -u)" 2>/dev/null | awk -F: '{print $6}'
  )"
  if [[ -z "${resolved_home}" || ! -d "${resolved_home}" || ! -w "${resolved_home}" ]]; then
    echo "HOME is not writable and the account home could not be resolved" >&2
    exit 2
  fi
  export HOME="${resolved_home}"
fi

if [[ -z "${TMPDIR:-}" || ! -d "${TMPDIR}" || ! -w "${TMPDIR}" ]]; then
  export TMPDIR="${KIMI_K3_BUILD_TMPDIR:-/tmp/k3p-build-$(id -u)}"
  mkdir -p "${TMPDIR}"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
python_bin="${PYTHON_BIN:-/opt/conda310/bin/python3}"
checkpoint="${CHECKPOINT_PATH:-/data0/luohaocheng.lhc/Kimi-K3-4layers-preflight}"
start_port="${START_PORT:-27188}"
kda_comm_backend="${KIMI_K3_KDA_COMM_BACKEND:-rs_ag}"
timestamp="$(date +%Y%m%d-%H%M%S)"
kda_backend="${KIMI_K3_KDA_BACKEND:-cula}"
run_root="${RUN_ROOT:-${HOME}/kimi_k3_perf_runs/${timestamp}-k3-${kda_comm_backend}-${kda_backend}-mega-64k}"
ops_overlay="${run_root}/runtime/ops"
server_log="${run_root}/launcher.log"
server_target="//example/kimi_k3_prefill_perf:kimi_k3_prefill_server"
server_pid=""

if [[ "${kda_backend}" != "cula" && "${kda_backend}" != "flash_kda" ]]; then
  echo "KIMI_K3_KDA_BACKEND must be cula or flash_kda, got ${kda_backend}" >&2
  exit 2
fi

cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM -- "-${server_pid}" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "${server_pid}" 2>/dev/null || return 0
      sleep 1
    done
    kill -KILL -- "-${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

mkdir -p \
  "${run_root}/runtime" \
  "${run_root}/measurements" \
  "${run_root}/snapshots" \
  "${run_root}/traces"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python not found: ${python_bin}; run this script inside lhc_GPU" >&2
  exit 2
fi
if [[ ! -f "${repo_root}/deps/http.bzl" ]]; then
  echo "RTP internal_source is missing: ${repo_root}/deps is unresolved" >&2
  echo "place this checkout under the normal RTP-LLM outer repository layout" >&2
  exit 2
fi
if [[ ! -f "${checkpoint}/config.json" ]]; then
  echo "4-layer checkpoint not found: ${checkpoint}" >&2
  exit 2
fi
if [[ "${kda_comm_backend}" != "rs_ag" && "${kda_comm_backend}" != "a2a" ]]; then
  echo "KIMI_K3_KDA_COMM_BACKEND must be rs_ag or a2a" >&2
  exit 2
fi
if curl --silent --fail "http://127.0.0.1:${start_port}/health" >/dev/null 2>&1; then
  echo "port ${start_port} already serves a healthy process; refusing to replace it" >&2
  exit 2
fi

(
  cd "${script_dir}/wheels"
  sha256sum --check SHA256SUMS
)
rm -rf "${ops_overlay}"
"${python_bin}" -m pip install \
  --no-deps --target "${ops_overlay}" \
  "${script_dir}/wheels/cuda_linear_attention-0.1.2+rtp.4db9fb9.1-cp310-cp310-linux_x86_64.whl" \
  "${script_dir}/wheels/deep_gemm-2.6.1-cp310-cp310-linux_x86_64.whl" \
  "${script_dir}/wheels/flash_kda-0.0.1-cp310-cp310-linux_x86_64.whl" \
  "${script_dir}/wheels/flash_linear_attention-0.5.0+rtp.3a9ce1c.3-py3-none-any.whl"

KDA_BACKEND="${kda_backend}" PYTHONPATH="${ops_overlay}" "${python_bin}" - <<'PY'
import inspect
import os
from importlib.metadata import version
from pathlib import Path

import deep_gemm
import torch

if torch.__version__ != "2.11.0+cu130" or torch.version.cuda != "13.0":
    raise RuntimeError(
        f"expected torch 2.11.0+cu130/CUDA 13.0, got "
        f"{torch.__version__}/{torch.version.cuda}"
    )
if torch.cuda.device_count() < 8:
    raise RuntimeError(f"expected 8 visible GPUs, got {torch.cuda.device_count()}")
if torch.cuda.get_device_capability(0) != (10, 3):
    raise RuntimeError(
        f"operator wheels target sm_103a, got "
        f"{torch.cuda.get_device_capability(0)}"
    )
if version("flash-linear-attention") != "0.5.0+rtp.3a9ce1c.3":
    raise RuntimeError("unexpected flash-linear-attention version")
if version("cuda-linear-attention") != "0.1.2+rtp.4db9fb9.1":
    raise RuntimeError("unexpected cuda-linear-attention version")
overlay = Path(os.environ["PYTHONPATH"].split(os.pathsep, 1)[0])
if not (overlay / "cula" / "kda" / "chunk.py").is_file():
    raise RuntimeError("cuLA chunk_kda Python entrypoint is unavailable")
if not list((overlay / "cula").glob("_cudac*.so")):
    raise RuntimeError("cuLA sm_103a extension is unavailable")
if os.environ["KDA_BACKEND"] == "flash_kda":
    import flash_kda
    import flash_kda_C

    if not hasattr(flash_kda, "get_workspace_size"):
        raise RuntimeError("FlashKDA Python API is incomplete")
    print(f"flash_kda={os.path.realpath(flash_kda.__file__)}")
    print(f"flash_kda_C={os.path.realpath(flash_kda_C.__file__)}")
required = {"activation_beta", "activation_linear_beta", "fast_math"}
missing = required.difference(
    inspect.signature(deep_gemm.fp8_fp4_mega_moe).parameters
)
if missing:
    raise RuntimeError(f"DeepGEMM wheel lacks K3 SiTU parameters: {missing}")
print(f"deep_gemm={os.path.realpath(deep_gemm.__file__)}")
print(f"fla={overlay / 'fla'}")
print(f"cula={overlay / 'cula'}")
PY

cd "${repo_root}"
bazel_startup_args=()
if [[ -n "${KIMI_K3_BAZEL_OUTPUT_BASE:-}" ]]; then
  mkdir -p "${KIMI_K3_BAZEL_OUTPUT_BASE}"
  bazel_startup_args+=("--output_base=${KIMI_K3_BAZEL_OUTPUT_BASE}")
fi
bazel_build_args=("--config=cuda13" "--config=sm10x")
if [[ -n "${KIMI_K3_XGRAMMAR_OVERRIDE:-}" ]]; then
  if [[ ! -d "${KIMI_K3_XGRAMMAR_OVERRIDE}" ]]; then
    echo "xgrammar override not found: ${KIMI_K3_XGRAMMAR_OVERRIDE}" >&2
    exit 2
  fi
  bazel_build_args+=(
    "--override_repository=xgrammar=${KIMI_K3_XGRAMMAR_OVERRIDE}"
  )
fi
bazelisk "${bazel_startup_args[@]}" \
  build "${bazel_build_args[@]}" "${server_target}"

model_source="${repo_root}/rtp_llm/models_py/model_desc/kimi_k3.py"
model_runfile="${repo_root}/bazel-bin/example/kimi_k3_prefill_perf/kimi_k3_prefill_server.runfiles/rtp_llm/rtp_llm/models_py/model_desc/kimi_k3.py"
if [[ ! -f "${model_runfile}" ]] || ! cmp --silent "${model_source}" "${model_runfile}"; then
  echo "Bazel runfiles do not match this checkout; refusing mislabeled profiling" >&2
  sha256sum "${model_source}" "${model_runfile}" 2>/dev/null || true
  exit 2
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD >"${run_root}/snapshots/git_head.txt"
  git status --short >"${run_root}/snapshots/git_status.txt"
  git diff HEAD --stat >"${run_root}/snapshots/git_diff_stat.txt"
else
  printf '%s\n' "source snapshot has no .git metadata" \
    >"${run_root}/snapshots/git_head.txt"
  : >"${run_root}/snapshots/git_status.txt"
  : >"${run_root}/snapshots/git_diff_stat.txt"
fi
cp "${script_dir}/wheels/SHA256SUMS" "${run_root}/snapshots/operator_sha256.txt"
nvidia-smi -q >"${run_root}/snapshots/nvidia_smi_before.txt"

export RUN_ROOT="${run_root}"
export OPS_OVERLAY="${ops_overlay}"
export CHECKPOINT_PATH="${checkpoint}"
export START_PORT="${start_port}"
export PYTHON_BIN="${python_bin}"
export KIMI_K3_KDA_BACKEND="${kda_backend}"
export KIMI_K3_KDA_COMM_BACKEND="${kda_comm_backend}"

setsid "${script_dir}/launch_prefill_server.sh" >"${server_log}" 2>&1 &
server_pid=$!
echo "${server_pid}" >"${run_root}/service.pid"
echo "[service] pid=${server_pid} log=${server_log}"

deadline=$((SECONDS + 14400))
until curl --silent --fail "http://127.0.0.1:${start_port}/health" >/dev/null; do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "service exited before becoming healthy; tail follows" >&2
    tail -200 "${server_log}" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "timed out waiting for service health" >&2
    exit 1
  fi
  sleep 5
done
echo "[service] healthy on port ${start_port}"

"${python_bin}" "${script_dir}/prefill_workload.py" \
  --base-url "http://127.0.0.1:${start_port}" \
  --length 65536 \
  --backend "${kda_backend}" \
  --kda-comm-backend "${kda_comm_backend}" \
  --output-dir "${run_root}/measurements" \
  --trace-dir "${run_root}/traces" \
  | tee "${run_root}/measurements/console.log"

nvidia-smi -q >"${run_root}/snapshots/nvidia_smi_after.txt"
echo "[done] run_root=${run_root}"
echo "[done] rank0_trace=${run_root}/traces/k3_${kda_comm_backend}_${kda_backend}_mega_prefill_65536_steady_wr0_1.json"

if [[ "${KEEP_SERVER:-0}" == "1" ]]; then
  trap - EXIT INT TERM
  echo "[service] KEEP_SERVER=1; process group ${server_pid} remains running"
fi
