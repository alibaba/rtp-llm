#!/usr/bin/env bash
set -euo pipefail

launch_prefill_server() {
  local script_dir repo_root checkpoint start_port cuda_devices python_bin
  local enable_cuda_graph enable_cuda_graph_debug_mode
  local decode_capture_config prefill_capture_config
  local flag_name flag_value server_runfiles server_binary
  local flashinfer_site_packages tvm_ffi_site_packages

  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd -- "${script_dir}/../.." && pwd)"
  : "${RUN_ROOT:?RUN_ROOT must point to this run artifact directory}"

  checkpoint="${CHECKPOINT_PATH:-/data0/luohaocheng.lhc/Kimi-K3-4layers-preflight}"
  start_port="${START_PORT:-27188}"
  cuda_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  python_bin="${PYTHON_BIN:-/opt/conda310/bin/python3}"
  enable_cuda_graph="${ENABLE_CUDA_GRAPH:-0}"
  enable_cuda_graph_debug_mode="${ENABLE_CUDA_GRAPH_DEBUG_MODE:-0}"
  decode_capture_config="${DECODE_CAPTURE_CONFIG:-}"
  prefill_capture_config="${PREFILL_CAPTURE_CONFIG:-}"

  for flag_name in enable_cuda_graph enable_cuda_graph_debug_mode; do
    flag_value="${!flag_name}"
    if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
      echo "${flag_name} must resolve to 0 or 1, got ${flag_value}" >&2
      exit 2
    fi
  done
  if [[ "${enable_cuda_graph_debug_mode}" == "1" && "${enable_cuda_graph}" != "1" ]]; then
    echo "ENABLE_CUDA_GRAPH_DEBUG_MODE=1 requires ENABLE_CUDA_GRAPH=1" >&2
    exit 2
  fi

  server_runfiles="${repo_root}/bazel-bin/rtp_llm/rtp_llm_server.runfiles"
  server_binary="${repo_root}/bazel-bin/rtp_llm/rtp_llm_server"
  flashinfer_site_packages="${server_runfiles}/pip_gpu_cuda13_torch_flashinfer_python/site-packages"
  tvm_ffi_site_packages="${server_runfiles}/pip_gpu_cuda13_torch_apache_tvm_ffi/site-packages"

  if [[ ! -x "${python_bin}" ]]; then
    echo "Python binary is not executable: ${python_bin}" >&2
    exit 2
  fi
  if [[ ! -x "${server_binary}" || ! -d "${server_runfiles}" ]]; then
    echo "missing Bazel server binary/runfiles: ${server_binary}" >&2
    echo "build //rtp_llm:rtp_llm_server first" >&2
    exit 2
  fi
  if [[ ! -f "${checkpoint}/config.json" ]]; then
    echo "checkpoint config not found: ${checkpoint}/config.json" >&2
    exit 2
  fi

  # Keep the UDS path below 108 bytes and large JIT caches under RUN_ROOT.
  export TMPDIR="${K3_PERF_TMPDIR:-/dev/shm/k3p-${start_port}-$$}"
  export PATH="$(dirname -- "${python_bin}"):${PATH}"
  export CUDA_VISIBLE_DEVICES="${cuda_devices}"
  export PYTHONPATH="${flashinfer_site_packages}:${tvm_ffi_site_packages}${PYTHONPATH:+:${PYTHONPATH}}"
  export PYTHONSAFEPATH=1
  export PYTHONUNBUFFERED=1
  export PYTHONFAULTHANDLER=1
  export TORCH_SHOW_CPP_STACKTRACES=1
  export TORCH_DISABLE_ADDR2LINE=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export FLA_TILELANG=0
  export RTP_LLM_STARTUP_TIMEOUT_S="${RTP_LLM_STARTUP_TIMEOUT_S:-14400}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export LOG_LEVEL="${LOG_LEVEL:-INFO}"
  export LOG_PATH="${RUN_ROOT}/logs/service"
  export START_PORT="${start_port}"
  export FRONTEND_SERVER_COUNT=1
  export MODEL_TYPE=kimi_k3
  export CHECKPOINT_PATH="${checkpoint}"
  export TOKENIZER_PATH="${TOKENIZER_PATH:-${checkpoint}}"
  export LOAD_METHOD=fastsafetensors
  export KIMI_K3_EXECUTION_MODE=optimized
  export KIMI_K3_FUSED_AG_GEMM="${KIMI_K3_FUSED_AG_GEMM:-auto}"
  export KIMI_K3_MEGA_MAX_TOKENS_PER_RANK=8192
  export KIMI_K3_ACCURACY_ALLOW_TOKEN_IDS=1
  export DSV4_MEGA_MOE_INPUT_PACKER=fused
  export DG_JIT_CACHE_DIR="${K3_PERF_DG_JIT_CACHE_DIR:-${RUN_ROOT}/runtime/deep_gemm_cache}"
  export TRITON_CACHE_DIR="${K3_PERF_TRITON_CACHE_DIR:-${RUN_ROOT}/runtime/triton_cache}"
  export FLASHINFER_WORKSPACE_BASE="${K3_PERF_FLASHINFER_WORKSPACE_BASE:-${RUN_ROOT}/runtime/flashinfer_workspace}"
  export DG_JIT_USE_NVRTC=0
  export DG_JIT_WITH_LINEINFO=0
  export DG_PRINT_CONFIGS="${DG_PRINT_CONFIGS:-1}"
  export TORCH_CUDA_PROFILER_DIR="${RUN_ROOT}/traces"
  export GEN_TIMELINE_SYNC=1

  unset REMOTE_RPC_SERVER_IP MODEL_SERVICE_CONFIG KIMI_K3_TENSOR_DUMP
  mkdir -p \
    "${TMPDIR}" \
    "${LOG_PATH}" \
    "${TORCH_CUDA_PROFILER_DIR}" \
    "${DG_JIT_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}" \
    "${FLASHINFER_WORKSPACE_BASE}" \
    "${RUN_ROOT}/work"
  echo \
    "[K3_PERF_CONFIG] role=PREFILL kda=cula(role-derived) graph=${enable_cuda_graph}" \
    >&2
  cd "${RUN_ROOT}/work"

  local -a server_args=(
    --role_type PREFILL
    --tp_size 8
    --dp_size 1
    --ep_size 8
    --world_size 8
    --local_world_size 8
    --max_seq_len 65537
    --max_context_batch_size 1
    --max_batch_tokens_size 65536
    --seq_size_per_block 4096
    --kernel_seq_size_per_block 128
    --kv_cache_mem_mb 8192
    --ssm_state_dtype fp32
    --warm_up 0
    --reuse_cache 0
    --enable_device_cache 1
    --concurrency_limit 1
    --use_deepep_moe 0
    --use_all_gather 0
    --enable_cuda_graph "${enable_cuda_graph}"
    --enable_cuda_graph_debug_mode "${enable_cuda_graph_debug_mode}"
    --load_method fastsafetensors
    --ft_core_dump_on_exception 0
    --shutdown_timeout 5
  )
  if [[ -n "${decode_capture_config}" ]]; then
    server_args+=(--decode_capture_config "${decode_capture_config}")
  fi
  if [[ -n "${prefill_capture_config}" ]]; then
    server_args+=(--prefill_capture_config "${prefill_capture_config}")
  fi
  exec "${server_binary}" "${server_args[@]}"
}

if [[ "${1:-}" == "__serve" ]]; then
  shift
  launch_prefill_server "$@"
fi

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
timestamp="$(date +%Y%m%d-%H%M%S)"
# Production PREFILL backends are role-derived and fixed to
# rs_ag / cuLA / FlashMLA / DeepGEMM MegaMoE.
run_root="${RUN_ROOT:-${HOME}/kimi_k3_perf_runs/${timestamp}-k3-rs_ag-cula-flashmla-mega-64k}"
server_log="${run_root}/launcher.log"
server_target="//rtp_llm:rtp_llm_server"
server_pid=""

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
checkpoint="$(realpath -e -- "${checkpoint}")"
checkpoint_fs="$(findmnt -T "${checkpoint}" -n -o FSTYPE)"
checkpoint_source="$(findmnt -T "${checkpoint}" -n -o SOURCE)"
case "${checkpoint}" in
  /mnt/nas/* | /mnt/nas1/*)
    echo "checkpoint must be on a local data disk, got ${checkpoint}" >&2
    exit 2
    ;;
esac
case "${checkpoint_fs}" in
  nfs* | cifs | smb* | fuse.*)
    echo "checkpoint filesystem must be local, got ${checkpoint_fs}" >&2
    exit 2
    ;;
esac
if [[ "${checkpoint_source,,}" == *nas* ]]; then
  echo "checkpoint mount source must not be NAS: ${checkpoint_source}" >&2
  exit 2
fi
if [[ ! -f "${checkpoint}/config.json" ]]; then
  echo "4-layer checkpoint not found: ${checkpoint}" >&2
  exit 2
fi
if [[ ! -f "${checkpoint}/model.safetensors.index.json" ]]; then
  echo "FastSafetensors index not found: ${checkpoint}/model.safetensors.index.json" >&2
  exit 2
fi
if curl --silent --fail "http://127.0.0.1:${start_port}/health" >/dev/null 2>&1; then
  echo "port ${start_port} already serves a healthy process; refusing to replace it" >&2
  exit 2
fi

cd "${repo_root}"
if [[ "$(hostname)" != "e01-cn-qiz4s5sfe01" ]]; then
  echo "K3 sources may only be built on L20-dev-115 (e01-cn-qiz4s5sfe01)" >&2
  exit 2
fi
if [[ "$(whoami)" != "luohaocheng.lhc" ]]; then
  echo "K3 sources must be built as luohaocheng.lhc" >&2
  exit 2
fi
if [[ ! -f /.dockerenv && ! -r /proc/1/cgroup ]]; then
  echo "run this build inside lhc_GPU" >&2
  exit 2
fi
repo_fs="$(findmnt -T "${repo_root}" -n -o FSTYPE)"
case "${repo_root}" in
  /data[0-9]*/* | /data/* | /ssd/*) ;;
  *) echo "refusing non-local source path: ${repo_root}" >&2; exit 2 ;;
esac
case "${repo_fs}" in
  nfs* | cifs | smb* | fuse.*)
    echo "refusing network source filesystem: ${repo_fs}" >&2
    exit 2
    ;;
esac
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
bazel_output_base="$(
  bazelisk "${bazel_startup_args[@]}" \
    info "${bazel_build_args[@]}" output_base
)"
bazel_output_base="$(realpath -m -- "${bazel_output_base}")"
bazel_output_fs="$(findmnt -T "${bazel_output_base}" -n -o FSTYPE)"
case "${bazel_output_base}" in
  /data[0-9]*/* | /data/* | /ssd/*) ;;
  *) echo "refusing non-local Bazel output path: ${bazel_output_base}" >&2; exit 2 ;;
esac
case "${bazel_output_fs}" in
  nfs* | cifs | smb* | fuse.*)
    echo "refusing network Bazel output filesystem: ${bazel_output_fs}" >&2
    exit 2
    ;;
esac
printf '%s\n' \
  "[build-preflight] host=$(hostname) container=lhc_GPU user=$(whoami)" \
  "[build-preflight] source=${repo_root} fs=${repo_fs}" \
  "[build-preflight] output=${bazel_output_base} fs=${bazel_output_fs}" \
  "[build-preflight] configs=--config=cuda13 --config=sm10x"
bazelisk "${bazel_startup_args[@]}" \
  build "${bazel_build_args[@]}" "${server_target}"

model_source="${repo_root}/rtp_llm/models_py/model_desc/kimi_k3.py"
server_runfiles="${repo_root}/bazel-bin/rtp_llm/rtp_llm_server.runfiles"
model_runfile="${server_runfiles}/rtp_llm/rtp_llm/models_py/model_desc/kimi_k3.py"
if [[ ! -f "${model_runfile}" ]] || ! cmp --silent "${model_source}" "${model_runfile}"; then
  echo "Bazel runfiles do not match this checkout; refusing mislabeled profiling" >&2
  sha256sum "${model_source}" "${model_runfile}" 2>/dev/null || true
  exit 2
fi

mapfile -t deep_gemm_packages < <(
  find -L "${server_runfiles}" -type d \
    -path '*/site-packages/deep_gemm' -print | sort -u
)
if [[ "${#deep_gemm_packages[@]}" -lt 1 ]]; then
  echo "DeepGEMM is missing from ${server_target} runfiles" >&2
  exit 2
fi
deep_gemm_site_packages="$(dirname -- "${deep_gemm_packages[0]}")"
SERVER_RUNFILES="${server_runfiles}" \
PYTHONPATH="${deep_gemm_site_packages}" \
  "${python_bin}" - <<'PY'
import inspect
import os

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
        f"DeepGEMM targets sm_103a, got {torch.cuda.get_device_capability(0)}"
    )
deep_gemm_import_path = os.path.abspath(deep_gemm.__file__)
server_runfiles = os.path.abspath(os.environ["SERVER_RUNFILES"])
if not deep_gemm_import_path.startswith(server_runfiles + os.sep):
    raise RuntimeError(
        f"DeepGEMM imported outside Bazel runfiles: {deep_gemm_import_path}"
    )
required = {"activation_beta", "activation_linear_beta", "fast_math"}
missing = required.difference(
    inspect.signature(deep_gemm.fp8_fp4_mega_moe).parameters
)
if missing:
    raise RuntimeError(f"DeepGEMM lacks K3 SiTU parameters: {missing}")
print(
    f"deep_gemm_import={deep_gemm_import_path} "
    f"resolved={os.path.realpath(deep_gemm.__file__)}"
)
PY

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
cp "${repo_root}/internal_source/deps/requirements_torch_gpu_cuda13.txt" \
  "${run_root}/snapshots/requirements_torch_gpu_cuda13.txt"
cp "${repo_root}/internal_source/deps/requirements_lock_torch_gpu_cuda13.txt" \
  "${run_root}/snapshots/requirements_lock_torch_gpu_cuda13.txt"
nvidia-smi -q >"${run_root}/snapshots/nvidia_smi_before.txt"

export RUN_ROOT="${run_root}"
export CHECKPOINT_PATH="${checkpoint}"
export START_PORT="${start_port}"
export PYTHON_BIN="${python_bin}"

setsid "${BASH_SOURCE[0]}" __serve >"${server_log}" 2>&1 &
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
  --output-dir "${run_root}/measurements" \
  --trace-dir "${run_root}/traces" \
  | tee "${run_root}/measurements/console.log"

nvidia-smi -q >"${run_root}/snapshots/nvidia_smi_after.txt"
echo "[done] run_root=${run_root}"
echo "[done] measured_trace_paths=${run_root}/measurements/run.json"

if [[ "${KEEP_SERVER:-0}" == "1" ]]; then
  trap - EXIT INT TERM
  echo "[service] KEEP_SERVER=1; process group ${server_pid} remains running"
fi
