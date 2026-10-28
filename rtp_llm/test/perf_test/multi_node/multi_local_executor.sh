#!/bin/bash
set -x;

enter_work_dir() {
  # Determine work dir: use env HIPPO_APP_WORKDIR if it is set, otherwise use home
  export SUB_DIR=${FT_SUB_DIR:-"."}
  export WORK_DIR=${FT_WORK_DIR:-${HIPPO_APP_WORKDIR:-$HOME}}/$SUB_DIR
  if [ ! -d "$WORK_DIR" ]; then
    mkdir -p $WORK_DIR
  fi
  cd $WORK_DIR
}

kill_processes() {
  # Kill all running processes related to rtp_llm and conda310
  echo "[$(date)] Starting process termination on $HOSTNAME";
  (ps axuww | grep rtp_llm_rank | grep -v run_worker.sh | awk '{print $2}' | xargs kill -9) || true;
  (ps axuww | grep rtp_llm_backend_server | grep -v run_worker.sh | awk '{print $2}' | xargs kill -9) || true;
  (ps axuww | grep rtp_llm_frontend_server_ | grep -v run_worker.sh | awk '{print $2}' | xargs kill -9) || true;
  (ps axuww | grep /opt/conda310/bin/python | grep -v run_worker.sh | grep -v multi_benchmark.py | awk '{print $2}' | xargs kill -9) || true;
  # Verify processes have been terminated
  echo "[$(date)] Verifying processes have been terminated on $HOSTNAME";
  (ps axuww | grep rtp_llm_rank | grep -v grep | grep -v run_worker.sh) || true;
  (ps axuww | grep rtp_llm_backend_server | grep -v grep | grep -v run_worker.sh) || true;
  (ps axuww | grep rtp_llm_frontend_server_ | grep -v grep | grep -v run_worker.sh) || true;
  (ps axuww | grep /opt/conda310/bin/python | grep -v run_worker.sh | grep -v multi_benchmark.py | awk '{print $2}' | xargs kill -9) || true;
}

checkout_code() {
  CLEAR_BAZEL_CACHE=false
  if [ -z "$OPEN_SOURCE_URL" ]; then
    if [ -d "RTP-LLM" ] && [ ! -d "RTP-LLM/internal_source" ]; then
      rm -rf RTP-LLM;
      CLEAR_BAZEL_CACHE=true;
    fi
    if [ ! -d "RTP-LLM" ]; then
      (git clone $GIT_REPO_URL) || exit 1;
    fi
    cd RTP-LLM;
    (git fetch origin && git reset --hard $GIT_CHECKOUT_REF) || exit 1;
    echo "开始更新子模块..."
    (git submodule sync --recursive) || exit 1;
    (git submodule update --init --recursive --remote) || exit 1;
    cd github-opensource;
    (git fetch origin && git reset --hard $OPEN_SOURCE_REF) || exit 1;
    echo "更新子模块成功"
    echo "当前commit id: $OPEN_SOURCE_REF"
    echo "开始下载大文件..."
    (sh .githooks/post-checkout) || exit 1;
    echo "成功下载大文件..."
    echo "子模块更新完成"
  else
    if [ -d "RTP-LLM" ] && [ -d "RTP-LLM/internal_source" ]; then
      rm -rf RTP-LLM;
      CLEAR_BAZEL_CACHE=true;
    fi
    if [ ! -d "RTP-LLM" ]; then
      mkdir -p RTP-LLM;
    fi
    cd RTP-LLM;
    if [ ! -d "github-opensource" ]; then
      (git clone $OPEN_SOURCE_URL github-opensource) || exit 1;
    fi
    cd github-opensource;
    (git fetch origin && git reset --hard $OPEN_SOURCE_REF) || exit 1;
    echo "当前commit id: $OPEN_SOURCE_REF"
    echo "开始下载大文件..."
    (sh .githooks/post-checkout) || exit 1;
    echo "成功下载大文件..."
  fi
  if [ "$CLEAR_BAZEL_CACHE" = true ]; then
    configure_native_install_platform || exit 1;
    (rm -rf $WORK_DIR/bazel_cache) || true;
    (bazelisk clean --expunge) || true;
  fi
}

configure_native_install_platform() {
  if [ "${BUILD_FROM_SCRATCH:-2}" -le 0 ]; then
    return 0
  fi
  local native_install_config
  native_install_config=$(/opt/conda310/bin/python3 - "$PWD" <<'RTP_NATIVE_CONFIG_PY'
import importlib.util
import os
from pathlib import Path
import platform
import re
import shlex
import sys


def resolve_native_install_config(root, compile_args, install_args, install_is_set, machine):
    """Match direct Native platform selectors, without probing SDKs or interpreting Bazel aliases."""
    spec = importlib.util.spec_from_file_location("rtp_perf_platform", Path(root) / "_build/platform.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    supported = set(module.get_platform_config_versions()) & set(module.get_config_to_extras())

    def selected(raw, label):
        tokens = shlex.split(raw)
        names = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "--config":
                i += 1
                if i == len(tokens) or tokens[i].startswith("-"):
                    raise ValueError("Missing --config value in " + label)
                names.append(tokens[i])
            elif token.startswith("--config="):
                names.append(token.split("=", 1)[1])
            i += 1
        direct = supported.intersection(names)
        if len(direct) != 1:
            raise ValueError(label + " must specify exactly one supported direct platform; aliases are not resolved")
        return next(iter(direct)), names

    def for_machine(name):
        if machine == "aarch64" and name.startswith("cuda") and not name.endswith("_arm"):
            arm = name + "_arm"
            if arm not in supported:
                raise ValueError("No defined ARM Native dependency configuration for " + name)
            return arm
        if machine != "aarch64" and name.endswith("_arm"):
            raise ValueError("ARM dependency configuration on a non-ARM host")
        return name

    compile_platform, compile_names = selected(compile_args, "BAZEL_BUILD_ARGS")
    if set(compile_names).intersection(module.DEPRECATED_PIP_PLATFORM_CONFIGS):
        raise ValueError("BAZEL_BUILD_ARGS requests a platform deprecated by Native metadata")
    # The existing unquoted shell variable expansion does not reinterpret quote
    # characters inside its value. Do not validate an argv different from Bazel's.
    compile_pattern = r"(?<!\S)--config(?:=|\s+)" + re.escape(compile_platform) + r"(?=\s|$)"
    if len(re.findall(compile_pattern, compile_args)) != compile_names.count(compile_platform):
        raise ValueError("Quoted compile platform selector does not match the original shell argv")
    wanted = for_machine(compile_platform)
    if not install_is_set:
        return "--config=" + wanted
    install_platform, names = selected(install_args, "RTP_BAZEL_CONFIG")
    if set(names).intersection(module.DEPRECATED_PIP_PLATFORM_CONFIGS):
        raise ValueError("RTP_BAZEL_CONFIG requests a platform deprecated by Native metadata")
    if for_machine(install_platform) != wanted:
        raise ValueError("Conflicting explicit platform selectors in BAZEL_BUILD_ARGS and RTP_BAZEL_CONFIG")
    # Preserve other install arguments verbatim. Native metadata requires the
    # --config= form and does not append _arm to an explicit CUDA selector.
    pattern = r"(?<!\S)--config(?:=|\s+)" + re.escape(install_platform) + r"(?=\s|$)"
    normalized, count = re.subn(pattern, "--config=" + wanted, install_args)
    if count != names.count(install_platform):
        raise ValueError("Quoted platform selector is not supported by Native metadata")
    return normalized


if __name__ == "__main__":
    try:
        print(resolve_native_install_config(sys.argv[1], os.environ.get("BAZEL_BUILD_ARGS", ""),
              os.environ.get("RTP_BAZEL_CONFIG", ""), "RTP_BAZEL_CONFIG" in os.environ, platform.machine()))
    except (OSError, ValueError, ImportError, AttributeError) as error:
        print("Native dependency configuration preflight failed: " + str(error), file=sys.stderr)
        raise SystemExit(1)
RTP_NATIVE_CONFIG_PY
  ) || return 1
  export RTP_BAZEL_CONFIG="$native_install_config"
}

install_requirements() {
  if [ "${BUILD_FROM_SCRATCH:-2}" -gt 1 ]; then
    export UV_SKIP_WHEEL_FILENAME_CHECK=${UV_SKIP_WHEEL_FILENAME_CHECK:-1}
    # Install Python dependencies via the current editable-install workflow.
    (/opt/conda310/bin/uv pip install --compile-bytecode --link-mode hardlink --python /opt/conda310/bin/python3 'setuptools>=64.0,<82' wheel 'tomli; python_version < "3.11"') || exit 1;
    (RTP_SKIP_BAZEL_BUILD=1 /opt/conda310/bin/uv pip install --compile-bytecode --link-mode hardlink --python /opt/conda310/bin/python3 -e ".[dev]" --no-build-isolation) || exit 1;
  fi
}

build_code() {
  if [ "${BUILD_FROM_SCRATCH:-2}" -gt 0 ]; then
    # Kill bazel build processes
    (ps axuww | grep 'bazelisk --batch --output_user_root' | grep -v grep | awk '{print $2}' | xargs kill -9) || true;
    # Build with all arguments
    BAZEL_TARGETS=${BAZEL_TARGETS:-"//:th_transformer //:rtp_compute_ops //:th_transformer_config"}
    (bazelisk --batch --output_user_root=$WORK_DIR/bazel_cache build ${BAZEL_TARGETS} ${BAZEL_BUILD_ARGS}) || exit 1;
    # Create symbolic links for proto files
    bazel_subdir=k8-opt
    if [ -d "bazel-out/aarch64-opt" ]; then
      bazel_subdir=aarch64-opt
    fi
    (ln -sf ../../../../bazel-out/${bazel_subdir}/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2_grpc.py rtp_llm/cpp/model_rpc/proto/) || exit 1;
    (ln -sf ../../../../bazel-out/${bazel_subdir}/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2.py rtp_llm/cpp/model_rpc/proto/) || exit 1;
    (ln -sf ../../../../bazel-out/${bazel_subdir}/bin/rtp_llm/cpp/model_rpc/proto/flexlb_schedule_service_pb2_grpc.py rtp_llm/cpp/model_rpc/proto/) || exit 1;
    (ln -sf ../../../../bazel-out/${bazel_subdir}/bin/rtp_llm/cpp/model_rpc/proto/flexlb_schedule_service_pb2.py rtp_llm/cpp/model_rpc/proto/) || exit 1;
    (ln -sf ../../../../bazel-out/${bazel_subdir}/bin/rtp_llm/cpp/cuda/deep_gemm/cutlass_hdr rtp_llm/cpp/cuda/deep_gemm/cutlass_hdr) || exit 1;
    (/opt/conda310/bin/uv pip install --compile-bytecode --link-mode hardlink --python /opt/conda310/bin/python bazel-bin/rtp_llm/rtp_llm_deep_gemm-0.2.0-py3-none-any.whl ) || exit 1;
  fi
}

configure_env() {
  # Configure fixed environment variables
  export FT_DISABLE_CUSTOM_AR=1;
  export MLA_OPS_TYPE="AUTO";
  export PYTHONPATH="./";
  export PYTHON_BIN="/opt/conda310/bin/python";
  export FT_CORE_DUMP_ON_EXCEPTION=1;
  export NCCL_TOPO_DUMP_FILE="/tmp/nccl_topo.xml";
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib64:/usr/local/nvidia/lib64/:/usr/local/cuda/lib64/:/usr/local/cuda/extras/CUPTI/lib64/";
  # Set CUDA_VISIBLE_DEVICES based on LOCAL_WORLD_SIZE if not already set
  if [ -z "$CUDA_VISIBLE_DEVICES" ] && [ -n "$LOCAL_WORLD_SIZE" ]; then
    CUDA_DEVICES=""
    for i in $(seq 0 $((LOCAL_WORLD_SIZE - 1))); do
      if [ -z "$CUDA_DEVICES" ]; then
        CUDA_DEVICES="$i"
      else
        CUDA_DEVICES="$CUDA_DEVICES,$i"
      fi
    done
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    echo "Set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (LOCAL_WORLD_SIZE=$LOCAL_WORLD_SIZE)"
  fi
  # Configure optional environment variables
  echo -e "\nActive environment variables:"
  (printenv | grep -E 'NCCL|ACCL|NVSHMEM' | sort) || exit 1
}

run_server() {
  if [ "${BUILD_FROM_SCRATCH:-2}" -eq 0 ]; then
    SERVER_CMD="$PYTHON_BIN ./rtp_llm/test/perf_test/multi_node/local_server_runner.py";
    if [ $CUDA_ASAN -eq 1 ]; then
        /usr/local/cuda/compute-sanitizer/compute-sanitizer --print-limit 10000 --target-processes all $SERVER_CMD || exit 1;
    else
      $SERVER_CMD || exit 1;
    fi
  fi
}

find_test_output_path() {
  # Get prefix of result directory
  TEST_OUTPUT_PATH_PREFIX="TEST_OUTPUT_"
  # Search for directories matching the prefix
  matching_dirs=($(ls -rt  | grep ${TEST_OUTPUT_PATH_PREFIX} | tail -n 1 2>/dev/null))
  # Check if any directories were found
  if [ ${#matching_dirs[@]} -eq 0 ]; then
    echo "[$(date)] ERROR: No directory found with prefix: $TEST_OUTPUT_PATH_PREFIX";
    echo "[$(date)] Available directories in current path:";
    ls -la | grep "^d" | awk '{print $9}' | grep -v "^\.$\|^\.\.$";
    exit 1;
  fi
  # Check if exactly one directory was found
  if [ ${#matching_dirs[@]} -gt 1 ]; then
    echo "[$(date)] ERROR: Multiple directories found with prefix: $TEST_OUTPUT_PATH_PREFIX";
    echo "[$(date)] Found directories:";
    for dir in "${matching_dirs[@]}"; do
      echo "  - $dir";
    done
    exit 1;
  fi
  # Exactly one directory found
  TEST_OUTPUT_PATH=$(readlink -f ${matching_dirs[0]})
  # Verify the directory exists and is accessible
  if [ ! -d "$TEST_OUTPUT_PATH" ]; then
    echo "[$(date)] ERROR: Directory $TEST_OUTPUT_PATH does not exist or is not accessible";
    exit 1;
  fi
}

local_build_script() {
  # Enter work dir
  enter_work_dir || exit 1;
  # Checkout code from git if not exist
  checkout_code || exit 1;
  # Apply patches if PATCH_COMMANDS is set
  if [ -n "$PATCH_COMMANDS" ]; then
    echo "Applying patches..."
    bash -euo pipefail -c "$PATCH_COMMANDS" || exit 1;
  fi
  # Align dependency metadata with the existing explicit build platform.
  configure_native_install_platform || exit 1;
  # Pip install requirements
  install_requirements || exit 1;
  # Build with all args
  build_code || exit 1;
}

local_kill_script() {
  # Kill all running processes related to rtp_llm and conda310
  kill_processes || exit 1;
}

local_copy_script() {
  # Enter work dir
  enter_work_dir || exit 1;
  cd RTP-LLM/github-opensource;
  # Find and return test output path to caller
  find_test_output_path || exit 1;
  echo "${TEST_OUTPUT_PATH}"
}

local_clean_script() {
  # Enter work dir
  enter_work_dir || exit 1;
  cd RTP-LLM/github-opensource;
  # Clean test output directory
  (rm -rf TEST_OUTPUT_*) || exit 1;
}

local_test_script() {
  # Kill all running processes related to rtp_llm and conda310
  kill_processes || exit 1;
  # Enter work dir
  enter_work_dir || exit 1;
  cd RTP-LLM/github-opensource;
  # Configure environment variables
  configure_env || exit 1;
  # Run server
  run_server || exit 1;
}

main() {
  SUB_CMD=${SUB_CMD}

  case ${SUB_CMD} in
      build)
          echo "execute local_build_script";
          local_build_script || exit 1;
          ;;
      kill)
          echo "execute local_kill_script";
          local_kill_script || exit 1;
          ;;
      copy)
          local_copy_script || exit 1;
          ;;
      clean)
          echo "execute local_clean_script";
          local_clean_script || exit 1;
          ;;
      test)
          echo "execute local_test_script";
          local_test_script || exit 1;
          ;;
  esac
}

main;
