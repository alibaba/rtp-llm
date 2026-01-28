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
    echo "开始创建软链接..."
    (unlink ./stub_source) || exit 1;
    (ln -sf ../internal_source ./stub_source) || exit 1;
    echo "软链接创建完成！"
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
    (rm -rf $WORK_DIR/bazel_cache) || true;
    (bazelisk clean --expunge) || true;
  fi
}

install_requirements() {
  if [ "${BUILD_FROM_SCRATCH:-2}" -gt 1 ]; then
    # Pip install requirements
    if [ `uname -m` == "aarch64" ]; then
      (/opt/conda310/bin/python3 -m pip install -r ./internal_source/deps/requirements_lock_cuda12_arm.txt) || exit 1;
    else
      (/opt/conda310/bin/python3 -m pip install -r ./internal_source/deps/requirements_lock_torch_gpu_cuda12.txt) || exit 1;
    fi
  fi
}

build_code() {
  if [ "${BUILD_FROM_SCRATCH:-2}" -gt 0 ]; then
    # Kill bazel build processes
    (ps axuww | grep 'bazelisk --batch --output_user_root' | grep -v grep | awk '{print $2}' | xargs kill -9) || true;
    # Build with all arguments
    (bazelisk --batch --output_user_root=$WORK_DIR/bazel_cache build //:th_transformer //rtp_llm:rtp_llm_lib ${BAZEL_BUILD_ARGS}) || exit 1;
    if [ $? -ne 0 ]; then
      echo "bazel build failed !";
      exit 1;
    fi
    # Create symbolic links for proto files
    bazel_subdir=k8-opt
    if [ -d "bazel-out/aarch64-opt" ]; then
      bazel_subdir=aarch64-opt
    fi
    (ln -sf ../../../../bazel-out/${bazel_subdir}/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2_grpc.py rtp_llm/cpp/model_rpc/proto/) || exit 1;
    (ln -sf ../../../../bazel-out/${bazel_subdir}/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2.py rtp_llm/cpp/model_rpc/proto/) || exit 1;
    (ln -sf ../../../../bazel-out/${bazel_subdir}/bin/rtp_llm/cpp/cuda/deep_gemm/cutlass_hdr rtp_llm/cpp/cuda/deep_gemm/cutlass_hdr) || exit 1;
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
