#!/bin/bash
set -x;

enter_work_dir() {
  # determine work dir: use env HIPPO_APP_WORKDIR if it is set, otherwise use home
  export SUB_DIR=${FT_SUB_DIR:-"."}
  export WORK_DIR=${FT_WORK_DIR:-${HIPPO_APP_WORKDIR:-$HOME}}/$SUB_DIR
  if [ ! -d "$WORK_DIR" ]; then
    mkdir -p $WORK_DIR
  fi
  cd $WORK_DIR
}

kill_processes() {
  # kill all running processes related to rtp_llm and conda310
  echo "[$(date)] Starting process termination on $HOSTNAME"
  ps axuww | grep rtp_llm_rank | grep -v run_worker.sh | awk '{print $2}' | xargs kill -9 2>/dev/null || true
  ps axuww | grep rtp_llm_backend_server | grep -v run_worker.sh | awk '{print $2}' | xargs kill -9 2>/dev/null || true
  ps axuww | grep rtp_llm_frontend_server_ | grep -v run_worker.sh | awk '{print $2}' | xargs kill -9 2>/dev/null || true
  ps axuww | grep /opt/conda310/bin/python | grep -v run_worker.sh | awk '{print $2}' | xargs kill -9 2>/dev/null || true
  # verify processes have been terminated
  echo "[$(date)] Verifying processes have been terminated on $HOSTNAME"
  ps axuww | grep rtp_llm_rank | grep -v grep | grep -v run_worker.sh
  ps axuww | grep rtp_llm_backend_server | grep -v grep | grep -v run_worker.sh
  ps axuww | grep rtp_llm_frontend_server_ | grep -v grep | grep -v run_worker.sh
  ps axuww | grep /opt/conda310/bin/python | grep -v grep | grep -v run_worker.sh
  # remove shared memory and semaphore files
  rm -rf /dev/shm/P*
  rm -rf /dev/shm/sem.loky*
  sleep 3
}

checkout_code() {
  # checkout code from git if not exist
  if [ ! -d "RTP-LLM" ]; then
    (git clone $GIT_REPO_URL) || exit 1;
  fi
  cd RTP-LLM;
  (git fetch origin && git reset --hard $GIT_CHECKOUT_REF) || exit 1;
}

install_requirements() {
  if [ "${BUILD_FROM_SCRATCH:-2}" -gt 1 ]; then
    # pip install requirements
    /opt/conda310/bin/python3 -m pip install -r ./deps/requirements_lock_torch_gpu_cuda12.txt
  fi
}

build_code() {
  if [ "${BUILD_FROM_SCRATCH:-2}" -gt 0 ]; then
    # build with all args
    bazelisk --batch --output_user_root=$WORK_DIR/bazel_cache build //:th_transformer //rtp_llm:rtp_llm_lib ${BAZEL_BUILD_ARGS}

    if [ $? -ne 0 ]; then
      echo "bazel build failed !"
      exit 1
    fi

    # create symbolic links for proto files
    ln -sf ../../../bazel-out/k8-opt/bin/rtp_llm/cpp/proto/model_rpc_service_pb2_grpc.py rtp_llm/cpp/proto/;
    ln -sf ../../../bazel-out/k8-opt/bin/rtp_llm/cpp/proto/model_rpc_service_pb2.py rtp_llm/cpp/proto/;
    ln -sf ../../../bazel-bin/rtp_llm/cpp/deep_gemm/cutlass_hdr rtp_llm/cpp/deep_gemm/cutlass_hdr;
  fi
}


configure_env() {
  # Configure fix environment variables
  export FT_SERVER_TEST=1;
  export FT_DISABLE_CUSTOM_AR=1;
  export MLA_OPS_TYPE="AUTO";
  export PYTHONPATH="./";
  export PYTHON_BIN="/opt/conda310/bin/python";
  export FT_CORE_DUMP_ON_EXCEPTION=1;
  export NCCL_TOPO_DUMP_FILE="/tmp/nccl_topo.xml";
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib64:/usr/local/nvidia/lib64/:/usr/local/cuda/lib64/";

  # Configure optional environment variables
  echo -e "\nActive environment variables:"
  printenv | grep -E 'NCCL|ACCL|NVSHMEM' | sort
}

run_server() {
  if [ "${BUILD_FROM_SCRATCH:-2}" -lt 2 ]; then
    SERVER_CMD="$PYTHON_BIN ./rtp_llm/test/perf_test/multi_node/local_server_runner.py";
    if [ $CUDA_ASAN -eq 1 ]; then
        /usr/local/cuda/compute-sanitizer/compute-sanitizer --print-limit 10000 --target-processes all \
        $SERVER_CMD
    else
      $SERVER_CMD;
    fi
  fi
}

local_build_script() {
  # determine work dir: use env HIPPO_APP_WORKDIR if it is set, otherwise use home
  enter_work_dir;
  # checkout code from git if not exist
  checkout_code;
  # pip install requirements
  install_requirements;
  # build with all args
  build_code;
}

local_kill_script() {
  kill_processes;
}

local_clean_script() {
  # determine work dir: use env HIPPO_APP_WORKDIR if it is set, otherwise use home
  enter_work_dir;
  cd RTP-LLM;
  rm -rf ${RM_PATTERN_PATH};
  rm -rf core-rtp_llm_rank-*
}

local_test_script() {
  # determine work dir: use env HIPPO_APP_WORKDIR if it is set, otherwise use home
  enter_work_dir;

  # kill all running processes related to rtp_llm and conda310
  kill_processes;

  # checkout code from git if not exist
  checkout_code;

  # pip install requirements
  install_requirements;

  # build with all args
  build_code;

  # configure environment variables
  configure_env;

  # run server
  run_server;
}

main() {
  SUB_CMD=${SUB_CMD}

  case ${SUB_CMD} in
      build)
          echo "execute local_build_script"
          local_build_script
          ;;
      kill)
          echo "execute local_kill_script"
          local_kill_script
          ;;
      clean)
          echo "execute local_clean_script"
          local_clean_script
          ;;
      test)
          echo "execute local_test_script"
          local_test_script
          ;;
  esac
}

main;
