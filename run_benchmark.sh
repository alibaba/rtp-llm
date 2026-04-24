#!/bin/bash
cd /root/yiyin/yiyin_rtp_llm_mi355x_benchmark/rtp-llm

SOURCE_DIR="$(pwd)"

# Sync source-only modules to installed package
INSTALLED_PKG_DIR="/opt/conda310/lib/python3.10/site-packages/rtp_llm"
echo "SOURCE_DIR=${SOURCE_DIR}"
echo "INSTALLED_PKG_DIR=${INSTALLED_PKG_DIR}"
if [ -d "${INSTALLED_PKG_DIR}" ]; then
    # Sync moriep_wrapper.py (not in installed package by default)
    if [ -f "${SOURCE_DIR}/rtp_llm/models_py/distributed/moriep_wrapper.py" ]; then
        cp "${SOURCE_DIR}/rtp_llm/models_py/distributed/moriep_wrapper.py" \
           "${INSTALLED_PKG_DIR}/models_py/distributed/moriep_wrapper.py" 2>/dev/null \
           && echo "Synced moriep_wrapper.py to installed package" \
           || echo "WARNING: Failed to sync moriep_wrapper.py"
    fi
    # Ensure test symlink exists
    if [ ! -e "${INSTALLED_PKG_DIR}/test" ]; then
        ln -s "${SOURCE_DIR}/rtp_llm/test" "${INSTALLED_PKG_DIR}/test"
        echo "Created symlink: ${INSTALLED_PKG_DIR}/test -> ${SOURCE_DIR}/rtp_llm/test"
    else
        echo "Symlink already exists: ${INSTALLED_PKG_DIR}/test"
    fi
fi

# Add aiter JIT directory to LD_LIBRARY_PATH for mori module dependencies
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/conda310/lib/python3.10/site-packages/aiter/jit"

export OMP_NUM_THREADS=8
export LD_PRELOAD=/opt/conda310/lib/libstdc++.so.6
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export HIP_VISIBLE_DEVICES=4,5,6,7
export SEQ_SIZE_PER_BLOCK=2048
export KERNEL_SEQ_SIZE_PER_BLOCK=16
export WARM_UP=0
export CONCURRENCY_LIMIT=128
export ENABLE_CUDA_GRAPH=0
export DECODE_CAPTURE_CONFIG=5,6,7,8
export QUANTIZATION=FP4_PER_GROUP_QUARK
export LOAD_PYTHON_MODEL=1
export LOAD_METHOD=fastsafetensors
# NOTE: LOAD_METHOD=fastsafetensors removed - fastsafetensors GDS check fails in this container
# The model will load via default safetensors path
export USE_ASM_PA=0
export WORLD_SIZE=4
export DP_SIZE=1
export TP_SIZE=4
export EP_SIZE=4
export DEVICE_RESERVE_MEMORY_BYTES=-16384000000
export RESERVER_RUNTIME_MEM_MB=40960
export START_PORT=10666
export ACT_TYPE=bf16
export TOKENIZER_PATH=/data/nvme0/models/Qwen3.5-397B-A17B-MXFP4
export CHECKPOINT_PATH=/data/nvme0/models/Qwen3.5-397B-A17B-MXFP4
export MODEL_TYPE=qwen35_moe
export USE_ALL_GATHER=0
export USE_DEEPEP_MOE=0
export USE_DEEPEP_LOW_LATENCY=0
export USE_MORI_EP=1
export FT_SERVER_TEST=1
export ROCM_DISABLE_CUSTOM_AG=True
export FT_DISABLE_CUSTOM_AR=True
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=0
export RCCL_ENABLE_P2P=0
export DIST_COMM_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export HACK_LAYER_NUM=5
export FAKE_BALANCE_EXPERT=1
export GEN_TIMELINE_SYNC=1
export INPUT_LEN_LIST="[2048]"
export BATCH_SIZE_LIST="[128]"
export IS_DECODE=1
export DECODE_TEST_LENGTH=20
export USE_BATCH_DECODE_SCHEDULER=1
export WORKER_INFO_PORT_NUM=10

echo "=== Starting benchmark ==="
echo "MODEL: $MODEL_TYPE"
echo "CHECKPOINT: $CHECKPOINT_PATH"
echo "PORT: $START_PORT"
echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
echo "==========================="

exec /opt/conda310/bin/python rtp_llm/test/perf_test/multi_node/local_server_runner.py 2>&1
