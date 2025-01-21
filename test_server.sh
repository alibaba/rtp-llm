#!/bin/bash
set -x;

export PYTHON_BIN=/opt/conda310/bin/python;
export XINSHI_HOME=/home/pengshixin.psx;
export JAVA_HOME=/opt/taobao/java;
export PYTHONUNBUFFERED=TRUE;

export PYTHONPATH=${XINSHI_HOME}/FasterTransformer/:${PYTHONPATH}
export PY_LOG_PATH=${XINSHI_HOME}/FasterTransformer/logs

# bazel build //maga_transformer:faster_transformer_all_tar

BAZEL_BIN=bazel
# if file .bazeliskrc exists, use bazelisk
if [ -f .bazeliskrc ]; then
    BAZEL_BIN=bazelisk
fi


bazelisk --output_user_root=~/temp/build_cache build //:th_transformer //maga_transformer:maga_transformer_lib --config=cuda12_6 --keep_going || {
    echo "bazel build failed";
    exit 1;
};

ln -s ../../../bazel-out/k8-opt/bin/maga_transformer/cpp/proto/model_rpc_service_pb2_grpc.py maga_transformer/cpp/proto/;
ln -s ../../../bazel-out/k8-opt/bin/maga_transformer/cpp/proto/model_rpc_service_pb2.py maga_transformer/cpp/proto/;


export CHECKPOINT_PATH="${XINSHI_HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775";
export MODEL_TYPE="qwen_tool";
export LD_LIBRARY_PATH=/opt/conda310/lib/:/usr/local/cuda/compat/:/usr/local/nvidia/lib64:/usr/lib64:/usr/local/cuda/lib64

export FT_SERVER_TEST=1

export TOKENIZER_PATH=${CHECKPOINT_PATH}
export LOAD_CKPT_NUM_PROCESS=8

export INT8_MODE=0
export TP_SIZE=1

export WORLD_SIZE=$TP_SIZE
export USE_GANG=FALSE
# export MERGE_LORA=0

export NCCL_NET_GDR_READ=1
export NCCL_NET_GDR_LEVEL=LOC
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
# export FT_DISABLE_CUSTOM_AR=1
# export NCCL_DEBUG=DEBUG

# export USE_RPC_MODEL=1
# export USE_NEW_DEVICE_IMPL=1

export MAX_SEQ_LEN=10240
export MAX_CONTEXT_BATCH_SIZE=1
export CONCURRENCY_LIMIT=8

# export KV_CACHE_MEM_MB=1024
# export SEQ_SIZE_PER_BLOCK=1024
# export ENABLE_FAST_GEN=1
# export FAST_GEN_MAX_CONTEXT_LEN=1024
export WARM_UP=1

# export KV_CACHE_MEM_MB=4096
# export SEQ_SIZE_PER_BLOCK=4
# export ENABLE_FMHA=OFF
# export ENABLE_TRTV1_FMHA=OFF
# export ENABLE_TRT_FMHA=OFF
# export ENABLE_PAGED_TRT_FMHA=OFF
# export ENABLE_OPENSOURCE_FMHA=OFF

# export FT_SERVER_TEST=1

export START_PORT=51255
export CUDA_VISIBLE_DEVICES=2,3,1,0

$PYTHON_BIN -m pdb ${XINSHI_HOME}/FasterTransformer/test_server.py;