#!/bin/bash
set -euo pipefail
source_root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$source_root"
/opt/conda310/bin/python rtp_llm/dash_sc/proto/create_grpc_proto.py
/opt/conda310/bin/python rtp_llm/cpp/model_rpc/proto/create_grpc_proto.py rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto .
/opt/conda310/bin/python rtp_llm/cpp/model_rpc/proto/create_grpc_proto.py rtp_llm/cpp/model_rpc/proto/flexlb_schedule_service.proto .
exec /tmp/rtp-bazelisk --batch --output_user_root=/tmp/dsv4-bazel-cache \
    build //:th_transformer //:th_transformer_config //:rtp_compute_ops //:th_grammar_tokenizer_info \
    --config=cuda13 --jobs=48 \
    --disk_cache=/home/admin/.cache/disk_cache \
    --repository_cache=/home/admin/.cache/bazel_cuda13_cache/cache/repos/v1 \
    --remote_cache= --remote_executor= --experimental_remote_downloader= \
    --linkopt=-L/tmp/rtp-glm53-link-libs \
    --linkopt=-Wl,-rpath-link,/opt/conda310/lib/python3.10/site-packages/nvidia/cusparselt/lib \
    --linkopt=-Wl,-rpath-link,/opt/conda310/lib/python3.10/site-packages/nvidia/nvshmem/lib \
    --repo_env=TF_CUDA_PATHS=/usr/local/cuda-13.2 \
    --action_env=PATH=/opt/conda310/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
    --host_action_env=PATH=/opt/conda310/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
    --repo_env=CUDNN_INSTALL_PATH=/opt/conda310/lib/python3.10/site-packages/nvidia/cudnn \
    --action_env=CUDNN_INSTALL_PATH=/opt/conda310/lib/python3.10/site-packages/nvidia/cudnn \
    --repo_env=NCCL_INSTALL_PATH=/opt/conda310/lib/python3.10/site-packages/nvidia/nccl \
    --action_env=NCCL_INSTALL_PATH=/opt/conda310/lib/python3.10/site-packages/nvidia/nccl \
    --action_env=TF_CUDA_COMPUTE_CAPABILITIES=10.3 \
    --host_action_env=TF_CUDA_COMPUTE_CAPABILITIES=10.3
