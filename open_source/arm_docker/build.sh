#! /bin/bash
set -x;

echo $0
echo $(readlink -f "$0")
DIR=$(dirname $(readlink -f "$0"))
echo $DIR

# These variables are to replace by your own values
BASE_IMAGE="alibaba-cloud-linux-3-registry.cn-hangzhou.cr.aliyuncs.com/alinux3/alinux3:231220.1"
IMAGE_NAME=${IMAGE_NAME:-"reg.docker.alibaba-inc.com/isearch/rtp_llm_arm_open_source_base"}
IMAGE_TAG=${IMAGE_TAG:-"0.0.3"}
CONDA_URL=${CONDA_URL:-"https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-aarch64.sh"}
BAZELISK_URL=https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-arm64

docker build \
    --build-arg BASE_OS_IMAGE=$BASE_IMAGE \
    --build-arg CONDA_URL=$CONDA_URL \
    --build-arg BAZELISK_URL=$BAZELISK_URL \
    -f arm.Dockerfile \
    --network=host \
    -t $IMAGE_NAME:$IMAGE_TAG \
    $DIR && \
docker tag $IMAGE_NAME:$IMAGE_TAG $IMAGE_NAME:latest

