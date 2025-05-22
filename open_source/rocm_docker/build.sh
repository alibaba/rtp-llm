#! /bin/bash
set +x;

echo $0
echo $(readlink -f "$0")
DIR=$(dirname $(readlink -f "$0"))
echo $DIR

# These variables are to replace by your own values
BASE_IMAGE="kis-registry.cn-wulanchabu.cr.aliyuncs.com/kis/amd:AMD-ROCM_6.3.0.2-alios-py_3.10_2024_12_18_13_42_00"
IMAGE_NAME=${IMAGE_NAME:-"reg.docker.alibaba-inc.com/isearch/rtp_llm_rocm_base"}
IMAGE_TAG=${IMAGE_TAG:-"rocm6.3.0.2_amdsmi_fix"}
CONDA_URL=${CONDA_URL:-"https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh"}
BAZELISK_URL=${BAZELISK_URL:-https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64}

docker build \
    --build-arg BASE_OS_IMAGE=$BASE_IMAGE \
    --build-arg CONDA_URL=$CONDA_URL \
    --build-arg BAZELISK_URL=$BAZELISK_URL \
    -f rocm6302.Dockerfile \
    --network=host \
    -t $IMAGE_NAME:$IMAGE_TAG \
    $DIR && \

docker tag $IMAGE_NAME:$IMAGE_TAG $IMAGE_NAME:latest
docker push $IMAGE_NAME:$IMAGE_TAG
docker push $IMAGE_NAME:latest

rm -rf $DIR/deps
