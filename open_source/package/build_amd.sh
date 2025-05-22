set -x;

BASE_IMAGE=reg.docker.alibaba-inc.com/isearch/rtp_llm_rocm_base
TARGET_IMAGE=reg.docker.alibaba-inc.com/isearch/rtp_llm_rocm
DEV_IMAGE=$BASE_IMAGE
BAZEL_ARGS="--config=rocm"

sh package_docker.sh ${BASE_IMAGE} ${TARGET_IMAGE} ${DEV_IMAGE} ${BAZEL_ARGS}
