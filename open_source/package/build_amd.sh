set -x;

BASE_IMAGE=reg.docker.alibaba-inc.com/isearch/maga_transformer_rocm_base
TARGET_IMAGE=reg.docker.alibaba-inc.com/isearch/maga_transformer_rocm
DEV_IMAGE=$BASE_IMAGE
BAZEL_ARGS="--config=rocm"

sh package_docker.sh ${BASE_IMAGE} ${TARGET_IMAGE} ${DEV_IMAGE} ${BAZEL_ARGS}
