set -x;

#BASE_IMAGE=reg.docker.alibaba-inc.com/isearch/rtp_llm_arm_base
BASE_IMAGE=reg.docker.alibaba-inc.com/isearch/rtp_llm_arm_open_source_base
TARGET_IMAGE=reg.docker.alibaba-inc.com/isearch/rtp_llm_arm
DEV_IMAGE=$BASE_IMAGE
BAZEL_ARGS="--config=arm"
export PLATFORM=aarch64

sh package_docker.sh ${BASE_IMAGE} ${TARGET_IMAGE} ${DEV_IMAGE} ${BAZEL_ARGS}
