#! /bin/bash
set -xe;

DIR=$(dirname $(readlink -f "$0"))
echo "dir = $DIR"

# check args
BASE_IMAGE=$1
if [ -z $BASE_IMAGE ]; then
    echo "Usage: $0 BASE_IMAGE TARGET_IMAGE DEV_IMAGE BAZEL_CONFIG"
    exit 1
fi
BASE_IMAGE_TAG=latest

TARGET_IMAGE=$2
if [ -z $TARGET_IMAGE ]; then
    echo "Usage: $0 BASE_IMAGE TARGET_IMAGE DEV_IMAGE BAZEL_CONFIG"
    exit 1
fi

DEV_IMAGE=$3
if [ -z $DEV_IMAGE ]; then
    echo "Usage: $0 BASE_IMAGE TARGET_IMAGE DEV_IMAGE BAZEL_CONFIG"
    exit 1
fi

# take 3rd and later args as bazel config
BAZEL_CONFIG="${@:4}"
if [ -z $BAZEL_CONFIG ]; then
    echo "Usage: $0 BASE_IMAGE TARGET_IMAGE DEV_IMAGE BAZEL_CONFIG"
    exit 1
fi

# render more build args
TAG=`date "+%Y_%m_%d_%H_%M"`_`git rev-parse --short HEAD`
# TODO: check arm
if [[ $PLATFORM == "aarch64" ]]; then
    WHEEL_TARGET=//rtp_llm:rtp_llm_aarch64
    WHEEL_OS=linux
else
    PLATFORM=x86_64
    WHEEL_TARGET=//rtp_llm:rtp_llm
    WHEEL_OS=manylinux1
fi
VERSION=`cat $DIR/../../open_source/version`
WHL_FILE=rtp_llm-$VERSION-cp310-cp310-${WHEEL_OS}_$PLATFORM.whl

# create temp container
TEMP_CONTAINER_NAME=packaging_temp_$TAG

# set cleanup trap
function docker_cleanup() {
    docker stop $TEMP_CONTAINER_NAME || true
    docker rm $TEMP_CONTAINER_NAME || true
}

trap docker_cleanup EXIT

# docker pull $DEV_IMAGE:$BASE_IMAGE_TAG
docker run --device /dev/fuse -v /mnt/:/mnt/ -v /dev/shm:/dev/shm --rm \
    --name=$TEMP_CONTAINER_NAME  --label com.search.type=dev \
    -v /home/:/home/ \
    -v ~/.cache/:/root/.cache/ \
    --net=host \
    -dit $DEV_IMAGE:$BASE_IMAGE_TAG /bin/bash
export DOCKER_PID=`docker inspect --format "{{ .State.Pid }}" $TEMP_CONTAINER_NAME`;

# build wheel inside temp container
docker cp $DIR/../.. $TEMP_CONTAINER_NAME:/FasterTransformer
docker cp ~/.ssh $TEMP_CONTAINER_NAME:/root/.ssh
docker exec -i $TEMP_CONTAINER_NAME chown -R root:root /root/.ssh
docker exec -i $TEMP_CONTAINER_NAME /bin/bash -c "ssh-keyscan gitlab.alibaba-inc.com >> ~/.ssh/known_hosts"
docker exec -i $TEMP_CONTAINER_NAME /bin/bash -c "cd /FasterTransformer/ && bazelisk build $BAZEL_CONFIG $WHEEL_TARGET"
rm $DIR/$WHL_FILE -f
docker cp $TEMP_CONTAINER_NAME:/FasterTransformer/bazel-bin/rtp_llm/$WHL_FILE $DIR/

# prepare start script
START_SH_DIR="$DIR/../../internal_source/package/maga_start.sh"
if [[ ! -f $START_SH_DIR ]]; then
    echo "internal source start script not found, use open source"
    START_SH_DIR="$DIR/../maga_start.sh"
fi
cp $START_SH_DIR $DIR/

# build docker
echo "FROM $BASE_IMAGE:$BASE_IMAGE_TAG" > /tmp/maga.Dockerfile
cat $DIR/maga.Dockerfile >> /tmp/maga.Dockerfile
docker build --build-arg WHL_FILE=$WHL_FILE --build-arg START_FILE=maga_start.sh \
       -f /tmp/maga.Dockerfile --network=host -t $TARGET_IMAGE:$TAG $DIR
docker tag $TARGET_IMAGE:$TAG $TARGET_IMAGE:latest
docker push $TARGET_IMAGE:$TAG
docker push $TARGET_IMAGE:latest
echo "docker name: $TARGET_IMAGE:$TAG"

#cleanup context
rm $DIR/$WHL_FILE -f
rm $DIR/maga_start.sh -f
