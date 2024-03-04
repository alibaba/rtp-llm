#! /bin/bash

echo $0

echo $(readlink -f "$0")

DIR=$(dirname $(readlink -f "$0"))

echo $DIR

USERNAME=$(whoami)
PASSWORD=""
CUDA11_DOCKER=false
CUDA12_DOCKER=false

SHORT_OPTS="p:onh"
LONG_OPTS="password:,cuda11_docker,cuda12_docker,help"

PARSED_OPTS=$(getopt -o "$SHORT_OPTS" --long "$LONG_OPTS"  -- "$@")

# 退出，如果解析失败
if [ $? -ne 0 ]; then
  exit 1
fi

eval set -- "$PARSED_OPTS"

# 解析参数
while true; do
  case $1 in
    -p|--password)
      PASSWORD="$2"
      shift 2
      ;;
    -o|--cuda11_docker)
      CUDA11_DOCKER=true
      shift
      ;;
    -n|--cuda12_docker)
      CUDA12_DOCKER=true
      shift
      ;;
    -h|--help)
      echo "用法:$0 [-p PASSWORD] [-o] [-n] "
      echo "-p password ,docker 仓库密码"
      echo "-o 打出nvidia maga deploy cuda11的docker"
      echo "-n 打出nvidia maga deploy cuda12的docker"
      shift
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "无效选项: $1" >&2
      exit 1
      ;;
  esac
done

set -xe

config_file=~/.docker/config.json

if [ ! -f "$config_file" ] && [ -z "$PASSWORD" ]; then
  echo "Docker config not exist and PASSWORD is empty"
  exit 1
fi

if [ -z "$PASSWORD" ]; then
  output=$(python -c "
import json
import base64

with open('${config_file}', 'r') as f:
    data = json.load(f)
auths = data.get('auths', {})

if auths:
    for auth_server, auth_data in auths.items():
        if auth_server == 'reg.docker.alibaba-inc.com':
            auth = auth_data.get('auth', '')
            if not auth:
                continue
            username = base64.b64decode(auth).decode().split(':', 1)[0]
            if username == '${USERNAME}':
              print(username)

  ")

  if [[ "$output" ]]; then
      echo "have login as user $output"
  else
      echo "try login in as user $USERNAME"
      docker login --username=$USERNAME --password=$PASSWORD reg.docker.alibaba-inc.com
  fi
fi


if ${CUDA12_DOCKER}; then
  IMAGE_SUFFIX="gpu_cuda12"
  DOCKERFILE_PREFIX="gpu_cuda12"
  BAZEL_CONFIG="--config=cuda12"
  NVIDIA_MAGA_BASE_IMAGE=reg.docker.alibaba-inc.com/isearch/maga_image_open_source_cuda12
  NVIDIA_MAGA_BASE_TAG="latest"
  REQUIREMENT_FILE=requirements_torch_gpu_cuda12.txt
else
  IMAGE_SUFFIX="gpu"
  DOCKERFILE_PREFIX="gpu"
  BAZEL_CONFIG=""
  NVIDIA_MAGA_BASE_IMAGE=reg.docker.alibaba-inc.com/isearch/maga_image_open_source
  NVIDIA_MAGA_BASE_TAG="latest"
  REQUIREMENT_FILE=requirements_torch_gpu.txt
fi

MAGA_DEPLOY_IMAGE=reg.docker.alibaba-inc.com/isearch/maga_deploy_image_open_source_$IMAGE_SUFFIX

echo "FROM $NVIDIA_MAGA_BASE_IMAGE:$NVIDIA_MAGA_BASE_TAG" > /tmp/nvidia_maga_deploy.Dockerfile
cat $DIR/nvidia_maga_deploy.Dockerfile >> /tmp/nvidia_maga_deploy.Dockerfile

# 输入是whl name，绝对路径
DIR_NAME=$(dirname "$WHL_FILE")
FILE_NAME=$(basename "$WHL_FILE")

# setup context
cp $DIR/../deps/requirements_base.txt $DIR/
cp $DIR/../deps/$REQUIREMENT_FILE $DIR/

if [ $? -ne 0 ]; then
  exit 1
fi

# rm $DIR/../../bazel-bin/maga_transformer/$WHL_FILE -f
rm $DIR/$FILE_NAME -f
cp $WHL_FILE $DIR/

docker build --build-arg REQUIREMENT_FILE=$REQUIREMENT_FILE --build-arg WHL_FILE=$FILE_NAME \
       -f /tmp/nvidia_maga_deploy.Dockerfile --network=host -t $MAGA_DEPLOY_IMAGE:$· $DIR
docker tag $MAGA_DEPLOY_IMAGE:$VERSION_TAG $MAGA_DEPLOY_IMAGE:latest
docker push $MAGA_DEPLOY_IMAGE:$VERSION_TAG
docker push $MAGA_DEPLOY_IMAGE:latest
echo "docker name: $MAGA_DEPLOY_IMAGE:$VERSION_TAG"

rm $DIR/$WHL_FILE -f
rm $DIR/requirements_base.txt -f
rm $DIR/$REQUIREMENT_FILE -f
