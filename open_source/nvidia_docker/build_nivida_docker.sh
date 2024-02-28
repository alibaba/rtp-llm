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
      echo "-o 使用nvidia cuda11的docker"
      echo "-n 使用nvidia cuda12的docker"
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

NVIDIA_MAGA_TAG=`date "+%Y_%m_%d_%H_%M"`_`git rev-parse --short HEAD`

NVIDIA_BASE_IMAGE=nvcr.io/nvidia/pytorch
NVIDIA_BASE_TAG=""
NVIDIA_MAGA_IMAGE=""

if ${CUDA11_DOCKER}; then
  NVIDIA_BASE_TAG="21.10-py3"
  NVIDIA_MAGA_IMAGE=reg.docker.alibaba-inc.com/isearch/maga_image_open_source
elif ${CUDA12_DOCKER}; then
  NVIDIA_BASE_TAG="23.10-py3"
  NVIDIA_MAGA_IMAGE=reg.docker.alibaba-inc.com/isearch/maga_image_open_source_cuda12
else
  echo "error, nvidia docker must be cuda11 or cuda12"
  exit 1
fi

# echo "pull nvidia docker $NVIDIA_BASE_IMAGE:$NVIDIA_BASE_TAG"
# docker pull $NVIDIA_BASE_IMAGE:$NVIDIA_BASE_TAG

echo "FROM $NVIDIA_BASE_IMAGE:$NVIDIA_BASE_TAG" > /tmp/nvidia_maga.Dockerfile
cat $DIR/nvidia_maga.Dockerfile >> /tmp/nvidia_maga.Dockerfile

echo "NVIDIA_MAGA_IMAGE = $NVIDIA_MAGA_IMAGE"
DOCKER_BUILDKIT=0 docker build -f /tmp/nvidia_maga.Dockerfile --network=host \
    --build-arg BAZEL_INSTALL_SH=bazel-6.4.0-installer-linux-x86_64.sh \
    -t $NVIDIA_MAGA_IMAGE:$NVIDIA_MAGA_TAG ./

docker tag $NVIDIA_MAGA_IMAGE:$NVIDIA_MAGA_TAG $NVIDIA_MAGA_IMAGE:latest
docker push $NVIDIA_MAGA_IMAGE:$NVIDIA_MAGA_TAG
docker push $NVIDIA_MAGA_IMAGE:latest
echo "docker name: $NVIDIA_MAGA_IMAGE:$NVIDIA_MAGA_TAG"
