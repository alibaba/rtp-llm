set -x;

SCRIPT_PATH="$( cd "$(dirname "$0")" || return ; pwd -P )"

if [ -z "${HIPPO_PROC_WORKDIR}" ]; then
  export TEMP_HIPPO_DIR=/tmp/hippo_temp
  mkdir -p $TEMP_HIPPO_DIR
  HIPPO_PROC_WORKDIR=$TEMP_HIPPO_DIR
fi

if [ -z "${HIPPO_APP_INST_ROOT}" ]; then
  export HIPPO_APP_INST_ROOT=$SCRIPT_PATH/../../
fi

DEFAULT_START_PORT=12233;
DEFAULT_PY_RTP_LOG_KEEP_COUNT=300;

START_PORT=${START_PORT:-${DEFAULT_START_PORT}};
PY_RTP_LOG_KEEP_COUNT=${PY_RTP_LOG_KEEP_COUNT:-${DEFAULT_PY_RTP_LOG_KEEP_COUNT}};

export LOG_DIR="$HIPPO_PROC_WORKDIR"/logs;
mkdir -p "$LOG_DIR";
ls $LOG_DIR | grep -Ev '[0-9]{4}_[0-9]{2}_[0-9]{2}__' | \
xargs -I {} bash -c 'mv ${LOG_DIR}/{} ${LOG_DIR}/`date -r ${LOG_DIR}/{} +"%Y_%m_%d__%H_%M_%S__"`{}';
ls ${LOG_DIR} -t | tail -n +${PY_RTP_LOG_KEEP_COUNT} | xargs -I {} rm -rf ${LOG_DIR}/{};

# $LOG_PATH used for py_inference `get_handler` func, do not remove
export LOG_PATH=$LOG_DIR

export STDOUT_FILE=$LOG_DIR/stdout
export STDERR_FILE=$LOG_DIR/stderr
export ENV_FILE=$LOG_DIR/env.txt

#logging level
export LOG_LEVEL="INFO"

# pyfsutil
export HADOOP_HOME=$HIPPO_APP_INST_ROOT/usr/local/hadoop/hadoop;
export JAVA_HOME=$HIPPO_APP_INST_ROOT/usr/local/java/jdk;
export FSLIB_DFS_STORAGE_LINKS=${HIPPO_ENV_STORAGE_LINKS-"dir|pov|alb"};
export PATH=$JAVA_HOME/bin:$PATH;
FSUTIL_DIR=$HIPPO_APP_INST_ROOT;

HOST_DRIVER_VERSION=${HOST_NVIDIA_DRIVER_VERSION}
# 如果环境变量为空，尝试通过 nvidia-smi 获取驱动版本
if [ -z "$HOST_DRIVER_VERSION" ]; then
  if command -v nvidia-smi &> /dev/null; then
    # 使用nvidia-smi获取驱动版本
    HOST_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n 1)
    if [ $? -eq 0 ] && [ -n "$NVIDIA_DRIVER_VERSION" ]; then
      HOST_DRIVER_VERSION=$NVIDIA_DRIVER_VERSION
    fi
  fi
fi

COMPAT_CUDA_PATH="/usr/local/cuda/compat/"

ADDITIONAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib64:/usr/local/nvidia/lib64/:\
${HIPPO_APP_INST_ROOT}/usr/lib64/nvidia/:\
${HIPPO_APP_INST_ROOT}/usr/local/cuda/lib64:/usr/local/cuda/lib64/:\
${HIPPO_APP_INST_ROOT}/inference_sdk/lib:\
${HIPPO_APP_INST_ROOT}/opt/taobao/java/jre/lib/amd64/server/:\
${FSUTIL_DIR}/lib:${FSUTIL_DIR}/lib64:${FSUTIL_DIR}/usr/local/lib:${FSUTIL_DIR}/usr/local/lib64"

NEED_COMPAT_CUDA_PATH=1
# 判断 HOST_NVIDIA_DRIVER_VERSION 是否为空
if [ -n "$HOST_DRIVER_VERSION" ]; then
  # 分割版本号
  IFS='.' read -r -a version_parts <<< "$HOST_DRIVER_VERSION"
  major_version=${version_parts[0]}
  minor_version=${version_parts[1]}

  # 检查前两位版本号是否 >= 535
  if [ "$major_version" -ge 535 ]; then
    # 去掉LD_LIBRARY_PATH中的 /usr/local/cuda/compat/
    ADDITIONAL_LD_LIBRARY_PATH=$(echo "$ADDITIONAL_LD_LIBRARY_PATH" | sed 's|:/usr/local/cuda/compat/||')
    NEED_COMPAT_CUDA_PATH=0
  fi
fi

if [ "$NEED_COMPAT_CUDA_PATH" -eq 1 ]; then
  FIXED_LD_LIBRARY_PATH="$BASE_LD_LIBRARY_PATH:$COMPAT_CUDA_PATH:$ADDITIONAL_LD_LIBRARY_PATH"
else
  FIXED_LD_LIBRARY_PATH="$BASE_LD_LIBRARY_PATH:$ADDITIONAL_LD_LIBRARY_PATH"
fi

export LD_LIBRARY_PATH="$FIXED_LD_LIBRARY_PATH"
export LD_LIBRARY_PATH_SETTED=1;

export FSLIB_PANGU_ENABLE_SEQUENTIAL_READAHEAD=${FSLIB_PANGU_ENABLE_SEQUENTIAL_READAHEAD-"true"}
export FSLIB_PANGU_ENABLE_BUFFER_WRITE=${FSLIB_PANGU_ENABLE_BUFFER_WRITE-"true"}

echo "START_PORT=${START_PORT}";

printenv > "$ENV_FILE";
if [ "${CMD}" ]; then
    echo "use cmd mode"
    ${CMD} >> "$STDOUT_FILE" 2>> "$STDERR_FILE";
else
    echo "use default mode"
    /opt/conda310/bin/python3 -m rtp_llm.start_server >> "$STDOUT_FILE" 2>> "$STDERR_FILE"
fi
