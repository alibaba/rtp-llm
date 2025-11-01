set -x
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PYI_PATH=$SCRIPT_DIR/../../rtp_llm/ops
BAZEL_BIN_PATH=$SCRIPT_DIR/../../bazel-bin
cd $BAZEL_BIN_PATH

export LD_LIBRARY_PATH=\
/opt/conda310/lib:\
$BAZEL_BIN_PATH/rtp_llm/libs:\
/usr/local/cuda-12.6/targets/x86_64-linux/lib:\
/usr/local/cuda-12.6/extras/CUPTI/lib64:\
/usr/local/PPU_SDK/sailSHMEM/lib:\
.:\
${LD_LIBRARY_PATH}

export PYTHONPATH=.:${PYTHONPATH}

rm -rf stubs

# 使用 run.py以：
# 1. os._exit(-1)避免torch析构时候的corrupted double-linked list
# 2. 提前load torch和th_transformer符号，避免有些pybind不在当前so里注册而报错的情况

/opt/conda310/bin/python3 $SCRIPT_DIR/run.py libth_transformer_config
/opt/conda310/bin/python3 $SCRIPT_DIR/run.py libth_transformer
/opt/conda310/bin/python3 $SCRIPT_DIR/run.py librtp_compute_ops

if [ -d stubs ] && [ "$(ls -A stubs)" ]; then
    cp -r stubs/* $PYI_PATH
else
    echo "Error: stubs directory is empty or not created"
    exit 1
fi

