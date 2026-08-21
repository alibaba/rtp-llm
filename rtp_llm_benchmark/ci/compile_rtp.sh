#!/bin/bash

set_env_para(){
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    WORKSPACE="${SCRIPT_DIR}/../../../"
    export RTP_PATH="${WORKSPACE}/rtp-llm"
    git config --global --add safe.directory ${RTP_PATH}
    export LOG_DIR="${WORKSPACE}/logs"
    [ ! -d ${LOG_DIR} ] && mkdir -p ${LOG_DIR}
    # unset PIP_EXTRA_INDEX_URL
    unset PIP_INDEX_URL
    sed -i 's|^\s*index-url\s*=.*|#&|'  ~/.config/pip/pip.conf
    sed -i 's/^/#/' $RTP_PATH/bazel/bazel_downloader.cfg
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    # tmp 
    #yum --disablerepo="*" --enablerepo=alinux3-updates install -y openblas openblas-devel
    #/opt/conda310/bin/python3 -m pip uninstall -y pytorch_triton_rocm
    /opt/conda310/bin/python -m pip uninstall -y triton
    /opt/conda310/bin/python -m pip install -y triton==3.5.0 --index-url https://pypi.org/simple
}

record_env() {
    output_file=${LOG_DIR}/env.md
    hostname=$(hostname)
    gcc_version=$(gcc --version | head -n1)
    hipblaslt_version=$(ls /opt/rocm/lib | grep hipblaslt.so | grep -oP '\d+(\.\d+)+')
    cpu_gpu_version=$(rocminfo | grep "Marketing Name:" | cut -d ':' -f2- | sort -u | sed 's/^ *//')
    commit_sha=$(git -C ${RTP_PATH} rev-parse HEAD)
    branch=$(git -C ${RTP_PATH} branch --show-current)
    echo "Hostname: $hostname" > $output_file
    echo "CPU-GPU Version: $cpu_gpu_version" >> $output_file
    echo "Image Name: $IMAGE_NAME" >> $output_file
    echo "GCC Version: $gcc_version" >> $output_file
    echo "HIPBLASLT Version: $hipblaslt_version" >> $output_file
    echo "Git Branch: $branch" >> $output_file
    echo "Git Commit SHA: $commit_sha" >> $output_file
    /opt/conda310/bin/python3 -m pip freeze >> $output_file
}

compile_rtp(){
    cd ${RTP_PATH}
    EXIT_CODE=0
    set -o pipefail
    # yum --disablerepo="*" --enablerepo=alinux3-os install -y patch
    # /opt/conda310/bin/python3 -m pip install -r ./deps/requirements_rocm.txt
    /opt/conda310/bin/python3 -m pip install --no-cache-dir -r ./deps/requirements_rocm.txt --index-url https://pypi.org/simple
    # /opt/conda310/bin/python3 -m pip install /mnt/raid0/yuzho/BACKUPS/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl # flash attn whl
    /opt/conda310/bin/python -m pip install ninja -i https://pypi.org/simple/
    /opt/conda310/bin/python -m pip install flash_attn --no-build-isolation --index-url https://pypi.org/simple

    # try to build
    bazelisk build //rtp_llm:rtp_llm //rtp_llm/dash_sc/proto:predict_v2_py --jobs 150 --verbose_failures --config=rocm 2>&1 | tee "${LOG_DIR}/bazelbuild.log"
    BUILD_RESULT=$?
    
    # if build failed and is because timeout, set PIP_TIMEOUT and try again
    if [ $BUILD_RESULT -ne 0 ]; then
        if grep -q -i "timeout\|timed out" "${LOG_DIR}/bazelbuild.log"; then
            echo "bazel build failed due to time out，set PIP_TIMEOUT=300 and try again..."
            export PIP_TIMEOUT=300
            bazelisk build //rtp_llm:rtp_llm //rtp_llm/dash_sc/proto:predict_v2_py --jobs 150 --verbose_failures --config=rocm 2>&1 | tee -a "${LOG_DIR}/bazelbuild.log" || EXIT_CODE=1
        else
            EXIT_CODE=1
        fi
    fi
    bash ${RTP_PATH}/rtp_llm/dash_sc/proto/link_py_proto.sh || true
    record_env
}

set_env_para
compile_rtp
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    # 被source调用
    return $EXIT_CODE
else
    # 直接执行
    exit $EXIT_CODE
fi
