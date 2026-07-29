#!/bin/bash
# workspace is $GITHUB_WORKSPACE. /mnt/raid0/rtp-actions-runner-yuzho

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ${SCRIPT_DIR}/compile_rtp.sh
COMPILE_EXIT_CODE=$?

# 如果编译失败，直接退出
if [ $COMPILE_EXIT_CODE -ne 0 ]; then
    echo "错误：编译失败，退出代码：$COMPILE_EXIT_CODE"
    exit $COMPILE_EXIT_CODE
fi

install_and_test_ut(){
    cd ${RTP_PATH}
    EXIT_CODE=0
    set -o pipefail
    # Running all the cases in " //tests:" whose names contain "rocm_"
    for case in $(bazelisk query 'kind("py_test", //tests:*)' | grep rocm_); do
        logfile="${LOG_DIR}/bazeltest_tests_rocm_all.log"
        echo "$case in $(basename "$logfile")" >> "${LOG_DIR}/test_cases_registry.txt"
        echo "Runing $case ..." | tee -a "$logfile"
        bazelisk test "$case" --config=rocm --test_env=HSA_NO_SCRATCH_RECLAIM=1 --test_output=all 2>&1 | tee -a "$logfile" || EXIT_CODE=1
    done

    # ------------------------------------------------------------------------------
    # [已移除 2026-07-21 by lcong]
    # 原本测试 //rtp_llm/cpp/devices/rocm_impl/test 的两段（devices:all 循环 +
    # gemm_op_test SWIZZLE 单例）已删除。原因：上游 PR 802 (commit 2a30fe5,
    # "feat: remove cpp device abstraction") 移除了整个 cpp/devices/ C++ 抽象层
    # 及其 C++ 测试；gemm 功能已转至 models_py/bindings/rocm/Gemm.cc (Python 侧)，
    # 旧的 C++ gemm_op_test / SWIZZLE 测试为「废弃删除」，非迁移。
    # rocm 相关单测现由下方 models_py/.../rocm/test 段覆盖。
    # ------------------------------------------------------------------------------

    # Running all the cases in "rtp_llm/models_py/modules/base/rocm/test", "rtp_llm/models_py/modules/factory/fused_moe/impl/rocm/test", "rtp_llm/models_py/modules/factory/linear/impl/rocm/test"
    for case in $(bazelisk query 'tests(//rtp_llm/models_py/modules/base/rocm/test:all + //rtp_llm/models_py/modules/factory/fused_moe/impl/rocm/test:all + //rtp_llm/models_py/modules/factory/linear/impl/rocm/test:all)'); do
        logfile="${LOG_DIR}/bazeltest_py_all.log"
        echo "$case in $(basename "$logfile")" >> "${LOG_DIR}/test_cases_registry.txt"
        echo "Running $case ..." | tee -a "$logfile"
        bazelisk test "$case" --config=rocm --test_env=HSA_NO_SCRATCH_RECLAIM=1 --test_output=all 2>&1 | tee -a "$logfile" || EXIT_CODE=1
    done
}

install_and_test_ut
exit $EXIT_CODE
# trigger Thu Jul 23 04:54:00 PM CST 2026
# trigger CI after iommu=pt Mon Jul 27 06:59:37 PM CST 2026
# clean rerun Tue Jul 28 09:58:22 AM CST 2026
