#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAVI_SCHED_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
VERSION=0.1.0
RELEASE_DIR=${NAVI_SCHED_DIR}/release
COMPILER_CONFIG=${1:-gcc}

if [[ $# -gt 1 ||
    ("${COMPILER_CONFIG}" != "clang" &&
    "${COMPILER_CONFIG}" != "gcc") ]]; then
    echo "usage: $0 [clang|gcc]" >&2
    exit 1
fi

export NAVI_SCHED_COMPILER_CONFIG=${COMPILER_CONFIG}
"${SCRIPT_DIR}/test_wheel.sh"
"${SCRIPT_DIR}/build_cpp_package.sh"
"${SCRIPT_DIR}/build_simulate_package.sh"
"${NAVI_SCHED_DIR}/test/simulator/test_simulate_package.sh"

printf 'NAVI_SCHED_COMPILER=%s\n' "${COMPILER_CONFIG}"
printf 'NAVI_SCHED_WHEEL=%s\n' \
    "${RELEASE_DIR}/navi_sched-${VERSION}-cp310-cp310-linux_x86_64.whl"
printf 'NAVI_SCHED_CPP_PACKAGE=%s\n' \
    "${RELEASE_DIR}/navi_sched_cpp-${VERSION}.tar.gz"
printf 'NAVI_SCHED_SIMULATE_PACKAGE=%s\n' \
    "${RELEASE_DIR}/navi_sched_simulate-${VERSION}.tar.gz"
