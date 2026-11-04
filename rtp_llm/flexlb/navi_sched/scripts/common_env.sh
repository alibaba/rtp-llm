#!/bin/bash

NAVI_SCHED_SCRIPT_DIR=$(
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
)
NAVI_SCHED_DIR=$(cd "${NAVI_SCHED_SCRIPT_DIR}/.." && pwd)
# navi_sched lives below rtp_llm/flexlb; the Bazel workspace is the
# repository root, three levels above this directory.
REPO_ROOT=$(cd "${NAVI_SCHED_DIR}/../../.." && pwd)

if [[ -n "${NAVI_SCHED_BAZEL_BIN:-}" ]]; then
    BAZEL_BIN=${NAVI_SCHED_BAZEL_BIN}
elif command -v bazelisk >/dev/null 2>&1; then
    BAZEL_BIN=bazelisk
else
    BAZEL_BIN=bazel
fi
export BAZELISK_HOME=${NAVI_SCHED_BAZELISK_HOME:-${HOME}/.cache/bazelisk}
PYTHON_BIN=${NAVI_SCHED_PYTHON_BIN:-/opt/conda310/bin/python3}
NAVI_SCHED_COMPILER_CONFIG=${NAVI_SCHED_COMPILER_CONFIG:-gcc}

if [[ "${NAVI_SCHED_COMPILER_CONFIG}" != "clang" &&
    "${NAVI_SCHED_COMPILER_CONFIG}" != "gcc" ]]; then
    echo "unsupported NaviSched compiler config: ${NAVI_SCHED_COMPILER_CONFIG}" >&2
    exit 1
fi

case "$(uname -m)" in
    x86_64)
        NAVI_SCHED_PLATFORM_TAG=linux_x86_64
        NAVI_SCHED_BAZEL_COMPILER_ARGS=("--config=${NAVI_SCHED_COMPILER_CONFIG}")
        NAVI_SCHED_BAZEL_PLATFORM_CONFIG=''
        ;;
    aarch64 | arm64)
        if [[ "${NAVI_SCHED_COMPILER_CONFIG}" != "gcc" ]]; then
            echo "NaviSched ARM packages require the GCC compiler config" >&2
            exit 1
        fi
        NAVI_SCHED_PLATFORM_TAG=linux_aarch64
        NAVI_SCHED_BAZEL_COMPILER_ARGS=(--config=gcc)
        NAVI_SCHED_BAZEL_PLATFORM_CONFIG=--config=arm
        ;;
    *)
        echo "unsupported NaviSched build architecture: $(uname -m)" >&2
        exit 1
        ;;
esac
export NAVI_SCHED_PLATFORM_TAG

if ! command -v "${BAZEL_BIN}" >/dev/null 2>&1; then
    echo "missing Bazel executable: ${BAZEL_BIN}" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "missing Python executable: ${PYTHON_BIN}" >&2
    exit 1
fi

PYTHON_VERSION=$("${PYTHON_BIN}" -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_LIBRARY_NAME="libpython${PYTHON_VERSION}.so.1.0"

PYTHON_HOME=$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)
PYTHON_LIB_DIR=${PYTHON_HOME}/lib
PYTHON_LIBRARY=${PYTHON_LIB_DIR}/${PYTHON_LIBRARY_NAME}
if [[ ! -f "${PYTHON_LIBRARY}" ]]; then
    echo "missing Python runtime library: ${PYTHON_LIBRARY}" >&2
    exit 1
fi

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export NAVI_PYTHON_HOME="${REPO_ROOT}/aios/navi/config_loader/python:\
${PYTHON_LIB_DIR}/python${PYTHON_VERSION}"

NAVI_SCHED_BAZEL_COMMON_ARGS=(
    "${NAVI_SCHED_BAZEL_COMPILER_ARGS[@]}"
    --config=${NAVI_SCHED_PYTHON_BAZEL_CONFIG:-py310}
)
if [[ -n "${NAVI_SCHED_BAZEL_PLATFORM_CONFIG}" ]]; then
    NAVI_SCHED_BAZEL_COMMON_ARGS+=("${NAVI_SCHED_BAZEL_PLATFORM_CONFIG}")
fi
if [[ -n "${NAVI_SCHED_BAZEL_JOBS:-}" ]]; then
    if [[ ! "${NAVI_SCHED_BAZEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "NAVI_SCHED_BAZEL_JOBS must be a positive integer" >&2
        exit 1
    fi
    NAVI_SCHED_BAZEL_COMMON_ARGS+=("--jobs=${NAVI_SCHED_BAZEL_JOBS}")
fi
if [[ -n "${NAVI_SCHED_PIP_REPOSITORY:-}" ]]; then
    if [[ ! -f "${NAVI_SCHED_PIP_REPOSITORY}/requirements.bzl" ]]; then
        echo "invalid NaviSched pip repository: ${NAVI_SCHED_PIP_REPOSITORY}" >&2
        exit 1
    fi
    NAVI_SCHED_BAZEL_COMMON_ARGS+=(
        "--override_repository=pip=${NAVI_SCHED_PIP_REPOSITORY}"
    )
fi
NAVI_SCHED_BAZEL_OPT_ARGS=(
    "${NAVI_SCHED_BAZEL_COMMON_ARGS[@]}"
    -c opt
)

navi_sched_bazel_output_bin() {
    "${BAZEL_BIN}" info "$@" bazel-bin
}
