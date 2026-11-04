#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_env.sh"
VERSION=0.1.0
PACKAGE_NAME=navi_sched_simulate-${VERSION}
RELEASE_DIR=${NAVI_SCHED_RELEASE_DIR:-${NAVI_SCHED_DIR}/release}
PACKAGE_DIR=${RELEASE_DIR}/${PACKAGE_NAME}
ARCHIVE_PATH=${RELEASE_DIR}/${PACKAGE_NAME}.tar.gz
ABI_CONFIG=${NAVI_SCHED_BAZEL_ABI_CONFIG:-old_abi}
if [[ "${ABI_CONFIG}" != "old_abi" && "${ABI_CONFIG}" != "new_abi" ]]; then
    printf 'invalid NaviSched Bazel ABI config: %s\n' "${ABI_CONFIG}" >&2
    exit 2
fi
BUILD_ARGS=(
    "${NAVI_SCHED_BAZEL_OPT_ARGS[@]}"
    --config=release
    --config="${ABI_CONFIG}"
)
BAZEL_BUILD_ARGS=("${BUILD_ARGS[@]}")
if [[ -n "${NAVI_SCHED_BAZEL_PROFILE:-}" ]]; then
    BAZEL_BUILD_ARGS+=("--profile=${NAVI_SCHED_BAZEL_PROFILE}")
fi

cd "${REPO_ROOT}"
"${BAZEL_BIN}" --batch build \
    "${BAZEL_BUILD_ARGS[@]}" \
    //rtp_llm/flexlb/navi_sched:navi_sched_simulate_core.so \
    //rtp_llm/flexlb/navi_sched:navi_sched_wheel
OUTPUT_BIN=$(navi_sched_bazel_output_bin "${BUILD_ARGS[@]}")

find "${PACKAGE_DIR}" -depth -delete 2>/dev/null || true
mkdir -p "${PACKAGE_DIR}/bin" "${PACKAGE_DIR}/lib" "${PACKAGE_DIR}/python"
cp "${NAVI_SCHED_DIR}/test/simulator/README.md" "${PACKAGE_DIR}/README.md"
cp "${NAVI_SCHED_DIR}/test/simulator/NaviSchedSimulateLauncher.sh" \
    "${PACKAGE_DIR}/bin/navi_sched_simulate_agent"
cp "${NAVI_SCHED_DIR}/test/simulator/NaviSchedSimulateLauncher.sh" \
    "${PACKAGE_DIR}/bin/navi_sched_simulate_client"
cp "${NAVI_SCHED_DIR}/test/simulator/NaviSchedSimulateLauncher.sh" \
    "${PACKAGE_DIR}/bin/navi_sched_simulate_master"
chmod 0755 "${PACKAGE_DIR}/bin/navi_sched_simulate_agent" \
    "${PACKAGE_DIR}/bin/navi_sched_simulate_client" \
    "${PACKAGE_DIR}/bin/navi_sched_simulate_master"
WHEEL_PATH=${OUTPUT_BIN}/aios/navi_sched/tools/navi_sched-${VERSION}-cp310-cp310-linux_x86_64.whl
unzip -q \
    "${WHEEL_PATH}" \
    -d "${PACKAGE_DIR}/python"
cp "${NAVI_SCHED_DIR}/test/simulator/NaviSchedSimulateRole.py" \
    "${PACKAGE_DIR}/python/navi_sched_simulate.py"
cp "${NAVI_SCHED_DIR}/test/simulator/NaviSchedSimulateEntry.py" \
    "${PACKAGE_DIR}/python/navi_sched_simulate_entry.py"
cp "${NAVI_SCHED_DIR}/test/simulator/NaviSchedSimulateMaster.py" \
    "${PACKAGE_DIR}/python/NaviSchedSimulateMaster.py"
cp "${OUTPUT_BIN}/aios/navi_sched/navi_sched_simulate_core.so" \
    "${PACKAGE_DIR}/python/navi_sched/_navi_sched_core.so"
strip --strip-unneeded "${PACKAGE_DIR}/python/navi_sched/_navi_sched_core.so"

for runtime_lib in liblz4.so libzstd.so libelf.so libev.so; do
    runtime_path=$(find "${OUTPUT_BIN}/_solib_k8" \
        -name "${runtime_lib}" -print -quit)
    if [[ -z "${runtime_path}" ]]; then
        printf 'missing runtime library: %s\n' "${runtime_lib}" >&2
        exit 1
    fi
    cp -L "${runtime_path}" "${PACKAGE_DIR}/lib/${runtime_lib}"
    chmod u+w "${PACKAGE_DIR}/lib/${runtime_lib}"
done
cp -L "${PYTHON_LIBRARY}" \
    "${PACKAGE_DIR}/lib/libpython3.10.so.1.0"
chmod u+w "${PACKAGE_DIR}/lib/libpython3.10.so.1.0"

tar -C "${RELEASE_DIR}" -czf "${ARCHIVE_PATH}" "${PACKAGE_NAME}"
printf '%s\n' "${ARCHIVE_PATH}"
