#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_env.sh"
"${SCRIPT_DIR}/verify_unified_baseline.sh"
WHEEL_NAME=navi_sched-0.1.0-cp310-cp310-${NAVI_SCHED_PLATFORM_TAG}.whl
WHEEL_STEM=${WHEEL_NAME%-cp310-cp310-${NAVI_SCHED_PLATFORM_TAG}.whl}
WHEEL_VERSION=${WHEEL_STEM#navi_sched-}
RELEASE_DIR=${NAVI_SCHED_RELEASE_DIR:-${NAVI_SCHED_DIR}/release}
RELEASE_PATH=${RELEASE_DIR}/${WHEEL_NAME}
INSTALL_DIR=${NAVI_SCHED_DIR}/install
TEST_DIR=${NAVI_SCHED_DIR}/tmp/wheel_test
VLLM_TEST_DEPS=${NAVI_SCHED_DIR}/tmp/vllm_test_deps
SANDBOX_BASE=${NAVI_SCHED_SANDBOX_BASE:-/tmp/bazel-sandbox}
BUILD_ARGS=(
    "${NAVI_SCHED_BAZEL_OPT_ARGS[@]}"
    --config=release
    --config=old_abi
    --copt=-fno-omit-frame-pointer
)
BUILD_PROFILE_ARGS=()
if [[ -n "${NAVI_SCHED_WHEEL_BAZEL_PROFILE:-}" ]]; then
    BUILD_PROFILE_ARGS+=("--profile=${NAVI_SCHED_WHEEL_BAZEL_PROFILE}")
fi

cleanup() {
    rm -rf "${INSTALL_DIR}" "${TEST_DIR}" "${VLLM_TEST_DEPS}"
}

cd "${REPO_ROOT}"
mkdir -p "${SANDBOX_BASE}"
nice -n 10 "${BAZEL_BIN}" --batch build \
    --sandbox_base="${SANDBOX_BASE}" \
    "${BUILD_ARGS[@]}" \
    "${BUILD_PROFILE_ARGS[@]}" \
    //rtp_llm/flexlb/navi_sched:navi_sched_wheel
OUTPUT_BIN=$(navi_sched_bazel_output_bin "${BUILD_ARGS[@]}")
WHEEL_PATH=${OUTPUT_BIN}/aios/navi_sched/tools/${WHEEL_NAME}

mkdir -p "${RELEASE_DIR}"
rm -f "${RELEASE_DIR}/${WHEEL_STEM}-py3-none-any.whl" \
    "${RELEASE_DIR}/navi_sched_raw-${WHEEL_VERSION}-py3-none-any.whl"
if [[ -f "${RELEASE_PATH}" ]]; then
    chmod u+w "${RELEASE_PATH}"
fi
cp "${WHEEL_PATH}" "${RELEASE_PATH}"
chmod 0644 "${RELEASE_PATH}"
cleanup
trap 'cleanup' EXIT
mkdir -p "${INSTALL_DIR}" "${TEST_DIR}"
"${PYTHON_BIN}" -m pip install \
    --no-deps \
    --target="${INSTALL_DIR}" \
    "${WHEEL_PATH}"

CORE_SO=${INSTALL_DIR}/navi_sched/_navi_sched_core.so
PYTHON_RUNTIME=${INSTALL_DIR}/navi_sched/libpython3.10.so.1.0
unzip -p "${WHEEL_PATH}" '*/WHEEL' | grep -q '^Root-Is-Purelib: false$'
unzip -p "${WHEEL_PATH}" '*/WHEEL' |
    grep -q "^Tag: cp310-cp310-${NAVI_SCHED_PLATFORM_TAG}$"
PYTHON_SYMBOL_COUNT=$(nm -D --defined-only "${CORE_SO}" |
    grep -Ec ' PyInit__navi_sched_core(@@NAVI_SCHED_PYBIND_0\.1)?$' || true)
if [[ "${PYTHON_SYMBOL_COUNT}" -eq 0 ]]; then
    echo "navi_sched wheel does not export its Python entry point" >&2
    exit 1
fi
PRIVATE_SYMBOL_COUNT=$(nm -D --defined-only "${CORE_SO}" |
    grep -Ec ' [A-Z] (CRYPTO_|EVP_|OPENSSL_|SSL_)' || true)
if [[ "${PRIVATE_SYMBOL_COUNT}" -ne 0 ]]; then
    echo "navi_sched wheel exports private crypto symbols" >&2
    exit 1
fi
if [[ ! -f "${PYTHON_RUNTIME}" ]]; then
    echo "navi_sched wheel is missing its Python runtime library" >&2
    exit 1
fi
if ! ldd "${CORE_SO}" | grep -Fq "libpython3.10.so.1.0 => ${PYTHON_RUNTIME}"; then
    echo "navi_sched extension does not resolve Python from its wheel" >&2
    ldd "${CORE_SO}" >&2
    exit 1
fi
if [[ ! -f "${INSTALL_DIR}/scripts/update_scheduler_config.sh" ]]; then
    echo "navi_sched wheel is missing the scheduler update command" >&2
    exit 1
fi

WHEEL_PATH="${WHEEL_PATH}" "${PYTHON_BIN}" - <<'PY'
import os
import zipfile

wheel_path = os.environ["WHEEL_PATH"]
with zipfile.ZipFile(wheel_path) as wheel:
    names = wheel.namelist()
    metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
    metadata = wheel.read(metadata_name).decode("utf-8")
    entry_points_name = next(
        name for name in names if name.endswith(".dist-info/entry_points.txt")
    )
    entry_points = wheel.read(entry_points_name).decode("utf-8")
    bad = [
        name
        for name in names
        if "test_plugin" in name
        or "TestKernel" in name
        or "TestResource" in name
        or "simulator" in name.lower()
        or "/testdata/" in name
        or name.startswith("rtp_llm/flexlb/navi_sched/vendor/navi/testdata/")
    ]
    if "navi_sched/_navi_sched_core.so" not in names:
        raise RuntimeError("navi_sched wheel is missing its extension module")
    if "navi_sched/libpython3.10.so.1.0" not in names:
        raise RuntimeError("navi_sched wheel is missing its Python runtime library")
    if "navi_sched_vllm_plugin.py" not in names:
        raise RuntimeError("navi_sched wheel is missing its lightweight vLLM plugin")
    if "scripts/update_scheduler_config.sh" not in names:
        raise RuntimeError("navi_sched wheel is missing the scheduler update command")
    if "navi_sched/lib/libnavi_sched.so" in names:
        raise RuntimeError("navi_sched wheel contains a duplicate NaviSched library")
if bad:
    raise RuntimeError("navi_sched wheel contains test-only files: %s" % bad)
if any(line.startswith("Requires-Dist:") for line in metadata.splitlines()):
    raise RuntimeError("navi_sched wheel must not declare Python dependencies")
expected_entry_point = (
    "navi_sched = navi_sched_vllm_plugin:register_vllm_plugin"
)
if expected_entry_point not in entry_points.splitlines():
    raise RuntimeError("navi_sched wheel has an invalid vLLM plugin entry point")
expected_replay_entry_point = "navi_sched_replay = navi_sched.replay_cli:main"
if expected_replay_entry_point not in entry_points.splitlines():
    raise RuntimeError("navi_sched wheel is missing the replay control command")
PY

PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${INSTALL_DIR}" \
    "${PYTHON_BIN}" -m navi_sched.replay_cli --help | grep -q '查询和控制'

for example in agent client embedded_cluster master vllm_route; do
    PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="${INSTALL_DIR}" \
        "${PYTHON_BIN}" \
        "${NAVI_SCHED_DIR}/example/${example}.py" --help \
        >/dev/null
done

cp "${SCRIPT_DIR}/test.py" "${TEST_DIR}"
cd "${TEST_DIR}"
PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${INSTALL_DIR}" \
    "${PYTHON_BIN}" test.py

PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${INSTALL_DIR}" \
    VLLM_NAVI_SCHED_ENABLE=0 \
    "${PYTHON_BIN}" -c 'import sys; import navi_sched_vllm_plugin;
navi_sched_vllm_plugin.register_vllm_plugin()
assert "navi_sched" not in sys.modules'

PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${INSTALL_DIR}" \
    "${PYTHON_BIN}" \
    "${NAVI_SCHED_DIR}/test/vllm/ProductionStackRouterTest.py"

PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${INSTALL_DIR}" \
    "${PYTHON_BIN}" \
    "${NAVI_SCHED_DIR}/test/vllm/VLLMBlockHashTest.py"

PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${INSTALL_DIR}" \
    "${PYTHON_BIN}" \
    "${NAVI_SCHED_DIR}/test/vllm/VLLMCachePolicyTest.py"

PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${INSTALL_DIR}" \
    "${PYTHON_BIN}" \
    "${NAVI_SCHED_DIR}/test/vllm/VLLMEngineObserverTest.py"

PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${INSTALL_DIR}" \
    timeout 120 "${PYTHON_BIN}" \
    "${NAVI_SCHED_DIR}/test/vllm/VLLMAdapterEndToEndTest.py"

PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${INSTALL_DIR}" \
    "${PYTHON_BIN}" \
    "${NAVI_SCHED_DIR}/test/dashserving/DashServingAdapterTest.py"

PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${INSTALL_DIR}" \
    timeout 180 "${PYTHON_BIN}" \
    "${NAVI_SCHED_DIR}/example/embedded_cluster.py" \
    --agent-count "${NAVI_SCHED_TEST_AGENT_COUNT:-10}" \
    --timeout-ms 30000 \
    --vis-output-dir "vis"
test -s "${TEST_DIR}/vis/master_graph.vis"
test -s "${TEST_DIR}/vis/master_schedule_graph.vis"
test -s "${TEST_DIR}/vis/agent_graph.vis"
test -s "${TEST_DIR}/vis/client_graph.vis"
