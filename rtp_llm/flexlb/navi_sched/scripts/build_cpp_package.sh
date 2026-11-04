#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_env.sh"
VERSION=0.1.0
PACKAGE_NAME=navi_sched_cpp-${VERSION}
RELEASE_DIR=${NAVI_SCHED_RELEASE_DIR:-${NAVI_SCHED_DIR}/release}
PACKAGE_DIR=${RELEASE_DIR}/${PACKAGE_NAME}
ARCHIVE_PATH=${RELEASE_DIR}/${PACKAGE_NAME}.tar.gz
BUILD_ARGS=(
    "${NAVI_SCHED_BAZEL_OPT_ARGS[@]}"
    --config=new_abi
    --config=release
)
BUILD_PROFILE_ARGS=()
if [[ -n "${NAVI_SCHED_BAZEL_PROFILE:-}" ]]; then
    BUILD_PROFILE_ARGS+=("--profile=${NAVI_SCHED_BAZEL_PROFILE}")
fi

cd "${REPO_ROOT}"
BUILD_COMMAND=(
    "${BAZEL_BIN}" --batch build
    "${BUILD_ARGS[@]}"
)
if (( ${#BUILD_PROFILE_ARGS[@]} )); then
    BUILD_COMMAND+=("${BUILD_PROFILE_ARGS[@]}")
fi
"${BUILD_COMMAND[@]}" \
    //rtp_llm/flexlb/navi_sched:libnavi_sched.so \
    //rtp_llm/flexlb/navi_sched:navi_sched_public_api_compile \
    //rtp_llm/flexlb/navi_sched:rtp_llm_adapter_public_api_compile
OUTPUT_BIN=$(navi_sched_bazel_output_bin "${BUILD_ARGS[@]}")

find "${PACKAGE_DIR}" -depth -delete 2>/dev/null || true
mkdir -p "${PACKAGE_DIR}/adapters/rtp_llm" \
    "${PACKAGE_DIR}/include" "${PACKAGE_DIR}/lib"
cp "${NAVI_SCHED_DIR}/cpp_package/BUILD.tpl" "${PACKAGE_DIR}/BUILD"
cp "${NAVI_SCHED_DIR}/cpp_package/WORKSPACE.tpl" "${PACKAGE_DIR}/WORKSPACE"
cp "${NAVI_SCHED_DIR}/cpp_package/adapters/rtp_llm/BUILD.tpl" \
    "${PACKAGE_DIR}/adapters/rtp_llm/BUILD"
printf '%s\n' "${VERSION}" >"${PACKAGE_DIR}/VERSION"
printf '%s\n' "cxx11" >"${PACKAGE_DIR}/CXX_ABI"
cp "${OUTPUT_BIN}/aios/navi_sched/libnavi_sched.so" \
    "${PACKAGE_DIR}/lib/libnavi_sched.so"
chmod u+w "${PACKAGE_DIR}/lib/libnavi_sched.so"
strip --strip-unneeded "${PACKAGE_DIR}/lib/libnavi_sched.so"
PUBLIC_SYMBOL_COUNT=$(nm -D --defined-only \
    "${PACKAGE_DIR}/lib/libnavi_sched.so" | c++filt |
    grep -Ec ' navi_sched::' || true)
if [[ "${PUBLIC_SYMBOL_COUNT}" -eq 0 ]]; then
    echo "navi_sched C++ package does not export its public API" >&2
    exit 1
fi
if [[ "${PUBLIC_SYMBOL_COUNT}" -gt 200 ]]; then
    echo "navi_sched C++ package exports too many symbols: ${PUBLIC_SYMBOL_COUNT}" >&2
    exit 1
fi
PUBLIC_SYMBOLS=$(nm -D --defined-only \
    "${PACKAGE_DIR}/lib/libnavi_sched.so" | c++filt)
for public_symbol in \
    'NaviSchedAgent::init' \
    'NaviSchedAgent::addBatchRequest' \
    'NaviSchedBatchFinishReporter::finishBatch' \
    'setRtpLLMForwardBatchHandler' \
    'retainRtpLLMReadyBatch' \
    'abortRetainedRtpLLMBatch' \
    'recordRtpLLMObservedPrefillBatch' \
    'startRetainedRtpLLMBatches'; do
    if ! grep -Eq "${public_symbol}" <<<"${PUBLIC_SYMBOLS}"; then
        echo "missing NaviSched public symbol: ${public_symbol}" >&2
        exit 1
    fi
done
for private_symbol in \
    '::Impl::' \
    'AgentReportR' \
    'MasterStateR' \
    'LatencyModelR' \
    'NonLinearPrefillModelR' \
    'AgentGraph' \
    'MasterCollectorManager' \
    'ScheduleKernel' \
    'NaviSchedClient' \
    'NaviSchedMaster'; do
    if grep -Eq "${private_symbol}" <<<"${PUBLIC_SYMBOLS}"; then
        echo "navi_sched C++ package exports internal symbol: ${private_symbol}" >&2
        exit 1
    fi
done
PRIVATE_SYMBOL_COUNT=$(nm -D --defined-only \
    "${PACKAGE_DIR}/lib/libnavi_sched.so" |
    grep -Ec ' (CRYPTO_|EVP_|OPENSSL_|SSL_)' || true)
if [[ "${PRIVATE_SYMBOL_COUNT}" -ne 0 ]]; then
    echo "navi_sched C++ package exports private crypto symbols" >&2
    exit 1
fi
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
    "${PACKAGE_DIR}/lib/${PYTHON_LIBRARY_NAME}"
chmod u+w "${PACKAGE_DIR}/lib/${PYTHON_LIBRARY_NAME}"
sed -i "s/libpython3\\.10\\.so\\.1\\.0/${PYTHON_LIBRARY_NAME}/g" \
    "${PACKAGE_DIR}/BUILD"
patchelf --set-rpath '$ORIGIN' "${PACKAGE_DIR}/lib/libnavi_sched.so"

PUBLIC_API_DEPFILE=$(find \
    "${OUTPUT_BIN}/aios/navi_sched/_objs" \
    -path '*/NaviSchedPublicApiCompileTest.pic.d' -print -quit)
if [[ -z "${PUBLIC_API_DEPFILE}" ]]; then
    echo "missing NaviSched public API dependency file" >&2
    exit 1
fi

copy_header() {
    local source_path=$1
    local include_path
    if [[ "${source_path}" == *'/_virtual_includes/'* ]]; then
        include_path=${source_path#*/_virtual_includes/}
        include_path=${include_path#*/}
    elif [[ "${source_path}" == bazel-out/*/bin/* ]]; then
        include_path=${source_path#*/bin/}
    else
        include_path=${source_path}
    fi
    mkdir -p "${PACKAGE_DIR}/include/$(dirname "${include_path}")"
    cp -L "${REPO_ROOT}/${source_path}" \
        "${PACKAGE_DIR}/include/${include_path}"
}

while IFS= read -r header; do
    copy_header "${header}"
done < <(sed ':next; /\\$/ { N; s/\\\n/ /; b next; }' \
    "${PUBLIC_API_DEPFILE}" | tr ' ' '\n' |
    sed 's/:$//' | grep -E '^(aios|bazel-out/.*/bin/aios)/.*\.h$' |
    grep -Ev '(^|/)(test|testdata)/|Test(Kernel|Resource)|Simulated' |
    sort -u)
copy_header "rtp_llm/flexlb/navi_sched/adapters/rtp_llm/RtpLLMAdapter.h"

FORBIDDEN_HEADER=$(find "${PACKAGE_DIR}/include" -type f |
    grep -E '/(test|testdata)/|Test(Kernel|Resource)|Simulated' | head -n 1 || true)
if [[ -n "${FORBIDDEN_HEADER}" ]]; then
    echo "C++ package contains test-only header: ${FORBIDDEN_HEADER}" >&2
    exit 1
fi
HEADER_COUNT=$(find "${PACKAGE_DIR}/include" -type f -name '*.h' | wc -l)
if [[ "${HEADER_COUNT}" -gt 2000 ]]; then
    echo "C++ package header count is too large: ${HEADER_COUNT}" >&2
    exit 1
fi
PACKAGE_SIZE_KB=$(du -sk "${PACKAGE_DIR}" | awk '{print $1}')
if [[ "${PACKAGE_SIZE_KB}" -gt 262144 ]]; then
    echo "C++ package is too large: ${PACKAGE_SIZE_KB} KiB" >&2
    exit 1
fi

printf '#include "rtp_llm/flexlb/navi_sched/adapters/rtp_llm/RtpLLMAdapter.h"\n' |
    g++ -std=c++17 -I"${PACKAGE_DIR}/include" -x c++ -fsyntax-only -
g++ -std=c++17 -I"${PACKAGE_DIR}/include" -fsyntax-only \
    "${NAVI_SCHED_DIR}/test/package/NaviSchedPublicApiCompileTest.cpp"

# Compile through the packaged Bazel targets so undeclared transitive headers fail.
SDK_CONSUMER_DIR=$(mktemp -d "${TMPDIR:-/tmp}/navi_sched_cpp_consumer.XXXXXX")
cleanup_sdk_consumer() {
    rm -rf "${SDK_CONSUMER_DIR}"
}
trap cleanup_sdk_consumer EXIT
printf 'workspace(name = "navi_sched_cpp_consumer")\n' > \
    "${SDK_CONSUMER_DIR}/WORKSPACE"
cp "${REPO_ROOT}/.bazeliskrc" "${SDK_CONSUMER_DIR}/.bazeliskrc"
printf '%s\n' \
    'cc_library(' \
    '    name = "sdk_compile",' \
    '    srcs = ["sdk_compile.cpp"],' \
    '    deps = ["@navi_sched_cpp//:navi_sched"],' \
    ')' >"${SDK_CONSUMER_DIR}/BUILD"
printf '%s\n' \
    '#include "rtp_llm/flexlb/navi_sched/adapters/rtp_llm/RtpLLMAdapter.h"' \
    'void verifyNaviSchedCppSdk() {}' >"${SDK_CONSUMER_DIR}/sdk_compile.cpp"
(cd "${SDK_CONSUMER_DIR}" &&
    "${BAZEL_BIN}" --batch build \
        --override_repository="navi_sched_cpp=${PACKAGE_DIR}" \
        //:sdk_compile)
cleanup_sdk_consumer
trap - EXIT

g++ -std=c++17 \
    -I"${PACKAGE_DIR}/include" \
    "${NAVI_SCHED_DIR}/test/package/NaviSchedPackageLinkTest.cpp" \
    -L"${PACKAGE_DIR}/lib" \
    -Wl,-rpath,"${PACKAGE_DIR}/lib" \
    -Wl,-rpath-link,"${PACKAGE_DIR}/lib" \
    -lnavi_sched \
    -o "${PACKAGE_DIR}/navi_sched_package_link_test"
"${PACKAGE_DIR}/navi_sched_package_link_test"
rm "${PACKAGE_DIR}/navi_sched_package_link_test"

if [[ $(find "${PACKAGE_DIR}/lib" -maxdepth 1 -type f \
    -name 'libnavi_sched.so*' | wc -l) -ne 1 ]]; then
    echo "navi_sched C++ package must contain one NaviSched library" >&2
    exit 1
fi

tar -C "${RELEASE_DIR}" -czf "${ARCHIVE_PATH}" "${PACKAGE_NAME}"
printf '%s\n' "${ARCHIVE_PATH}"
