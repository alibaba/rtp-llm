#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAVI_SCHED_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
VERSION=0.1.0
PACKAGE_NAME="navi_sched_simulate-${VERSION}.tar.gz"
ARCHIVE_PATH="${NAVI_SCHED_DIR}/release/${PACKAGE_NAME}"
# NAVI_SCHED_OSS_UPLOAD_DIR selects the SuezOps package channel.
UPLOAD_DIR=${NAVI_SCHED_OSS_UPLOAD_DIR:-test}
# NAVI_SCHED_EMP_ID owns the SuezOps package created after upload.
EMP_ID=${NAVI_SCHED_EMP_ID:-103303}
# NAVI_SCHED_OSS_ENDPOINT overrides the internal OSS endpoint for uploads.
OSS_ENDPOINT=${NAVI_SCHED_OSS_ENDPOINT:-oss-cn-hangzhou-internal.aliyuncs.com}
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
OSS_PREFIX=oss://alibaba-aios-package/app/navi_sched/package
OSS_PATH=${OSS_PREFIX}/${UPLOAD_DIR}/${VERSION}/${TIMESTAMP}/${PACKAGE_NAME}
PUBLIC_PREFIX=https://alibaba-aios-package.oss-cn-hangzhou.aliyuncs.com
PACKAGE_URL=${PUBLIC_PREFIX}/${OSS_PATH#oss://alibaba-aios-package/}
OSSUTIL=${NAVI_SCHED_OSSUTIL:-ossutil}

"${SCRIPT_DIR}/build_simulate_package.sh"

if ! command -v "${OSSUTIL}" >/dev/null 2>&1; then
    OSSUTIL="${HOME}/.local/bin/ossutil"
fi
if [[ ! -x "${OSSUTIL}" ]]; then
    echo "ossutil is required to upload ${ARCHIVE_PATH}" >&2
    exit 1
fi

"${OSSUTIL}" mkdir "${OSS_PATH%/*}/" -e "${OSS_ENDPOINT}"
"${OSSUTIL}" cp "${ARCHIVE_PATH}" "${OSS_PATH}" --parallel=10 -e "${OSS_ENDPOINT}"
echo "uploaded: ${OSS_PATH}"
"${SCRIPT_DIR}/SuezOpsPackage.py" \
    --package-url "${PACKAGE_URL}" \
    --version "${VERSION}" \
    --emp-id "${EMP_ID}"
