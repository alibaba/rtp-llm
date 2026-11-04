#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_env.sh"
REPLAY_DIR=${NAVI_SCHED_REPLAY_DIR:-\
    ${NAVI_SCHED_DIR}/tmp/cost_scheduler_debug/replays}
GENERATED_REPORT=${NAVI_SCHED_DIR}/tmp/cost_scheduler_debug/comparison_report.json
PACKAGED_REPORT=${NAVI_SCHED_DIR}/test/data/cost_scheduler/comparison_report.json
if [[ -n "${NAVI_SCHED_COMPARISON_REPORT:-}" ]]; then
    REPORT_FILE=${NAVI_SCHED_COMPARISON_REPORT}
elif [[ -f "${GENERATED_REPORT}" ]]; then
    REPORT_FILE=${GENERATED_REPORT}
else
    REPORT_FILE=${PACKAGED_REPORT}
fi
VIEWER_PORT=${NAVI_SCHED_REPLAY_VIEWER_PORT:-18780}
VIEWER=${NAVI_SCHED_DIR}/tools/replay_viewer/serve.py

VIEWER_URL=$("${PYTHON_BIN}" "${VIEWER}" \
    --port "${VIEWER_PORT}" --replay-dir "${REPLAY_DIR}" \
    --report-file "${REPORT_FILE}")

cat <<EOF
NaviSched replay viewer 已在后台运行，不占用当前 shell。
Replay 目录: ${REPLAY_DIR}
报告文件: ${REPORT_FILE}
访问地址: ${VIEWER_URL}
EOF
