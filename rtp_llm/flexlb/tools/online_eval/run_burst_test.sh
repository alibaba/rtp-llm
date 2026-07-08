#!/opt/homebrew/bin/bash
#
# FlexLB Mock Engine — Burst-traffic stress test (real trace + multi-speed replay)
#
# Uses the real production trace `trace_30min.jsonl` (8332 requests, 13.3 min,
# avg 10.5 QPS, 13 natural burst segments) with 6 replay speed levels
# (5/10/15/20/30/50x) to amplify pressure.
#
# Pattern: follows run_stability_test.sh
#   1. Pre-build FlexLB JAR
#   2. For each speed: start monitor → run_online_eval.sh → stop monitor
#   3. Generate analysis report via analyze_burst_results.py
#
# Usage:   bash run_burst_test.sh
# Custom:  SPEEDS="10 20" bash run_burst_test.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXLB_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Ensure bash 5+ is in PATH (macOS default bash 3.2 lacks mapfile)
export PATH="/opt/homebrew/bin:${PATH}"

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment)
# ---------------------------------------------------------------------------
DEFAULT_SPEEDS=(5 10 15 20 30 50)
if [[ -n "${SPEEDS:-}" ]]; then
    read -r -a SPEEDS <<< "${SPEEDS}"
elif [[ -n "${SPEEDS_OVERRIDE:-}" ]]; then
    read -r -a SPEEDS <<< "${SPEEDS_OVERRIDE}"
else
    SPEEDS=("${DEFAULT_SPEEDS[@]}")
fi

N_PREFILL="${N_PREFILL:-2}"
N_DECODE="${N_DECODE:-4}"
SCHEDULE_MODE="${SCHEDULE_MODE:-batch}"
LOAD_BALANCE_STRATEGY="${LOAD_BALANCE_STRATEGY:-COST_BASED_PREFILL}"
DECODE_LOAD_BALANCE_STRATEGY="${DECODE_LOAD_BALANCE_STRATEGY:-COST_BASED_DECODE}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1024}"
SLA_TTFT_MS="${SLA_TTFT_MS:-500}"
ZERO_OUTPUT_POLICY="${ZERO_OUTPUT_POLICY:-one}"
LIMIT="${LIMIT:-0}"
DURATION_S="${DURATION_S:-0}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-2}"

# Batcher defaults (consistent with run_online_eval.sh)
FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES="${FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES:-2}"
FLEXLB_BATCH_FIXED_WAIT_MS="${FLEXLB_BATCH_FIXED_WAIT_MS:-220}"
FLEXLB_BATCH_PREDICT_THRESHOLD_MS="${FLEXLB_BATCH_PREDICT_THRESHOLD_MS:-550}"

# Decode concurrency limit (was 132 in previous test; 2000 to eliminate NO_AVAILABLE_WORKER rejections)
DECODE_CONCURRENCY_LIMIT="${DECODE_CONCURRENCY_LIMIT:-2000}"

# gRPC timeout: 3600s (1 hour) — large output_len requests need long decode time
TIMEOUT_MS="${TIMEOUT_MS:-3600000}"

# Ports (match run_online_eval.sh defaults)
FLEXLB_HTTP_ADDR="${FLEXLB_HTTP_ADDR:-127.0.0.1:7001}"
FLEXLB_MANAGEMENT_PORT="${FLEXLB_MANAGEMENT_PORT:-7002}"
MOCK_BASE_GRPC_PORT="${MOCK_BASE_GRPC_PORT:-55151}"
MOCK_HTTP_PORT=$((MOCK_BASE_GRPC_PORT - 1))

# Maven
MAVEN_PROFILES="${MAVEN_PROFILES:-opensource,!internal}"
FLEXLB_JAR="${FLEXLB_DIR}/flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar"

# Output
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/run}"
BURST_RUN_ID="burst_$(date +%Y%m%d_%H%M%S)"
BURST_DIR="${RUN_ROOT}/${BURST_RUN_ID}"
mkdir -p "${BURST_DIR}"

# ---------------------------------------------------------------------------
# Pre-build FlexLB JAR (avoid JAVA_TOOL_OPTIONS leaking into Maven)
# ---------------------------------------------------------------------------
if [[ ! -f "${FLEXLB_JAR}" ]]; then
    echo "============================================"
    echo "Pre-building FlexLB JAR..."
    echo "============================================"
    (cd "${FLEXLB_DIR}" && ./mvnw -P"${MAVEN_PROFILES}" -pl flexlb-api -am package -DskipTests)
    if [[ ! -f "${FLEXLB_JAR}" ]]; then
        echo "ERROR: JAR build failed: ${FLEXLB_JAR}" >&2
        exit 1
    fi
fi
echo "JAR: ${FLEXLB_JAR}"
echo ""

# ---------------------------------------------------------------------------
# Main loop: one round per replay speed
# ---------------------------------------------------------------------------
ROUND_DIRS=()
MONITOR_PID=""
cleanup_burst() {
    if [[ -n "${MONITOR_PID}" ]]; then
        echo "Cleaning up monitor PID=${MONITOR_PID}..." >&2
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
    fi
}
trap cleanup_burst EXIT

echo "============================================"
echo "FlexLB Burst Traffic Stress Test"
echo "(Real trace + multi-speed replay)"
echo "============================================"
echo "Speeds: ${SPEEDS[*]}"
echo "Trace:  data/online_logs/trace_30min.jsonl (8332 requests, 13.3 min)"
echo "Config: N_PREFILL=${N_PREFILL} N_DECODE=${N_DECODE} SCHEDULE_MODE=${SCHEDULE_MODE}"
echo "        LOAD_BALANCE_STRATEGY=${LOAD_BALANCE_STRATEGY} DECODE_LOAD_BALANCE_STRATEGY=${DECODE_LOAD_BALANCE_STRATEGY}"
echo "        MAX_CONCURRENCY=${MAX_CONCURRENCY} SLA_TTFT_MS=${SLA_TTFT_MS}"
echo "        LIMIT=${LIMIT} ZERO_OUTPUT_POLICY=${ZERO_OUTPUT_POLICY} DURATION_S=${DURATION_S}"
echo "        MAX_INFLIGHT_BATCHES=${FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES} WAIT_MS=${FLEXLB_BATCH_FIXED_WAIT_MS}"
echo "        DECODE_CONCURRENCY_LIMIT=${DECODE_CONCURRENCY_LIMIT}"
echo "        TIMEOUT_MS=${TIMEOUT_MS}"
echo "Monitor interval: ${MONITOR_INTERVAL}s"
echo "Output: ${BURST_DIR}"
echo "============================================"
echo ""

for speed in "${SPEEDS[@]}"; do
    ROUND_ID="burst_${speed}x_$(date +%Y%m%d_%H%M%S)"
    ROUND_DIR="${RUN_ROOT}/${ROUND_ID}"
    mkdir -p "${ROUND_DIR}"
    ROUND_DIRS+=("${ROUND_DIR}")

    echo ">>> Speed ${speed}x starting (output: ${ROUND_DIR})"

    # 1. Start stability monitor in background
    python3 "${SCRIPT_DIR}/stability_monitor.py" \
        --flexlb-http-addr "${FLEXLB_HTTP_ADDR}" \
        --management-port "${FLEXLB_MANAGEMENT_PORT}" \
        --mock-http-port "${MOCK_HTTP_PORT}" \
        --interval "${MONITOR_INTERVAL}" \
        --output "${ROUND_DIR}/monitor.jsonl" &
    MONITOR_PID=$!
    echo "    Monitor PID=${MONITOR_PID}, output: ${ROUND_DIR}/monitor.jsonl"

    # 2. Run online eval (environment-driven, zero modification)
    #    Uses default TRACE_FILE (real trace), only changes REPLAY_SPEED
    echo "    Starting run_online_eval.sh (REPLAY_SPEED=${speed})..."
    set +e
    env \
        REPLAY_SPEED="${speed}" \
        LIMIT="${LIMIT}" \
        DURATION_S="${DURATION_S}" \
        ZERO_OUTPUT_POLICY="${ZERO_OUTPUT_POLICY}" \
        RUN_ROOT="${RUN_ROOT}" \
        RUN_ID="${ROUND_ID}" \
        N_PREFILL="${N_PREFILL}" \
        N_DECODE="${N_DECODE}" \
        SCHEDULE_MODE="${SCHEDULE_MODE}" \
        LOAD_BALANCE_STRATEGY="${LOAD_BALANCE_STRATEGY}" \
        DECODE_LOAD_BALANCE_STRATEGY="${DECODE_LOAD_BALANCE_STRATEGY}" \
        MAX_CONCURRENCY="${MAX_CONCURRENCY}" \
        SLA_TTFT_MS="${SLA_TTFT_MS}" \
        FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES="${FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES}" \
        FLEXLB_BATCH_FIXED_WAIT_MS="${FLEXLB_BATCH_FIXED_WAIT_MS}" \
        FLEXLB_BATCH_PREDICT_THRESHOLD_MS="${FLEXLB_BATCH_PREDICT_THRESHOLD_MS}" \
        DECODE_CONCURRENCY_LIMIT="${DECODE_CONCURRENCY_LIMIT}" \
        TIMEOUT_MS="${TIMEOUT_MS}" \
        bash "${SCRIPT_DIR}/run_online_eval.sh" 2>&1 | tee "${ROUND_DIR}/eval.stdout"
    EVAL_EXIT=$?
    set -e

    if [[ ${EVAL_EXIT} -ne 0 ]]; then
        echo "    WARNING: run_online_eval.sh exited with code ${EVAL_EXIT}"
    fi

    # 3. Stop monitor
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""

    echo "<<< Speed ${speed}x done"
    echo ""

    # 4. Sleep between rounds for port release
    if [[ "${speed}" != "${SPEEDS[-1]}" ]]; then
        echo "    Waiting 3s for port release..."
        sleep 3
    fi
done

# ---------------------------------------------------------------------------
# Generate analysis report
# ---------------------------------------------------------------------------
echo "============================================"
echo "Generating burst traffic analysis report..."
echo "============================================"

SPEEDS_STR="${SPEEDS[*]}"
REPORT_PATH="${BURST_DIR}/burst_traffic_report.md"
DOCS_REPORT_PATH="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)/docs/flexlb-burst-traffic-test-report.md"

python3 "${SCRIPT_DIR}/analyze_burst_results.py" \
    --run-dirs "${ROUND_DIRS[@]}" \
    --speeds ${SPEEDS_STR} \
    --sla-ttft-ms "${SLA_TTFT_MS}" \
    --output "${REPORT_PATH}"

# Also copy to docs/ for discoverability
mkdir -p "$(dirname "${DOCS_REPORT_PATH}")"
cp "${REPORT_PATH}" "${DOCS_REPORT_PATH}" 2>/dev/null || true

echo ""
echo "============================================"
echo "Burst traffic test complete"
echo "============================================"
echo "Report: ${REPORT_PATH}"
echo "Also at: ${DOCS_REPORT_PATH}"
echo ""
echo "Per-speed outputs:"
for dir in "${ROUND_DIRS[@]}"; do
    speed_label="$(basename "${dir}" | sed 's/^burst_//; s/_.*$//')"
    echo "  - ${speed_label}: ${dir}/"
    echo "      summary: ${dir}/load_client/summary.json"
    echo "      monitor: ${dir}/monitor.jsonl"
    echo "      flexlb:  ${dir}/flexlb.log"
done
echo ""
echo "View report: cat ${REPORT_PATH}"
