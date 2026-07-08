#!/opt/homebrew/bin/bash
#
# FlexLB L1 — Mock engine 调度层高压力稳定性测试
#
# 三轮递增压力 (REPLAY_SPEED=10/20/50)，每轮：
#   1. 后台启动 stability_monitor.py (轮询 JVM/inflight/mock 指标)
#   2. 通过 JAVA_TOOL_OPTIONS 注入 GC 日志 + HeapDump
#   3. 调用 run_online_eval.sh (零修改复用，环境变量驱动)
#   4. 停止监控
# 最终生成多轮对比报告 (PASS/FAIL 判定)。
#
# 用法:  bash run_stability_test.sh
# 自定义: STABILITY_SPEEDS="10 20" bash run_stability_test.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 确保 bash 5+ 在 PATH 中 (macOS 自带 bash 3.2 不支持 mapfile)
export PATH="/opt/homebrew/bin:${PATH}"

# ---------------------------------------------------------------------------
# 配置 (均可通过环境变量覆盖)
# ---------------------------------------------------------------------------
DEFAULT_SPEEDS=(10 20 50)
if [[ -n "${STABILITY_SPEEDS:-}" ]]; then
  read -r -a SPEEDS <<< "${STABILITY_SPEEDS}"
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
LIMIT="${LIMIT:-1000}"
ZERO_OUTPUT_POLICY="${ZERO_OUTPUT_POLICY:-one}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-2}"

# 端口 (与 run_online_eval.sh 默认值一致)
FLEXLB_HTTP_ADDR="${FLEXLB_HTTP_ADDR:-127.0.0.1:7001}"
FLEXLB_MANAGEMENT_PORT="${FLEXLB_MANAGEMENT_PORT:-7002}"
MOCK_BASE_GRPC_PORT="${MOCK_BASE_GRPC_PORT:-55151}"
MOCK_HTTP_PORT=$((MOCK_BASE_GRPC_PORT - 1))

# 输出目录
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/run}"
STABILITY_RUN_ID="stability_$(date +%Y%m%d_%H%M%S)"
STABILITY_DIR="${RUN_ROOT}/${STABILITY_RUN_ID}"
mkdir -p "${STABILITY_DIR}"

# ---------------------------------------------------------------------------
# 预构建 FlexLB jar (避免 JAVA_TOOL_OPTIONS 泄漏至 Maven 进程)
# ---------------------------------------------------------------------------
FLEXLB_JAR="${FLEXLB_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}/flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar"
MAVEN_PROFILES="${MAVEN_PROFILES:-opensource,!internal}"
if [[ ! -f "${FLEXLB_JAR}" ]]; then
    echo "预构建 FlexLB jar..."
    (cd "$(cd "${SCRIPT_DIR}/../.." && pwd)" && ./mvnw -P"${MAVEN_PROFILES}" -pl flexlb-api -am package -DskipTests)
    if [[ ! -f "${FLEXLB_JAR}" ]]; then
        echo "ERROR: jar 构建失败: ${FLEXLB_JAR}" >&2
        exit 1
    fi
fi
echo "JAR: ${FLEXLB_JAR}"
echo ""

ROUND_DIRS=()

MONITOR_PID=""
cleanup_stability() {
    if [[ -n "${MONITOR_PID}" ]]; then
        echo "清理监控进程 PID=${MONITOR_PID}..." >&2
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
    fi
}
trap cleanup_stability EXIT

echo "============================================"
echo "FlexLB L1 稳定性测试"
echo "============================================"
echo "压力级别: ${SPEEDS[*]}"
echo "配置: N_PREFILL=${N_PREFILL} N_DECODE=${N_DECODE} SCHEDULE_MODE=${SCHEDULE_MODE}"
echo "      LOAD_BALANCE_STRATEGY=${LOAD_BALANCE_STRATEGY} DECODE_LOAD_BALANCE_STRATEGY=${DECODE_LOAD_BALANCE_STRATEGY}"
echo "      MAX_CONCURRENCY=${MAX_CONCURRENCY} SLA_TTFT_MS=${SLA_TTFT_MS} LIMIT=${LIMIT}"
echo "      ZERO_OUTPUT_POLICY=${ZERO_OUTPUT_POLICY}"
echo "监控间隔: ${MONITOR_INTERVAL}s"
echo "输出目录: ${STABILITY_DIR}"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# 主循环: 每个压力级别一轮
# ---------------------------------------------------------------------------
for speed in "${SPEEDS[@]}"; do
  ROUND_ID="stability_${speed}x_$(date +%Y%m%d_%H%M%S)"
  ROUND_DIR="${RUN_ROOT}/${ROUND_ID}"
  mkdir -p "${ROUND_DIR}"
  ROUND_DIRS+=("${ROUND_DIR}")

  echo ">>> 第 ${speed}x 轮开始 (输出: ${ROUND_DIR})"

  # 1. 后台启动监控采集器
  python3 "${SCRIPT_DIR}/stability_monitor.py" \
    --flexlb-http-addr "${FLEXLB_HTTP_ADDR}" \
    --management-port "${FLEXLB_MANAGEMENT_PORT}" \
    --mock-http-port "${MOCK_HTTP_PORT}" \
    --interval "${MONITOR_INTERVAL}" \
    --output "${ROUND_DIR}/monitor.jsonl" &
  MONITOR_PID=$!
  echo "    监控进程 PID=${MONITOR_PID}, 输出: ${ROUND_DIR}/monitor.jsonl"

  # 2. 通过 JAVA_TOOL_OPTIONS 注入 GC 日志 + HeapDump (零文件修改)
  #    JVM 自动拾取此环境变量
  export JAVA_TOOL_OPTIONS="-Xlog:gc*:${ROUND_DIR}/gc.log:time,uptime,level,tags:filecount=1,filesize=50m -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=${ROUND_DIR}/"

  # 3. 调用 run_online_eval.sh (环境变量驱动，零修改)
  echo "    启动 run_online_eval.sh (REPLAY_SPEED=${speed})..."
  set +e
  env \
    REPLAY_SPEED="${speed}" \
    RUN_ROOT="${RUN_ROOT}" \
    RUN_ID="${ROUND_ID}" \
    N_PREFILL="${N_PREFILL}" \
    N_DECODE="${N_DECODE}" \
    SCHEDULE_MODE="${SCHEDULE_MODE}" \
    LOAD_BALANCE_STRATEGY="${LOAD_BALANCE_STRATEGY}" \
    DECODE_LOAD_BALANCE_STRATEGY="${DECODE_LOAD_BALANCE_STRATEGY}" \
    MAX_CONCURRENCY="${MAX_CONCURRENCY}" \
    SLA_TTFT_MS="${SLA_TTFT_MS}" \
    LIMIT="${LIMIT}" \
    ZERO_OUTPUT_POLICY="${ZERO_OUTPUT_POLICY}" \
    JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS}" \
    bash "${SCRIPT_DIR}/run_online_eval.sh" 2>&1 | tee "${ROUND_DIR}/eval.stdout"
  EVAL_EXIT=$?
  set -e

  if [[ ${EVAL_EXIT} -ne 0 ]]; then
    echo "    WARNING: run_online_eval.sh 退出码 ${EVAL_EXIT} (可能 Master 崩溃)"
  fi

  # 4. 停止监控采集器
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true

  # 清除 JAVA_TOOL_OPTIONS 避免影响后续操作
  unset JAVA_TOOL_OPTIONS

  echo "<<< 第 ${speed}x 轮完成"
  echo ""

  # 5. 轮间 sleep 确保端口释放
  if [[ "${speed}" != "${SPEEDS[-1]}" ]]; then
    echo "    等待 3s 确保端口释放..."
    sleep 3
  fi
done

# ---------------------------------------------------------------------------
# 生成对比报告
# ---------------------------------------------------------------------------
echo "============================================"
echo "生成稳定性测试对比报告..."
echo "============================================"

REPORT_PATH="${STABILITY_DIR}/stability_report.md"

python3 "${SCRIPT_DIR}/generate_stability_report.py" \
  --run-dirs "${ROUND_DIRS[@]}" \
  --sla-ttft-ms "${SLA_TTFT_MS}" \
  --output "${REPORT_PATH}"

echo ""
echo "============================================"
echo "稳定性测试完成"
echo "============================================"
echo "报告: ${REPORT_PATH}"
echo ""
echo "各轮输出:"
for dir in "${ROUND_DIRS[@]}"; do
  echo "  - ${dir}/"
  echo "      summary: ${dir}/load_client/summary.json"
  echo "      monitor: ${dir}/monitor.jsonl"
  echo "      gc log:  ${dir}/gc.log"
  echo "      flexlb:  ${dir}/flexlb.log"
done
echo ""
echo "查看报告: cat ${REPORT_PATH}"
