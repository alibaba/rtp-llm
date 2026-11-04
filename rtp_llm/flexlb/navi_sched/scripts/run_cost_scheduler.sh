#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_env.sh"
INPUT_REPLAY=${NAVI_SCHED_DIR}/test/data/cost_scheduler/max_cache_hit_10_agents_midpoint.json
OUTPUT_DIR=${NAVI_SCHED_DIR}/tmp/cost_scheduler_debug
REPLAY_OUTPUT_DIR=${OUTPUT_DIR}/replays
OUTPUT=${OUTPUT_DIR}/cost_scheduler_output.txt
RUN_TESTS=false
COST_LAMBDA=${NAVI_SCHED_COST_LAMBDA:-0.5}
COST_ALPHA=${NAVI_SCHED_COST_ALPHA:-512.0}
COST_ALPHA_DECAY=${NAVI_SCHED_COST_ALPHA_DECAY:-1.0}
COST_MIN_ALPHA=${NAVI_SCHED_COST_MIN_ALPHA:-0.0}
COST_MAX_LOOP_COUNT=${NAVI_SCHED_COST_MAX_LOOP_COUNT:-2}

usage() {
    cat <<'EOF'
用法: run_cost_scheduler.sh [--test]

编译 CostScheduler，并加载一份固定的中段调度现场。

  --test  运行 replay 前先执行 NaviSched data 和 role 单元测试。
EOF
}

while (($# > 0)); do
    case "$1" in
        --test)
            RUN_TESTS=true
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ ! -f "${INPUT_REPLAY}" ]]; then
    echo "missing CostScheduler replay: ${INPUT_REPLAY}" >&2
    exit 1
fi

cd "${REPO_ROOT}"

REQUEST_COUNT=$(grep -c '"request": {' "${INPUT_REPLAY}")
AGENT_COUNT=$(grep -c '"agent_id":' "${INPUT_REPLAY}")

cat <<EOF
============================================================
CostScheduler 开发回放

目标函数    : ${NAVI_SCHED_DIR}/scheduler/CostSchedulerKernelInternal.cpp
并行图      : ${NAVI_SCHED_DIR}/scheduler/CostSchedulerGraph.cpp
实现入口    : CostSchedulerKernelInternal::forwardNodeRange()
Cost Kernel : navi_sched.cost_scheduler_k
参数        : lambda=${COST_LAMBDA}, alpha=${COST_ALPHA}, \
alpha_decay=${COST_ALPHA_DECAY}, min_alpha=${COST_MIN_ALPHA}, \
max_loop_count=${COST_MAX_LOOP_COUNT}
输入 Replay       : ${INPUT_REPLAY}
  请求数量  : ${REQUEST_COUNT}
  Agent 数量: ${AGENT_COUNT}
输出目录          : ${OUTPUT_DIR}

执行过程:
  1. 使用 Clang opt 和 Python 3.10 编译调度 replay runner。
  2. 加载固定现场并执行一次 CostScheduler。
  3. 输出 context、cache 命中和 CostScheduler SchedInfo 表格。
  4. 保存包含 CostScheduler SchedInfo 的结果 replay。
============================================================
EOF

if [[ "${RUN_TESTS}" == true ]]; then
    echo "==> 执行 NaviSched 单元测试"
    "${BAZEL_BIN}" test "${NAVI_SCHED_BAZEL_OPT_ARGS[@]}" \
        --test_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
        //rtp_llm/flexlb/navi_sched:navi_sched_data_test \
        //rtp_llm/flexlb/navi_sched:navi_sched_role_test
fi

echo "==> 编译 schedule replay runner"
"${BAZEL_BIN}" build "${NAVI_SCHED_BAZEL_OPT_ARGS[@]}" \
    //rtp_llm/flexlb/navi_sched:schedule_replay_runner
OUTPUT_BIN=$(navi_sched_bazel_output_bin \
    "${NAVI_SCHED_BAZEL_OPT_ARGS[@]}")
REPLAY_RUNNER=${OUTPUT_BIN}/aios/navi_sched/schedule_replay_runner
if [[ ! -x "${REPLAY_RUNNER}" ]]; then
    echo "missing replay runner: ${REPLAY_RUNNER}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${REPLAY_OUTPUT_DIR}"
RESULT=${REPLAY_OUTPUT_DIR}/01_cost_optimization.json

echo
echo "==> CostScheduler 固定现场结果"
"${REPLAY_RUNNER}" \
    navi_sched.cost_scheduler_k "${INPUT_REPLAY}" \
    --cost-lambda "${COST_LAMBDA}" \
    --cost-alpha "${COST_ALPHA}" \
    --cost-alpha-decay "${COST_ALPHA_DECAY}" \
    --cost-min-alpha "${COST_MIN_ALPHA}" \
    --cost-max-loop-count "${COST_MAX_LOOP_COUNT}" \
    --context-summary \
    --replay-output "${RESULT}" |
    tee "${OUTPUT}"

cat <<'EOF'

结果字段:
  scheduler_info                    本轮 CostScheduler 的调度输入表格
    sched_infos[node][request]      对应位置的 cost、queue_wait_ms 和 weight
    node/request 标识               从同一份 replay context 中关联输出
  scheduled                         成功分配到 Agent 的请求数
  rejected                          被策略拒绝的请求数
  invalid_targets                   选择了候选集合外 Agent 的请求数
  avg_cache_hit_tokens              每个请求平均复用的输入 token 数
  avg_estimated_queue_ms            目标 Agent 的平均预计排队时间
  avg_estimated_batch_execution_ms  每个已分配 batch 的平均预计执行时间
EOF

echo
echo "完整输出:"
echo "  ${OUTPUT}"
echo "结果 replay:"
echo "  ${RESULT}"
echo "Web 查看（独立后台启动，不占用当前 shell）:"
echo "  ${NAVI_SCHED_DIR}/scripts/run_replay_viewer.sh"
