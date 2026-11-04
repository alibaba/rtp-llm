#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_env.sh"

REQUEST_COUNT=${1:-400}
NODE_COUNT=${2:-4000}
LOOP_COUNT=${3:-2}
PARALLELISM=${4:-10}
REPEAT_COUNT=${5:-300}

if (($# > 5)); then
    echo "usage: $0 [REQUESTS NODES LOOPS PARALLELISM REPEATS]" >&2
    exit 2
fi
for value in "${REQUEST_COUNT}" "${NODE_COUNT}" "${LOOP_COUNT}" \
    "${PARALLELISM}" "${REPEAT_COUNT}"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "benchmark arguments must be positive integers" >&2
        exit 2
    fi
done

cd "${REPO_ROOT}"
cat <<EOF
CostScheduler benchmark
  compiler : clang opt
  python   : ${PYTHON_BIN} (${PYTHON_VERSION})
  matrix   : ${REQUEST_COUNT} requests x ${NODE_COUNT} nodes
  optimizer: ${LOOP_COUNT} loops, ${PARALLELISM} workers
  repeats  : ${REPEAT_COUNT}
EOF

"${BAZEL_BIN}" build "${NAVI_SCHED_BAZEL_OPT_ARGS[@]}" \
    //rtp_llm/flexlb/navi_sched:cost_scheduler_graph_benchmark \
    //rtp_llm/flexlb/navi_sched:cost_scheduler_master_graph_vis
OUTPUT_BIN=$(navi_sched_bazel_output_bin \
    "${NAVI_SCHED_BAZEL_OPT_ARGS[@]}")
BENCHMARK_BIN=${OUTPUT_BIN}/aios/navi_sched/cost_scheduler_graph_benchmark
MASTER_VIS_BIN=${OUTPUT_BIN}/aios/navi_sched/cost_scheduler_master_graph_vis
if [[ ! -x "${BENCHMARK_BIN}" || ! -x "${MASTER_VIS_BIN}" ]]; then
    echo "missing benchmark binary: ${BENCHMARK_BIN}" >&2
    exit 1
fi

echo "  binary   : ${BENCHMARK_BIN}"
RUNTIME_DIR=${NAVI_SCHED_DIR}/tmp/cost_scheduler_benchmark
mkdir -p "${RUNTIME_DIR}"
if [[ ! -e "${RUNTIME_DIR}/aios" ]]; then
    ln -s "${REPO_ROOT}/aios" "${RUNTIME_DIR}/aios"
fi
cd "${RUNTIME_DIR}"
"${BENCHMARK_BIN}" "${REQUEST_COUNT}" "${NODE_COUNT}" \
    "${LOOP_COUNT}" "${PARALLELISM}" "${REPEAT_COUNT}"

echo "Benchmarking the steady-state MasterScheduleGraph without vis"
"${MASTER_VIS_BIN}" "${REQUEST_COUNT}" "${NODE_COUNT}" \
    "${LOOP_COUNT}" "${PARALLELISM}" no_vis "${REPEAT_COUNT}"
NO_VIS_STEM=master_schedule_no_vis_${REQUEST_COUNT}r_${NODE_COUNT}n_${LOOP_COUNT}l_${PARALLELISM}w_samples.tsv
mv master_schedule_no_vis_samples.tsv "${NO_VIS_STEM}"
echo "  saved samples: ${RUNTIME_DIR}/${NO_VIS_STEM}"

echo "Collecting one full MasterScheduleGraph vis run"
"${MASTER_VIS_BIN}" "${REQUEST_COUNT}" "${NODE_COUNT}" \
    "${LOOP_COUNT}" "${PARALLELISM}" vis
VIS_STEM=master_schedule_${REQUEST_COUNT}r_${NODE_COUNT}n_${LOOP_COUNT}l_${PARALLELISM}w
cp master_schedule_graph.vis "${VIS_STEM}.vis"
cp master_schedule_graph_vis_summary.tsv "${VIS_STEM}_summary.tsv"
echo "  saved vis: ${RUNTIME_DIR}/${VIS_STEM}.vis"
echo "  saved summary: ${RUNTIME_DIR}/${VIS_STEM}_summary.tsv"
