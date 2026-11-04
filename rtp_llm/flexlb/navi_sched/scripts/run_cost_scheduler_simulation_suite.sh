#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common_env.sh"

SIMULATOR=${NAVI_SCHED_DIR}/test/simulator/run_traffic_replay.sh
SUITE_DIR=${NAVI_SCHED_DIR}/tmp/cost_scheduler_suite
REPORT=${NAVI_SCHED_DIR}/tmp/cost_scheduler_debug/comparison_report.json
PACKAGED_REPORT=${NAVI_SCHED_DIR}/test/data/cost_scheduler/comparison_report.json
CLIENT_COUNT=10
TIME_COMPRESSION=1
COMPLETION_TIMEOUT_SECONDS=7200
REQUEST_TIMEOUT_MS=$((COMPLETION_TIMEOUT_SECONDS * 1000))
COST_MAX_LOOP_COUNT=50
WINDOW_MAX_LOOP_COUNT=200
COST_PARALLELISM=10
TOKEN_NORMALIZATION_POWER=0
SINGLE_REQUEST_COUNT=53751
MULTI_REQUEST_COUNT=90291
RESUME=false
FAST_FIRST=false
declare -A REPORT_CANDIDATES=()

usage() {
    cat <<'EOF'
usage: run_cost_scheduler_simulation_suite.sh [--resume] [--fast-first]

Runs the canonical CostScheduler simulation matrix and refreshes the Web report.
With --resume, a case is reused only after its summary passes all validations.
With --fast-first, lambda=1 boundary cases are deferred to a later full resume.
EOF
}

while (($# > 0)); do
    case "$1" in
        --resume)
            RESUME=true
            ;;
        --fast-first)
            FAST_FIRST=true
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

declare -a SOURCE_NAMES=(4091 69eb f233 aae5 ddec)
declare -a SERVICE_NAMES=(
    deepseek-v4-pro-model-4091
    deepseek-v4-pro-model-69eb
    deepseek-v4-pro-model-f233
    deepseek-v4-pro-model-aae5
    deepseek-v4-pro-model-ddec
)
declare -a MODERATE_AGENT_COUNTS=(33 41 9 17 54)

mkdir -p "${SUITE_DIR}" "$(dirname "${REPORT}")"

for name in "${SOURCE_NAMES[@]}"; do
    path=${NAVI_SCHED_DIR}/tmp/joint_${name}.middle5m.jsonl
    if [[ ! -s "${path}" ]]; then
        echo "missing multi-biz traffic input: ${path}" >&2
        exit 1
    fi
done

make_cases_file() {
    local output=$1
    local mode=$2
    local -n agent_counts=$3
    "${PYTHON_BIN}" - "${output}" "${mode}" "${NAVI_SCHED_DIR}" \
        "${SOURCE_NAMES[*]}" "${SERVICE_NAMES[*]}" \
        "${agent_counts[*]}" <<'PY'
import json
import sys

output, mode, root, names, services, counts = sys.argv[1:]
names = names.split()
services = services.split()
counts = [int(value) for value in counts.split()]
sources = [
    {
        "name": f"agent_biz_{name}",
        "traffic_log": f"{root}/tmp/joint_{name}.middle5m.jsonl",
        "service_name": service,
    }
    for name, service in zip(names, services)
]
if mode == "mixed":
    value = {
        "mode": "mixed",
        "biz_name": "agent_biz_mixed",
        "agent_count": counts[0],
        "client_count": 10,
        "sources": sources,
    }
else:
    value = [
        {
            **source,
            "biz_name": source["name"],
            "agent_count": count,
            "client_count": 2,
        }
        for source, count in zip(sources, counts)
    ]
with open(output, "w", encoding="utf-8") as target:
    json.dump(value, target, indent=2)
    target.write("\n")
PY
}

MODERATE_CASES=${SUITE_DIR}/independent_154.json
SHARED_154_CASES=${SUITE_DIR}/shared_154.json
SHARED_145_CASES=${SUITE_DIR}/shared_145.json
declare -a SHARED_154_COUNT=(154)
declare -a SHARED_145_COUNT=(145)
declare -a SINGLE_LAMBDAS=(
    0 0.25 0.5 0.75 0.8 0.85 0.9 0.95 0.975 0.99 0.995 0.999
    0.9995 0.9997 0.9998 0.9999 0.99995 1
)
declare -a BATCH_WINDOWS=(30 50 75 100)
declare -a MULTI_LAMBDAS=(
    0 0.1 0.25 0.5 0.75 0.9 0.95 0.99 0.999 1
)
declare -a LOOP_CAPS=(2 5 10 20)
SIMULATED_FORMULA_PARAMETERS='[-964.0,10.0,1.4,20.0,0.1,0.09,0.0,1.4,1.0,0.0,0.0]'
SIMULATED_LATENCY_STATE=${SUITE_DIR}/simulated_formula.state
SIMULATED_LATENCY_STATE_AUDIT=${SIMULATED_LATENCY_STATE}.json
if [[ "${FAST_FIRST}" == true ]]; then
    SINGLE_LAMBDAS=("${SINGLE_LAMBDAS[@]:0:${#SINGLE_LAMBDAS[@]}-1}")
    MULTI_LAMBDAS=("${MULTI_LAMBDAS[@]:0:${#MULTI_LAMBDAS[@]}-1}")
fi
make_cases_file "${MODERATE_CASES}" independent MODERATE_AGENT_COUNTS
make_cases_file "${SHARED_154_CASES}" mixed SHARED_154_COUNT
make_cases_file "${SHARED_145_CASES}" mixed SHARED_145_COUNT
"${PYTHON_BIN}" "${NAVI_SCHED_DIR}/test/simulator/latency_model_state.py" \
    --output "${SIMULATED_LATENCY_STATE}" \
    --parameters-json "${SIMULATED_FORMULA_PARAMETERS}" \
    >"${SIMULATED_LATENCY_STATE_AUDIT}"

SUITE_FINGERPRINT=$(
    {
        find \
            "${NAVI_SCHED_DIR}/common" \
            "${NAVI_SCHED_DIR}/config" \
            "${NAVI_SCHED_DIR}/data" \
            "${NAVI_SCHED_DIR}/debug" \
            "${NAVI_SCHED_DIR}/navi_sched" \
            "${NAVI_SCHED_DIR}/resource" \
            "${NAVI_SCHED_DIR}/scheduler" \
            "${NAVI_SCHED_DIR}/service" \
            "${NAVI_SCHED_DIR}/test/simulator" \
            -type f \( -name '*.h' -o -name '*.cpp' -o \
            -name '*.py' -o -name '*.sh' \) -print0
        printf '%s\0' "${NAVI_SCHED_DIR}/BUILD"
        find "${NAVI_SCHED_DIR}/scripts" -maxdepth 1 \
            -type f -name '*.sh' -print0
    } |
        sort -z |
        xargs -0 sha256sum |
        sha256sum |
        awk '{print $1}'
)

append_report_run() {
    local group=$1
    local label=$2
    local scheduler=$3
    local lambda_value=$4
    local loop_count=$5
    local summary=$6
    if [[ "${REPORT_CANDIDATES[${summary}]:-}" == 1 &&
        -f "${summary}" ]]; then
        report_args+=(
            --grouped-run "${group}" "${label}" "${scheduler}"
            "${lambda_value}" "${loop_count}" "${summary}"
        )
    fi
}

refresh_report() {
    local temporary_report=${REPORT}.tmp
    local report_lambda=
    local report_suffix=
    local report_window=
    local report_loop_cap=
    local -a report_args=(
        --output "${temporary_report}"
        --title "NaviSched scheduler experiments"
        --client-count "${CLIENT_COUNT}"
        --time-compression "${TIME_COMPRESSION}"
    )
    if [[ -s "${REPORT}" ]]; then
        report_args+=(--base-report "${REPORT}")
    fi
    append_report_run "Strategy baseline" random random - - \
        "${SUITE_DIR}/single_random/traffic_replay_summary.json"
    append_report_run "Strategy baseline" max_cache_hit max_cache_hit - - \
        "${SUITE_DIR}/single_max_cache_hit/traffic_replay_summary.json"
    append_report_run "Strategy baseline" \
        "FlexLB selector proxy (Navi collection window)" flexlb_baseline - - \
        "${SUITE_DIR}/single_flexlb_baseline/traffic_replay_summary.json"
    append_report_run "Strategy baseline" cost cost 0.5 \
        "${COST_MAX_LOOP_COUNT}" \
        "${SUITE_DIR}/single_cost_0_5/traffic_replay_summary.json"
    for report_lambda in "${SINGLE_LAMBDAS[@]}"; do
        report_suffix=${report_lambda/./_}
        append_report_run "Single-biz lambda sweep" "l${report_suffix}" cost \
            "${report_lambda}" "${COST_MAX_LOOP_COUNT}" \
            "${SUITE_DIR}/single_cost_${report_suffix}/\
traffic_replay_summary.json"
    done
    for report_window in "${BATCH_WINDOWS[@]}"; do
        append_report_run "Batch window sweep" \
            "${report_window} requests / ${report_window} ms" cost 0 \
            "${WINDOW_MAX_LOOP_COUNT}" \
            "${SUITE_DIR}/single_cost_window_${report_window}/\
traffic_replay_summary.json"
    done
    for report_lambda in "${MULTI_LAMBDAS[@]}"; do
        report_suffix=${report_lambda/./_}
        append_report_run "Independent multi-biz lambda sweep" \
            "l${report_suffix}" cost "${report_lambda}" \
            "${COST_MAX_LOOP_COUNT}" \
            "${SUITE_DIR}/independent_154_cost_${report_suffix}/\
traffic_replay_summary.json"
        append_report_run "Unified mixed-pool lambda sweep" \
            "l${report_suffix}" cost "${report_lambda}" \
            "${COST_MAX_LOOP_COUNT}" \
            "${SUITE_DIR}/shared_154_cost_${report_suffix}/\
traffic_replay_summary.json"
    done
    local strategy_group="Independent and unified strategy baseline"
    append_report_run "${strategy_group}" "independent random" random - - \
        "${SUITE_DIR}/independent_154_random/traffic_replay_summary.json"
    append_report_run "${strategy_group}" "independent max cache hit" \
        max_cache_hit - - \
        "${SUITE_DIR}/independent_154_max_cache_hit/\
traffic_replay_summary.json"
    append_report_run "${strategy_group}" "unified random" random - - \
        "${SUITE_DIR}/shared_154_random/traffic_replay_summary.json"
    append_report_run "${strategy_group}" "unified max cache hit" \
        max_cache_hit - - \
        "${SUITE_DIR}/shared_154_max_cache_hit/traffic_replay_summary.json"
    append_report_run "${strategy_group}" \
        "independent FlexLB selector proxy" flexlb_baseline - - \
        "${SUITE_DIR}/independent_154_flexlb_baseline/\
traffic_replay_summary.json"
    append_report_run "${strategy_group}" \
        "unified FlexLB selector proxy" flexlb_baseline - - \
        "${SUITE_DIR}/shared_154_flexlb_baseline/\
traffic_replay_summary.json"
    append_report_run "${strategy_group}" "independent cost" cost 0 \
        "${COST_MAX_LOOP_COUNT}" \
        "${SUITE_DIR}/independent_154_cost_0/traffic_replay_summary.json"
    append_report_run "${strategy_group}" "unified cost" cost 0 \
        "${COST_MAX_LOOP_COUNT}" \
        "${SUITE_DIR}/shared_154_cost_0/traffic_replay_summary.json"
    local pool_group="Unified pool capacity validation"
    append_report_run "${pool_group}" \
        "shared 154 agents" cost 0 "${COST_MAX_LOOP_COUNT}" \
        "${SUITE_DIR}/shared_154_cost_0/traffic_replay_summary.json"
    append_report_run "${pool_group}" \
        "shared 145 agents" cost 0 "${COST_MAX_LOOP_COUNT}" \
        "${SUITE_DIR}/shared_145_cost_0/traffic_replay_summary.json"
    append_report_run "Small-lambda loop cap sweep" "cap 50" cost 0 \
        "${COST_MAX_LOOP_COUNT}" \
        "${SUITE_DIR}/shared_154_cost_0/traffic_replay_summary.json"
    for report_loop_cap in "${LOOP_CAPS[@]}"; do
        append_report_run "Small-lambda loop cap sweep" \
            "cap ${report_loop_cap}" cost 0 "${report_loop_cap}" \
            "${SUITE_DIR}/shared_154_loop_${report_loop_cap}/\
traffic_replay_summary.json"
    done
    if ! "${PYTHON_BIN}" \
        "${NAVI_SCHED_DIR}/tools/replay_viewer/\
make_comparison_report.py" \
        "${report_args[@]}"; then
        return 1
    fi
    mv "${temporary_report}" "${REPORT}"
}

validate_summary() {
    local summary=$1
    local scheduler=$2
    local expected_requests=$3
    local expected_agents=$4
    local expected_lambda=$5
    local expected_loops=$6
    local expected_wait_count=$7
    local expected_wait_time_ms=$8
    local fingerprint_file=${summary}.suite_fingerprint
    if [[ ! -s "${fingerprint_file}" ||
        "$(<"${fingerprint_file}")" != "${SUITE_FINGERPRINT}" ]]; then
        return 1
    fi
    "${PYTHON_BIN}" - "${summary}" "${scheduler}" \
        "${expected_requests}" "${expected_agents}" \
        "${expected_lambda}" "${expected_loops}" \
        "${expected_wait_count}" "${expected_wait_time_ms}" \
        "${TIME_COMPRESSION}" "${SIMULATED_FORMULA_PARAMETERS}" \
        "${SIMULATED_LATENCY_STATE_AUDIT}" <<'PY'
import json
import math
import sys

(path, scheduler, request_count, agent_count,
 lambda_value, loop_count, wait_count, wait_time_ms,
 time_compression, formula_parameters, state_audit_path) = sys.argv[1:]
try:
    with open(path, encoding="utf-8") as source:
        summary = json.load(source)
    with open(state_audit_path, encoding="utf-8") as source:
        state_audit = json.load(source)
    assert summary["scheduler_name"] == scheduler
    assert summary["agent_readiness"] == "all_agents_ready"
    assert summary["request_count"] == int(request_count)
    assert len(summary["execution_agents"]) == int(agent_count)
    assert summary["schedule_wait_count"] == int(wait_count)
    assert summary["schedule_wait_time_ms"] == int(wait_time_ms)
    assert math.isclose(
        float(summary["replay_execution_time_compression"]),
        float(time_compression))
    assert summary["latency_time_basis"] == "uncompressed_ms"
    launch = summary["experiment_launch_config"]
    assert launch["formula_parameters"] == json.loads(formula_parameters)
    assert launch["latency_model_state_file"]
    assert launch["latency_model_state_sha256"] == state_audit["sha256"]
    prediction = summary["decision_prediction_accuracy"]["execution"]
    assert prediction["mape_pct"] < 20.0
    loss = summary["cost_scheduler_loss"]["distribution"]
    assert loss["count"] > 0
    metrics = summary["scheduler_metrics"]
    assert metrics["schedule_count"] > 0
    assert metrics["average_requests_per_schedule"] > 0
    assert metrics["latency_us_distribution"]["count"] > 0
    if scheduler == "cost":
        config = summary["scheduler_config"]
        assert math.isclose(float(config["lambda"]), float(lambda_value))
        assert int(config["max_loop_count"]) == int(loop_count)
        assert math.isclose(
            float(config["latency_token_normalization_power"]), 0.0)
        effective = summary["effective_scheduler_config"]
        assert math.isclose(
            float(effective["latency_token_normalization_power"]), 0.0)
        expected_budget_us = math.ceil(
            int(wait_time_ms) / float(time_compression)) * 1000
        assert int(config["optimization_time_budget_us"]) == expected_budget_us
except (AssertionError, KeyError, OSError, TypeError, ValueError):
    raise SystemExit(1)
PY
}

execute_case() {
    local name=$1
    local scheduler=$2
    local request_count=$3
    local agent_count=$4
    local lambda_value=$5
    local alpha=$6
    local loop_count=$7
    local cases_file=${8:-}
    local wait_count=${9:-30}
    local wait_time_ms=${10:-30}
    local output=${SUITE_DIR}/${name}
    echo "==> run ${name}"
    local scheduler_config=
    if [[ "${scheduler}" == cost ]]; then
        scheduler_config=$(printf \
            '{"lambda":%s,"alpha":%s,"max_loop_count":%s,"parallelism":%s,"latency_token_normalization_power":%s}' \
            "${lambda_value}" "${alpha}" "${loop_count}" \
            "${COST_PARALLELISM}" "${TOKEN_NORMALIZATION_POWER}")
    fi
    NAVI_SCHED_TRAFFIC_SCHEDULER=${scheduler} \
        NAVI_SCHED_TRAFFIC_SCHEDULER_CONFIG=${scheduler_config} \
        NAVI_SCHED_TRAFFIC_AGENT_COUNT=${agent_count} \
        NAVI_SCHED_TRAFFIC_CLIENT_COUNT=${CLIENT_COUNT} \
        NAVI_SCHED_TRAFFIC_CASES_FILE=${cases_file} \
        NAVI_SCHED_TRAFFIC_TIME_COMPRESSION=${TIME_COMPRESSION} \
        NAVI_SCHED_TRAFFIC_CACHE_CAPACITY_BYTES=$((300 * 1024 * 1024 * 1024)) \
        NAVI_SCHED_TRAFFIC_NOISE_STDDEV_RATIO=0.01 \
        NAVI_SCHED_TRAFFIC_RANDOM_SEED=1 \
        NAVI_SCHED_TRAFFIC_FORMULA_PARAMETERS_JSON=${SIMULATED_FORMULA_PARAMETERS} \
        NAVI_SCHED_TRAFFIC_LATENCY_MODEL_STATE_FILE=${SIMULATED_LATENCY_STATE} \
        NAVI_SCHED_TRAFFIC_TIMEOUT_MS=${REQUEST_TIMEOUT_MS} \
        NAVI_SCHED_TRAFFIC_SCHEDULE_WAIT_COUNT=${wait_count} \
        NAVI_SCHED_TRAFFIC_SCHEDULE_WAIT_TIME_MS=${wait_time_ms} \
        NAVI_SCHED_TRAFFIC_COMPLETION_TIMEOUT_SECONDS=${COMPLETION_TIMEOUT_SECONDS} \
        NAVI_SCHED_TRAFFIC_TEST_DIR=${output} \
        "${SIMULATOR}" >"${output}.log" 2>&1
    printf '%s\n' "${SUITE_FINGERPRINT}" > \
        "${output}/traffic_replay_summary.json.suite_fingerprint"
    validate_summary "${output}/traffic_replay_summary.json" \
        "${scheduler}" "${request_count}" \
        "${agent_count}" "${lambda_value}" "${loop_count}" \
        "${wait_count}" "${wait_time_ms}"
}

run_case() {
    local name=$1
    local scheduler=$2
    local request_count=$3
    local agent_count=$4
    local lambda_value=$5
    local alpha=$6
    local loop_count=$7
    local cases_file=${8:-}
    local wait_count=${9:-30}
    local wait_time_ms=${10:-30}
    local output=${SUITE_DIR}/${name}
    local summary=${output}/traffic_replay_summary.json

    if [[ "${RESUME}" == true ]] && validate_summary \
        "${summary}" "${scheduler}" "${request_count}" \
        "${agent_count}" "${lambda_value}" "${loop_count}" \
        "${wait_count}" "${wait_time_ms}"; then
        echo "==> reuse ${name}"
    else
        execute_case "$@"
    fi
    REPORT_CANDIDATES["${summary}"]=1
    refresh_report
}

echo "==> build simulate package with Clang opt and Python ${PYTHON_VERSION}"
"${SCRIPT_DIR}/build_simulate_package.sh"

for window in "${BATCH_WINDOWS[@]}"; do
    run_case "single_cost_window_${window}" cost \
        "${SINGLE_REQUEST_COUNT}" 40 0 512 \
        "${WINDOW_MAX_LOOP_COUNT}" "" "${window}" "${window}"
done

run_case single_random random "${SINGLE_REQUEST_COUNT}" 40 - - -
run_case single_max_cache_hit max_cache_hit \
    "${SINGLE_REQUEST_COUNT}" 40 - - -
run_case single_flexlb_baseline flexlb_baseline \
    "${SINGLE_REQUEST_COUNT}" 40 - - -

for lambda_value in "${MULTI_LAMBDAS[@]}"; do
    suffix=${lambda_value/./_}
    run_case "independent_154_cost_${suffix}" cost \
        "${MULTI_REQUEST_COUNT}" 154 "${lambda_value}" 128 \
        "${COST_MAX_LOOP_COUNT}" \
        "${MODERATE_CASES}"
    run_case "shared_154_cost_${suffix}" cost \
        "${MULTI_REQUEST_COUNT}" 154 "${lambda_value}" 128 \
        "${COST_MAX_LOOP_COUNT}" \
        "${SHARED_154_CASES}"
done

run_case independent_154_random random "${MULTI_REQUEST_COUNT}" 154 - - - \
    "${MODERATE_CASES}"
run_case independent_154_max_cache_hit max_cache_hit \
    "${MULTI_REQUEST_COUNT}" 154 - - - "${MODERATE_CASES}"
run_case shared_154_random random "${MULTI_REQUEST_COUNT}" 154 - - - \
    "${SHARED_154_CASES}"
run_case shared_154_max_cache_hit max_cache_hit \
    "${MULTI_REQUEST_COUNT}" 154 - - - "${SHARED_154_CASES}"
run_case independent_154_flexlb_baseline flexlb_baseline \
    "${MULTI_REQUEST_COUNT}" 154 - - - "${MODERATE_CASES}"
run_case shared_154_flexlb_baseline flexlb_baseline \
    "${MULTI_REQUEST_COUNT}" 154 - - - "${SHARED_154_CASES}"
run_case shared_145_cost_0 cost "${MULTI_REQUEST_COUNT}" 145 0 128 \
    "${COST_MAX_LOOP_COUNT}" \
    "${SHARED_145_CASES}"
for loop_cap in "${LOOP_CAPS[@]}"; do
    run_case "shared_154_loop_${loop_cap}" cost \
        "${MULTI_REQUEST_COUNT}" 154 0 128 "${loop_cap}" \
        "${SHARED_154_CASES}"
done

for lambda_value in "${SINGLE_LAMBDAS[@]}"; do
    suffix=${lambda_value/./_}
    run_case "single_cost_${suffix}" cost \
        "${SINGLE_REQUEST_COUNT}" 40 "${lambda_value}" 512 \
        "${COST_MAX_LOOP_COUNT}"
done

refresh_report
cp "${REPORT}" "${PACKAGED_REPORT}"

echo "==> refreshed ${REPORT}"
echo "==> refreshed ${PACKAGED_REPORT}"
