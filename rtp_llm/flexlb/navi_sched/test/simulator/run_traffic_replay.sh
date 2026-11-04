#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAVI_SCHED_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${NAVI_SCHED_PYTHON_BIN:-/opt/conda310/bin/python3}
DEFAULT_ARCHIVE=${NAVI_SCHED_DIR}/release/navi_sched_simulate-0.1.0.tar.gz
ARCHIVE=${NAVI_SCHED_SIMULATE_ARCHIVE:-${DEFAULT_ARCHIVE}}
DEFAULT_TRAFFIC_LOG=${SCRIPT_DIR}/data/deepseek-pro-v4.2026-08-04.h09.deepseek-v4-pro-model-4091.5m.jsonl
TRAFFIC_LOG=${NAVI_SCHED_TRAFFIC_LOG:-${DEFAULT_TRAFFIC_LOG}}
SERVICE_NAME=${NAVI_SCHED_TRAFFIC_SERVICE_NAME:-deepseek-v4-pro-model-4091}
TRAFFIC_CASES_FILE=${NAVI_SCHED_TRAFFIC_CASES_FILE:-}
SCHEDULER_NAME=${NAVI_SCHED_TRAFFIC_SCHEDULER:-random}
SCHEDULER_CONFIG=${NAVI_SCHED_TRAFFIC_SCHEDULER_CONFIG:-}
AGENT_COUNT=${NAVI_SCHED_TRAFFIC_AGENT_COUNT:-20}
CLIENT_COUNT=${NAVI_SCHED_TRAFFIC_CLIENT_COUNT:-20}
THREAD_NUM=${NAVI_SCHED_TRAFFIC_THREAD_NUM:-64}
TIME_COMPRESSION=${NAVI_SCHED_TRAFFIC_TIME_COMPRESSION:-2}
BLOCK_SIZE=${NAVI_SCHED_TRAFFIC_BLOCK_SIZE:-256}
DEFAULT_CACHE_CAPACITY_BYTES=$((300 * 1024 * 1024 * 1024))
CLUSTER_CACHE_CAPACITY_BYTES=${NAVI_SCHED_TRAFFIC_CACHE_CAPACITY_BYTES:-\
    ${DEFAULT_CACHE_CAPACITY_BYTES}}
TOKEN_CACHE_BYTES=${NAVI_SCHED_TRAFFIC_TOKEN_CACHE_BYTES:-$((10 * 1024))}
NOISE_STDDEV_RATIO=${NAVI_SCHED_TRAFFIC_NOISE_STDDEV_RATIO:-0.01}
RANDOM_SEED=${NAVI_SCHED_TRAFFIC_RANDOM_SEED:-1}
REQUIRE_CACHE_EVICTION=${NAVI_SCHED_TRAFFIC_REQUIRE_CACHE_EVICTION:-1}
FORMULA_PARAMETERS_JSON=${NAVI_SCHED_TRAFFIC_FORMULA_PARAMETERS_JSON:-}
LATENCY_MODEL_STATE_FILE=${NAVI_SCHED_TRAFFIC_LATENCY_MODEL_STATE_FILE:-}
LATENCY_FIT_BATCH_SIZE=${NAVI_SCHED_TRAFFIC_LATENCY_FIT_BATCH_SIZE:-100000}
LATENCY_FIT_INTERVAL=${NAVI_SCHED_TRAFFIC_LATENCY_FIT_INTERVAL:-0}
LATENCY_LEARNING_RATE=${NAVI_SCHED_TRAFFIC_LATENCY_LEARNING_RATE:-0.011}
LATENCY_WARMUP_SAMPLE_COUNT=${NAVI_SCHED_TRAFFIC_LATENCY_WARMUP_SAMPLE_COUNT:-10}
REPLAY_SAMPLE_INTERVAL=${NAVI_SCHED_TRAFFIC_REPLAY_SAMPLE_INTERVAL:-0}
REPLAY_SAMPLE_SEQUENCES=${NAVI_SCHED_TRAFFIC_REPLAY_SAMPLE_SEQUENCES:-10000,30000}
SCHEDULE_WAIT_COUNT=${NAVI_SCHED_TRAFFIC_SCHEDULE_WAIT_COUNT:-30}
SCHEDULE_WAIT_TIME_MS=${NAVI_SCHED_TRAFFIC_SCHEDULE_WAIT_TIME_MS:-30}
REQUEST_TIMEOUT_MS=${NAVI_SCHED_TRAFFIC_TIMEOUT_MS:-1800000}
COMPLETION_TIMEOUT_SECONDS=${NAVI_SCHED_TRAFFIC_COMPLETION_TIMEOUT_SECONDS:-1800}
MIXED_TRAFFIC_LOG=$(mktemp \
    "${TMPDIR:-/tmp}/navi_sched_mixed_traffic.XXXXXX.jsonl")
trap 'rm -f "${MIXED_TRAFFIC_LOG}"' EXIT

if [[ ! -f "${ARCHIVE}" ]]; then
    echo "missing simulate archive: ${ARCHIVE}" >&2
    exit 1
fi
if ((THREAD_NUM <= 0 || CLUSTER_CACHE_CAPACITY_BYTES <= 0 || \
    TOKEN_CACHE_BYTES <= 0 || BLOCK_SIZE <= 0 || SCHEDULE_WAIT_COUNT <= 0 || \
    SCHEDULE_WAIT_TIME_MS < 0 || LATENCY_FIT_BATCH_SIZE <= 0 || \
    LATENCY_FIT_INTERVAL < 0 || LATENCY_WARMUP_SAMPLE_COUNT < 0)); then
    echo "thread, cache, and schedule wait values are invalid" >&2
    exit 1
fi
if ((REQUEST_TIMEOUT_MS <= 0)); then
    echo "request timeout must be positive" >&2
    exit 1
fi
if ((COMPLETION_TIMEOUT_SECONDS <= 0)); then
    echo "completion timeout must be positive" >&2
    exit 1
fi
if [[ "${REQUIRE_CACHE_EVICTION}" != 0 &&
    "${REQUIRE_CACHE_EVICTION}" != 1 ]]; then
    echo "cache eviction requirement must be 0 or 1" >&2
    exit 1
fi
TRAFFIC_CASES_JSON=$(
    "${PYTHON_BIN}" - \
        "${TRAFFIC_CASES_FILE}" "${TRAFFIC_LOG}" "${SERVICE_NAME}" \
        "${AGENT_COUNT}" "${CLIENT_COUNT}" "${MIXED_TRAFFIC_LOG}" <<'PY'
import json
import os
import sys

(
    cases_file,
    traffic_log,
    service_name,
    agent_count,
    client_count,
    mixed_traffic_log,
) = sys.argv[1:]


def record_service_name(record):
    value = record.get("service_name")
    if value is not None:
        return value
    values = record.get("service_names")
    return values[0] if isinstance(values, list) and values else None


def record_timestamp_ms(record):
    value = record.get(
        "timestamp_ms", record.get("timestamp", record.get("reqTime")))
    if value is None:
        raise RuntimeError("traffic record has no timestamp")
    if isinstance(value, float):
        value = round(value * 1000)
    return int(value)


def build_mixed_case(config):
    required = ("biz_name", "agent_count", "client_count", "sources")
    if any(name not in config for name in required):
        raise RuntimeError("mixed traffic must define " + ", ".join(required))
    sources = config["sources"]
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("mixed traffic sources must be a non-empty array")
    records = []
    source_metrics = []
    sequence = 0
    for source_index, source in enumerate(sources):
        if not isinstance(source, dict) or any(
                name not in source for name in ("traffic_log", "service_name")):
            raise RuntimeError(
                "each mixed source must define traffic_log and service_name")
        path = os.path.abspath(str(source["traffic_log"]))
        source_service = str(source["service_name"])
        if not os.path.isfile(path) or not source_service:
            raise RuntimeError(f"invalid mixed traffic source: {path}")
        source_count = 0
        with open(path, encoding="utf-8") as input_file:
            for line_index, line in enumerate(input_file):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record_service_name(record) != source_service:
                    continue
                record["timestamp_ms"] = record_timestamp_ms(record)
                record["service_name"] = "navi_sched_mixed_traffic"
                record["traffic_source"] = source.get("name", source_service)
                records.append((record["timestamp_ms"], sequence,
                                source_index, line_index, record))
                sequence += 1
                source_count += 1
        if not source_count:
            raise RuntimeError(f"no traffic records found for {source_service}")
        source_metrics.append({
            "name": str(source.get("name", source_service)),
            "service_name": source_service,
            "request_count": source_count,
        })
    records.sort(key=lambda item: (item[0], item[1]))
    request_ids = set()
    with open(mixed_traffic_log, "w", encoding="utf-8") as output:
        for _, _, source_index, line_index, record in records:
            request_id = str(record.get("request_id", ""))
            if request_id in request_ids:
                request_id = f"{source_index}:{line_index}:{request_id}"
                record["request_id"] = request_id
            request_ids.add(request_id)
            json.dump(record, output, separators=(",", ":"))
            output.write("\n")
    return {
        "traffic_log": mixed_traffic_log,
        "service_name": "navi_sched_mixed_traffic",
        "biz_name": str(config["biz_name"]),
        "agent_count": int(config["agent_count"]),
        "client_count": int(config["client_count"]),
        "traffic_mode": "mixed",
        "source_metrics": source_metrics,
    }


if cases_file:
    with open(cases_file, encoding="utf-8") as source:
        cases = json.load(source)
else:
    cases = [{
        "traffic_log": traffic_log,
        "service_name": service_name,
        "biz_name": "agent_biz",
        "agent_count": int(agent_count),
        "client_count": int(client_count),
    }]
if isinstance(cases, dict):
    if cases.get("mode") != "mixed":
        raise RuntimeError("traffic object mode must be mixed")
    cases = [build_mixed_case(cases)]
if not isinstance(cases, list) or not cases:
    raise RuntimeError("traffic cases must be an array or mixed object")
required = ("traffic_log", "service_name", "biz_name", "agent_count", "client_count")
biz_names = set()
for case in cases:
    if not isinstance(case, dict) or any(name not in case for name in required):
        raise RuntimeError("each traffic case must define " + ", ".join(required))
    case["traffic_log"] = os.path.abspath(str(case["traffic_log"]))
    case["service_name"] = str(case["service_name"])
    case["biz_name"] = str(case["biz_name"])
    case["agent_count"] = int(case["agent_count"])
    case["client_count"] = int(case["client_count"])
    if not os.path.isfile(case["traffic_log"]):
        raise RuntimeError(f"missing traffic log: {case['traffic_log']}")
    if (not case["service_name"] or not case["biz_name"] or
            case["agent_count"] <= 0 or case["client_count"] <= 0):
        raise RuntimeError("traffic case values must be non-empty and positive")
    if case["biz_name"] in biz_names:
        raise RuntimeError(f"duplicate traffic biz: {case['biz_name']}")
    biz_names.add(case["biz_name"])
    case.setdefault("traffic_mode", "independent")
print(json.dumps(cases, separators=(",", ":")))
PY
)

mapfile -t CASE_ROWS < <(
    "${PYTHON_BIN}" - "${TRAFFIC_CASES_JSON}" <<'PY'
import json
import sys

for case in json.loads(sys.argv[1]):
    print("\t".join(str(case[name]) for name in (
        "traffic_log", "service_name", "biz_name", "agent_count", "client_count")))
PY
)
declare -a CASE_TRAFFIC_LOGS=()
declare -a CASE_SERVICE_NAMES=()
declare -a CASE_BIZ_NAMES=()
declare -a CASE_AGENT_COUNTS=()
declare -a CASE_CLIENT_COUNTS=()
declare -a AGENT_BIZ_NAMES=()
declare -a CLIENT_CASE_INDEXES=()
declare -a CLIENT_LOCAL_INDEXES=()
AGENT_COUNT=0
CLIENT_COUNT=0
for case_index in "${!CASE_ROWS[@]}"; do
    IFS=$'\t' read -r traffic_log service_name biz_name agent_count client_count \
        <<<"${CASE_ROWS[case_index]}"
    CASE_TRAFFIC_LOGS+=("${traffic_log}")
    CASE_SERVICE_NAMES+=("${service_name}")
    CASE_BIZ_NAMES+=("${biz_name}")
    CASE_AGENT_COUNTS+=("${agent_count}")
    CASE_CLIENT_COUNTS+=("${client_count}")
    for ((index = 0; index < agent_count; ++index)); do
        AGENT_BIZ_NAMES+=("${biz_name}")
    done
    for ((index = 0; index < client_count; ++index)); do
        CLIENT_CASE_INDEXES+=("${case_index}")
        CLIENT_LOCAL_INDEXES+=("${index}")
    done
    AGENT_COUNT=$((AGENT_COUNT + agent_count))
    CLIENT_COUNT=$((CLIENT_COUNT + client_count))
done

DEFAULT_TEST_DIR=${NAVI_SCHED_DIR}/tmp/traffic_replay_${AGENT_COUNT}x${CLIENT_COUNT}
TEST_DIR=${NAVI_SCHED_TRAFFIC_TEST_DIR:-${DEFAULT_TEST_DIR}}
mkdir -p "$(dirname "${TEST_DIR}")"
TEST_DIR=$(cd "$(dirname "${TEST_DIR}")" && pwd)/$(basename "${TEST_DIR}")

AGENT_CACHE_CAPACITY_BYTES=$((CLUSTER_CACHE_CAPACITY_BYTES / AGENT_COUNT))
if ((AGENT_CACHE_CAPACITY_BYTES <= 0)); then
    echo "cluster cache capacity is too small for agent count" >&2
    exit 1
fi

PORT_BASE=$(
    "${PYTHON_BIN}" - "${AGENT_COUNT}" <<'PY'
import socket
import sys

agent_count = int(sys.argv[1])
for base in range(40000, 65000 - agent_count - 3, agent_count + 3):
    sockets = []
    try:
        for port in range(base, base + agent_count + 2):
            value = socket.socket()
            value.bind(("0.0.0.0", port))
            sockets.append(value)
        print(base)
        break
    except OSError:
        pass
    finally:
        for value in sockets:
            value.close()
else:
    raise RuntimeError("no contiguous port range for traffic replay")
PY
)
MASTER_GRPC_PORT=${PORT_BASE}
MASTER_SCHEDULE_WAIT_TIME_MS=$(
    "${PYTHON_BIN}" - \
        "${SCHEDULE_WAIT_TIME_MS}" "${TIME_COMPRESSION}" <<'PY'
import math
import sys

print(math.ceil(int(sys.argv[1]) / float(sys.argv[2])))
PY
)

mapfile -t EXPECTED_COUNTS < <(
    "${PYTHON_BIN}" - "${TRAFFIC_CASES_JSON}" <<'PY'
import json
import sys

for case in json.loads(sys.argv[1]):
    request_count = 0
    with open(case["traffic_log"], encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            if json.loads(line).get("service_name") == case["service_name"]:
                request_count += 1
    if not request_count:
        raise RuntimeError(
            f"no traffic records found for service {case['service_name']}")
    client_count = case["client_count"]
    quotient, remainder = divmod(request_count, client_count)
    for index in range(client_count):
        print(quotient + (index < remainder))
PY
)
if ((${#EXPECTED_COUNTS[@]} != CLIENT_COUNT)); then
    echo "failed to calculate per-client request counts" >&2
    exit 1
fi
EXPECTED_FINISHED_REQUEST_COUNT=$(printf '%s\n' "${EXPECTED_COUNTS[@]}" |
    awk '{sum += $1} END {print sum + 0}')
declare -a REPLAY_SEQUENCES=()
if ((REPLAY_SAMPLE_INTERVAL > 0)); then
    for ((sequence = 0;  \
    sequence < EXPECTED_FINISHED_REQUEST_COUNT;  \
    sequence += REPLAY_SAMPLE_INTERVAL)); do
        REPLAY_SEQUENCES+=("${sequence}")
    done
else
    IFS=',' read -r -a REPLAY_SEQUENCES <<<"${REPLAY_SAMPLE_SEQUENCES}"
fi
for sequence in "${REPLAY_SEQUENCES[@]}"; do
    if [[ ! "${sequence}" =~ ^[0-9]+$ ]]; then
        echo "invalid replay sample sequence: ${sequence}" >&2
        exit 1
    fi
done

declare -a AGENT_PIDS=()
declare -a CLIENT_PIDS=()
MASTER_PID=

stop_pid_group() {
    local timeout_seconds=$1
    shift
    local pid
    for pid in "$@"; do
        [[ -z "${pid}" ]] || kill -TERM "${pid}" 2>/dev/null || true
    done
    for _ in $(seq 1 $((timeout_seconds * 10))); do
        local running=0
        for pid in "$@"; do
            if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
                running=1
                break
            fi
        done
        [[ ${running} -eq 0 ]] && break
        sleep 0.1
    done
    for pid in "$@"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    done
    for pid in "$@"; do
        [[ -z "${pid}" ]] || wait "${pid}" 2>/dev/null || true
    done
}

stop_roles() {
    # Master must remain available for the Agent stopping handshake.
    stop_pid_group 10 "${CLIENT_PIDS[@]:-}"
    stop_pid_group 25 "${AGENT_PIDS[@]:-}"
    stop_pid_group 10 "${MASTER_PID}"
}

cleanup() {
    local status=$?
    # A signal may interrupt a child wait.  Do not re-enter cleanup while
    # terminating the remaining role processes.
    trap - EXIT HUP INT TERM
    stop_roles
    rm -f "${MIXED_TRAFFIC_LOG}"
    if [[ ${status} -ne 0 ]]; then
        for log_file in "${TEST_DIR}"/*.log; do
            [[ -f "${log_file}" ]] || continue
            printf '%s\n' "--- ${log_file} ---" >&2
            tail -80 "${log_file}" >&2
        done
    fi
    return ${status}
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

rm -rf "${TEST_DIR}"
mkdir -p "${TEST_DIR}"
tar -xzf "${ARCHIVE}" -C "${TEST_DIR}"
PACKAGE_DIR=${TEST_DIR}/navi_sched_simulate-0.1.0
RUNTIME_DIR=${PACKAGE_DIR}/runtime
mkdir -p "${RUNTIME_DIR}/master"

EXPERIMENT_CONFIG_JSON=$(
    "${PYTHON_BIN}" - \
        "${SCHEDULER_NAME}" "${SCHEDULER_CONFIG}" \
        "${SCHEDULE_WAIT_COUNT}" "${SCHEDULE_WAIT_TIME_MS}" \
        "${AGENT_COUNT}" "${CLIENT_COUNT}" "${THREAD_NUM}" \
        "${TIME_COMPRESSION}" "${BLOCK_SIZE}" \
        "${CLUSTER_CACHE_CAPACITY_BYTES}" "${AGENT_CACHE_CAPACITY_BYTES}" \
        "${TOKEN_CACHE_BYTES}" "${NOISE_STDDEV_RATIO}" "${RANDOM_SEED}" \
        "${FORMULA_PARAMETERS_JSON}" "${LATENCY_MODEL_STATE_FILE}" \
        "${LATENCY_FIT_BATCH_SIZE}" \
        "${LATENCY_FIT_INTERVAL}" "${LATENCY_LEARNING_RATE}" \
        "${LATENCY_WARMUP_SAMPLE_COUNT}" "${REPLAY_SAMPLE_INTERVAL}" \
        "${REQUEST_TIMEOUT_MS}" "${COMPLETION_TIMEOUT_SECONDS}" <<'PY'
import json
import hashlib
import os
import sys

(
    scheduler_name, scheduler_config, schedule_wait_count,
    schedule_wait_time_ms, agent_count, client_count, thread_num,
    time_compression, block_size, cluster_cache_capacity_bytes,
    agent_cache_capacity_bytes, token_cache_bytes, noise_stddev_ratio,
    random_seed, formula_parameters, latency_model_state_file,
    latency_fit_batch_size,
    latency_fit_interval, latency_learning_rate,
    latency_warmup_sample_count, replay_sample_interval,
    request_timeout_ms, completion_timeout_seconds,
) = sys.argv[1:]
print(json.dumps({
    "scheduler_name": scheduler_name,
    "scheduler_config": json.loads(scheduler_config or "{}"),
    "schedule_wait_count": int(schedule_wait_count),
    "schedule_wait_time_ms": int(schedule_wait_time_ms),
    "agent_count": int(agent_count),
    "client_count": int(client_count),
    "thread_num": int(thread_num),
    "time_compression": float(time_compression),
    "block_size": int(block_size),
    "cluster_cache_capacity_bytes": int(cluster_cache_capacity_bytes),
    "agent_cache_capacity_bytes": int(agent_cache_capacity_bytes),
    "token_cache_bytes": int(token_cache_bytes),
    "noise_stddev_ratio": float(noise_stddev_ratio),
    "random_seed": int(random_seed),
    "formula_parameters": json.loads(formula_parameters or "[]"),
    "latency_model_state_file": latency_model_state_file,
    "latency_model_state_sha256": (
        hashlib.sha256(open(latency_model_state_file, "rb").read()).hexdigest()
        if latency_model_state_file and os.path.isfile(latency_model_state_file)
        else ""
    ),
    "latency_fit_batch_size": int(latency_fit_batch_size),
    "latency_fit_interval": int(latency_fit_interval),
    "latency_learning_rate": float(latency_learning_rate),
    "latency_warmup_sample_count": int(latency_warmup_sample_count),
    "replay_sample_interval": int(replay_sample_interval),
    "request_timeout_ms": int(request_timeout_ms),
    "completion_timeout_seconds": int(completion_timeout_seconds),
}, sort_keys=True, separators=(",", ":")))
PY
)
printf '%s\n' "${EXPERIMENT_CONFIG_JSON}" \
    >"${TEST_DIR}/requested_experiment_config.json"
printf 'requested_experiment_config=%s\n' "${EXPERIMENT_CONFIG_JSON}" \
    >"${TEST_DIR}/experiment_config.log"

SCHEDULER_CONFIG_JSON=$(
    "${PYTHON_BIN}" - "${SCHEDULER_CONFIG}" <<'PY'
import json
import sys

config = json.loads(sys.argv[1] or "{}")
if not isinstance(config, dict):
    raise ValueError("scheduler config must be a JSON object")
print(json.dumps(config, sort_keys=True, separators=(",", ":")))
PY
)
printf \
    '{"biz_name":"master_biz","scheduler_name":"%s",''"scheduler_config":%s,"max_wait_schedule_count":%d,''"max_wait_schedule_time_ms":%d,"time_compression":%s,"local_nodes":[' \
    "${SCHEDULER_NAME}" "${SCHEDULER_CONFIG_JSON}" \
    "${SCHEDULE_WAIT_COUNT}" "${MASTER_SCHEDULE_WAIT_TIME_MS}" \
    "${TIME_COMPRESSION}" \
    >"${RUNTIME_DIR}/master/config.json"
for index in $(seq 0 $((AGENT_COUNT - 1))); do
    ((index == 0)) || printf ',' >>"${RUNTIME_DIR}/master/config.json"
    agent_port=$((PORT_BASE + index + 2))
    printf '{"biz_name":"%s","ip":"127.0.0.1","grpc_port":%d,"support_heartbeat":true}' \
        "${AGENT_BIZ_NAMES[index]}" "${agent_port}" \
        >>"${RUNTIME_DIR}/master/config.json"
done
printf ']}\n' >>"${RUNTIME_DIR}/master/config.json"

for index in $(seq 0 $((AGENT_COUNT - 1))); do
    agent_port=$((PORT_BASE + index + 2))
    runtime_dir=${RUNTIME_DIR}/agent_${index}
    mkdir -p "${runtime_dir}"
    agent_latency_model_state_file=
    if [[ -n "${LATENCY_MODEL_STATE_FILE}" ]]; then
        agent_latency_model_state_file=${runtime_dir}/latency_model.state
        cp "${LATENCY_MODEL_STATE_FILE}" "${agent_latency_model_state_file}"
    fi
    (
        cd "${runtime_dir}"
        printf 'effective_agent_launch_config=%s\n' \
            "${EXPERIMENT_CONFIG_JSON}"
        exec env PATH=/usr/bin:/bin LD_LIBRARY_PATH="${PACKAGE_DIR}/lib" \
            "${PACKAGE_DIR}/bin/navi_sched_simulate_agent" \
            --biz "${AGENT_BIZ_NAMES[index]}" \
            --grpc-port "${agent_port}" \
            --server-port "${agent_port}" \
            --thread-num "${THREAD_NUM}" \
            --time-compression "${TIME_COMPRESSION}" \
            --cache-capacity-bytes "${AGENT_CACHE_CAPACITY_BYTES}" \
            --token-cache-bytes "${TOKEN_CACHE_BYTES}" \
            --block-size "${BLOCK_SIZE}" \
            --formula-parameters-json "${FORMULA_PARAMETERS_JSON}" \
            --latency-model-state-file "${agent_latency_model_state_file}" \
            --latency-fit-batch-size "${LATENCY_FIT_BATCH_SIZE}" \
            --latency-fit-interval "${LATENCY_FIT_INTERVAL}" \
            --latency-learning-rate "${LATENCY_LEARNING_RATE}" \
            --latency-warmup-sample-count "${LATENCY_WARMUP_SAMPLE_COUNT}" \
            --noise-stddev-ratio "${NOISE_STDDEV_RATIO}" \
            --random-seed "${RANDOM_SEED}" \
            --queue-trace-output "${runtime_dir}/traffic_queue_trace.csv" \
            --runtime-dir "${runtime_dir}"
    ) >"${TEST_DIR}/agent_${index}.log" 2>&1 &
    AGENT_PIDS+=("$!")
done

for index in $(seq 0 $((AGENT_COUNT - 1))); do
    agent_pid=${AGENT_PIDS[index]}
    agent_log=${RUNTIME_DIR}/agent_${index}/navi_sched.log
    for _ in $(seq 1 600); do
        if grep -q "snapshot start success" "${agent_log}" 2>/dev/null; then
            break
        fi
        if ! kill -0 "${agent_pid}" 2>/dev/null; then
            wait "${agent_pid}" || true
            echo "traffic replay agent ${index} exited during initialization" >&2
            exit 1
        fi
        sleep 0.05
    done
    if ! grep -q "snapshot start success" "${agent_log}" 2>/dev/null; then
        echo "traffic replay agent ${index} initialization timed out" >&2
        exit 1
    fi
done

(
    cd "${RUNTIME_DIR}/master"
    printf 'effective_master_launch_config=%s\n' \
        "${EXPERIMENT_CONFIG_JSON}"
    exec env PATH=/usr/bin:/bin LD_LIBRARY_PATH="${PACKAGE_DIR}/lib" \
        NAVI_SCHED_MASTER_BIZ_NAME=master_biz \
        NAVI_SCHED_MASTER_CONFIG_FILE=config.json \
        NAVI_SCHED_MASTER_GRPC_PORT="${MASTER_GRPC_PORT}" \
        NAVI_SCHED_MASTER_LOG_FILE=navi_sched.log \
        NAVI_SCHED_MASTER_SNAPSHOT_DIR=navi_sched \
        NAVI_SCHED_SIMULATE_STATS_FILE="${TEST_DIR}/stats.json" \
        "${PACKAGE_DIR}/bin/navi_sched_simulate_master"
) >"${TEST_DIR}/master.log" 2>&1 &
MASTER_PID=$!

# Start traffic only after every Agent has completed its Master handshake.
ready_agent_count=0
for _ in $(seq 1 1200); do
    latest_snapshot=$(find \
        "${RUNTIME_DIR}/master/navi_sched/master_snapshot" \
        -type f -name '*_snapshot' 2>/dev/null |
        sort | tail -1 || true)
    if [[ -n "${latest_snapshot}" ]]; then
        ready_agent_count=$(grep -a '^  count: [0-9]*$' \
            "${latest_snapshot}" | tail -1 | awk '{print $2}')
        if [[ "${ready_agent_count:-0}" -eq "${AGENT_COUNT}" ]]; then
            break
        fi
    fi
    if ! kill -0 "${MASTER_PID}" 2>/dev/null; then
        wait "${MASTER_PID}" || true
        echo "traffic replay master exited during initialization" >&2
        exit 1
    fi
    sleep 0.1
done
if [[ "${ready_agent_count:-0}" -ne "${AGENT_COUNT}" ]]; then
    echo "traffic replay master discovered ${ready_agent_count:-0} of ""\
${AGENT_COUNT} agents" >&2
    exit 1
fi
for index in $(seq 0 $((AGENT_COUNT - 1))); do
    if ! kill -0 "${AGENT_PIDS[index]}" 2>/dev/null; then
        wait "${AGENT_PIDS[index]}" || true
        echo "traffic replay agent ${index} exited before client startup" >&2
        exit 1
    fi
done

for index in $(seq 0 $((CLIENT_COUNT - 1))); do
    case_index=${CLIENT_CASE_INDEXES[index]}
    local_client_index=${CLIENT_LOCAL_INDEXES[index]}
    runtime_dir=${RUNTIME_DIR}/client_${index}
    mkdir -p "${runtime_dir}"
    (
        replay_args=()
        for sequence in "${REPLAY_SEQUENCES[@]}"; do
            replay_args+=(--traffic-collect-replay-sequence "${sequence}")
        done
        cd "${runtime_dir}"
        exec env PATH=/usr/bin:/bin LD_LIBRARY_PATH="${PACKAGE_DIR}/lib" \
            "${PACKAGE_DIR}/bin/navi_sched_simulate_client" \
            --biz client_biz \
            --master-biz master_biz \
            --node "master_biz,127.0.0.1,${MASTER_GRPC_PORT}" \
            --traffic-log "${CASE_TRAFFIC_LOGS[case_index]}" \
            --traffic-biz "${CASE_BIZ_NAMES[case_index]}" \
            --traffic-service-name "${CASE_SERVICE_NAMES[case_index]}" \
            --traffic-client-count "${CASE_CLIENT_COUNTS[case_index]}" \
            --traffic-client-index "${local_client_index}" \
            --traffic-timeout-ms "${REQUEST_TIMEOUT_MS}" \
            --block-size "${BLOCK_SIZE}" \
            "${replay_args[@]}" \
            --thread-num "${THREAD_NUM}" \
            --time-compression "${TIME_COMPRESSION}" \
            --runtime-dir "${runtime_dir}"
    ) >"${TEST_DIR}/client_${index}.log" 2>&1 &
    CLIENT_PIDS+=("$!")
done

for index in $(seq 0 $((CLIENT_COUNT - 1))); do
    expected_count=${EXPECTED_COUNTS[index]}
    client_pid=${CLIENT_PIDS[index]}
    for _ in $(seq 1 $((COMPLETION_TIMEOUT_SECONDS * 20))); do
        if grep -q "traffic replay completed, requestCount \[${expected_count}\]" \
            "${RUNTIME_DIR}/client_${index}/navi_sched.log" 2>/dev/null; then
            break
        fi
        if ! kill -0 "${client_pid}" 2>/dev/null; then
            wait "${client_pid}" || true
            echo "traffic replay client ${index} exited before completion" >&2
            exit 1
        fi
        sleep 0.05
    done
    grep -q "traffic replay completed, requestCount \[${expected_count}\]" \
        "${RUNTIME_DIR}/client_${index}/navi_sched.log"
done

for _ in $(seq 1 $((COMPLETION_TIMEOUT_SECONDS * 2))); do
    latest_snapshot=$(find "${RUNTIME_DIR}/master/navi_sched/master_snapshot" \
        -type f -name '*_snapshot' 2>/dev/null | sort | tail -1)
    if [[ -n "${latest_snapshot}" ]]; then
        finished_request_count=$(grep -a '^  finished_requests: [0-9]*$' \
            "${latest_snapshot}" | tail -1 | awk '{print $2}')
        if [[ "${finished_request_count:-0}" -eq "${EXPECTED_FINISHED_REQUEST_COUNT}" ]]; then
            break
        fi
    fi
    if ! kill -0 "${MASTER_PID}" 2>/dev/null; then
        echo "traffic replay master exited before receiving all completions" >&2
        exit 1
    fi
    sleep 0.5
done
if [[ "${finished_request_count:-0}" -ne "${EXPECTED_FINISHED_REQUEST_COUNT}" ]]; then
    echo "traffic replay master received ${finished_request_count:-0} of ""\
${EXPECTED_FINISHED_REQUEST_COUNT} completions" >&2
    exit 1
fi
cp "${latest_snapshot}" "${TEST_DIR}/completed_master_snapshot.txt"
COMPLETION_NS=$(date +%s%N)

if grep -R -q "EC_TIMEOUT" "${RUNTIME_DIR}"; then
    echo "traffic replay encountered an unexpected graph timeout" >&2
    exit 1
fi

stop_roles
CLIENT_PIDS=()
MASTER_PID=
AGENT_PIDS=()

summary_args=(
    "${TEST_DIR}" "${TRAFFIC_CASES_JSON}" "${SCHEDULER_NAME}"
    "${SCHEDULER_CONFIG_JSON}" "${SCHEDULE_WAIT_COUNT}"
    "${SCHEDULE_WAIT_TIME_MS}" "${RANDOM_SEED}"
    "${AGENT_COUNT}" "${CLIENT_COUNT}" "${CLUSTER_CACHE_CAPACITY_BYTES}"
    "${AGENT_CACHE_CAPACITY_BYTES}" "${BLOCK_SIZE}" "${TOKEN_CACHE_BYTES}"
    "${TIME_COMPRESSION}" "${NOISE_STDDEV_RATIO}"
    "${REQUIRE_CACHE_EVICTION}" "${COMPLETION_NS}"
    "${EXPERIMENT_CONFIG_JSON}"
)
"${PYTHON_BIN}" - "${summary_args[@]}" <<'PY'
import csv
import json
import math
import os
import re
import sys

(
    test_dir,
    traffic_cases_json,
    scheduler_name,
    scheduler_config_json,
    schedule_wait_count,
    schedule_wait_time_ms,
    random_seed,
    agent_count,
    client_count,
    cluster_cache_capacity_bytes,
    agent_cache_capacity_bytes,
    block_size,
    token_cache_bytes,
    time_compression,
    noise_stddev_ratio,
    require_cache_eviction,
    completion_ns,
    experiment_config_json,
) = sys.argv[1:]
traffic_cases = json.loads(traffic_cases_json)
experiment_config = json.loads(experiment_config_json)
scheduler_config = json.loads(scheduler_config_json)
if isinstance(scheduler_config, str):
    scheduler_config = (
        json.loads(scheduler_config) if scheduler_config else {}
    )
schedule_wait_count = int(schedule_wait_count)
schedule_wait_time_ms = int(schedule_wait_time_ms)
random_seed = int(random_seed)
agent_count = int(agent_count)
client_count = int(client_count)
cluster_cache_capacity_bytes = int(cluster_cache_capacity_bytes)
agent_cache_capacity_bytes = int(agent_cache_capacity_bytes)
block_cache_bytes = int(block_size) * int(token_cache_bytes)
time_compression = float(time_compression)
noise_stddev_ratio = float(noise_stddev_ratio)
require_cache_eviction = bool(int(require_cache_eviction))
if scheduler_name == "cost":
    scheduler_config["optimization_time_budget_us"] = (
        math.ceil(schedule_wait_time_ms / time_compression) * 1000
    )
case_requests = []
case_prompt_token_counts = []
case_source_prompt_token_counts = []
request_sources_by_biz = {}
client_cases = []
for case in traffic_cases:
    requests = []
    prompt_token_counts = []
    source_prompt_token_counts = {}
    request_sources = {}
    with open(case["traffic_log"], encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("service_name") != case["service_name"]:
                continue
            timestamp = record.get(
                "timestamp_ms", record.get("timestamp", record.get("reqTime")))
            if isinstance(timestamp, float):
                timestamp = round(timestamp * 1000)
            prompt_token_count = record.get("input_length")
            if prompt_token_count is None:
                prompt_token_count = record.get("prompt_token_count")
            if prompt_token_count is None:
                prompt_token_count = record.get("usage", {}).get("inputTokens")
            if prompt_token_count is None:
                raise RuntimeError("traffic record has no prompt token count")
            prompt_token_count = int(prompt_token_count)
            prompt_token_counts.append(prompt_token_count)
            source_name = record.get("traffic_source", case["service_name"])
            requests.append((int(timestamp), len(requests), source_name))
            source_prompt_token_counts.setdefault(source_name, []).append(
                prompt_token_count)
    requests.sort(key=lambda item: item[0])
    request_sources = {
        request_id: request[2]
        for request_id, request in enumerate(requests, 1)
    }
    case_requests.append(requests)
    case_prompt_token_counts.append(prompt_token_counts)
    case_source_prompt_token_counts.append(source_prompt_token_counts)
    request_sources_by_biz[case["biz_name"]] = request_sources
    for local_index in range(case["client_count"]):
        client_cases.append((case, local_index))
if len(client_cases) != client_count:
    raise RuntimeError("traffic case client counts do not match runtime clients")
agent_biz_names = []
for case in traffic_cases:
    agent_biz_names.extend([case["biz_name"]] * case["agent_count"])
if len(agent_biz_names) != agent_count:
    raise RuntimeError("traffic case agent counts do not match runtime agents")
expected = []
for case, requests in zip(traffic_cases, case_requests):
    quotient, remainder = divmod(len(requests), case["client_count"])
    expected.extend(
        quotient + (index < remainder)
        for index in range(case["client_count"]))

timings = {
    "batch_wait_ms": [],
    "queue_wait_ms": [],
    "execution_latency_ms": [],
    "client_total_latency_ms": [],
}
request_timings = {}
schedule_losses = {}
schedule_loss_biz_names = {}
schedule_latencies_us = {}
schedule_request_counts = {}
replay_start_pattern = re.compile(
    r"traffic replay started, startTimeMs \[(\d+)\]")
replay_start_times_ms = {
    case["biz_name"]: set() for case in traffic_cases
}
biz_timings = {
    case["biz_name"]: {name: [] for name in timings}
    for case in traffic_cases
}
source_timings = {
    case["biz_name"]: {
        source_name: {name: [] for name in timings}
        for source_name in source_prompt_token_counts
    }
    for case, source_prompt_token_counts in zip(
        traffic_cases, case_source_prompt_token_counts)
}
for index, count in enumerate(expected):
    case, _ = client_cases[index]
    biz_name = case["biz_name"]
    path = os.path.join(
        test_dir, "navi_sched_simulate-0.1.0", "runtime",
        f"client_{index}", "traffic_request_latency.jsonl")
    with open(path, encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    if len(records) != count:
        raise RuntimeError(f"client {index} expected {count} results, got {len(records)}")
    for record in records:
        timing = record["request_timing"]
        request_timings[(biz_name, record["request_id"])] = timing
        source_name = request_sources_by_biz[biz_name][
            int(record["request_id"])]
        schedule_id = int(record["schedule_id"])
        schedule_loss = float(record["schedule_loss"])
        schedule_latency_us = int(record["schedule_latency_us"])
        previous = schedule_losses.get(schedule_id)
        if previous is not None and previous != schedule_loss:
            raise RuntimeError(
                f"schedule {schedule_id} returned inconsistent loss")
        previous_biz_name = schedule_loss_biz_names.get(schedule_id)
        if previous_biz_name is not None and previous_biz_name != biz_name:
            raise RuntimeError(
                f"schedule {schedule_id} returned for multiple traffic bizs")
        schedule_losses[schedule_id] = schedule_loss
        schedule_loss_biz_names[schedule_id] = biz_name
        previous_latency_us = schedule_latencies_us.get(schedule_id)
        if (previous_latency_us is not None and
                previous_latency_us != schedule_latency_us):
            raise RuntimeError(
                f"schedule {schedule_id} returned inconsistent latency")
        schedule_latencies_us[schedule_id] = schedule_latency_us
        schedule_request_counts[schedule_id] = (
            schedule_request_counts.get(schedule_id, 0) + 1)
        for name, values in timings.items():
            values.append(timing[name])
            biz_timings[biz_name][name].append(timing[name])
            source_timings[biz_name][source_name][name].append(timing[name])
    log_path = os.path.join(
        test_dir, "navi_sched_simulate-0.1.0", "runtime",
        f"client_{index}", "navi_sched.log")
    with open(log_path, encoding="utf-8") as source:
        start_matches = replay_start_pattern.findall(source.read())
    if len(start_matches) != 1:
        raise RuntimeError(f"client {index} did not emit one replay start time")
    replay_start_times_ms[biz_name].add(int(start_matches[0]))
if any(len(values) != 1 for values in replay_start_times_ms.values()):
    raise RuntimeError("clients in one traffic biz did not use one start time")
replay_start_time_ms = min(
    next(iter(values)) for values in replay_start_times_ms.values())
wall_duration_ms = int(completion_ns) / 1_000_000 - replay_start_time_ms

with open(os.path.join(test_dir, "stats.json"), encoding="utf-8") as source:
    stats = json.load(source)
if stats["finished_request_count"] != len(timings["client_total_latency_ms"]):
    raise RuntimeError("master finished count does not match client timing records")
if stats["cache_hit_tokens"] + stats["cache_miss_tokens"] <= 0:
    raise RuntimeError("traffic replay did not perform cache matching")
for model in stats["latency_models"]:
    if model["revision"] <= 1:
        raise RuntimeError("latency model did not fit any execution batches")

with open(os.path.join(test_dir, "completed_master_snapshot.txt"),
          encoding="utf-8") as source:
    completed_master_snapshot = source.read()

prediction_models = []
for block in completed_master_snapshot.split("\n  Agent\n")[1:]:
    validation = re.search(
        r"^    latency_validation: samples=(\d+), unmatched_batches=(\d+)$",
        block, re.MULTILINE)
    queue = re.search(
        r"^      queue_ms: predicted_mean=([0-9.eE+-]+), "
        r"actual_mean=([0-9.eE+-]+), backend_mean=[0-9.eE+-]+, "
        r"bias_mean=([0-9.eE+-]+), mae=([0-9.eE+-]+)$",
        block, re.MULTILINE)
    execution = re.search(
        r"^      execution_ms: master_predicted_mean=([0-9.eE+-]+), "
        r"agent_pre_update_predicted_mean=([0-9.eE+-]+), "
        r"actual_mean=([0-9.eE+-]+), master_bias_mean=([0-9.eE+-]+), "
        r"master_mae=([0-9.eE+-]+), master_mape_pct=([0-9.eE+-]+), "
        r"agent_bias_mean=([0-9.eE+-]+), agent_mae=([0-9.eE+-]+), "
        r"agent_mape_pct=([0-9.eE+-]+)$",
        block, re.MULTILINE)
    total = re.search(
        r"^      total_ms: predicted_mean=([0-9.eE+-]+), "
        r"actual_mean=([0-9.eE+-]+), bias_mean=([0-9.eE+-]+), "
        r"mae=([0-9.eE+-]+), mape_pct=([0-9.eE+-]+)$",
        block, re.MULTILINE)
    if not validation or not queue or not execution or not total:
        continue
    prediction_models.append({
        "prediction_sample_count": int(validation.group(1)),
        "prediction_unmatched_batch_count": int(validation.group(2)),
        "master_predicted_queue_mean_ms": float(queue.group(1)),
        "actual_queue_mean_ms": float(queue.group(2)),
        "queue_actual_minus_predicted_bias_ms": float(queue.group(3)),
        "queue_mae_ms": float(queue.group(4)),
        "master_predicted_execution_mean_ms": float(execution.group(1)),
        "agent_pre_update_predicted_execution_mean_ms": float(execution.group(2)),
        "actual_execution_mean_ms": float(execution.group(3)),
        "execution_actual_minus_master_predicted_bias_ms": float(
            execution.group(4)),
        "execution_master_prediction_mae_ms": float(execution.group(5)),
        "execution_master_prediction_mape_pct": float(execution.group(6)),
        "execution_actual_minus_agent_pre_update_predicted_bias_ms": float(
            execution.group(7)),
        "execution_agent_pre_update_prediction_mae_ms": float(
            execution.group(8)),
        "execution_agent_pre_update_prediction_mape_pct": float(
            execution.group(9)),
        "master_predicted_total_mean_ms": float(total.group(1)),
        "actual_total_mean_ms": float(total.group(2)),
        "total_actual_minus_predicted_bias_ms": float(total.group(3)),
        "total_mae_ms": float(total.group(4)),
        "total_mape_pct": float(total.group(5)),
    })
prediction_sample_count = sum(
    model["prediction_sample_count"] for model in prediction_models)
if prediction_sample_count <= 0:
    raise RuntimeError("simulator produced no decision prediction samples")

def prediction_weighted_mean(field):
    return sum(
        model["prediction_sample_count"] * model[field]
        for model in prediction_models
    ) / prediction_sample_count

decision_prediction_accuracy = {
    "sample_count": prediction_sample_count,
    "unmatched_batch_count": sum(
        model["prediction_unmatched_batch_count"]
        for model in prediction_models),
    "queue": {
        "predicted_mean_ms": prediction_weighted_mean(
            "master_predicted_queue_mean_ms"),
        "actual_mean_ms": prediction_weighted_mean("actual_queue_mean_ms"),
        "actual_minus_predicted_bias_ms": prediction_weighted_mean(
            "queue_actual_minus_predicted_bias_ms"),
        "mae_ms": prediction_weighted_mean("queue_mae_ms"),
    },
    "execution": {
        "predicted_mean_ms": prediction_weighted_mean(
            "master_predicted_execution_mean_ms"),
        "actual_mean_ms": prediction_weighted_mean("actual_execution_mean_ms"),
        "actual_minus_predicted_bias_ms": prediction_weighted_mean(
            "execution_actual_minus_master_predicted_bias_ms"),
        "mae_ms": prediction_weighted_mean(
            "execution_master_prediction_mae_ms"),
        "mape_pct": prediction_weighted_mean(
            "execution_master_prediction_mape_pct"),
    },
    "agent_pre_update_execution": {
        "predicted_mean_ms": prediction_weighted_mean(
            "agent_pre_update_predicted_execution_mean_ms"),
        "actual_mean_ms": prediction_weighted_mean("actual_execution_mean_ms"),
        "actual_minus_predicted_bias_ms": prediction_weighted_mean(
            "execution_actual_minus_agent_pre_update_predicted_bias_ms"),
        "mae_ms": prediction_weighted_mean(
            "execution_agent_pre_update_prediction_mae_ms"),
        "mape_pct": prediction_weighted_mean(
            "execution_agent_pre_update_prediction_mape_pct"),
    },
    "total": {
        "predicted_mean_ms": prediction_weighted_mean(
            "master_predicted_total_mean_ms"),
        "actual_mean_ms": prediction_weighted_mean("actual_total_mean_ms"),
        "actual_minus_predicted_bias_ms": prediction_weighted_mean(
            "total_actual_minus_predicted_bias_ms"),
        "mae_ms": prediction_weighted_mean("total_mae_ms"),
        "mape_pct": prediction_weighted_mean("total_mape_pct"),
    },
}

fit_pattern = re.compile(
    r"simulated model fit summary, batchCount \[(\d+)\], "
    r"predictionBiasMs \[([0-9.eE+-]+)\], "
    r"predictionMaeMs \[([0-9.eE+-]+)\], "
    r"predictionRmseMs \[([0-9.eE+-]+)\], "
    r"predictionMeanRelativeAbsoluteError \[([0-9.eE+-]+)\]")
model_fit = []
cache_agents = []
execution_agents = []
cache_pattern = re.compile(
    r"simulated cache summary, cachedBlockCount \[(\d+)\], "
    r"evictedBlockCount \[(\d+)\]")
execution_pattern = re.compile(
    r"simulated execution summary, batchCount \[(\d+)\], "
    r"requestCount \[(\d+)\], logicalBusyMs \[([0-9.eE+-]+)\], "
    r"logicalElapsedMs \[([0-9.eE+-]+)\]")
for index in range(agent_count):
    path = os.path.join(
        test_dir, "navi_sched_simulate-0.1.0", "runtime",
        f"agent_{index}", "navi_sched.log")
    with open(path, encoding="utf-8") as source:
        log = source.read()
    matches = fit_pattern.findall(log)
    if len(matches) > 1:
        raise RuntimeError(f"agent {index} emitted multiple model fit summaries")
    cache_matches = cache_pattern.findall(log)
    if not cache_matches:
        raise RuntimeError(f"agent {index} did not emit a cache summary")
    cached_blocks, evicted_blocks = max(
        cache_matches, key=lambda item: (int(item[1]), int(item[0])))
    cache_agents.append({
        "agent_index": index,
        "cached_block_count": int(cached_blocks),
        "evicted_block_count": int(evicted_blocks),
    })
    execution_matches = execution_pattern.findall(log)
    if len(execution_matches) != 1:
        raise RuntimeError(f"agent {index} did not emit one execution summary")
    batch_count, request_count, busy_ms, elapsed_ms = execution_matches[0]
    batch_count = int(batch_count)
    request_count = int(request_count)
    if matches:
        fit_batch_count, bias_ms, mae_ms, rmse_ms, relative_mae = matches[0]
        if int(fit_batch_count) != batch_count:
            raise RuntimeError(
                f"agent {index} fit and execution batch counts differ")
        model_fit.append({
            "batch_count": batch_count,
            "prediction_bias_ms": float(bias_ms),
            "prediction_mae_ms": float(mae_ms),
            "prediction_rmse_ms": float(rmse_ms),
            "prediction_mean_relative_absolute_error": float(relative_mae),
        })
    elif batch_count:
        raise RuntimeError(f"agent {index} did not emit one model fit summary")
    else:
        model_fit.append({
            "batch_count": 0,
            "prediction_bias_ms": None,
            "prediction_mae_ms": None,
            "prediction_rmse_ms": None,
            "prediction_mean_relative_absolute_error": None,
        })
    execution_agents.append({
        "agent_index": index,
        "biz_name": agent_biz_names[index],
        "batch_count": batch_count,
        "request_count": request_count,
        "average_batch_size": request_count / batch_count if batch_count else 0.0,
        "logical_busy_ms": float(busy_ms),
        "logical_elapsed_ms": float(elapsed_ms),
    })
active_model_fit = [item for item in model_fit if item["batch_count"] > 0]
total_fit_batch_count = sum(item["batch_count"] for item in active_model_fit)
if total_fit_batch_count <= 0:
    raise RuntimeError("traffic replay did not execute any batches")
prediction_mae_ms = sum(
    item["batch_count"] * item["prediction_mae_ms"] for item in active_model_fit
) / total_fit_batch_count
prediction_rmse_ms = (
    sum(item["batch_count"] * item["prediction_rmse_ms"] ** 2
        for item in active_model_fit) / total_fit_batch_count
) ** 0.5
prediction_bias_ms = sum(
    item["batch_count"] * item["prediction_bias_ms"] for item in active_model_fit
) / total_fit_batch_count
prediction_relative_mae = sum(
    item["batch_count"] * item["prediction_mean_relative_absolute_error"]
    for item in active_model_fit
) / total_fit_batch_count
cache_evicted_block_count = sum(
    item["evicted_block_count"] for item in cache_agents)
if require_cache_eviction and cache_evicted_block_count <= 0:
    raise RuntimeError("traffic replay did not trigger cache eviction")

def distribution(values):
    values.sort()
    def percentile(value):
        return values[min(len(values) - 1, int((len(values) - 1) * value))]
    return {
        "average_ms": sum(values) / len(values),
        "p50_ms": percentile(0.50),
        "p90_ms": percentile(0.90),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": values[-1],
    }

def comparison(values):
    predicted = [item[0] for item in values]
    actual = [item[1] for item in values]
    errors = [left - right for left, right in values]
    absolute_errors = [abs(value) for value in errors]
    predicted_average = sum(predicted) / len(predicted)
    actual_average = sum(actual) / len(actual)
    covariance = sum(
        (left - predicted_average) * (right - actual_average)
        for left, right in values)
    denominator = (
        sum((value - predicted_average) ** 2 for value in predicted) *
        sum((value - actual_average) ** 2 for value in actual)
    ) ** 0.5
    return {
        "sample_count": len(values),
        "predicted_average_ms": predicted_average,
        "actual_average_ms": actual_average,
        "bias_ms": sum(errors) / len(errors),
        "mae_ms": sum(absolute_errors) / len(absolute_errors),
        "rmse_ms": (sum(value ** 2 for value in errors) / len(errors)) ** 0.5,
        "absolute_error_distribution": distribution(absolute_errors),
        "correlation": covariance / denominator if denominator else None,
    }

def optional_distribution(values):
    return distribution(values) if values else None

def scalar_distribution(values):
    ordered = sorted(values)
    average = sum(ordered) / len(ordered)
    def percentile(value):
        return ordered[min(
            len(ordered) - 1, int((len(ordered) - 1) * value))]
    return {
        "count": len(ordered),
        "average": average,
        "stddev": (
            sum((value - average) ** 2 for value in ordered) /
            len(ordered)
        ) ** 0.5,
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }

def cache_metrics(values):
    hit_tokens = sum(item[0] for item in values)
    miss_tokens = sum(item[1] for item in values)
    total_tokens = hit_tokens + miss_tokens
    return {
        "request_count": len(values),
        "cache_hit_tokens": hit_tokens,
        "cache_miss_tokens": miss_tokens,
        "cache_hit_rate": hit_tokens / total_tokens,
        "cache_hit_token_distribution": scalar_distribution(
            [item[0] for item in values]),
        "cache_miss_token_distribution": scalar_distribution(
            [item[1] for item in values]),
    }

cost_scheduler_loss = None
biz_schedule_losses = {case["biz_name"]: [] for case in traffic_cases}
if schedule_losses:
    loss_records = [
        {
            "schedule_id": schedule_id,
            "value": schedule_losses[schedule_id],
        }
        for schedule_id in sorted(schedule_losses)
    ]
    cost_scheduler_loss = {
        "distribution": scalar_distribution(
            [record["value"] for record in loss_records]),
        "records": loss_records,
    }
    for schedule_id, value in schedule_losses.items():
        biz_schedule_losses[schedule_loss_biz_names[schedule_id]].append(value)

scheduler_metrics = None
if schedule_latencies_us:
    scheduler_metrics = {
        "schedule_count": len(schedule_latencies_us),
        "average_requests_per_schedule": (
            sum(schedule_request_counts.values()) /
            len(schedule_request_counts)
        ),
        "request_count_distribution": scalar_distribution(
            list(schedule_request_counts.values())),
        "latency_us_distribution": scalar_distribution(
            list(schedule_latencies_us.values())),
    }

queue_traces = {}
agent_batches = {index: {} for index in range(agent_count)}
biz_cache_values = {case["biz_name"]: [] for case in traffic_cases}
source_cache_values = {
    case["biz_name"]: {
        source_name: []
        for source_name in source_prompt_token_counts
    }
    for case, source_prompt_token_counts in zip(
        traffic_cases, case_source_prompt_token_counts)
}
biz_batch_sizes = {case["biz_name"]: [] for case in traffic_cases}
source_batch_sizes = {
    case["biz_name"]: {
        source_name: []
        for source_name in source_prompt_token_counts
    }
    for case, source_prompt_token_counts in zip(
        traffic_cases, case_source_prompt_token_counts)
}
for index in range(agent_count):
    path = os.path.join(
        test_dir, "navi_sched_simulate-0.1.0", "runtime",
        f"agent_{index}", "traffic_queue_trace.csv")
    with open(path, encoding="utf-8") as source:
        for record in csv.DictReader(source):
            request_id = int(record["request_id"])
            biz_name = agent_biz_names[index]
            request_key = (biz_name, request_id)
            if request_key in queue_traces:
                raise RuntimeError(f"duplicate queue trace request {request_key}")
            total_token_count = int(record["total_token_count"])
            cache_hit_token_count = int(record["cache_hit_token_count"])
            if not 0 <= cache_hit_token_count <= total_token_count:
                raise RuntimeError(
                    f"invalid cache stats for request {request_key}")
            source_name = request_sources_by_biz[biz_name][request_id]
            cache_value = (
                cache_hit_token_count,
                total_token_count - cache_hit_token_count,
            )
            biz_cache_values[biz_name].append(cache_value)
            source_cache_values[biz_name][source_name].append(cache_value)
            queue_traces[request_key] = {
                "agent_index": index,
                "batch_id": int(record["batch_id"]),
                "batch_ready_time_us": int(record["batch_ready_time_us"]),
                "batch_start_time_us": int(record["batch_start_time_us"]),
                "total_token_count": total_token_count,
                "cache_hit_token_count": cache_hit_token_count,
                "source_name": source_name,
            }
            batch_id = int(record["batch_id"])
            batch = agent_batches[index].setdefault(batch_id, {
                "batch_ready_time_us": int(record["batch_ready_time_us"]),
                "batch_start_time_us": int(record["batch_start_time_us"]),
                "request_ids": [],
            })
            batch["request_ids"].append(request_id)

actual_cache_hit_tokens = sum(
    hit_tokens
    for values in biz_cache_values.values()
    for hit_tokens, _ in values)
actual_cache_miss_tokens = sum(
    miss_tokens
    for values in biz_cache_values.values()
    for _, miss_tokens in values)
if (actual_cache_hit_tokens != stats["cache_hit_tokens"] or
        actual_cache_miss_tokens != stats["cache_miss_tokens"]):
    raise RuntimeError("per-request cache stats do not match master totals")

for index, batches in agent_batches.items():
    for batch in batches.values():
        biz_name = agent_biz_names[index]
        batch_size = len(batch["request_ids"])
        biz_batch_sizes[biz_name].append(batch_size)
        source_names = {
            request_sources_by_biz[biz_name][request_id]
            for request_id in batch["request_ids"]
        }
        for source_name in source_names:
            source_batch_sizes[biz_name][source_name].append(batch_size)
        execution_latencies = [
            float(request_timings[(agent_biz_names[index], request_id)]
                  ["execution_latency_ms"])
            for request_id in batch["request_ids"]
        ]
        batch["execution_latency_ms"] = (
            sum(execution_latencies) / len(execution_latencies))
    agent_batches[index] = sorted(
        batches.values(), key=lambda batch: batch["batch_start_time_us"])

batch_timer_wakeup_delays = []
batch_availability_gaps = []
for batches in agent_batches.values():
    for current, following in zip(batches, batches[1:]):
        deadline_time_us = (
            current["batch_start_time_us"] +
            current["execution_latency_ms"] / time_compression * 1000.0)
        availability_gap_ms = max(
            0.0,
            (min(following["batch_ready_time_us"],
                 following["batch_start_time_us"]) - deadline_time_us) /
                1000.0 * time_compression)
        timer_wakeup_delay_ms = max(
            0.0,
            (following["batch_start_time_us"] -
             max(deadline_time_us, following["batch_ready_time_us"])) /
                1000.0 * time_compression)
        batch_availability_gaps.append(availability_gap_ms)
        batch_timer_wakeup_delays.append(timer_wakeup_delay_ms)

schedule_to_start = []
timing_to_trace = []
execution_work_to_start = []
ready_execution_work_to_start = []
future_ready_execution_work = []
non_execution_gaps = []
timer_wakeup_delays = []
batch_availability_gaps_to_start = []
unexplained_gaps = []
underprediction_decomposition = []
seen_replays = set()
optimization_loop_counts = []
optimization_stop_reasons = {}
cost_effective_config_keys = (
    "lambda", "alpha", "alpha_decay", "min_alpha", "max_loop_count",
    "optimization_time_budget_us", "parallelism",
    "latency_weight_concavity", "latency_token_normalization_power",
    "initial_weight_seed",
    "discrete_check_interval", "discrete_gap_threshold",
    "discrete_repair_alpha_decay",
)
flexlb_effective_config_keys = (
    "predictor_type", "compute_token_cost", "cache_hit_token_cost",
    "use_reported_load", "score_tie_random_enabled",
    "score_tie_threshold_pct", "score_tie_threshold_ms",
    "hotspot_multiplier", "imbalance_multiplier",
)
effective_config_keys = {
    "cost": cost_effective_config_keys,
    "flexlb_baseline": flexlb_effective_config_keys,
}.get(scheduler_name, ())
effective_config_counts = {}
for index in range(client_count):
    path = os.path.join(
        test_dir, "navi_sched_simulate-0.1.0", "runtime",
        f"client_{index}", "traffic_schedule_replay.jsonl")
    if not os.path.exists(path):
        continue
    with open(path, encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            replay = json.loads(line)
            replay_biz_name = replay["biz_name"]
            replay_key = (
                replay_biz_name,
                replay["schedule_time_us"],
                tuple(item["request"]["request_id"]
                      for item in replay["requests"]),
            )
            if replay_key in seen_replays:
                continue
            seen_replays.add(replay_key)
            scheduler_info = replay.get("scheduler_info", {})
            if effective_config_keys and scheduler_info:
                effective_config = {
                    key: scheduler_info[key]
                    for key in effective_config_keys
                    if key in scheduler_info
                }
                config_key = json.dumps(
                    effective_config, sort_keys=True, separators=(",", ":"))
                effective_config_counts[config_key] = (
                    effective_config_counts.get(config_key, 0) + 1)
            for run in scheduler_info.get("optimization_runs", []):
                actual_loop_count = run.get("actual_loop_count")
                if actual_loop_count is not None:
                    optimization_loop_counts.append(
                        int(actual_loop_count))
                stop_reason = run.get("stop_reason")
                if stop_reason:
                    optimization_stop_reasons[stop_reason] = (
                        optimization_stop_reasons.get(stop_reason, 0) + 1)
            nodes = {}
            for node in replay["nodes"]:
                for address in (
                        node.get("collector_address"),
                        node.get("server_address")):
                    if address:
                        nodes[address] = node
            for decision in replay["decisions"]:
                request_id = decision["request_id"]
                request_key = (replay_biz_name, request_id)
                trace = queue_traces.get(request_key)
                timing = request_timings.get(request_key)
                target = decision.get("target", {})
                node = (nodes.get(target.get("agent_address")) or
                        nodes.get(target.get("address")))
                if not trace or not timing or not node:
                    continue
                predicted_ms = float(
                    node["queue_total_predicted_latency_ms"])
                schedule_time_us = int(replay["schedule_time_us"])
                schedule_wait_ms = max(
                    0.0,
                    (trace["batch_start_time_us"] - schedule_time_us) /
                    1000.0 * time_compression)
                queue_wait_ms = max(
                    0.0,
                    (trace["batch_start_time_us"] -
                     trace["batch_ready_time_us"]) /
                    1000.0 * time_compression)
                actual_execution_ms = 0.0
                ready_execution_ms = 0.0
                future_ready_execution_ms = 0.0
                timer_wakeup_delay_ms = 0.0
                batch_availability_gap_ms = 0.0
                batches = agent_batches[trace["agent_index"]]
                for batch_index, batch in enumerate(batches):
                    batch_start_time_us = batch["batch_start_time_us"]
                    if batch_start_time_us >= trace["batch_start_time_us"]:
                        break
                    batch_finish_time_us = (
                        batch_start_time_us +
                        batch["execution_latency_ms"] /
                        time_compression * 1000.0)
                    overlap_us = max(
                        0.0,
                        min(batch_finish_time_us,
                            trace["batch_start_time_us"]) -
                        max(batch_start_time_us, schedule_time_us))
                    actual_execution_ms += (
                        overlap_us / 1000.0 * time_compression)
                    if batch["batch_ready_time_us"] <= schedule_time_us:
                        ready_execution_ms += (
                            overlap_us / 1000.0 * time_compression)
                    else:
                        future_ready_execution_ms += (
                            overlap_us / 1000.0 * time_compression)
                    if batch_index + 1 >= len(batches):
                        continue
                    following = batches[batch_index + 1]
                    availability_overlap_us = max(
                        0.0,
                        min(following["batch_ready_time_us"],
                            following["batch_start_time_us"],
                            trace["batch_start_time_us"]) -
                        max(batch_finish_time_us, schedule_time_us))
                    timer_overlap_us = max(
                        0.0,
                        min(following["batch_start_time_us"],
                            trace["batch_start_time_us"]) -
                        max(batch_finish_time_us,
                            following["batch_ready_time_us"],
                            schedule_time_us))
                    batch_availability_gap_ms += (
                        availability_overlap_us / 1000.0 * time_compression)
                    timer_wakeup_delay_ms += (
                        timer_overlap_us / 1000.0 * time_compression)
                schedule_to_start.append((predicted_ms, schedule_wait_ms))
                timing_to_trace.append((
                    float(timing["queue_wait_ms"]), queue_wait_ms))
                execution_work_to_start.append((
                    predicted_ms, actual_execution_ms))
                ready_execution_work_to_start.append((
                    predicted_ms, ready_execution_ms))
                future_ready_execution_work.append(
                    future_ready_execution_ms)
                non_execution_gaps.append(
                    schedule_wait_ms - actual_execution_ms)
                timer_wakeup_delays.append(timer_wakeup_delay_ms)
                batch_availability_gaps_to_start.append(
                    batch_availability_gap_ms)
                unexplained_gaps.append(
                    schedule_wait_ms - actual_execution_ms -
                    timer_wakeup_delay_ms - batch_availability_gap_ms)
                underprediction_decomposition.append({
                    "total": schedule_wait_ms - predicted_ms,
                    "execution_model": actual_execution_ms - predicted_ms,
                    "timer_wakeup": timer_wakeup_delay_ms,
                    "batch_availability": batch_availability_gap_ms,
                    "unexplained": (
                        schedule_wait_ms - actual_execution_ms -
                        timer_wakeup_delay_ms - batch_availability_gap_ms),
                })

if scheduler_name == "cost" and not schedule_to_start:
    raise RuntimeError("traffic replay produced no queue prediction samples")
if scheduler_metrics:
    scheduler_metrics["sampled_loop_count_distribution"] = (
        scalar_distribution(optimization_loop_counts)
        if optimization_loop_counts else None
    )
    scheduler_metrics["sampled_stop_reasons"] = optimization_stop_reasons

queue_prediction_accuracy = None
queue_timing_consistency = None
if schedule_to_start:
    queue_prediction_accuracy = {
        "replay_context_count": len(seen_replays),
        "schedule_to_batch_start": comparison(schedule_to_start),
        "execution_work_to_batch_start": comparison(execution_work_to_start),
        "ready_execution_work_to_batch_start": comparison(
            ready_execution_work_to_start),
        "future_ready_execution_work_to_batch_start": distribution(
            future_ready_execution_work),
        "non_execution_gap_to_batch_start": distribution(non_execution_gaps),
        "timer_wakeup_delay_to_batch_start": distribution(timer_wakeup_delays),
        "batch_availability_gap_to_batch_start": distribution(
            batch_availability_gaps_to_start),
        "unexplained_gap_to_batch_start": distribution(unexplained_gaps),
        "batch_timer_wakeup_delay": optional_distribution(
            batch_timer_wakeup_delays),
        "batch_availability_gap": optional_distribution(
            batch_availability_gaps),
        "mean_underprediction_decomposition_ms": {
            "total": sum(
                item["total"] for item in underprediction_decomposition) /
                len(underprediction_decomposition),
            "execution_model": sum(
                item["execution_model"]
                for item in underprediction_decomposition) /
                len(underprediction_decomposition),
            "ready_queue": sum(
                actual - predicted
                for predicted, actual in ready_execution_work_to_start) /
                len(ready_execution_work_to_start),
            "future_ready_execution": sum(future_ready_execution_work) /
                len(future_ready_execution_work),
            "timer_wakeup": sum(
                item["timer_wakeup"]
                for item in underprediction_decomposition) /
                len(underprediction_decomposition),
            "batch_availability": sum(
                item["batch_availability"]
                for item in underprediction_decomposition) /
                len(underprediction_decomposition),
            "unexplained": sum(
                item["unexplained"]
                for item in underprediction_decomposition) /
                len(underprediction_decomposition),
            "identity_max_absolute_error": max(
                abs(item["total"] - item["execution_model"] -
                    item["timer_wakeup"] - item["batch_availability"] -
                    item["unexplained"])
                for item in underprediction_decomposition),
        },
    }
    queue_timing_consistency = comparison(timing_to_trace)
effective_scheduler_config = None
effective_scheduler_config_sample_count = 0
if effective_config_keys:
    if len(effective_config_counts) != 1:
        raise RuntimeError(
            f"{scheduler_name} effective config is missing or inconsistent: "
            f"{effective_config_counts}")
    config_key, effective_scheduler_config_sample_count = next(
        iter(effective_config_counts.items()))
    effective_scheduler_config = json.loads(config_key)
    def config_value_matches(requested, effective):
        if (isinstance(requested, (int, float)) and
                not isinstance(requested, bool) and
                isinstance(effective, (int, float)) and
                not isinstance(effective, bool)):
            return math.isclose(
                float(requested), float(effective),
                rel_tol=1e-12, abs_tol=1e-12)
        return requested == effective

    config_mismatches = {
        key: {
            "requested": scheduler_config[key],
            "effective": effective_scheduler_config.get(key),
        }
        for key in effective_config_keys
        if key in scheduler_config and
        not config_value_matches(
            scheduler_config[key], effective_scheduler_config.get(key))
    }
    if config_mismatches:
        raise RuntimeError(
            f"{scheduler_name} requested/effective config mismatch: "
            f"{config_mismatches}")
logical_replay_duration_ms = max(
    requests[-1][0] - requests[0][0] for requests in case_requests)
biz_metrics = []
for case_index, (case, requests) in enumerate(
        zip(traffic_cases, case_requests)):
    biz_name = case["biz_name"]
    agents = [
        agent for agent in execution_agents
        if agent["biz_name"] == biz_name
    ]
    utilizations = [
        (agent["logical_busy_ms"] / agent["logical_elapsed_ms"]
         if agent["logical_elapsed_ms"] else 0.0)
        for agent in agents
    ]
    biz_metrics.append({
        "biz_name": biz_name,
        "service_name": case["service_name"],
        "request_count": len(requests),
        "agent_count": len(agents),
        "client_count": case["client_count"],
        "logical_replay_duration_ms": requests[-1][0] - requests[0][0],
        "prompt_token_distribution": scalar_distribution(
            case_prompt_token_counts[case_index]),
        "actual_cache": cache_metrics(biz_cache_values[biz_name]),
        "batch_size_distribution": scalar_distribution(
            biz_batch_sizes[biz_name]),
        "traffic_sources": [
            {
                "name": name,
                "request_count": len(values),
                "prompt_token_distribution": scalar_distribution(values),
                "request_share": len(values) / len(requests),
                "actual_cache": cache_metrics(
                    source_cache_values[biz_name][name]),
                "batch_size_distribution": scalar_distribution(
                    source_batch_sizes[biz_name][name]),
                "latency_distribution": {
                    latency_name: distribution(latency_values)
                    for latency_name, latency_values in
                    source_timings[biz_name][name].items()
                },
            }
            for name, values in
            case_source_prompt_token_counts[case_index].items()
        ],
        "latency_distribution": {
            name: distribution(values)
            for name, values in biz_timings[biz_name].items()
        },
        "average_utilization": sum(utilizations) / len(utilizations),
        "min_utilization": min(utilizations),
        "max_utilization": max(utilizations),
        "utilization_stddev": (
            sum(
                (value - sum(utilizations) / len(utilizations)) ** 2
                for value in utilizations
            ) / len(utilizations)
        ) ** 0.5,
        "cost_scheduler_loss": (
            scalar_distribution(biz_schedule_losses[biz_name])
            if biz_schedule_losses[biz_name] else None
        ),
    })

result = {
    "experiment_launch_config": experiment_config,
    "request_count": len(timings["client_total_latency_ms"]),
    "agent_readiness": "all_agents_ready",
    "scheduler_name": scheduler_name,
    "scheduler_config": scheduler_config,
    "effective_scheduler_config": effective_scheduler_config,
    "effective_scheduler_config_sample_count": (
        effective_scheduler_config_sample_count),
    "schedule_wait_count": schedule_wait_count,
    "schedule_wait_time_ms": schedule_wait_time_ms,
    "random_seed": random_seed,
    "traffic_mode": traffic_cases[0].get("traffic_mode", "independent"),
    "prompt_token_distribution": scalar_distribution([
        value
        for values in case_prompt_token_counts
        for value in values
    ]),
    "latency_time_basis": "uncompressed_ms",
    "replay_execution_time_compression": time_compression,
    "wall_duration_ms": wall_duration_ms,
    "logical_replay_duration_ms": logical_replay_duration_ms,
    "logical_drain_time_ms": max(
        0.0,
        wall_duration_ms * time_compression -
            logical_replay_duration_ms),
    "biz_metrics": biz_metrics,
    "cluster_cache_capacity_bytes": cluster_cache_capacity_bytes,
    "agent_cache_capacity_bytes": agent_cache_capacity_bytes,
    "block_cache_bytes": block_cache_bytes,
    "latency_distribution": {
        name: distribution(values) for name, values in timings.items()
    },
    "cache_hit_tokens": stats["cache_hit_tokens"],
    "cache_miss_tokens": stats["cache_miss_tokens"],
    "cache_hit_rate": stats["cache_hit_tokens"] /
        (stats["cache_hit_tokens"] + stats["cache_miss_tokens"]),
    "cache_evicted_block_count": cache_evicted_block_count,
    "cache_agents": cache_agents,
    "execution_agents": execution_agents,
    "latency_model_fit": {
        "batch_count": total_fit_batch_count,
        "prediction_bias_ms": prediction_bias_ms,
        "prediction_mae_ms": prediction_mae_ms,
        "prediction_rmse_ms": prediction_rmse_ms,
        "prediction_mean_relative_absolute_error": prediction_relative_mae,
        "agents": model_fit,
    },
    "latency_models": stats["latency_models"],
    "decision_prediction_accuracy": decision_prediction_accuracy,
    "queue_prediction_accuracy": queue_prediction_accuracy,
    "queue_timing_consistency": queue_timing_consistency,
    "cost_scheduler_loss": cost_scheduler_loss,
    "scheduler_metrics": scheduler_metrics,
}
summary_path = os.path.join(test_dir, "traffic_replay_summary.json")
with open(summary_path, "w", encoding="utf-8") as output:
    json.dump(result, output, indent=2)
console_result = dict(result)
if cost_scheduler_loss:
    console_result["cost_scheduler_loss"] = {
        "distribution": cost_scheduler_loss["distribution"],
        "records": os.path.join(test_dir, "traffic_replay_summary.json"),
    }
print(json.dumps(console_result, indent=2))
PY
