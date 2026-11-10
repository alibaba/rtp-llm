#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  update_scheduler_config.sh --set key=value ...
  update_scheduler_config.sh --hardware HARDWARE --set key=value ...
  update_scheduler_config.sh --get
  update_scheduler_config.sh --patch-json '{"key":value,...}'

默认请求本机 Master 的 scheduler-config 接口：127.0.0.1:7001。
Master 字段在 Master 内生效，Agent 字段通过已有的 Master->Agent 同步链路下发。
如需特殊环境，可用 --host/--port 或 --url 覆盖默认地址。

 Options:
  --host HOST          Master HTTP 地址，默认 127.0.0.1
  --port PORT          Master HTTP 端口，默认 7001
  --url URL            完整接口地址，覆盖 --host/--port
  --timeout SECONDS    curl 超时时间，默认 15 秒
  --hardware HARDWARE  只更新 fingerprint 中 hardware 完全匹配的 Agent
  --get                只查询当前生效配置，不修改
  --patch-json JSON    传入一个 JSON 对象；可与 --set 合并
  --set KEY=VALUE      修改字段，可重复；VALUE 使用 JSON 写法，例如 0.01、4、true、"text"
  -h, --help           显示本帮助

可配置字段（字段名保持英文，含义如下）：
  lambda                              调度代价中延迟代价的权重系数。
  alpha                               优化器初始步长，必须大于 0。
  alpha_decay                        优化器步长衰减系数，范围为 (0, 1]。
  min_alpha                          优化器允许的最小步长。
  min_loop_count                     最低梯度迭代轮数（优先于优化时间预算，受最大轮数限制）。
  max_loop_count                     单次调度决策允许的最大优化迭代次数。
  optimization_time_budget_us        单次优化的 CPU 时间预算，-1 表示不限制。
  parallelism                        优化器并行度，0 表示自动选择。
  latency_weight_concavity            延迟代价权重的凹度，0 表示关闭。
  latency_token_normalization_power   token 数量归一化指数，范围为 [0, 1]。
  initial_weight_seed                 优化器权重初始化使用的确定性随机种子。
  discrete_check_interval             检查离散调度结果的迭代间隔。
  discrete_gap_threshold              触发离散结果修正的相对差距阈值。
  discrete_repair_alpha_decay         离散结果修正阶段的步长衰减系数。
  max_wait_schedule_count             一个调度批次最多等待的请求数。
  max_wait_schedule_time_ms           调度批次最多等待的时间，单位毫秒。
  queue_overload_start_cycle_multiplier  开始过载限流的排队周期倍数，0 表示关闭。
  queue_overload_full_cycle_multiplier   达到满限流的排队周期倍数。
  overload_control_strategy           execution_utilization（默认）、latency（平均积压）、min_latency（最小可信积压）或 post_cost_latency（选址后按目标组排队）。
  execution_utilization_global_scaling 整集群聚合 running/waiting 后放缩，默认 false。
  execution_utilization_target        等效 Prefill 执行占比目标，默认 1.10（110%）。
  execution_utilization_kp            PI 比例系数，默认 0.05。
  execution_utilization_ki_per_second PI 每秒积分系数，默认 0.01。
  admission_timeout_ms                Master 准入请求超时时间，单位毫秒。
  batch_timeout_ms                    Agent 凑批等待超时时间，单位毫秒。
  closed_batch_keep_ms                已完成 batch 状态保留时间，单位毫秒。
  latency_fit_batch_size              累积多少个新样本后触发一次在线拟合。
  latency_fit_interval                在线拟合的额外触发间隔，0 表示使用默认批量触发。
  latency_learning_rate               延迟模型 Adam 基础学习率。
  latency_online_learning_rate_scale  在线更新时相对基础学习率的倍率。
  latency_adaptive_learning_rate      在线学习率控制器对象，用 --patch-json 设置：
    enabled                           是否启用控制器，true/false。
    mode                              adaptive（默认）、error_rate 或 error_rate_hinge（超过阈值才加速）。
    error_rate_base_gain              误差率到基础学习率倍率的增益，必须大于 0。
    error_rate_base_power             基础误差率的幂次，1 为线性，2 为平方，范围 1-4。
    error_rate_base_threshold         error_rate_hinge 的误差率阈值，范围 0-1，默认 0.1。
    error_rate_normalizer_floor_ms    误差率分母下限，单位毫秒，默认 100。
    error_rate_minimum_streak          error_rate_hinge 连续同方向大误差达到该次数才加速，范围 1-64，默认 1。
    base_scale_minimum/maximum        基础学习率倍率的下限和上限。
    residual_learning_rate_minimum/maximum  residual 学习率范围；两种 error_rate 模式要求两者相等。
  latency_error_clip_ms             Absolute training error clip in ms; default 5000, 0 disables.
  latency_current_sample_weight      当前在线样本在 loss 中的相对权重。
  latency_nonlinear_scale             非线性延迟项中的 K 参数，必须为正。
  queue_learning_rate                 queue 模型学习率。
  latency_check_gradient              是否启用延迟梯度额外校验，true/false。
  latency_cold_start_fit_epochs       冷启动全历史拟合的 epoch 数。
  latency_history_limit               保留的延迟历史样本总数上限。
  latency_history_bucket_capacity     每个历史桶最多保留的样本数。
  latency_persist_interval             按 revision 计的模型落盘间隔。
  latency_persist_interval_ms          模型落盘的最小时间间隔，单位毫秒。
  latency_warmup_sample_count         warmup 阶段跳过、不参与拟合的初始样本数。
  latency_model_trust_min_revision     模型变为 trusted 所需的最小本地 revision。
  latency_zero_anchor_max_ms           零点校验允许的最大延迟，单位毫秒。
  latency_zero_anchor_min_samples      零点校验所需的最少低负载样本数。
  latency_validation_enabled           是否启用候选模型校验，true/false。
  residual_learning_rate              residual 快速补偿模型学习率；0 时关闭预测和更新。
  residual_update_error_floor_ms       单次 residual 更新的绝对误差上限，单位毫秒。
  residual_update_error_ratio          单次 residual 更新的相对误差上限。
  residual_idle_decay_start_ms        无流量多久后开始衰减 residual，单位毫秒。
  residual_idle_decay_end_ms          无流量多久后 residual 衰减到 0，单位毫秒。
  time_compression                     测试时钟压缩倍率，线上通常为 1。
EOF
}

host="${NAVI_MASTER_HOST:-127.0.0.1}"
port="${NAVI_MASTER_HTTP_PORT:-7001}"
url=""
timeout="15"
hardware=""
get_only=0
patch_json="{}"
sets=()

while (($# > 0)); do
    case "$1" in
    --host)
        (($# >= 2)) || { echo "--host requires a value" >&2; exit 2; }
        host="$2"
        shift 2
        ;;
    --port)
        (($# >= 2)) || { echo "--port requires a value" >&2; exit 2; }
        port="$2"
        shift 2
        ;;
    --url)
        (($# >= 2)) || { echo "--url requires a value" >&2; exit 2; }
        url="$2"
        shift 2
        ;;
    --timeout)
        (($# >= 2)) || { echo "--timeout requires a value" >&2; exit 2; }
        timeout="$2"
        shift 2
        ;;
    --hardware)
        (($# >= 2)) || { echo "--hardware requires a value" >&2; exit 2; }
        hardware="$2"
        [[ -n "$hardware" ]] || { echo "--hardware must not be empty" >&2; exit 2; }
        shift 2
        ;;
    --get)
        get_only=1
        shift
        ;;
    --patch-json)
        (($# >= 2)) || { echo "--patch-json requires a value" >&2; exit 2; }
        patch_json="$2"
        shift 2
        ;;
    --set)
        (($# >= 2)) || { echo "--set requires KEY=VALUE" >&2; exit 2; }
        sets+=("$2")
        shift 2
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "unknown option: $1" >&2
        usage
        exit 2
        ;;
    esac
done

if [[ -z "$url" ]]; then
    url="http://${host}:${port}/scheduler-config"
fi

if ((get_only)); then
    ((${#sets[@]} == 0)) || {
        echo "--get cannot be combined with --set" >&2
        exit 2
    }
    [[ "$patch_json" == "{}" ]] || {
        echo "--get cannot be combined with --patch-json" >&2
        exit 2
    }
    request_body="$(python3 - "$hardware" <<'PY'
import json
import sys

payload = {}
if sys.argv[1]:
    payload["hardware"] = sys.argv[1]
print(json.dumps(payload, separators=(",", ":")))
PY
)"
else
    request_body="$(python3 - "$hardware" "$patch_json" "${sets[@]}" <<'PY'
import json
import sys

try:
    patch = json.loads(sys.argv[2])
except json.JSONDecodeError as error:
    raise SystemExit(f"--patch-json must be a JSON object: {error}")
if not isinstance(patch, dict):
    raise SystemExit("--patch-json must be a JSON object")

for assignment in sys.argv[3:]:
    if "=" not in assignment:
        raise SystemExit(f"--set requires KEY=VALUE: {assignment}")
    key, value = assignment.split("=", 1)
    if not key:
        raise SystemExit("--set key must not be empty")
    try:
        patch[key] = json.loads(value)
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"value for {key} must be JSON (quote strings): {error}"
        )

if not patch:
    raise SystemExit("update requires --set or --patch-json")
payload = {
    "update": True,
    "configJson": json.dumps(patch, separators=(",", ":")),
}
if sys.argv[1]:
    payload["hardware"] = sys.argv[1]
print(json.dumps(payload, separators=(",", ":")))
PY
)"
fi

response_file="$(mktemp "${TMPDIR:-/tmp}/navi-scheduler-config.XXXXXX")"
trap 'rm -f "$response_file"' EXIT
set +e
if command -v curl >/dev/null 2>&1; then
    http_status="$(curl --silent --show-error \
        --max-time "$timeout" \
        --header 'Content-Type: application/json' \
        --header 'Accept: application/json' \
        --request POST \
        --data-binary "$request_body" \
        --output "$response_file" \
        --write-out '%{http_code}' \
        "$url")"
    request_status=$?
else
    http_status="$(python3 - "$url" "$request_body" "$timeout" "$response_file" <<'PY'
import sys
import urllib.error
import urllib.request
from pathlib import Path

url, body, timeout, output_path = sys.argv[1:]
request = urllib.request.Request(
    url,
    data=body.encode("utf-8"),
    headers={
        "Accept": "application/json",
        "Content-Type": "application/json",
    },
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=float(timeout)) as response:
        Path(output_path).write_bytes(response.read())
        print(response.status)
except urllib.error.HTTPError as error:
    Path(output_path).write_bytes(error.read())
    print(error.code)
except Exception as error:
    print(error, file=sys.stderr)
    raise SystemExit(1)
PY
    )"
    request_status=$?
fi
set -e
if ((request_status != 0)); then
    cat "$response_file" >&2
    echo "scheduler-config request failed (transport exit ${request_status})" >&2
    exit "$request_status"
fi
if [[ ! "$http_status" =~ ^2[0-9][0-9]$ ]]; then
    cat "$response_file" >&2
    echo "scheduler-config returned HTTP ${http_status}" >&2
    exit 1
fi
response="$(cat "$response_file")"
printf '%s\n' "$response"

python3 - "$response" <<'PY'
import json
import sys

try:
    response = json.loads(sys.argv[1])
except json.JSONDecodeError as error:
    raise SystemExit(f"scheduler-config returned invalid JSON: {error}")
if response.get("accepted") is not True:
    raise SystemExit(2)
PY
