# Master `/batch_schedule` API

一次性向 caller 分配 N 个 worker 槽位。

## Endpoint

```
POST /rtp_llm/batch_schedule
Content-Type: application/json
```

**Scope**：本接口**只支持单 role 单阶段批量散开**。多 role 部署（解耦 PD / VL）请用 `/schedule` 并发派发，不要用此接口。

**调度策略**：当前仅支持 `ROUND_ROBIN`。target role 必须在 `FLEXLB_CONFIG` 里配为 `ROUND_ROBIN`。其他策略（如 `SHORTEST_TTFT`）调本接口会返 `INVALID_REQUEST`，未来版本会扩展，wire 协议已向前兼容（见 §Request 的 `sub_requests` 字段）。

## Request

**最简形态（推荐起步）**：

```json
{
  "batch_count": 3
}
```

**完整形态（含可选字段）**：

```json
{
  "batch_count": 3,
  "sub_requests": [
    {"request_id": 1001, "seq_len": 1280},
    {"request_id": 1002, "seq_len": 980},
    {"request_id": 1003, "seq_len": 2200}
  ]
}
```

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `batch_count` | ✅ | int | 申请的 worker 槽位数。范围 `[1, BATCH_SCHEDULE_MAX_COUNT]`，超出返 `INVALID_REQUEST` |
| `sub_requests` | ❌ | array | 可选；提供时长度必须 == `batch_count`。**当前版本接收并校验长度，但策略层不读**（RR 不需要 per-request 信息），为未来支持 SHORTEST_TTFT 等负载感知策略保留 |

**`sub_requests[]` 单元素字段**（沿用 `/schedule` 的 `Request` DTO 形状，全部可选）：

| 字段 | 类型 | 当前是否被读 | 未来用途 |
|---|---|---|---|
| `request_id` | int64 | ❌ | SHORTEST_TTFT batch 启用后用作 `localTaskMap` key（届时变必填）|
| `seq_len` | int64 | ❌ | SHORTEST_TTFT batch 启用后用于估算 prefill 时间，让长短 prompt 散到不同 worker |
| `block_cache_keys` | int64[] | ❌ | 更晚的 cache-aware batch 策略才会用 |
| `generate_timeout` | int64 | ❌ | SLO-aware 策略保留 |
| `request_time_ms` | int64 | ❌ | caller 侧时间戳，监控保留 |

caller 现在可以**选择只传 `batch_count`**（最简，RR 也不需要更多）；也可以**提前传 `sub_requests`**（forward-compatible，未来策略升级时无需 caller 配合改造）。

## Response

### 成功（HTTP 200）

```json
{
  "success": true,
  "code": 200,
  "server_status": [
    {"server_ip": "10.1.2.1", "http_port": 28100, "grpc_port": 28101},
    {"server_ip": "10.1.2.2", "http_port": 28100, "grpc_port": 28101},
    {"server_ip": "10.1.2.1", "http_port": 28100, "grpc_port": 28101}
  ],
  "real_master_host": "10.1.0.1:7001"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `success` | bool | 成功固定 `true` |
| `code` | int | 成功固定 `200` |
| `server_status[]` | array | 长度 = `batch_count`，按 RR cursor 顺序排列。如果 caller 传了 `sub_requests`，则 `server_status[i]` 对应 `sub_requests[i]` |
| `server_status[].server_ip` | string | worker IP |
| `server_status[].http_port` | int | worker Python 层端口（tokenize / OpenAI 兼容 / 转 C++ engine）|
| `server_status[].grpc_port` | int | worker C++ engine gRPC 直连端口（= `http_port + 1`）|
| `real_master_host` | string | 当前 master `ip:port`，方便 caller 排障 |

caller 调 worker 时 `http_port` 和 `grpc_port` 选其一即可：调 `http_port` 走 Python 层（适合传 prompt 字符串）、调 `grpc_port` 直进 C++ engine（适合自己 tokenize 过、追求最低延迟）。

### 失败（HTTP 500）

```json
{
  "success": false,
  "code": 8406,
  "error_message": "batch_count must be in [1, 1000]",
  "real_master_host": "10.1.0.1:7001"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `success` | bool | 失败固定 `false` |
| `code` | int | 错误码，详见下表 |
| `error_message` | string | 错误原因描述 |
| `real_master_host` | string | 同成功响应 |

失败响应**不输出** `server_status` 字段（避免 caller 误解析 null 数组）。

## 错误码

| 场景 | code | error_message |
|---|---|---|
| `batch_count` 缺失 / ≤ 0 / 超过 `BATCH_SCHEDULE_MAX_COUNT` | `INVALID_REQUEST` (8406) | `"batch_count must be in [1, <MAX>]"` |
| 提供 `sub_requests` 但 `length != batch_count` | `INVALID_REQUEST` (8406) | `"sub_requests length <M> != batch_count <N>"` |
| 0 个 role 注册（master 未就绪 / 配置缺失） | `NO_AVAILABLE_WORKER` (8400) | `"master not ready or MODEL_SERVICE_CONFIG missing"` |
| 多个 role 都有注册 worker（多 role 部署） | `INVALID_REQUEST` (8406) | `"batch_schedule only supports single-role deployments; multi-stage deployments (disaggregated PD / VL) should use /schedule per request. Detected roles: [...]"` |
| 单 role 但该 role 无 alive worker | role-specific（`NO_PREFILL_WORKER` 8402 / `NO_DECODE_WORKER` 8403 / `NO_PDFUSION_WORKER` 8404 / `NO_VIT_WORKER` 8405） | role 标准错误信息 |
| 该 role 策略不支持 batch（没配 `ROUND_ROBIN`） | `INVALID_REQUEST` (8406) | `"strategy for role <ROLE> does not support batch_schedule"` |
| Master 不可达（slave 收到时上游 master 没响应） | `NO_AVAILABLE_WORKER` (8400) | `"master unreachable"` |

## 行为说明

### Worker 分配机制（当前 RR 实现）

- master 后台每 20ms gRPC 心跳一次，维护每个 role 的 worker 状态
- 收到 `batch_count = N` 时：
  1. 校验 `batch_count` 范围 + `sub_requests` 长度（若提供）
  2. 调用 `EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getRoleTypeList()`（**和 `/schedule` 用同一份数据源**）获取当前有注册 worker 的 role 列表
     - 0 个 → `NO_AVAILABLE_WORKER`
     - 多个 → `INVALID_REQUEST`，提示改用 `/schedule`
     - 恰好 1 个 → 进入下一步
  3. 该 role 对应的 LoadBalancer 不支持 batch（没配 `ROUND_ROBIN`）→ `INVALID_REQUEST`
  4. 调 `RoundRobinLoadBalancer.selectBatch(N, role)`：一次原子 op 占连续 N 个 cursor 号，从 role 的 alive worker 列表里按 cursor 顺序取 N 台
     - alive 列表为空 → 返回 role-specific 错误（例：`NO_PDFUSION_WORKER`）
     - `N > alive 数量` 时 cursor 自然 wrap（例：`N=20` 对 10 台 worker → 每台 2 个）

### Master 不记账（当前 RR 行为）

caller 调本接口拿到 worker 后**调用 worker 的过程 master 完全不感知**：

- 无 `localTaskMap` 记账、无心跳对账、无 lost detection、无 rollback
- caller 与 worker 之间的失败语义全交给 caller：超时/失败时 caller 自己重试或换 worker
- 因此 batch_schedule 路径上 cursor 推进的副作用之外没有任何残留状态

> 未来 SHORTEST_TTFT batch 启用后，master 会记账（写 `localTaskMap`），但 caller 端语义不变（依然是"返回 N 个 target，caller 自己调"）。caller 撒谎 / 不调 worker 的容错由 worker 心跳的 `markTasksAsLost` 自愈，~20-40ms。

### 顺序语义

`server_status` 数组顺序：
- 不传 `sub_requests`：按 RR cursor 依次推进得到的 worker 顺序
- 传 `sub_requests`：`server_status[i]` ↔ `sub_requests[i]`（按下标对位）

caller 想保留 RR 均匀分布特性，建议把自己手头的 N 个业务请求按下标对位派发到 `server_status[i]`。也可以打乱、跳过、并行派发——master 不强制，只保证返回的这 N 个 slot 在 cursor 序上是连续的。

### slave 转发到 master

多 master + ZooKeeper 选主部署下，caller 调到的实例如果不是 leader，会自动 HTTP 转发到 leader。转发失败时返回 `NO_AVAILABLE_WORKER`，**不 fallback 本地处理**（避免多 master 同时调度造成账本分裂）。

## curl 示例

**最简（推荐起步）**：

```bash
curl -X POST http://master:7001/rtp_llm/batch_schedule \
  -H 'Content-Type: application/json' \
  -d '{"batch_count": 3}'
```

**Forward-compatible（提前传可选字段，今天忽略，未来策略升级自动生效）**：

```bash
curl -X POST http://master:7001/rtp_llm/batch_schedule \
  -H 'Content-Type: application/json' \
  -d '{
    "batch_count": 3,
    "sub_requests": [
      {"request_id": 1001, "seq_len": 1280},
      {"request_id": 1002, "seq_len": 980},
      {"request_id": 1003, "seq_len": 2200}
    ]
  }'
```

## Python 示例

```python
import requests

def get_batch_workers(batch_count, master_url="http://master:7001", sub_requests=None):
    """
    sub_requests: 可选 list of dict，每个含 request_id + seq_len 等字段。
                  当前版本仅校验长度，未来 SHORTEST_TTFT 启用后会按 seq_len 散开长短 prompt。
    """
    payload = {"batch_count": batch_count}
    if sub_requests is not None:
        payload["sub_requests"] = sub_requests
    resp = requests.post(
        f"{master_url}/rtp_llm/batch_schedule",
        json=payload,
        timeout=2,
    ).json()
    if not resp["success"]:
        raise RuntimeError(f"batch_schedule failed: code={resp['code']}, msg={resp['error_message']}")
    return resp["server_status"]


def batch_dispatch(prompts, master_url):
    targets = get_batch_workers(len(prompts), master_url)
    # 按下标对位派发，保留 RR 均匀分布
    for prompt, target in zip(prompts, targets):
        # 走 Python 层（带 tokenize）
        requests.post(
            f"http://{target['server_ip']}:{target['http_port']}/generate",
            json={"prompt": prompt},
        )
        # 或直连 C++ engine（自己已 tokenize）
        # send_grpc(target['server_ip'], target['grpc_port'], token_ids=[...])


def batch_dispatch_forward_compatible(prompt_records, master_url):
    """prompt_records: list of {'request_id': int, 'seq_len': int, 'prompt': str}"""
    sub_requests = [
        {"request_id": r["request_id"], "seq_len": r["seq_len"]}
        for r in prompt_records
    ]
    targets = get_batch_workers(len(prompt_records), master_url, sub_requests=sub_requests)
    for record, target in zip(prompt_records, targets):
        requests.post(
            f"http://{target['server_ip']}:{target['http_port']}/generate",
            json={"prompt": record["prompt"], "request_id": record["request_id"]},
        )
```

## 部署侧

### FLEXLB_CONFIG

target role 配为 `ROUND_ROBIN`：

```json
{
  "loadBalanceStrategy": "ROUND_ROBIN",
  ...
}
```

或在按 role 粒度配置策略时把该 role 设为 `ROUND_ROBIN`。

### Env

| env | 默认 | 说明 |
|---|---|---|
| `BATCH_SCHEDULE_MAX_COUNT` | `1000` | `batch_count` 上限。超过返 `INVALID_REQUEST`，防 DoS |
| `HIPPO_ROLE` | 必填 | master 启动必需的任意非空字符串 |
| `MODEL_SERVICE_CONFIG` | 必填 | worker 集群地址配置（见 flexlb 公共文档）|
| `FLEXLB_SYNC_CONSISTENCY_CONFIG` | 可选 | 开 ZK 选主时设；不设则单实例 master |

## 设计取舍

本接口当前**故意做窄**：单 role + RR 策略 + 不记账。换取的是：

- 协议极简，caller 端最少只传一个 int
- master 零 in-transit 状态，cursor 自带 stampede 免疫
- 失败/重试语义全在 caller 侧，master 不参与

**未来扩展全部 non-breaking**（wire 已预留 `sub_requests` 可选字段）：

- **支持 SHORTEST_TTFT batch**（避热点 + 长短 prompt 自动散开）：caller 开始填 `sub_requests[].request_id` + `seq_len` 即生效，无需 wire 改动
- **支持 cache-aware batch**：caller 开始填 `sub_requests[].block_cache_keys` 即生效
- **支持 SLO-aware**：caller 开始填 `sub_requests[].generate_timeout` 即生效
- **支持多 role / 多阶段批量**：response shape 升级为嵌套，需独立 PR；本接口不动

## 相关代码

- `rtp_llm/flexlb/flexlb-api/.../HttpLoadBalanceServer.java::batchScheduleRequest`
- `rtp_llm/flexlb/flexlb-sync/.../scheduler/DefaultRouter.java::batchSchedule`
- `rtp_llm/flexlb/flexlb-sync/.../strategy/RoundRobinLoadBalancer.java::selectBatch`
- `rtp_llm/flexlb/flexlb-common/.../dao/loadbalance/BatchScheduleRequest.java`
- `rtp_llm/flexlb/flexlb-common/.../dao/loadbalance/BatchScheduleResponse.java`
- `rtp_llm/flexlb/flexlb-common/.../dao/loadbalance/BatchScheduleTarget.java`
- `rtp_llm/flexlb/flexlb-common/.../dao/loadbalance/Request.java`（`sub_requests[]` 元素 DTO）
