# Master `/batch_schedule` API

一次性向 caller 分配 N 个 worker 槽位。

## Endpoint

```
POST /rtp_llm/batch_schedule
Content-Type: application/json
```

**Scope**：本接口**只支持单 role 单阶段批量散开**。多 role 部署（解耦 PD / VL）请用 `/schedule` 并发派发，不要用此接口。目标 role 必须在 `FLEXLB_CONFIG` 里配为 `ROUND_ROBIN` 策略。

**调度行为**：固定 Round-Robin。**master 不做负载预测、不记账、不做心跳对账**。

## Request

```json
{
  "batch_count": 3
}
```

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `batch_count` | ✅ | int | 申请的 worker 槽位数。范围 `[1, BATCH_SCHEDULE_MAX_COUNT]`，超出返 `INVALID_REQUEST` |

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
| `server_status[]` | array | 长度 = `batch_count`，按 RR cursor 顺序排列 |
| `server_status[].server_ip` | string | worker IP |
| `server_status[].http_port` | int | worker Python 层端口（tokenize / OpenAI 兼容 / 转 C++ engine）|
| `server_status[].grpc_port` | int | worker C++ engine gRPC 直连端口（= `http_port + 1`） |
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
| 0 个 role 注册（master 未就绪 / 配置缺失） | `NO_AVAILABLE_WORKER` (8400) | `"master not ready or MODEL_SERVICE_CONFIG missing"` |
| 多个 role 都有注册 worker（多 role 部署） | `INVALID_REQUEST` (8406) | `"batch_schedule only supports single-role deployments; multi-stage deployments (disaggregated PD / VL) should use /schedule per request. Detected roles: [...]"` |
| 单 role 但该 role 无 alive worker | role-specific（`NO_PREFILL_WORKER` 8402 / `NO_DECODE_WORKER` 8403 / `NO_PDFUSION_WORKER` 8404 / `NO_VIT_WORKER` 8405） | role 标准错误信息 |
| 该 role 策略不支持 batch（没配 `ROUND_ROBIN`） | `INVALID_REQUEST` (8406) | `"strategy for role <ROLE> does not support batch_schedule"` |
| Master 不可达（slave 收到时上游 master 没响应） | `NO_AVAILABLE_WORKER` (8400) | `"master unreachable"` |

## 行为说明

### Worker 分配机制

- master 后台每 20ms gRPC 心跳一次，维护每个 role 的 worker 状态
- 收到 `batch_count = N` 时：
  1. 调用 `EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getRoleTypeList()`（**和 `/schedule` 用同一份数据源**）获取当前有注册 worker 的 role 列表
     - 0 个 → `NO_AVAILABLE_WORKER` (`"master not ready..."`)
     - 多个 → `INVALID_REQUEST` (`"batch_schedule only supports single-role deployments..."`)，提示 caller 改用 `/schedule`
     - 恰好 1 个 → 进入下一步
  2. 该 role 对应的 LoadBalancer 不支持 batch（没配 `ROUND_ROBIN`）→ `INVALID_REQUEST`
  3. 调 `RoundRobinLoadBalancer.selectBatch(N, role)`：从 role 的 alive worker 列表里按 cursor 顺序取 N 台
     - alive 列表为空 → 返回 role-specific 错误（例：`NO_PDFUSION_WORKER`）
     - `N > alive 数量` 时 cursor 自然 wrap（例：`N=20` 对 10 台 worker → 每台 2 个）

### Master 不记账

caller 调本接口拿到 worker 后**调用 worker 的过程 master 完全不感知**：

- 无 `localTaskMap` 记账、无心跳对账、无 lost detection、无 rollback
- caller 与 worker 之间的失败语义全交给 caller：超时/失败时 caller 自己重试或换 worker
- 因此 batch_schedule 路径上 cursor 推进的副作用之外没有任何残留状态

### 顺序语义

`server_status` 数组顺序 = RR cursor 依次推进得到的 worker 顺序。caller 如果想保留 RR 均匀分布特性，建议把自己手头的 N 个业务请求**按下标对位**派发到 `server_status[i]`。

caller 也可以打乱、跳过、并行派发——master 不强制，只保证它返回的这 N 个 slot 在 cursor 序上是连续的。

### slave 转发到 master

多 master + ZooKeeper 选主部署下，caller 调到的实例如果不是 leader，会自动 HTTP 转发到 leader。转发失败时返回 `MASTER_UNREACHABLE`，**不 fallback 本地处理**（避免多 master 同时调度造成账本分裂）。

## curl 示例

```bash
curl -X POST http://master:7001/rtp_llm/batch_schedule \
  -H 'Content-Type: application/json' \
  -d '{"batch_count": 3}'
```

## Python 示例

```python
import requests

def get_batch_workers(batch_count, master_url="http://master:7001"):
    resp = requests.post(
        f"{master_url}/rtp_llm/batch_schedule",
        json={"batch_count": batch_count},
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
| `MODEL_SERVICE_CONFIG` | 必填 | worker 集群地址配置（见 flexlb 公共文档） |
| `FLEXLB_SYNC_CONSISTENCY_CONFIG` | 可选 | 开 ZK 选主时设；不设则单实例 master |

## 设计取舍

本接口**故意做窄**：固定 RR、单 role、不记账。换取的是：

- 协议极简，caller 端只传一个 int
- master 零 in-transit 状态，cursor 自带 stampede 免疫
- 失败/重试语义全在 caller 侧，master 不参与

未来如果需要：
- **支持其他调度策略**（如 ShortestTTFT 用于避热点）：需要重新引入 `request_id` 入参 + `localTaskMap` 记账，**会是破坏性协议变更**
- **支持多 role 同时调度**：需要 caller 在 request 里显式指定 role
- **支持 cache-aware 调度**：需要 caller 传 `block_cache_keys` 等 per-request metadata

预留扩展空间但本期不实现。

## 相关代码

- `rtp_llm/flexlb/flexlb-api/.../HttpLoadBalanceServer.java::batchScheduleRequest`
- `rtp_llm/flexlb/flexlb-sync/.../scheduler/DefaultRouter.java::batchSchedule`
- `rtp_llm/flexlb/flexlb-sync/.../strategy/RoundRobinLoadBalancer.java::selectBatch`
- `rtp_llm/flexlb/flexlb-common/.../dao/loadbalance/BatchScheduleRequest.java`
- `rtp_llm/flexlb/flexlb-common/.../dao/loadbalance/BatchScheduleResponse.java`
- `rtp_llm/flexlb/flexlb-common/.../dao/loadbalance/BatchScheduleTarget.java`
