# KVCM remote cache

## 版本与制品

来源 commit 统一记录在 `deps/kvcm.bzl`，已包含本次 SDK／内源适配。客户端 RPM 和服务器包须从同一内源／开源 SDK／真实 PACE 组合产出；RPM 必须包含匹配的头文件和动态库。新增虚接口与 StartWrite 参数改变 ABI，旧 RPM 和开源 PACE stub 均不兼容。

制品记录尚未填写。依赖 KVCM 的目标（含 remote smoke 的全量 `//...` 构建）会被门禁拦截，普通无 KVCM 依赖的目标不受影响。产出后填写真实 URL、SHA256 和按 `_source_id()` 拼接的来源标识。

使用上述 commit 构建无需再应用已合入的补丁；基线至提交的 diff 归档仅用于审计。

## 配置

| 参数／环境变量 | 默认 | 语义 |
|---|---:|---|
| `kvcm_default_query_type`／`KVCM_DEFAULT_QUERY_TYPE` | 2 | Instance 默认模式：1=batch、2=prefix、3=SWA、4=Mamba |
| `kvcm_query_type`／`KVCM_QUERY_TYPE` | 0 | 请求模式；0 使用 Instance 默认值 |
| `kvcm_sw_size`／`KVCM_SW_SIZE` | 0 | SWA 窗口大小，单位为 cache key／block，SWA 模式必须大于 0 |
| `kvcm_read_backend_type`／`KVCM_READ_BACKEND_TYPE` | 0 | 0=普通查询；1=3fs、2=mooncake、3=PACE DRAM、4=NFS、5=VCNS 3fs、9=PACE SSD |
| `kvcm_min_replica_count`／`KVCM_MIN_REPLICA_COUNT` | 0 | StartWrite 最少可读副本数；0 由服务端按 1 处理 |

`KVCM_CLIENT_CONFIG` 的显式 JSON 优先于自动生成的 Instance 配置，`default_query_type` 缺省为 2；请求仍可通过 `kvcm_query_type` 覆盖。每个 backend 绑定一个默认 Instance。按后端查询固定 batch，此时显式 `kvcm_query_type` 只允许 0 或 1；普通 Mamba／SWA 查询使用 `kvcm_read_backend_type=0`。

`--kvcm_model_sdk_config`（环境变量 `RECO_MODEL_SDK_CONFIG`）分别配置单一 DRAM 或 SSD：

- DRAM：`[{"type":"pace","sdk_log_level":"INFO"}]`。
- SSD：`[{"type":"pace_ssd","sdk_log_level":"INFO"}]`；读取可设 `kvcm_read_backend_type=9`，服务端须用 `ST_TAIRMEMPOOL_SSD`、`media_type=5`。

存储地址／媒体由服务端 storage config 下发。正常写入候选只配置所选数据后端；事件存储单列在 `event_report_storage_candidates`。

SWA／batch 的 miss 保留原 key 位置。复用要求：FULL 完整前缀、LINEAR 最终状态、SWA 完整窗口，并满足所有 TP rank；缺失 URI 不算命中。混合 LINEAR＋SWA 可先写 FULL＋LINEAR 部分，读取仍要求 SWA 完整。原有 IOV、多 pool/group、FULL＋LINEAR、同布局 TP 沿用。

自动 Instance identity 包含默认模式和注册 group 配置，升级可能切换缓存命名空间；自定义 ID 须与服务端现存配置一致。`KVCacheConfig` pickle 为版本 8／74 项，读取兼容 1～7，进程间须同一构建。

## RPC 入口

`RpcService/ExecuteFunction` → `KVCacheManager::executeFunction` → `KVCMStorageBackend::execute`，只接受 TP0、本 backend 的 Instance；SDK 失败返回非 OK RPC 状态。

| `RemoteOperationRequestPB.op` | SDK 方法 | 返回 |
|---|---|---|
| `REMOTE_OPERATION_MATCH_LOCATION_LEN` | `MatchLocationLen` | `matched_blocks`，单位为 block |
| `REMOTE_OPERATION_MATCH_META` | `MatchMeta` | `locations`、原始 `metas` 字符串 |
| `REMOTE_OPERATION_REMOVE_CACHE` | `RemoveCache` | 成功／失败 |
| `REMOTE_OPERATION_GET_LOCATIONS_BY_BACKEND` | `GetCacheLocationsByBackend` | key 对齐的 `backend_locations`，含空项、type、spec size、URI |
| `REMOTE_OPERATION_GET_HOST_CACHE_STATE` | `GetHostCacheState` | host、本地长度、P2P 拉取及最终长度 |
| `REMOTE_OPERATION_MATCH_LOCATION` | `MatchLocation` | 原始位置，batch／SWA 保留空项 |

protobuf JSON 示例：

```json
{"remote_request":{"op":"REMOTE_OPERATION_MATCH_LOCATION_LEN","trace_id":"cache-length","metadata":{"query_type":2,"block_keys":[101,102,103]}}}
```

`metadata` 支持 token、offset／bool mask、窗口、detail level、backend、spec name、medium、P2P host count。backend 查询只支持 batch，非空 spec-name 列表须与 key 一一对应；host-state 只支持 prefix／Mamba，提供元数据查询。

设置 `kvcm_read_backend_type` 后，位置经 group／rank 映射进入 TP payload 请求，URI 交给 `TransferClient::LoadKvCaches`。事件 URI 仅描述位置，不作为本次 payload 后端。

## I/O 契约

RTP 要求 `sdk_config.drain_on_timeout=true`：deadline 阻止排队 I/O，已提交任务收尾后才释放调用方引用。SDK 默认预算 12s、TP 广播 15s、PACE 内部同步兜底 10s，起点不同；排队及 drain 可能使返回超出外层预算。每个执行 rank 持本地 block pin 至 SDK 返回。其他后端的底层取消能力仍受 SDK 契约限制，尤其是 Mooncake soft timeout。

写入按 offset／bool mask 映射原 key，actual URI 必须同序回填 spec；失败走 FinishWrite 中止，空 session 尽力关闭，失败由服务端短时过期兜底。PACE fallback 保留 hostname：DRAM 沿用 `PREFER_LOCAL`（0），避免媒体值 2 碰撞旧 `ONLY_REMOTE`；SSD 用 `LOC_DEFAULT | MEDIA_TYPE_LOCALSSD`（5）。

## 服务端事件配置

Publisher 状态机、完整性和拓扑限制见 [事件上报](backend/kv_cache_event_publisher.md)。对应 storage 必须放入 Instance Group 的 `event_report_storage_candidates`：

```json
{"global_unique_name":"rtp_hbm_events","storage_type":"ST_EVENT_REPORT_L1P5","event_report":{"heartbeat_timeout_ms":30000,"cleanup_grace_ms":300000,"liveness_check_interval_ms":5000,"snapshot_min_interval_ms":1000},"check_storage_available_when_open":false}
```

## 验收边界

本补丁仅经静态审查，未编译、未测试或运行真实 PACE I/O。实际 ABI、超时／DMA／TP 时序和端到端响应未验证；完整 remote-cache smoke 属于后续验收。
