# Master 观测接口契约

业务接口、平台钩子和运维配置接口保持兼容。测试框架迁移自己的调用方，不以删除共享接口为前提。接口出现在 Mock 模块中，不代表调用方只能连接 Mock；未知外部调用按共享接口保留。

`master/info` 被 frontend 探测使用；`notify_master` 被选主通知使用；`hook/*` 被部署脚本使用。它们不并入诊断入口。`schedule_snapshot`、`queue_snapshot`、日志和流量策略更新以及健康检查也保持原合同。`inflight_status`、`server_latency` 和 reset 的旧消费者可以继续运行；新观测不再扩充这些旧响应。测试干预仍使用显式启用、字段白名单和 token 保护的控制接口。

## 只读快照

`GET /rtp_llm/debug/snapshot?include=workers,scheduler` 返回 `schema_version: 2`。`workers` 和 `scheduler` 组件版本均为 1。未知参数、重复参数、未知组件和超出预算的请求返回 400。`limit` 缺省 1024，上限 2048；`budget_ms` 缺省 100，上限 250；响应最多 1 MiB。仅允许一个同步采集，忙时返回 429。字节预算不足返回 503，不提供伪完整数据。

顶层提供进程级 `instance_id`、每次采集的 `snapshot_id`、采集起止 epoch 毫秒、`capabilities`、`status` 和组件。组件有独立采集时间；组合响应不承诺全局原子性。服务端不缓存响应。客户端超时是传输预算，服务端执行预算不能由它无限扩大。

Worker 范围明确为 `prefill_decode_endpoint_registry`，不代表服务发现全集，也不覆盖 PDFusion 或 VIT。目录取自两种角色的不可变 routing directory，Worker 内的轮询、应用和缓存来源独立更新。读取期间目录变化，或条数/时间预算耗尽，返回 `partial`；覆盖不完整时不能根据缺席判定 Worker 已撤出。即便覆盖完整，也只表示此次稳定目录遍历，没有连续时间窗的无异常保证。

Worker 身份由 `(instance_id, worker_id, generation)` 组成。`worker_id` 为角色和地址；generation、状态和缓存版本以十进制字符串传输。所有年龄由 Master 单调时钟计算，单位毫秒；未知值为 null，不能补零或默认健康。

| 字段 | 来源与含义 |
|---|---|
| `lifecycle_state` | WorkerStatus generation 的 active/retiring，不等于容量可用 |
| `last_reported_alive`、`poll_response_age_ms` | 已验证成功的状态轮询，沿用已有 PollHealth 语义 |
| `applied_observation_age_ms`、`applied_status_version` | 成功应用的状态或验证成功的同版本心跳；年龄从转换为不可变回包观测时开始，包含应用等待 |
| `received_cache_version`、`indexed_cache_version` | 收到的缓存状态和已应用索引版本 |
| `index_initialized`、`cached_key_count` | 索引初始化情况和收到的缓存 key 数；不证明集合 membership |

同版本心跳只有在已有 endpoint 的处理成功后才刷新应用观测，不重放版本化状态。回包转换时间不是网络收包时间；该观测不涵盖转换前的等待。应用观测也不承诺后续所有 scheduler projection 已完成。观测字段不参与调度、不新增状态失鲜准入策略；不能从它推导现有版本没有实现的准入结果。

`scheduler` 只返回 queued 和 inflight 总数。普通数值时序使用 Prometheus；需要关联 Worker 身份和 generation 的观测使用快照。全量缓存 key、请求历史和昂贵的一致性扫描不属于此接口。读取不发 Worker RPC、不落盘、不修复状态，也不获取 endpoint admission 或 cache mutation 锁。

## 框架消费

`adapters.master_diagnostics.MasterDiagnosticsClient` 校验 schema、组件版本、时钟、身份与字段类型。通过 `component()` 读取所需组件；缺失、partial、截断和不支持都显式失败。旧版 `DebugClient` 的请求行 schema 不用于 Worker 快照。

能力检查应在大规模注入和发流前完成。不支持的能力不能回退为成功轮询年龄、零计数或默认健康。Master 重启后 instance 变化，不能继续用前一进程的 generation 和累计计数判定恢复。原始证据与功能结果分别保留，后续采集失败不能覆盖已经证实的异常。
