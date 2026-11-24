# P2P 超时与 KV 资源保护

Decode 请求失败时，如果接收任务可能已经派发，目标 KV 进入隔离。所有 rank 确认停写后立即释放。隔离满 5 分钟仍未确认停写时，执行强制释放。

## 部署端口

Prefill 向 Decode 的各个 worker 发起 P2P KV 数据传输。TCP 和 RDMA 接收端均绑定 Decode worker 的 `cache_store_listen_port + 1`，即默认端口布局中的 `START_PORT + rank_id × worker_info_port_num + 3`。升级 PD 部署时，应允许 Prefill 访问这些 Decode 入站端口，并为同一 IP 上的各 worker 保留独立端口。

## 处理顺序

1. 到达 transfer deadline 后，请求进入失败终态，后台发送 `CANCEL_READ`，取消未开始的任务。正在传输的任务只记录取消意图，等待真实完成回调。
2. RDMA watchdog 先结束超时 RPC。如果回调仍未到达，在 deadline 加 `P2P_TRANSFER_NOT_DONE_RESOURCE_HOLD_MS`（默认 10000ms）后关闭相关 connection。等待完成回调期间，对应 KV 计入滞留量。
3. `P2PConnectorAsyncReadContextChecker` 在后台继续持有目标资源。取消广播成功后，按 10～100ms 的退避间隔查询全部 rank 的 lease。单轮查询最多等待 500ms，并受 `P2P_LEASE_QUERY_TIMEOUT_MS` 限制。每个请求同时最多执行一个控制任务，不同请求最多并行四个。
4. 全部 rank 都满足 `sealed && started_ops == finished_ops` 时，立即释放该上下文持有的 KV 引用。
5. 查询失败、取消失败或仍有未完成任务时，继续持有 KV 并重试。单个请求从进入隔离起满 5 分钟后，即使未确认所有 rank 停写，也会释放 KV 引用并记录 WARN。强制释放依赖先前提交的 RDMA 或 H2D 操作已停止访问这些块。

## 滞留量

统计对象为失败后仍在等待物理完成的目标 KV，按 `(cache group, block ID)` 去重。启用 PP 时，分子和分母按该 Decode 副本的所有 PP stage 和 TP rank 加总，包含 KV scale。正常传输中的请求不计入滞留量。

## 状态观测

`P2P retained KV` 日志区分仍在等待取消确认、尚未收到租约查询结果、最近一次查询失败，以及已收到租约状态但仍在等待停写的上下文。取消或查询失败日志会携带首个已知错误。若收到了 worker 错误，日志会包含对应 rank。

新增 Gauge 指标使用前缀 `rtp_llm_p2p_connector_decode_scheduler_`：

- `retained_kv_bytes`：滞留 KV 字节数。
- `retained_kv_percent`：占总容量的百分比。
- `retained_context_count`：滞留上下文数。
- `oldest_kv_hold_ms`：最长持有时间。


存在滞留时每 30 秒打印容量、占比和最长持有时间，全部释放后打印恢复日志。回收定时器独立于控制 RPC。

Prefill 请求状态表最多保留 1,048,576 个活跃及终态记录。终态记录保留一小时用于防止重放。表满时返回 `RESOURCE_EXHAUSTED`，已有请求仍可完成，保留期到期后会腾出空间。
