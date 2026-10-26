# P2P 超时与 KV 资源保护

请求超时不代表 RDMA 已停止访问内存。Decode 超时后可立即返回失败，但目标 KV 必须继续持有，直到所有 rank 都确认物理传输停止。

## 处理顺序

1. 到达 transfer deadline 后，请求进入失败终态，后台发送 `CANCEL_READ`，取消未开始的任务。正在传输的任务只记录取消意图，等待真实完成回调。
2. RDMA watchdog 先结束超时 RPC；若回调仍未到达，在 deadline 加 `P2P_TRANSFER_NOT_DONE_RESOURCE_HOLD_MS`（默认 10000ms）后关闭相关 connection。断链本身不能作为安全释放 KV 的依据。
3. `P2PConnectorAsyncReadContextChecker` 在后台继续持有目标资源。取消广播成功后，按 10～100ms 的退避间隔查询全部 rank 的 lease；单轮查询超时由 `P2P_LEASE_QUERY_TIMEOUT_MS` 限制，内部最多等待 500ms。
4. 只有全部 rank 都满足 `sealed && started_ops == finished_ops`，才能释放该上下文持有的 KV 引用。释放不要求调用方先销毁失败上下文。
5. 查询失败、取消失败或仍有未完成任务时，持续持有并重试；不再因持有时间到期而 abort，也不按时间强制释放。

## 泄漏阈值

环境变量 `P2P_KV_LEAK_EXIT_THRESHOLD_PERCENT` 默认 **30**，允许 1～100；也可通过 `--p2p_kv_leak_exit_threshold_percent` 设置。

统计对象为失败后仍在等待物理完成的目标 KV，按 `(cache group, block ID)` 去重，再按各 group 的实际块字节数和层数加权。分母为 Decode 本地 KV 总容量，包含 KV scale。正常传输中的请求不计入滞留量。

滞留量**严格超过**配置比例时，rank 0 在保留 KV 引用的状态下退出进程；等于阈值时继续持有和轮询。KV 分片布局沿用现有对称部署假设。

## 状态观测

新增 Gauge 指标使用前缀 `rtp_llm_p2p_connector_decode_scheduler_`：

- `retained_kv_bytes`：滞留 KV 字节数。
- `retained_kv_percent`：占总容量的百分比。
- `retained_context_count`：滞留上下文数。
- `oldest_kv_hold_ms`：最长持有时间。

存在滞留时每 30 秒打印容量、占比、阈值和最长持有时间；全部释放后打印恢复日志。后台容量检查不依赖控制 RPC 完成，控制线程繁忙时也会按秒唤醒。

Prefill 的资源预持有、终态 tombstone 和 MR/rkey 错误处理沿用现有逻辑；此改动不重试业务请求。
