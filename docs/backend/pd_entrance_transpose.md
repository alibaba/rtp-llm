# PD Decode 入口与同版本发布说明

本版本统一采用 Decode 入口：Frontend 把请求发到 Decode；Prefill 计算首 token，并通过 StartLoad 交付首 token、采样状态和 KV 传输信息；后续生成和流式输出由 Decode 完成。`DECODE_ENTRANCE` / `--decode_entrance` 仍可解析，任一取值都使用 Decode 入口。

## 发布契约

Frontend、FlexLB master、Prefill、Decode 必须来自同一发布版本，使用配套的外源 commit、内源 commit 和构建产物。不能把“服务启动成功”当作协议兼容证明。

本次切换不支持这四部分跨新旧版本混跑：

- P/D handoff 由 Decode 发起，使用 GetPeerInfo 查询 Prefill 拓扑，通过 StartLoad 获取 KV 和首 token。
- 新请求从 Decode.GenerateStreamCall 获取输出；旧 EnqueueGroup / FetchResponse 不提供过渡实现。
- 新 Decode 要求 Prefill 的 GetPeerInfo / StartLoad 与当前分组 KV 布局一致。KV 传输、远端缓存和 host-cache 对端仍须遵守同构建协议契约。

当前没有自动版本握手或混版本降级。版本配套和路由分组由发布编排保证，不能依赖运行时自动识别并修复误配。

## 切换步骤

1. 记录同一轮完整功能 CI 使用的内外源 commit，准备该版本配套的四类产物。CI 未通过前，不视为可发布。
2. 建立独立的新版本路由组，配置正确的 Prefill、Decode 和前端地址。旧版和新版不能共享会随机选取两代实例的路由池；新组未齐备时不接流量。
3. 核对各角色实际版本、TP/CP、模型及 KV 布局。验证新组的普通请求、Master 预入队、流式取消、TCP/RDMA 传输和真实 smoke 输出。
4. 流量按完整路由组切换到新版本，确保每个请求的 FE→Master→D→P 都留在同一版本组。禁止逐个替换在线 P/D、让另一侧继续使用旧协议。
5. 旧组停止接收新请求后，等待在途请求、物理传输和资源持有全部结束，再回收旧实例。取消 ACK、RPC 超时或 Decode 本地终态都不能单独证明 Prefill 侧也已清理完成。

新 Decode 发布取消终态的条件是本地生成结束、活动读取结束且 P2P 资源持有已释放；Prefill 不可达时，远端取消记录独立重试，并在本地安全终态后按现有 10 分钟保留期退出。到期只删除轻量控制记录，不把 Prefill 标为已清理；本地读取或资源持有未结束时不适用该回收规则。不得据此强制释放仍在传输的 KV。

取消注册表分别为业务请求记录和准入前取消记录设 1,048,576 条上限。业务额度已满时，新请求返回并发容量错误。提前取消额度已满时，无法登记新 ID 的 Cancel 返回 gRPC RESOURCE_EXHAUSTED。已有请求的取消和重复取消仍可执行。本地生命周期事件推进单条请求的清理，仍在等待本地物理清理的请求继续保留。Decode 通过共享 RPC 完成队列确认 Prefill 取消，每个 Decode Server 同时最多处理一轮远端确认。Prefill 批量入队时为等待 `StartLoad` 注册一次性定时器，及时挂接会撤销定时器。

容量拒绝或取消超过 10 分钟仍未完成本地清理时，`PD cancellation registry pressure` 日志记录两类额度的占用、上次告警后的拒绝数及未完成清理的超期数。单个注册表最多每 30 秒告警一次。

Prefill 多模态瞬时错误可用 PREFILL_RETRY_TIMES 配置额外尝试次数（默认0），PREFILL_RETRY_TIMEOUT_MS 限制追加尝试窗口（默认0不另加限制）。每次从原始请求重建输入并保留原始起始时间和 deadline；不可重试错误、取消和请求过期立即停止。该重试不重放入队或 KV 传输，Decode 继续消费 FE 改写的 token IDs。

旧启动参数及对应环境变量仍可解析。旧入口专用的线程池、等待、重试和轮询参数，以及旧传输的提前停止参数保留配置兼容。实际通信、取消与资源释放使用新路径的 deadline 和租约。

KV 单次通信默认预算为 5 秒，从 worker 开始发送计时，上限为请求剩余时间。等待 Prefill 注册、产出 KV 和返回首 token 使用请求 deadline。冷启动或建链需要更长预算时，可调整 `--load_cache_timeout_ms`。RDMA 在传输 deadline 后等待 `--p2p_transfer_not_done_resource_hold_ms` 再尝试关闭连接。Decode 通过租约查询确认停写才释放 KV，`--p2p_lease_query_timeout_ms` 控制单次查询的上限，`--p2p_kv_leak_exit_threshold_percent` 控制滞留量达到何种程度时退出进程。

旧同步 Decode RPC 的 `rtp_llm_rpc_` 阶段指标 `allocate_resource_rt_us`、`load_cache_from_prefill_rt_us`、`local_generate_rt_us`、`load_cache_min_rt_us`、`load_cache_max_rt_us` 和 `load_cache_polling_cost_us` 随旧路径下线。新链路可用 `rtp_llm_stream_loading_cache_latency_us` 观察流的加载耗时，用 `rtp_llm_p2p_connector_decode_schedule_cost_time_us` 和 `rtp_llm_p2p_connector_decode_worker_total_cost_time_us` 分别观察协调与 worker 耗时。这些指标的计时边界不同，已启用 tracing 的部署还可查看 Decode 生成和 KV 加载 span。

## 回滚

保留旧版本完整路由组。需要回滚时将流量切回该组，先排空新组再回收；不能只回滚其中一个角色。新旧组使用独立 KV 资源，不跨协议版本复用在途 KV 或缓存状态。

本文说明发布要求，不表示上述 CI 或部署验证已经完成；以对应准确 commit 的完整 CI 和部署证据为准。

旧 RPC 指标名继续注册。Decode 入口上报首次 KV 分配、Prefill 缓存加载和本地生成阶段耗时。旧 RemoteLoad 的逐 rank 最短、最长耗时及轮询开销在新路径不再产生样本。
