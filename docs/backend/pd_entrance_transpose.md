# PD Decode 入口与同版本发布说明

本版本统一采用 Decode 入口：Frontend 把请求发到 Decode；Prefill 计算首 token，并通过 StartLoad 交付首 token、采样状态和 KV 传输信息；后续生成和流式输出由 Decode 完成。`DECODE_ENTRANCE` / `--decode_entrance` 已移除。

## 发布契约

Frontend、FlexLB master、Prefill、Decode 必须来自同一发布版本，使用配套的外源 commit、内源 commit 和构建产物。不能把“服务启动成功”当作协议兼容证明。

本次切换不支持这四部分跨新旧版本混跑：

- P/D handoff 由 Decode 发起，使用 GetPeerInfo 查询 Prefill 拓扑，通过 StartLoad 获取 KV 和首 token。单 stage 同样返回完整的 Prefill KV 布局；Decode 以对端公布的 CP 和物理布局规划传输，不再用本地 CP 配置推导或校验 Prefill 的 CP 大小。
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

KV 单次通信默认预算为 5 秒，从 worker 开始发送计时，上限为请求剩余时间。等待 Prefill 注册、产出 KV 和返回首 token 使用请求 deadline。冷启动或建链需要更长预算时，可调整 `--load_cache_timeout_ms`。RDMA 在传输 deadline 后等待 `--p2p_transfer_not_done_resource_hold_ms` 再尝试关闭连接。Decode 确认所有 rank 停写后释放 KV。`--p2p_lease_query_timeout_ms` 控制单次查询的上限。

Decode 入口继续上报 `rtp_llm_rpc_allocate_resource_rt_us`、`rtp_llm_rpc_load_cache_from_prefill_rt_us` 和 `rtp_llm_rpc_local_generate_rt_us`，分别对应首次 KV 分配、KV 加载和本地生成。旧 RemoteLoad 的逐 rank 最短/最长耗时和轮询开销不再产生样本。流级加载指标与 P2P 协调/worker 指标仍可用于分层定位，计时边界不相同。

GetPeerInfo 使用最多 15 秒的探测预算，并受请求剩余时间约束；探测超时归为连接超时，允许前端按连接失败策略重路由。`--load_cache_timeout_ms` 只控制 KV 通信，不控制拓扑探测。

## 多模态与 CP 部署约束

`VIT_SEPARATION_LOCAL` 下，Decode 入口需要本地 ViT 展开 token layout，Prefill 需要本地 ViT 生成 embedding。两侧 checkpoint 都必须包含完整的多模态权重，并为 ViT 权重和缓存池预留显存；默认自动 KV 定容会扣除这部分占用。独立 ViT 部署应在 Decode、Prefill 和前端一致配置远端 ViT 分离模式，避免闲置的本地 ViT 实例。v41 等模型特有的本地处理约束仍以模型实现为准。

PD handoff 的 token IDs、span 和特征哈希是跨角色缓存身份契约。两侧必须使用相同模型权重、预处理和确定性的 ViT 执行；Prefill 发现哈希不一致会返回 `MM_PROCESS_ERROR`，不会在 Decode 已持有旧 token IDs 时单侧改写身份，也不会通过重试掩盖确定性失配。发布验收应包含真实多模态请求和哈希失配负例。

CP 方法是角色相关配置：Prefill 使用实际通信方法（`ALL_GATHER`、`ALL_GATHER_WITH_OVERLAP` 或 `ALLTOALL`），Decode 使用 `PREFILL_CP` 表示上游开启 CP。分片开关须配套，fixed/SWA Decode 还需明确 `prefill_cp_size`。将未启用实际 CP 的 Prefill 与期望 CP 分片的 Decode 配对属于配置错误，应拒绝，不能通过放宽布局校验兼容。


## 配置默认值

默认使用 TCP。启用 `CACHE_STORE_RDMA_MODE` 后才使用 RDMA；各参数也可通过同名小写 CLI 选项覆盖。

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `LOAD_CACHE_TIMEOUT_MS` | 5000 ms | KV 单次通信预算 |
| `MAX_RPC_TIMEOUT_MS` | 7200000 ms | 请求未指定超时时的默认生成期限 |
| `PREFILL_RETRY_TIMES` | 0 | 多模态瞬时错误的额外尝试次数 |
| `PREFILL_RETRY_TIMEOUT_MS` | 0 | 追加尝试预算，0 表示仅受请求 deadline 约束 |
| `P2P_TRANSFER_NOT_DONE_RESOURCE_HOLD_MS` | 10000 ms | RDMA 传输超时后等待物理完成，再尝试关闭连接 |
| `P2P_LEASE_QUERY_TIMEOUT_MS` | 500 ms | Decode 单轮 lease 查询超时，可调低，内部上限为 500 ms |
| `P2P_CANCEL_BROADCAST_TIMEOUT_MS` | 1000 ms | P2P 取消广播的 gRPC 超时 |
| `P2P_CANCELLED_KEYS_TTL_MS` | 3600000 ms | 保留已取消请求的记录，用于拒绝迟到的传输 |
| `CACHE_STORE_TCP_WORKER_QUEUE_SIZE` | 500 | TCP/RDMA 控制面的 worker 队列容量 |
| `RDMA_TRANSFER_WORKER_THREAD_COUNT` | 16 | RDMA 传输 worker 线程数 |
| `RDMA_TRANSFER_WORKER_QUEUE_SIZE` | 100 | RDMA 传输 worker 队列容量 |
| `P2P_RDMA_ENABLE_H2D_COPY` | false | 启用 pinned host staging 后再做 H2D copy |
| `P2P_RDMA_STAGING_TOTAL_BYTES` | 0 | staging 容量，启用 H2D 时须至少容纳一个模型 KV block |
| `P2P_PREFILL_SENDER_THREAD_COUNT` | 4 | Prefill 异步发送线程数 |
| `P2P_PREFILL_SENDER_QUEUE_SIZE` | 10000 | Prefill 异步发送队列容量 |

升级时从启动模板移除以下旧选项及对应的环境变量。

- 入口及 Prefill 等待：`DECODE_ENTRANCE`、`PREFILL_MAX_WAIT_TIMEOUT_MS`、`PREFILL_PREPARE_RESOURCE_POOL_SIZE`、`PREFILL_STOP_STREAM_WAIT_TIMEOUT_MS`。
- 旧重试和轮询：`DECODE_RETRY_INTERVAL_MS`、`RDMA_CONNECT_RETRY_TIMES`、`DECODE_POLLING_KV_CACHE_STEP_MS`。
- 旧 P2P 超时：`CACHE_STORE_RDMA_TRANSFER_WAIT_TIMEOUT_MS`、`P2P_READ_STEAL_BEFORE_DEADLINE_MS`、`P2P_READ_RETURN_BEFORE_DEADLINE_MS`、`P2P_RESOURCE_STORE_TIMEOUT_CHECK_INTERVAL_MS`、`P2P_LAYER_CACHE_BUFFER_STORE_TIMEOUT_MS`、`P2P_KV_FORCE_RELEASE_TIMEOUT_MS`。

PR 早期版本的进程退出阈值 `P2P_KV_LEAK_EXIT_THRESHOLD_PERCENT`、`P2P_KV_LEAK_EXIT_THRESHOLD_MIN_PERCENT` 和 `P2P_KV_LEAK_EXIT_THRESHOLD_MAX_PERCENT` 已在后续提交中移除。

## 回滚

保留旧版本完整路由组。需要回滚时将流量切回该组，先排空新组再回收；不能只回滚其中一个角色。新旧组使用独立 KV 资源，不跨协议版本复用在途 KV 或缓存状态。

本文说明发布要求，不表示上述 CI 或部署验证已经完成；以对应准确 commit 的完整 CI 和部署证据为准。

旧 RPC 指标名继续注册。Decode 入口上报首次 KV 分配、Prefill 缓存加载和本地生成阶段耗时。旧 RemoteLoad 的逐 rank 最短、最长耗时及轮询开销在新路径不再产生样本。

前端启动时校验主机墙钟必须处于请求 ID 的可编码窗口（2020-01-01 至约 2037-06）。窗口外会明确拒绝启动并要求同步时钟，避免健康探针正常却所有业务请求报 500。运行后回拨仍由共享逻辑槽保持单调；真正耗尽编码空间时拒绝分配。
