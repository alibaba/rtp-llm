# FlexLB Mock Engine

Java Mock 模拟 Prefill / Decode 的调度、KV 容量、执行耗时、请求生命周期和故障，用于验证 FlexLB。它不执行模型、不分配真实 KV tensor，也不能用模拟耗时证明 GPU 的绝对性能。

功能与 workload 测试统一从 [online_eval](../tools/online_eval/README.md) 进入。编译、Java 21 前置和运行命令见[编译与运行](../tools/online_eval/docs/development/build-and-runtime.md)；实例参数、流量与门禁以场景 YAML 为准。JavaLoadClient 环境变量的隔离与字段校验共用 [runtime/load_client.py](../tools/online_eval/src/runtime/load_client.py) 中的名称清单；取值以场景配置为准。

## 实现与配置入口

| 组件 | 职责 |
|---|---|
| `JavaMockEngineCluster` | 逻辑 P/D 引擎、队列、gRPC、请求所有权与状态报告 |
| `MockPerformanceModel`、`PrefillTimeFormula` | 性能 JSON、公式、执行耗时及有界噪声 |
| `MockPrefillBatchPolicy` | FIFO 候选准入、token 预算与 CP 计价 |
| `MockLruBlockCache` | Device KV lease、前缀匹配与驱逐 |
| `MockControlServer`、`DynamicEngineManager` | HTTP 控制、故障和动态引擎管理 |
| `JavaLoadClient` | 播放、Schedule、Fetch / Generate、请求终态证据 |

性能配置由 `--performance-config` 指向的 JSON 提供。Prefill 可以使用固定耗时或公式，Decode 使用声明的 step 曲线或校准参数；角色 `scale` 与全局 `sleep_scale` 调整模拟时间。噪声参数必须有明确的标准差和绝对上限，不能与 `jitter_pct` 同时启用。配置字段与校验以 `MockPerformanceModel` 为准，采集档案和模型偏离规则见[数据规则](../tools/online_eval/data/README.md)。

## FIFO 与 Master 派发

引擎内部组批和 Master 的 decision / dispatcher 是两根轴。BATCH 的 `EnqueueBatch` 和 NON_BATCH 的直接请求都进入引擎的候选池；Master 批次不等于引擎执行批次。

`prefill.fifo` 存在时使用 FIFO 策略，并拒绝同时声明
`prefill.max_batch_requests`、`prefill.max_batch_tokens` 或 `prefill.direct_batch_size_max`，
避免两套批预算相互遮蔽。不启用 FIFO 的路径使用独立 regroup 配置，其合同不应套用到 FIFO。

FIFO 的主要语义：

- `max_requests` 限制 stream 数。`max_batch_tokens` 同时限制完整 token 总量和最长完整序列长度乘总序列数，采用严格小于边界。多返回序列按 `num_return_sequences` 计宽；beam search 的首次 prefill 宽度为 1。
- 首条候选保留真实 FIFO 的例外：未命中 context 长度小于 `max_seq_len` 时，可以超过批 token 预算。独立输入校验和物理 KV 容量检查仍然生效。
- `max_batch_tokens_without_cache` 是停止继续准入的计算量配额。CP padding 逐序列计算后乘宽度；当前候选可以使累计量越过配额，下一条停止准入。
- `cp_enabled` 未声明时由 `cp_size > 1` 推导；未启用 CP 时 `cp_size` 必须为 1。`force_single` 默认 true，仅在 CP 启用时生效。请求上限、CP 宽度与模型长度使用目标部署的有效配置。
- `max_inited_kv_streams` 限制持有非空 KV lease 的请求数。达到上限后，已初始化 KV 的请求仍可推进，空 lease 不占配额。
- `max_batch_kv_len`、`max_waiting_requests` 和 `prefill.max_waiting_batches` 是 mock 专属约束，FIFO 默认不启用。只有显式设置 `prefill.fifo.fault_limits_enabled: true` 时才用于故障实验；实际物理 KV pool 容量始终生效。

真实参考是 [FIFOSchedulerConfig](../../cpp/config/ConfigModules.h) 与 [FIFOScheduler](../../cpp/engine_base/schedulers/FIFOScheduler.cc)。FIFO 准入对齐不表示 Decode、多序列输出或完整缓存状态机与真实引擎等价；耗时模型仍需相同输入下的执行证据验证。

## 请求、状态与资源

BATCH 默认 `--auto-fetch false`：Decode 先保留 KV 并报告 `KV_ALLOCATED`，Prefill 完成后释放计算槽，但保留 deferred context 和 connector KV，直到 Fetch、取消或过期。Fetch 可以在 Prefill 完成之前或之后接入。NON_BATCH 已有客户端流。正常测试需要回读完整输出；显式 schedule-only 实验的 auto-fetch 不等于客户端 Fetch 证据。`no_respond` 表示服务端 RPC 黑洞。

WorkerStatus 在移除 running 条目前先发布终态记录；running、保留的 completion 和 finished cursor 在同一监视器内取快照。队列操作和文件写入在快照临界区之外执行。
`MOCK_COMPLETION_RETAIN_WINDOW` 控制保留容量，超出容量立即淘汰最旧记录；读取不消费记录。慢消费者无法补取已淘汰的 completion。
`MOCK_STATUS_SNAPSHOT_LOG=true` 配合 `--events-file` 可记录状态 RPC 快照，排查终态遗漏。

取消通过 gRPC `RpcService/Cancel`、HTTP `/cancel_request` 或进程内测试通道使用同一合同。停止 Decode、崩溃或强制移除会向所属 Prefill 响应队列投递 `8209 REMOTE_GENERATE_FAILED`；正常排空不产生断链错误。业务错误走 `error_info` 数据帧，响应队列只接受一个终态，不模拟真实 gRPC trailing status 或 keepalive 时延。

协议和所有权细节统一见[请求生命周期](../tools/online_eval/docs/architecture/request-lifecycle.md)。

## Device 与 Memory cache

Device pool 容量、已引用 block 与可回收前缀相互独立；压力不能靠增加命中率绕过物理容量或 reserve 水位。
`prefill.enable_gpu_prefix_tree`、`decode.enable_gpu_prefix_tree` 控制角色的 GPU 前缀树驱逐；关闭后使用普通访问顺序 LRU，已引用 block 仍受保护。

`prefill.memory_cache` 是可选的 host 前缀元数据缓存，缺省关闭，不为 Decode 创建 memory cache。`capacity_blocks` 以完整逻辑 cache-key block 计量，比较 CP 引擎容量时需换算物理 block token 数和 CP 宽度。

匹配先读取 GPU 前缀，再连续扩展 memory 前缀；只有两者的非重叠总命中减少 compute token。Memory hit 仍需分配 GPU KV。成功 Prefill 写入完整 block；取消或失败不写入。Master 的 cache-key 状态包含两层的并集，但执行容量始终只计 GPU。

Memory 默认使用前缀树驱逐：读刷新 recency，在飞读写保护容量，只驱逐最旧的合格叶节点。`prefill.memory_cache.enable_prefix_tree: false` 仅用于匹配显式关闭该策略的真实配置。成功 H2D 读取后 host 条目被消费并归还容量；失败或取消只释放保护，条目仍可复用。崩溃清空两层缓存。

该模型的 H2D / D2H 复制是原子的元数据操作，不模拟 tensor、DMA 重叠、host pin、写入队列、partial block 或 disk tier；配置不接受 copy-delay 和 copy-lifecycle 开关。驱逐 lifetime 是插入到驱逐的时长，不是过期时间。

## HTTP 控制与采集

控制端口为 `baseGrpcPort - 1`。请求可用逻辑 `engine` 名或 gRPC `port` 选择引擎。

| 接口 | 方法 | 用途 |
|---|---|---|
| `/health`、`/snapshot`、`/requests` | GET | 健康、状态快照与近期请求证据 |
| `/metrics` | GET | Prometheus；默认按 role 汇总，`?per_engine=true` 展示逐引擎样本 |
| `/inject`、`/clear_inject` | POST | 注入或清除故障 |
| `/set_perf`、`/set_kv_pressure`、`/set_queue_depth` | POST | 显式修改模拟参数、KV 压力或准入限制 |
| `/stop_engine`、`/start_engine`、`/cancel_request` | POST | 引擎停启与取消 |
| `/add_engine`、`/remove_engine` | POST | 动态拓扑 |

`/remove_engine` 先关闭新工作 RPC 准入，再撤销 discovery。默认 graceful 模式保留既有工作直到排空或达到 `drain_timeout_ms`；超时会转为强制清理并报告 `drained=false`。显式 `mode=abrupt` 立即拆除。Generate、Enqueue、RemoteGenerate 和迟到 Fetch 被关闭的入口以 UNAVAILABLE 拒绝；状态与清理接口保留。`admission` 记录关闭时刻及受理/拒绝计数，关闭后受理计数不得增长。关停结果和业务门禁分别判断。

采集只需 cluster 控制端口的一个 scrape target；指标合同见 [METRICS.md](METRICS.md)。`/snapshot`、事件和日志用于专门的协议证据，不作为 Prometheus 失败后的静默回退。

默认独立 loopback IP 保持 Master 的逐引擎身份标签。macOS 不保证整个 `127.0.0.0/8` 可达；本机跨地址连接时用 `--unique-engine-ips=false`。外部 Pod 使用可达地址，不能发布 loopback；规则见[Whale 配置](../tools/online_eval/docs/whale/configuration.md)。

## Whale 与测试

[Whale 镜像入口](whale/README.md) 保留组件 Docker context；部署、发现与验收统一见 [Whale runbook](../tools/online_eval/docs/whale/README.md)。Bundle 中的 `MOCK_BUNDLE_FILE_DISCOVERY=1` 显式选择文件发现，`MOCK_DISCOVERY_FILE` 指向由动态引擎接口维护的 discovery 文件。运行时逻辑拓扑在重启后按初始配置重建，不等于平台独立 Engine Pod 的生命周期。

Java 测试位于 `src/test/java/org/flexlb/mockengine/`，通过 FlexLB 根目录的 Maven wrapper 执行；实际类和测试数量以源码及测试结果为准。Python 回归、dry-run 和真实场景验收见[新增 case](../tools/online_eval/docs/development/adding-cases.md)。
