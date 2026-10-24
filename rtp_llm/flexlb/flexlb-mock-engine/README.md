# FlexLB Mock Engine

Java Mock 模拟 Prefill / Decode 的调度、KV 容量、执行耗时、请求生命周期和故障，用于验证 FlexLB。它不执行模型、不分配真实 KV tensor，也不能用模拟耗时证明 GPU 的绝对性能。

功能与 workload 测试统一从 [online_eval](../tools/online_eval/README.md) 进入。编译、Java 21 前置和运行命令见[编译与运行](../tools/online_eval/docs/development/build-and-runtime.md)；实例参数、流量与门禁以场景 YAML 为准。JavaLoadClient 环境变量的隔离与字段校验共用 [runtime/load_client.py](../tools/online_eval/src/runtime/load_client.py) 中的名称清单；取值以场景配置为准。

## 实现与配置入口

CacheStatus 的额外编码优化由 `--cache-status-compact-encoding true` 开启，默认关闭。
独立 Whale Mock role 对应环境变量 `MOCK_CACHE_STATUS_COMPACT_ENCODING=1`。
开启后用每次 RPC 独立持有的 `long[]` 快照编码自然 keys，绕过临时 Set 和 protobuf map；
Device、Host 和外部 fixture 的重复 keys 只输出一次。响应仍是标准 CacheStatusPB，
同版本省略 keys，`need_cache_keys=false` 回显客户端版本并只返回容量等标量。
编码开关只减少 Java 对象分配，不改变同步协议；不能用于推断 real 的序列化成本。

同步协议依据为 real 的 [KVCacheManager::getKVCacheInfo](../../cpp/cache/KVCacheManager.cc)
和 [BlockTreeCache::getKeySnapshot](../../cpp/cache/block_tree_cache/BlockTreeCache.cc)：
key 版本初始为 -1，只在逻辑节点创建、最终移除时递增；GPU/Host 副本搬运、引用计数和容量变化不改变 key 版本。
版本和 key 集合在同一临界区取快照。客户端版本不同时返回当前完整集合；版本相同时返回空集合。
空集合是否表示清空索引由响应版本是否变化决定，容量始终实时读取。

| 组件 | 职责 |
|---|---|
| `JavaMockEngineCluster` | 逻辑 P/D 引擎、队列、gRPC、请求所有权与状态报告 |
| `MockPerformanceModel`、`PrefillTimeFormula` | 性能 JSON、公式、执行耗时及有界噪声 |
| `MockPrefillBatchPolicy` | FIFO 候选准入、token 预算与 CP 计价 |
| `MockLruBlockCache` | Device KV lease、前缀匹配与驱逐 |
| `MockVitEmbeddingCache`、`MockVitHashCache`、`MockTensorPool` | ViT 分层驻留、独立 sidecar、连续空间和事件回收；规则见[引擎配置](../tools/online_eval/docs/architecture/engine-configuration.md) |
| `MockControlServer`、`DynamicEngineManager` | HTTP 控制、故障和动态引擎管理 |
| `JavaLoadClient` | 播放、Schedule、Fetch / Generate、请求终态证据 |

行为配置通过 `--engine-config` 导入 real 最终值与 allocator 容量，
`--calibration` 只接受性能模型。字段所有权、导出、支持边界与回读统一见
[引擎配置](../tools/online_eval/docs/architecture/engine-configuration.md)。

运行时性能控制使用单个不可变快照发布，单次耗时计算使用同一快照的系数。`/set_perf` 校验后一次发布 `prefill_fixed_ms`、`decode_scale`；被拒绝的请求不改变状态，scheduler 容量不由性能覆盖控制。


引擎内部组批与 Master 派发相互独立，Master 批次不等于执行批次。
生产入口使用导入的 FIFO 配额；物理 KV 容量独立生效。
未缓存 token 配额是停止继续准入的软配额，完整 token 与 padding 矩形另行计价。
首条候选保留 real 的单流例外；请求长度与物理池检查仍然生效。

## 请求、状态与资源

BATCH 默认 `--auto-fetch false`：Decode 先保留 KV 并报告 `KV_ALLOCATED`，Prefill 完成后释放计算槽，但保留 deferred context 和 connector KV，直到 Fetch、取消或过期。Fetch 可以在 Prefill 完成之前或之后接入。NON_BATCH 已有客户端流。正常测试需要回读完整输出；显式 schedule-only 实验的 auto-fetch 不等于客户端 Fetch 证据。`no_respond` 表示服务端 RPC 黑洞。

请求 `timeout_ms` 使用跨准备、KV 等待、Prefill、Fetch 和 Decode 的绝对预算，缺省或非正数不设置请求 deadline。Fetch 接入 TTL 与帧间等待是独立约束，不重置请求预算。Decode ALLOCATE 对永久容量不足立即返回领域错误 602；临时不足等待容量代际变化，直到释放、取消或请求超时。`decode_retry_*` 不作为容量等待上限，`/set_perf` 拒绝这些旧覆盖字段。

P 已产生完整输出时，远程路径协商并发送 `PREFILL_COMPLETE`；本地路径使用同一资源结算规则，不执行额外 Decode step。EnqueueBatch 在分配资源前整批拒绝重复 request_id，并拒绝与单 DP 引擎不符的 rank 及缺少 input 的成员。

WorkerStatus 在移除 running 条目前先发布终态记录；running、保留的 completion 和 finished cursor 在同一监视器内取快照。队列操作和文件写入在快照临界区之外执行。
完成记录固定保留 1000 条，插入时淘汰最旧记录；读取不消费记录。慢消费者无法补取已淘汰的 completion。
`status_version` 是读取时的微秒墙钟；完成版本以进程启动时的微秒墙钟为种子，每次发布递增。响应 finished cursor 是客户端游标与本次返回记录版本的最大值，没有新记录时回显客户端游标。
`MOCK_STATUS_SNAPSHOT_LOG=true` 配合 `--events-file` 可记录状态 RPC 快照，排查终态遗漏。

取消通过 gRPC `RpcService/Cancel`、HTTP `/cancel_request` 或进程内测试通道使用同一合同。停止 Decode、崩溃或强制移除会向所属 Prefill 响应队列投递 `8209 REMOTE_GENERATE_FAILED`；正常排空不产生断链错误。容量和请求超时通过 gRPC status 与 `ErrorDetailsPB` trailer 返回；下游有效领域错误优先于传输层兜底。请求业务错误通过 gRPC `onError` 和领域错误详情结束响应，响应队列只接受一个终态；不模拟真实 keepalive 时延。

单引擎关闭先停止准入、关闭调用并打断回调，验证 transport 与该引擎回调均退出后才释放端口和所有者。关闭失败保留资源归属，允许重试；线程忽略中断时不能报告关闭成功。

协议和所有权细节统一见[请求生命周期](../tools/online_eval/docs/architecture/request-lifecycle.md)。

## Device 与 Memory cache

Native-token 请求以部署级 角色 `kv_cache.reuse_cache` 和请求 `reuse_cache` 共同决定复用；本地缓存层由部署配置决定，`kv_cache.enable_device_cache` 由 real 有效配置提供，host 层由 `kv_cache.enable_memory_cache` 和容量观测决定，不使用请求中的本地 tier 开关。实际复用 key 上限为 `floor((input_len - 1) / block_size)`，至少保留一个 token 计算。显式回放 block keys 保留合成 trace 的计数语义，不作为 native-token 命中边界的验证依据。

Device pool 容量、已引用 block 与可回收前缀相互独立；压力不能靠增加命中率绕过物理容量或 reserve 水位。
容量描述中的 `enable_gpu_prefix_tree` 控制角色的 GPU 前缀树驱逐；关闭后使用普通访问顺序 LRU，已引用 block 仍受保护。

host cache 是可选的前缀元数据缓存，不为 Decode 创建 memory cache。`memory_blocks` 以完整逻辑 cache-key block 计量，比较 CP 引擎容量时需换算物理 block token 数和 CP 宽度。

匹配先读取 GPU 前缀，再连续扩展 memory 前缀；只有两者的非重叠总命中减少 compute token。Memory hit 仍需分配 GPU KV。成功 Prefill 写入完整 block；取消或失败不写入。Master 的 cache-key 状态包含两层的并集，但执行容量始终只计 GPU。

Memory 默认使用前缀树驱逐：读刷新 recency，在飞读写保护容量，只驱逐最旧的合格叶节点。`enable_memory_prefix_tree: false` 仅用于匹配显式关闭该策略的真实配置。成功 H2D 读取后 host 条目被消费并归还容量；失败或取消只释放保护，条目仍可复用。崩溃清空两层缓存。

H2D / D2H 模拟描述符队列、完成回调与资源保护，不执行真实 tensor 复制、DMA 或 host pin，也不模拟 partial block 或 disk tier；行为参数来自 real 有效配置，复制耗时来自校准值。驱逐 lifetime 是插入到驱逐的时长，不是过期时间。

Whale/KMonitor 的周期采样间隔固定为 1 秒，与 stdout 调试统计周期独立。WorkerStatus / CacheStatus 的 QPS 和 metadata 读取耗时在每次读取完成时上报，耗时不包含 protobuf map 构造、编码和网络发送；周期采样不重放 RPC、scheduler 或 forward 事件。P 的 model forward 指标只计模型耗时，MTP step 指标包含执行阶段注入的停顿，不计入队和等待执行 lane 的时间；Mock 不模拟 native executor 的 bookkeeping 成本。ViT 保留 real worker 的常量状态版本 1 和无增量版本的目录快照协议。

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

### 大缓存同步 fixture 的编码开销

合成 cache fixture 的 gRPC 响应直接从不可变的 key 几何快照编码标准
`CacheStatusPB`，不在 Mock 中为每个版本重新构造完整 protobuf Map。
该路径仍返回相同的 key、bool 值、版本、容量和 block size；在途响应持有自己的
快照，不会被后续 churn 改写。cold / changed / future version 返回全量 keys；
same version 返回空 keys。冻结 churn 后，相同版本轮询不再产生 key 载荷。
自然 key 和 fixture key 先合并版本再判断是否变化，不能用合并后的客户端游标直接判断 fixture 自身版本。

协议依据为 `LocalRpcServer::GetCacheStatus` 和 `model_rpc_service.proto`；
`CacheFixtureWireTest` 用原始 protobuf parser、完整响应字节及真实 gRPC 客户端验证。
这只降低 Java Mock 的对象分配，不模拟 C++ 引擎的序列化 CPU 成本，也不改变
fixture 与真实推理 KV 容量相互独立的边界。Real 的全量响应仍需复制 key 列表、
构造 protobuf Map；这里的直接编码是 Mock 的负载生成优化，不代表 real 没有这些成本。
该路径适合验证 Master 的解析、索引和健康隔离，不能用于验收真实引擎侧的大响应
资源消耗或心跳延迟。实验仍须按实际响应条目/s、bytes/s 验收同步强度。
