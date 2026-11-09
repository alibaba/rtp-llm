# 组件配置与指标参考

[返回使用说明](../README.md)

本文记录真实 Master 适配、RTP/ViT、KV、Compute、Client 协议与指标口径。Case 的配置文件组织方式见 [Case 指南](case-guide.md)，内部职责见 [架构说明](architecture.md)。文中路径均相对框架目录，除非另有说明。

## 真实 Master 适配

`org.flexlb.MasterNode` 与生产 Application、同步器和调度器在同一 JVM，不另起 Master 子进程，也不实现 Mock 调度器。Case 的 `environment.master.config` 使用原生环境变量字符串：`FLEXLB_CONFIG` 是业务 JSON，`MODEL_SERVICE_CONFIG` 是路由 JSON。

Runner 先启动独占 ZK，再启动 Engine、Master 和 Client，与 YAML 书写顺序无关。已启动 Engine 的 `endpoints.json` 形成 topology/discovery；Runner 只替换 `MODEL_SERVICE_CONFIG` 的 hosts/discovery_file，保留 service_id、组和角色域名，不读取线上 VIP，不改写 `FLEXLB_CONFIG`。当前只支持一个 endpoint 组、分离角色和 gRPC；ViT 使用相同的发现和实例生命周期。

发现使用实际 Engine gRPC 端口，生产 WorkerAddressService 再转换内部 HTTP 身份，不假定 Mock 控制端口相邻。原始组件配置、实际环境、effective-config 和 discovery 分开保存；Master 暴露生产 `/prometheus`。

只绑定 `MasterStateApi` 的两个读取接口：

| 接口 | 行为 |
| --- | --- |
| `master/get` | 默认返回所选活动 Master 的服务地址列表；`is_leader: true/false` 按各 Master 的生产 isMaster 声明筛选主/非主节点。数量由 expect 判断，不伪造唯一主备。 |
| `master/engine/get` | 读取各 Master 的 WorkerDirectory，按角色映射 Engine 服务地址，逐 Master 返回。未知地址报错，不能按部署数量补齐或将各节点取并集。 |

单/多 Master 全部使用生产 ZK 选举，不提供 standalone 分支。独立 ZookeeperNode 使用已有依赖、独立 JVM/数据目录；Runner 分配地址和本次 HIPPO_ROLE，运行时同步配置为 needConsistency=true、会话超时 6s，并保存实际值。ready 必须等到真实选举产生 Leader。

Master 使用 127.0.0.1、127.0.0.2 等地址，相同 HTTP 端口，gRPC=HTTP+2，适配生产以 IP 作为身份和共用转发端口的约束。端口启动时被抢占则失败，不静默换协议。Linux 默认支持 127/8；macOS 缺少别名时明确报错。实验之间不共享 ZK 或选举路径。清理先停止组件，再停止 ZK，最后停止 Prometheus，保留日志和数据。

接线测试分别覆盖 DIRECT/NON_BATCH 的 Generate、QUEUE/FIXED_WINDOW/BATCH 的 Enqueue→Fetch。`python/mock_framework/execution/roles/test/test_master_lifecycle.py` 在两种模式下持续运行同一 Client 计划，停止主节点、补回新 IP 的 Master、再停止旧节点；在旧请求 deadline 之后检查新主持续成功接流，并检查独立进程退出、各组件独立产物和当前采集目标。切换错误如实保留，不要求整个过渡期零错误。协议夹具测试和真实 Master 接线测试分开；用生产 pre_stop 钩子观察选举不等于完成 Master 生命周期动作。

## RTP 节点与公共校验

原生输入与配置不叠加 Mock 自定的 URL 长度、图片数量、图片边长、特征行数、并发 1024 等上限。保留输入格式、正数和整数溢出校验，以及原生配置对应的计算、队列和缓存预算。

节点仅监听本机回环地址；每个实例独立绑定原生 gRPC、控制 HTTP 和指标端点，共 JVM 不合并这些端点。`endpoints.json` 写实际地址、支持的 RPC 和阶段范围；`/health` 只列已绑定的公共接口，当前 Engine 没有绑定 Case 查询接口，Python 查询直接访问 Prometheus。

```yaml
environment:
  prefill:
    engine: rtp
    config:
      SEQ_SIZE_PER_BLOCK: "64"
      MAX_SEQ_LEN: "32768"
      MAX_BATCH_TOKENS_SIZE: "8192"
      REUSE_CACHE: "true"
      TEST_BLOCK_NUM: "1025"
      RESERVE_BLOCK_RATIO: "5"
      mock.compute.forward.expression: "param(baseMs, 1) + 0.01*totalComputeTokens"
```

原生配置保留 real 的名称和单位，例如 `concurrency_limit`、`seq_size_per_block`、`test_block_num`；同一字段同时写大小写两种形式会报错。P/D 的 `concurrency_limit` 默认 32，必须为正整数，控制每轮请求数。Mock 必须显式提供 `test_block_num`；`seq_size_per_block` 缺省或为 0 时取 64，`max_seq_len` 缺省或为 0 时取 8192。本地没有 GPU 探测，不自动推导物理 KV 容量。一个 Engine 表示完整计算体，不创建 CP/TP/DP rank 或模拟通信，不根据并行度隐式换算容量、block size 或已校准的公式。具体支持项和输入规则见下文“Engine 配置规则”。

RTP 请求公共支持范围集中在 `RequestValidation`，本地单 token、P/D Prefill 及 Decode ALLOCATE 均使用它：adapter/unique_key、多 beam/多返回、stop words、sp_edit/sp_input_lookup、额外 logits/hidden/prob/loss 输出明确拒绝。路径只保留自身角色、长度、路由和协议阶段规则。测试通过实际 gRPC 同时验证 max_new_tokens=1/2 和 D ALLOCATE，防止参数只在一条路径被拒绝。

### 执行和 P/D 交接

- P 的 `GenerateStreamCall(max_new_tokens=1)` 由 PrefillRpcServer 接入，不启用 D。请求经过 FIFO→选中时分配 KV→共享 Compute→真实等待→编码输出→释放；排队不占 KV。
- 多 token P/D 要求 `can_use_pd_separation=true` 且恰好一个有效 DECODE 地址。PrefillRpcServer 先向 D ALLOCATE，确认后启动 P，随后 LOAD、发 P 首 token、等待 LOAD 确认、发 GENERATE 并转发 D 输出。
- D 的 DecodeRpcServer 处理原生协议，AbstractRtpScheduler 管理资源和队列，只有 GENERATE 后进入计算队列；每个 Engine 一轮一个 Forward，按配置合并多个请求。初始 KV 分配遇临时不足等待容量变化；生成期间每轮先增长 KV，失败则终止，成功后才执行与提交输出。

PD 只执行调度器之间的交接，不模拟 KV 数据传输。P/D 只使用原生 RemoteGenerate TCP 连接，不注册自定义回调服务，也不跨节点比对 prompt、块布局或 cache keys。P 必须等自身 EngineScheduler 报告首轮 Forward 完成后才发送 LOAD；D 对已分配的请求执行零耗时异步传输事件，完成后回复 LOAD ACK。P 收到 ACK 才释放源 KV，然后发送 GENERATE 或 PREFILL_COMPLETE，D EngineScheduler 决定开始计算或结束。两端块大小各自独立，KV 的申请、持有、释放及 finished 记录仍归各自 EngineScheduler 管理。

接入层信任正常 P 生产者发送的 LOAD 和首 Token 完成事实，不模拟损坏的 KV 数据、假冒来源、迟到的跨请求交接或生产传输恢复。请求取消、总 deadline、节点关闭和下游 RPC 失败仍必须终止在途任务并回收资源。零耗时传输没有独立加载计时器；real 的 `load_cache_timeout_ms` 在此模型中忽略。

原生 EnqueueBatch/EnqueueGroup 当前只支持 DP rank 0，支持文本及经 ViT 准备的输入。Batch 接入已选择 PD 协议，不套用普通 GenerateStreamCall 的分流条件：max_new_tokens=1 也正常接纳，can_use_pd_separation 不改变 Batch 协议。首 Token 达到长度上限时，通过 PREFILL_COMPLETE 结束 D，不重复输出 Token，也不执行 Decode Forward。普通单 Token RPC 仍走本地 Prefill；非 PD 的多 Token 本地生成和 prefill_only 尚未模拟，明确返回 UNIMPLEMENTED。

入队 ACK 前必须取得 D 分配确认；Prefill 完成后推进 LOAD 和 Decode，整个流程不等待 Fetch。Fetch 仅从保留的输出容器单次接入，不持有调度器或 KV 资源。重复 ID 的整批项拒绝，其他非法项各自返回错误；成功项不因其他项失败而回滚。

已接受的批量请求不随入队 RPC 结束而取消；请求 timeout 覆盖全程，Fetch 接入期限默认 10min，接入后撤销。Fetch 重复为 ALREADY_EXISTS，过期/缺失为 NOT_FOUND。Cancel 只面向批量请求；活动项 ACCEPTED，未登记项 TOMBSTONED，已完成项 NOT_FOUND；抢占锁定原生 8429，墓碑和完成记录保留 10min。ACK 不承诺远端已经释放，直接 Generate 通过 RPC 取消。

P/D 都按实际 Forward 计时，token 是重复输入末 token 的模拟载荷，不做词表推理。Engine RPC 始终返回本轮新增 token，`aux.output_len` 始终累计；用户侧 incremental/累计展示由 Frontend 处理。P 首 token 计入总长度，非推测模式 D 执行 N−1 轮。Client TTFT 必须来自实际接收，不能使用 Engine aux 或计算耗时替代。

请求总预算取 timeout_ms 和 RPC deadline 较早者，不逐阶段重置；取消、超时和关闭只释放一次，旧回调不能影响复用身份的新请求。602/603/605 使用原生 ErrorDetailsPB。P 转发采用受控拉取/有限 Inbox，遵守发送就绪；共享连接不随单请求取消关闭。请求数不叠加固定保护上限，TaskHistory 保留 1024 项。旧游标越过历史窗口时，按 real 的恢复方式继续返回完整当前状态、仍保留的完成记录和推进后的游标，不以 OUT_OF_RANGE 阻断健康同步；超前游标不倒退。每次缺口读取在组件日志中记录 `finished_history_gap`，包含 requested_version、dropped_through、latest_version、retained，不能把这段完成记录当成完整请求历史。生产 proto 没有历史完整性字段，缺口证据保存在实验组件日志；需要完整逐请求证据时不能依赖 WorkerStatus 的有限窗口。real 当前为 1000 条/5s 保留，Mock 的 1024 条是明确的模拟边界，不宣称保留策略完全相同。

WorkerStatus 报真实等待/执行任务、finished/version，available_concurrency 保持生产默认 0，KV 以 token 为单位；CacheStatus 提供 block size 与版本。P/D 入队前检查 prompt 的最大 KV 容纳长度及上下文预留；临时容量不足仍等待。期望输出可以超过剩余上下文，实际生成长度取 max_new_tokens 与 max_seq_len−input 的较小值。P 的首 token 已达到 D 上限时，D 只回送结束标志，不执行 Forward 或重复输出 token。成功且 reuse 与缓存层开启时提交完整有效块；P 提交 prompt，D 提交 prompt 与已计算的生成前缀，失败不提交。没有完整动态 batch、GPU 或混合 KV 布局模拟。

## Block 容量与 RTP 缓存

每个 P/D Engine 有一个统一 KV 索引和 LRU，分别约束缓存保留容量与 device 驻留容量。`test_block_num - 1` 是 device 可用总块数；`mock.cache.total.blocks` 是统一池总块数，默认等于 device 块数且不得更小，不包含 null block。首次分配的 `reserve_block_ratio` 只按 device 容量计算，Decode 增长可使用预留块。

所有活动请求的不同物理块占用 device 额度，包括缓存命中、KV Loading 和等待 PD 交接的请求；共享前缀只计一次。命中也必须取得 device 额度才返回 Ready。最后一个活动引用释放后归还额度，缓存 key 可继续留在统一池。仅在统一池需要空间时执行 LRU，不因 device 暂时不足驱逐历史缓存。未持有 device 额度的缓存恢复视为瞬时，不模拟 Host/device 数据搬运、带宽、层级归属或额外队列。

Scheduler 接口保持 `Ready / Loading / Failed`；初次申请临时容量不足时等待容量变化，单请求超过 device 减预留后的容量立即失败；Decode 每轮增长前检查 device，增长失败按容量错误结束。WorkerStatus、CacheStatus 容量和现有 DEVICE 指标始终报告 device 约束，key 列表及 item num 报告统一索引，可大于 device 总块数。free 按已分配块优先填满空闲 device 的瞬时模型计数，available 表示可供活动请求使用的额度，不等同于未缓存空槽。

容量不足时，按 LRU 淘汰可回收的前缀树叶子。请求持有的块不会被淘汰。成功终态只发布已计算的完整块，失败或取消只释放占用。Master CacheStatus 与 KVCM Reporter 使用同一份有效块状态。P/D 网络交接仍走原生协议。

## 通用 V 与输入准备

Engine 是运行实例的概念，prefill/decode/vit 是 Engine 的角色；RTP/vLLM 是 Engine 后端，不能与 P/D/V 混为一层。当前配置中的 `component.engine` 字段选择 P/D 后端；V 统一实现于 `java/engine 的 vit 包`，不需要选择后端，也不与 `java/engine 的 RTP 实现` 并列为后端。以后 P/D 使用哪个后端都复用这一份 V。V 的媒体协议与 TCP 客户端仍在 `java/protocol`，Client/P 只依赖该协议模块。

P/D/V 各实例拥有独立 JVM、端口、队列、缓存和产物目录。统一产物为 `java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar`。

原生 `RemoteMultimodalEmbedding` 的一个请求可含多张图。缓存未命中的每份唯一媒体进入同一条异步准备链：准入 → 执行器排队 → 获取尺寸 → GPU 就绪队列 → Forward → 结果发布。RPC 等待线程、准备执行器和 Forward 执行线程分开；HTTP 慢读不会占用 Forward 线程。缓存命中直接返回，在途重复请求共用同一生产者和额度。`vit_concurrency` 限制准备执行器工作线程数，线程持续持有到该媒体计算完成；`vit_max_queue_size` 是额外等待额度，已接纳的唯一媒体总数不超过两者之和，GPU 就绪队列也按这个总预算派生。满时返回 RESOURCE_EXHAUSTED。任务额度由 VitPreparation 持有，直到共享结果发布完成后才释放；ViT 合批循环的完成通知只表示计算阶段结束。缓存条目的成功/失败发布采用身份校验，旧任务和重复完成不会覆盖新条目。

`gpu_max_batch_size=1` 串行执行一份就绪媒体工作项且不等待合批；大于 1 时按工作项数和收集窗口合批。同一 RPC 的多图分别准备，可进入不同 Forward，全部结果就绪后才完成 RPC。单张图的 patch/row 不拆分。最后一个等待者取消时，尚未执行的任务移出队列，进行中的下载中断；已经选入 Forward 的任务保留额度到实际计算/发布结束。状态查询不经过这些阻塞执行器。

`environment.vit` 不指定 engine 身份，配置示例：

```yaml
vit:
  deployment:
    replicas: 1
  config:
    patch_size: 14
    downsample_ratio: 3
    hidden_size: 4096
    min_pixels: 295936
    max_image_tokens: 1024
    mock.vit.skip.download: true  # 本地假 URL 回放；Whale 使用默认 false
    gpu_max_batch_size: 8
    gpu_batch_wait_ms: 10
    vit_concurrency: 64
    vit_max_queue_size: 64
    mm_cache_gpu_max_bytes: 67108864
    mm_cache_cpu_max_bytes: 268435456
    mm_hash_key_cache_max_bytes: 16777216
    mock.compute.forward.expression: "2 + 0.00001 * totalImagePixels"
```

Mock 启动不读取或挂载模型目录。ViT 配置显式提供上例中的五个几何参数。P/D 的 `max_seq_len` 默认 8192，需对齐目标模型时显式传入有效长度。多模态的 P 和 Java Client 使用相同的 `image_token_id`。参与缓存匹配的请求发起方和 Engine 使用相同的 Block 大小和有符号 int64 `cache_key_hash_seed`，本地文本用例可使用默认 seed 0。

`mock.vit.skip.download` 默认 false，Whale ENV 对应 `MOCK_VIT_SKIP_DOWNLOAD`。默认模式 GET 真实 HTTP(S) URL，仅读取图片头尺寸，不生成 RGB tensor；请求中的预处理 width/height 不冒充原图尺寸。支持 JDK ImageIO 图片格式（含 PNG/JPEG/GIF/BMP）及 WebP 头尺寸，其他格式明确拒绝。使用 real 的 `download_headers`（JSON 字符串）、`mm_image_max_file_size_kb` 和 `mm_timeout_ms`；文件大小取 mm_image_max_file_size_kb（正数时）和 DSV4.1 原生 `deepseek_v41_processor.py::load_image_source` 的 10 MiB 上限中较小值（`mm_image_max_file_size_kb <= 0` 仅关闭通用限制）；下载总期限最多 60 秒，并受任务剩余期限约束，瞬态网络错误及 HTTP 408/429/500/502/503/504 最多尝试三次，重试等待 200/400ms 且不超出剩余期限。队列等待同样计入任务期限。

本地假 URL 设置 `mock.vit.skip.download=true`，从回放宽高获取尺寸；仍走同一执行器、准入、队列和完成流程，不模拟下载耗时，也不伪造下载指标。真实 URL 回放可省略宽高。几何继续使用 V4.1 `plan_image_grid`/`num_image_tokens`，参数来自 ViT 的显式配置；视频、音频和 tensor 输入未实现。

回放数据保留原始 token、URL/尺寸与单个占位位置 Image.offset；已删除 Image.length，旧图片数据需要重新生成。Client 与 P/D 显式配置相同的 image_token_id，LoadClient 将回放中的图片位置编码为该值。直接构造请求的测试工具在不设置图片标记时仍使用 -200 约定；对接真实 Frontend 时使用其实际标记。标记出现次数必须与媒体数一致。P 的内部 Prepared 保存规范化语义 token 和 cache-only spans，缓存 hash 不再替换语义 token。Native GenerateInput 不携带布局；仅 P→D 的 GenerateRequest.multimodal_cache_spans 携带缓存身份。P/D 按原生逐块滚动算法将语义 token 与重叠 span 身份共同混入 key；同一媒体身份必须在两端得到相同 key。参与匹配的组件应显式对齐 cache-key seed。

V 以 miss 的图片组成 ForwardBatch，复用公式和 SimulatedCompute。计算完成后登记缓存、通知等待者；Compute 不直接改变缓存。命中和在途合并不重复计入 forward。提供 real 名称的 py_rtp_vit_embedding_forward_rt、batch_size、cache 指标及 mock_vit_theoretical_forward_rt；元数据模拟不能代替真实模型计时校准。

Embedding Device/Host 按逻辑 rows*hidden_size*BF16(2 bytes) 计费，各自 LRU；Device 淘汰转存 Host，Host 淘汰移除 embedding。Hash 独立按行数和 key 字节计费，因此 hash_hit 与 embedding_hit 不等价。在途计算使用独立 claim，失败移除，旧 claim 的迟到完成不能覆盖新 claim。消费者持有不可变结果，不依赖缓存继续驻留。缓存 key 对齐 Linux real MultimodalInput 的 URL/type/preprocess 身份，hash 内容是稳定模拟值，不宣称与实际 tensor 的 hash 相等。

Client 按 Master 的 V-only route 选 V，读取元数据、计算路由长度和缓存 key，再带 selected_vit 与原始输入请求完整 P/D 路由；Master 切换使 pin 失效时重新取一次 V 路由。P 在输入准备完成后才把规范化 token 序列交给 KV/EngineScheduler，并用同一输入准备 D ALLOCATE 和 PD 交接。P 的输入准备发生在调度锁外，受原请求 deadline/取消约束；这段准备暂不计入 P EngineScheduler 的队列快照，但计入 Client 端到端耗时。D 接收已准备输入，不重复编码。

P 的本地单 token、P/D 单请求和 Master batch 都由 PrefillRpcServer 管理输入准备，RtpRpcService 只负责路由。提交 EngineScheduler 的单个与批量入口只接受已准备输入，不在调度器里调用 V。接入层直接用 Session 登记 PENDING，并负责过滤已取消的 Batch 成员、整理逐项结果；提交成功后在同一执行锁内交给 EngineScheduler；准备失败同样发布 finished 增量，但不增加计算请求和队列指标。开始时间和 deadline 在接入时确定，显式组不会在准备结束后重置预算。

VP 同 JVM 和跨进程都走 TCP gRPC，只传 shape/dtype、split_size 和逐行 INT32 hash，不生成 embedding payload，也不伪造 RDMA lease。Mock P 校验形状、长度和展开 span 后使用；此响应不能交给真实 tensor 解码器，VP 网络带宽和 GPU 内存复制不在模拟范围内。

P/D/V 的入口统一为 `java -jar java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar`。本地实例读取各自配置文件。

独立 JVM 可选择两种输入方式：设置 `MOCK_CONFIG_FILE` 时，只读取文件中的 id/component_role/component.config；未设置时，只读取业务 ENV，身份由 `ROLE_TYPE=PREFILL|DECODE|VIT` 提供；可用 `MOCK_NODE_ID` 显式指定，否则使用 role 与发布地址生成诊断身份。文件和 ENV 不叠加，没有第二套默认值。P/D 文件还需提供 `component.engine: rtp`。

独立入口使用 real 的 `START_PORT`（默认 8088），HTTP 为 START_PORT，gRPC 为 START_PORT+1。`MOCK_BIND_HOST`、`MOCK_ADVERTISE_HOST` 只控制进程监听和发布地址；监听通配地址且未显式指定发布地址时，通过主机名解析本机 IP。镜像默认监听 0.0.0.0，本地 Host 默认监听 127.0.0.1。

EngineMain 在本地发布配置、端点和 ready，Whale 模式仅记录启动日志。HTTP 健康检查与原生 CheckHealth 使用相同检查链。内部执行故障使进程非零退出，TERM 触发资源释放。

镜像和 CI 使用方法见 [Whale 镜像与启动](whale-engine.md)。无论是否共 JVM，VP/PD 都走 TCP、只传元数据。

## Compute、Forward、Decode 和 EOS

每个组件只配置一份 Compute；Forward、Decode 进度和 EOS 都归 Compute，不再配置组级 decode/generation，也不保留旧 compute.expression 入口。

```yaml
environment:
  prefill:
    engine: rtp
    config:
      mock.compute:
        forward:
          expression: "1.23 * (1 + 0.01*totalComputeTokens)"
        eos:
          probability: 0.0025
          seed: 20260922
  decode:
    engine: rtp
    config:
      mock.compute:
        forward:
          expression: "19.5 + 0.175*batchSize"
        decode:
          tokens.per.step: 2.6
        eos:
          probability: 0.0025
          seed: 20260922
```

RtpConfig 只加载 ComputeSpec；ComputeModel 组合 FormulaForward、可选噪声、DecodeStepModel 和 StoppingModel。非默认 Decode 进度参数只允许配置在 D，P/D 都可以配置 EOS。旧字段、未知字段、null、错误类型或缺少必要 seed 明确拒绝，没有别名和兼容分支。模型校验来自 Java，Python 只编译组件配置结构。effective-config 只保存生效值及解析后的模型参数，原始输入由 Runner 的 config.json 保留；不再产生 pending_config。

调度前 Compute.prepare 固定请求快照、本轮进度和 KV 需求。Forward 完成后 Compute.completed 按 EOS 和长度上限截断本轮输出，返回 committedTokens/finished。EngineScheduler 只校验请求仍有效并提交结果、更新队列、处理资源和 PD；不判断正常结束是 EOS 还是达到上限。取消、超时、关闭和计算失败仍走独立失败路径，不能成为正常 finished，也不能因旧回调再提交 token。

P 每轮只产首 token，finished 表示整个请求是否结束，不表示 Prefill 阶段是否完成。P 阶段完成后退出计算集合，保留传输需要的 KV；即使首 token EOS，也仍完成 LOAD 确认，再通过 PREFILL_COMPLETE 让 D 结束，不启动 D Forward。PD 不再传预测的 stop_after，不由 P 决定 D 的未来停止长度。

ForwardBatch 保留请求长度、实际缓存命中、已生成长度、stepTokens、优先级和图片 shape。FormulaForward 只计算 Duration，不保留没有实际替换入口的 ForwardModel 接口；SimulatedCompute 只执行可取消等待；公共 Compute 的产出计算不修改请求或队列。一个 Engine 同时至多一个 Forward，完成结果只提交一次。

公式复用生产 ArithmeticFormula 的 param(name, value)、sum 等函数，单位 ms，保留到 ns；非法变量、负值、非有限值和溢出拒绝。逐请求变量只在 sum 中使用：inputTokens/hitCacheTokens/computeTokens/hasHitCache/outputTokens/generatedTokens/contextTokens/stepTokens/priority/imageCount/imagePixels；batch 使用 batchSize、dpRank、total/max 派生变量。RTP 额外绑定 cpSize、cpEnabled（0/1）、tpSize、dpSize，只有显式引用才参与计算，不自动缩放已有拟合公式。默认 FormulaForward 不猜测多个 DP 的组合方式。

倍率直接乘入 forward.expression，不提供独立 scale/noise.scale。噪声位于 forward.noise，默认 stddev.ms/stddev.ratio=0；启用时要求 seed 和正 max.deviation.ms，扰动有界、最终耗时不小于 0。seed、Engine 身份、角色和 batch 身份固定一次耗时样本，取消批次不补完成指标。当前 DeepSeek Case 不启用噪声。

Decode 只配置每轮平均输出量 `tokens.per.step`，默认 1。小数用相邻两轮累计目标的差分实现，例如 2.6 每轮输出 2 或 3 个 token。每轮每请求计一个计算输入。EOS 或长度上限可以截短本轮输出，KV 只为本轮可能提交的前缀预留。

EOS 由 `StoppingModel` 按请求和绝对 token 位置决定，遵守 `min_new_tokens`。`ignore_eos` 屏蔽 EOS，实际输出仍受 `max_new_tokens` 和上下文长度限制。标准回放使用 `ignore_eos=true`，按输入文件给出的目标长度回放。

real 依据：MtpExecutor::decodeStep 先 target verify Forward，再执行 speculative_sampler；MtpBatchStreamProcessor::prepareDecodeSpecUpdateInfo 提取接受后的 token，GenerateStream::specUpdate 经 NormalGenerateStream::updateOutput 检查 EOS/max；CompleteTokenIds::matchEosToken 截断到第一个 EOS。PrefillRpcServer::remoteLoadCacheEnd 保留 LOAD ACK、释放源 KV 和 PREFILL_COMPLETE 顺序。当前 Mock 不支持 stop words/beam search 等未实现条件，显式请求会拒绝。

## Engine 配置规则

Case 的 `component.config` 可内联 YAML，或引用相对 Case 目录的配置文件。Python 只展开点号和组织实例，不维护 Engine 参数默认值。Java 的 `EngineConfig` 负责输入转换，`RtpConfig`、`VitConfig` 负责默认值、范围和语义；没有配置源基类或覆盖层。

已实现且语义对齐的 real 参数沿用 real 命名。未实现的 real 参数直接忽略，值不参与校验，也不进入 effective-config。例如磁盘/远端缓存、独立加载/重试超时、异步传输队列、模型路径、设备与尚未模拟的并行通信参数；出现这些参数不代表 Mock 实现了对应能力。原生参数拼写错误也会被忽略，因此复用 real 配置后应核对 effective-config 中真正生效的字段。

Mock 专用参数使用小写点号名称，例如 `mock.compute.forward.expression`；也可写成嵌套 YAML。未知 Mock 参数、大小写错误、空对象、null、路径冲突和重复输入均报错。Mock 不接受旧的 compute/memory_cache_blocks 等别名作为覆盖入口。`capacity.*`、`mock.cache.memory.blocks` 等旧容量入口和 `device_eviction_policy` 均不支持；LRU 固定。`device_blocks`、`reserve_blocks` 只是 effective-config 的派生输出，不能用于配置覆盖。不要在 Mock Case 放入不生效的 `memory_cache_size_mb`、`enable_disk_cache`、`block_tree_*_queue_max_size` 或独立加载/重试超时。`enable_memory_cache=true` 和 `enable_remote_cache=true` 仍拒绝启动；扩大统一池无需打开这些开关。

### 已实现的 real 参数

| 参数 | 默认值/约束 | 行为 |
| --- | --- | --- |
| seq_size_per_block / max_seq_len / test_block_num | 64 / 8192 / 必填；前两者为 0 时用默认值，test_block_num ≥ 2 | 基础哈希/物理块 token 数、单请求序列上限、每 rank 的 Device 物理块数（含 null block）。P 的逻辑分配粒度为基础块大小乘 prefill_cp_size。 |
| concurrency_limit | 32，正整数 | P/D 每轮请求数上限；real 的 ConcurrencyConfig 输入经 engine_config.py 写入内部 max_generate_batch_size。 |
| reserve_block_ratio | 5，0–100 | 首次分配预留百分比。 |
| reuse_cache / enable_device_cache | false / true | 前缀复用和 Device 索引。 |
| tp_size / dp_size | 1 / 1，正整数 | real 并行配置；tp_size 用于原生状态/接入元数据，二者分别绑定公式 tpSize/dpSize，不模拟 rank 间执行。 |
| prefill_cp_size | 1（0 视为默认） | Mock P 每组包含的基础缓存块数。大于 1 时只取每组末尾的键，逻辑块大小为基础块大小乘该值。D 的本地 FULL KV 始终按基础块管理。 |
| cp_rotate_method | DISABLED | 沿用 real PrefillCPConfig；ALL_GATHER、ALL_GATHER_WITH_OVERLAP、ALLTOALL 开启通用 CP，cpSize=tp_size、cpEnabled=1，否则为 1/0。PREFILL_CP 的计算策略不模拟；它与通用 CP 都可启用下述 P KV 分片布局。 |
| max_batch_tokens_size | 0，启动时解析为 max_context_batch_size × max_seq_len | P 的 batch token/矩形准入；合法 singleton 可超过预算。D 不使用该预算选批，仍通过 WorkerStatus 报告。 |
| max_context_batch_size | 1，正整数 | 只在 max_batch_tokens_size 缺省或为 0 时参与预算推导，不直接限制 batch 请求数。 |
| max_batch_tokens_without_cache / max_inited_kv_cache_streams | 0 / 0 | 仅 P 生效；0 不启用对应限制。 |
| disable_decode_first_malloc_device_reuse | false | real 内部 CacheConfig 属性，文件输入；不杜撰公开 ENV 名。 |
| gpu_max_batch_size / gpu_batch_wait_ms | 1 / 10 ms | 仅 V；batch=1 为逐请求执行且等待为 0；>1 同时限制 Forward 工作块数与媒体数，较大图片组分轮执行。 |
| vit_concurrency / vit_max_queue_size | 64 / 64 | 异步工作线程上限 / 额外等待额度；总准入和 GPU 就绪队列容量为两者之和。 |
| mm_cache_gpu_max_bytes / mm_cache_cpu_max_bytes | 各 2 GiB | V embedding 缓存逻辑预算，单位 byte。 |
| mm_hash_key_cache_max_bytes | 256 MiB | V hash 缓存逻辑预算，单位 byte。 |

real 默认值和映射以 `rtp_llm/server/server_args`、`rtp_llm/config/engine_config.py`、`rtp_llm/config/py_config_modules.py` 和 `rtp_llm/cpp/config/ConfigModules.h` 为依据。Mock 不用真实 GPU 分配这些预算，也不按实例伪造物理资源消耗。

### Mock 参数

| 参数 | 含义 |
| --- | --- |
| mock.cache.total.blocks | 统一池总块数，不含 null block；默认 test_block_num - 1，不得小于 device 可用总块数。ENV：MOCK_CACHE_TOTAL_BLOCKS。 |
| mock.scheduler.mode | P 调度方式，默认 `fifo` 按单请求入队、自主组 Forward batch，拒绝 `EnqueueBatch`/`EnqueueGroup`；`group` 显式启用组语义。ENV 为 `MOCK_SCHEDULER_MODE`；D 始终使用普通调度。 |
| mock.vit.skip.download | 默认 false；true 时只跳过实际图片读取，使用输入宽高，其余异步调度不变。 |
| mock.compute.forward.expression | 必填，Forward 公式，结果单位 ms。 |
| mock.compute.forward.noise.stddev.ms / stddev.ratio / max.deviation.ms / seed | 有界噪声，详见 Compute；数值后缀均相对 mock.compute.forward.noise。 |
| mock.compute.decode.tokens.per.step | D 平均每轮进度，与真实投机机制不同。 |
| mock.compute.eos.probability / seed | P/D 的概率停止模型。 |

### ENV 输入

real 公开参数按其 server_args 声明读取 ENV。P/D 上表的公开项对应大写下划线名，例如 `CONCURRENCY_LIMIT`、`SEQ_SIZE_PER_BLOCK`、`TP_SIZE`、`DP_SIZE`、`CP_ROTATE_METHOD`。V 的 batch 参数分别使用 `VIT_GPU_MAX_BATCH_SIZE`、`VIT_GPU_BATCH_WAIT_MS`，队列参数为 `VIT_CONCURRENCY`、`VIT_MAX_QUEUE_SIZE`；缓存参数为 `MM_CACHE_GPU_MAX_BYTES`、`MM_CACHE_CPU_MAX_BYTES`、`MM_HASH_KEY_CACHE_MAX_BYTES`。不会从这些 ENV 名反推另一套默认值。

Mock 名称的点号转换为下划线并大写，例如 `mock.compute.decode.tokens.per.step` → `MOCK_COMPUTE_DECODE_TOKENS_PER_STEP`。`MOCK_COMPUTE_*`、`MOCK_CACHE_*`、`MOCK_VIT_*` 是严格校验的业务命名空间；框架启动变量另行处理。只读取已声明的业务字段，其他进程 ENV 不写入配置产物。

```sh
ROLE_TYPE=DECODE MOCK_NODE_ID=decode-0 \
SEQ_SIZE_PER_BLOCK=128 MAX_SEQ_LEN=32768 TEST_BLOCK_NUM=1025 \
CONCURRENCY_LIMIT=32 MOCK_COMPUTE_FORWARD_EXPRESSION='19.5 + 0.175 * batchSize' \
MOCK_COMPUTE_DECODE_TOKENS_PER_STEP=2.6 \
START_PORT=8088 \
java -jar java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar
```

## DSV4.1 Engine / KVCM 契约

ViT 的 GetWorkerStatus 总是显式提供 `mm_inflight_inputs`（包括真实零值），计数来自去重后的媒体所有权；缓存命中不增加，重复等待者不重复增加，排队/执行/缓存结果发布完成前不提前减少。RPC 等待者数不是该指标。状态查询不经过阻塞的结果等待执行器。真实图片由 Java 下载，ImageIO 读取 PNG/JPEG/GIF/BMP 尺寸，WebP 读取格式头；仅提取尺寸，不做完整像素解码校验。文件字节的 SHA-256 摘要与模型几何参数共同生成模拟特征身份，不执行神经网络或物化 embedding 张量，不依赖 Python、Pillow 或解码子进程。未知格式明确拒绝，不回退到请求中的尺寸提示；需要的新格式按实际输入补齐。

V 仍按 URL/预处理 key 缓存和合并在途请求，命中时不重新下载；同 URL 内容变化需等缓存失效后重新读取。同一份文件通过不同 URL 获取时，在模型几何参数相同的条件下产生相同模拟特征身份；不同文件编码不保证 hash 相同，也不承诺与真实权重的 hash 数值一致。本地 `mock.vit.skip.download=true` 使用输入宽高和原有模拟身份，其余调度与计算路径相同。

P 在 GENERATE 阶段发送 `prefill_kv_handoff.computed_kv_end`，为已计算 prompt 的排他边界，不包含首个采样 token。D 校验请求身份、已 LOAD 阶段和边界；缺省字段保留原生旧调用兼容，显式错误断言拒绝。未实现的 logprobs 请求明确返回 UNIMPLEMENTED，不伪造概率输出。

本地统一池固定采用 LRU；不再接受 `device_eviction_policy`。

缓存元信息由 `engine/cache/CacheMetadata` 描述，来自 KvCacheManager/BlockTree 已提交的本地有效块。提交、淘汰和清空推进版本。Master 的原生 GetCacheStatus 返回可复用集合与容量，版本未改变时省略 key。

`KvCacheManager` 拥有并管理 Reporter 的配置、启动、查询和关闭；LLMMockEngine 只管理缓存管理器的生命周期。报告线程退出和 HOST_DOWN 在缓存锁外完成，结束后再清理 KV。

`engine/cache/KvcmReporter` 使用同一来源，按 real Meta HTTP `/api/registerInstance`、`/api/reportEvent` 编码注册、NODE_REGISTER、HEARTBEAT、BLOCK_SNAPSHOT、BLOCK_ADD、BLOCK_DELETE 和 HOST_DOWN。当前 Full-KV 沿用本地 HBM 元信息命名空间发布统一池的有效块；扩大统一池后，这些事件表示本实例可恢复的逻辑缓存，不表示每个 key 都实际驻留 device，也不模拟分层位置。RTP 装配方提供注册模型、布局和 spec，Reporter 管理发现与传输资源。

变更使用有界内存队列（100000 条），每次最多取 1000 条，按同 key 最后一次转换合并，避免 Meta 先处理 ADD 再处理 DELETE 导致结果逆序。注册和增量由同一个执行者每 20ms 处理，快照与心跳分别使用独立执行者，慢快照不阻塞 ADD/DELETE 或心跳。它们均不占用 EngineLoop 线程。

快照在缓存锁内一次捕获 HBM long[] 与版本，在锁外流式 JSON 编码，不构建逐 key JSON 对象树；仍保留完整请求体，并非零拷贝。对齐原生 Publisher：快照失败（包括限流）丢弃本轮，不重放、不暂停增量；下一次周期或服务端恢复请求捕获新快照。重新注册后旧快照回包不影响新注册，增量不按快照 ACK 过滤。注册和增量失败按原生默认 500ms→16s 指数退避；不使用 retry_after_ms 暂停整个数据通道。控制请求超时 1.5 秒，快照上传超时 30 秒。KVCM 响应先汇总 header 与全部 item 的恢复信号，再判断是否失败，前面的普通错误不会遮蔽后面的注册失效。心跳连续 snapshot_required=true 只触发一次，false 或重新注册重置。正常心跳或非空增量 ACK 启动/刷新 60 秒恢复窗口；超期后的心跳失败触发一次重新注册，注册和快照 ACK 不启动或续期该窗口，避免恢复风暴。队列溢出和增量失败请求新快照，不维护磁盘事件历史。关闭停止快照和心跳、收尾增量执行者，再尽力上报 HOST_DOWN；意外退出由服务端心跳失效处理。快照间隔 `kv_cache_event_snapshot_interval_ms` 必须为正。

Master 与 KVCM 是两个独立消费者，不要求跨服务原子提交。KVCM 不可用时不会阻塞调度或 Master RPC；最近错误、注册状态和待发送数量可从 Engine 端点的 kvcm 状态查看。事件位置表示 Engine 当前持有的本地缓存。
```yaml
kv_cache_event_publisher_type: kvcm
kv_cache_event_manager_endpoint: http://<meta-http-host>:<port>
kv_cache_event_instance_group: mock-experiment
kv_cache_event_instance_id: deployment_128
kv_cache_event_snapshot_interval_ms: 600000
```

manager_endpoint 是显式 Meta HTTP 覆盖地址，优先于发现；未配置时，`engine/cache/KvcmDiscovery` 使用 kvcm_enable_vipserver、kvcm_vipserver_domain 或 kvcm_server_address，通过原生 GetClusterInfo 发现 Meta HTTP leader。成功地址缓存到失败后失效，依次尝试 bootstrap 候选，不做周期轮询。instance_id 必须显式配置为共享查询空间；host_ip_port 缺省为发布 IP 加 gRPC 端口减一，与 DSV Master 身份一致。正式环境需使用匹配的 model_name 和实例组。

`ExecuteFunction` 采用原生 FunctionRequest/Response：READ/WRITE 当前未提交任何外部 I/O，返回 FAILED_BUFFER_SAFE 且 payload_buffer_safe=true；其他未实现的 KVCM SDK 元数据操作返回 UNIMPLEMENTED。ShutdownRequest 的跨 rank 停机协调未模拟，返回 UNIMPLEMENTED，不生成排空证明；实例停止继续使用已有的进程生命周期和 TERM 清理路径。

当前缓存明确使用统一逻辑容量，只注册一个 Full spec；不建模 DSV4.1 Full/SWA/Linear 多资源混合池。GDR、真实远端 payload 读写及完成 Decode KV 向远端 DRAM 的提交均未实现。注册/事件接线测试不等于这些能力已经验证，不能用本模型给出上述容量和真实命中结论。端点能力明确标记 payload_io=false。

## Client 原生收发核心

`NativeClient` 接收 FlexlbScheduleRequestPB 和单请求总 timeout。每次请求固定一份主备快照，先向主节点 Schedule，传输失败时最多向备节点尝试一次；收到业务响应、deadline 超时、主动取消或进入 Engine 取流后都不重发。要求成功响应及有效 P 地址。已入队走 FetchResponse，其他走 GenerateStreamCall，并写入 Master 返回的 P/D/V 路由。不提供不 Fetch、不消费完整流、Engine 代报 TTFT 或 schedule-only 模式。

每个 Client JVM 独立运行 MasterDiscovery：每秒刷新 topology.json 并 POST 真实 `/rtp_llm/master/info`，单次心跳 500ms、最多 4 个并发探测；连续两次失败判不健康。对齐 real Frontend `server/host_service.py` 的选择规则：优先健康且声明自己为主的节点，否则保留仍健康的原目标，再按 IP 选择健康候选；备用节点从未声明为主的其他健康节点中选择。发现暂时漏掉的健康节点仍继续探测，失败后停止保留，不再发现且已不健康的记录在最后探测 30s 后清理。文件损坏或非法端点属于执行错误，会终止整个 Client 播放；没有健康 Master 是请求失败，不直连 Engine，也不按列表轮流广播。

发送边界对齐 `server/master_client.py`：只有 Schedule 的传输失败尝试一次备节点；业务拒绝、deadline 和取流失败不尝试其他 Master。整个请求仍受同一个总 deadline 约束，重试消耗剩余预算。这个策略不承诺 exactly-once：响应丢失后原主可能已经接收请求，保留与 real Frontend 相同的传输重试风险。Client QPS 仍按逻辑请求 issued 计一次，备节点尝试不会再发放流量预算。

每次只预取一个响应，验证身份、INT32 载荷/形状、递增长度和终态；最终长度必须达到 min_new_tokens，ignore_eos 时必须等于 max_new_tokens，提前 finished 不能冒充固定长度回放成功；finished 后仍须正常结束，空流/截断/终态后额外帧均失败，不保留全部输出。firstToken 从发送的单调时钟到实际收到首 token，包含调度和网络；没有首 token 就没有数值。completion 记录发送墙钟、可空 TTFT、总耗时、收到 token 数及终态/错误码；后续失败保留已观测 TTFT，不读取 aux 耗时代替。

同一 deadline 覆盖调度和取流，并受更短原生 timeout 限制。取消/错误/超时结束当前 RPC，向该请求最后尝试的 Master 尽力发一次有界 Cancel，不随新的选举结果改变取消目标；Cancel 失败写日志，不承诺远端释放。连接复用、请求上下文独立，close 取消活动并等待有界 Cancel 后关闭所拥有连接。

协议测试使用夹具，真实接线测试使用生产 Master，二者都真实消费 P/D 输出。LoadClient 回放、client-0 协调和 start/stop 已接通，主备发送只改变 Schedule 目标选择与传输失败重试边界。Client 逐请求执行记录已接通；回放源支持 JSONL 和 Protobuf＋Zstd，worker 分片使用 Protobuf＋Zstd，仓库不提供旧采集格式转换工具；Engine 与 Master 的完整服务端证据仍未接通。

## 指标、采集和查询

RTP 后端的 RtpMetrics 使用 Prometheus simpleclient 0.16.0，每个节点独立注册表，`GET /metrics` 标准文本导出。事件发生时更新累计量，抓取时读取瞬时量，没有常驻采样线程。业务标签保留 role；原生 TPS 按请求 priority 分桶，单 DP worker 的 dp_rank 为 0；Prometheus 为每个采集目标设置 instance=实际 RPC 服务地址，KV 分池使用 pool_name=device/host、tier=DEVICE/HOST，额外维度仅下表枚举，不放 request ID、cache key 或异常文本。

| 指标 | 类型 | 含义 |
| --- | --- | --- |
| rtp_llm_requests_started_total | Counter | 校验并接纳后加一，入口拒绝不计。 |
| rtp_llm_framework_requests_total | Counter，outcome=success/cancelled/timeout/error | 每个已接纳请求唯一终态，节点关闭计 error，不代替 Client 成功率。 |
| rtp_llm_host_kv_tokens | Gauge | 本地 Host total/free token 容量，与 Device 分开。 |
| rtp_llm_cache_lookup_tokens_total / rtp_llm_cache_hit_tokens_total | Counter | 分配时实际输入/历史命中 token，取消不撤销已发生查找。 |
| rtp_llm_forward_tokens_total | Counter，kind=input/compute/output | 完成 Forward 的 token；input/compute 汇总本批全部执行成员，output 只计有效提交。计算完成但成员已取消时仍计 input/compute，output 为零；计算中止不补完成样本。 |
| rtp_llm_model_forward_us | 事件：Prometheus Summary / KMonitor Gauge | Prefill 每次完成 Forward 的原始耗时样本，微秒。当前生产 MTP 兼容口径下 Decode 缺席。 |
| rtp_llm_step_latency_us | 事件：Prometheus Summary / KMonitor Gauge | P/D 完成的非空执行步耗时，微秒，从调度返回到处理完本轮输出。 |
| rtp_llm_context_tps / rtp_llm_context_tps_with_cache | Gauge，priority | 未命中计算量 / 对应有效执行秒数；with_cache 使用全部输入量。各 priority 共用整批执行时间。 |
| rtp_llm_generate_tps / rtp_llm_total_tps | Gauge，priority | Real 原生合同是报告窗口内的 token 数，不是 token/s。D 使用停止截断和取消过滤前的验收量；P total 使用计算量。 |
| rtp_llm_context_tokens / rtp_llm_context_tokens_with_cache | Gauge，priority | 与 context TPS 配对的报告窗口 token 数。 |
| rtp_llm_context_wall_tps / rtp_llm_context_wall_tps_with_cache | Gauge，priority | 对应 context token 数 / 实际报告窗口秒数。 |
| rtp_llm_wall_tps_report_interval_us | Gauge | P/D 实际 TPS 报告窗口，微秒；不按 priority 重复计数。 |
| rtp_llm_context_forward_seconds_total / rtp_llm_context_forward_with_cache_seconds_total | Counter | 与 context TPS 同一执行边界的累计秒数；不是仅模型 Forward 时间。 |
| py_rtp_vit_embedding_tps | Prometheus `_total` Counter / KMonitor 累计速率 | 成功实际计算的 embedding token 累计量，Prometheus rate 查询得到 token/s；缓存命中和 in-flight 去重不重复计数，失败/过期不计。 |
| rtp_llm_sp_estimate_tpot_us | 事件：Prometheus Summary / KMonitor Gauge | Decode 本轮执行耗时 ÷（模拟验收 token 总数 / 本轮参与 stream 数），微秒；不使用停止截断后的交付量。 |
| rtp_llm_gather_model_input_us | 事件：Prometheus Summary / KMonitor Gauge | P/D 构造批输入的耗时，微秒；不含选批、KV 准备和模拟 Forward。 |
| mock_client_tpot_seconds | Histogram | 成功请求收到首尾 token 的时间差，除以首 token 后的 token 数。单 token 请求不采样。 |
| mock_model_forward_actual_us | Gauge | P/D 最近一次完成 Forward 的模拟实际耗时，微秒，用于与计划值对照。 |
| mock_model_forward_planned_us | Gauge | 同一个完成 Forward 的计划耗时，微秒，包含公式和可选噪声。 |
| rtp_llm_model_forward_seconds | Histogram | 完成 Forward 实际耗时秒，含 bucket/sum/count，排除排队，不是 TTFT。 |
| rtp_llm_scheduler_requests | Gauge，state=waiting/running/loading/ready | 调度器报告的四类互斥请求数。 |
| rtp_llm_kv_tokens | Gauge，state=total/available/free/held | token 容量，available 含可驱逐缓存，不等于 free。 |

TPOT 的计时从选批和 KV 准备完成后、开始构造批输入时开始，到模拟计算完成回调进入输出处理前结束。它独立于原有 `rtp_llm_step_latency_us` 的“调度返回到处理完输出”边界。分母使用现有简化 Decode 模型给出的本轮验收 token 数，对应 Real 验证阶段 `accept_len` 的采样位置，不新增 draft、verify 或 acceptance 内部指标。EOS、max_new_tokens、max_seq_len 限制只截断交付量，不回写本轮验收统计。选批前已取消的 stream 不参与；入批后取消但计算已完成的 stream 仍参与该批统计，其 token 不交付；整轮计算中止而无完成结果时不产生 TPOT 样本。Prefill 不产生该 Decode 指标。

RTP 原生观测直接导出：`rtp_llm_wait_stream_size`、`rtp_llm_running_stream_size`、`rtp_llm_kv_cache_left_seq`（Tokens）、`rtp_llm_kv_cache_{total,free,available,used}_blocks`（Blocks）和 `rtp_llm_kv_cache_used_ratio`（0–100，不可用比例）。Host 暂只上报 total/free/used Blocks，不用 free 冒充 available。loading/ready 仍是调度器阶段辅助观测，不冒充 real 的加载缓存队列。

Prefill 的 context TPS 用计算 token／执行时间，含缓存 TPS 用输入 token／执行时间；执行边界从本批调度开始到处理完输出，与辅助累计执行秒数一致。墙钟 TPS 使用对应 token／实际 TPS 窗口。Decode 的 generate 和 total 保留 Real 的窗口计数，使用本轮验收 token，包含完成计算后才取消的成员，不受 EOS、最大输出长度截断；已提交输出独立保留在 Forward output Counter 中，用它计算真实交付吞吐。不得把原生 generate/total 当作 token/s，也不得对它们求 rate。窗口计数的可比性依赖实际报告间隔，跨部署不能直接比较不同长度窗口的原始数值。

TPS 的各 priority 共用整批有效执行时间和墙钟窗口，分桶分子相加等于本实例总量。空闲报告 priority=0 的零值；仍有 Forward 在执行且没有新的完成结果时不报假零、不重置 TPS 窗口，直到完成或失败后结束这段窗口。失败不虚构执行 token。QPS 有独立的周期边界，终态统计不受长 Forward 延迟影响。`rtp_llm_framework_qps` 统计终态，不是接纳；error 包含 timeout，cancel 单列。Real 与 Mock 周期边界可能不同，不能要求逐点相等。

没有 real 同名指标的真实观测，以 RTP Prometheus 辅助 Counter/Histogram 保留供 Python 门禁和统计使用，不拿它们的累计值直接冒充线上 QPS/TPS Gauge。理论 Forward、ViT 运行图片数和 hash 缓存预算保留 `mock_` 前缀。此次更名不保留旧 `mock_engine_*` 别名；旧实验保持原始 view.json 和 TSDB，不承诺新查询可以直接读取旧指标名。

P/D 已补 `rtp_llm_{context,generate}_batch_size`、执行 Token 数、最长序列、逐请求等待/首 Token/总延迟、输入/输出长度与迭代数；`rtp_llm_stream_loading_cache_latency_us` 只统计 allocator loading 阶段，PD 协议等待不再混入。Query batch 的 native 语义是 beam 数，不能用调度批大小代替。

ViT 使用相同的物理元数据与 Prometheus instance 发现；QPS、Embedding 队列/等待/Forward/合批、请求图片数/输出长度，以及缓存命中/未命中/在途合并/淘汰和逻辑驻留量均按 real 名称导出。实际读取时提供 rtp_llm_vit_image_fetch_rt_us（μs）；成功完成图片准备后采样总耗时、下载及其余耗时（ms）。其余耗时包括读尺寸、内容摘要和几何计算。跳过读取时这些指标无样本。py_rtp_vit_preprocess_queue_size 是尚未完成准备的工作数（含排队和执行），py_rtp_vit_embedding_rt 仅计成功工作的就绪排队加 Forward，py_rtp_vit_request_rt 覆盖阻塞媒体调用的成功/失败全程，包括缓存命中等待但不含 RPC 传输和输出编码。未实现的像素预处理和 CUDA Graph 指标不补零。详见 [指标覆盖与边界](../views/README.md)。

事件指标统一通过 EventMetrics.observe 记录。Prometheus 使用无客户端分位数的 Summary，导出累计 `_sum` / `_count`，不再导出同名窗口均值 Gauge；抓取和空闲期间不清零。单实例和跨实例均值均使用 sum/count 的区间增量比值，跨实例先分别汇总，再相除。KMonitor 在事件发生时以原名逐样本提交，由 SDK 保留样本数、总和及极值，不再从周期快照提交事件均值或累计辅助量。SDK 出口在构造时选定，没有全局 observer 或原始样本队列；业务埋点不区分 SDK。周期线程只负责状态、原生窗口计数和已计算的速率。Prometheus 的事件 QPS 和 ViT embedding TPS 导出原名加 `_total` 的累计 Counter，由查询 `rate(...[1m])` 计算每秒速率；读取不清零、不推进窗口，多次抓取不会消费事件。KMonitor 仍以原名提交原始计数，由 SDK 计算速率；原生 TPS 窗口计数和已计算的速率快照保持 Gauge。原有 Forward 最近值与 Histogram 口径保持不变。 下表的 `engine/decode/tps` 用于衡量按墙钟计的已提交输出 token 速率。

累计量预建 0，抓取不清零；分母为 0 不补比率。以下查询在 Python `execution/step/` 的对应指标接口中实现，直接调用本次托管 Prometheus：

| 接口 | 口径 |
| --- | --- |
| engine/cache/hit_rate | P 历史命中 token 增量／输入 token 增量，token 加权；可选 targets。 |
| engine/prefill/tps | 原生 rtp_llm_context_tps：每个采样点先合并实例内 priority，再取实例区间均值之和。 |
| engine/prefill/tps_with_cache | 原生 rtp_llm_context_tps_with_cache：先合并 priority，再取实例区间均值之和，输入含命中。 |
| engine/decode/tps | D 完成 Forward 输出 token 增量／墙钟时间。 |
| client/success_rate | 必填 range 区间内成功终态计数增量／全部终态计数增量，先对各 Client 算增量再聚合；可选 targets。 |
| prometheus/query | 固定 query_time 的一次 scalar 或 float instant vector，保留标签、数值和 sample_time。 |

普通接口在本次独占 Prometheus 内按 role 和 instance 地址过滤，先逐条 Counter 算增量再聚合以处理 reset；hit_rate 不能平均节点比率。显式空 targets 报错，选择器转义；每个选中实例的分子/分母至少需要两次采样，重复或缺失序列报错。range 在固定 end 执行单次 instant query，窗口=end−start，精度 ms，低于 1ms 拒绝；latest 用检查前 60s 内最近两次采样的 irate，不平均整分钟。缺少采样或没有可计算 Forward 时报错。

`client/success_rate` 的 range 是必填的 `[start, end]`（start < end <= 0），按固定查询时刻定位窗口，使用 Counter 的 increase，按完成时间统计，分母包含 success/error/timeout/cancelled。它没有最小请求数门槛；窗口无终态时分母为零，或任一 Client 的终态序列采集不完整时均报错。至少两次抓取是计数器增量的计算条件，不是请求数要求。近期完成请求可能在更早时发出；此口径不表示同一发送阶段的成功率。

`cases/master_ha_failover/case.yaml` 使用已转换的 DeepSeek trace，按 YAML 中的 `replay_speed` 回放贯穿杀 A、恢复 A、杀 B、恢复 B，检查结束后提前停止。操作通过已保存的地址列表引用，恢复前不杀另一侧；恢复后检查原主不被抢占。Case 在保守等待后直接检查最近完成请求的成功率，最后收集 Client 终态，不使用进度接口、最少样本数、精确事件窗口或 HA 专用执行器。启动配方仍由 `mock-case launch <case>/case.yaml` 生成；双 Master 要求本机可绑定两个回环 IP。该 Case 沿用原 DeepSeek preset 的本地块容量和 reuse_cache，已移除 Host、远端和水位配置。参数来自 legacy_unverified preset，配置和数据迁移不代表端到端验收通过。

原始 PromQL 不自动加过滤。matrix/string/native histogram、非有限值、查询 warning 或后端失败都不作为有效门禁；普通指标恰好一个有限值。原始查询返回结构化结果，不隐式聚合或对各样本套用数值门禁。查询不重试，HTTP 和求值均受步骤剩余预算限制。

### Runner 托管 Prometheus

启动配方可加入：

```yaml
prometheus:
  binary: /path/to/prometheus
  scrape_interval: 1s
  startup_timeout: 30s
```

Runner 不下载 binary；默认 scrape 1s、startup 30s，目前验证 Prometheus 3.5.0。它在独立运行目录启动采集器，从结构化日志读取系统分配端口，URL 仅保存在 Python 运行上下文与采集产物中，不向 Engine 配置注入查询地址。

组件 ready 后必须提供合法 metrics URL。Runner 原子更新 file discovery，初始节点全部成功抓取两次才执行 phase；新增 Engine 完成两次基线采样后扩容操作才返回，下线前完成最后采样，再从活动 targets 移除，不另建采集历史台账。每次独立 TSDB、job=mock，不要求用户定义实验 ID。当前 Python 查询只支持 Runner 托管 Prometheus，不接受组件内 prometheus.url/job 作为备用入口。

收尾等待每个目标在收尾开始后成功采样，再停止组件、ZK，最后 TERM Prometheus，超时才 KILL。最后采样失败、采集器异常退出会标记数据不完整并令实验 ERROR。启动失败/取消仍清理并保留现场。

prometheus/ 保存配置、targets、版本、日志、data/TSDB/WAL 和 collection.json；complete 仅说明采集边界及关闭成功，不代替业务指标或中途 up 判断。门禁 FAIL/主动取消也可以有完整采集。曲线不导出 JSON，数据不随收尾删除，后续回放服务在线查询；目前仍用默认保留策略，没有长期归档或正式报告服务。已完成实验可使用保存的大盘快照通过 View 回看，详见 [View 指南](view-guide.md)；Client 已提供 issued/finished Counter 及 TTFT/请求耗时 Histogram。

### RTP 调度实现边界

调度器内单个请求对象统一称为 `GenerateStream`，沿用 real 的生成请求术语，保存输入、KV 引用、生成进度和协议回调；局部变量使用 `stream`。完成历史中的版本化记录称为 `FinishedTask`，不再使用含糊的 `Entry`。

`AbstractRtpScheduler` 共享请求状态、KV 引用、PD 生命周期、Forward 提交和取消，并提供运行状态和指标采样源；`RtpFifoScheduler` 仅做普通 FIFO 准入，`RtpPrefillGroupScheduler` 单独拥有显式组队列、残余成员和普通/组之间的选择。组标识不改变普通请求的调度方式。选择在 Engine 启动时完成，不从 Master 配置推断，也不在运行中按请求切换。`fifo` 模式应配合 Master 的 `NON_BATCH` 交付；它仍会动态组成多请求 Forward batch。两种调度器共用 `KvCacheManager.prepare()`。本地分配同步返回。Java 装配可给 Prefill 注入异步前缀恢复源，加载失败时 KV 层保留本地命中并返回可计算的租约。默认启动使用本地池，KVCM 上报仍只处理元数据。

`LLMMockEngine` 装配指标导出，采集只短暂复制调度状态；指标构造与 SDK 发送在业务锁外。`RtpMetrics` 读取调度状态快照并按各指标的口径采样。

### KVCM 元信息

KVCM Reporter 发布有效块事件、快照和心跳。它不承担 KV payload 读写。旧的 Host 容量、远端缓存开关和水位配置应从 Mock 配方移除。

### 状态与耗时口径（对照本分支 real）

real 的依据为 `cpp/engine_base/stream/GenerateStream.cc`、`GenerateStateMachine.cc`、`cpp/model_rpc/RpcServerRuntimeMeta.h`、`LocalRpcServer.cc`、`cpp/metrics/RtpLLMMetrics.cc` 和 `cache/block_tree_cache/storage_backend/kvcm/KVCMStorageBackend.cc`。

- WorkerStatus 的 `phase/is_waiting` 描述当前任务：缓存准备中、有 Lease 为 KV_ALLOCATED/is_waiting=true，首次进入 Forward 为 RUNNING。real LocalRpcServer 未填写 waiting_query_len/running_query_len，Mock 同样保留 protobuf 默认 0；不能用这两个字段判断在途请求，应读任务列表。
- 活动 TaskInfo 的 execution_time_ms=-1、waiting_time_ms=0，终态才提交实际耗时。waiting 不是累计队列驻留时间：每次离开 WAITING 更新为请求起点至本次转换的时间；正常加载后再次进入 RUNNING 时，包含加载和重新排队。若在 LOADING_CACHE 内取消，保留此前离开 WAITING 时的等待值。execution_time_ms=终态总耗时−waiting_time_ms，因此取消加载的执行耗时可能包含加载时间，不能解释成纯计算。D 的正常 GENERATE 开始重置执行计时起点，原请求 deadline 不变。
- `rtp_llm_wait_stream_size` 仅统计等待调度队列；`rtp_llm_loading_cache_stream_size` 仅统计 allocator CACHE_LOADING，不含 D 的 ALLOCATE/LOAD 协议等待。通用 rtp_llm_scheduler_requests 的 loading 仍可包含协议等待，两者不是同一语义。
- `rtp_llm_wait_latency_us` 使用上述冻结的 waiting；`rtp_llm_stream_enqueue_to_canrun_us` 为首次调度准入之前的等待，`rtp_llm_stream_canrun_to_running_us` 包含缓存准备到首次运行；`rtp_llm_stream_loading_cache_latency_us` 为进入加载到调度器观察完成，`rtp_llm_stream_load_done_to_running_us` 为观察完成到首次运行。单位均为微秒；按 real 的终态触发和 REPORT_GAUGE 行为省略零样本，不把 PD 零耗时传输的 RPC 等待冒充缓存加载耗时。

### 缓存与远端指标

| 名称 | 单位／触发边界 |
| --- | --- |
| rtp_llm_device_input_length | tokens，完整输入 key 数乘 block stride，不是原始输入的未取整长度。 |
| rtp_llm_stream_cache_device_reuse_length / memory_reuse_length / remote_reuse_length | tokens，缓存准备成功或安全回退时的有效来源；查询命中未读成功不算 remote reuse。 |
| rtp_llm_kv_cache_load_qps / load_fail_qps | 次/s，异步准备结算；失败回退或取消计失败，纯本地立即完成不计异步 load。 |
| rtp_llm_kv_cache_load_prepare_latency_us / load_wait_latency_us | 微秒，同步准备耗时／准备返回后至调度器结算的等待。 |
| rtp_llm_kv_cache_match_to_ready_latency_us | 微秒，准备起点至结算；保留 real 的 load_attempted=true/false 标签。 |
| rtp_llm_remote_match_qps / match_fail_qps / match_time_us | 次/s、微秒，实际调用远端 match；失败和无匹配分别处理。 |
| rtp_llm_remote_match_reuse_block_num / remote_valid_hit_ratio | 远端匹配块数／(本地+远端匹配块数)/请求 key 数，ratio 为 0–1；非零远端匹配时才记录 ratio，尚不保证 read 成功。 |
| rtp_llm_remote_read_qps / read_fail_qps / read_task_cost_time_us / read_token_num | 次/s、微秒、tokens；按实际 read 调用，只有成功读取才记录非零逻辑 tokens，不乘 TP 或 payload group 数。 |
| rtp_llm_remote_write_qps / write_fail_qps / write_task_cost_time_us / write_cache_block_num | 次/s、微秒、blocks；包含 write 完成确认。适配器返回实际成功发布块数，已存在可返回 0；后台写失败可见，不改变已完成请求结果。 |
| rtp_llm_kv_cache_load_ref_blocks / store_ref_blocks / active_blocks | blocks，Device 池拉取当前状态；统计去重块数，不是引用次数；取消请求后未停止的 I/O 仍占用引用。沿用 pool_name/tier 标签。 |
| mock_remote_cache_in_flight / mock_remote_cache_rejected_qps | 次、次/s，operation=read/write；Mock 自身逻辑在途任务、未实际调用后端的任务。无完全对应的 native 统计边界，所以保留 mock 前缀。 |

延迟/长度事件沿用上述 EventMetrics 规则；Prometheus 的累计 sum/count 保留事件权重，KMonitor 接收原始样本。没有真实 SDK、StartWrite/FinishWrite 分段或 payload 时，不导出对应 SDK、写会话分段、字节或带宽数据。指标回调失败不改变 I/O 结果。

请求类零值规则对齐 Real 的 REPORT_GAUGE：等待/首 token 耗时、输出长度、迭代次数和 reuse_length 为零时不产生样本。KV 复用类使用直接 Report，kv_cache_reuse_length 等仍包含零命中样本；不能统一过滤零值。

### 输出交付边界

Scheduler 只发布累计进度，删除逐轮输出等待标记、网络可写性筛选及 OUTPUT 调度阶段。`Listener.progressed(request)` 是可合并的锁外变化通知；RPC 读取不可变的 `request.progress()`，计算不依赖通知完成。正常结束和 KV 释放发生在计算侧，最后一个进度通知排在终态通知之前。

`GeneratedOutput` 替代消息缓冲队列，输出积压状态为 O(1)，按 generated/sent 游标即时生成固定合成 token，单帧最多 4096 token。D 不提前构造每轮响应。P 转发 D 的帧时只保留累计长度与最新元信息；传输 Inbox 的两帧受控拉取仍有固定上限。积压进度允许合并，不能用收到的帧数推断 Forward 次数。终态错误优先于尚未发送的数据，成功必须先补齐数据再结束。

GENERATE 后 D 不因 P 输出流断连或取消而停止；初始 ALLOCATE/LOAD、D 自身超时、显式调度取消和节点关闭保持其原有约束。该假设只用于合成 Mock，不模拟链路故障导致的生产行为。

计算结束后，RPC 为剩余输出保留固定十分钟交付期限（独立于 Fetch 接入期限）。到期移除输出注册记录，并通过串行发送路径以 DEADLINE_EXCEEDED 结束连接；不调用计算取消。正常交付或连接取消时撤销计时器。若底层 onNext 尚未返回，注册清理仍执行，错误通知等该调用返回后发送，不并发调用 observer。

LLM Mock Engine 的装配类统一命名为 `execution/LLMMockEngine`，缓存管理类为 `cache/KvCacheManager`。`execution/EngineLoop` 主链使用 enqueue → loop/step → schedule → forward → processOutputs，沿用 real 的普通执行术语。请求控制入口为 updateRequest，KV 回调为 onKvTransferCompleted；RPC 输出发送不参与 Forward 就绪判断。


### ViT 指标口径

对照本分支 real 的 `mm_process_engine.py`、`mm_scheduler.py`、`mm_preprocess.py` 和 `mm_embedding_cache.py`。这是源码合同对齐，不代表已核验 TH 生产镜像的版本。当前异步缓存路径按去重后的单媒体 entry 执行；多图 RPC、内部工作、Forward batch 是三个计数单位。

| 指标 | 样本与边界 |
| --- | --- |
| `py_rtp_vit_qps` | `source=mm_embedding`；唯一未命中工作真正开始执行时加一。多图可产生多个工作；缓存命中和 in-flight 消费者不加，异步提交启动的工作也计入；排队期间过期不补 QPS。 |
| `py_rtp_vit_success_qps` | 唯一工作成功发布一次；无 source 标签。缓存重复消费不加。 |
| `py_rtp_vit_error_qps` | 首次观察到失败时计一次；entry 标志防止共享结果的不同异常重复计数，异常标志防止工作与 RPC 层重复计数。入口校验和超时也可能计入错误，因此不以 success/QPS 代替客户端成功率。 |
| `py_rtp_vit_request_rt` | ms，`source=get_embedding_result`、`status=success/error`；阻塞调用入口到返回或失败，包含下载、共享等待，不包含异步 ACK、RPC 传输、输出编码。纯 hash 命中不调用该路径；hash 缺失只处理缺失媒体。 |
| `py_rtp_vit_image_count` / `py_rtp_vit_embedding_length` | 每个阻塞调用的图片数 / 成功返回的 embedding 行数。缓存返回仍记录；异步提交 ACK 不记录。 |
| `py_rtp_vit_embedding_tps` | 成功实际计算的 embedding 行数，SDK 原生累计后折算速率；Prometheus 导出 `py_rtp_vit_embedding_tps_total`，查询 `rate(...[1m])` 得到墙钟每秒速率。失败、缓存和共享消费者不重复计入。 |
| `py_rtp_vit_embedding_batch_size` / `forward_rt` | 每次 Forward 的工作 chunk 数 / 实际模拟执行毫秒。不是 RPC 数，也不是 embedding 行数。 |
| `py_rtp_vit_embedding_queue_size` / `queue_wait_rt` / `embedding_rt` | 等待 Forward 的队列项数 / 首次入批前等待毫秒 / 成功工作提交 GPU 队列到 Forward 完成毫秒。最后一个不含下载和调用者晚取结果的时间。 |
| `py_rtp_vit_embedding_overload_qps` | GPU scheduler 拒绝提交次数；CPU admission 拒绝走错误指标，不混为 GPU 队列溢出。 |
| `py_rtp_vit_preprocess_queue_size` | 尚未完成准备的媒体数，含排队和执行中；不含等待 Forward 的工作。 |
| `py_rtp_vit_preprocess_rt` / `download_rt` / `preprocess_other_rt` | 成功准备的总毫秒 / 下载毫秒 / 两者之差；跳过下载不制造样本。Mock 非下载部分只有读取尺寸、摘要及几何规划，没有真实像素处理。 |
| `rtp_llm_vit_image_fetch_rt_us` | 成功下载微秒；当前 DeepSeek V4.1 图片规划使用 `model=deepseek_v41,mm_type=image`。 |
| `py_rtp_vit_embedding_cache_{hit,miss,inflight,eviction}_qps` | 每 key 获取时区分驻留命中、新工作、共享等待；eviction 只计逻辑结果被容量淘汰，GPU→CPU 保留不计。GPU 淘汰且 CPU 放不下也必须计。 |
| `py_rtp_vit_embedding_cache_tokens` / `bytes` | GPU+CPU 逻辑 embedding 驻留量。Mock 用形状计费，不代表真实 Tensor、GPU 内存或 JVM 物理内存。 |

本地 View 显式筛选上述 source；请求耗时合并成功/失败的原始 sum/count 后求平均，不平均实例均值。没有模拟的 CUDA Graph、GreenNet、预处理进程 IPC、像素 decode/resize/processor、真实 Tensor 数据面和代理转发指标保持无样本，不用估计值伪装真实测量。RPC handler/lifecycle、payload bytes 不由 engine 请求耗时或逻辑缓存 bytes 冒充。


### 状态 RPC 的 RT 与 QPS

`rtp_llm_rpc_cache_status_total_rt_us` 从进入查询计到缓存快照返回，包含缓存锁等待、版本检查和必要的 key 复制；不包含 protobuf 构建、响应提交、网络传输或指标上报。边界对照 real `LocalRpcServer::getCacheStatusInfo`。

`rtp_llm_rpc_worker_status_total_rt_us` 计到缓存容量和任务状态采集完成；同样排除后续 protobuf 构建和响应提交，对照 real `GetWorkerStatus` 的 `request_after_ws_time_us`。这两个指标均不是客户端观察的完整 RPC 延迟；计时边界修正不代表消息构建、序列化的实际成本消失。

对应 QPS 保持每次状态查询计一次。对比部署时先区分每实例和集群总量，再核对 Master 数量、轮询周期及重试机制，不通过调整 QPS 补偿不同轮询负载。


`rtp_llm_rpc_onflight_request` 对照 real `GenerateContext::collectBasicMetrics`：生成请求结束时，采样当前实例尚未退出的生成请求数，包含正在结束的本次请求，然后释放该请求的计数。P 覆盖本地生成以及等待 D 的整个 P/D 请求，批量 Enqueue 按每个子请求计，不按批 RPC 次数；Fetch 不另加一次。D 从 RemoteGenerate 打开计数，RPC 取消/断连时释放；已接受 GENERATE 的计算仍按既有规则独立继续，后续计算结束不能重复释放。成功、失败、取消都只释放一次；health、WorkerStatus、CacheStatus 不计入。

这是 real 的终态事件 Gauge：SDK 收到原始样本，Prometheus 导出 sum/count，本地 View 先合并原始权重再求均值。它不是 Scheduler running 数，也不是定时采集的瞬时 RPC 并发；例如单个串行请求在结束时采样为 1，空闲时没有新样本，不应期待它自然上报 0。Mock 的输出交付已经解耦，逻辑请求终态与异步输出排空的边界不同，该指标不衡量待交付数据量或网络压力。

### 请求终止与 ViT 结果所有权

Scheduler/EngineLoop 的外部终止入口为 `abort` / `abortRequest`，必须提供失败原因；成功仅由内部计算和 P/D 状态转换决定。终止不等待输出交付，KV 使用权仍在持有时延迟释放资源。

ViT Cache Claim 和 Forward Work 只提供只读 CompletionStage；调用方转换得到的 CompletableFuture 是独立副本，完成或取消副本不会修改共享结果。Claim.close() 幂等，每个消费者只归还自己的引用，最后一个消费者离开才触发放弃处理。


### CP：只模拟缓存布局，不模拟 rank 通信

数据集只提供语义 token、媒体信息和期望输出，不知道 CP。本地回放用 Client 和 Mock P 的启动配置指定相同的 `seq_size_per_block`、`cache_key_hash_seed` 和 `prefill_cp_size`。例如基础块 128、CP4 时，Client 发 K4、K8，并将请求的 `cache_key_block_size` 设为 512。CP size 缺省时按 1，Client 缺省基础块大小时不生成路由键。`cp_rotate_method` 和 `tp_size` 仍用于已有的计算模型和 Prefill 准入。

Whale 使用真实 Frontend 时，需要让 Mock P 遵守真实 Frontend 的缓存路由合同：相同 `cache_key_hash_seed` 和基础 `seq_size_per_block`，并对齐 CP 分片配置。基础块为 128 时，Mock P 设置 `prefill_cp_size=4`，按 128-token 块滚动哈希并保留 K4、K8。媒体身份也在基础块边界参与哈希。P 的 KV 分配、复用、完整块提交、CacheStatus.block_size 和 token 容量统一使用 512 的逻辑边界；`test_block_num` 仍是每 rank 物理块数（含 null block）。有效配置额外显示只读的 `cache_key_block_size`。

P→D 仍传完整语义 token 和媒体 cache spans，ALLOCATE/LOAD/GENERATE 接口不新增 CP 字段，不传预先抽样的 P cache keys。D 从同一语义输入按自己的本地布局生成 key；即使设置 `prefill_cp_size=4`，它的 FULL KV 仍按基础块管理，与 real `KVCacheManager` 的 DECODE 分支一致。P 分片、D 不分片时，二者的 key 数和容量粒度不同是预期行为，不应强行设置成相同。

Mock 模拟的是 FULL KV 的 key/容量/复用边界及既有性能公式，不启动 CP rank，不模拟 collective、真实 KV payload 拆分或张量搬运。KVCM 注册的基础 block_size 仍保留 128，不能因 P canonical key 的跨度为 512 就改注册命名空间。未配置分片时保持原有行为；不会根据 TH 的某一套配置自动给所有部署开启 CP4。
