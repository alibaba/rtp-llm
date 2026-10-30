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

接线测试分别覆盖 DIRECT/NON_BATCH 的 Generate、QUEUE/FIXED_WINDOW/BATCH 的 Enqueue→Fetch。`tests/test_master_lifecycle.py` 在两种模式下持续运行同一 Client 计划，停止主节点、补回新 IP 的 Master、再停止旧节点；在旧请求 deadline 之后检查新主持续成功接流，并检查独立进程退出、各组件独立产物和当前采集目标。切换错误如实保留，不要求整个过渡期零错误。协议夹具测试和真实 Master 接线测试分开；用生产 pre_stop 钩子观察选举不等于完成 Master 生命周期动作。

## RTP 节点与公共校验

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
      compute.forward.expression: "param(baseMs, 1) + 0.01*totalComputeTokens"
```

原生字段支持 server_args 小写或 Whale 大写 env 名，同一字段同时提供两种写法报错；整数/布尔接受对应原生文本，其余原生内容保留，但不能据此宣称功能已模拟。当前要求显式 block size、max sequence length 和 test_block_num，只支持 `dp_size=1`、`dp_rank=0`；P/D 批量上限取 1–1024，默认 1。Mock 一个 Engine 表示完整计算体，不拆 CP/TP/DP rank 或模拟 rank 间通信；内部并行的影响由整 Engine 的性能公式、容量和调度配置表达。block size 是本 Engine 的逻辑分配粒度，PD 不根据 CP 数乘除。配置导入时应先统一容量口径。不加载模型层/head 或 GPU 布局，不从部署内存推导 KV 块数。

RTP 请求公共支持范围集中在 `RequestValidation`，本地单 token、P/D Prefill 及 Decode ALLOCATE 均使用它：adapter/unique_key、多 beam/多返回、stop words、sp_edit/sp_input_lookup、额外 logits/hidden/prob/loss 输出明确拒绝。路径只保留自身角色、长度、路由和协议阶段规则。测试通过实际 gRPC 同时验证 max_new_tokens=1/2 和 D ALLOCATE，防止参数只在一条路径被拒绝。

### 执行和 P/D 交接

- P 的 `GenerateStreamCall(max_new_tokens=1)` 走 LocalPrefill，不启用 D。请求经过 FIFO→选中时分配 KV→共享 Compute→真实等待→编码输出→释放；排队不占 KV。
- 多 token P/D 要求 `can_use_pd_separation=true` 且恰好一个有效 DECODE 地址。PrefillGeneration 先向 D ALLOCATE，确认后启动 P，随后 LOAD、发 P 首 token、等待 LOAD 确认、发 GENERATE 并转发 D 输出。
- D 的 DecodeAdmission 处理原生协议，RtpScheduler 管理资源和队列，只有 GENERATE 后进入计算队列；每个 Engine 一轮一个 Forward，按配置合并多个请求。初始 KV 分配遇临时不足等待容量变化；生成期间每轮先增长 KV，失败则终止，成功后才执行与提交输出。

PD 只执行调度器之间的交接，不模拟 KV 数据传输。P/D 只使用原生 RemoteGenerate TCP 连接，不注册自定义回调服务，也不跨节点比对 prompt、块布局或 cache keys。P 必须等自身 Scheduler 报告首轮 Forward 完成后才发送 LOAD；D 对已分配的请求执行零耗时异步传输事件，完成后回复 LOAD ACK。P 收到 ACK 才释放源 KV，然后发送 GENERATE 或 PREFILL_COMPLETE，D Scheduler 决定开始计算或结束。两端块大小各自独立，KV 的申请、持有、释放及 finished 记录仍归各自 Scheduler 管理。

接入层信任正常 P 生产者发送的 LOAD 和首 Token 完成事实，不模拟损坏的 KV 数据、假冒来源、迟到的跨请求交接或生产传输恢复。请求取消、总 deadline、节点关闭和下游 RPC 失败仍必须终止在途任务并回收资源。零耗时传输没有独立加载计时器；显式传入的 load_cache_timeout_ms 记入 pending_config，不宣称生效。

原生 EnqueueBatch/EnqueueGroup 当前只支持 DP rank 0，支持文本及经 ViT 准备的输入。Batch 接入已选择 PD 协议，不套用普通 GenerateStreamCall 的分流条件：max_new_tokens=1 也正常接纳，can_use_pd_separation 不改变 Batch 协议。首 Token 达到长度上限时，通过 PREFILL_COMPLETE 结束 D，不重复输出 Token，也不执行 Decode Forward。普通单 Token RPC 仍走本地 Prefill；非 PD 的多 Token 本地生成和 prefill_only 尚未模拟，明确返回 UNIMPLEMENTED。

入队 ACK 前必须取得 D 分配确认；Prefill 完成后推进 LOAD 和 Decode，整个流程不等待 Fetch。Fetch 仅从保留的输出容器单次接入，不持有调度器或 KV 资源。重复 ID 的整批项拒绝，其他非法项各自返回错误；成功项不因其他项失败而回滚。

已接受的批量请求不随入队 RPC 结束而取消；请求 timeout 覆盖全程，Fetch 接入期限默认 10min，接入后撤销。Fetch 重复为 ALREADY_EXISTS，过期/缺失为 NOT_FOUND。Cancel 只面向批量请求；活动项 ACCEPTED，未登记项 TOMBSTONED，已完成项 NOT_FOUND；抢占锁定原生 8429，墓碑和完成记录保留 10min。ACK 不承诺远端已经释放，直接 Generate 通过 RPC 取消。

P/D 都按实际 Forward 计时，token 是重复输入末 token 的模拟载荷，不做词表推理。incremental 返回新 token，否则返回累计输出；aux.output_len 始终累计。P 首 token 计入总长度，非推测模式 D 执行 N−1 轮。Client TTFT 必须来自实际接收，不能使用 Engine aux 或计算耗时替代。

请求总预算取 timeout_ms 和 RPC deadline 较早者，不逐阶段重置；取消、超时和关闭只释放一次，旧回调不能影响复用身份的新请求。602/603/605 使用原生 ErrorDetailsPB。P 转发采用受控拉取/有限 Inbox，遵守发送就绪；共享连接不随单请求取消关闭。Mock 保护上限是 1024 活动请求，TaskHistory 保留 1024 项。旧游标越过历史窗口时，按 real 的恢复方式继续返回完整当前状态、仍保留的完成记录和推进后的游标，不以 OUT_OF_RANGE 阻断健康同步；超前游标不倒退。每次缺口读取在组件日志中记录 `finished_history_gap`，包含 requested_version、dropped_through、latest_version、retained，不能把这段完成记录当成完整请求历史。生产 proto 没有历史完整性字段，缺口证据保存在实验组件日志；需要完整逐请求证据时不能依赖 WorkerStatus 的有限窗口。real 当前为 1000 条/5s 保留，Mock 的 1024 条是明确的模拟边界，不宣称保留策略完全相同。

WorkerStatus 报真实等待/执行任务、finished/version，available_concurrency 保持生产默认 0，KV 以 token 为单位；CacheStatus 提供 block size 与版本。D 当前要求 input+max_new_tokens<max_seq_len，未实现完整真实截断。显式投机按 RTP reserveStep 预留，成功且 reuse 与缓存层开启时写回完整 prompt 块；失败不提交，生成内容不额外建索引。没有完整动态 batch、GPU 或混合 KV 布局模拟。

## Block 容量与 RTP 缓存

共享 `BlockPool` 只管理物理容量和引用。每个实例/存储层各自建池，每个所有者持有独立 Reference，close 幂等，不能释放其他所有者份额或混用池。REQUEST/CACHE/LOAD/STORE/EVICTION 分别计数；只有 CACHE 引用的块算可回收容量，free 和 available 分开，分配失败不部分扣账，释放唤醒等待者。

`KvCache` 和 `BlockTree` 目前固定实现 RTP Full-KV 语义，文件保留在 `java/shared/cache`，不为未知的 vLLM 行为增加策略层或适配接口。每个 Engine 各自拥有 Device/Host 池、前缀索引和请求租约，同 JVM 不共享缓存内容。树内 `lru/lfu/fifo` 只改变合法候选的排序，Device 和 Host 可以分别配置。一次 match 用同一访问序号刷新整条命中路径，release/重叠 publish 不刷新；每次成功进入新存储层都会刷新该层的 FIFO 入层顺序。

正式 `acquire` 在同一锁内完成前缀匹配、取得使用引用、申请缺少的 Device 容量和本地读取；没有先查询再占用的 Scheduler 接口。状态/前缀快照只读，不保证后续请求仍然命中。本地 Host/Device 读取和转存瞬间完成，不建立本地 loading 队列、异步任务或带宽模型；PD 网络交接仍按原有生命周期执行。

对齐当前 real BlockTree：开启 Device cache 时，Host 读取成功后把已有有效前缀注册到 Device，再移除 Host 副本；关闭 Device cache 时，Host 源保留，Device 工作副本不注册。申请失败不能消费 Host。驱逐候选按所在层的直接子节点判断叶子，不要求整棵树的全局叶子；Full-KV 前缀被删除时，所有失效后缀跨层清理。Device 水位触发时向 Host 转存；Host 分配失败时保留 Device 源，不为了转存强制删除 Host 缓存，也不退化成丢弃 Device。未配置 Host 时水位回收可以直接删除 Device 索引。紧急 Device 分配允许直接回收 Device 缓存及失效后缀，不额外删除无关 Host 根。层间移动保持逻辑 key 和索引版本，真正插入/删除才改变 key 版本。Host 不计入 WorkerStatus/CacheStatus 的 Device 执行容量。

容量使用原生 `test_block_num` 和 `reserve_block_ratio`。Device 可用块数=test_block_num−1；Host 使用 `memory_cache_blocks` 个块。reserve 默认 5%，约束首次分配，运行中增长可以使用预留。需求超过单实例上限为永久不足，暂时被其他请求占用为可重试；增长一次补齐差额，失败不留下半次新增分配。容量不足后的等待/结束决定仍在 Scheduler，运行中的 RTP Decode 增长失败会结束该请求，不实现主动换出。显式投机的首次 D 分配也包含 reserveStep，在 PD 交接前准备目标容量。

容量失败处理沿用上述边界：首次 `acquire` 返回 RETRYABLE 时，P 留在 waiting、D 留在容量等待阶段；同一容量版本不重复申请，资源释放后唤醒重试，并保留请求原 deadline。PERMANENT 立即结束单请求，返回容量错误 602；等待中超时返回 603，取消和超时后不再分配。运行中 Decode 的 `ensureCapacity` 无论暂时还是永久不足，都结束该请求，等待其 Compute/PD 使用归零后释放引用，其他请求继续；释放的容量可供等待请求重新准入。不增加抢占、请求换出或重新排队重算接口。

Host 转存目标不足由 KV 内部保留源，不直接上升为请求失败。成功终态的 `publish` 抛异常时，RequestState 将请求改为内部错误，并在 finally 中关闭租约；失败/取消不发布。`acquire/ensureCapacity` 的预期容量失败必须以 Failure 返回，未预期的内部异常按 Engine 故障停止处理，不能伪装成容量不足继续运行。

Scheduler 与 KV 管理器按 Token 交互：`acquire(requiredTokens, keys)` 首次取得资源，`Lease.ensureCapacity(requiredTokens)` 保证本轮容量，`reusedTokens()` 返回复用 Token 数。容量的 Block 换算和向上取整封装在 KV 管理器中；已有容量足够或需求变小时不缩容、不重复申请，超过容量上限返回永久不足。容量配置、物理池和状态快照仍使用原生 Block 口径。

不再每轮调用 markValid，也不在 Lease 中重复保存计算进度。Scheduler 在成功终态且所有 Compute/PD 使用结束后调用 `publish(keys, validTokens)`，一次声明有效范围并注册完整 prompt 块；KV 管理器校验有效范围未超过已分配容量、key 数未超过完整有效块数。`publish` 不关闭租约，`close` 只释放请求引用。投机拒绝区域、预留空间、失败/取消请求不发布，生成内容暂不额外建立前缀索引。多个请求可共享已注册的不可变 Device 块。缓存索引引用与请求物理引用独立：索引可以在请求运行中注销或迁往 Host，请求仍保留原 Device 块，最后一个引用释放才归还物理容量。acquire/publish 仅在同步事务内临时保护完整前缀。状态中的 referenced 只统计仍同时被索引和请求引用的 Device 物理块；注销或转存后仍被请求持有的块计入 held。Host-only 的私有 Device 副本与 Host 记作两层真实占用。

Device/Host 水位按物理占用比例工作：达到 high 后，尽量回收到 low，未达到 low 时保留触发状态。加载、发布、增长和请求释放完成后同步检查，顺序为 Host、Device，不建后台扫描线程；运行中请求的引用可能使物理占用暂时无法下降。`enable_memory_cache` 要求显式正数 `memory_cache_blocks`。Disk/Remote 仍拒绝开启。Host 传输超时保留在 pending_config，本地瞬间完成不使用该超时。Host-only 写回暂保留可容纳的部分前缀，尚未对齐 real 的整批目标预分配与整批失败回滚；Device+Host 主路径不依赖此简化。异步传输队列、Disk/Remote 和 Hybrid/SWA/Linear 布局仍不实现。

## 通用 V 与输入准备

Engine 是运行实例的概念，prefill/decode/vit 是 Engine 的角色；RTP/vLLM 是 Engine 后端，不能与 P/D/V 混为一层。当前配置中的 `component.engine` 字段选择 P/D 后端；V 统一实现于 `java/shared/vit`，不需要选择后端，也不与 `java/backend/rtp` 并列为后端。以后 P/D 使用哪个后端都复用这一份 V。V 的媒体协议与 TCP 客户端仍在 `java/shared/multimodal`，Client/P 只依赖该协议模块。

P/D/V 都实现 `EngineInstance` 生命周期句柄，各自拥有端口、队列、缓存和产物目录；V 拥有自己的调度器。本地同一个 `EngineHost` 平等管理三种 Engine；Whale 每 Pod 直接启动一个 Engine，无需 EngineHost。统一产物是 `java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar`，同时包含通用单实例入口和本地宿主入口，不为 V 单独打包。Master 始终独立 JVM。

原生 `RemoteMultimodalEmbedding` 的一个请求可含多张图，并发请求由 V 调度器合成 batch，无额外 Batch RPC。`gpu_max_batch_size=1` 表示单请求执行并忽略收集窗口；否则按请求数、图数和收集窗口选 FIFO batch。队列和 RPC 等待者均有上限，超出返回 RESOURCE_EXHAUSTED。Client/P 只依赖 shared/multimodal，不依赖 V 服务模块。首版不拆分一个请求的图片组，超过单批图数预算明确拒绝。取消/超时仅释放当前等待者；同 key 的计算可被多个请求共享，无消费者且尚未执行的工作不再 forward，已开始的 batch 可以完成并登记缓存。

`environment.vit` 不指定 engine 身份，配置示例：

```yaml
vit:
  deployment:
    replicas: 1
  config:
    patch_size: 14
    spatial_merge_size: 2
    hidden_size: 4096
    gpu_max_batch_size: 8
    gpu_max_batch_images: 200
    gpu_batch_wait_ms: 10
    mm_max_queue_size: 1024
    mm_cache_gpu_max_bytes: 67108864
    mm_cache_cpu_max_bytes: 268435456
    mm_hash_key_cache_max_bytes: 16777216
    compute.forward.expression: "2 + 0.00001 * totalImagePixels"
```

图像输入携带明确的 width/height（原生 mm_preprocess_config），不下载/解码实际图片。输出行数为 ceil(width/(patch_size*spatial_merge_size))*ceil(height/(patch_size*spatial_merge_size))。这是可配置的 Mock 几何模型，不宣称复现某个真实视觉模型的 resize/crop。首版只接受 IMAGE URL 身份和显式尺寸；视频、音频、tensor 输入、V4.1 prepared patches 在 V 模块边界明确拒绝，不由 Client 或 P 的核心校验器按多模态类别拒绝。数据文件的 Image.length 要事先按模型几何生成，旧数据仅离线转换，不新增运行时兼容格式。

V 以 miss 的图片组成 ForwardBatch，复用公式和 SimulatedCompute。计算完成后登记缓存、通知等待者；Compute 不直接改变缓存。命中和在途合并不重复计入 forward。提供 real 名称的 py_rtp_vit_embedding_forward_rt、batch_size、cache 指标及 mock_vit_theoretical_forward_rt；元数据模拟不能代替真实模型计时校准。

Embedding Device/Host 按逻辑 rows*hidden_size*BF16(2 bytes) 计费，各自 LRU；Device 淘汰转存 Host，Host 淘汰移除 embedding。Hash 独立按行数和 key 字节计费，因此 hash_hit 与 embedding_hit 不等价。在途计算使用独立 claim，失败移除，旧 claim 的迟到完成不能覆盖新 claim。消费者持有不可变结果，不依赖缓存继续驻留。缓存 key 对齐 Linux real MultimodalInput 的 URL/type/preprocess 身份，hash 内容是稳定模拟值，不宣称与实际 tensor 的 hash 相等。

Client 按 Master 的 V-only route 选 V，读取元数据、准备特征 token，再带 selected_vit 请求完整 P/D 路由；Master 切换使 pin 失效时重新取一次 V 路由。P 在输入准备完成后才把规范化 token 序列交给 KV/Scheduler，并用同一输入准备 D ALLOCATE 和 PD 交接。P 的输入准备发生在调度锁外，受原请求 deadline/取消约束；这段准备暂不计入 P Scheduler 的队列快照，但计入 Client 端到端耗时。D 接收已准备输入，不重复编码。

本地单 token 的输入准备由 RpcService 负责，P/D 单请求和 Master batch 由 PrefillGeneration 负责；LocalPrefill 的单个与批量提交只接受已准备输入，拒绝尚带媒体内容的请求，不调用 V。本地入口保留准备前的请求开始时间；各路径沿用原请求 deadline，不在准备完成后重置超时预算。

VP 同 JVM 和跨进程都走 TCP gRPC，只传 shape/dtype、split_size 和逐行 INT32 hash，不生成 embedding payload，也不伪造 RDMA lease。Mock P 校验形状、长度和展开 span 后使用；此响应不能交给真实 tensor 解码器，VP 网络带宽和 GPU 内存复制不在模拟范围内。

Whale 的 P/D/V 统一使用 `java -jar java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar`，无需 Python Runner、`MOCK_HOST_PORT`、`MOCK_COMPONENTS_DIR` 或宿主管理服务。入口 `EngineMain` 从 `MOCK_CONFIG_FILE` 中的 component_role 决定启动哪个 Engine；设置 MOCK_OUTPUT_DIR、MOCK_READY_FILE，以及固定 MOCK_RPC_PORT、MOCK_CONTROL_PORT。实例配置采用与 Runner 相同的 id/component_role/component.config 结构，P/D 另外提供 component.engine。通过 MOCK_BIND_HOST=0.0.0.0 和 MOCK_ADVERTISE_HOST=PodIP 区分监听和发现地址。每个实例独立写入 effective-config、endpoints 和 ready；关闭删除 ready，旧产物目录拒绝覆盖。共享 JVM 的实例沿用同一进程地址，各自端口不同。无论是否共 JVM，VP/PD 都走 TCP、只传元数据。

## Compute、Forward、Decode 和 EOS

每个组件只配置一份 Compute；Forward、Decode 进度和 EOS 都归 Compute，不再配置组级 decode/generation，也不保留旧 compute.expression 入口。

```yaml
environment:
  prefill:
    engine: rtp
    config:
      compute:
        forward:
          expression: "1.23 * (1 + 0.01*totalComputeTokens)"
        eos:
          probability: 0.0025
          seed: 20260922
  decode:
    engine: rtp
    config:
      compute:
        forward:
          expression: "19.5 + 0.175*batchSize"
        decode:
          tokens_per_step: 2.6
        eos:
          probability: 0.0025
          seed: 20260922
```

RtpConfig 只加载 ComputeSpec；ComputeModel 组合 FormulaForward、可选噪声、DecodeStepModel 和 StoppingModel。非默认 Decode 进度参数只允许配置在 D，P/D 都可以配置 EOS。旧字段、未知字段、null、错误类型或缺少必要 seed 明确拒绝，没有别名和兼容分支。模型校验来自 Java，Python 只编译组件配置结构。effective-config 顶层保存生效值，requested_config 保存请求配置，pending_config 单列仍未实现的原生参数。

调度前 Compute.prepare 固定请求快照和本轮验证量/KV 需求，不抽投机接受量。Forward 完成后 Compute.completed 依次计算接受后的候选产出、EOS 和长度截断，返回 committedTokens/finished。Scheduler 只校验请求仍有效并提交结果、更新队列、处理资源和 PD；不判断正常结束是 EOS 还是达到上限。取消、超时、关闭和计算失败仍走独立失败路径，不能成为正常 finished，也不能因旧回调再提交 token。

P 每轮只产首 token，finished 表示整个请求是否结束，不表示 Prefill 阶段是否完成。P 阶段完成后退出计算集合，保留传输需要的 KV；即使首 token EOS，也仍完成 LOAD 确认，再通过 PREFILL_COMPLETE 让 D 结束，不启动 D Forward。PD 不再传预测的 stop_after，不由 P 决定 D 的未来停止长度。

ForwardBatch 保留请求长度、实际缓存命中、已生成长度、stepTokens、优先级和图片 shape。FormulaForward 只计算 Duration，不保留没有实际替换入口的 ForwardModel 接口；SimulatedCompute 只执行可取消等待；公共 Compute 的产出计算不修改请求或队列。一个 Engine 同时至多一个 Forward，完成结果只提交一次。

公式复用生产 ArithmeticFormula 的 param(name, value)、sum 等函数，单位 ms，保留到 ns；非法变量、负值、非有限值和溢出拒绝。逐请求变量只在 sum 中使用：inputTokens/hitCacheTokens/computeTokens/hasHitCache/outputTokens/generatedTokens/contextTokens/stepTokens/priority/imageCount/imagePixels；batch 使用 batchSize、dpRank、total/max 派生变量。RTP 额外绑定 cpSize、cpEnabled（0/1）、tpSize、dpSize，只有显式引用才参与计算，不自动缩放已有拟合公式。默认 FormulaForward 不猜测多个 DP 的组合方式。

倍率直接乘入 forward.expression，不提供独立 scale/noise.scale。噪声位于 forward.noise，默认 stddev_ms/stddev_ratio=0；启用时要求 seed 和正 max_deviation_ms，扰动有界、最终耗时不小于 0。seed、Engine 身份、角色和 batch 身份固定一次耗时样本，取消批次不补完成指标。当前 DeepSeek Case 不启用噪声。

Decode 有两种互斥模型。默认普通模式每轮一个 token；平均模式 tokens_per_step>=1 按 ceil((n+1)*v)−ceil(n*v) 产生候选数量，每轮每请求计一个 target 输入，不虚构 draft/accept 统计。显式投机模式配置 draft_tokens、acceptance_rate、seed：抽取接受前缀 A，候选产出为 A+1；固定独立接受率 α、草稿长 k 时 E[C]=1+α+…+α^k。这是可复现的统计近似，不模拟真实 MTP 的 logits、draft 张量和独立 draft KV；P 的初始化成本包含在 P 整体公式中。

长度边界在接受结果后统一处理。显式投机模式按固定 k 个草稿、k+1 个验证位置及 seq_len+k+1 的 KV 预留执行（普通 Decode reserveStep=0），对齐当前 RTP 的固定验证窗口；不能因为只剩 R 个输出位置就提前缩小验证量。候选产出 5 个而只剩 4 个时，只提交 4 个并返回 finished=true，验证成本仍按完整计划累计。平均进度模型每轮仍计一个 target 输入，KV 仅覆盖最终可能提交的输出前缀。EOS 只检查接受后的候选 token，按绝对输出位置生成固定概率事件；相同 seed、请求身份和概率下，改变 batch、轮宽、接受模型或调用顺序不会重抽已经走过的位置。拒绝的草稿不参与结束判断。

StoppingModel 遵守 min_new_tokens，ignore_eos 只屏蔽 EOS，不屏蔽长度上限。完成判定的上限同时受 max_new_tokens 与 maxSequence−inputTokens 约束；RTP 接入仍保留前述较严格的序列准入规则；显式投机在首次分配和后续增长中另外预留 reserveStep。EOS 位于本轮中间时截断实际产出，但本轮已执行的验证成本和在途资源引用不撤销。固定概率只是 Mock EOS hazard，不宣称真实输出分布。标准回放发送 ignore_eos=true，所以 Case 可保留 P/D 的 EOS 参数而不再次缩短数据中的目标长度。

real 依据：MtpExecutor::decodeStep 先 target verify Forward，再执行 speculative_sampler；MtpBatchStreamProcessor::prepareDecodeSpecUpdateInfo 提取接受后的 token，GenerateStream::specUpdate 经 NormalGenerateStream::updateOutput 检查 EOS/max；CompleteTokenIds::matchEosToken 截断到第一个 EOS。PrefillRpcServer::remoteLoadCacheEnd 保留 LOAD ACK、释放源 KV 和 PREFILL_COMPLETE 顺序。当前 Mock 不支持 stop words/beam search 等未实现条件，显式请求会拒绝。

## P/D 配置治理

组件只接受已登记、角色适用的原生参数；支持小写字段或对应大写环境变量名，两者同时出现、未知参数、已删除参数和错放角色的参数报错。Python 不复制这份字段表，RTP 的字段归属与待实现参数校验位于 RtpSettings，实际值解析位于 RtpConfig。没有静默透传一批未消费配置的入口。

CP/TP/DP 保留在执行配置及 effective-config 中，不转成来源说明。一个 Mock Engine 仍表示整体计算单元；不创建内部 rank，不模拟并行通信。当前执行域仍只支持 dp_size=1、dp_rank=0，其他值明确拒绝；保留 dp_size 并不表示已经支持内部多 DP。并行度作为公式输入，已校准的公式、容量和 block size 不再隐式换算。

下表是当前 DeepSeek HA 配置的处理结果。待实现参数保留实际数值、校验类型与范围，但只进入 pending_config，不表示行为已经对齐。Host 容量、策略和水位已经接通；开启尚不存在的 disk/remote cache 仍拒绝启动；其他待实现项允许保留并警告，相关调度、压力和超时结论不得作为已对齐结果。

| 参数 | P | D | 当前处理 |
| --- | --- | --- | --- |
| cp_size / cp_enabled | 4 / true | 1 / false | 保留，公式可读取；不模拟内部 CP。 |
| tp_size / dp_size | 4 / 1 | 1 / 1 | 保留；tp_size 还用于原生状态上报。 |
| max_seq_len | 1048576 | 1048576 | 生效。 |
| seq_size_per_block / test_block_num | 512 / 21554 | 128 / 221485 | 本地容量生效；PD 按 token 交接。 |
| reserve_block_ratio | 1 | 8 | 生效，百分比。 |
| max_generate_batch_size | 64 | 128 | P/D 统一入口，生效；删除 max_context_batch_size。 |
| max_batch_tokens_size | 512000 | 1048576 | P token/矩形准入生效；D 不作为每轮 verify token 上限。 |
| max_batch_tokens_without_cache | 50000 | 不配置 | P 普通队列按未缓存 token 配额准入；D 不配置。 |
| max_inited_kv_cache_streams | 128 | 不配置 | P 初始化 KV 数量限制和超大组退回条件均生效；D 不配置。 |
| reuse_cache / enable_device_cache / device_eviction_policy | false / true / lru | false / true / lru | 保留现有语义，不为接 Host 而打开 reuse。 |
| enable_memory_cache / memory_cache_blocks | true / 52295 | false / 不配置 | P Host 容量生效，本地瞬间周转。 |
| memory_eviction_policy | lru | 不配置 | P Host 策略/水位生效；D 不配置。 |
| block_tree_device_evict_low/high_watermark_ratio | 0.82 / 0.9 | 0.82 / 0.9 | Device 水位生效；释放资源时同步回收到低水位。 |
| block_tree_memory_evict_low/high_watermark_ratio | 0.9 / 0.94 | 不配置 | P Host 策略/水位生效；D 不配置。 |
| block_tree_business_queue_max_size / block_tree_transfer_queue_max_size | 10000 / 10000 | 10000 / 10000 | 待实现队列背压，不以无界队列冒充等价实现。 |
| memory_cache_sync_timeout_ms | 10000 | 不配置 | P 本地瞬间传输不使用，保留 pending_config。 |
| enable_disk_cache / enable_remote_cache | false / false | false / false | 明确关闭；开启拒绝。 |
| prefill_retry_times | 0 | 不配置 | 当前不重试；非零值列为待实现。 |
| prefill_max_wait_timeout_ms | 600000 | 不配置 | P 待实现独立等待期限。 |
| max_rpc_timeout_ms | 7200000 | 7200000 | 待实现独立 RPC 上限；已有请求总期限不因此失效。 |
| load_cache_timeout_ms | 不配置 | 可记录原配置 | 零耗时交接不使用独立加载超时，D 显式配置记入 pending_config。 |
| compute | 1.23 * (原公式)，不配噪声 | 19.5+0.175*batchSize | 保留，不按 CP/TP 二次缩放。 |
| compute.eos | probability=0.0025，seed=20260922 | 同 P | 参数保留；固定长度回放通过 ignore_eos 屏蔽 EOS。 |
| compute.decode.tokens_per_step | 不配置 | 2.6 | D 平均进度模型，不等于真实投机参数。 |

以下参数从当前 Case 删除，不复刻 native 线程池、轮询和描述符分批：block_tree_transfer_worker_count、memory_cache_max_descriptors_per_transfer_batch、prefill_prepare_resource_pool_size、prefill_stop_stream_wait_timeout_ms、decode_retry_interval_ms。取消、资源释放和有界请求生命周期仍必须实现，不能随这些参数一起删除。

decode_retry_times、decode_retry_timeout_ms 在当前 real C++ 执行代码中没有读取处，不迁入 Mock。当前 P 的 cp_force_single_prefill=false、D 不执行对应 Prefill 准入，均省略；未来 P 启用该限制时只需增加逻辑调度约束。prefill_retry_timeout_ms 当前因重试次数为 0 省略，非零重试的次数与预算一起待实现。

## Client 原生收发核心

`NativeClient` 接收 FlexlbScheduleRequestPB 和单请求总 timeout。每次请求固定一份主备快照，先向主节点 Schedule，传输失败时最多向备节点尝试一次；收到业务响应、deadline 超时、主动取消或进入 Engine 取流后都不重发。要求成功响应及有效 P 地址。已入队走 FetchResponse，其他走 GenerateStreamCall，并写入 Master 返回的 P/D/V 路由。不提供不 Fetch、不消费完整流、Engine 代报 TTFT 或 schedule-only 模式。

每个 Client JVM 独立运行 MasterDiscovery：每秒刷新 topology.json 并 POST 真实 `/rtp_llm/master/info`，单次心跳 500ms、最多 4 个并发探测；连续两次失败判不健康。对齐 real Frontend `server/host_service.py` 的选择规则：优先健康且声明自己为主的节点，否则保留仍健康的原目标，再按 IP 选择健康候选；备用节点从未声明为主的其他健康节点中选择。发现暂时漏掉的健康节点仍继续探测，失败后停止保留，不再发现且已不健康的记录在最后探测 30s 后清理。文件损坏或非法端点属于执行错误，会终止整个 Client 播放；没有健康 Master 是请求失败，不直连 Engine，也不按列表轮流广播。

发送边界对齐 `server/master_client.py`：只有 Schedule 的传输失败尝试一次备节点；业务拒绝、deadline 和取流失败不尝试其他 Master。整个请求仍受同一个总 deadline 约束，重试消耗剩余预算。这个策略不承诺 exactly-once：响应丢失后原主可能已经接收请求，保留与 real Frontend 相同的传输重试风险。Client QPS 仍按逻辑请求 issued 计一次，备节点尝试不会再发放流量预算。

每次只预取一个响应，验证身份、INT32 载荷/形状、递增长度和终态；最终长度必须达到 min_new_tokens，ignore_eos 时必须等于 max_new_tokens，提前 finished 不能冒充固定长度回放成功；finished 后仍须正常结束，空流/截断/终态后额外帧均失败，不保留全部输出。firstToken 从发送的单调时钟到实际收到首 token，包含调度和网络；没有首 token 就没有数值。completion 记录发送墙钟、可空 TTFT、总耗时、收到 token 数及终态/错误码；后续失败保留已观测 TTFT，不读取 aux 耗时代替。

同一 deadline 覆盖调度和取流，并受更短原生 timeout 限制。取消/错误/超时结束当前 RPC，向该请求最后尝试的 Master 尽力发一次有界 Cancel，不随新的选举结果改变取消目标；Cancel 失败写日志，不承诺远端释放。连接复用、请求上下文独立，close 取消活动并等待有界 Cancel 后关闭所拥有连接。

协议测试使用夹具，真实接线测试使用生产 Master，二者都真实消费 P/D 输出。标准 Protobuf 输入的 Player、LoadClient、client0 协调和 start/stop 已接通，主备发送只改变 Schedule 目标选择与传输失败重试边界。Client 逐请求执行记录已接通；运行时只读标准 Protobuf＋Zstd 输入，仓库不提供旧采集格式转换工具；Engine 与 Master 的完整服务端证据仍未接通。

## 指标、采集和查询

RTP 后端的 RtpMetrics 使用 Prometheus simpleclient 0.16.0，每个节点独立注册表，`GET /metrics` 标准文本导出。事件发生时更新累计量，抓取时读取瞬时量，没有常驻采样线程。业务标签只保留 role；Prometheus 为每个采集目标设置 instance=实际 RPC 服务地址，KV 分池使用 pool_name=device/host、tier=DEVICE/HOST，额外维度仅下表枚举，不放 request ID、cache key 或异常文本。

| 指标 | 类型 | 含义 |
| --- | --- | --- |
| rtp_llm_requests_started_total | Counter | 校验并接纳后加一，入口拒绝不计。 |
| rtp_llm_framework_requests_total | Counter，outcome=success/cancelled/timeout/error | 每个已接纳请求唯一终态，节点关闭计 error，不代替 Client 成功率。 |
| rtp_llm_host_kv_tokens | Gauge | 本地 Host total/free token 容量，与 Device 分开。 |
| rtp_llm_cache_lookup_tokens_total / rtp_llm_cache_hit_tokens_total | Counter | 分配时实际输入/历史命中 token，取消不撤销已发生查找。 |
| rtp_llm_forward_tokens_total | Counter，kind=input/compute/output | 完成 Forward 的 token；input/compute 汇总本批全部执行成员，output 只计有效提交。全部取消的 Forward 不补完成样本。 |
| rtp_llm_model_forward_us | Gauge | 最近一个完成 Forward 的实际耗时，微秒；计时到执行完成通知，不包含等待调度域锁和结果提交。 |
| mock_model_forward_planned_us | Gauge | 同一个完成 Forward 的计划耗时，微秒，包含公式和可选噪声。 |
| rtp_llm_model_forward_seconds | Histogram | 完成 Forward 实际耗时秒，含 bucket/sum/count，排除排队，不是 TTFT。 |
| rtp_llm_scheduler_requests | Gauge，state=waiting/running/loading/ready | 调度器报告的四类互斥请求数。 |
| rtp_llm_kv_tokens | Gauge，state=total/available/free/held | token 容量，available 含可驱逐缓存，不等于 free。 |

RTP 原生观测直接导出：`rtp_llm_wait_stream_size`、`rtp_llm_running_stream_size`、`rtp_llm_kv_cache_left_seq`（Tokens）、`rtp_llm_kv_cache_{total,free,available,used}_blocks`（Blocks）和 `rtp_llm_kv_cache_used_ratio`（0–100，不可用比例）。Host 暂只上报 total/free/used Blocks，不用 free 冒充 available。loading/ready 仍是调度器阶段辅助观测，不冒充 real 的加载缓存队列。

Prefill 的 context TPS 用计算 Tokens／执行时间，含缓存 TPS 用输入 Tokens／执行时间；墙钟 TPS 用对应 Tokens／实际窗口。Decode generate TPS 统计提交的输出 Tokens，total TPS 统计模型计算 Tokens，二者在投机采样时不同。`rtp_llm_framework_qps` 统计终态，不是接纳；error 包含 timeout，cancel 单列。上述速率 Gauge 的窗口是两次采集间的单调时钟间隔，无后台采样线程；空闲窗口报告零。real 默认按后台周期报告，窗口边界可能不同，不能要求逐点相等。

没有 real 同名指标的真实观测，以 RTP Prometheus 辅助 Counter/Histogram 保留供 Python 门禁和统计使用，不拿它们的累计值直接冒充线上 QPS/TPS Gauge。理论 Forward、ViT 运行图片数和 hash 缓存预算保留 `mock_` 前缀。此次更名不保留旧 `mock_engine_*` 别名；旧实验保持原始 view.json 和 TSDB，不承诺新查询可以直接读取旧指标名。

P/D 已补 `rtp_llm_{context,generate}_batch_size`、执行 Token 数、最长序列、逐请求等待/首 Token/总延迟、输入/输出长度与迭代数；Decode 传输等待接入 `rtp_llm_stream_loading_cache_latency_us`。Query batch 的 native 语义是 beam 数，不能用调度批大小代替。

ViT 使用相同的物理元数据与 Prometheus instance 发现；QPS、Embedding 队列/等待/Forward/合批、请求图片数/输出长度，以及缓存命中/未命中/在途合并/淘汰和逻辑驻留量均按 real 名称导出。毫秒指标不转换成微秒名，未实现的下载、预处理子进程和 CUDA Graph 不补零。详见 [指标覆盖与边界](../views/README.md)。

新增事件 Gauge 为采集窗口均值，无事件的窗口不发耗时样本；其 `_sum` / `_count` 辅助累计量不清零。QPS 使用真实窗口秒数换算，空闲归零；窗口与 KMonitor 固定周期不保证逐点相同。原有 Forward 最近值与 Histogram 口径保持不变。

累计量预建 0，抓取不清零；分母为 0 不补比率。以下查询在 Python `interfaces/metrics.py` 实现，直接调用本次托管 Prometheus：

| 接口 | 口径 |
| --- | --- |
| engine/cache/hit_rate | P 历史命中 token 增量／输入 token 增量，token 加权；可选 targets。 |
| engine/prefill/tps | P 实际计算 token 增量／含实际计算的 Forward 耗时增量。 |
| engine/prefill/tps_with_cache | P 输入 token 增量／有输入的 Forward 耗时增量，输入含命中。 |
| engine/decode/tps | D 完成 Forward 输出 token 增量／墙钟时间。 |
| client/success_rate | 必填 range 区间内成功终态计数增量／全部终态计数增量，先对各 Client 算增量再聚合；可选 targets。 |
| prometheus/query | 固定 query_time 的一次 scalar 或 float instant vector，保留标签、数值和 sample_time。 |

普通接口在本次独占 Prometheus 内按 role 和 instance 地址过滤，先逐条 Counter 算增量再聚合以处理 reset；hit_rate 不能平均节点比率。显式空 targets 报错，选择器转义；每个选中实例的分子/分母至少需要两次采样，重复或缺失序列报错。range 在固定 end 执行单次 instant query，窗口=end−start，精度 ms，低于 1ms 拒绝；latest 用检查前 60s 内最近两次采样的 irate，不平均整分钟。缺少采样或没有可计算 Forward 时报错。

`client/success_rate` 的 range 是必填的 `[start, end]`（start < end <= 0），按固定查询时刻定位窗口，使用 Counter 的 increase，按完成时间统计，分母包含 success/error/timeout/cancelled。它没有最小请求数门槛；窗口无终态时分母为零，或任一 Client 的终态序列采集不完整时均报错。至少两次抓取是计数器增量的计算条件，不是请求数要求。近期完成请求可能在更早时发出；此口径不表示同一发送阶段的成功率。

`cases/master_ha_failover/case.yaml` 使用已转换的 DeepSeek trace，按 YAML 中的 `replay_speed` 回放贯穿杀 A、恢复 A、杀 B、恢复 B，检查结束后提前停止。操作通过已保存的地址列表引用，恢复前不杀另一侧；恢复后检查原主不被抢占。Case 在保守等待后直接检查最近完成请求的成功率，最后收集 Client 终态，不使用进度接口、最少样本数、精确事件窗口或 HA 专用执行器。启动配方仍由 `mock-case launch <case>/case.yaml` 生成；双 Master 要求本机可绑定两个回环 IP。该 Case 保留原 DeepSeek preset 的 Host 开关、容量与水位，不为演示命中而打开原配置关闭的 reuse_cache。参数来自 legacy_unverified preset，配置和数据迁移不代表端到端验收通过。

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
