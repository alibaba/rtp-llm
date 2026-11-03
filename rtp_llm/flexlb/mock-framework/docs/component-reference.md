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
      mock.compute.forward.expression: "param(baseMs, 1) + 0.01*totalComputeTokens"
```

原生配置保留 real 的名称和单位，例如 `concurrency_limit`、`seq_size_per_block`、`test_block_num`；同一字段同时写大小写两种形式会报错。P/D 的 `concurrency_limit` 默认 32，范围 1–1024，控制每轮请求数。Mock 必须显式提供 `seq_size_per_block`、`max_seq_len`、`test_block_num`：本地没有 GPU/模型配置，不能复用 real 的硬件探测和自动容量推导。一个 Engine 表示完整计算体，不创建 CP/TP/DP rank 或模拟通信，不根据并行度隐式换算容量、block size 或已校准的公式。具体支持项和输入规则见下文“Engine 配置规则”。

RTP 请求公共支持范围集中在 `RequestValidation`，本地单 token、P/D Prefill 及 Decode ALLOCATE 均使用它：adapter/unique_key、多 beam/多返回、stop words、sp_edit/sp_input_lookup、额外 logits/hidden/prob/loss 输出明确拒绝。路径只保留自身角色、长度、路由和协议阶段规则。测试通过实际 gRPC 同时验证 max_new_tokens=1/2 和 D ALLOCATE，防止参数只在一条路径被拒绝。

### 执行和 P/D 交接

- P 的 `GenerateStreamCall(max_new_tokens=1)` 走 LocalPrefill，不启用 D。请求经过 FIFO→选中时分配 KV→共享 Compute→真实等待→编码输出→释放；排队不占 KV。
- 多 token P/D 要求 `can_use_pd_separation=true` 且恰好一个有效 DECODE 地址。PrefillGeneration 先向 D ALLOCATE，确认后启动 P，随后 LOAD、发 P 首 token、等待 LOAD 确认、发 GENERATE 并转发 D 输出。
- D 的 DecodeAdmission 处理原生协议，RtpScheduler 管理资源和队列，只有 GENERATE 后进入计算队列；每个 Engine 一轮一个 Forward，按配置合并多个请求。初始 KV 分配遇临时不足等待容量变化；生成期间每轮先增长 KV，失败则终止，成功后才执行与提交输出。

PD 只执行调度器之间的交接，不模拟 KV 数据传输。P/D 只使用原生 RemoteGenerate TCP 连接，不注册自定义回调服务，也不跨节点比对 prompt、块布局或 cache keys。P 必须等自身 Scheduler 报告首轮 Forward 完成后才发送 LOAD；D 对已分配的请求执行零耗时异步传输事件，完成后回复 LOAD ACK。P 收到 ACK 才释放源 KV，然后发送 GENERATE 或 PREFILL_COMPLETE，D Scheduler 决定开始计算或结束。两端块大小各自独立，KV 的申请、持有、释放及 finished 记录仍归各自 Scheduler 管理。

接入层信任正常 P 生产者发送的 LOAD 和首 Token 完成事实，不模拟损坏的 KV 数据、假冒来源、迟到的跨请求交接或生产传输恢复。请求取消、总 deadline、节点关闭和下游 RPC 失败仍必须终止在途任务并回收资源。零耗时传输没有独立加载计时器；real 的 `load_cache_timeout_ms` 在此模型中忽略。

原生 EnqueueBatch/EnqueueGroup 当前只支持 DP rank 0，支持文本及经 ViT 准备的输入。Batch 接入已选择 PD 协议，不套用普通 GenerateStreamCall 的分流条件：max_new_tokens=1 也正常接纳，can_use_pd_separation 不改变 Batch 协议。首 Token 达到长度上限时，通过 PREFILL_COMPLETE 结束 D，不重复输出 Token，也不执行 Decode Forward。普通单 Token RPC 仍走本地 Prefill；非 PD 的多 Token 本地生成和 prefill_only 尚未模拟，明确返回 UNIMPLEMENTED。

入队 ACK 前必须取得 D 分配确认；Prefill 完成后推进 LOAD 和 Decode，整个流程不等待 Fetch。Fetch 仅从保留的输出容器单次接入，不持有调度器或 KV 资源。重复 ID 的整批项拒绝，其他非法项各自返回错误；成功项不因其他项失败而回滚。

已接受的批量请求不随入队 RPC 结束而取消；请求 timeout 覆盖全程，Fetch 接入期限默认 10min，接入后撤销。Fetch 重复为 ALREADY_EXISTS，过期/缺失为 NOT_FOUND。Cancel 只面向批量请求；活动项 ACCEPTED，未登记项 TOMBSTONED，已完成项 NOT_FOUND；抢占锁定原生 8429，墓碑和完成记录保留 10min。ACK 不承诺远端已经释放，直接 Generate 通过 RPC 取消。

P/D 都按实际 Forward 计时，token 是重复输入末 token 的模拟载荷，不做词表推理。Engine RPC 始终返回本轮新增 token，`aux.output_len` 始终累计；用户侧 incremental/累计展示由 Frontend 处理。P 首 token 计入总长度，非推测模式 D 执行 N−1 轮。Client TTFT 必须来自实际接收，不能使用 Engine aux 或计算耗时替代。

请求总预算取 timeout_ms 和 RPC deadline 较早者，不逐阶段重置；取消、超时和关闭只释放一次，旧回调不能影响复用身份的新请求。602/603/605 使用原生 ErrorDetailsPB。P 转发采用受控拉取/有限 Inbox，遵守发送就绪；共享连接不随单请求取消关闭。Mock 保护上限是 1024 活动请求，TaskHistory 保留 1024 项。旧游标越过历史窗口时，按 real 的恢复方式继续返回完整当前状态、仍保留的完成记录和推进后的游标，不以 OUT_OF_RANGE 阻断健康同步；超前游标不倒退。每次缺口读取在组件日志中记录 `finished_history_gap`，包含 requested_version、dropped_through、latest_version、retained，不能把这段完成记录当成完整请求历史。生产 proto 没有历史完整性字段，缺口证据保存在实验组件日志；需要完整逐请求证据时不能依赖 WorkerStatus 的有限窗口。real 当前为 1000 条/5s 保留，Mock 的 1024 条是明确的模拟边界，不宣称保留策略完全相同。

WorkerStatus 报真实等待/执行任务、finished/version，available_concurrency 保持生产默认 0，KV 以 token 为单位；CacheStatus 提供 block size 与版本。D 当前要求 input+max_new_tokens<max_seq_len，未实现完整真实截断。显式投机按 RTP reserveStep 预留，成功且 reuse 与缓存层开启时提交完整有效块；P 提交 prompt，D 提交 prompt 与已计算的生成前缀，失败不提交。没有完整动态 batch、GPU 或混合 KV 布局模拟。

## Block 容量与 RTP 缓存

共享 `BlockPool` 只管理物理容量和引用。每个实例/存储层各自建池，每个所有者持有独立 Reference，close 幂等，不能释放其他所有者份额或混用池。REQUEST/CACHE/LOAD/STORE/EVICTION 分别计数；只有 CACHE 引用的块算可回收容量，free 和 available 分开，分配失败不部分扣账，释放唤醒等待者。

`KvCache` 和 `BlockTree` 目前固定实现 RTP Full-KV 语义，文件保留在 `java/shared/cache`，不为未知的 vLLM 行为增加策略层或适配接口。每个 Engine 各自拥有 Device/Host 池、前缀索引和请求租约，同 JVM 不共享缓存内容。树内 `lru/lfu/fifo` 只改变合法候选的排序，Device 和 Host 可以分别配置。一次 match 用同一访问序号刷新整条命中路径，release/重叠 publish 不刷新；每次成功进入新存储层都会刷新该层的 FIFO 入层顺序。

Scheduler 统一调用 `prepare` 取得 `CacheLoad`；其本地阶段使用 `acquire`，在同一锁内完成前缀匹配、取得使用引用、申请缺少的 Device 容量和本地读取。状态/前缀快照只读，不保证后续请求仍然命中。本地 Host/Device 读取和转存瞬间完成，不建立本地 loading 队列、异步任务或带宽模型；PD 网络交接仍按原有生命周期执行。

对齐当前 real BlockTree：开启 Device cache 时，Host 读取成功后把已有有效前缀注册到 Device，再移除 Host 副本；关闭 Device cache 时，Host 源保留，Device 工作副本不注册。申请失败不能消费 Host。驱逐候选按所在层的直接子节点判断叶子，不要求整棵树的全局叶子；Full-KV 前缀被删除时，所有失效后缀跨层清理。Device 水位触发时向 Host 转存；Host 分配失败时保留 Device 源，不为了转存强制删除 Host 缓存，也不退化成丢弃 Device。未配置 Host 时水位回收可以直接删除 Device 索引。紧急 Device 分配允许直接回收 Device 缓存及失效后缀，不额外删除无关 Host 根。层间移动保持逻辑 key 和索引版本，真正插入/删除才改变 key 版本。Host 不计入 WorkerStatus/CacheStatus 的 Device 执行容量。

容量使用原生 `test_block_num` 和 `reserve_block_ratio`。Device 可用块数=test_block_num−1；Host 使用 `mock.cache.memory.blocks` 个块。reserve 默认 5%，约束首次分配，运行中增长可以使用预留。需求超过单实例上限为永久不足，暂时被其他请求占用为可重试；增长一次补齐差额，失败不留下半次新增分配。容量不足后的等待/结束决定仍在 Scheduler，运行中的 RTP Decode 增长失败会结束该请求，不实现主动换出。显式投机的首次 D 分配也包含 reserveStep，在 PD 交接前准备目标容量。

容量失败处理沿用上述边界：首次 `acquire` 返回 RETRYABLE 时，P 留在 waiting、D 留在容量等待阶段；同一分配版本不重复申请，资源释放或新前缀变为可复用后唤醒重试，并保留请求原 deadline。PERMANENT 立即结束单请求，返回容量错误 602；等待中超时返回 603，取消和超时后不再分配。运行中 Decode 的 `ensureCapacity` 无论暂时还是永久不足，都结束该请求，等待其 Compute/PD 使用归零后释放引用，其他请求继续；释放的容量可供等待请求重新准入。不增加抢占、请求换出或重新排队重算接口。

Host 转存目标不足由 KV 内部保留源，不直接上升为请求失败。成功终态的 `publish` 抛异常时，RequestState 将请求改为内部错误，并在 finally 中关闭租约；失败/取消不发布。`acquire/ensureCapacity` 的预期容量失败必须以 Failure 返回，未预期的内部异常按 Engine 故障停止处理，不能伪装成容量不足继续运行。

Scheduler 与 KV 管理器按 Token 交互：`prepare(requiredTokens, keys, deadline, remoteLookup)` 首次准备资源，`Lease.ensureCapacity(requiredTokens)` 保证本轮容量，`CacheLoad.Result.reusedTokens` 返回已就绪的总复用 Token 数，`Lease.reusedTokens()` 仍描述初次本地匹配。容量的 Block 换算和向上取整封装在 KV 管理器中；已有容量足够或需求变小时不缩容、不重复申请，超过容量上限返回永久不足。容量配置、物理池和状态快照仍使用原生 Block 口径。

不再每轮调用 markValid，也不在 Lease 中重复保存计算进度。Scheduler 在成功终态且所有 Compute/PD 使用结束后调用 `publish(keys, validTokens)`，一次声明有效范围并注册完整有效块；P 仅提交 prompt，D 同时提交已计算的输出前缀，排除最后一个未计算 KV 的采样 token；KV 管理器校验有效范围未超过已分配容量、key 数未超过完整有效块数。`publish` 不关闭租约，`close` 只释放请求引用。投机拒绝区域、预留空间、失败/取消请求不发布，生成内容与输入使用同一 CacheSequence 算法建立前缀索引；当前输出模型重复原生 GENERATE 给出的首 token。多个请求可共享已注册的不可变 Device 块。缓存索引引用与请求物理引用独立：索引可以在请求运行中注销或迁往 Host，请求仍保留原 Device 块，最后一个引用释放才归还物理容量。acquire/publish 仅在同步事务内临时保护完整前缀。状态中的 referenced 只统计仍同时被索引和请求引用的 Device 物理块；注销或转存后仍被请求持有的块计入 held。Host-only 的私有 Device 副本与 Host 记作两层真实占用。

Device/Host 水位按物理占用比例工作：达到 high 后，尽量回收到 low，未达到 low 时保留触发状态。加载、发布、增长和请求释放完成后同步检查，顺序为 Host、Device，不建后台扫描线程；运行中请求的引用可能使物理占用暂时无法下降。`enable_memory_cache` 要求显式正数 `mock.cache.memory.blocks`。Disk/Remote 及 Host 传输超时参数直接忽略，本地瞬间完成不使用这些参数。Host-only 写回暂保留可容纳的部分前缀，尚未对齐 real 的整批目标预分配与整批失败回滚；Device+Host 主路径不依赖此简化。远端准备流程与逻辑读写接口已实现，具体远端存储适配和真实 payload 未实现；Disk 仍不实现。明确不建模按网络层分池或 Hybrid/SWA/Linear 多资源 KV 布局，所有层统一折算为一套逻辑 KV 容量。

## 通用 V 与输入准备

Engine 是运行实例的概念，prefill/decode/vit 是 Engine 的角色；RTP/vLLM 是 Engine 后端，不能与 P/D/V 混为一层。当前配置中的 `component.engine` 字段选择 P/D 后端；V 统一实现于 `java/shared/vit`，不需要选择后端，也不与 `java/backend/rtp` 并列为后端。以后 P/D 使用哪个后端都复用这一份 V。V 的媒体协议与 TCP 客户端仍在 `java/shared/multimodal`，Client/P 只依赖该协议模块。

P/D/V 都实现 `EngineInstance` 生命周期句柄，各自拥有端口、队列、缓存和产物目录；V 拥有自己的调度器。本地同一个 `EngineHost` 平等管理三种 Engine；Whale 每 Pod 直接启动一个 Engine，无需 EngineHost。统一产物是 `java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar`，同时包含通用单实例入口和本地宿主入口，不为 V 单独打包。Master 始终独立 JVM。

原生 `RemoteMultimodalEmbedding` 的一个请求可含多张图。缓存未命中的每份唯一媒体进入同一条异步准备链：准入 → 执行器排队 → 获取尺寸 → GPU 就绪队列 → Forward → 结果发布。RPC 等待线程、准备执行器和 Forward 执行线程分开；HTTP 慢读不会占用 Forward 线程。缓存命中直接返回，在途重复请求共用同一生产者和额度。`vit_concurrency` 限制准备执行器工作线程数，线程持续持有到该媒体计算完成；`vit_max_queue_size` 是额外等待额度，已接纳的唯一媒体总数不超过两者之和，GPU 就绪队列也按这个总预算派生。满时返回 RESOURCE_EXHAUSTED。任务额度由 VitPreparation 持有，直到共享结果发布完成后才释放；Scheduler 的完成通知只表示计算阶段结束。缓存条目的成功/失败发布采用身份校验，旧任务和重复完成不会覆盖新条目。

`gpu_max_batch_size=1` 串行执行一份就绪媒体工作项且不等待合批；大于 1 时按工作项数和收集窗口合批。同一 RPC 的多图分别准备，可进入不同 Forward，全部结果就绪后才完成 RPC。单张图的 patch/row 不拆分。最后一个等待者取消时，尚未执行的任务移出队列，进行中的下载中断；已经选入 Forward 的任务保留额度到实际计算/发布结束。状态查询不经过这些阻塞执行器。

`environment.vit` 不指定 engine 身份，配置示例：

```yaml
vit:
  deployment:
    replicas: 1
  config:
    checkpoint_path: ./model
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

模型元数据沿用 real 的入口：Case 的 `checkpoint_path` 指向模型目录（相对路径以 case.yaml 所在目录为基准），Whale 使用 `CHECKPOINT_PATH`。仅读取该目录的 config.json，不加载权重；V 必须提供，纯文本 P/D 和 Client 可省略。多模态运行时 Client、P/D、V 指向同一份模型配置。文件保持 real 结构，例如：

```json
{
  "image_token_id": 1000,
  "vision_config": {
    "patch_size": 14,
    "downsample_ratio": 3,
    "min_pixels": 295936,
    "max_image_tokens": 1024
  },
  "text_config": {"hidden_size": 4096}
}
```

image_token_id 必须使用所选模型的实际值，示例值不代表 DSV4.1 的固定标记。text_config.hidden_size 控制视觉结果的逻辑字节预算。无关模型字段忽略；所需几何字段缺失或非法直接报错。

`mock.vit.skip.download` 默认 false，Whale ENV 对应 `MOCK_VIT_SKIP_DOWNLOAD`。默认模式 GET 真实 HTTP(S) URL，仅读取图片头尺寸，不生成 RGB tensor；请求中的预处理 width/height 不冒充原图尺寸。支持 JDK ImageIO 图片格式（含 PNG/JPEG/GIF/BMP）及 WebP 头尺寸，其他格式明确拒绝。使用 real 的 `download_headers`（JSON 字符串）、`mm_image_max_file_size_kb` 和 `mm_timeout_ms`；沿用 DSV4.1 下载的 10 MiB 硬上限、60 秒总下载期限，并受任务剩余期限约束，瞬态失败最多尝试三次。队列等待同样计入任务期限。

本地假 URL 设置 `mock.vit.skip.download=true`，从回放宽高获取尺寸；仍走同一执行器、准入、队列和完成流程，不模拟下载耗时，也不伪造下载指标。真实 URL 回放可省略宽高。几何继续使用 V4.1 `plan_image_grid`/`num_image_tokens`，参数来自模型 config.json 的 vision_config；视频、音频和 tensor 输入未实现。

回放数据保留原始 token、URL/尺寸与单个占位位置 Image.offset；已删除 Image.length，旧图片数据需要重新生成。Client 与 P/D 从同一模型目录读取 image_token_id，LoadClient 将回放中的图片位置编码为该值。直接构造请求的测试工具在不提供模型目录时仍使用 -200 约定；部署真实模型输入时必须提供对应目录。标记出现次数必须与媒体数一致。P 的内部 Prepared 保存规范化语义 token 和 cache-only spans，缓存 hash 不再替换语义 token。Native GenerateInput 不携带布局；仅 P→D 的 GenerateRequest.multimodal_cache_spans 携带缓存身份。P/D 按原生逐块滚动算法将语义 token 与重叠 span 身份共同混入 key；同一媒体身份必须在两端得到相同 key。当前 cache-key seed 固定为 0，不宣称不同 checkpoint 的命名空间已自动推导。

V 以 miss 的图片组成 ForwardBatch，复用公式和 SimulatedCompute。计算完成后登记缓存、通知等待者；Compute 不直接改变缓存。命中和在途合并不重复计入 forward。提供 real 名称的 py_rtp_vit_embedding_forward_rt、batch_size、cache 指标及 mock_vit_theoretical_forward_rt；元数据模拟不能代替真实模型计时校准。

Embedding Device/Host 按逻辑 rows*hidden_size*BF16(2 bytes) 计费，各自 LRU；Device 淘汰转存 Host，Host 淘汰移除 embedding。Hash 独立按行数和 key 字节计费，因此 hash_hit 与 embedding_hit 不等价。在途计算使用独立 claim，失败移除，旧 claim 的迟到完成不能覆盖新 claim。消费者持有不可变结果，不依赖缓存继续驻留。缓存 key 对齐 Linux real MultimodalInput 的 URL/type/preprocess 身份，hash 内容是稳定模拟值，不宣称与实际 tensor 的 hash 相等。

Client 按 Master 的 V-only route 选 V，读取元数据、计算路由长度和缓存 key，再带 selected_vit 与原始输入请求完整 P/D 路由；Master 切换使 pin 失效时重新取一次 V 路由。P 在输入准备完成后才把规范化 token 序列交给 KV/Scheduler，并用同一输入准备 D ALLOCATE 和 PD 交接。P 的输入准备发生在调度锁外，受原请求 deadline/取消约束；这段准备暂不计入 P Scheduler 的队列快照，但计入 Client 端到端耗时。D 接收已准备输入，不重复编码。

本地单 token 的输入准备由 RpcService 负责，P/D 单请求和 Master batch 由 PrefillGeneration 负责；LocalPrefill 的单个与批量提交只接受已准备输入，拒绝尚带媒体内容的请求，不调用 V。本地入口保留准备前的请求开始时间；各路径沿用原请求 deadline，不在准备完成后重置超时预算。

VP 同 JVM 和跨进程都走 TCP gRPC，只传 shape/dtype、split_size 和逐行 INT32 hash，不生成 embedding payload，也不伪造 RDMA lease。Mock P 校验形状、长度和展开 span 后使用；此响应不能交给真实 tensor 解码器，VP 网络带宽和 GPU 内存复制不在模拟范围内。

P/D/V 的单实例入口统一为 `java -jar java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar`。本地 Host 使用每实例配置文件，多个实例共享 JVM 时仍保留各自配置；不从进程 ENV 覆盖实例业务参数。

独立 JVM 可选择两种输入方式：设置 `MOCK_CONFIG_FILE` 时，只读取文件中的 id/component_role/component.config；未设置时，只读取业务 ENV，身份由 `ROLE_TYPE=PREFILL|DECODE|VIT` 提供；可用 `MOCK_NODE_ID` 显式指定，否则使用 role 与发布地址生成诊断身份。文件和 ENV 不叠加，没有第二套默认值。P/D 文件还需提供 `component.engine: rtp`。

独立入口使用 real 的 `START_PORT`（默认 8088），HTTP 为 START_PORT，gRPC 为 START_PORT+1。`MOCK_BIND_HOST`、`MOCK_ADVERTISE_HOST` 只控制进程监听和发布地址；监听通配地址且未显式指定发布地址时，通过主机名解析本机 IP。镜像默认监听 0.0.0.0，本地 Host 默认监听 127.0.0.1。

EngineMain 不要求输出目录或 ready 文件，也不写 effective-config/endpoints。启动时输出一条包含生效配置、实际端点的 JSON 日志；每秒检查现有 Engine 健康状态，内部不可恢复故障使进程非零退出。HTTP `/health` 与原生 CheckHealth 使用同一条不采集指标的检查链，健康为 200/status=ok，故障为 503。TERM 触发关闭与资源释放。实验文件只由本地 EngineHost 发布，并继续保留目录防覆盖约束。

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

调度前 Compute.prepare 固定请求快照和本轮验证量/KV 需求，不抽投机接受量。Forward 完成后 Compute.completed 依次计算接受后的候选产出、EOS 和长度截断，返回 committedTokens/finished。Scheduler 只校验请求仍有效并提交结果、更新队列、处理资源和 PD；不判断正常结束是 EOS 还是达到上限。取消、超时、关闭和计算失败仍走独立失败路径，不能成为正常 finished，也不能因旧回调再提交 token。

P 每轮只产首 token，finished 表示整个请求是否结束，不表示 Prefill 阶段是否完成。P 阶段完成后退出计算集合，保留传输需要的 KV；即使首 token EOS，也仍完成 LOAD 确认，再通过 PREFILL_COMPLETE 让 D 结束，不启动 D Forward。PD 不再传预测的 stop_after，不由 P 决定 D 的未来停止长度。

ForwardBatch 保留请求长度、实际缓存命中、已生成长度、stepTokens、优先级和图片 shape。FormulaForward 只计算 Duration，不保留没有实际替换入口的 ForwardModel 接口；SimulatedCompute 只执行可取消等待；公共 Compute 的产出计算不修改请求或队列。一个 Engine 同时至多一个 Forward，完成结果只提交一次。

公式复用生产 ArithmeticFormula 的 param(name, value)、sum 等函数，单位 ms，保留到 ns；非法变量、负值、非有限值和溢出拒绝。逐请求变量只在 sum 中使用：inputTokens/hitCacheTokens/computeTokens/hasHitCache/outputTokens/generatedTokens/contextTokens/stepTokens/priority/imageCount/imagePixels；batch 使用 batchSize、dpRank、total/max 派生变量。RTP 额外绑定 cpSize、cpEnabled（0/1）、tpSize、dpSize，只有显式引用才参与计算，不自动缩放已有拟合公式。默认 FormulaForward 不猜测多个 DP 的组合方式。

倍率直接乘入 forward.expression，不提供独立 scale/noise.scale。噪声位于 forward.noise，默认 stddev.ms/stddev.ratio=0；启用时要求 seed 和正 max.deviation.ms，扰动有界、最终耗时不小于 0。seed、Engine 身份、角色和 batch 身份固定一次耗时样本，取消批次不补完成指标。当前 DeepSeek Case 不启用噪声。

Decode 有两种互斥模型。默认普通模式每轮一个 token；平均模式 tokens.per.step>=1 按 ceil((n+1)*v)−ceil(n*v) 产生候选数量，每轮每请求计一个 target 输入，不虚构 draft/accept 统计。显式投机模式配置 draft.tokens、acceptance.rate、seed：抽取接受前缀 A，候选产出为 A+1；固定独立接受率 α、草稿长 k 时 E[C]=1+α+…+α^k。这是可复现的统计近似，不模拟真实 MTP 的 logits、draft 张量和独立 draft KV；P 的初始化成本包含在 P 整体公式中。

长度边界在接受结果后统一处理。显式投机模式按固定 k 个草稿、k+1 个验证位置及 seq_len+k+1 的 KV 预留执行（普通 Decode reserveStep=0），对齐当前 RTP 的固定验证窗口；不能因为只剩 R 个输出位置就提前缩小验证量。候选产出 5 个而只剩 4 个时，只提交 4 个并返回 finished=true，验证成本仍按完整计划累计。平均进度模型每轮仍计一个 target 输入，KV 仅覆盖最终可能提交的输出前缀。EOS 只检查接受后的候选 token，按绝对输出位置生成固定概率事件；相同 seed、请求身份和概率下，改变 batch、轮宽、接受模型或调用顺序不会重抽已经走过的位置。拒绝的草稿不参与结束判断。

StoppingModel 遵守 min_new_tokens，ignore_eos 只屏蔽 EOS，不屏蔽长度上限。完成判定的上限同时受 max_new_tokens 与 maxSequence−inputTokens 约束；RTP 接入仍保留前述较严格的序列准入规则；显式投机在首次分配和后续增长中另外预留 reserveStep。EOS 位于本轮中间时截断实际产出，但本轮已执行的验证成本和在途资源引用不撤销。固定概率只是 Mock EOS hazard，不宣称真实输出分布。标准回放发送 ignore_eos=true，所以 Case 可保留 P/D 的 EOS 参数而不再次缩短数据中的目标长度。

real 依据：MtpExecutor::decodeStep 先 target verify Forward，再执行 speculative_sampler；MtpBatchStreamProcessor::prepareDecodeSpecUpdateInfo 提取接受后的 token，GenerateStream::specUpdate 经 NormalGenerateStream::updateOutput 检查 EOS/max；CompleteTokenIds::matchEosToken 截断到第一个 EOS。PrefillRpcServer::remoteLoadCacheEnd 保留 LOAD ACK、释放源 KV 和 PREFILL_COMPLETE 顺序。当前 Mock 不支持 stop words/beam search 等未实现条件，显式请求会拒绝。

## Engine 配置规则

Case 的 `component.config` 可内联 YAML，或引用相对 Case 目录的配置文件。Python 只展开点号和组织实例，不维护 Engine 参数默认值。Java 的 `EngineConfig` 负责输入转换，`RtpConfig`、`VitConfig` 负责默认值、范围和语义；没有配置源基类或覆盖层。

已实现且语义对齐的 real 参数沿用 real 命名。未实现的 real 参数直接忽略，值不参与校验，也不进入 effective-config。例如磁盘/远端缓存、独立加载/重试超时、异步传输队列、模型路径、设备与尚未模拟的并行通信参数；出现这些参数不代表 Mock 实现了对应能力。原生参数拼写错误也会被忽略，因此复用 real 配置后应核对 effective-config 中真正生效的字段。

Mock 专用参数使用小写点号名称，例如 `mock.compute.forward.expression`；也可写成嵌套 YAML。未知 Mock 参数、大小写错误、空对象、null、路径冲突和重复输入均报错。Mock 不接受旧的 compute/memory_cache_blocks 等别名作为覆盖入口。

### 已实现的 real 参数

| 参数 | 默认值/约束 | 行为 |
| --- | --- | --- |
| seq_size_per_block / max_seq_len / test_block_num | 必填；正数，test_block_num ≥ 2 | 逻辑分配粒度、序列上限、Device 物理块数（含 null block）。 |
| concurrency_limit | 32，1–1024 | P/D 每轮请求数上限；real 的 ConcurrencyConfig 输入经 engine_config.py 写入内部 max_generate_batch_size。 |
| reserve_block_ratio | 5，0–100 | 首次分配预留百分比。 |
| reuse_cache / enable_device_cache | false / true | 前缀复用和 Device 索引。 |
| enable_memory_cache | false | Host 缓存开关；开启时须同时给出正数 mock.cache.memory.blocks。 |
| tp_size / dp_size | 1 / 1，正整数 | real 并行配置；tp_size 用于原生状态/接入元数据，二者分别绑定公式 tpSize/dpSize，不模拟 rank 间执行。 |
| cp_rotate_method | DISABLED | 沿用 real PrefillCPConfig；ALL_GATHER、ALL_GATHER_WITH_OVERLAP、ALLTOALL 开启通用 CP，cpSize=tp_size、cpEnabled=1，否则为 1/0。PREFILL_CP 是独立策略，当前未模拟，prefill_cp_size 等专用参数忽略。 |
| max_batch_tokens_size | 0 | P 的 batch token/矩形准入；0 仍可执行 singleton。 |
| max_batch_tokens_without_cache / max_inited_kv_cache_streams | 0 / 0 | 仅 P 生效；0 不启用对应限制。 |
| device_eviction_policy / memory_eviction_policy | lru | real 内部配置属性，支持 lru/lfu/fifo；real server_args 没有对应公开 ENV，仅文件输入。 |
| disable_decode_first_malloc_device_reuse | false | real 内部 CacheConfig 属性，文件输入；不杜撰公开 ENV 名。 |
| block_tree_device_evict_low/high_watermark_ratio | 0.82 / 0.90 | Device 水位。 |
| block_tree_memory_evict_low/high_watermark_ratio | 0.90 / 0.94 | Host 水位。 |
| gpu_max_batch_size / gpu_batch_wait_ms | 1 / 10 ms | 仅 V；batch=1 为逐请求执行且等待为 0；>1 同时限制 Forward 工作块数与媒体数，较大图片组分轮执行。 |
| vit_concurrency / vit_max_queue_size | 64 / 64 | 异步工作线程上限 / 额外等待额度；总准入和 GPU 就绪队列容量为两者之和。 |
| mm_cache_gpu_max_bytes / mm_cache_cpu_max_bytes | 各 2 GiB | V embedding 缓存逻辑预算，单位 byte。 |
| mm_hash_key_cache_max_bytes | 256 MiB | V hash 缓存逻辑预算，单位 byte。 |

real 默认值和映射以 `rtp_llm/server/server_args`、`rtp_llm/config/engine_config.py`、`rtp_llm/config/py_config_modules.py` 和 `rtp_llm/cpp/config/ConfigModules.h` 为依据。Mock 不用真实 GPU 分配这些预算，也不按实例伪造物理资源消耗。

### Mock 参数

| 参数 | 含义 |
| --- | --- |
| mock.scheduler.mode | P 调度方式，默认 `fifo` 按单请求入队、自主组 Forward batch，拒绝 `EnqueueBatch`/`EnqueueGroup`；`group` 显式启用组语义。ENV 为 `MOCK_SCHEDULER_MODE`；D 始终使用普通调度。 |
| mock.cache.memory.blocks | P/D Host 逻辑块预算，默认 0；不是 real 的 memory_cache_size_mb。本模型不加载层/head/dtype 来计算每块字节数，所以忽略 memory_cache_size_mb，不能把 MB 当块数。 |
| mock.vit.skip.download | 默认 false；true 时只跳过实际图片读取，使用输入宽高，其余异步调度不变。 |
| mock.compute.forward.expression | 必填，Forward 公式，结果单位 ms。 |
| mock.compute.forward.noise.stddev.ms / stddev.ratio / max.deviation.ms / seed | 有界噪声，详见 Compute；数值后缀均相对 mock.compute.forward.noise。 |
| mock.compute.decode.tokens.per.step | D 平均每轮进度，与真实投机机制不同。 |
| mock.compute.decode.draft.tokens / acceptance.rate / seed | D 简化投机模型；相对 mock.compute.decode。 |
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
java -jar java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar
```

## DSV4.1 Engine / KVCM 契约

ViT 的 GetWorkerStatus 总是显式提供 `mm_inflight_inputs`（包括真实零值），计数来自去重后的媒体所有权；缓存命中不增加，重复等待者不重复增加，排队/执行/缓存结果发布完成前不提前减少。RPC 等待者数不是该指标。状态查询不经过阻塞的结果等待执行器。当前没有 GPU 预取、graph 和内容检查模型；实际下载仅用于读取尺寸，不模拟 tensor 预处理。

P 在 GENERATE 阶段发送 `prefill_kv_handoff.computed_kv_end`，为已计算 prompt 的排他边界，不包含首个采样 token。D 校验请求身份、已 LOAD 阶段和边界；缺省字段保留原生旧调用兼容，显式错误断言拒绝。未实现的 logprobs 请求明确返回 UNIMPLEMENTED，不伪造概率输出。

共享缓存元信息由 `shared/cache/CacheMetadata` 描述，来源始终是 KvCache/BlockTree 已提交的有效块，不包含仅分配但尚未计算的工作块。快照记录逻辑 key、HBM/Host 驻留和版本；插入、介质迁移、淘汰与清空推进同一版本，并在缓存锁内投递短变更通知。Master 的原生 GetCacheStatus 仍返回逻辑可复用集合与容量，版本未改变时省略 key；Host 迁移不会误删除仍可复用的逻辑 key。

`shared/cache/KvcmReporter` 消费同一来源，按 real Meta HTTP `/api/registerInstance`、`/api/reportEvent` 编码注册、NODE_REGISTER、HEARTBEAT、BLOCK_SNAPSHOT、BLOCK_ADD、BLOCK_DELETE 和 HOST_DOWN。当前 Full-KV 只发布实际 HBM 驻留；迁移到 Host 发送 HBM 删除，恢复到 HBM 发送新增，Host-only 缓存不会冒充 HBM。注册模型、布局和 spec 由 RTP 装配方明确传入；Reporter 和共享 KvcmDiscovery 不依赖 RtpConfig 或 Scheduler；Reporter 统一拥有发现和传输资源的关闭责任。旧 backend/rtp/KvcmPublisher 已删除，没有双实现或兼容入口。

变更使用有界内存队列（100000 条），每次最多取 1000 条，按同 key 最后一次转换合并，避免 Meta 先处理 ADD 再处理 DELETE 导致结果逆序。网络工作在独立线程中，每 20ms 检查增量、每秒心跳；网络失败、队列溢出或服务端要求恢复时，以当前快照修复，不扫描差集代替变更，也不保存磁盘事件历史。快照带本地版本界限，已被快照覆盖的队列事件会丢弃，避免旧事件覆盖新快照。正常关闭尽力上报 HOST_DOWN；意外退出由服务端心跳失效处理。

Master 与 KVCM 是两个独立消费者，不要求跨服务原子提交。KVCM 不可用时不会阻塞调度或 Master RPC；最近错误、注册状态和待发送数量可从 Engine 端点的 kvcm 状态查看。远端 KV 已提供统一准备流程和逻辑读写接口，但未安装真实存储适配；事件位置仅声明本地持有缓存，不能当成成功的远端 payload 写入。
```yaml
kv_cache_event_publisher_type: kvcm
kv_cache_event_manager_endpoint: http://<meta-http-host>:<port>
kv_cache_event_instance_group: mock-experiment
kv_cache_event_instance_id: deployment_128
kv_cache_event_snapshot_interval_ms: 600000
```

manager_endpoint 是显式 Meta HTTP 覆盖地址，优先于发现；未配置时，shared/cache/KvcmDiscovery 使用 kvcm_enable_vipserver、kvcm_vipserver_domain 或 kvcm_server_address，通过原生 GetClusterInfo 发现 Meta HTTP leader。成功地址缓存到失败后失效，依次尝试 bootstrap 候选，不做周期轮询。instance_id 必须显式配置为共享查询空间；host_ip_port 缺省为发布 IP 加 gRPC 端口减一，与 DSV Master 身份一致。显式装配远端适配器并开启 enable_remote_cache=true 后，即使关闭事件发布仍可注册并保持控制心跳；未装配适配器却开启此项会在启动时报错。NODE_REGISTER 不宣称有 HBM 事件；心跳 extra_info 中的 rtp_remote_cache_enabled 单独记录，不能被解释成数据面已经接通。正式环境需使用匹配的 model_name 和实例组。

`ExecuteFunction` 采用原生 FunctionRequest/Response：READ/WRITE 当前未提交任何外部 I/O，返回 FAILED_BUFFER_SAFE 且 payload_buffer_safe=true；其他未实现的 KVCM SDK 元数据操作返回 UNIMPLEMENTED。ShutdownRequest 的跨 rank 停机协调未模拟，返回 UNIMPLEMENTED，不生成排空证明；实例停止继续使用已有的进程生命周期和 TERM 清理路径。

当前缓存明确使用统一逻辑容量，只注册一个 Full spec；不建模 DSV4.1 Full/SWA/Linear 多资源混合池。GDR、真实远端 payload 读写及完成 Decode KV 向远端 DRAM 的提交均未实现。注册/事件接线测试不等于这些能力已经验证，不能用本模型给出上述容量和真实命中结论。端点能力明确标记 payload_io=false。

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

P/D 已补 `rtp_llm_{context,generate}_batch_size`、执行 Token 数、最长序列、逐请求等待/首 Token/总延迟、输入/输出长度与迭代数；`rtp_llm_stream_loading_cache_latency_us` 只统计 allocator loading 阶段，PD 协议等待不再混入。Query batch 的 native 语义是 beam 数，不能用调度批大小代替。

ViT 使用相同的物理元数据与 Prometheus instance 发现；QPS、Embedding 队列/等待/Forward/合批、请求图片数/输出长度，以及缓存命中/未命中/在途合并/淘汰和逻辑驻留量均按 real 名称导出。实际读取时提供 py_rtp_vit_download_rt（ms）和 rtp_llm_vit_image_fetch_rt_us（μs）；跳过读取时不补零。py_rtp_vit_preprocess_queue_size 是尚未完成准备的工作数（含排队和执行），py_rtp_vit_embedding_rt 仅计成功工作的就绪排队加 Forward，py_rtp_vit_request_rt 覆盖阻塞媒体调用的成功/失败全程，包括缓存命中等待但不含 RPC 传输和输出编码。未实现的预处理子进程和 CUDA Graph 不补零。详见 [指标覆盖与边界](../views/README.md)。

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

### RTP 调度实现边界

`RtpSchedulerBase` 共享请求状态、KV 引用、PD 生命周期、Forward 提交、取消和指标；`RtpScheduler` 仅做普通 FIFO 准入，`RtpGroupScheduler` 单独拥有显式组队列、残余成员和普通/组之间的选择。组标识不改变普通请求的调度方式。选择在 Engine 启动时完成，不从 Master 配置推断，也不在运行中按请求切换。`fifo` 模式应配合 Master 的 `NON_BATCH` 交付；它仍会动态组成多请求 Forward batch。本地准备立即完成；普通调度器支持远端异步准备，显式 group 暂不支持远端适配器。

### 缓存准备与远端接口

`KvCache.prepare` 返回准备操作或明确的永久/可重试容量错误。关闭远端时只执行本地匹配和分配，不创建传输线程。开启时，P 对本地缺失的连续前缀异步查询、读取；查到位置不增加复用量，读取成功才发布有效块并完成准备。读取失败保留有效本地前缀，由普通调度器重算缺失部分；请求总 deadline 不重置。D 的 PD 加载仍使用原有流程，不重复查询远端。

`shared/cache/remote/RemoteCache` 仅定义逻辑 match/read/write，位置凭据由适配器自己持有，KvCache 持有本地容量和引用。接口执行在锁外虚拟线程中，整个查询/读取共用一次任务；最多保留 1024 个在途读写，无额外等待队列。达到上限时读取回退本地，写入阶段返回异常。接口返回/抛异常必须表示已停止访问本次资源；interrupt 只请求取消，不作为释放证明。尚未实现真实数据传输、存储 SDK、KVCM 写会话或 KVS。

普通 RtpScheduler 将未就绪请求放入独立 CACHE_LOADING 阶段；完成通知只唤醒 EngineLoop，maintain 在调度锁内更新有效前缀、把请求放回 waiting 队尾。loading 不阻塞其他请求，也计入本轮准入预算与已初始化 KV 请求数。回调不查找可复用的 request_id；取消后原操作不能再发布。LOAD 引用一直持有到工作线程退出，即使请求已取消。

成功请求通过 `Lease.publishAndStore` 发布有效完整块，并以独立 STORE 引用提交后台写；请求终态不等写入，失败不回滚有效本地缓存。返回的 CompletionStage 反映写入及资源释放结果。节点关闭请求取消在途任务，未安全结束的任务继续持有引用，禁止提前复用块。`enable_remote_cache=false` 为默认；当前命令行/Whale 入口未安装适配器，不能直接开启为真实远端模式。测试仅通过显式 Java 装配注入可控适配器，不提供运行时假存储。

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

延迟/长度 native Gauge 使用采集窗口内事件平均值，QPS 使用事件数/窗口墙钟；辅助 _sum/_count 保留跨采集累计值，空闲窗口不重放旧延迟。没有真实 SDK、StartWrite/FinishWrite 分段或 payload 时，不导出对应 SDK、写会话分段、字节或带宽数据。指标回调失败不改变 I/O 结果。
