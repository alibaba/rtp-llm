# NaviSched Resource 与 Kernel 设计

## 1. 拆分原则

NaviSched 用 Resource 容纳可复用状态、跨激活逻辑和配置，用 Kernel 描述图中数据的到达、
顺序和输出。角色类只负责启动 Navi、注入跨 Navi 边界的资源、暴露 Client/Master/Agent
语义接口；它们不在 Python 或角色线程里执行调度循环、拟合循环或 collector 循环。

资源 stage 的使用规则如下：

- `RS_BIZ`：同一个 biz 的多张图可复用的状态，由 Navi 自动按需创建；这是默认选择。
- `RS_EXTERNAL`：仅用于上层执行器向 Navi 注入事件，或 Navi 向上层回调的边界。
- `RS_KERNEL`：只在 Kernel 级独占状态能明显减少重复逻辑时使用；当前核心状态无需为了
  拆分而过细地使用它。

当前所有 Kernel 依赖都不是 optional，统一使用 `KERNEL_DEPEND_ON`。依赖缺失时 Navi 会在
Kernel 执行前报错，`compute()` 不需要重复判空。若未来确有空和非空都成立的业务语义，才
使用 optional 依赖并在使用点检查。

## 2. 三个角色的资源边界

### Client

`ClientGraphRunner` 按通用 `bizName` 缓存 meta，并运行 Client 到 Master 的短图。它不是
Resource，因为它管理的是角色拥有的 runGraph 生命周期。请求携带框架给出的 token 数和
只读 block hash 数组，NaviSched 不 tokenize，也不计算正式 hash。

### Master

`MasterCollectorManager` 在出现请求流量前订阅并遍历所有发现到的 biz。每个 endpoint 的
`Spec` 单独拥有一张长驻 collector 图。所有 Agent 的 Navi 配置仍是 `partCount=1`、
`partId=0`；Spec 才是访问单个实例的实际地址。

Gig snapshot 变化会立即刷新 collector。建图失败、启动图失败或 collector 因错误结束时，
管理器只记录一次待重试状态，由一秒周期的重试线程重新读取 Gig 并重建失败连接；健康连接
不会被周期扫描或重建。正常 EOF 表示 Agent 已按协议结束，仍保留到 Gig 删除节点或启动 meta
发生变化。

`MasterStateR` 是 Master biz 的共享状态，调度图和采集图都依赖它。快照线程通过其公开
快照接口读取数据，不在格式化过程中持有节点 map 锁。

### Agent

Agent 对服务端执行器只注入凑批、延迟输入、forward 请求和 Report bridge 这些必须跨 Navi
边界交互的 Resource。Cache、状态上报、历史和模型由 Navi 自动创建，所有图内计算发生在 Agent
唯一的常驻图中。

## 3. Resource 说明

### `AgentReportR`

- Stage：`RS_BIZ`，由 Navi 自动创建。
- 依赖：`AgentCacheR` 和外部 `AgentReportBridgeR`。
- 持有握手、server 元数据、订阅者 AsyncPipe 与状态上报入口。
- 接收请求生命周期、原子 Cache 更新、延迟模型变化和 Agent 停止事件。
- 新 collector 订阅时依次得到当前握手和完整 Cache 快照；停止时向每个订阅者发送 stopping 标记和 EOF。
- 被 `AgentReportSourceKernel` 强依赖。

### `AgentCacheR`

- Stage：`RS_BIZ`，由 Navi 自动创建，容量通过其 `resource_config.total_blocks` 设置。
- 持有本机完整 block hash 集合，应用 Cache 增删并生成全量快照。
- 请求入队时按输入顺序计算连续前缀命中；首个 block miss 后停止匹配。

`AgentReportBridgeR` 是薄的 `RS_EXTERNAL` Resource，只让角色 facade 在 Navi 初始化后取得
`AgentReportR`。它保存 weak pointer，不持有状态，也不参与 Cache 或上报逻辑。

### `AgentBatchR`

- Stage：`RS_EXTERNAL`。
- 边界：调度结果或服务端请求到通用 batch callback。
- 以 batch id 归组请求；数量凑齐立即释放，等待超时释放部分请求，关闭后的迟到请求直接
  释放。
- 保存有界的已关闭 batch 标记，避免迟到请求再次打开旧 batch。
- 被 `AgentBatchKernel` 强依赖。

### `AgentLatencyInputR`

- Stage：`RS_EXTERNAL`。
- 边界：服务端执行负载和真实 batch 延迟到 Agent 常驻图。
- 用一个 AsyncPipe 接收排队/运行负载和完成 batch 样本。
- 不向上层暴露内部特征、模型参数或预测方法。
- 被 `AgentLatencyKernel` 强依赖。

### `AgentRequestR`

- Stage：`RS_EXTERNAL`。
- 边界：Client forward 到嵌入式执行器的请求与完成回调。
- 将 `NaviSchedForwardRequest` 交给上层执行器注册的语义 handler；正式代码只定义这个
  通用边界，不感知 handler 的具体实现。
- 被 `AgentRequestKernel` 强依赖。

### `AgentHistoryR`

- Stage：`RS_BIZ`。
- 持有有界延迟样本、拟合/保存计数和历史文件。
- 加载时校验 formula、binary version、revision 和参数；公式或二进制不匹配时拒绝旧参数，
  但保留样本用于冷启动拟合。
- 向拟合和落盘提供不可变历史快照；当旧快照仍被使用时才 copy-on-write，避免序列化时在
  历史锁下复制完整数据。

### `NonLinearPrefillModelR`

- Stage：`RS_BIZ`。
- 当前唯一公式的资源：九参数非线性 prefill 模型及其 Adam 拟合状态。
- 在加载、预测和拟合前严格校验九个参数。
- 未出现第二种公式前不预设无用的模型基类或派生层次。

### `LatencyModelR`

- Stage：`RS_BIZ`，依赖 `NonLinearPrefillModelR` 与 `AgentHistoryR`。
- 协调模型预测、批量样本拟合、冷启动、模型 revision 与持久化请求。
- 参数以不可变快照整体替换；预测只能观察到完整旧快照或完整新快照。
- 参数变化时通知 `AgentReportR`，由 collector 流推送给 Master。
- 一次真实推理 batch 的规模由 `NaviSchedRequestStats` 向量的元素数表示；
  `latencyFitBatchSize` 是多少个已完成推理 batch 触发一次模型更新；Agent 凑批的
  `expectedBatchSize` 是独立概念。
- 一个已完成推理 batch 只追加一个样本：将其中所有查询的特征聚合，标签为该 batch 的实际
  执行延迟。队列预测也以 batch 为单位聚合后调用同一模型，不能把 batch 延迟作为多个单请求
  样本的标签。

### `MasterStateR`

- Stage：`RS_BIZ`。
- 节点索引分为两层：尚在建连、握手或停止阶段的节点保存在按 Agent ID 打平的 pending map；
  完成握手且服务地址有效的节点进入对应 biz 的 ready map。
- 每个 biz 的 ready map 使用 COW 更新。调度只取得目标 biz 的不可变 map 快照，随后遍历时不再
  持有节点索引锁，也不会复制其他 biz 的节点。
- 每张 collector 图持有自己创建的 `shared_ptr<NodeState>`，后续上报直接更新该对象；同 Agent ID
  的新图替换旧图后，旧图结束时通过指针身份校验，不能删除新图节点。
- 节点索引锁只保护 pending/ready 间的迁移、增删和查找；每个节点独立的锁保护高频上报、
  Cache 和调度读取。
- 一份 Cache 更新在同一次节点写锁内处理删除、增加和全量替换，调度不会读到中间态。
- 同一 Agent collector 流按到达顺序更新状态，不依赖 Agent 与 Master 的时钟同步。
- Agent 上报的 `eventTimeUs` 只用于调试展示；队列数据时效使用 Master 收到上报时记录的本地时间。
- 提供调度输入快照，不实现用户的节点选择算法。

### `MasterScheduleR`

- Stage：`RS_BIZ`。
- 连接 Client 的多个短图与 Master 的唯一常驻调度图。
- 持有常驻图输入 AsyncPipe 和提交计数，不持有策略、节点状态或跨请求历史。
- 每次提交同时携带只读请求和来源短图的 response AsyncPipe。内部单调提交 id 与用户
  request id 分离，仅用于本地追踪。

## 4. Kernel 说明

### `ScheduleKernel`

读取 Client NamedData，创建当前短图的 response AsyncPipe，将两者提交给
`MasterScheduleR`。下一次被 response pipe 唤醒时输出 `MasterResponseToClientData`。它不
保留请求历史，也不实现策略。

### `MasterScheduleContextKernel`

持有 Master 常驻图的唯一等待队列。请求数据和一次性定时激活共用 `MasterScheduleR` 提供的
request pipe；由数量阈值、最长等待时间或请求 timeout 触发调度。每轮按目标 biz 构造一个或
多个 `MasterScheduleContextData`，其中同时保存策略输入和来源请求的 response pipe。某个请求
开启收集时，它还会在策略执行前旁路记录当轮 replay 输入。没有请求时不注册 timer；request
pipe EOF 后排空等待请求，再输出 EOF。
运行期无法注册下一次 timer 时返回 `EC_ABORT`，因为此时图 timer 已不可用，常驻图无法继续
保证等待请求会被唤醒。

### Scheduler 执行路径

Master 根据配置选择一条 Scheduler 执行路径。普通策略由一个 Scheduler Kernel 接收
`MasterScheduleContextData`，通过 `NaviSchedScheduleContext` 读取 request 集合、node 快照和
Cache 命中结果，为每个请求选择节点或拒绝请求，再输出同一个 Data。CostScheduler 则展开为
按 Agent 分片的梯度阶段、required barrier、按 request 分片的投影阶段和 required barrier；
全部阶段仍传递同一个 Context，最后统一输出 decisions。

策略不接触 request pipe、定时器、biz 分组和 response pipe。无状态同步策略不需要 Resource；
有配置或状态时由使用它的 Kernel 依赖对应 `RS_KERNEL` Resource。

### `MasterScheduleResultKernel`

接收 Scheduler 输出的同一个 `MasterScheduleContextData`，校验 decisions 数量和错误状态，
将最终 decisions 补入已采集的 replay，并完成每个来源请求的 response pipe。输入 EOF 时输出
常驻图 EOF；它不执行节点选择，也不保存跨轮状态。

### `ClientForwardKernel`

位于 Client 短图的决策之后。读取 Agent 地址并 fork 目标 Agent 图，把请求和决策作为
NamedData 传入；不另起上层可见 RPC 协议。限流或失败决策不进入该 Kernel 的执行路径。

### `AgentReportSourceKernel`

将 `AgentReportR` 的一个语义事件转换为一份 `AgentUpdateToMasterData` 并输出到唯一的
分布式边。完整 envelope 由 Resource 先组合，所以同一次事件只造成一次跨图 IO 激活。

### `MasterUpdateKernel`

接收一个 Agent envelope，并按控制、请求、Cache、延迟等 section 的语义顺序更新
`MasterStateR`。同一事件的多个 section 在一次本地激活中生效，避免拆成多次状态更新。

### `AgentBatchKernel`

从 `AgentBatchR` 的唯一 activation AsyncPipe 接收主动唤醒、周期唤醒和结束信号，触发 ready
callback。batch timeout 注册到同一条 pipe；Kernel 按当前 batch 状态推进，不在 Navi worker
中条件等待，也不依赖唤醒来源。

### `AgentLatencyKernel`

读取 `AgentLatencyInputR`，将服务端语义事件转成模型特征，记录完成 batch、预测队列时延，
在达到配置的样本量时拟合，并将变化的模型参数发布给 `AgentReportR`。落盘请求从旁路边
交给 `AgentHistorySaveKernel`。预测队列时延由 queued batch 预测总和与 running batch 预测
剩余时延组成；running 已执行时间只从自身预测值中扣除，超过预测值后不再减少后续队列时延。

### `AgentHistorySaveKernel`

消费延迟 Kernel 的保存旁路，调用 `AgentHistoryR` 保存当前不可变模型快照。Agent 正常停止
时接收 EOF，完成最后一次保存后结束输出。

### `AgentRequestKernel`

将 Client forward 图送入的请求交给 `AgentRequestR` 的上层 handler，并按 handler 的完成
结果驱动图结束。它不包含模型执行逻辑。

## 5. 长驻图中的时序保证

```text
Client NamedData -> ScheduleKernel -> MasterScheduleR pipe
                 -> MasterScheduleContextKernel
                 -> 同一个 MasterScheduleContextData
                 -> Scheduler Kernel 或 CostScheduler 并行 DAG
                 -> context.selectNode / reject
                 -> 同一个 MasterScheduleContextData
                 -> MasterScheduleResultKernel -> response pipe -> Client 输出

AgentReportR pipe -> AgentReportSourceKernel -> 一条分布式边
                  -> MasterUpdateKernel -> MasterStateR
```

同一 runGraph 会话内 Navi 流保证可靠、有序。连接结束只影响其所在图；重新发现 endpoint
时由新的图和握手恢复。所有周期触发使用 Navi 的 `EvTimer`，没有额外的调度、拟合或模拟
执行线程。

跨常驻图生命周期的 External Resource 保留最后一次注入的 pipe。图结束时该 pipe 被 Navi
置为 terminated，后续入口调用 `setData()` 会立即返回 `EC_ASYNC_TERMINATED`，请求不会进入
Resource 的等待队列。角色重建常驻图后，新 Kernel 在 `init()` 中用新 pipe 覆盖旧 pipe，入口
随即恢复。Agent 模拟执行入口失效时，本次实际 batch 的多个请求统一结束并回滚负载计数。

## 6. 扩展约束

- 新策略组合 Master 图节点，复用现有请求、状态和决策数据，不把策略下沉到 Python。
- 新公式先增加 C++ Resource/Kernel、formula id、参数数量校验和持久化/通信校验；没有第二
  种公式时不引入抽象基类。
- 新 Agent 状态优先加入已有的 `AgentUpdateToMasterData`，而不是增加跨图边。
- 新框架适配只转译框架已有数据；不复制 Agent 的 Cache、inflight、凑批、拟合或历史逻辑。
- simulate 只能链接测试 build；正式发布物不得带模拟 Kernel、HTTP 伪装或测试启动器。

## 7. 测试模拟实现

simulate build 在正式 Agent 接口之后绑定测试执行器，正式 Resource、Kernel 和发布包均不依赖
这部分代码。

`SimulatedExecutionR` 是 `RS_KERNEL` Resource。它由模拟执行 Kernel 自动创建，通过一个测试
专用 bridge 取得正式 Agent facade，并在 `AgentRequestR` 的通用边界注册 completion handler。
它复用 `AgentBatchR` 凑批，持有测试 Cache、LRU 淘汰状态以及等待和执行中的 batch；执行完成后
只调用正式 Agent 的 batch、Cache 和延迟汇报接口。

`SimulatedExecutionKernel` 通过 Navi periodic pipe 驱动截止时间，同一时刻只执行一个 batch，
完成后才启动下一批。停止时拒绝新请求并排空已经接收的 batch。模拟执行没有模型服务、私有
执行线程或第二套通信协议。

```text
Client forward -> 正式 AgentRequestR handler 边界
               -> SimulatedExecutionR -> 正式 AgentBatchR
               -> SimulatedExecutionKernel periodic pipe
               -> 正式 Agent batch/cache/latency 汇报接口
```
