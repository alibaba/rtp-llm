# NaviSched 代码阅读顺序

本文按一次真实运行从启动到停止的顺序组织。不要从单个 Kernel 开始
读；先理解角色、常驻图和 Resource 的归属，Kernel 的职责才会清晰。

## 1. 先看对外语义接口

按以下顺序阅读：

1. `NaviSchedCommon.h`
   - 公开 C++ 数据契约：Agent 配置、请求 token 统计、Cache hash 和凑批回调。
   - block hash 由调用方以 `shared_ptr<const NaviSchedBlockHashes>` 传入；
     NaviSched 不负责 tokenize，也不负责计算 hash。

2. `scheduler/ScheduleContext.h`、`scheduler/SchedulerBaseKernel.h`
   - Scheduler 就是 Master 常驻图中可替换的策略 Kernel。具体 Kernel 继承
     `SchedulerBaseKernel`，只覆写请求与节点的匹配逻辑。
   - 无状态策略只需要 Kernel；需要配置、状态或其他 Resource 时，再由具体 Kernel
     依赖自己的 `RS_KERNEL` Resource。
   - Master 只根据 `scheduler_name` 将
     `navi_sched.<scheduler_name>_scheduler_k` 画入常驻图，不持有策略对象。
   - `NaviSchedScheduleContext` 是该目录内部的策略调用上下文；策略为每个请求调用
     `selectNode()` 或 `reject()`，不属于上层 API。

3. `navi_sched/agent.py`、`navi_sched/master.py`、
   `navi_sched/client.py`、`bindings/Pybindings.cpp`
   - Python 是三个 C++ 角色类的薄封装。C++ 头文件才是权威接口；Python
     只做列表到 C++ 对象的转换，并在可能阻塞的角色调用期间释放 GIL。

4. `config/NaviSchedConfigBuilder.cpp`
   - 将角色配置转换为 Navi 配置。每个角色都是一个 biz，固定
     `part_count = 1`、`part_id = 0`。
   - 从这里看 local node、CM2、VIP 订阅、任务队列、kernel 列表和
     Resource 配置如何进入 Navi。

5. `example/embedded_cluster.py`
   - 最短的完整执行示例：启动 Agent、Master、Client，等待 Agent meta，
     发送调度请求，最后依次停止所有角色。

## 2. 三个角色共用的启动与停止逻辑

先读 `service/NaviSchedRole.cpp`，再读 Client、Master 和内部的
`NaviSchedAgentImpl`。公开的 `NaviSchedAgent` 只是面向模型服务的 facade，不暴露角色继承和
Navi 初始化细节。

`NaviSchedRole::init()` 的公共启动顺序是：

1. 创建角色私有的 `navi::Navi`，设置 `waitNoQuery`。
2. 按需初始化 KMonitor。
3. 创建并初始化 `GigRpcServer`；按需启动 gRPC server。
4. 调用 `Navi::initWithGigRpcServer()`，让 Navi 与角色共同持有同一个
   `shared_ptr<GigRpcServer>`。
5. 按需启动 snapshot 的 `LoopThread`。

`NaviSchedRole::update()` 会先停 snapshot，构造 `NaviConfig`，执行角色
的 update hook，调用 `Navi::update()`，最后重启 snapshot。具体角色在
`onUpdated()` 中创建 Resource 并启动常驻图。`stop()` 按 snapshot、Navi、
gRPC server、KMonitor 的顺序释放公共资源。

## 3. 先启动 Agent

按以下顺序阅读：

1. `service/NaviSchedAgent.cpp`
2. `service/NaviSchedAgentImpl.h`
3. `resource/AgentReportR.{h,cpp}`
4. `graph/AgentGraph.cpp`
5. `kernel/AgentBatchKernel.cpp` 与 `resource/AgentBatchR.cpp`
6. `kernel/AgentLatencyKernel.cpp`
7. `resource/AgentLatencyInputR.cpp`
8. `resource/LatencyModelR.cpp`、`resource/NonLinearPrefillModelR.cpp`、
   `resource/AgentHistoryR.cpp`

### Agent 初始化

`NaviSchedAgent` 在 `Navi::update()` 前创建四个外部边界 Resource：

- `AgentBatchR`：保存未凑齐的调度 batch 及其 callback。
- `AgentLatencyInputR`：接收执行负载与已完成 batch 的延迟样本。
- `AgentReportBridgeR`：在 Navi 初始化后把自动创建的 `AgentReportR` 交回角色 facade。
- `AgentRequestR`：保存可选的框架请求转发 handler。

`AgentReportR` 和 `AgentCacheR` 是 Navi 自动创建的 `RS_BIZ` Resource。`AgentReportR`
依赖 `AgentCacheR` 与外部 `AgentReportBridgeR`，负责接收生命周期/cache 汇报并向 Master
采集流扇出；角色不会手动创建或注入它。

`NaviSchedAgentImpl::onUpdated()` 为一个 Agent biz 只启动一张常驻
`AgentGraph`。正常图中有两个相互独立的分支：

```text
AgentBatchR graph pipe -> agent_batch_k -> batch_finished
AgentLatencyInputR pipe -> agent_latency_k -> agent_history_save_k
                                      |             |
                                      +-> latency_finished
```

`AgentGraph` 使用很长的 timeout，正常停止时由 Resource 持有的 pipe 结束。
它不会为每一个查询创建新图。

### Agent 凑批路径

LLM 适配层调用 `NaviSchedAgent::addBatchRequest()`。`AgentBatchR` 按
Master 返回的 `batchId` 保存请求，并在以下任一情况调用 callback：

- 收齐 `expectedBatchSize` 个请求；
- 到达 `batch_timeout_ms`。

首个未完成 batch 通过唯一的 activation pipe 唤醒常驻图。`AgentBatchKernel` 只在有未完成
batch 时把下一超时点注册到同一条 pipe；每次激活先消费 pipe 状态，再调用
`AgentBatchR::expireBatches()` 并决定是否注册下一次。请求唤醒和周期唤醒不需要区分，EOF
只表示 Agent 图结束。整个过程不在 Navi worker 中做条件等待。

周期注册是一次性的。凑齐 batch 后或没有待处理 batch 时，Kernel 会把对应注册置为 inactive。
LocalSubGraph 只保存拥有 periodic pipe 的 Node 快照，不保存或回收 pipe；inactive Node 在该
Node 下一次创建 AsyncPipe 时延迟移出快照。timeout 扫描对普通 pipe 不加锁，对 periodic pipe
使用 trylock；与 Kernel 调整定时器竞争时跳过本轮，调整完成后的 timeout refresh 会补扫。
撤销失败只记录告警，不终止常驻图；运行期注册失败表示图 timer 已不可用，对应 Kernel 返回
`EC_ABORT` 终止当前常驻图。

### Agent 汇报与延迟路径

适配层只调用 `addBatchRequest`、batch reporter 的 `startBatch/finishBatch`，以及
`reportCacheInserted/reportCacheRemoved`。请求接收、batch 就绪、开始、完成和异常结束事件由
Agent 根据这些调用自动生成。

- Cache 增删立即进入 `AgentReportR`。
- batch 生命周期和延迟样本进入 `AgentLatencyInputR`，唤醒常驻图中的 `AgentLatencyKernel`。
- `AgentLatencyKernel` 对负载汇报预测队列总延迟。该值包含当前 running batch 的预测剩余时延
  和后续 queued batch 的预测时延总和；running 已执行时间最多扣到自身预测值，不会侵蚀后续
  queued batch。每个完成执行 batch 聚合为一个特征样本，
  用该 batch 的实际执行延迟更新 `LatencyModelR`，仅在参数变化时汇报新的模型快照。
- `latencyFitBatchSize` 计数的是完成执行 batch 样本，不是 batch 内请求数，也不是凑批的
  `expectedBatchSize`。预测与 simulate 真值使用同一批特征聚合和公式。
- `LatencyModelR` 持有当前不可变快照；`NonLinearPrefillModelR` 保存
  九参数公式与拟合逻辑；`AgentHistoryR` 保存有限历史与持久化模型状态。
- `AgentHistorySaveKernel` 位于独立图分支，磁盘写入不会阻塞拟合和汇报。

## 4. 启动 Master 并建立 Agent 流

按以下顺序阅读：

1. `service/NaviSchedMaster.cpp`
2. `graph/MasterCollectorManager.cpp`
3. `kernel/AgentReportSourceKernel.cpp`
4. `kernel/MasterUpdateKernel.cpp`
5. `resource/MasterStateR.{h,cpp}`

### Master 初始化

`NaviSchedMaster::onUpdated()` 创建 biz 级的 `MasterStateR` 与
`MasterScheduleR`，然后启动两套互相独立的常驻机制：

1. `MasterScheduleGraph`：本地长驻调度图。
2. `MasterCollectorManager`：订阅 Gig 发现到的所有 biz，并对每一个带
   gRPC-stream 端口的 endpoint 建一张独立的长驻采集图。

采集管理器通过 `SearchService::getBizNames()` 和 `getBizNodeInfos()`
读取初始拓扑，并在拓扑 snapshot 通知后刷新。每个 endpoint 都按 IP 和
gRPC-stream 端口直接访问，因此某一台 Agent 的图结束后，只需要关闭并
重建这一台的采集图，不影响其他 Agent。

拓扑变化立即生效；连接图异常结束以及建图、启动图直接失败只设置重试标记，
`sched_retry` 每秒处理一次失败连接。这个线程没有失败任务时不遍历 Gig，也不检查健康
collector，避免节点规模增大后产生固定扫描成本。正常 EOF 不进入重试，Gig 节点删除和启动
meta 变化继续由 snapshot 通知处理。

### Agent 到 Master 的流式图

每个 Agent endpoint 对应一张如下分布式图：

```text
Agent 子图                                         Master 子图
agent_report_source_k -- AgentUpdateToMasterData --> master_update_k
                                                          |
                                                      finished/eof
```

`AgentReportSourceKernel` 创建 async pipe 并订阅 `AgentReportR`。订阅时
立即发送 handshake 和当前全量 cache 状态；之后 Agent 产生的增量更新在同一条
严格保序的可靠流中推送。

`MasterUpdateKernel` 将一条 `AgentUpdateToMasterData` 拆为可选的 control、
request/load、cache 和 latency section，并写入 `MasterStateR`。该 Resource
才是真正的状态归属：

- 正在建连、尚未握手或已经 stopping 的节点保存在按 Agent ID 打平的 pending map；
- 握手完成且服务地址有效的节点进入对应 biz 的 COW ready map，只有这里的节点参与调度；
- 每节点读写锁保护的可变状态和 KV prefix hash；
- 可整体替换的延迟模型快照；
- 调试历史和 collector 绑定关系。

collector 图收到第一条有效上报时创建自己的 `NodeState`，之后通过该指针直接更新，不再按
Agent ID 重复查找。新实例使用相同 Agent ID 建图时会替换节点索引中的旧对象；旧图结束只删除
自己持有的对象。图结束后，节点会从当时所在的 pending 或 ready map 删除。

同一个 collector 图中的更新按 Navi 流到达顺序应用，不比较 Agent 时间戳。
`AgentUpdateToMasterData::eventTimeUs` 只记录 Agent 产生该事件的本地时间，供 summary、snapshot
和问题排查使用；Master 不用它排序，也不与 Master 本地时钟相减。队列状态的时效由 Master
收到上报时记录的本地 `queueReportReceiveTimeUs` 计算。

Agent 进程实例由 Gig 发布的 `AgentInstanceR` meta 区分。这个 meta 只用于
`MasterCollectorManager` 判断是否需要重建 collector，不进入 `MasterStateR` 的运行时节点状态。
新 collector 订阅后固定先收到包含当前最新延迟模型的 handshake，再收到一份全量 Cache 快照；
已有 collector 不因模型更新重复握手，后续模型与 Cache 变化都走增量更新。

这里是检查 cache prefix 匹配和 Agent 下线处理的第一处入口。

## 5. 启动 Client 并跟踪一次调度请求

按以下顺序阅读：

1. `service/NaviSchedClient.cpp`
2. `graph/ClientGraphRunner.cpp`
3. `kernel/ScheduleKernel.cpp`
4. `resource/MasterScheduleR.cpp`
5. `graph/MasterScheduleGraph.cpp`
6. `scheduler/MasterScheduleContextKernel.cpp`
7. `scheduler/SchedulerBaseKernel.cpp`
8. 所选策略的 `scheduler/*SchedulerKernel.cpp`，以及它实际依赖的可选 Resource。

### Client 短图

`NaviSchedClient::schedule()` 创建 `ClientRequestToMasterData`，交给
`ClientGraphRunner`。

Runner 直接使用外部传入的 block hash 序列发送 `schedule` 请求。每次请求都会
创建一张短暂的分布式图；Master 子图中只有
`navi_sched.schedule_k`。请求作为 named data 写入子图，最终从图输出读取
`MasterResponseToClientData`。

### 请求进入 Master 常驻图

`ScheduleKernel::init()` 是 Client 短图到 Master 常驻图的桥：

1. 为当前请求创建 response async pipe。
2. 通过 `MasterStateR` 校验并统计客户端请求。
3. 向 `MasterScheduleR` 提交
   `MasterScheduleRequestData(request, responsePipe)`。
4. `compute()` 只等待该 async pipe；Master 返回结果时，输出客户端图数据
   与 eof。

`MasterScheduleR` 是持有常驻图输入 pipe 的小 Resource。常驻图尚未完成
初始化时，它先保存请求；之后直接写入 pipe。图结束后 Resource 保留 terminated pipe，后续
请求会立即失败而不是重新进入无人消费的等待队列；新常驻图初始化时再用新 pipe 覆盖它。

### Master 常驻批调度

`MasterScheduleGraph` 包含固定的 `MasterScheduleContextKernel`、配置选择的
`navi_sched.<scheduler_name>_scheduler_k` 和固定的 `MasterScheduleResultKernel`。只有前置
Context Kernel 持有 `MasterScheduleR` 提供的请求 pipe。只有请求等待时，它才把一次性
periodic 调度注册到同一个 pipe；请求数据和定时激活均唤醒该 Kernel。每轮调度后，它先 drain
策略执行期间到达的请求，再从所有等待请求的 `enqueueTimeUs` 计算最早剩余等待时间，作为
下一次一次性 timer 的延迟；不是固定周期重注册。

每次激活后，它先 drain 所有输入。满足以下任一条件即调度：

- 等待请求数达到 `max_wait_schedule_count`；
- 最早请求已等待 `max_wait_schedule_time_ms`；
- 某请求的 timeout 要求立即启动下一次调度；
- 停止过程中请求 pipe 收到 eof。

随后它按 `max_wait_schedule_count` 取走本轮请求，剩余请求保留到下一轮，并按 `bizName`
生成一个或多个 `MasterScheduleContextData`。同一个 Data 依次经过策略 Kernel 和 Result
Kernel；若本次调度耗时已经跨过下一批请求的最近截止时间，Context Kernel 会立即自唤醒。

具体调用链是：

```text
MasterScheduleContextKernel
  -> MasterScheduleContextData(requests, nodes, response pipes, replay)
  -> XxxSchedulerKernel
  -> NaviSchedScheduleContext::selectNode / reject
  -> 框架在同一个 context 中保存每个输入请求的选择结果
  -> MasterScheduleResultKernel
  -> MasterResponseToClientData response pipe
```

策略拿到的是当轮固定的节点信息快照。Cache 匹配始终通过 context 访问对应节点；在线节点读取
正式 Cache，离线 replay runner 则预先构造提供同一接口的假节点和假 Cache。节点信息在构造
context 时只读取一次，不会在策略循环中重复复制或持有 Master 节点 map 的锁。

## 6. 可选的 Client 到 Agent 转发

正常选址链路读清楚后，再读这一段：

1. `kernel/ClientForwardKernel.cpp`
2. `kernel/AgentRequestKernel.cpp`
3. `resource/AgentRequestR.cpp`
4. `adapters/rtp_llm/` 中对应的 C++ 插桩代码。

`NaviSchedClientConfig::enableAgentForward = false` 时，Client 只将选中的
地址返回给调用方，由 LLM 框架保持原有发送路径。

该选项为 true 时，Client 图在 `schedule_k` 后增加 `client_forward_k`。
它根据目标 Agent 的直连 gRPC 地址 fork 一张子图，将原始请求和调度结果
作为 named data 传给 Agent。Agent 的 `agent_request_k` 将语义请求提交给
`AgentRequestR`，后者调用上层执行器注册的 completion handler，最后通过 fork 图返回调度
结果。正式 NaviSched 不关心 handler 是模型框架还是其他执行器；未注册 handler 时请求明确
失败。forward 是可选的数据路径，不替代正常的仅选址集成。

## 7. 按相反顺序跟踪停止

排查泄漏、常驻图卡住或 VIS 缺失时，按下列顺序阅读：

1. `NaviSchedAgent::stop()`：结束 Agent request handler、batch 和 latency
   常驻图 pipe，发送带 eof 的 `agent_stopping`，再停止 Navi。
2. `MasterUpdateKernel`：应用该 control 更新；Master 将该 Agent 标记为 inactive
   并返回 stop ack。Agent 实例身份由 Gig 发布的 `navi_sched.agent_instance_r` meta
   判断，meta 变化时 `MasterCollectorManager` 重建 collector。
3. `MasterCollectorManager::closeRoute()`：等待受影响的采集图结束并删除
   绑定；其他 Agent 的采集图继续运行。
4. `NaviSchedMaster::stop()`：结束 `MasterScheduleR`。Context Kernel 排空请求并输出 EOF，
   EOF 依次经过选中的同步 Scheduler Kernel 或 CostScheduler 并行 DAG，再由 Result Kernel
   结束调度常驻图，随后停止采集图和 Navi。
5. `NaviSchedClient::stop()`：在公共角色停止后释放短图 runner。

`graph/GraphVisRecorder.cpp` 是统一的 VIS 保存逻辑。开启 `enableVis` 时，
当前工作目录会产生 `client_graph.vis`、`master_graph.vis`、
`master_schedule_graph.vis` 和 `agent_graph.vis`。性能测试必须关闭 VIS。

## 8. 最后再读框架适配层

核心流程读完后再看适配，避免把框架插桩误认为调度核心：

1. `rtp_llm_adapter.md`，然后 `adapters/rtp_llm/RtpLLMAdapter.{h,cpp}`。
2. `test/rtp_llm/RtpLLMEndToEndTest.py`，确认 C++ 插入点。

适配层应保持薄：创建和配置 Agent/Client，传入外部计算好的 block hash，
通过 batch reporter 和 Cache 增删接口汇报真实执行状态。调度、拓扑流、批超时、
延迟拟合、历史持久化和 Master 状态始终属于本文前述的 C++ Navi 图和
Resource。

## 9. 最后阅读测试模拟

只有需要理解本地集群回放时，才继续阅读 `test/simulator/`：

1. `SimulatedAgentAccess.cpp` 将测试执行器绑定到正式 `NaviSchedAgent`。
2. `SimulatedExecutionR.cpp` 在正式 `AgentRequestR` 边界注册测试 handler，并复用正式凑批、
   Cache 和汇报 API。
3. `SimulatedExecutionKernel.cpp` 用 Navi periodic pipe 串行驱动 batch 执行。
4. `TrafficReplayR.cpp` 加载按 service 切分的查询数据；Traffic Replay Kernel 按统一时间基准
   调用正式 Client forward。

simulate 只存在于测试链接目标。正式 Agent request 流程和正式发布包不引用这些类型。
