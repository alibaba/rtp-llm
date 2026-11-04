# NaviSched 角色适配教程

本文面向把 NaviSched 接入一个推理框架的开发者，说明 Client、Master、Agent 三个角色各自
需要部署什么、调用什么接口，以及每个字段表达的事实。NaviSched 负责收集节点状态、选择节点
和可选的请求转发；模型执行、tokenize、流式响应、取消和框架自身的 KV Cache 仍由推理框架
负责。

已有实现的边界说明见 [整体设计](navi_sched_design.md)、[RTP-LLM 适配](rtp_llm_adapter.md)
。Client 和 Master 通过 Python API 接入；嵌入 Prefill
执行器的 C++ 代码只包含 `navi_sched/NaviSchedAgent.h`。

## 1. 先确定部署形态

一个独立可选的推理 `ip:port` 对应一个 Agent。Client 和 Master 可以独立扩缩容：

```text
业务入口 / Router
        |
        v
   NaviSched Client ---- NaviSched Master
        |                       ^
        |                       |
        +-- 可选 forward --------+---- Agent 状态与 Cache 增量
                                |
                         推理 Agent 1 ... 推理 Agent N
```

常规模式中，Client 返回 `NaviSchedDecision`，业务入口用 `target` 中的地址继续调用框架
原有 RPC/HTTP/SSE 发送逻辑。开启 Client forward 后，Client 在取得一个目标后，将请求交给
目标 Agent；这个模式只适用于 Agent 已有非阻塞接收入口的场景。两种模式都不替换模型框架的
执行器。

角色启动顺序建议为：先启动 Master 并完成发现配置，再启动 Agent，最后启动 Client。Client
在 Master 尚未发现 Agent 时可能得到空目标，这是一个正常的“当前无可用节点”调度结果。

## 2. 通用数据与时间单位

### Block hash

`NaviSchedBlockHashes` 是 `std::vector<uint64_t>`。它必须来自推理框架自身的 KV Cache
key 或 rolling block hash，Client 请求和 Agent Cache 更新必须使用同一套值。NaviSched 不在
生产链路计算 token hash、block size 或 hash 算法。

`NaviSchedBlockHashesConstPtr` 是共享只读数组。调用方可直接交给 NaviSched，不必先复制；
调用期间及异步处理期间该共享对象必须保持有效。Python 列表会在绑定层转换为对应的 C++ 数组。

### 时间与计数

- 名称以 `Ms` 结尾的值单位为毫秒。
- 名称以 `Us` 结尾的值单位为微秒；C++ 调用方使用
  `autil::TimeUtility::currentTime()` 取得当前值。
- `timeoutMs` 必须为正数，表示从 Client 发起调度到取得决策的总时限。
- `NaviSchedBatchRequest.totalTokenCount` 和完成时的 `NaviSchedRequestStats` 都使用 token 数；
  节点 Cache 容量和增删仍使用完整 block 数。
- `requestId` 在 Client、Agent 和框架回调之间必须稳定一致。它用于关联生命周期、决策和
  调试 replay。

### Cache 更新

Cache 总 block 容量在构造 Agent 时通过 `totalBlocks` 提供，初始化后不动态改变。运行期间框架只在
完整 block 真正插入或删除时调用 `reportCacheInserted()`、`reportCacheRemoved()`，参数是本次
变化的共享只读 hash 数组；Agent 自动生成事件时间。

## 3. 角色启动与发现配置

Python 使用方直接构造 `NaviSchedMaster`、`NaviSchedClient` 或 `NaviSchedAgent` 并调用 `init()`。
`biz_name` 属于各角色本身；Client 在每次 `schedule()` 中通过 `target_biz_name` 指定要调度的
Agent biz。服务发现和运行容量由部署侧的角色配置提供，使用方不需要直接操作 Navi 的内部对象。

`init()` 成功后角色开始工作；停止顺序始终是业务入口先停止接收新请求，再调用 `stop()`。
Agent 的 `stop()` 会先上报 stopping 并等待 Master 确认或超时，再关闭 collector；调用方不应
绕过它直接销毁进程。

C++ 只在嵌入 Prefill 执行器时使用 `NaviSchedAgent`。它的构造配置和字段说明见 [API](api.md)
的“C++ Agent API”，Client 和 Master 没有需要由 C++ 使用方直接构造的公开接口。

## 4. Master：发现节点并安排调度窗口

Master 没有逐请求的公开“调度 API”。它由 Client 的分布式图请求驱动，并通过 Agent collector
持续更新节点状态。业务侧只需要创建、配置、观察和停止它。

Master 的 `biz_name` 是 Client 用于发现和请求 Master 的 Navi biz。它不是 Master 绑定的
prefill biz；一个 Master 可同时接收多个 Agent biz 的状态，并按每个
`schedule(..., target_biz_name=...)` 请求独立调度。

```python
from navi_sched import NaviSchedMaster

master = NaviSchedMaster(
    "master",
    max_wait_schedule_count=30,
    max_wait_schedule_time_ms=30,
    scheduler_name="random",
)
master.init()
```

`NaviSchedMaster` 的构造参数、发现配置和停止方式见 [API](api.md)。它只负责运行调度中心；
状态查询后续通过 RPC 提供，不通过 Python 读取内部节点快照。

调度会按 biz 分组。一次调度的多个请求不必分到同一个 Agent，也不保证一个 Agent 获得请求；具体
策略只从本轮节点状态、已计算 Cache 命中和队列预测数据中作决定。某个 Client 的 response pipe
已经超时，只会影响该请求，不能让 Master 常驻图退出。

## 5. Client：在发送模型请求之前选择节点

Client 的唯一业务 API 是 `schedule`。必须在框架已经得到准确 prompt token 数和 block hash 后、
在原有模型 RPC/HTTP 请求发送之前调用。它返回成功不等于模型执行成功，只表示调度图返回了一个
决策。

```python
from navi_sched import NaviSchedClient, NaviSchedRequest

client = NaviSchedClient("client", "master")
client.init()
request = NaviSchedRequest(
    prompt_token_count=prompt_token_count,
    block_hashes=rolling_hashes,
    request_id=request_id,
    block_size=kv_cache_block_size,
)
decision = client.schedule(
    request, target_biz_name="prefill", timeout_ms=1000)
```

`schedule()` 的参数和 `NaviSchedDecision` 字段见 [API](api.md)。Python API 中 `target` 为 `None` 表示当前没有可用
节点或策略选择了限流；调用方按 `reason` 走已有的拒绝、重试或降级逻辑。forward 仅用于模拟与
测试，说明集中在 API 文档末尾。

## 6. Agent：在执行器插桩点上报事实

Agent 在每个独立可选推理服务进程中创建一次。构造函数中的 `serverIp`、`serverPort` 是 Master
最终返回给 Client 的推理服务地址，不是 Agent 自己的 collector gRPC 地址。

```cpp
navi_sched::NaviSchedAgentConfig agentConfig;
agentConfig.service = serviceConfig;
agentConfig.runtime.bizName = "prefill";
navi_sched::NaviSchedAgent agent(
    agentConfig, modelServerPort, modelServerIp, cacheTotalBlocks);
agent.init();
```

Agent 公开 API 应按下面的生命周期调用。所有函数都只做异步状态投递；调用方不得等待 Master
响应或在 Navi worker 内同步等待 callback。

### 6.1 接收请求并等待 batch

每个到达 Agent 的请求都调用 `addBatchRequest()`。如果 Master 没有为它分配 batch，传入
`batchId=0`、`expectedBatchSize=0`，Agent 会为它生成独立 batch。入队参数是
`NaviSchedBatchRequest`：`totalTokenCount` 是输入 token 总数，`blockHashes` 是框架已计算的
完整 block hash。Agent 根据本机 Cache 自动计算连续前缀命中，框架不传预测命中量。

```cpp
agent.addBatchRequest(batchRequest, decision.batchId, decision.expectedBatchSize,
    [&executor](const navi_sched::NaviSchedBatchPtr &batch) {
        executor.enqueue(batch); // 只入队，不能在这里执行模型。
    });
```

关闭时先停止接收新请求，再从 callback 之外调用 `agent.stop()`；Agent 会等待已经开始的请求接收
和 batch 就绪 callback 返回，不能在 callback 内调用 `stop()`。

`NaviSchedBatch.timedOut=false` 表示按数量凑齐；`true` 表示达到 Agent 的
batch timeout 后释放部分 batch。迟到请求会得到新的本地 batch id，不能加入已经释放的 batch。
callback 可能由 Agent 常驻图触发，因此只能把任务交回框架执行器，不能在 callback 中执行模型或
同步等待。

Agent 在请求被接受、batch 就绪和超时释放时自动维护 inflight、batch waiting、queued batch 和
超时累计值，并向 Master 异步汇报；框架不再单独调用请求接收或队列负载接口。

### 6.2 串行执行并完成 batch

执行器开始一个 batch 前调用其 `finishReporter->startBatch(requestIds)`。Agent 保证同一节点只有一个
running batch；返回 `false` 时本 batch 不能开始。执行完成后，用相同 reporter 调用一次
`finishBatch(requestStats)`，数组必须完整覆盖本 batch 的请求：

```cpp
auto reporter = batch->finishReporter;
if (!reporter->startBatch(batch->requestIds)) {
    return;
}
auto finishedStats = executor.run(batch->requestIds);
reporter->finishBatch(finishedStats);
```

每项 `NaviSchedRequestStats` 包含 `requestId`、`totalTokenCount` 和执行器实际生效的
`cacheHitTokenCount`。该真实值用于延迟拟合，不复用入队时的预测值。

Agent 从 `startBatch()` 到 `finishBatch()` 自动计算整个 batch 的执行时延。一个物理 batch 只产生
一个拟合样本；累计 `latencyFitBatchSize` 个 batch 样本后拟合并向 Master 推送一次模型参数。
队列预测维护为待执行 batch 预测时延之和，batch 加入和移除时增量更新，不要求框架扫描或上报队列。
当前 running batch 会按已经执行的时间递减预测剩余时延，但最多扣除该 batch 自身的预测值，
不会把执行超时转化为后续 queued batch 的负时延。

如果错误分支忘记调用 `finishBatch()`，最后一个 reporter 引用析构时会自动异常结束 batch、释放
inflight 并累计异常结束次数。这是防泄漏兜底，不应替代正常完成调用。

### 6.3 Cache 上报

| API | 何时调用 | 说明 |
| --- | --- | --- |
| `reportCacheInserted(blockHashes)` | 完整 block 真正写入 KV cache 后 | 只传本次新增的 hash。 |
| `reportCacheRemoved(blockHashes)` | LRU 淘汰或显式释放完整 block 后 | 只传本次不再可复用的 hash。 |

不要把查询命中当作 Cache 插入。两个接口都是增量上报，容量只由 Agent 构造参数初始化一次。
Agent 内部自动维护完整 Cache 集合。Master 新建或重建 collector 时，Agent 先发送握手，再发送
一份全量 Cache 快照；连接建立后只发送增删量。

## 7. Python 角色最小骨架

Master 是正式 Python 发布入口；Client 和 Agent 也可用 Python 进行测试或 Router 集成。下面的
代码只展示角色职责，不包含框架的 tokenization 和真实执行：

```python
from navi_sched import NaviSchedAgent, NaviSchedClient, NaviSchedMaster

master = NaviSchedMaster(
    "master",
    max_wait_schedule_count=30,
    max_wait_schedule_time_ms=30,
    scheduler_name="random",
)
master.init()

agent = NaviSchedAgent(
    "prefill", server_ip="10.0.0.8", server_port=8000, name="agent")
agent.init()

client = NaviSchedClient(
    "client", "master", name="client")
client.init()

# 框架接入点：取得真实 token 数和同一 KV key 空间的 uint64 block hash 后调用。
decision = client.schedule(128, "prefill", block_hashes=[101, 202], timeout_ms=1000)
```

测试结束时按 `client.stop()`、`agent.stop()`、`master.stop()` 顺序关闭。生产进程应把这些调用
接到框架的正常 shutdown hook，而不是依赖 Python 析构时机。

## 8. RTP-LLM 与 vLLM 放置插桩的位置

### RTP-LLM

- Frontend 完成 tokenization 和 rolling hash 后调用 Client；拿到 `target.address` 后继续调用
  原有 `ModelRpcClient.enqueue`，不要让 NaviSched 代替 RTP RPC。
- 每个 Prefill `LocalRpcServer` 创建一个 Agent，并以它对外提供的模型 RPC `ip:port` 发布。
- 请求进入本地 batch 管理时调用 `addBatchRequest`；执行器通过 batch reporter 调用
  `startBatch`、`finishBatch`。
- KVCacheManager/SharedBlockCache 在 block 插入和淘汰后分别调用 Cache 增删接口。

### vLLM

- DashServing Router 已拿到上游生成的 `prompt_token_ids`，用与 vLLM 完全一致的
  `sha256_cbor` 链式算法生成完整 block hash，再调用 Client。拿到 `target.address` 后继续调用
  DashServing 原有 Prefill transport，不改变模型请求协议。
- 每个独立可选的 vLLM Prefill endpoint 创建一个 Agent。vLLM observer 在真实
  `execute_model` 前后驱动 batch reporter，不使用 NaviSched 调度结果中的 batch id 替代
  vLLM 自己形成的执行 batch。
- Agent 直接读取 vLLM `Request.block_hashes`、实际已计算 token 和 KV event，向 Master 报告
  真实 Cache 命中、完整 block 插入与淘汰，不在 Prefill 侧重新计算 hash。
- observer 在 vLLM 已经形成执行 batch 时接入，因此 Agent 的队列状态不包含仍停留在 vLLM
  Scheduler waiting queue 中的请求；评估预测队列时要单独考虑这段等待。
- 当前只支持纯文本、单候选请求。多模态、LoRA、`cache_salt` 和 `n != 1` 会在 Router 调度前
  拒绝，不能使用不完整的 hash 或重复 request id 静默运行。
- Router 与 Prefill 必须使用相同的 KV block size。当前专用部署默认固定为 16；vLLM backend
  如果将最终 block size 改成其他值，Agent 会在启动时拒绝不一致配置。

完整部署参数、hash 规则和调用顺序见 [DashServing 与 vLLM 适配](vllm_dashserving_adapter.md)。

## 9. 调试、replay 和错误处理

`collect_replay=True` 是策略开发调试开关。调用 `schedule()` 后，返回 decision 的 `replay`
字段包含当轮请求及其 block hash、节点快照、每请求每节点的前缀 Cache 命中数及决策结果，
但不保存完整 Cache hash 集合。离线 runner 根据命中数构造假节点和假 Cache，通过正式 Scheduler
接口复现策略输入。调度结果会先驱动 Client forward，replay 从独立输出随后返回，不影响请求
进入 Agent 凑批。
不要为所有在线请求打开它。

正常运行关闭 `enableVis`。需要检查图时才临时开启，并在图执行结果已读取到 EOF 后再读取 vis；
未读到 EOF 的图结果可能不完整。

常见错误与处理方式：

| 现象 | 原因 | 正确处理 |
| --- | --- | --- |
| Client 成功但 `target is None` | 当前无可用节点、策略限流或未调度 | 按 `reason` 走框架已有重试/拒绝逻辑，不发送空地址。 |
| Cache 命中异常低 | Client 和 Agent 使用了不同 hash、或 Agent 未及时上报 cache 增删 | 统一采用框架原始 `uint64` hash，并在插入、淘汰和释放后上报完整增量。 |
| Master 看到过高 inflight | batch 没有正常完成，且 reporter 被长期持有 | 每个 callback 返回的 batch 最终调用一次 reporter `finishBatch`，错误路径及时释放 reporter。 |
| 队列预测始终为 0 | batch 尚未就绪、执行太快，或没有有效延迟模型 | 检查 callback、reporter 生命周期和 Agent 模型 revision。 |
| Agent 常驻图后续全超时 | callback/forward handler 在 Navi worker 中阻塞或返回错误 | callback 只入队，所有耗时执行交给框架原有执行器。 |

接入完成后的最低验证应覆盖：Agent 上线、Client 选点、框架原 transport 执行成功、请求生命周期
闭合、Cache 新增/删除同步、队列预测上报、Agent 正常 stop 后 Master 下线，以及 Agent 重启后以
新的启动 meta 重新建立 collector。
