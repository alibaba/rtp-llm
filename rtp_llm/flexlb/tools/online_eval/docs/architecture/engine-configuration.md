# Real 与 Mock 的引擎配置

Mock 的行为配置消费 real 完成解析和推导后的值，不再维护另一套启动默认值。
它只支持能表达的行为；配置通过不代表真实 GPU 性能或完整状态机已经通过差分验证。

## 配置所有权

| 输入 | 内容 | 生效方式 |
|---|---|---|
| `--mock.engine_config` | `schema_version`、源码来源、按角色区分的 scheduler、model、KV 策略与 allocator 容量；V 使用真实 VitConfig 字段 | 启动时校验，修改后重启 |
| `--mock.calibration` | P 公式、固定耗时与 host load 延迟、D step 曲线与 token 增量、噪声、EOS 模型；V 的假特征宽度、保护限制和 forward 延迟 | 启动时加载；控制 API 可覆盖声明的校准项 |
| 进程参数 | 逻辑实例数、地址、端口、JVM、RPC/完成线程、显式 auto-fetch、监控 | 由运行环境负责 |

校准文件不能包含 block size、FIFO、缓存复用、reserve 或等待队列门禁。
`mock.forward.prefill.expression` 是 Mock 的执行公式；Master 的预测公式由 Master 配置独立负责。
旧 `--performance`、容量 CLI、`--master-config` 与 `MOCK_PERFORMANCE_CONFIG*` 不再接受。
`/set_perf` 接受嵌套的 `mock.timing`、`mock.rpc`、`mock.forward.prefill`（fixed_ms/jitter/pause）和 `mock.forward.decode`（scale/jitter/pause），以及 `engine`/`port` 定位字段；不能修改 real 行为门禁。
故障仍由显式故障接口配置。

## Mock 命名与时间模型

Mock 自有配置使用 `mock.<领域>.<字段>` 路径，JSON 使用嵌套对象；叶子中的单位和复合词用下划线，例如 `mock.timing.time_scale`。进程入口使用 `--mock.<字段>`，例如 `--mock.rpc_threads`。real 原生字段保留其下划线名字、单位和作用域，不加 Mock 别名。`engine`/`port` 是控制请求的目标选择器，request_id、cache keys 等协议数据不是配置项。Shell 的 `MOCK_*` 环境变量只是 launcher 的传入绑定，内部生成同一份配置，不建立第二套耗时模型。

```json
{
  "mock": {
    "timing": {"seed": 42, "time_scale": 0.1},
    "forward": {
      "prefill": {
        "jitter": {"stddev_ms": 2, "stddev_ratio": 0.02, "max_deviation_ms": 20},
        "pause": {"probability": 0.01, "duration_ms": 100}
      },
      "decode": {"jitter": {"stddev_ratio": 0.03, "max_deviation_ms": 5}},
      "vit": {"fixed_ms": 1}
    },
    "rpc": {
      "get_worker_status": {
        "request": {"delay_ms": 1},
        "response": {
          "delay_ms": 2,
          "jitter": {"stddev_ms": 1, "max_deviation_ms": 3},
          "pause": {"probability": 0.01, "duration_ms": 200}
        }
      }
    }
  }
}
```

默认 seed 为 0、time_scale 为 1；抖动、附加 RPC 延迟和停顿默认关闭。抖动标准差为 `hypot(stddev_ms, mean_ms * stddev_ratio)`，采样高斯残差并裁剪到 `±max_deviation_ms`；非零抖动必须配置正的裁剪边界。每个事件独立按 probability 决定是否添加 duration_ms 停顿，不代表持续慢节点、跨事件相关抖动或特定硬件故障分布。最终时间为 `max(0, mean + jitter + pause) * time_scale`，包括 RPC 延迟与 forward 停顿一起缩放；使用纳秒调度保留亚毫秒模拟时间。host_load_ms 也按相同全局比例缩放；real 的请求期限、TTL、重试与资源预算保持真实墙钟。

P 每个实际调度 batch、D 每次实际 forward step、V 每个实际 GPU batch 采样一次。RPC 每次调用的 request/response 两侧分别采样一个偏移，方法名支持 `get_worker_status`、`get_cache_status`、`enqueue_batch`、`cancel`、`fetch_response`、`generate_stream_call`、`remote_generate`、`check_health`、`remote_multimodal_embedding`、`get_multimodal_hashes`、`async_submit_embedding`、`wait_green_net_verdict`、`release_rdma_lease`；每条流中的消息保留原来的间隔和先后次序，完成排在消息之后，不逐帧重新抽取偏移。

seed 与 engine 身份、方法/阶段及该阶段的事件序号共同决定随机数流，其他 engine 或方法的流量不会消耗本流序号。相同配置、身份和事件序列可复现采样；并发请求与事件的对应顺序、操作系统调度和实际墙钟耗时不保证逐次相同。运行时通过 /set_perf 覆盖配置会重置该 engine 的采样序列；已调度事件保留当时的采样值。

request 延迟发生在 handler 收到请求之前，response 延迟发生在快照或输出产生之后；WorkerStatus/CacheStatus 的版本和数据保持产生时的值。对齐依据是 real `LocalRpcServer.cc` 的 GetWorkerStatus/GetCacheStatus 在 handler 内读取和构建快照，网络传输不重新生成快照或推进业务状态。延迟使用定时器，不在 RPC worker 中 sleep；调用在延迟期间仍占用 in-flight 额度。deadline、cancel、shutdown 清理延迟事件，request 延迟结束后不能再接纳已取消调用。已由 EnqueueBatch 接纳的逻辑请求仍按 real 的独立生命周期执行，传输取消不回滚成功入队。

`/inject` 使用 `{"engine":"prefill-0","mock":{"fault":{"type":"enqueue_error","enabled":true}}}`。`/prefill_formula` 使用 `mock.forward.prefill.expression`；`/decode_model` 使用 `mock.forward.decode` 的三个系数；`/output_length` 使用 `mock.forward.decode.eos`。旧顶层耗时字段、noise、jitter_pct、enqueue/generate delay 和旧 CLI 不接受。采集档案保留原始格式与校验值，由数据加载层显式投影为运行配置，不作为旧运行参数入口。

指标 `mock_timing_<阶段>_{base,jitter,pause,planned,elapsed,timer_lag}_us` 分别记录缩放后的分量、计划时间、实际等待及调度迟到；jitter 可为负。它们是 Mock 诊断指标。forward 停顿计入真实 forward/step 阶段，不归入队列等待。RPC 延迟不会伪造 engine 内部计算耗时。

## 默认值与推导

代码默认值以 real 正常启动后的生效值为准：先处理 CLI/env 与 server args 默认值，再应用模型及平台推导。
C++ 结构体初值可能被 server args 覆盖，例如 `prefill_retry_timeout_ms` 的 C++ 初值为 20，
但 server args 默认 0；Mock 的代码默认使用 0。没有覆盖时的 `concurrency_limit` 为 32，
ViT async concurrency 为 64，CPU embedding/Hash 字节预算分别为 2 GiB/256 MiB，图片上限为 100 MiB，KV reuse 默认关闭。
显式 engine-config 档案继续优先于代码默认值；synthetic 实验档案不因代码默认修正而改写。
GPU cache、模型长度、block size 和 allocator 容量仍从必填 engine-config 读取，不补造平台或模型观测值。

`max_context_batch_size` 与 `max_batch_tokens_size` 是 real 中的两个不同配置。
real 在 token 上限为 0 时执行 `max_batch_tokens_size = max_context_batch_size * max_seq_len`；
显式 token 上限非 0 时不使用这个推导。例如 context=4、序列长度=8192、token 上限=0，
最终 token 预算为 32768；显式指定 token 上限=16384 则最终仍为 16384。
Mock 只导入最终的同名 `max_batch_tokens_size`，没有把 context 参数换名，也不额外增加一个 context 请求数门禁。

## 获取 real 配置

配置采集由独立 Mock 工具完成，不向 real 引擎启动流程添加自动导出配置的环境变量。
Mock 工具 `rtp_llm/flexlb/tools/mock_bundle/real_config_snapshot.py` 提供 `snapshot` 和
`vit_snapshot`，可由独立采集脚本对已经完成解析、推导的 real 配置对象生成快照。
工具不会自行启动模型或取得生产进程中的对象，也不会自动记录 CLI/环境变量来源。
HTTP 图像下载的响应头与正文共享下载期限，正文停滞不能占用预处理线程直到连接自行结束。
ViT 导出必须显式提供观测到的 `work_budget`（可为 null）、`hash_cache_accounting` 和
`greennet_enabled` 布尔值；遗漏观测不能被解释为功能关闭。

allocator 容量仍需从真实运行环境独立观测并提供，不能从显存 MB 推算 block 数。
每个角色选择匹配 rank 的观测；异构 rank 需要分别建立 Mock 配置。
取得快照与容量文件后，用 Mock 工具合并：

```bash
python3 rtp_llm/flexlb/tools/mock_bundle/real_config_snapshot.py \
  --snapshot <Prefill 有效配置文件> \
  --snapshot <Decode 有效配置文件> \
  --snapshot <ViT 有效配置文件> \
  --capacity <Prefill 容量文件> \
  --capacity <Decode 容量文件> \
  --output <目标 engine-config.json>
```

快照必须来自相同源码版本；重复角色、重复容量或观测源码版本不符会失败。
没有容量观测的 P/D 快照不能启动 Mock。合成实验显式使用 `source.kind=synthetic`
和 `capacity.kind=synthetic`，说明实验来源并给出容量，不能标为 real。
测试场景只允许覆盖 synthetic 容量，不得覆盖真实 allocator 观测。

合成 Prefill 缓存实验可通过 `prefill_cache_policy.reuse_cache` 显式覆盖复用开关；省略时继承档案，不隐式开启，也不改变 Decode 或共享档案。要求缓存预热的实验必须保证复用已开启，并在 `model_override` 中说明偏离。

Master 遇到同 request/batch 的历史成功完成记录与活跃 CANCELING overlay 并存时，
保留该控制记录的所有者，等待权威 CANCELED 终态后结算。不同 batch 的 overlay 不延迟当前任务结算。

## 支持范围

P/D 使用 FIFO 的最终请求数、完整 token/padding 预算、未缓存 token 配额、
已初始化 KV stream 配额、CP 配置、角色 block size 和缓存策略。
`0` 的推导由 real 完成；Mock 不再解释未解析的 `max_batch_tokens_size=0`。

### 调度器与队列

real 的 `NormalEngine::initScheduler` 默认选择 `FIFOScheduler`。
`use_batch_decode_scheduler=true` 优先选择固定 batch 的 `BatchDecodeScheduler`；
否则 `pdfusion_scheduler_mode=ratio` 仅在 PDFUSION 角色选择 `PDFusionRatioScheduler`。
Mock 当前只接受显式导出的 FIFO，另外两种模式拒绝启动，不能当作 FIFO 运行。

FIFO 表示按到达顺序扫描候选，不表示严格队首阻塞，也不表示把所有入口混成一条队列。

| 角色/入口 | real 队列规则 | Mock 对应 |
|---|---|---|
| P `generate_stream` | 普通 waiting；逐项扫描，放不下的候选留在原位置，继续检查后续候选 | 普通 lane；执行时重查 prefix、KV 与 token 配额 |
| P `EnqueueBatch` | `enqueueMultiple → enqueueGroup`，进入独立 group waiting；每轮最多检查队首组，保留剩余组员的位置；不与普通 lane 或下一组混合 | `MockFifoPrefillQueue` 的 group lane；组残余留在队首 |
| P 组超过请求数/已初始化 stream 上限 | 退回普通 waiting 的队尾；token 超额不触发整组退回 | 同样按组大小判定，保留原 batch ID |
| P 两个 lane 都有工作 | 空执行边界在普通和 group 间交替；一个 lane 无可执行候选时尝试另一个；后续组不越过受阻的队首组 | 独立边界选择；每条 lane 内稳定扫描 |
| D ALLOCATE | 位于 scheduler 之外；临时 KV 压力按资源 generation 等待，受请求 deadline/cancel 约束；已分配 KV 在等待 P/Fetch 时仍持有 | 独立 reservation；不算 scheduler waiting/running |
| D scheduler | 无硬等待长度上限；执行一个 step 时新请求留在 waiting，下一 step 边界按 inclusive 请求上限补齐 | step 边界补齐；未初始化者检查已初始化 stream 上限，已有 KV 的候选可越过该门禁 |
| P Host KV loading | 成功初始化后进入 loading，普通 load 完成回 waiting 尾重新准入；group 内所有成员 ready 才占独立执行边界；pending I/O 不阻塞另一 lane | 独立异步 copy completion，保留 GPU lease、host read pin；普通与 group loading 分池，copy 失败退回重算 |
| P/D 普通取消与超时 | Error 先唤醒 consumer；scheduler 访问该 stream 时提交 FINISHED；错误优先于缓冲输出 | consumer 可先收到 gRPC 错误；资源和 finished-task 留到调度边界，后续 group 不提前提交 FINISHED；保留第一个错误原因 |
| P/D 运行中取消 | 先记录取消意图；已执行 forward 仍拥有其 KV，边界后才终结并释放 | 保留运行槽、KV 和任务状态到执行边界；下游取消期间 P 回报 CANCELING，释放后回报 CANCELED |
| V URL RPC / 异步接单 | `vit_concurrency + vit_max_queue_size`；按整批新 miss 原子预留，同 key 共享；最后一个 owner 只能取消尚未运行的 async task | 独立 async admission，与 GPU waiting 队列分开 |
| V preprocess | local 模式在 async worker 同步预处理；worker 模式按 `mm_preprocess_max_workers` 执行，内部待处理队列无独立上限；pending 包含运行与等待 | 下载、解码、几何计算先于 GPU；JVM worker 模拟进程工作池 |
| V GPU waiting | `mm_max_queue_size` 限等待项；取消项仍计入物理队列，直到消费者取走；先跳过取消项再收集 batch | 有界 GPU waiting；已 claim RUNNING 的 future 不能取消 |
| V GPU 收集/完成 | 窗口为 0 时不收集额外请求；一个 forward 失败则整批失败；stop 让已运行 forward 完成，不启动下一批 | 相同窗口、取消 claim 和整批失败规则；async 队列 deadline 在运行前检查，GPU 等待使用剩余预算 |

普通 P lane 使用 `max_batch_tokens_without_cache` 作为下一候选的停止配额；
real 的 group admission 不使用该配额。完整逻辑 token 和 padding 矩形预算在两条 lane 都生效。
prefill-only 可填满完整 token 上限，generation 保留一个 token；两种执行模式不能混合。

`EnqueueBatch` 在单 DP 下先检查整个 envelope 的重复 request ID，再按本地 rank 过滤，
把所有有效 `dp_slots` 按原顺序展平成一个 group。缺失 input 和 rank 错误逐请求返回；
整个 batch 出现重复 ID 时整批拒绝。batch ID 覆盖各请求的 `group_id`，`group_size` 使用完整组的 input 数量。

所有 context 先注册 active，再分别提交准备任务，最后统一等待结果。prepare 失败逐请求终结，
不等待其他成员才能提交自己的失败状态；其余成员继续。准备成功后先存储 context，在 R1 检查取消，
再模拟 `NormalEngine::enqueueMultiple`：创建 stream 的错误先剔除，随后检查剩余 stream 是否混合执行模式，
最后逐请求检查长度与 KV 最大可容纳长度，仅有效成员组成一次 scheduler group。
执行模式按 `max_new_tokens` 判断，不使用被最大序列长度截断后的模拟输出长度。
P→D ALLOCATE 清除 `group_id`、`group_size`、`group_timeout`，D 不继承 P 的 group 调度边界。

每个 context 独立仲裁 terminal cause。prepare 或 group 操作未退出时，priority Cancel 先发布 CANCELING，
操作退出后才能完成清理；其他成员的 prepare 不延长该请求自己的操作所有权。
R1 前获胜的取消不进入 engine；R1 后的取消在 scheduler 返回后的 R2 和 ACK 发布处再次检查。
非取消错误已经终结时，迟到的 Cancel 返回 NOT_FOUND，不改写原错误；取消先获胜时保留 8429。
scheduler 已接单的请求只在调度边界释放运行资源。

ACK 按请求发布，每个输入恰有一个 success 或 error；成功仅表示入队。
发布前请求期限耗尽返回 DEADLINE_EXCEEDED，不能返回成功；success 已发布后到达的 Cancel 不改写该 ACK。
停止接单时逐请求返回 UNAVAILABLE，正常停止不返回空 ACK；关闭准备池时显式结算移出等待队列的任务，避免 batch join 永久挂起。
batch RPC 的传输 Context 不拥有已接受的逻辑请求；逻辑请求继续受自身 timeout、priority Cancel 和 Fetch lease 约束。
显式 ACK 丢弃、黑洞和 crash 故障注入属于 Mock harness，不采用正常 ACK 的完整性保证。

队列审查依据为 [FIFOScheduler.cc](../../../../../cpp/engine_base/schedulers/FIFOScheduler.cc)、
[FIFOSchedulerBase.cc](../../../../../cpp/engine_base/schedulers/FIFOSchedulerBase.cc)、
[GenerateStateMachine.cc](../../../../../cpp/engine_base/stream/GenerateStateMachine.cc)、
[PrefillBatchRpcServer.cc](../../../../../cpp/model_rpc/PrefillBatchRpcServer.cc)、
[DecodeRpcServer.cc](../../../../../cpp/model_rpc/DecodeRpcServer.cc)、
[RpcServerRuntimeMeta.h](../../../../../cpp/model_rpc/RpcServerRuntimeMeta.h)、
[mm_process_engine.py](../../../../../multimodal/mm_process_engine.py) 和
[mm_scheduler.py](../../../../../multimodal/mm_scheduler.py)。

real 的 gRPC `LocalRpcServer::GetWorkerStatus` 不填旧的 `available_concurrency`、
`waiting_query_len` 和 `running_query_len`，这些字段保持 protobuf 默认值 0。
Mock 同样回报 0。调度状态以任务列表的 `phase`、`is_waiting` 和取消进度为准；
Mock 的 `/snapshot` 与指标仍提供独立诊断队列深度，不能当作 real 填充的字段。
WorkerStatus 的活动表是 RPC runtime inventory，包含 ALLOCATE 等待、已分配 KV 和执行中的 stream，
不等于 scheduler running 队列。D 在 ALLOCATE 开始就注册 RECEIVED，分配后为 KV_ALLOCATED，
GENERATE 入队并被 scheduler 选中后才为 RUNNING；ALLOCATE 失败也产生 finished delta。
P 计算完成释放执行槽，runtime 仍保留到 Fetch 推进且 D LOAD ACK 到达，届时先释放 connector KV，再发布成功完成。
P 的成功 stream 指标也在这个 runtime dequeue 点上报，首 token 时间仍取计算产生时间。
real 的 finished TaskInfo 保留初始 PENDING phase；Mock 同样回报 PENDING/is_waiting=true，不能用该字段反推结束前的执行阶段。
D 不继承 P 的 group/batch identity，PD 首 token 在 D 入队前已存在；普通自回归剩余 N-1 次 forward，
直接 D 输入从 0 token 开始。KV 增长在下一次 forward 开始前预留，失败时不执行该 forward；
MTP 仍用校准的平均 token 增量，不模拟真实接受长度和异步 device 上界。
下游优先级取消保留 P Fetch 的 8429，同时 D 的本地普通取消记录为 8100。

P 的历史执行完成与同 batch 的 `CANCELING` 控制 overlay 可同时出现；Master 保留该请求归属，直到 `CANCELED` 确认。异 batch 的 overlay 不会延迟完成结算。
`/snapshot.rpc_runtime` 的 callback worker 和 in-flight 调用额度属于 Mock harness，
不能当作 real `RtpLLMOp` 的 gRPC `MAX_POLLERS` 或 engine waiting 容量。
RPC 线程池过载与 engine 调度队列的完全等价仍需单独验证。

物理池只支持单个 FULL group、相同的逻辑与物理 block 粒度及 LRU 驱逐。
多 group、不同 block 粒度、MTP 子池和不支持的 scheduler 不会被压成一个总量。
real 的容量导出对这些布局标记 `unsupported_topology`，Mock 明确拒绝。
单池仍只模拟元数据、引用和容量，不模拟真实 KV tensor、DMA、disk 或远端 KV 读取。
Decode memory cache、disk 和远端 KV cache 配置不支持。
Host KV copy 使用独立完成事件，按 `LOADING_CACHE → WAITING → RUNNING` 推进。
`prefill.host_load_ms` 是校准文件中的复制耗时，不是行为门禁；默认 0 仍经过独立事件和调度轮次。
loading 中的请求持有完整 GPU 容量及 host pin；复制成功后才能消费 host 条目，取消时释放 pin，
普通复制失败退回重算；materialization RETRYABLE 释放本轮资源后重新初始化，PERMANENT/INTERNAL 发布 602，GPU 资源由 scheduler 提交 FINISHED 后回收。

Host 搬运分为 business、LOAD transfer、background transfer 三个池。business 默认 4 workers，
配置的 business queue 额外保留 64 个 LOAD 槽；completion 优先，最多连续处理 4 个 LOAD 后处理一个等待的 background。
transfer 的两个方向池各使用 `block_tree_transfer_worker_count`，队列 0 表示无界。
三个池都沿用 `BlockTreeTaskPool` 的 BACKGROUND 子队列门禁：非零容量 Q 大于 64 时，BACKGROUND 最多等待 Q−64 项，否则最多 Q 项。
business 总容量是配置值加 64，因此其 BACKGROUND 上限等于配置值；transfer 总容量直接使用配置值，其 background 子队列也会受到上述减 64 规则约束。
`memory_cache_max_descriptors_per_transfer_batch` 限制每批 descriptor，`memory_cache_sync_timeout_ms` 覆盖排队和搬运。
可选 store 失败不使生成失败；源 GPU 引用和目标 host reservation 保留到搬运结算。
owner stop 先拒绝新业务，再排空业务 body、两个方向的 transfer、回到 business 的 settlement，最后关闭三个池。
原始 pool close 丢弃尚未运行的队列项，不触发伪 timeout，等待已运行任务结束，不 interrupt worker。
水位达到 `ceil(total * high)` 后持续回收到 `floor(total * low)`；pinned 条目不可驱逐。
GPU→host 先持有两端资源，成功后才释放源；pending release 不重复计入水位需求。
这里只模拟单 rank、单 FULL pool 的元数据与事件，真实 DMA、CRC、跨 rank settlement 和带宽不在验证范围。

P 准备重试依次包围连接准备、multimodalProcess 和 D ALLOCATE，最多执行 `max(0, prefill_retry_times)+1` 次。
`prefill_retry_timeout_ms<=0` 关闭重试期限；首次尝试允许开始，后续尝试检查剩余重试预算。
channel readiness 同时受 request/retry/max RPC 预算及 15 秒硬上限限制；已开始的 ALLOCATE RPC 只受 request/max RPC 预算限制。
准备池大小使用 real 解析后的 `max(128, configured>0 ? configured : max(1, concurrency_limit)*2)`，其等待队列同大小。
同组 slot 的 MM 与 ALLOCATE 均在准备池内并行，完成后汇合；失败 slot 保留具体错误并释放该 attempt 的资源。
MM 成功后跨 D ALLOCATE 重试复用；仅 903/904/906/907 可重试。transport 的 UNAVAILABLE/deadline 转 907，其他未分类错误不重试。
`prefill_max_wait_timeout_ms` 从 Fetch/自动 continuation 等待 WAITING 开始，进入 LOADING/RUNNING 后结束；不重启这段计时。
LOAD 采用 request/max RPC/load cache 三者最小预算，load cache 配置非正时回落 5000 ms；阶段超时为 8300；等待运行超时为 8213。
P 到 D 的传输失败在缺少下游领域详情时按观察阶段区分 ALLOCATE（8207）、LOAD（8208）、GENERATE（8209）；已有领域详情优先保留。
Fetch 失败通过带领域详情的 gRPC 终态返回，不作为普通成功数据帧交付。

D 当前 RemoteGenerate 不调用 `decode_retry_times/timeout_ms`，这两个字段只读回，不增加 ALLOCATE/LOAD/GENERATE 重执行。

V 分别使用 `vit_concurrency + vit_max_queue_size` 的异步接单额度，
`mm_max_queue_size` 的 forward 等待队列，以及 batch 请求数、图片数和收集窗口。
普通 URL 的 `RemoteMultimodalEmbedding` 实际调用 real 的 `get_embedding_result`，与 async submit/hash 共享按 key 的异步接单；
新增多图工作先认领 entry，再整体预留异步额度；准入失败时终结本次新增 entry，撤销本次请求所有权。每个 URL 各产生一个 MMWorkItem；模型 `preprocess_batch_size` 不合并这些独立的 async job。
V4.1 typed RPC 对应 `submit_v41`，绕过 async C+Q 和 URL preprocess，完整验证请求及总图片上限后，所有图片作为一次 scheduler 请求接单。
同请求多图不拆成多个 GPU 入队请求；即使编码缓存全部命中，也先经过图片数和 waiting 队列门禁。
图片超限返回 INVALID_ARGUMENT/902；GPU waiting 队列满返回 RESOURCE_EXHAUSTED/903；URL async admission 满返回 RESOURCE_EXHAUSTED/409。
MM 错误详情保留到 P，903 可重试，902/409 不重试。
异步任务按 `request_id` 集合持有 entry，同一 ID 的重复认领不增加 owner；取消只移除该请求的所有权，
无其他 owner 的排队任务可取消，已运行任务保留到执行结束。提交部分失败只清理本次 claims，不能取消同 ID 的较早工作。
GPU/CPU embedding 预算都为零时，real 不保留 pending entry，同 key 也不合并计算；Hash 预算不改变这一规则。
`AsyncSubmitEmbedding` 的 ACK 表示提交成功，不取决于后台是否已完成；其后台期限取本批正媒体期限的最大值，
没有正值时使用 `mm_timeout_ms`，不继承 ACK RPC 的等待期限。
Hash RPC 先验证所有 missing key 的 URL 输入，再按一批接单；响应保留请求 key 的顺序和重复项，
总 hash 行数上限为 1048576。首次补算响应不带 generation，后续缓存探测返回驻留 Hash 的 generation。
错误保留 `ErrorDetailsPB` 和 `x-rtp-vit-error-reported`，后台失败按结果上报一次，调用方和代理不重复计数。
`py_rtp_vit_qps/success_qps` 在实际计算入口/成功边界上报，缓存命中和 typed 路径不增加这两个指标；
`mock_vit_rpc_*` 单独记录 RPC 流量。缓存 hit/miss/inflight 在认领时上报，eviction 在驱逐时上报，
resident tokens/bytes 在完成、回迁、移除和清理后上报，不用定时 snapshot 替代 real 的事件指标。
typed 编码缓存位于 model forward 内，使用独立 item-count LRU，读取 real 同名环境变量 `V41_IMAGE_ENCODE_CACHE_ITEMS`（默认 16，非正关闭）；
可信 content hash、processor identity 和网格共同作 key，空 hash 绕过缓存。它不占通用 URL embedding/hash 的字节容量。
GPU、CPU embedding 按实际输出 tensor 的元素数、dtype、目标设备及 tensor 对象身份去重计费；
同设备的结果按元素对齐打包，每个池一次申请连续区间。Hash 是独立 CPU sidecar，不重复计入 embedding 池。
三个池均为固定容量的分配元数据，按 8 字节对齐 first-fit，合并相邻空闲区间，不搬动活跃区间做压实。
GPU 淘汰下放 CPU；CPU 容量不足先淘汰冷项，必要时淘汰含 CPU tensor 的 GPU 热项；
Embedding 总字节足够但连续区间不足时继续驱逐，分配失败重试一次后旁路缓存，不把成功 forward 改成失败。
Hash 按总计费先驱逐，池分配仍失败则直接旁路该项。
单项超过 GPU 容量尝试存 CPU，连 CPU 也放不下则不保留。CPU 读取恢复失败是请求错误；
恢复成功后的缓存提升失败只放弃驻留，已恢复的请求结果仍可用。

`hash_cache_accounting` 由 Mock 测试工具 `vit_cache_reference.py` 只读测量 real 运行时的 Python 字符串、列表、tensor wrapper、
记录及字典项的计费布局，Mock 必须显式导入，不能使用 JVM 对象大小或固定常数猜测。
Hash 查询元数据不刷新 LRU；消费且 generation 一致才刷新。超大替换删除自身旧 generation，不冲掉无关记录。
URL 原始字节 LRU 独立于 embedding 字节预算。local preprocess 使用当前进程的 `url_cache_item_num`；
spawn preprocess 各子进程独立导入默认 10 项缓存，父进程的 resize 不传播到子进程，Mock 保留该实际作用域。
预处理 IPC 提交损坏时重建池并重试一次；结果读取遇到损坏、或连续超时达到 worker 数时重建，清空旧 worker 的 URL 缓存和 pending 统计。
预处理耗时只在结果被成功消费时上报，迟到完成不补报。Mock 使用逻辑线程 worker，不模拟操作系统进程终止机制。
缓存命中仍校验当前文件大小限制。`mm_cache_item_num` 保留 real 的模型内部旧缓存配置，
当前通用 URL scheduler 不使用它；V4.1 编码缓存由上面的独立环境变量控制。

GPU/CPU pool 不申请真实显存，payload 仍为不可变 JVM 数据。回收依赖完成事件，事件未完成的块计入
used/pending，不能重新分配；默认 CPU 模拟事件立即完成，测试通过受控事件覆盖延迟回收和搬运失败。
这些事件用于验证所有权和容量，不拟合 CUDA/DMA 延迟、forward workspace 或真实 OOM。
模拟输出的 shape/dtype 由现有模型/校准输入决定，比较容量时必须先匹配输出 tensor 布局。
只支持 inline gRPC，不接受 RDMA 和远端 embedding cache。
`greennet_enabled` 导出 real provider 的最终启用状态；Mock 必须显式接受 `false`，启用检查时拒绝配置，
不能把未实现的检查当作通过。该字段是运行时能力投影，不是新的 VitConfig 调参项。

V 的 `work_budget` 来自初始化完成后的 real `MMScheduler`，必须显式为 `null` 或五维预算对象。
五维为 `input_patches`、`output_tokens`、`estimated_workspace_bytes`、
`max_attention_segment`、`attention_work`；patches、tokens、workspace、attention_work 按总和累加，segment 按最大值聚合，0 表示不限制。
通用 scheduler 按稳定顺序切 chunk；不可拆的单项 cost 超限单独执行，图片数超限拒绝。
batch 收集遇到放不下的 chunk，保留为下一批首项并停止扫描。
只对第一块应用新请求队列门禁；续块排回队尾并绕过该门禁，防止单 worker 自锁。
超时保留已运行 forward，但停止后续 chunk；forward 出错则整批请求失败。

当前源码的具体模型均继承接口默认 `get_batch_work_budget() = None`，没有开启模型专属估算。
Mock 的图片服务据此执行 `null` 预算；若导入非空预算但没有匹配的模型 estimator，明确拒绝启动。
五维预算、分块和续排的通用行为有单独回归，不能把测试用估算当作真实模型容量公式。

监控遵循实际拥有者：P/D waiting、loading、running、admitted context、oldest age 在 scheduler round 采样；
周期 scrape 不重复插入 scheduler 样本。group fallback 按 real counter 注册，只有发生时上报。
D ALLOCATE 预留不进入 scheduler waiting/loading/running。
V queue size 包含 waiting 与 pending carry-over，入口容量只检查 waiting；batch size 在取消 claim 后采样。
queue wait、forward、embedding request RT 的单位均为毫秒，embedding RT 仅报告成功且未超时请求。


## 回读和验证

P/D 控制口与 V HTTP 口的 `GET /effective_config` 返回导入配置和模拟边界。
P/D 同时返回初始校准、当前公式/step/EOS 覆盖与 generation。
校准更新完成后应回读该接口；generation 变化不代表引擎行为配置发生热更新。

先验证同一组 CLI/env 输入经过 real parser、默认值补齐和 finalize 得到的结果，
再验证 Mock 的准入、容量不足、等待、取消与释放边界。
纯 Mock 测试通过只能证明模拟器合同，真实差分仍需要目标 real 引擎的执行证据。

## 配置差异与支持范围

此处比较的是最终配置的行为，不能用同名字段、JSON 可读取或 UT 数量代替行为等价。
real 的字段来自 `cpp/config/ConfigModules.h`、`config/py_config_modules.py` 和对应 server args；
Mock 的公共入口是 `MockEngineConfig`、`MockVitServer` 和 `JavaMockEngineCluster.Config`。
调度只有 FIFO 一条实现；测试向量也使用 real 的 scheduler 参数名。旧 direct/coalesce/regroup 模式、并发 lane 和可选队列状态模式已删除。

### 双方都有的配置

| real 最终字段 | Mock 接收位置 | 行为契约 |
|---|---|---|
| `RuntimeConfig.max_generate_batch_size` | `scheduler.max_generate_batch_size` | 每轮 stream 准入额度；不是 RPC 并发或等待长度 |
| `FIFOSchedulerConfig.max_batch_tokens_size` | 同名 scheduler 字段 | 完整 token 与 padding 矩形预算；保留 real singleton fast path |
| `max_batch_tokens_without_cache` | 同名 scheduler 字段 | 普通 lane 的下一候选停止配额；不是候选累计不能超过的硬 cap；group 不应用 |
| `max_inited_kv_cache_streams` | 同名 scheduler 字段 | 全部 scheduler 已初始化 KV 的 stream 数；已初始化候选不重复占用 |
| `cp_force_single_prefill` | 同名 scheduler 字段 | CP 启用时限制 context admission；`cp_size` 从 real TP/CP 最终值推导 |
| `ModelConfig.max_seq_len` | `model.max_seq_len` | singleton 判断及 speculative reserve 长度校验；不能直接把所有超限归为 LACK_MEM |
| `KVCacheConfig.seq_size_per_block` | 同名 KV 字段 | 逻辑/物理同粒度的单 FULL pool；与 allocator 观测交叉校验 |
| `reuse_cache`、`enable_device_cache`、`enable_memory_cache` | 同名 KV 字段 | 缓存策略与驻留层开关；host hit 仍需 GPU 容量 |
| `reserve_block_ratio` | 同名 KV 字段及 allocator `reserve_blocks` | real 为百分比；最终块数用 allocator 观测，不以配置预算冒充实际容量 |
| `device_eviction_policy`、`memory_eviction_policy` | 同名 KV 字段 | 只支持 LRU；其他值明确拒绝，不默默按 LRU 执行 |
| `PDSepConfig.decode_retry_interval_ms` | `pd_sep.decode_retry_interval_ms` | ALLOCATE 取消检查周期 `max(value, 50)`；仅资源 generation 变化后重新分配 |
| `VitConfig.vit_concurrency`、`vit_max_queue_size` | 同名 V 字段 | async admitted 总额为两者之和，同 key 共享；不等于 GPU queue cap |
| `gpu_max_batch_size`、`gpu_max_batch_images`、`gpu_batch_wait_ms` | 同名 V 字段 | 使用 real resolved 值；serial 要求 wait=0、images=sys.maxsize |
| `mm_max_queue_size` | 同名 V 字段 | 新请求只看 waiting；pending carry-over 计入监控，续块绕过 newcomer 门禁 |
| `mm_timeout_ms` | 同名 V 字段 | embedding 阶段预算；媒体级正值覆盖默认，仍受原请求剩余期限约束，不替代 RPC deadline |
| `mm_image_max_file_size_kb` | 同名 V 字段 | 文件限制为 KB×1024；不再从性能校准文件提供输入文件容量 |
| `mm_cache_gpu_max_bytes`、`mm_cache_cpu_max_bytes`、`mm_hash_key_cache_max_bytes` | 同名 V 字段 | 三个独立预算；固定逻辑池、连续空间及 pending 回收，GPU 淘汰下放 CPU；不模拟物理分配/搬运耗时 |
| Mock 工具对 real Hash cache 运行时的只读测量 | `hash_cache_accounting` | Python 元数据布局，必填；不要求修改 real cache 类 |
| `url_cache_item_num`、`mm_cache_item_num` | 同名 V 字段 | URL 原始数据按条数 LRU，local/spawn 作用域见上文；旧 embedding 条数配置保留给模型内部 |
| real provider `is_enabled()` | `greennet_enabled` | 最终能力状态，必填；检查开启时 Mock 拒绝启动 |
| `get_batch_work_budget()` 初始化结果 | `vit.work_budget` | 不是 VitConfig 原生字段；从实际 scheduler 导出的派生值。当前具体模型均为 None |

### 异名或旧名容易导致的误解

| Mock 字段/旧名 | real 对应 | 处理 |
|---|---|---|
| 旧 `prefill.fifo.max_requests` | `max_generate_batch_size` | 已删除；配置和测试向量均使用 real 名字 |
| 旧 `max_inited_kv_streams` | `max_inited_kv_cache_streams` | 已删除；使用 real 名字 |
| 旧 `direct_batch_size_max` / `max_batch_requests` / `max_waiting_batches` | 无对应 real 配置 | 已删除；使用 real scheduler 预算，故障限流使用显式注入 |
| 旧 `report_queued_as_kv_allocated` | D 实际资源阶段 | 已删除；普通等待为 RECEIVED，PD 预分配为 KV_ALLOCATED |
| 旧 `block_size` | `seq_size_per_block` | 从 engine-config 导入；校准文件不能修改 |
| 旧 `prefill_concurrency` / `decode_concurrency` | 容易混淆 RPC 与 scheduler 容量 | 不再作为公共配置；P forward 串行调度，D batch 容量来自 scheduler |
| `capacity.device_blocks` / `memory_blocks` | allocator 最终可用 pool 块数 | 观测值，不对应 `kv_cache_mem_mb` / `memory_cache_size_mb` 的直接别名 |
| `capacity.reserve_step` | NormalEngine 计算后的 speculative reserve step | 不是每 step 增加的 token 数；prefill-only 不保留 speculative 空间 |
| `disable_decode_first_malloc_device_reuse` / prefix-tree flags | allocator 实际运行策略 | 作为容量观测附带的行为，不能仅用 `reuse_cache` 代替 |
| `mock.fetch_attach_timeout_ms` | request `fetch_attach_timeout_ms`，缺省 600000 ms | Mock 默认覆盖项；不是 `prefill_max_wait_timeout_ms` 或请求总超时 |
| 旧 V `max_input_bytes` | `mm_image_max_file_size_kb` | 公共输入文件限制改用 real 字段和单位；旧校准字段拒绝 |
| V `mm_transport_mode` | `output_transport.mode` | 扁平投影；目前只接受 grpc，RDMA 明确拒绝 |

### Mock 有、real 没有同等配置

以下参数不能包装成 real 容量或 real 性能设置：

| 参数组 | 所属范围 |
|---|---|
| `mock.timing`、`mock.forward`、`mock.rpc` | 模拟计算耗时或输出长度；real 模型实际计算、采样和 MTP 接受率替代这些参数 |
| `mock.vit.hidden_size`、`mock.forward.vit.fixed_ms`、`mock.vit.max_output_bytes` | 假特征宽度、耗时和 JVM 输出保护；不是 real 模型隐藏维度及输出容量的完整等价 |
| `mock.auto_fetch`、逻辑 P/D 实例数、mock.unique_engine_ips、端口布局、发现文件 | 测试部署方式及协议驱动 |
| `rpc_threads`、`rpc_max_calls`、completion/event-loop 线程 | JVM harness 容量；不是 C++ MAX_POLLERS、prepare pool 或 engine queue |
| `queue_depth_limit`、status/ACK 故障注入 | 显式 Mock 故障实验；独立于 scheduler 的队列与预算模型 |
| `source`、`provenance`、`capacity.kind` / `layout` | 配置来源与支持布局的元数据，不是推理算法开关 |

### PD、host 与预处理配置

| real 配置 | Mock 运行语义 |
|---|---|
| `prefill_retry_times`、`prefill_retry_timeout_ms` | 准备阶段次数与预算；不重放计算、LOAD 或输出 |
| `prefill_max_wait_timeout_ms` | WAITING 独立阶段期限，8204；与 Fetch TTL 和总 request deadline 分开 |
| `decode_retry_times`、`decode_retry_timeout_ms` | 当前 real RemoteGenerate 不使用；导入并读回，不赋予额外行为 |
| `load_cache_timeout_ms`、`max_rpc_timeout_ms` | channel/RPC/LOAD 的独立预算；LOAD 失败取消其 D reservation |
| `prefill_prepare_resource_pool_size` | 使用 real 解析结果，worker 数与 waiting queue 容量相同；slot 并行准备 |
| `memory_cache_sync_timeout_ms`、`memory_cache_max_descriptors_per_transfer_batch`、`block_tree_transfer_worker_count`、`block_tree_business_queue_max_size`、`block_tree_transfer_queue_max_size` | business 与两个方向 transfer 队列、descriptor 分批和排队/搬运期限 |
| `block_tree_device_evict_low/high_watermark_ratio`、`block_tree_memory_evict_low/high_watermark_ratio` | 单池水位与异步 GPU→host 所有权；0/0 关闭 |
| V `mm_preprocess_max_workers`、`use_local_preprocess`、`download_headers` | 独立 worker 或同步预处理；HTTP JSON headers；完成后才能进入 GPU 队列 |

### real 有、Mock 尚未等价实现

这些字段不能因为没有出现在导出 JSON 中就视为无效或已对齐。
下表是明确的支持边界；涉及这些机制的实验不能用当前 Mock 证明 real 等价。

| real 配置 | Mock 状态及原因 |
|---|---|
| `max_context_batch_size` | 已由 real finalize 为 `max_batch_tokens_size`；属于被折叠的输入，不是缺失运行门禁 |
| `pdfusion_scheduler_mode`、`decode_prefill_ratio`、`use_batch_decode_scheduler`、`batch_decode_scheduler_*` | PDFUSION/ratio/batch-decode 不支持，不能按 FIFO 替代 |
| `decode_polling_kv_cache_step_ms`、`decode_polling_call_prefill_ms` | KV transport 轮询不逐次模拟；stage budget 不证明 RDMA poll 行为 |
| `prefill_stop_stream_wait_timeout_ms`、`rdma_connect_retry_times` | stop wait 数值导入；没有真实 cache-store entry/transport teardown，因此不模拟其物理等待。GPU 所有权始终由 scheduler/搬运结算释放，不由此期限强制释放 |
| `CacheStoreConfig` 的 RDMA/QP/thread/transfer deadline、steal/return deadline、TRANSFER_NOT_DONE hold、resource store 检查间隔 | 不模拟真实 transport 和 worker 持有 GPU 的安全窗口 |
| `kv_cache_mem_mb`、`memory_cache_size_mb`、`test_block_num`、`fp8_kv_cache`、`ssm_state_dtype`、`kernel_seq_size_per_block`、`max_block_size_per_item`、`use_block_cache` | 部分决定 real allocator 布局/块数；Mock 消费最终观测，不重新执行显存规划、dtype 和 kernel 分配 |
| disk watermark、多 rank/group watermark | 当前只支持 device/memory 单池水位；磁盘与跨 rank 驱逐未模拟 |
| `enable_disk_cache`、`disk_cache_*`、`enable_remote_cache`、`kvcm_*` | disk/remote KV 不支持，开关启用时拒绝；真实数据面/控制面不由本地 LRU 模拟 |
| `dsv4_fixed_pool_blocks`、`dsv4_hca_state_pool_blocks`、`dsv4_fixed_pool_use_memory`、`linear_step`、多 group/MTP 子池 | 不支持的容量布局通过 allocator observation 标记并拒绝，不压成总块数 |
| `enable_crc`、`kv_cache_event_*`、`block_tree_full_prefix_scan_interval_ms` | KV 内容校验、发布与一致性扫描未模拟；Mock metadata keys 不证明数据正确 |
| `multi_task_prompt*` | real 系统前缀展开未等价；输入长度应使用 real 展开后的形状，不能假设原始 token 长度相同 |
| V `mm_video_max_file_size_kb`、进程崩溃重建 | 不支持视频；JVM worker 不模拟 Python multiprocessing 的进程死亡/pipe 重建 |
| V `mm_cache_item_num`、`url_cache_item_num`、`use_igraph_cache`、`igraph_*`、`default_key` | 模型内缓存、URL 缓存和 igraph 不等价，不能拿 embedding 字节缓存代替 |
| V `mm_remote_cache_*`、`output_transport.control/rdma.*` | 远端 embedding cache 和 RDMA 不支持；相应开关拒绝 |
| V `vit_trt`、`trt_cache_*`、`biencoder_preprocess`、`extra_input_in_mm_embedding`、`extra_data_path`、`local_extra_data_path` | 模型实现及预处理分支未模拟 |
| V `vit_separation`、`vit_proxy_load_balance_strategy` | 拓扑/代理调度属于部署与 proxy；当前 V endpoint 不模拟整套 real proxy |
| V `disable_access_log`，Runtime `warm_up*`、`model_warm_up`、`output_dispatcher_worker_count`、`reserve_runtime_mem_mb`、`specify_gpu_arch`、`model_name`、`worker_*addrs` | real 日志、启动、运行时及硬件配置，不是 Mock scheduler 参数 |
| GrammarConfig、logits/sampling、tokenizer、输出词表、adapter、模型错误与 CUDA OOM/async bookkeeping | 不执行模型，不能通过 scheduler UT 声称对齐这些请求错误及清理机制 |

## 请求错误与终态契约

| real 分支 | 当前 Mock 行为 | 验证边界 |
|---|---|---|
| QueryConverter wire 校验 | 负数 token、显式 prefill-only 非零 token、prefill-only + prompt logits 返回 605 | 不模拟完整模型参数/词表校验 |
| bare zero / prompt scoring | 未标记的零输出仍为 prefill-only；prompt logits 归一化为一个 token，禁用 reuse 与 PD | 只对齐调度配置归一化，不生成真实 logits |
| GenerateConfig prefill-only 校验 | 禁止 generation-only 输出、beam/多返回及正 min_new_tokens；D 拒绝 prefill-only | 多序列正常生成整体不支持 |
| mixed explicit group | 按归一化执行模式拒绝混合 group，返回 domain 605 | 同批错误与成功成员仍保留独立请求身份 |
| speculative reserve 长度 | `input + reserve_step > max_seq_len` 返回 511；prefill-only reserve=0；等号允许 | 不以 `input>=max_seq_len` 粗暴替代 |
| 输入超过整个 FULL pool token 上限 | 返回 607 | 与 reserve/可用块不足的 602 分开 |
| P 初始化临时 KV 暂缺 | 保留 waiting，继续扫描；失败候选不锁定本轮执行模式 | 不模拟 allocator internal corruption |
| P 初始化永久不足 | scheduler 提交 602，清理 P/D 资源；不一直挂在 waiting，也不伪造普通 cancel |
| D ALLOCATE | 临时暂缺按 generation 等待；永久 602；deadline 603；取消 8100 | D 真实 host/disk load 不支持 |
| D scheduler init/growth 失败 | terminal 602，不按 P context 初始化重试 | 不模拟 MTP 设备快照与 host table commit |
| P host load | 普通复制失败重算；materialization RETRYABLE 重新初始化，PERMANENT/INTERNAL 602 | 不验证真实 allocator 内容、DMA 或跨 rank 错误 |
| 普通 cancel / timeout | 第一个 Error 原因保留；consumer 可先收到 gRPC 错误；FINISHED 和 KV 释放等 scheduler 边界 | priority preemption 的 context finalization 另行处理 |
| priority cancel | 记录 CANCELING，等待相应本地/下游操作退出，清理后发布 8429/CANCELED；absent fence 与重复取消有保留窗口 | prepare 池使用相同逻辑容量；跨进程崩溃时序未差分验证 |
| error code 编码 | domain CANCELLED=8100 用于 TaskInfo/ErrorDetails；RpcErrorPB 的枚举 CANCELLED=2 不能直接当 domain 错误 | real transport 两套编码必须分别解释 |
| scheduler stop / process crash | Mock 有独立 drain、crash epoch 和重复终态保护；不执行真实 worker join、设备 event 和 deferred release | 不证明 real 异步 bookkeeping 的 stop/resource 安全窗口 |
| P forward 中途下游断链 | 原始下游错误立即通知 consumer；GPU lease 与 inflight 由 forward/scheduler 边界结算，不能在 RPC error callback 提前释放 | 只验证 gRPC socket 与模拟 forward，不验证 RDMA worker |
| buffered output 与 Error | Error 优先于尚未消费的输出；已提交 consumer terminal 不被后来 timeout 覆盖 | Native output dispatch / MTP worker 差分未验证 |

有限状态与错误分支的回归只能证明上述范围。完整 real 引擎还包含未模拟的 model forward、
MTP、异步 device bookkeeping、allocator 与 transport 分支，不能统称为“每个判断已对齐”。
