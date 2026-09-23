# Batch 执行录制、GPU Kernel 关联与形状回放设计

2026-09-17 当前实现（日志路径已更新）：仅输出 `logs/batch_schedule.<PID>.log` 和 owner 目录的 `manifest.json`。请求组成、输入/KV 长度和已知 TTFT 嵌入 batch 的 requests 数组；不采集或输出录制专用的入队/首次调度时间及 scheduler_step_id；不再输出 request_events.jsonl、executions.jsonl 或 requests.jsonl，不采集请求结束时间和终态。下文涉及这些文件及终态的内容属于历史设计或后续目标，以使用说明及代码阅读文档的当前实现为准。

日志配置调整：batch 输出由 alog.conf 的 batch_schedule logger / batchScheduleAppender 统一管理，录制器通过 `submitBatch(...)` 入队，由 worker 检查 INFO 级别后调用 `logBinaryMessage` 写入，默认配置为 `logs/batch_schedule.%p.log`，alog 将 `%p` 展开为当前进程 PID，不再位于 owner 目录。按进程分文件可避免同机多个 DP owner 的长 JSON 写入交错；自定义配置也应保持进程间文件隔离。多 DP 录制需收集各进程的 `logs/batch_schedule.*.log`。分析工具使用 `--batch-file` 指定实际文件；owner 目录仅保留 manifest。动态 appender、录制器手动分片以及 `RTP_LLM_RECORD_FILE_BYTES` 已移除；batch 文件的轮转和格式以 alog 配置为准。

2026-09-16 范围收敛：本期仅流量录制。新增 profiler scope、自动采集窗口、captures.jsonl 和回放自动 trace 采集均已移除，原有 profiler 功能不变。batch 默认持续追加到各进程的 batch_schedule.<PID>.log，不按 batch 建文件；默认禁用大小轮转，需要轮转时设置 alog.conf 的 max_file_size。下文 kernel 关联/回放设计作为后续目标，不是本期交付承诺。

状态：设计目标及实现跟踪。日期：2026-09-08。更新：2026-09-14，补充请求生命周期事件及第一版实现。

实现跟踪：基础录制、请求事件、trace 分析和 native 形状回放代码已接入，尚未完成全部生产验收。当前支持范围、配置和限制见 [使用说明](references/batch_recording.md)；下文保留完整设计目标。

2026-09-15 实现策略调整：保留上层录制器，JSONL 底层改用 alog FileAppender 异步输出，按 best effort 接受缺失数据，不再承诺全量无损。manifest 区分提交量与未知的实际落盘量；离线分析按每个已观察请求检查事件完整性，不因其他请求丢失而清空全部耗时。下文旧版严格完整性要求仅作原设计背景，当前行为以使用说明和代码阅读指南为准。

采集策略更新：`RTP_LLM_RECORD_DECODE` 默认关闭，只有设为 `1` 才采集包含 decode 的 batch。关闭时整批跳过（包含混合 batch），保留纯 prefill 和请求生命周期事件；完整连续长度录制及 decode 回放需要显式开启该开关。profiler 窗口独立控制。

## 1. 目标与第一版边界

保存线上每次模型执行中逐请求的 token/KV 长度配对，在指定采集窗口关联每个 GPU kernel 的执行时间，离线重建相同 batch 执行形状，并输出原始执行与回放的对比报告。

第一版交付：

- 完整逐请求长度快照，不能截断长 batch，也不能用均值或独立直方图代替逐请求配对。
- 全链路唯一执行标识，关联调度 step、模型 forward、rank 和 kernel。
- 连续长度录制，以及指定窗口的逐 kernel 追踪。
- 请求生命周期事件：enqueue、first_scheduled、first_token、finish/cancel/error，用于计算引擎侧请求延迟。
- 执行层形状回放：普通 prefill、decode、混合 batch、chunked prefill、异步 decode；覆盖 eager、CUDA Graph 和 TP/DP 标识。
- 数据完整性检查、长度分布分析、逐 kernel 对比报告。

第一版的执行单位是一次模型 forward。模型内部计算与通信 kernel 纳入该 forward；采样、KV 搬运等外部阶段单独标识，不能误计入模型耗时。记录到的其他 GPU activity 保留，不强行归属某个 forward。

MTP/draft/verify 在协议中预留子执行身份，第一版检测到未适配路径时明确报告 unsupported，不按普通 decode 回放。多模态和特殊 cache 布局必须有对应回放适配器才能声明支持。

形状回放不保证复现输出内容、MoE 路由、缓存数据内容、在线调度行为或线上竞争负载。报告必须标明这些差异。

## 2. 仓库现状与改动位置

| 位置 | 当前行为 | 计划改动 |
| --- | --- | --- |
| `rtp_llm/cpp/normal_engine/NormalModelInputGatherer.cc` | 组装逐请求输入；decode 排在 context 前；异步 decode 可使用 GPU 长度 | 在真实输入形成时采集不可变快照，保持 slot 对应关系 |
| `rtp_llm/cpp/normal_engine/NormalExecutor.cc` | gather、TP 同步、forward 已有 profiling scope | 分配/传播执行身份，提交录制任务，包围 forward |
| `rtp_llm/cpp/models/ModelInputsLogger.cc` | 调试输出最多 64 个 tensor 元素，GPU 值可能缺失，内容主要为字符串 | 保留调试用途；新增独立结构化 recorder |
| `rtp_llm/cpp/utils/ProfilingScope.h` | 支持动态 scope 名称 | scope 只携带短执行标识，不塞入长度数组 |
| `rtp_llm/cpp/engine_base/TorchProfiler.cc` | step 窗口启停、异步保存 | 增加 capture 身份、窗口清单、完整性状态 |
| 实际 CUDA Graph 选择/执行路径 | graph 选择发生在下层 | 补充实际 bucket、padding、capture/replay/fallback 信息 |
| `rtp_llm/test/perf_test/` | 已有压测与 batch=1 cache grid | 新增执行快照回放入口和关联分析工具 |

实现前继续定位所有实际 graph 路径及 TP 输入打包结构；表中的新接口和文件名均为拟议设计。

## 3. 数据模型

### 3.1 身份与文件布局

使用以下层级，禁止依赖各 rank 独立自增后碰巧对齐：

```text
session_id（一次录制会话）
  replica_id / dp_rank（独立调度域）
    scheduler_step_id（一次调度）
      exec_id（一次模型 forward）
        world_rank（该卡的实际执行和 kernel）
```

同一 TP 组由输入所有者产生执行 ID，并随输入传播。进程重启生成新的 session/实例身份。请求标识使用会话内稳定匿名 ID；`sequence_id` 表示同一请求在本轮执行中展开出的序列位置索引，从 0 开始，每个请求独立编号；不保证跨轮稳定，不能据此关联不同轮次的逻辑序列或 beam。预留 parent_execution_id、model_role、substep_index。

每次会话输出：

```text
manifest.json                 # schema、环境、模型、拓扑、采集配置、文件清单
request_events.jsonl           # 请求入队、首次调度、首 token 和终态事件
batches.<owner>.jsonl          # 每次执行的完整逐请求快照
executions.<rank>.jsonl        # 各 rank 实际执行形状、状态、graph 信息
trace.<capture>.<rank>.json    # profiler 原始输出
capture.<capture>.json        # 参与 rank、执行范围、采集/丢失状态
kernel_events.<rank>.jsonl     # 离线归一化后的 kernel 记录
report.json / report.html     # 机器可读结果与可视化报告
```

使用版本化 JSONL；第一版不引入在线数据库。manifest 记录代码版本、模型配置/权重版本标识、GPU、驱动、CUDA、PyTorch、profiler 版本、TP/DP 配置、dtype/量化、attention backend、cache group 布局、block size、graph 配置及异步执行开关。不要复制整个环境变量集合。

### 3.2 逐请求长度协议

| 字段 | 定义 |
| --- | --- |
| `batch_slot` | 实际模型输入中的序列顺序，不能仅按请求顺序猜测 |
| `request_id` | 稳定匿名请求标识 |
| `sequence_id` | 请求在本轮执行中的序列位置索引，从 0 开始，缺省为 0；不保证跨轮稳定 |
| `phase` | prefill、decode；预留 draft、verify |
| `input_len` | 原始输入长度 |
| `q_len` | 本次 forward 对该序列实际计算的 token 数 |
| `kv_len` | 本次 forward 前已经存在的逻辑 KV token 数 |
| `reuse_len` | 初始命中前缀长度；来源确认后写入，未知为 null |
| `chunk_index` | chunked prefill 的分块序号，非 chunk 可为空 |
| `length_source` | 当前为 cpu_input_snapshot；旧录制可能为 host_input 或 device_snapshot |
| `is_fake` | 是否为同步/填充构造的序列 |

`q_len` 和 `kv_len` 为可选字段：CPU 快照中 q <= 0 或 KV < 0 时，仅省略对应字段，仍保留整个 batch、所有请求及其顺序和其他有效字段。`kv_len=0` 是有效值，表示没有已有 KV 前缀。字段缺失表示未知，消费端不得默认补零；依赖完整长度的统计应排除缺失样本，完整 batch 形状回放应将包含缺失长度的 batch 标为不可回放。槽位数量与长度数组大小不一致仍属于结构错误。

仅当两个字段都有效时，标准全注意力可用 `q_len + kv_len` 表示本次逻辑 attention 长度，但不能把它等同于物理 KV 容量或滑窗可见长度。

当前代码映射：prefill 的 q 来自 `currentExecuteTokens(i).size()`，KV 来自实际 `prefix_lengths`。普通同步 decode 的 q=1，KV 来自 `sequence_lengths`，其值为 `seqLength()-1`。decode 的 `input_lengths` 是原始输入长度，不能当作 q。异步 decode 必须使用本次真实 GPU 输入长度快照，不能直接读取可能滞后的 CPU seqLength。

`GenerateStream::prefixLength()` 当前返回 reuse_length_；`initialReuseLength()` 另有字段。初始命中与后续分块已有前缀不能混用，接入时用专门用例核验。

示例（仅展示核心字段）：

```json
{
  "schema_version": 1,
  "session_id": "s001",
  "replica_id": "replica0",
  "dp_rank": 0,
  "scheduler_step_id": 9120,
  "exec_id": "18231",
  "model_role": "target",
  "requests": [
    {"batch_slot": 0, "request_id": "r17", "sequence_id": 0, "phase": "decode", "input_len": 1024, "q_len": 1, "kv_len": 2048},
    {"batch_slot": 1, "request_id": "r18", "sequence_id": 0, "phase": "prefill", "input_len": 4608, "q_len": 512, "kv_len": 4096}
  ]
}
```

### 3.3 实际执行与 kernel 协议

各 rank 的 execution 记录包含：执行键、真实设备身份、逻辑/物理 batch size、真实/填充 token 数、execution_mode、graph bucket、graph 实例/launch 标识（可获取时）、warmup/fake/skip 状态、执行成功/失败状态、时间戳和时钟域。

特殊 cache 适配器补充 group 类型、布局、有效长度和容量元数据。第一版不导出 GPU 指针或完整 KV 数据，也不声称复现物理地址布局。

每个 kernel event 包含：执行键（无法关联则 null）、capture_id、rank、device、context、stream、原始名称、start_ns、duration_ns、correlation_id、graph/node 标识（可获取时）、association_method 和 association_status。时间转换保留原始时钟来源；跨主机时间不直接相减。

同名 kernel 的每次调用都保留，汇总时再聚合。无法归属、窗口边界不完整、后台任务分别标识，不能丢弃后伪装成完整追踪。

### 3.4 请求生命周期事件协议

每个事件一行 JSON，由持有请求生命周期的 owner 记录，TP follower 不重复输出。多 owner 分别写入自己子目录中的 `request_events.jsonl`，manifest 列出分片，不允许多个进程写同一个文件。请求键为 `session_id + replica_id + dp_rank + request_id`，与 batch 快照使用同一匿名请求 ID。

| event | 打点位置与计时边界 | 次数 |
| --- | --- | --- |
| `enqueue` | 引擎成功接收入队的状态转换处，在调度器可见请求前确定时间戳 | 每请求一次 |
| `first_scheduled` | 调度/执行路径首次确定提交该请求进入实际执行 batch；候选检查、预分配、撤销的选择不算 | 最多一次 |
| `first_token` | 输出路径首次发布可供上层消费的新生成 token；不含 prompt echo、空输出、draft token | 最多一次 |
| `finish` | 请求状态机提交正常完成终态，包括 EOS、长度限制等正常结束 | 三种终态合计最多一次 |
| `cancel` | 请求状态机确认取消并提交终态；仅收到取消信号不算 | 同上 |
| `error` | 请求状态机提交不可恢复失败终态；可重试错误不算 | 同上 |

具体接入函数在实现时核验。时间戳在状态转换处采集，不能使用 writer 写盘时间。first_token 是引擎输出发布时刻，不是 GPU 首次生成或客户端收包时刻；非流式请求可能到最终输出才产生此事件。多序列请求在首次发布任一序列的新 token 时记录一次，并携带该次执行中请求内的序列位置索引 `sequence_id`，不表示跨轮稳定身份；终态以整个请求为单位。

| 字段 | 定义 |
| --- | --- |
| `schema_version` | 第一版为 1 |
| `session_id / replica_id / dp_rank / request_id` | 请求关联键 |
| `owner_id / clock_id` | owner 进程实例及单调时钟域，重启更换身份 |
| `event / event_seq` | 事件类型及请求内递增序号，从 0 开始；提交日志队列前分配，丢弃不复用 |
| `timestamp_monotonic_ns` | 状态转换处的单调时钟时间，用于耗时计算 |
| `timestamp_unix_ns` | 同处采集的 UTC epoch 时间，仅用于检索和粗粒度跨系统对照 |
| `scheduler_step_id / exec_id` | first_scheduled 必须关联首次执行；first_token 在可追溯时关联生产执行；其他未知为 null |
| `sequence_id` | 首 token 对应执行中请求内的序列位置索引，不保证跨轮稳定；请求级事件为 null |
| `output_mode` | streaming 或 non_streaming |
| `reason / error_code` | 终态原因和稳定错误码，不适用为 null；不写原始 prompt 或异常内容 |
| `generated_tokens` | 终态时各序列累计已发布的新生成 token 数之和，不含 prompt/draft；未知为 null |

示例（实际一行一个对象）：

```json
{"schema_version":1,"session_id":"s001","replica_id":"replica0","dp_rank":0,"request_id":"r17","owner_id":"owner-01","clock_id":"owner-01-monotonic","event":"first_scheduled","event_seq":1,"timestamp_monotonic_ns":120010000000,"timestamp_unix_ns":1789344000010000000,"scheduler_step_id":9120,"exec_id":"18231","sequence_id":null,"output_mode":"streaming","reason":null,"error_code":null,"generated_tokens":null}
```

典型序列为 `enqueue → first_scheduled → first_token → finish`。允许 `enqueue → cancel/error`，以及没有首 token 的正常结束。首次调度后执行失败仍保留 first_scheduled，并最终记录 error；首 token 后发生 cancel/error 也保留 first_token。取消与正常完成竞争时，以请求状态机实际提交的终态为准，在同一锁或原子状态转换保护下保证首次事件和终态只生成一次，不在析构或多个回调中重复打点。

同一 clock_id 且事件完整时可计算：

- `engine_request_latency_ns = terminal - enqueue`，terminal 为 finish/cancel/error，成功、取消和失败分别统计。
- `initial_queue_latency_ns = first_scheduled - enqueue`，仅表示首次等待，不含抢占后的累计排队。
- `engine_first_token_latency_ns = first_token - enqueue`，按 output_mode 分组，不等同于客户端 TTFT。
- `post_first_schedule_latency_ns = terminal - first_scheduled`，包含后续等待、计算和输出开销，不是请求独占 GPU 时间。

缺少端点或时钟域不同，结果为 null 并记录原因，不能补零。仅凭首 token 和终态不能还原每 token 间隔或 P99 TPOT，需要后续逐 token 事件或独立压测。

请求事件与 GPU 窗口独立：lengths/window 模式下，在配置范围内连续记录全部真实已入队请求，不按请求采样，包括尚未进入 batch 就取消的请求。fake/warmup 请求不写入；入队前校验拒绝和网络失败不在计时范围内。动态启停不伪造历史事件：manifest 记录覆盖区间和在途请求标识，跨界请求标为 partial。崩溃或缺少终态不能自动推断为 cancel/error。

使用有界异步队列，event_seq 在状态转换时确定，文件允许乱序。队列满整条丢弃并按事件类型计数；manifest 记录事件生成、入队、写入、丢弃数量及 writer 错误。结合请求事件序号、必要事件、覆盖区间与文件收尾状态检查完整性，无法排除丢失时不标为 complete。崩溃可能丢失未落盘计数，不能仅靠丢弃计数为零证明完整。

## 4. 录制流程与并发约束

1. 调度/执行线程分配 exec_id，gather 时获得 slot、请求身份和实际输入长度。
2. CPU 输入在其有效期内复制为 recorder 自有数据；GPU 输入在确定的流依赖下冻结长度快照。
3. 执行身份随 TP 输入传播。forward 范围加入 exec_id，graph 路径补充实际执行元数据。
4. 录制任务进入有界队列，后台线程等待快照可读，完成序列化、轮转与写盘。
5. 成功、失败、跳过均形成可核验终态；异步日志允许乱序，用执行键关联。

GPU 快照不得在后台直接读取可被下轮覆盖的 tensor。先在有序流上复制到自有设备缓冲，再通过 event 依赖异步复制到 pinned host；若能证明源不可变且生命周期足够，可省略中间副本。使用独立 copy stream 时必须显式建立生产/消费依赖，并持有全部缓冲直到 copy 完成。录制复制范围标为 recorder overhead，分析时不计入模型 kernel。

不能用 `.cpu()` 或每步 `cudaDeviceSynchronize()` 实现长度录制。队列/缓冲池满时整条丢弃并计数，不写半个 batch，不阻塞推理。完整记录要求与过载丢弃并不矛盾：无丢失窗口才可标记 complete。

文件按大小轮转，保留策略可配置。writer 异常关闭本次录制并暴露状态，推理继续；服务退出时做有时限 drain，并记录未落盘数量。capture manifest 最后完成提交，异常退出留下 incomplete 状态。

## 5. GPU 追踪与归属

第一版优先复用 Torch/Kineto，通过执行范围与 CUDA launch correlation 归属 kernel。解析器按采集工具和版本适配，不假定所有 trace 的 external ID 字段相同。

关联链路是 execution scope → CPU launch → GPU activity。禁止仅用 GPU 时间落入 CPU scope 的区间来推断归属。跨线程 launch 必须传播执行上下文；共享/后台通信不能强行归属。

CUDA Graph 要关联某次 replay 的 launch 及其 node activity；单独 graphNodeId 在多次 replay 间不唯一。只有 graph 总时间时将 kernel coverage 标为不支持，不均摊成单 kernel 时间。

M0 先验证当前 PyTorch/Kineto、驱动和 graph 路径。如果无法提供完整 graph node 归属，第一版保留 Nsight Systems node trace 作为回放/受控采集后端，并补充 NVTX 执行身份标记。Torch RecordFunction 不视为自动存在的 NVTX 标记。两种 profiler 后端互斥启用，不能未经验证并行争用采集资源。原生 CUPTI collector 留为后续目标。

窗口使用 capture_id 和明确的起止执行序号，所有参与 rank 确认启停。窗口内强制同时录制长度快照；采集启动/结束在推理线程遵守现有 profiler 线程归属。关闭时收尾在途活动，边界执行若缺少完整 activity，标为 partial。禁止为了长度触发窗口在主线程同步 GPU；GPU 才有的触发条件采用后续窗口策略。

参考：[CUPTI external correlation](https://docs.nvidia.com/cupti/api/structCUpti__ActivityExternalCorrelation.html)、[Nsight Systems graph/node tracing](https://docs.nvidia.com/nsight-systems/UserGuide/index.html?highlight=sm)。节点级追踪存在额外开销，需实测；不承诺线上全时逐 kernel 采集低开销。

## 6. 形状回放

新增离线执行入口，复用模型初始化、输入构造、KV 管理和 forward，按录制 slot 顺序直接提交 batch。HTTP 重发无法保证恢复原 batch，因此不作为该模式的执行入口。

流程：校验 manifest/支持矩阵 → 选取执行快照 → 配置相同模型与拓扑 → 构造有效 KV 和输入 → warmup/graph 准备 → 无 profiler 基线测量 → 开启 profiler 重复测量 → 生成对比报告。

- 使用固定随机种子产生合法 token；通过 cache 适配器分配并初始化有限数值的 KV，不能读未初始化内存。
- 标准位置编码按已有 KV 长度构造；不支持的 position/cache 布局拒绝回放，输出具体原因。
- 保留逐请求 q/KV 配对、顺序、混合类型；恢复实际 graph bucket 和 padding。无法恢复时明确标记 mismatch。
- 每轮测量前在计时范围外重建基准状态，避免前轮写 KV、采样或长度增长污染下轮。
- 第一版默认逐快照独立回放；多个快照可顺序执行，但不宣称恢复请求生命周期、cache 共享或线上到达时间。
- 多卡重建同一执行的全部参与 rank；记录各 rank 的 kernel 与通信差异，不通过重发请求重新分配 DP。
- MoE 可以提供形状回放，但报告标记 route_uncontrolled；不以 kernel 数量或路由完全一致作为其保证。

拟议工具接口（实现后才可执行）：

```text
batch_replay --record-dir DIR --execution-ids IDS --warmup 5 --repeat 20
batch_trace_analyze --record-dir DIR --replay-dir DIR --output DIR
```

## 7. 分析与对比输出

报告展示 batch size、逐请求 q/KV 联合分布、各执行类型占比、graph bucket/padding，以及按长度条件筛选后的 kernel 热点。

补充请求总耗时、首次排队、引擎首 token 延迟和终态占比，列出 partial 请求及事件丢失情况。通过请求键关联其参与的 exec_id；这些 batch 的 GPU 耗时属于共享执行，不能按请求数或 token 数均摊为请求独占耗时。

耗时口径分别输出：单 kernel duration；各 kernel duration 总和；GPU activity 区间并集；首个相关 GPU activity 到最后一个结束的跨度；CPU 提交耗时。多 stream 重叠时这些数值不能互换。第一版不把跨度称为精确关键路径。

原始/回放对比按配置指纹、执行形状和 rank 匹配；按 kernel 名称/可用算子范围展示调用次数、总耗时、分位数和变化率，不强行把第 N 个 kernel 与另一侧第 N 个匹配。报告同时列出 unmatched kernel、缺失 rank、丢失记录、graph mismatch 和配置差异。

每个统计组显示样本数；单次线上执行不能生成有意义的分位数。记录 profiler 开销与无 profiler 基线，区分测量扰动和回放差异。窗口外执行只有长度数据，GPU 耗时显示 unavailable。

## 8. 配置与可观测性

拟议新增配置：record mode（off/lengths/window）、输出目录、最大文件/总保留量、队列容量、快照缓冲预算、capture start/steps、profile backend、rank 范围。缺省关闭。第一版不采集 prompt/token 内容。

暴露低基数指标：录制条数、丢弃条数、队列深度、缓冲占用、快照字节数、writer 错误、录制 CPU 开销、capture 状态。请求 ID、执行 ID 和完整长度数组只进入文件，不作为 metrics label。

支持矩阵按模型/cache adapter、执行模式、profiler 后端及版本写入 manifest；未知组合返回 unsupported，不默认为兼容。

## 9. 实施顺序与验收

| 阶段 | 交付 | 通过条件 |
| --- | --- | --- |
| M0：可行性验证 | eager/graph、多卡最小 trace，输入字段语义核验 | 已知 launch 集合能够正确归属，graph 每次 replay 能区分；确定可用后端 |
| M1：结构化录制 | recorder、执行 ID、异步快照、请求事件、配置和指标 | 长度与实际输入一致，batch 不截断，异步无旧值，首次/终态事件去重，过载有明确缺口 |
| M2：追踪闭环 | 窗口控制、rank 清单、kernel 关联器 | 完整窗口内已知 kernel 归属正确，缺失/边界能检测 |
| M3：回放和报告 | 支持矩阵内的执行回放、对比工具 | q/KV/slot/graph 形状一致，输出基线及 kernel 对比和全部限制 |

必须覆盖的验证：

- batch size=1、64、超过 64；相同总 token 但不同逐请求配对。
- KV=0、有前缀命中、chunked prefill、混合 prefill/decode。
- 异步 decode 连续多步，与受控测试中的真实 device 输入逐项核对。
- TP 多 rank、DP 独立调度域、warmup/fake/skip；缺失 rank 可检测。
- eager、多次 graph replay、不同 bucket、graph fallback。
- 队列满、writer 失败、采集边界、异常退出；不得输出假的 complete。
- 请求正常完成、排队中/执行中取消、首 token 前后失败、空输出、非流式输出、多序列首次输出：核验事件语义与耗时公式。
- 并发取消/完成、重复回调、抢占再调度：首次事件最多一次，终态最多一个；writer 乱序不改变状态转换时间，TP follower 不重复输出。
- 请求事件丢失、崩溃、动态启停、时钟域不一致：标为 partial/null，不伪造终态或耗时。
- 在受控无丢失用例中要求快照完整率、已知目标 kernel 关联率 100%；线上报告实际覆盖率及分母，不把后台 kernel 混入分母。
- 对录制关闭、仅长度录制、窗口 profiler 三种模式分别测吞吐、TTFT、TPOT、CPU/内存/磁盘开销。

建议仅长度录制的初始性能目标：固定环境重复对照下吞吐下降不超过 1%，P99 TPOT 增幅不超过 2%；这是待基准确认的目标，不是现有性能承诺。窗口 profiler 单独报告成本，不套用该目标。未达到指标时先分析快照复制和写盘成本，不通过静默采样改变“完整录制”的语义。

## 10. 后续 TODO

| 优先级 | 目标 | 依赖与完成标准 |
| --- | --- | --- |
| P1 | 请求时序回放 | 复用第一版请求事件，补充前端到达、抢占、重算、前缀共享和回放驱动；可比较调度形成的 batch 分布 |
| P2 | 逐 token / 逐序列时间线 | 补充 token 发布、各序列生命周期及前端边界，支持 ITL/TPOT 分布和客户端延迟分析 |
| P1 | MTP/Eagle 等投机执行 | 适配 draft/verify 子执行、提议/接受长度和 cache 回滚；不能假设 decode q=1 |
| P1 | 特殊 KV 布局适配 | 支持滑窗、压缩 KV、多 group/DSV4 等有效长度与物理布局；给出逐模型验证矩阵 |
| P1 | MoE 路由复现 | 记录专家 token 分布/通信量；评估内容或中间状态回放，区分强制路由与自然路由 |
| P1 | PD 分离与跨机链路 | 关联 prefill/decode、KV 传输、跨机时钟和请求迁移 |
| P2 | 自适应采集窗口 | 根据稀有长度组合、延迟异常和 graph fallback 触发，并报告采样偏差 |
| P2 | 原生 CUPTI collector | 在兼容性、graph node、资源占用和版本支持验证后，评估持续追踪收益 |
| P2 | 完整 GPU 工作与关键路径 | 扩展 sampler、KV copy、通信及跨流依赖，输出关键路径而非简单 duration 总和 |
| P2 | 更高保真 cache 回放 | 重建共享块、分配/释放顺序、碎片和缓存热度；与纯形状模式分开报告 |
| P2 | 多模态/内容回放 | 定义可选内容载荷、保存策略和模型专用输入适配，不扩大默认录制内容 |
| P2 | 大规模分析与归档 | 转换 Parquet、远端归档、长周期分布比较、版本性能回归门禁 |

上述 TODO 不阻塞支持矩阵内第一版交付；当前线上必需但尚无适配器的模型/执行路径，必须先补充对应适配后才能宣称该路径完成。
