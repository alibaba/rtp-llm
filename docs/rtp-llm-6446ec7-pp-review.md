# RTP-LLM PP 实现审查：6446ec7

- 审查对象：[6446ec7e17d8579fba56399cfc7ccedbb72b94f8](https://github.com/alibaba/rtp-llm/commit/6446ec7e17d8579fba56399cfc7ccedbb72b94f8)
- 提交主题：`feat(engine): support pipeline parallelism with PD and speculative decoding`
- 审查日期：2026-10-16
- 验证范围：阅读该提交的执行、调度、采样、序列化、KV 拓扑及相关测试代码；对调度条件做了简化状态模拟。未运行多 GPU 集成测试或性能压测。

## 审查结论

这次实现有一个明确的请求错误隔离问题，以及多处可以从代码确认的调度、数据传输和拓扑限制。建议优先处理 beam 请求隔离、PDFUSION prefill 公平性，以及 DSpARK proposal 概率的状态归属。

下文区分代码行为与性能推断。数据量示例是按 tensor 形状计算的理论值，不是实测延迟或显存峰值。P1 表示建议优先处理，不表示所有问题都已在多 GPU 环境复现。

## 1. [P1] 不支持的 beam 请求在执行线程触发断言

### 触发条件与证据

`PPBatchStreamProcessor::gatherSamplingPlan()` 使用如下断言拒绝 beam search：

```cpp
RTP_LLM_CHECK_WITH_INFO(!stream->hasNumBeams(), ...);
```

此时请求已被 scheduler 选中并标记为 inflight。该宏会抛出异常，开启对应 core dump 配置时直接调用 `abort()`。`NormalEngine::loop()` 没有将这个异常转换为单请求错误的处理。

在 PP、未开启 speculative decoding 的服务上，`num_beams > 1` 请求可能打断 engine 执行，并让其他 stage 停在等待通信的位置。Python 侧针对 speculative 请求的兼容性检查不能覆盖这个场景。

### 改进建议

在入队前校验 PP 请求能力并返回 `INVALID_PARAMS`。执行阶段如发现不兼容请求，应报告该请求错误、移出本轮 batch 并清理 inflight 状态，避免用进程级断言验证用户参数。

### 代码位置

- [beam 检查：PPBatchStreamProcessor.cc:63](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.cc#L63)
- [断言行为：AssertUtils.cc:13](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/utils/AssertUtils.cc#L13)
- [执行循环：NormalEngine.cc:769](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/NormalEngine.cc#L769)

## 2. [P1，调度] PDFUSION 的 prefill 存在饥饿条件

### 触发条件与证据

`PPScheduler::schedule()` 对 PDFUSION 使用以下策略：

```cpp
case RoleType::PDFUSION:
    if (scheduled_streams.empty()) {
        admitWaitingStreams(scheduled_streams);
    }
```

只要本轮存在 ready decode，就完全跳过新 prefill 的 admission。结合 ring 收结果的方式，流水线填满后可以形成持续循环：

```text
A 的结果回来 → 调度 A 下一轮
B 的结果回来 → 调度 B 下一轮
C 的结果回来 → 调度 C 下一轮
……
每轮都有 ready decode，新 prefill 持续等待
```

即使某个 decode batch 只剩一个长请求，也不会利用该轮插入新的 prefill batch。新请求需要等到出现一次没有 ready decode 的调度轮次。

按源码的 ring 推进和 admission 条件所做的简化状态模拟能够维持该循环；它不等同于真实 GPU 集成测试。预期影响是长输出负载下 TTFT 增大。由于 waiting 请求的超时和错误清理也在 admission 路径中，这些处理同样可能被推迟。

### 改进建议

增加有界公平性，例如限制连续 decode batch 数量，或在 waiting prefill 达到等待时间阈值时插入一个独立 prefill batch。这个调整不要求在同一 batch 混合 prefill 与 decode。

### 代码位置

- [调度分支：PPScheduler.cc:329](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/engine_base/schedulers/PPScheduler.cc#L329)
- [ring 收结果位置：PPExecutor.cc:1413](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L1413)

## 3. [高优先级性能] DSpARK 概率矩阵每轮绕完整条流水线

### 当前数据路径

proposal 概率 `q[B,K,V]` 的生产者和消费者都在 last stage，但当前路径是：

```text
Last stage GPU 生成 q
    → CPU
    → execution result 返回 Stage 0
    → 保存到 stream
    → 下一轮重新拼入 sampling plan
    → Stage 1 → Stage 2 → … → Last stage
    → GPU
    → rejection sampling
```

这会额外引入 D2H/H2D、GLOO 传输、序列化以及 CPU 内存复制。

举例：`B=32、K=4、draft vocab=32768、FP32` 时，一份 q 为：

```text
32 × 4 × 32768 × 4 bytes = 16 MiB
```

PP=4 时，返回 Stage 0 再逐 stage 送回 last stage，四条逻辑传输链路合计搬运 64 MiB。这是逻辑传输量；实际跨机网络流量取决于部署布局。

### 改进建议

将 q 按 `(request_id, proposal_epoch)` 留在 last-stage GPU。Stage 0 接收 accepted tokens、下一轮 proposal token IDs 和版本号即可。batch 重排时，last stage 按请求标识 gather 对应概率；请求结束时释放。PD 初次交接使用单独的初始化路径。

不能直接丢弃 q：随机 speculative decoding 的 rejection sampling 需要与 proposal 对应的概率分布。这里建议改变状态归属，而不是省略概率信息。

### 代码位置

- [draft 概率搬回 CPU：PPExecutor.cc:1158](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L1158)
- [下一轮重新打包 q：PPBatchStreamProcessor.cc:205](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.cc#L205)
- [verify 再搬回 GPU：PPExecutor.cc:774](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L774)

## 4. [性能] 每轮全量传输 token history 和静态采样配置

### 当前行为与代价

`gatherSamplingPlan()` 每轮重新分配以下 tensor，并复制每个请求的完整历史：

```cpp
token_ids[total_batch_size, max_seq_len + 1]
```

grammar、banned combos 和 sampling 参数也重复进入 plan。中间 stage 的 root 会反序列化完整 plan，再转发原始 payload，尽管采样部分只有 last stage 消费。TP 非 root 使用较小的 metadata-only plan，不能将这里的开销理解成所有 TP rank 都传一份完整采样计划。

例如 `B=64、L=128K`，仅 int32 token history 就约为 32 MiB/轮，还没有计入 KV tables、q 和其他输入。greedy decode、未开启 history penalty 的请求也走这个全量历史路径。

### 改进建议

拆分请求注册信息、每轮执行信息与采样状态：

- 首次注册发送配置及 prompt/history。
- 每轮发送新提交 token、执行位置和必要的 block table 更新。
- 中间 stage 只接收 forward 所需信息。
- last stage 维护采样所需历史，并处理请求结束清理。

### 代码位置

- [全量历史复制：PPBatchStreamProcessor.cc:196](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.cc#L196)
- [plan 序列化：PPSerialization.cc:357](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPSerialization.cc#L357)

## 5. [架构限制] Stage 0 的逻辑 KV 管理与物理 cache tag 耦合

### 当前约束

拓扑校验要求后续 stage 的所有 cache tag 都存在于 Stage 0：

```cpp
if (canonical_index.find(group.tag) == canonical_index.end()) {
    return fail("... leading PP stage must own every cache group ...");
}
```

例如，Stage 0 只有 FULL attention、Stage 1 只有 LINEAR attention，即使各自显存充足且模型前向可以分段，该拓扑也会因 Stage 0 缺少 LINEAR tag 而被拒绝。

同类型 BlockTree group set 还要求完整出现或完全缺席，部分成员重叠被拒绝。相关测试明确固定了这些约束，因此它们属于当前实现边界，不是已经确认的 KV 数值错误。

### 改进建议

Stage 0 持有全局 tag 并集和 allocator 元数据；每个 stage 只为本地 layers 分配物理 KV。各 tag 的容量由真正拥有该 tag 的 stage 协商，block-copy/transfer 元数据按接收 stage 的本地 group 投影。

共享 tag 取最小容量是合理的保守策略。需要解除的是“逻辑管理权必须对应 Stage 0 本地物理存储”的耦合，否则会限制异构 KV 类型模型的非均匀切层。

### 代码位置

- [Stage 0 tag 限制：PPTopologyValidator.cc:322](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/cache/PPTopologyValidator.cc#L322)
- [部分 group set 被拒绝的测试：PPTopologyValidatorTest.cc:96](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/cache/test/PPTopologyValidatorTest.cc#L96)

## 6. [性能/显存] Activation snapshot 的释放粒度过粗

### 当前行为与代价

发送 activation 前执行 `clone()`，是为了防止下一轮 forward 覆盖仍在传输的 model output buffer。但通信 ticket 强引用 snapshot，直到对应 ring slot 再次复用时才释放。

因此发送完成后，snapshot 仍可能留在显存中。`pp_size + 1` 个 slot 可以保留多轮 activation，这部分不能只按一份额外 hidden 估算。

例如，单个 BF16 tensor 的形状为 `8192 tokens × 8192 hidden` 时占用 128 MiB。如果还传 residual、辅助 hidden 或同时保留多个 slot，显存占用继续增加。实际峰值取决于 tensor 集合、batch 大小和通信完成时间。

### 改进建议

分离请求/result ring 与 send buffer 的生命周期。通信完成后及时回收 snapshot，或使用有容量上限的 buffer pool。当前 `PPTransport` 只有 `wait()`，可以补充非阻塞完成查询或等价的 event 回收机制。

### 代码位置

- [activation clone：PPExecutor.cc:183](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L183)
- [ticket 持有 tensor：ExecOps.cc:710](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/models_py/bindings/core/ExecOps.cc#L710)
- [通信接口：PPTransport.h:31](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPTransport.h#L31)

## 7. 需要压测评估的执行限制

### 优先支持 decode 节点上 PP target verify 的 CUDA Graph

#### 当前限制与优化目标

本建议针对 speculative decoding 中 decode 节点的 **target verify**，不是 prompt prefill，也不是要求把 scheduler、acceptance、sampling 和 draft 全部放进同一个 graph。

当前 PP target 被显式限制为 eager，有两道开关：

- `PPExecutor` 创建 target `PyWrappedModel` 时传入 `allow_cuda_graph=false`。
- `PyWrappedModel` 的 graph 启用条件还要求 `pp_size_ == 1`。

但非 PP 路径已经有 target verify graph：`num_tokens_per_bs = gen_num_per_cycle + 1`，角色为 `CudaGraphRole::TARGET_VERIFY`。因此应在已有 verify graph 能力上适配 PP stage 的输入输出，而不是把它作为 generation-prefill graph 的扩展，也不能只解除上述开关便认为适配完成。

固定每轮投机长度 K 时，verify 输入可按 `B × (K+1)` 分桶。接受长度由 forward 后的验证决定；接受长度变化本身不阻止本轮 target forward 使用 graph。下一轮 token、position、KV 长度和 block table 可以更新到固定地址的输入 buffer。实际输入不满足已捕获形状或后端约束时，应走经过通信一致性检查的 fallback；可变 K、PD 首轮和混合 batch 需要单独验证。

#### 捕获粒度：每个 rank 的本地 forward

非 PP 实现中，每个 rank 创建自己的 runner 和 graph，按 bucket 捕获自己的 forward，并在运行时 replay 本地 graph；不是 TP root 捕获后广播，也不是一个集中持有的跨设备 graph。PP 可以沿用这个粒度，让每个 stage 内的各 rank 捕获本地 target 分片。

`syncCudaGraphCaptureRanks()` 在 Python model 初始化后、`initializeCapture()` 前执行 WORLD barrier，确保所有 rank 完成初始化后再进入捕获流程。它不是每轮 replay 的 barrier，也不能替代后续 collective 的顺序约束。

同一通信组内，warmup、capture 和 replay 的通信调用必须匹配。若 MoE 使用 EP，dispatch/combine 仍需跨 rank：warmup 实际执行通信，capture 捕获 graph-compatible 的通信路径，replay 每轮实际传输当前 token 数据。动态 routing 必须由支持捕获的通信/算子后端处理，不能假定任意动态 dispatch 都可以直接 capture。EP=1 不一定有跨 rank expert dispatch，但启用 TP 时仍可能有 TP collective。

#### 推荐实施边界

第一步把 PP stage 间通信留在 graph 外，只捕获本地 target forward：

```text
plan / attention metadata 准备（graph 外）
    → 接收本轮 activation（首 stage 使用 token 输入）
    → 更新固定地址的 graph 输入 buffer
    → replay 本 stage、本 rank 的 target forward
    → 中间 stage 发送 activation / last stage 进入后处理
```

需要同时补齐以下内容：

1. **PP 输入输出适配。** capture 使用与运行时一致的 `pp_intermediates` 结构，覆盖 hidden、residual 及模型需要的辅助 tensor；last stage 保留 MTP 所需的 target hidden 输出。不能只适配输入，仍沿用仅处理最终 hidden 的输出假设。
2. **静态 buffer 与分桶。** 为每个 bucket 准备固定地址、形状兼容的输入输出和 attention metadata；正确处理 padding、KV 写入位置及无效行，避免 padding 污染真实请求的 cache。
3. **异步生命周期。** 用 event 连接 receive → 输入更新 → replay → send。graph 输出被异步发送或被 draft 消费期间，不得由下一轮 replay 覆盖；初版可保留 snapshot 拷贝，再评估有界 buffer pool 或多套 graph/buffer 的成本。下一轮更新输入也不能覆盖尚未读完的上一轮输入。
4. **通信与 fallback 协调。** 同一 TP/EP 通信组按一致的协议执行 warmup、capture 和 replay，不能让某个 rank 根据局部条件任意切换 graph/eager 或跳过 collective。不同 PP stage 可以有各自的 graph，但必须保留 stage 间发送顺序和 batch 对应关系；初版不要求把 PP P2P 一起 capture。

#### 验证与收益判断

先在相同 PP 拓扑下比较 eager 与 graph，验证 logits、KV 更新和 speculative acceptance 一致性，再测 CPU launch 开销、各 stage target forward 时间、端到端 decode 延迟、吞吐及额外 graph/buffer 显存。覆盖 bucket 边界、batch 缩小/重排、不同接受长度、PD 首轮、取消/结束、在途发送、多轮 replay 和带 TP/EP 通信的 MoE。

小 batch decode 的 launch 开销可能更值得优化，但 graph 不消除 TP/EP/PP 通信，也不解决 last stage draft/sampling 的负载不均。非 PP graph 可作为参考基线，不能把不同拓扑下的全部性能差异归因于 graph。以上收益尚未经过多 GPU 压测。

#### 代码位置

- [PP target 禁用 graph：PPExecutor.cc:403](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L403)
- [启用条件中的 pp_size 限制：PyWrappedModel.h:236](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/models/PyWrappedModel.h#L236)
- [verify token 数与 graph role：PyWrappedModel.h:433](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/models/PyWrappedModel.h#L433)
- [capture 前 WORLD barrier：PyWrappedModel.h:50](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/models/PyWrappedModel.h#L50)
- [本地 runner 初始化及 barrier 调用：PyWrappedModel.h:478](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/models/PyWrappedModel.h#L478)
- [按 bucket 捕获：cuda_graph_decode.cc:42](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/cuda_graph/cuda_graph_decode.cc#L42)
- [本地 forward capture 与输出保存：cuda_graph_runner.cc:1993](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc#L1993)

### 默认均匀分层没有计入 last stage 额外负载

默认按 target layer 数量均分，但完整 draft model、lm_head 和 sampling 集中在 last stage，可能同时造成计算瓶颈和 KV 容量瓶颈。

实现提供了自定义分层扩展点，但没有自动按这些工作量平衡。应按各 stage 的实测执行时间和显存占用调整分层，不能仅按 layer 数量判断是否均衡。

代码：[默认分层：pp_layout.py:17](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/config/pp_layout.py#L17)。

## 8. PPExecutionPlan 的转发机制与按需反序列化优化

### 当前实现：首 stage 编码一次，后续 stage 各自解码

每个执行轮次中，Stage 0 构造并序列化 plan。后续 stage 接收原始 payload，反序列化为本地对象，同时保留原始字节供转发使用。中间 stage 不重新序列化本地 plan。

以 PP=4 的 TP root lane 为例：

```text
Stage 0：buildPlan → serializePlan → payload
                                      ↓
Stage 1：接收 payload → deserializePlan → 本地 plan
             └─ 原始 payload 转发 ──────↓
Stage 2：接收 payload → deserializePlan → 本地 plan
             └─ 原始 payload 转发 ──────↓
Stage 3：接收 payload → deserializePlan → 本地 plan
```

接收和转发的关键代码是：

```cpp
// receivePlan()
plan_payload = receiveObject();
auto plan = pp_serialization::deserializePlan(plan_payload);

// processImpl()
if (!isLastStage()) {
    asyncSendPlan(plan_payload, inflight.plan_sends);
}
```

对 PP=N 的一个完整 plan，每轮序列化 1 次、反序列化 N−1 次、stage 间传输 N−1 次。TP 非 root lane 使用精简的 `metadata_only` plan；完整采样信息沿各 stage 的 TP root 传递。

当前反序列化会为 tensor 分配独立的 pinned CPU storage 并复制数据，并不是直接引用 payload 的零拷贝视图。因此，即便不重复序列化，中间 stage 仍承担了不必要的采样字段解析和内存分配成本。

代码：[接收与解码：PPExecutor.cc:153](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L153)、[构造与转发：PPExecutor.cc:1236](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L1236)。

### 第一阶段优化：sampling 继续传输，中间 stage 跳过解析

可以保留现有逐 stage 转发路径，为 payload 增加有长度边界的段。以下是建议格式，不是当前 wire format：

```text
公共 header：version、metadata_only、shutdown
model_input_size   | model_input
sampling_size      | sampling_plan
output_config_size | output_config
其余执行与生命周期字段
```

中间 stage 解析本地执行所需字段，遇到仅供 last-stage root 消费的段时，校验长度后直接移动读取指针，继续转发原始 payload：

```cpp
deserializeModelInput(reader);
reader.skip(sampling_size);
reader.skip(output_config_size);
// 按本地需要解析其他字段。
forward(original_payload);
```

last-stage root 才完整解析 sampling 和 output 配置。实现时应调整序列化版本，并让 `skip()` 校验剩余字节数，不能先构造 tensor 再丢弃。

此方案能减少中间 stage 的 tensor 分配、pinned memory 占用、数据复制和字符串等对象构造；不会减少网络传输量，也不会消除接收完整 payload 所需的内存。更进一步的优化才是将采样专用 payload 从 Stage 0 直接发送给 last stage，或维护增量请求状态。

### finished_request_ids 与 shutdown 的实际位置和消费方

对象结构中，两者的位置不同：

```cpp
struct PPExecutionPlan {
    GptModelInputs model_input;  // 包含 shutdown
    PPSamplingPlan sampling_plan;
    PPOutputConfig output_config;
    std::vector<int64_t> finished_request_ids;
    // 其他字段省略。
};
```

`finished_request_ids` 由 Stage 0 scheduler 在请求进入 `FINISHED` 时收集，经 `ScheduleOutput` 放入 plan。当前只有 last-stage root 消费它，用于清理请求级采样状态：

```cpp
if (isLastStage() && isStageRoot()) {
    for (const auto request_id : plan.finished_request_ids) {
        sampling_states_.erase(request_id);
    }
}
```

因此，中间 stage 可以跳过 `finished_request_ids` 的解析，只需原样转发。该字段不是每个 stage 都需要解析的控制信息；但不能从传输数据中删除，否则 last-stage sampling state 无法按请求结束清理。当前完整 wire payload 将这组 ID 放在 `draft_next_position_ids` 后，以数量和 ID 列表编码。

`shutdown` 则是每个 stage 都需要读取的控制信息。Stage 0 的 `finish()` 调用 `processImpl(empty_output, 0, true)` 发出终止 plan，各 stage 转发并处理该轮后执行：

```cpp
stopped_ = model_input.shutdown;
```

其他 rank 的 `finish()` 在 `stopped_` 为 false 时继续处理上游 plan，收到终止标志后等待在途通信完成。虽然 `shutdown` 在对象中属于 `model_input`，当前 wire format 已将它单独编码到公共 header：

```cpp
w.val<uint32_t>(kVersion);
w.flag(metadata_only);
w.flag(plan.model_input.shutdown);
```

TP 非 root 的 metadata-only plan 也携带该标志，随后它还参与 stage 内的模型输入同步。

| 字段 | 中间 stage | Last-stage root |
| --- | --- | --- |
| `shutdown` | 解析并执行终止流程 | 解析并执行终止流程 |
| `finished_request_ids` | 可跳过解析，保留原始字节转发 | 解析并删除 sampling state |
| `sampling_plan` | 可跳过解析，保留原始字节转发 | 解析并执行采样 |
| `output_config` | 可跳过解析，保留原始字节转发 | 解析并生成对应输出 |

代码位置：

- [plan 字段：PPTypes.h:86](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPTypes.h#L86)
- [last stage 清理状态：PPExecutor.cc:1253](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L1253)
- [终止流程：PPExecutor.cc:1189](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L1189)
- [当前 wire format：PPSerialization.cc:357](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPSerialization.cc#L357)

优化后应验证：中间 stage 跳过解析后仍原样转发 payload；last stage 的采样字段保持一致；仅有结束 ID 的空 plan 仍清理状态；shutdown 在 TP root 和非 root 上都能正常传播；错误的段长度能被拒绝。这里记录的是优化建议，尚未修改执行代码。

## 9. 将 PP 通信接收、发送与执行主流程解耦

### 目标与当前耦合点

接收、转发以及通信完成后的 buffer 回收可以从执行线程拆出去；执行主流程只在消费输入或复用 buffer 时建立必要依赖。通信解耦不能消除 forward 对本轮输入的依赖，也不能让同一请求在上一轮结果未提交时随意进入下一轮。

当前代码已经使用异步发送，但仍有以下耦合：

- `receiveObject()` 依次接收长度和 payload，并等待两次接收完成，执行线程会阻塞。
- 收到 plan 后，要经过本 stage 的 TP 输入同步和 slot 复用等待，才转发给下一 stage。
- Stage 0 在 ring 指定的位置接收 result，接收推进与执行轮次绑定。
- send ticket 及其持有的 tensor 随 slot 复用回收，通信完成后仍可能继续占用内存。

代码：[对象收发：PPExecutor.cc:131](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L131)、[plan 同步与转发：PPExecutor.cc:1236](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L1236)、[result 接收时机：PPExecutor.cc:1413](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L1413)。

以下内容是后续改造建议，尚未实现。

### Plan：后台接收、按序转发、按需解析

通信线程可以持续接收下一轮 payload，保留原始字节供发送端使用，并将本 stage 所需的解析结果放入有界 ready queue：

```text
收到 payload
    ├─ 原始字节进入有序发送队列，转发下一 stage
    └─ 按需解析 → ready queue

执行线程
    └─ 取本轮输入 → TP 同步 → 等待本轮 activation → forward
```

原始 payload 的转发不依赖本 stage 的 forward，也不需要等待采样字段解析。结合第 8 节的按需反序列化，可以让下游提前准备输入，同时让本 stage 提前接收下一轮 plan。

必须给 ready queue、待发送 payload 和在途 batch 设置容量或字节预算，避免上游推进过快而堆积 CPU/pinned memory。发送完成前保留 payload 的所有权。

这里的“提前转发”必须与后文的有序消息协议一起实现，不能只把现有 `asyncSendPlan()` 移到任意后台线程。

### TP 非 root 的模型输入仍由本 stage 同步

跨 PP 的精简 plan 不代表 TP 非 root 执行时缺少模型输入。现有实现中：

| 输入 | TP 非 root 的来源 |
| --- | --- |
| tokens、lengths、KV block tables 等模型输入 | 本 stage 的 TP root，通过 `tpSyncModelInputs()` 同步 |
| hidden/residual 等 activation | 上一 stage 对应 TP rank 的 PP P2P 传输 |
| sampling plan | 非 root 不执行采样，不需要完整采样计划 |

通信解耦后仍要维持这些依赖。第一阶段可以保留 TP collective 在执行线程上的确定顺序，先独立处理 CPU plan 收发；如进一步预取 TP 输入，则需要独立输入 buffer 和明确的就绪事件。

### Activation：提前接收，用 GPU event 建立依赖

可以在通信 stream 上预先接收 activation，在计算 stream 真正消费数据时等待对应 event：

```text
接收 stream：接收 activation A → A_ready event
计算 stream：等待 A_ready      → forward A → A_done event
发送 stream：等待 A_done       → 发送 A 的输出
```

这些等待可以是 GPU stream 间的依赖，CPU 不必同步等待整个数据传输。提前接收需要先知道 tensor 的 shape、dtype，并准备独立的接收 buffer；可以通过提前发送 activation metadata 或可复用 buffer pool 实现。

解耦不能直接取消当前的 `clone()`。如果下一轮 forward 会覆盖 model output buffer，就仍需要 snapshot。只有让 forward 输出到通信持有的独立 buffer，并确保发送完成后才复用，才能安全减少拷贝。发送和接收 buffer 都必须覆盖实际 GPU 读写的生命周期，不能仅凭 CPU 函数返回判断可以释放。

### Result：独立接收，单线程提交请求状态

Stage 0 可以使用独立的 result 接收逻辑：

```text
result 到达 → completion queue
                    ↓
scheduler/执行线程在安全点消费
                    ↓
更新 stream → 清除 inflight → 允许下一轮调度
```

这样可以提前排空传输通道，不必等 ring 转到某个位置才开始接收。但仅把 result 放入 completion queue，并不会自动缩短请求下一轮调度的等待；主线程还需要在调度前等合适的安全点及时消费完成事件。

接收线程应负责接收和入队，不直接修改 scheduler 的请求状态、KV 分配或请求生命周期。保持单一状态提交线程，可以避免与调度、取消和 KV 回收竞争。result 必须准确关联原始 inflight batch；如果允许接收与执行独立推进，建议显式携带 batch ID，并按既定顺序提交状态。

### 消息顺序是解耦前必须处理的协议约束

当前 plan 和 activation metadata 都通过 CPU 对象发送，使用同一条 GLOO lane，匹配依赖对称的调用顺序，例如：

```text
plan A → activation metadata A → plan B → activation metadata B
```

如果后台线程把 plan B 提前插到 activation metadata A 前，而接收端仍按旧顺序读取，就可能把一种消息当成另一种消息解析。因此应采用以下一种方式：

- 每条连接使用单一发送队列及统一的接收分发逻辑，保持协议定义的消息顺序；或者
- 分离控制通道，并使用 `message_type + batch_id` 等信息将 plan、activation metadata 和 result 路由到对应队列。

如果同一通道允许消息交错，仅添加 batch ID 不够，还需要消息类型、长度边界和相应的分发逻辑。

GPU P2P 与 TP collective 的提交也必须保持跨 rank 一致的顺序。多个线程不能任意交错调用同一 communicator。CUDA stream/event 解决数据就绪依赖，不能替代通信协议的顺序约束。

代码：[按设备选择 GLOO/NCCL：PPTransport.cc](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPTransport.cc)、[activation metadata 与 tensor 发送：PPExecutor.cc:183](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L183)。

### 实施顺序与验证重点

1. 先明确消息边界、顺序和 batch 关联方式，再将 CPU plan/result 收发队列化。
2. 在有序协议下提前转发 plan，并对中间 stage 跳过采样专用字段解析。
3. 让主线程在安全点消费 completion queue，保持请求、取消及 KV 状态的单线程提交。
4. 引入 activation 预接收、event 依赖和有容量限制的 buffer pool。
5. 在通信完成时及时回收 buffer，再评估是否能够减少 activation snapshot 拷贝。

需要覆盖正常 batch、空 plan、fake batch、取消、请求结束以及 shutdown。终止消息不能越过仍需执行的旧 batch；停机应按序停止提交、排空在途结果和通信，再释放模型、buffer 与 communicator。验证指标包括主线程阻塞时间、plan 到达与 forward 启动间隔、completion queue 消费延迟、队列字节数及峰值显存。

## 10. Verify 采样将完整 token history 展开 K+1 份

### 当前行为与代价

这与第 4 节的全量 plan 传输是两个不同层次的开销：即使将历史保留在 last stage、plan 改为增量，last stage 内部仍会为 verify 采样展开历史。

`allocateSamplerInputs()` 为 verify 分配并清零约 `[B × (K+1), L+K+1]` 的 pinned CPU token tensor，其中 B 为请求数、K 为投机 token 数、L 为本批最大历史长度。`fillSamplerInputs()` 对每个 verify 行复制该请求的完整历史，并追加该行可见的 draft prefix：

```text
verify row 0：committed history
verify row 1：committed history + draft[0]
verify row 2：committed history + draft[0:2]
……
verify row K：committed history + draft[0:K]
```

随后 `Sampler::forward()` 将整个 token tensor 搬到 GPU。以 B=64、L=128×1024、K=4、int32 为例，仅展开的历史主体就约 160 MiB/轮，不包含 plan、额外 buffer 或概率矩阵。异步 H2D 可以改变等待位置，但不能消除这部分传输量。

### 改进建议与验证

1. 为不使用历史相关 penalty/processor 的请求增加轻量采样路径，避免仅因通用接口而物化完整历史。
2. 通用路径考虑“共享 committed history + 每个 verify 行的短 draft prefix”，改造消费者按逻辑索引读取；不能只把 tensor 换成 view 而不检查算子的连续布局和写入要求。
3. 对频次惩罚等维护增量统计；no-repeat-ngram、grammar 等按各自语义处理，不能一概删除历史。
4. 暂时保留原算子的路径可先复用分桶 buffer，并在保证未写区域不会被读取的前提下减少清零。

验证不同历史长度、K、batch 内长度差异和 processor 组合下的采样语义；分别统计 pinned 分配/清零、CPU memcpy 时间、H2D 字节数及峰值显存。该建议不涉及改变 verify 后 accepted tokens 的 CPU/GPU 消费路径。

代码：[采样输入分配：PPBatchStreamProcessor.cc:343](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.cc#L343)、[逐 verify 行复制历史：PPBatchStreamProcessor.cc:435](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.cc#L435)、[完整 token tensor H2D：Sampler.cc:130](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/models/Sampler.cc#L130)。

## 11. 增加面向 PP 流水线占用的 microbatch 调度

### 当前行为与限制

`PPScheduler::evaluateRunningStreams()` 把所有非 inflight、仍为 RUNNING 的请求加入本轮 batch；本轮选中的请求随后全部标记为 PP inflight。它没有根据流水线占用，主动把 ready 请求分成多个 microbatch。

`pp_size + 1` 个 ring slot 只是允许维护多个在途 batch，不会自动把一个 batch 拆开。例如 PP=4、只有一组可装进单批的请求且没有新请求补入时，整组请求作为 A 逐 stage 执行，等结果提交后才能再次调度；推进空 plan 不会产生新的有效计算 batch，可能留下流水线空洞。

### 改进建议与验证

- 将 ready 请求数、在途 batch 数、各 stage 实测耗时和目标延迟纳入本轮 batch 大小决策，让多个独立请求组形成在途 microbatch。
- 除请求数外，引入 token/计算成本预算；decode 的上下文长度不同，attention 成本也不同。
- 与 CUDA Graph bucket 联动，避免拆出大量形状不合适、GEMM 效率很低的小批次。
- 保留请求的 inflight 约束；未选中的 ready 请求留待后续轮次，并补充有界公平性。

不能固定假设“拆成 PP 份最优”：拆小会增加通信次数并降低单次计算效率。单个 query 的自回归依赖也不能通过请求级 microbatch 消除。先做 microbatch 大小扫描，比较各 stage 空闲比例、吞吐、TPOT、排队延迟和 fake/empty batch 比例。

代码：[ready 请求收集：PPScheduler.cc:88](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/engine_base/schedulers/PPScheduler.cc#L88)、[调度与 inflight 标记：PPScheduler.cc:308](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/engine_base/schedulers/PPScheduler.cc#L308)、[PP 执行入口：NormalEngine.cc:989](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/NormalEngine.cc#L989)。

## 12. MTP draft chain 改用增量 TP 输入同步

### 当前行为与代价

`runDraftStep()` 的后续 MTP draft decode 循环中，每步都调用 `tpSyncModelInputs(draft_input, ...)`。通用同步不仅处理新 token，还处理 shape hints、KV block table/group 信息、sequence lengths、position、hidden，以及非 root 的 tensor 分配和 pack/unpack。

需要注意：`tpSyncModelInputs()` 已经按 CPU/GPU 分类做 packed broadcast，不能把“合并多个小广播”当作尚未实现的主要优化。这里应减少的是每步重复处理、复制和传输的不变字段。

### 改进建议与验证

在一段 batch 布局稳定的 draft chain 开始时同步基础状态，后续采用专用的增量同步协议：

- KV table、batch layout 等没有变化时保留在各 rank，不重复广播。
- 每步同步新 token 及确实变化的数据；可以确定性更新的长度和 position 在各 rank 本地推进。
- hidden 能否使用各 rank 本地模型输出，需要按模型和 TP 布局验证，不能直接删除广播。
- batch 重排、cache 映射或布局变化时，用版本号/epoch 失效并重新同步基础状态。
- 复用有生命周期保障的同步 buffer，减少非 root 分配及 pack/unpack 拷贝。

该优化可以在现有 accepted-token CPU 处理流程不变的情况下实施。验证 K、TP 数、跨机 TP、长 block table 和不同 position 编码；测每个 draft step 的同步时间、传输字节数和分配次数。所有 rank 必须维持一致的通信顺序。

代码：[每步 draft 的全量 TP 同步：PPExecutor.cc:1132](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L1132)、[通用模型输入同步：ModelTypes.cc:268](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/models/ModelTypes.cc#L268)、[参与广播的字段及打包：ModelTypes.cc:600](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/models/ModelTypes.cc#L600)。

## 13. 可选输出按请求裁剪，避免整批大 tensor 回传

### 当前行为与适用范围

`gatherOutputConfig()` 对 `return_logits`、`return_hidden_states` 等标志按 batch 做 OR。只要有请求需要对应输出，`fillOptionalOutputs()` 就将相应整批 tensor 搬到 CPU，经 result 通道回到 Stage 0，之后才按请求切片。

因此少量需要额外输出的请求，可能使整个 batch 承担 D2H、序列化及网络传输成本。该问题主要针对普通非 SP 路径：当前 SP 已关闭其中多种额外输出，不能把它当成所有 MTP 请求的固定开销。Prompt logits 已有 per-request 配置，也应与这些 batch-level 标志区分。

### 改进建议与验证

- 引入 per-request 输出 mask/index，在 last stage 先选出需要返回的行或 token 区间，再 D2H 和序列化。
- 结果协议携带对应的 request/sequence/row 映射；Stage 0 不能再按整批 dense tensor 的位置直接切片。
- 大输出可使用单独通道或有界队列，避免阻塞小 token result；拆分时必须维持请求完成、取消和输出齐备的语义。
- 同时检查 loss 等计算能否按请求选择执行，而不只是裁剪传输结果。

用“一个请求要求 logits/hidden，其余请求不要求”的混合 batch 验证，比较实际返回值、D2H/网络字节数、峰值内存和其他请求的延迟。

代码：[batch-level 输出标志：PPBatchStreamProcessor.cc:231](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.cc#L231)、[可选输出整体 D2H：PPBatchStreamProcessor.cc:681](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.cc#L681)。

## 14. 减少多层辅助 hidden 的逐 stage 累积复制与转发

### 当前行为与适用范围

对于启用了多层 target 辅助特征捕获的模型，`begin_pp_aux_hidden_capture()` 为当前 stage 分配新的特征 buffer，复制上游已经收集的 prefix，再加入本 stage 特征。中间 stage 将 `mtp_aux_hidden_states` 与主 hidden/residual 一起转发，发送时还会执行 snapshot clone。

因此，较早 stage 产生的辅助特征可能经过多轮 buffer 复制和中继传输。该成本取决于捕获层位置、数量和 token 布局，并非所有 MTP 配置都存在；估算 activation 流量时不能只计算一份主 hidden。

### 改进建议与验证

先做低风险优化：无新增辅助特征的 stage 直接透传已有 tensor，并明确其 ownership 与完成事件；有新增特征时评估分块保存/传输，减少对既有 prefix 的重拷贝。不要在修改生命周期前直接移除 snapshot。

较大的改造是把辅助特征和主 activation 分流，评估由产生特征的 stage 直接发往 last stage，按 batch/layer ID 组装。这可减少逻辑中继，但也会增加 last stage 的接收压力和协议复杂度，收益取决于物理网络拓扑，不能假设跨节点总流量必然按相同比例下降。

验证全局 layer 顺序、各 batch 隔离、在途 buffer 复用，以及 CP 下 rank-local token 顺序和 padding；统计每条 PP 链路的主 activation/辅助特征字节数、复制耗时及显存峰值。

代码：[辅助特征 prefix 分配与复制：module_base.py:255](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/models_py/model_desc/module_base.py#L255)、[Qwen3Next 辅助特征捕获与转发：qwen3_next.py:1823](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/models_py/model_desc/qwen3_next.py#L1823)、[发送 snapshot：PPExecutor.cc:183](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L183)。

## 15. 用 per-slot buffer 生命周期减少下一轮 forward 前的 CPU 等待

### 当前同步边界

`waitLastRun()` 调用 `last_compute_done_.synchronize()`，会阻塞 CPU。`processImpl()` 在下一轮 target forward 前调用它，随后 `releaseAllModelBuffers()` 释放 executor、target 和 draft 持有的 buffer。

这不意味着 CPU 完全不能提前准备下一批：当前 plan 构建/接收、TP 输入同步及 activation 接收已经发生在等待前。被挡住的是后续模型 forward 内部的输入准备和 GPU 操作提交。

```text
CPU：准备 B 的 plan / 接收 activation
          → 等待 A 的 GPU 完成
          → 释放旧 buffer
          → 准备并提交 B 的 forward
```

同步具有正确性目的。例如 A 的异步 H2D 尚未读完 pinned CPU buffer，CPU 不能提前修改或释放它；Python 模型也可能复用跨轮 workspace。不能直接删除 `synchronize()`。

### 改进建议与收益边界

将必要的输入、临时 tensor 和 host buffer 所有权改成 per-slot，记录实际消费者的完成事件并延迟回收；审查模型内部共享 workspace 后，再允许 CPU 提前准备、提交下一轮。

```text
CPU：提交 A → 准备并提交 B → 准备 C
GPU：执行 A ──────────→ 执行 B ─────→ 执行 C
       A 的 buffer 保留至其消费者完成后才回收
```

这不是要求 A、B 在同一个 GPU 上同时计算。同一 compute stream 可以保持 GPU 操作顺序，跨 stream 使用 event 建立依赖；但 stream 顺序不能替代 CPU 内存、对象及共享 workspace 的生命周期管理。若 buffer 还被 P2P send 或其他 stream 消费，应等所有消费者完成，而不只是 target forward 结束。

先测 `executor.pp.wait_last_batch` 的等待时间，以及 A 完成到 B GPU 开始之间的间隔。如果走到等待点时 A 已经结束，或 B 已被通信/其他依赖限制，这项优化收益可能很小。验证慢 H2D、慢发送、连续多轮、graph 输出复用、空 batch 与 shutdown，并限制在途 buffer 总量。

代码：[CPU 等待及 buffer 释放：PPExecutor.cc:549](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L549)、[下一轮 forward 前的等待位置：PPExecutor.cc:1303](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L1303)。

## 16. [可靠性，高优先级] PP 通信等待缺少有效的故障收敛期限

### 当前行为与风险边界

PP NCCL group 和配套的 GLOO group 都以 `timeout=timedelta(days=36500)` 创建，约为 100 年。`PythonP2PWork::wait()` 直接调用 Python work 的 `wait()`，这一层没有额外的单次等待期限或取消机制。

风险主要是某个 stage 仍存活但执行线程停滞、异常分支漏发消息，或各 rank 进入不匹配的通信序列。下游可能等待 plan/activation，Stage 0 等待 result，无法依靠合理的通信超时及时结束故障。进程退出或连接断开可能由底层提前检测，因此不能表述成“所有故障都必然等 100 年”。

正常停机路径同样受此影响：`NormalEngine::stop()` 设置停止状态后 join 执行线程，但修改 `running_` 本身不会唤醒阻塞中的通信 wait。正常 shutdown sentinel 和 drain 协议不能替代故障场景下的超时退出机制。

### 改进建议与验证

- 分别配置初始化、运行期通信和停机 drain 的期限，避免用极长 process-group timeout 作为唯一边界。
- 增加独立的 stage 进度 watchdog，记录最后推进的 batch、phase、peer，区分慢计算、慢通信与失去进展。
- 不可恢复的协议/设备错误应触发流水线级失败收敛，明确终止受影响请求，并通过可执行的 communicator/process teardown 或外部监管路径退出。
- 不要简单捕获异常后跳过 batch 继续执行；未经恢复协议验证，这会进一步破坏消息和 collective 匹配。

故障注入应覆盖：保持进程存活但停止某个 stage 的执行进度、遗漏匹配发送、进程退出、连接故障，以及等待期间发起 stop。验证失败传播、请求错误、停机时限和资源释放。当前仅确认源码上的等待边界问题，未运行多 GPU 故障注入。

代码：[PP NCCL/GLOO group 超时：collective_torch.py:378](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/models_py/distributed/collective_torch.py#L378)、[P2P wait：ExecOps.cc:719](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/models_py/bindings/core/ExecOps.cc#L719)、[停止与 join：NormalEngine.cc:755](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/NormalEngine.cc#L755)。

## 17. Grammar 冷编译阻塞 last stage 执行线程

### 当前行为与影响

Stage 0 的 `GenerateStream` 创建时已经初始化过 logits processor，但 grammar backend/cache 是进程内对象，不能假设 last stage 因此已经具有编译结果。last stage 首次处理请求时，会在采样路径中重新创建 processor/matcher：

```text
target forward → sampleTokens → initSamplingStates
    → initLogitsProcessors → createMatcherFromKey
    → 等待 grammar 编译结果 → 采样 / draft / 发送结果
```

`XGrammarBackend` 已有线程池、缓存、编译去重、队列限制和超时，但 `compileNow()` 的调用方仍通过 `future.wait_for()` 等待。不能把“编译在线程池执行”理解为“last stage 执行线程不会被阻塞”。

在 last-stage cache miss 时，新 grammar 请求可能拖慢同批普通请求，并延迟后续 batch，使压力向前传递。已有缓存命中路径不应计入同等冷编译成本；实际影响需要按 schema 复杂度和命中率测量。

### 改进建议与验证

将首次请求注册与模型执行分开，提前向 last stage 传递 grammar/config 并异步预编译；以 processor ready 作为实际 batch admission 的条件之一，避免进入流水线后才等待编译。编译结果可共享，但请求 matcher 的可变状态仍应独立，并处理注册后取消、失败和清理。

保留现有编译并发、队列和缓存限制，不应通过无界预编译换取低延迟。验证冷/热 schema、编译超时和取消；混合新 schema 请求与普通请求，观察普通请求 TPOT/p99、last-stage 空闲/阻塞时间及编译队列长度。

代码：[Stage 0 processor 初始化：GenerateStream.cc:136](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/engine_base/stream/GenerateStream.cc#L136)、[last stage 采样状态初始化：PPBatchStreamProcessor.cc:261](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPBatchStreamProcessor.cc#L261)、[编译等待：XGrammarBackend.cc:490](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/engine_base/grammar/XGrammarBackend.cc#L490)、[matcher 创建入口：XGrammarBackend.cc:532](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/engine_base/grammar/XGrammarBackend.cc#L532)。

## 18. Prefill token budget 是软约束，不能充当单次执行的硬上限

### 当前预算语义

`PPScheduler::fitsPrefillTokenLimits()` 对空 batch 的第一条请求，在其 context length 小于模型 `max_seq_len` 时直接返回 true。这意味着它可能超过 `max_batch_tokens_size` 而通过这一层检查。其他输入长度、KV 容量等检查仍存在，不能说请求完全不受限制。

另外，`max_batch_tokens_without_cache` 检查的是已经接纳的 token 数是否达到阈值，而不是加上当前 candidate 后是否超限，因此更接近接纳软水位。

例如，空 batch、token budget 为 16K、单请求为 100K 且其他检查通过时，不能指望该 budget 自动把请求切成多个 16K chunk。大 prefill 在 PP 下可能增加 activation、辅助 hidden 和 snapshot 峰值，并让某个 stage 长时间处理同一批，造成后续短请求的队头阻塞。这是预算语义与执行能力限制，不等于已证明必然 OOM，也不应一概视为 PP 独有的问题。

### 改进建议与验证

明确区分 batching 软预算、单次执行硬安全上限和 chunked prefill 能力。如果引入硬上限，应同时提供分块或明确拒绝路径，不能让超大请求永久留在 waiting。

硬上限需按各 stage 的实际执行能力评估，包括 attention workspace、辅助特征、在途 snapshot 和可用显存。CP 场景必须考虑 rank-local padding 与实际通信布局，不能仅用总 token 数除以 CP 数作为峰值估计。

验证接近/超过软预算的首请求、多个候选累积越过水位、长短请求混合，以及不同 CP/PP 配置下的峰值显存、单批执行时间和短请求排队延迟。

代码：[无缓存 token 软水位检查：PPScheduler.cc:240](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/engine_base/schedulers/PPScheduler.cc#L240)、[首条请求放行：PPScheduler.cc:264](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/engine_base/schedulers/PPScheduler.cc#L264)。

## 19. [可观测性] 聚合指标不能代表完整流水线的阶段耗时

### 当前缺口

`reportResultMetrics()` 只在 first-stage TP root 上报。代码明确留下 TODO：中间/末端 stage-local forward、tail draft、TP sync、EPLB 和 grammar 指标尚待补充。

- 聚合的 `model_forward_us` 主要来自 first stage 的本地调用，不能当作整条 PP 的 target GPU 时间。
- result 返回了 last stage 的部分 sampling 计时，但不包含完整 tail 执行分解。
- SP phase timings 保留零占位，零不表示该阶段实际没有耗时。
- 包围 `model_->forward()` 的 CPU wall time 不天然等于 GPU kernel 执行时间。

已有 profiling scope 可用于 trace，但不能替代常态运行时的 stage-level 指标。缺少这些信息容易把 last stage 的 draft/grammar 阻塞误判为通信或 Stage 0 调度问题，影响分层和 batch 大小决策。

### 改进建议与验证

每个 stage 分别记录 plan/activation 等待、输入准备与 TP sync、target GPU 时间、sampling/grammar、draft GPU 时间、发送及 buffer 等待、空闲时间。CPU 时间和 GPU event 时间明确区分，GPU 计时应异步回收，避免埋点本身引入全局同步。

按 stage/rank 聚合指标；batch ID 用于 trace 关联而非高基数监控标签。跨主机 trace 关联还应考虑时钟偏差，不能直接相减未校准的主机时间戳。以可控的 stage 延迟注入和 trace 对照验证归因，检查吞吐是否重复计数、零占位是否被误读，并测埋点开销。

代码：[指标上报及 stage-local TODO：PPExecutor.cc:1494](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L1494)、[采样路径计时：PPExecutor.cc:879](https://github.com/alibaba/rtp-llm/blob/6446ec7e17d8579fba56399cfc7ccedbb72b94f8/rtp_llm/cpp/normal_engine/pipeline/PPExecutor.cc#L879)。

## 不应直接认定为 bug 的事项

缺少显式 batch ID 不等于 activation 必然串 batch。有序 P2P、对称的 send/recv 调用以及一致的执行轮次可以保证匹配。显式 batch ID、消息类型和 tensor 数量校验主要增强故障定位与协议检查，不能仅凭缺少 tag 就断言当前实现乱序。

类似地，activation clone 本身是保证异步发送安全的一种合理方案；应优化其分配与回收方式，而不是直接移除拷贝。共享 KV tag 取 owner stage 容量最小值也有明确的正确性目的，不应把它单独列为错误。

## 建议处理顺序与验证

最新补充的第 16 节通信超时与停机故障收敛应列入高优先级可靠性工作，与请求错误隔离一起处理。第 19 节的 stage-level 指标是性能归因的基础；第 17 节优先用冷 grammar 混合负载验证，第 18 节则先明确软预算与硬上限的产品语义，再决定分块或拒绝策略。这些结论来自代码检查，尚未完成多 GPU 故障注入或性能压测。

| 顺序 | 工作 | 建议验证 |
| --- | --- | --- |
| 1 | 将 beam 等不兼容请求转换为单请求错误 | 不兼容请求与正常请求并发提交，确认正常请求继续完成、所有 stage 保持可用 |
| 2 | 增加 PDFUSION prefill 有界公平性 | 长 decode 占满流水线时持续注入 prefill，检查 admission 等待时间、TTFT 和超时清理 |
| 3 | 将 DSpARK q 保留在 last stage | 验证 batch 重排、混合 PD 首轮、取消和 proposal epoch；测 CPU/GLOO 流量 |
| 4 | 拆分全量 plan 与增量请求状态 | 验证首次注册、后续更新、结束清理和历史一致性；测长上下文序列化开销 |
| 5 | 优化 activation buffer 回收 | 测发送完成至 buffer 释放的间隔及峰值显存，验证复用不覆盖在途数据 |
| 6 | 解耦逻辑 cache topology 与 Stage 0 物理层 | 覆盖后续 stage 独有 tag、部分 group set、PD 两端不同 PP 切分 |
| 7 | 为 PP target verify 接入 rank-local CUDA Graph，并单独优化 stage 负载平衡 | 在相同 PP 拓扑比较 eager/graph 的正确性、延迟与显存；覆盖 TP/EP collective、bucket 切换和在途输出复用；分别测 target、draft、sampling、通信耗时 |

新增性能项建议先按下表测量，再依据实际瓶颈插入上述实施顺序；不以未压测的理论开销代替收益排序。

| 测量优先级 | 新增工作 | 建议验证 |
| --- | --- | --- |
| 1 | 减少 verify 的 K+1 份完整历史展开 | 分离 CPU 分配/清零/memcpy 与 H2D 开销，覆盖历史相关 processor 的正确性 |
| 2 | draft chain 增量 TP 同步 | 对比每步同步字节数、分配次数及延迟；验证 KV/layout 变更和跨 rank 顺序 |
| 3 | PP-aware microbatch 调度实验 | 扫描 batch 大小，比较各 stage 空闲比例、吞吐、TPOT 和排队延迟 |
| 4 | 按请求裁剪可选输出 | 混合输出需求下检查映射正确性、D2H/网络流量及其他请求延迟 |
| 5 | 辅助 hidden 透传与复制优化 | 测各链路流量、复制次数和峰值显存；覆盖多捕获层及 CP 布局 |
| 6 | per-slot buffer 与下一轮 forward 提前提交 | 先确认 wait_last_batch 和 GPU 批次间隙，再验证全消费者完成后回收及内存上限 |
