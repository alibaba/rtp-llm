# GlobalQueueCoordinator 简化与 750P/750D 性能验证

2026-09-11。代码与功能验证已完成；性能没有全部通过。

- 功能测试：1491 个，0 失败、0 错误、1 跳过。
- BATCH：3000 QPS 通过；10000 QPS 有改善，但未稳定达到吞吐／延迟门槛。
- NON_BATCH：10000 QPS 两次均通过；3000 QPS 吞吐达标，但两次客户端长尾失败，不能宣称没有延迟回归。
- 所有下表中的正式测量请求都返回成功，且覆盖全部 750P/750D 路由；失败的是性能断言。另有一次冷启动诊断超时，单独记录，不混入正式测量。

## 提交与行为

| 提交 | 内容 |
| --- | --- |
| `a856c8d375` | 保留有界扫描；移除工作区中的 PlanningBudget 和 endpoint 冲突排除机制；容量判断归选路／入队所有 |
| `263b6aa101` | 用统一 PlacementWaitQueue 替代 endpoint／selector 两套等待逻辑；删除后继容量探测与同 worker 栅栏；补充行为测试 |
| `2f35fb4c86` | 发压不再跳过迟到时隙；客户端延迟计入计划发送至实际发送的等待；增加按时间预热参数 |

FIFO／PRIORITY 保证调度尝试顺序，不保证不同资源需求的请求严格按到达顺序进入 worker。
失败请求可以等待，后续请求可以尝试同一 worker。已等待的容量域每次先放行一个重试；
成功或取消归还重试机会，失败则等待新事件。这一规则集中在等待队列，不需要 Coordinator
探测下一个请求、维护暂停后继、保存冲突历史或再算一份全局容量账本。

10000 请求积压回归保留原门槛：一次 Decode 空位释放只新增 2 次选路（成功一次、确认满一次）；
重复心跳新增 0 次选路，没有丢请求或重复预留资源。

详细规则见 [调度说明](../../global-queue-coordinator.md)。

## 测试边界与环境

- Apple M5 Pro，15 逻辑 CPU，48 GiB 内存，macOS 26.3.2 arm64；Corretto Java 21.0.8。
- 每次测试 JVM 固定 `-Xms8g -Xmx8g`；默认 15 planner，client、Master 与 mock 在同一机器、同一测试 JVM 中。
- BATCH / FIXED_WINDOW：10 ms、最多 16 条；750 个真实 Netty Prefill mock RPC server，750 个逻辑 Decode endpoint。
- NON_BATCH / SINGLE：750P/750D 路由端点；真实 client→Master gRPC→路由响应，不发送 EnqueueBatch。
- 两种模式都使用仓库现有真实长度／脱敏 token 形态的 payload 模板，不替换成小请求。
- 此处 E2E 是调度响应链路，不包含 GPU 推理或输出流读取。BATCH 的模拟状态完成可能早于 ACK；日志保留 `terminal_without_ack_count`，不能把全部请求都称为经过完整 ACK。
- 每档按目标速率预热 10 秒，再测量 10 秒：3000 档 30000 条，10000 档 100000 条。各模式依次执行 3000、10000 档。
- 门槛：client 与 Master 完成 QPS ≥目标的 98%，客户端／服务端 P99 ≤250 ms，BATCH 等待 P99 ≤50 ms。未放宽延迟门槛。
- 客户端 P99 从**计划发送时间**计算，包含发压线程迟到；服务端 P99 使用服务端时钟。两者不可混用。

基线是开始任务时的未提交工作区（HEAD `5d5ea8ed2a` 加原工作区 diff），不是裸 HEAD。
已在 `/tmp/flexlb-queue-simplify-20260911/before` 保存源文件，在 `before.patch` 保存 diff，
并在独立 `baseline-source` 目录测试。原有 ArithmeticFormula、WhitelistMetricsFilterConfig
及其测试改动在前后两边完全相同，保留为未提交改动，不混入本次提交。

## 相同参数的正式前后对照

以下均为实际发出约 3000／10000 QPS 的测量，默认 15 planner，10 秒预热。
QPS 包含全部请求完成所需的尾部排空时间。单次样本不代表生产容量上限。

| 模式 | 目标 QPS | 版本 | client 完成 QPS | Master 完成 QPS | client P99 ms | Master P99 ms | 请求错误 | 门槛 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| BATCH | 3000 | 基线 | 2996.8 | 3002.6 | 165.289 | 54 | 0 | 通过 |
| BATCH | 3000 | 简化后 | 2996.8 | 3001.0 | 161.380 | 115 | 0 | 通过 |
| BATCH | 10000 | 基线 | 8269.7 | 8280.7 | 2108.132 | 2099 | 0 | 未通过 |
| BATCH | 10000 | 简化后 | 8834.7 | 8844.6 | 1112.342 | 904 | 0 | 未通过 |
| NON_BATCH | 3000 | 基线 | 2999.9 | 3000.3 | 129.036 | 15 | 0 | 通过 |
| NON_BATCH | 3000 | 简化后 | 2999.9 | 3000.9 | 571.490 | 52 | 0 | 客户端 P99 未通过 |
| NON_BATCH | 10000 | 基线 | 9998.4 | 10000.9 | 70.974 | 28 | 0 | 通过 |
| NON_BATCH | 10000 | 简化后 | 9998.7 | 10000.8 | 47.122 | 18 | 0 | 通过 |

NON_BATCH 简化后按完全相同参数复测，保留失败，不挑选最好样本：

| 目标 QPS | client 完成 QPS | Master 完成 QPS | client P99 ms | Master P99 ms | 请求错误 | 门槛 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3000 | 2999.8 | 3000.8 | 690.538 | 69 | 0 | 客户端 P99 未通过 |
| 10000 | 9997.8 | 10003.7 | 171.156 | 75 | 0 | 通过 |

NON_BATCH 3000 档服务端延迟低于客户端延迟，发压迟到对长尾有贡献；但本次没有充分隔离
客户端与服务端，也没有对该档单独完成因果定位，不能把回归直接归咎于环境。

## 其他测量与诊断

1. 原发压器把迟到的时隙直接跳过。标称 10000 QPS 时实际只发约 6134 QPS，完成约 6126 QPS。
   因此它的低延迟不能用作“已验证 10000 QPS”的证据。
2. 修正发压器、仅按每 P 16 条预热的初轮 BATCH 10K：基线完成 9290.3 QPS / client P99
   773.625 ms；简化后完成 9822.2 QPS / 190.542 ms，单轮通过。固定 10 秒预热后未稳定
   复现，所以不以这轮作为最终达标结论。初轮 BATCH 3000 也有长尾失败，原始日志全部保留。
3. 4 planner 的额外诊断更差：BATCH 10K 完成 6694.7 QPS / client P99 4914.894 ms。
   该轮开启 JFR，因此也不能作为无 profiler 的严格 A/B。未采用该调参，默认仍为 15。
4. 4-planner JFR 的 1422 个执行样本中，705 个栈包含 DefaultRouter.selectRole、546 个包含
   CostBasedPrefillStrategy.evaluateCandidates；观察到约 550 ms 的 G1 Full Compaction Pause。
   栈计数会重叠，且记录包含准备阶段。证据指向继续研究路由候选计算、payload 分配与 GC，
   不支持继续给 Coordinator 加特殊分支。采样摘要保存在 `profile-summary.json`。
5. 另一次新 JVM 不经过前面的 3000 档、直接以 10K 预热的诊断失败：100000 条中 54006
   成功、45994 deadline 超时。它是冷启动失败，未计入上述有预热／前置档的正式测量。
   完整日志与 JFR 保存在 `/tmp/flexlb-queue-simplify-20260911/`。

本轮没有修改生产线程默认值、RPC 序列化或选路算法来追求某次压测过线。
尚未解决的是 BATCH 10K 稳态门槛和 NON_BATCH 3000 客户端长尾；需要继续定位，不能标记性能验收全部通过。

## 复现与证据

在 flexlb 根目录执行：

```bash
./tools/run_queue_performance.sh /tmp/flexlb-queue-perf
```

脚本即使 BATCH 失败也会继续 NON_BATCH，保存每种模式的完整日志与退出码。
默认与正式对照参数一致；可用 `FLEXLB_PERF_HEAP`、`FLEXLB_PERF_DURATION_MS`、
`FLEXLB_PERF_WARMUP_MS`、`FLEXLB_PERF_TARGET_QPS` 覆盖，但更改参数后应重新做同条件基线。

功能回归命令：

```bash
./mvnw -B -P 'opensource,!internal' -pl flexlb-api,flexlb-mock-engine -am test
```

- [结构化结果](results.json) 保留全部正式、初轮、调参结果和性能断言失败。
- `baseline-warm-perf.log.gz` / `refactored-warm-perf.log.gz`：正式 BATCH 前后对照。
- `baseline-nonbatch-perf.log.gz` / `refactored-nonbatch-perf.log.gz`：正式 NON_BATCH 前后对照。
- `refactored-nonbatch-repeat-perf.log.gz`：NON_BATCH 复测。
- `functional-tests.log.gz`：1491 个功能测试的完整记录。
- 其余压缩日志保留预热／旧发压器／线程数诊断，不与正式参数混用。
