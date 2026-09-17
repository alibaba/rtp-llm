# 单次快照内增量预测：实现与性能

## 结论与边界

在基线 `159933443e8193631ee06ba23f326016df6aec47`（已包含 JVM 字节码编译和聚合循环融合）之上，实现追加式批预测。配置公式保持不变；真实 WorkerBatcher 与路由队列投影均已接入。未部署线上。

测量使用用户提供的 DeepSeek V4 完整公式，副本见 formula.txt。实验参数 maxPredictedExecutionMs=700、maxCollectionWaitMs=700，batch 上限分别为 1/8/32/64。token/KV 容量在本基准中设置为不主导截断的 1M/2M。请求的计算 token 均匀分布在 100～4195、命中 token 在 0～4095。这是固定种子的合成负载，不是线上流量回放。

## 实现边界

- ArithmeticFormulaCompiler 使用原 AST 生成 append(item, sums) 和 evaluate(batch, sums)，每个独立 sum 保留原顺序的累加器，重复项复用。外层表达式按原顺序执行。超出 JVM 编译限制时保留全量适配路径。
- FormulaPredictor 创建本次操作独占的 BatchPrediction；AppendBindings 逐项更新统计和变量，不重建每个前缀的 PrefillBatchFeatures。
- GroupPlanner 通过显式 append 回调通知新成员。原 list 回调接口保留，适配到同一个选择实现。
- 真实 BATCH 交付与路由模拟使用同一个预测器能力。非公式预测器通过默认全量实现兼容。
- 路由模拟的 AppendPlanning 保存已评估前缀的原始 double；服务时间只从同一个规划游标获取确实评估过的前缀，经原 PredictionBoundary 验证和 ceil。probe 后未计算的后缀仍正常预测。
- 会话不跨快照、模型或 planner 共享。插入、删除、重排或超时剔除后的新规划创建新会话。没有全局预测结果缓存、没有浮点减法回滚，不修改 TTL、请求状态机或资源记账。
- 嵌套 sum 沿用原求值器的标量子作用域语义，并已加入差分验证。

## 结果

单位：耗时 µs/次；分配量 bytes/次。每个版本 3 个独立 JVM，每个场景 3 轮预热、3 轮测量，每轮至少 400ms；表格为 9 次测量中位数。JVM 交替运行，单线程，不与 Maven 测试并行。

| 场景 | 基线 µs | 增量 µs | 加速 | 基线分配 B | 增量分配 B |
|---|---:|---:|---:|---:|---:|
| select_1 | 0.070 | 0.074 | 0.94× | 256.0 | 600.0 |
| select_8 | 1.101 | 0.392 | 2.81× | 3624.0 | 1216.0 |
| select_32 | 11.030 | 1.370 | 8.05× | 30888.0 | 3040.0 |
| select_64 | 24.201 | 1.915 | 12.63× | 55193.8 | 4176.5 |
| fleet5_depth_0 | 0.208 | 0.182 | 1.15× | 424.0 | 392.0 |
| fleet5_depth_32 | 151.108 | 18.779 | 8.05× | 358720.0 | 48240.0 |
| fleet5_depth_128 | 438.894 | 49.441 | 8.88× | 1007264.0 | 107120.0 |
| fleet5_depth_512 | 1636.612 | 177.628 | 9.21× | 3725464.0 | 386752.0 |

`select_N` 包括完整 GroupPlanner 前缀选择、特征构造、变量绑定、求值与超限判断，N 是请求数上限；700ms 上限可能让实际 batch 提前结束。

`fleet5_depth_N` 是五个候选机器各有 N 条排队请求时的冻结队列投影总耗时，包含多批次规划、过期索引与服务时间计算。队列快照预先构造，不包括实时 endpoint 快照获取、缓存匹配、路由锁、RPC、GPU 执行。它不是 Master 端到端压测。

原始 wall time、线程 CPU time、线程分配量及 checksum 均在 before/after 文本内。summary.json 保存中位数。对象分配量来自 ThreadMXBean，而不是存活堆大小。checksum 仅做基准防错；严格等价性以差分测试为准。

小 batch 的会话创建成本可能抵消收益，单条场景需要单独看；空队列控制组的细小变化也不能归因于长队列增量优化。本地结果不能直接换算为线上整机 CPU 降幅、集群吞吐或 P99 改善。

## 正确性验证

- common/cache/grpc/sync 完整 reactor：1,203 个测试全部通过；随后补充的精确前缀复用测试通过（该测试类 6 项全通过），合计覆盖 1,204 个不同测试。
- 8 种表达式 × 100 组输入 × 64 个前缀，共 51,200 次比较；包含实际完整公式、非线性批总量、嵌套 sum、scalar 变量、负零、非有限数和大整数转换边界。
- 500 个随机表达式 × 32 个前缀，共 16,000 次比较。保持非 NaN 的 double 位值一致，NaN 按 doubleToLongBits 规范化。
- 120 次删除、插入、重排后比较选择和 readiness；150 个包含过期成员、不同优先级/probe 位置的场景比较完整 Candidate。
- 并发独立会话、模型替换、真实交付策略 append 接入、超限撤回到前一预测、毫秒取整，以及不把 probe 前缀预测误用于完整 batch 均有测试。

## 复现

JDK 21，仓库 flexlb 根目录：

```bash
./mvnw -B -P 'opensource,!internal' -pl flexlb-sync -am test
```

将基线提交的 `rtp_llm/flexlb` 和 `rtp_llm/cpp/model_rpc/proto` 导出到独立临时目录，在该基线 flexlb 目录执行：

```bash
./mvnw -B -P 'opensource,!internal' -pl flexlb-sync -am -DskipTests test
```

再从当前版本运行（JAVA_HOME 指向 JDK 21）：

```bash
python3 tools/incremental-prediction/run.py /absolute/baseline/rtp_llm/flexlb /tmp/incremental-results
```

脚本独立编译/运行两个版本的基准，基线前缀选择使用原全量 API，增量版本使用追加 API；五候选投影使用两版各自真实的 RouteProjection/BatchDeliveryStrategy。依赖 classpath 来自当前版本 IncrementalPredictionTest 的 Surefire 报告。

环境：macOS ARM64、Apple M5 Pro、Microsoft OpenJDK 21.0.9，固定堆 512MiB。未使用 JMH，未模拟 64 个 planner 的并发负载；结果用于评估本次算法变化和单次操作的成本。
