# 远端 FlexLB 性能验证（2026-09-24）

按下列源码快照分别记录未提交工作区的验证结果，不能用 Git HEAD 代替源码身份。

## 初始快照：轮询简化后的环境与源码

- 主机：`luoli.hn@11.163.39.110`；容器：用户指定的 `luoli_gpu`。
- 目录：`/home/luoli.hn/work/rtp_llm_4/github-opensource/rtp_llm/flexlb`。
- 容器默认 Java 8；本次在 `/opt/flexlb-jdk21-20260924` 安装并使用 Temurin 21.0.12.1+1。
- JDK 归档对照 Adoptium 元数据校验 SHA256：`ce79869e1307ed8ee1e2baa86a412b1eb5b75d10a01006d788a6f968bcfaee94`。
- 488 个源码、资源与构建文件在远端逐项 SHA256 校验通过。清单摘要：`d76f6a1524a5ddc21b57070462837c2046f7d3e8b752be327f05ebab88057d95`。
- 同步前远端备份：主机 `/tmp/flexlb-before-polling-sync-20260924-2104.tar.gz`。
- 测试原始日志：容器 `/tmp/flexlb-polling-perf-20260924/`；运行脚本：容器 `/tmp/flexlb-polling-perf.sh`。
- 性能门槛保持原值；全部 Maven 调用串行运行。

## 初始快照：已验证结果

`ProductionCaliberDecodeTest`：8 项通过，无失败/跳过；首次执行使用 clean 构建，清除已删除类的旧产物。

| 场景 | 实测 | 原门槛 | 结果 |
| --- | ---: | --- | --- |
| 4 streams × 500 tokens | 511 tok/s | 519 ±15% | 通过 |
| 128 streams × 100 tokens | 7,688 tok/s | 7,726 ±15% | 通过 |
| 1 stream × 500 tokens | 129 tok/s | 119 ±20% | 通过 |

这是 Mock Decode 模型在 Linux 容器上的计时验证，不是 GPU 模型推理吞吐，也不能单独证明此前本地低批量锚点失败的原因。

## Sync / API 性能结果

- Sync：`WorkerBatcherPerformanceTest`、`PrefillAdmissionFailurePerformanceTest` 共 **4 项通过**；稳定队列捕获在 0/1/32/128/512 深度均为 **0 B/op**。
- API：完整 `api-performance-regression` profile，使用原默认 2 GB 堆；**16 项中 14 项通过、2 项失败**。
- 突发用例：client 4,460.6 QPS，低于 5,000 QPS 门槛。首个吞吐断言失败，不能声称后续 Master 吞吐与 P99 断言已通过。
- 4P/8D、10,000 QPS：delivery wait P99 为 51 ms，超过 50 ms 门槛。
- 这两项仍待定位和修复；未放宽阈值，未以通过的其他用例替代它们。

## 750P/750D 队列检查：4 项全部通过

使用 `tools/run_queue_performance.sh`，BATCH/NON_BATCH、3,000/10,000 offered QPS，原 0.98 吞吐比例门槛，8 GB 堆。

首次脚本在环境采集时被 Git 的目录所有者检查中止，尚未启动测试。通过本次进程的 `GIT_CONFIG_*` 为该仓库配置 `safe.directory` 后重跑，日志目录为容器 `/tmp/flexlb-polling-perf-20260924/queue-verified/`。没有修改全局 Git 配置。

| 投递方式 | 目标 QPS | 客户端 QPS | Master QPS | Master P99 | 投递等待 P99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BATCH | 3,000 | 2,996.8 | 3,004.3 | 13 ms | 11 ms |
| BATCH | 10,000 | 9,988.9 | 10,003.9 | 27 ms | 26 ms |
| NON_BATCH | 3,000 | 2,996.7 | 3,000.4 | <1 ms | 1 ms |
| NON_BATCH | 10,000 | 9,995.5 | 10,001.9 | <1 ms | 1 ms |

每个场景预热与测量各 10 秒；BATCH/NON_BATCH 两次 Maven 调用均 BUILD SUCCESS。这里的通过不能替代默认 API profile 中失败的两个场景。

## 后续定位

单独运行突发用例并通过 JFR 采集其测试 fork，采样会影响计时，诊断结果不作为无采样验收通过证据。采样严格匹配本次 Maven 子进程，不接触其他 Java 服务。

初始快照未通过最终性能验收。后续快照及结果见下文。


## 第二个快照：队列投影单遍构建

JFR 诊断记录中，620 个执行采样有 377 个落在全局规划线程；244 个采样的首个 FlexLB 栈帧为 `RouteTimelineProjector.projectWithPredictions`。这指向投影扫描开销，但采样不能精确归因每次分配。删掉中间有效请求列表，将过期过滤、重复 ID 检查和 probe 插入合并到投影数组的一次构建中。

- 源码清单：`/tmp/flexlb-projection-sources.sha256`，488 文件逐项校验通过；摘要 `b524ed7c4c03bede5253386ed346a696e19febfb3456110792bfee6dc062cb83`。
- 原始日志：容器 `/tmp/flexlb-projection-perf-20260924/`；本地归档 `/tmp/flexlb-projection-perf-results.tar.gz`。
- 默认 API profile：**16 项中 15 项通过，1 项失败**。
- 8192 请求突发：client **5,078.2 QPS**、Master **5,159.9 QPS**，均过 5,000 门槛；Master P99 **857 ms**，仍高于 **250 ms**。只有一次对照，不能把全部计时变化归因于这次代码改动。
- 初始快照失败的 4P/8D、10,000 QPS 场景本次通过；尚不足以证明该临界值稳定性问题已消除。
- 750P/750D 四场景均通过，两次 Maven BUILD SUCCESS，原门槛未变：

| 投递方式 | 目标 QPS | 客户端 QPS | Master QPS | Master P99 | 投递等待 P99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BATCH | 3,000 | 2,996.8 | 3,002.0 | 14 ms | 11 ms |
| BATCH | 10,000 | 9,978.0 | 9,995.9 | 94 ms | 27 ms |
| NON_BATCH | 3,000 | 2,993.6 | 3,000.7 | <1 ms | 1 ms |
| NON_BATCH | 10,000 | 9,992.8 | 10,003.7 | 33 ms | 1 ms |

## 第三个快照：删除 Session 中转层

删除 `RouteProjection.Session`，线程直接持有投影器；合并 reset 与 project 的入口，删除静态中转与重复候选谓词。`sync` 生产 Java 从第二个快照的 28,816 行降至 **28,759 行**。这一版的本地完整 API reactor 通过；三个独立 review 无遗留问题。

- 源码清单：`/tmp/flexlb-projector-layer-sources.sha256`，488 文件逐项校验通过；摘要 `abb09de5991f8916acde20044887e752e1a32639e9cfd705799d8a5090ffe313`。
- 指定容器上仅对剩余突发场景做规划线程数实验，未修改生产默认配置，也未放宽验收门槛。
- 原始日志：容器 `/tmp/flexlb-planner-experiment-20260924/`。

| 规划线程数 | 客户端 QPS | Master QPS | Master P99 | 路由提交 P99 | batch wait P99 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 5,686.6 | 5,780.5 | 676 ms | 310 ms | 377 ms |
| 32 | 5,653.7 | 5,743.0 | 908 ms | 618 ms | 388 ms |

两次均因 Master P99 超过 250 ms 失败，线程数调整不足以解决问题，因此没有更改默认值。第二个快照的完整性能结果不能代替第三个快照的完整验收。

**当前结论：功能回归和三方审查通过，25,000 行目标及完整性能验收均未完成。**

## 第四个快照：Endpoint 统一投影缓存，移除 DIRECT Batcher（2026-09-25）

本快照包含上次远端之后的累计简化，sync 生产 Java **28,019 行**；不能把全部性能差异归因于最后一次缓存归属迁移。

- 指定 host 路径和 `luoli_gpu` 容器执行，JDK `/opt/flexlb-jdk21-20260924`。API 保持默认 2 GB 堆；750P/750D 使用既有脚本默认 8 GB 堆。未放宽吞吐/P99 门槛。
- host 旧源码备份 `/tmp/flexlb-before-direct-removal-20260925.tar.gz`。首次工具目录同步因容器生成的 `__pycache__` 无删除权限失败，随后排除缓存同步，未改测试源码/门槛来处理环境问题。
- `/tmp/flexlb-direct-removal-sources.sha256` 共 **600 文件逐项校验通过**，清单摘要 `3260515b21ee4a817a7956776c9ed8ddc7f32ebd9b095f353dd20c773044970c`。容器日志 `/tmp/flexlb-direct-removal-perf-20260925/`，本地原始归档 `/tmp/flexlb-direct-removal-results.tar.gz`。
- Decode 锚点 **8/8 通过**；sync 性能 **4/4 通过**。
- 完整 API 性能 **16 项中 14 项通过、2 项失败**。8192 请求突发 client/master QPS **5,313.2 / 5,404.3** 通过 5,000 门槛，但 Master P99 **783 ms > 250 ms**。阶段 P99：grpc queue 7 ms、route submit 525 ms、batch wait 352 ms、dispatch ACK 15 ms、ACK response 2 ms。
- 另一失败是 **1P/2D、10,000 QPS**，delivery wait P99 **62 ms > 50 ms**（Master P99 64 ms；batch wait P99 61 ms）。不能用整体吞吐通过掩盖该尾延迟失败。

750P/750D 四场景均通过：

| 投递方式 | 目标 QPS | 客户端 QPS | Master QPS | Master P99 | 投递等待 P99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BATCH | 3,000 | 2,996.8 | 3,000.5 | 12 ms | 11 ms |
| BATCH | 10,000 | 9,989.0 | 10,003.4 | 27 ms | 24 ms |
| NON_BATCH | 3,000 | 2,999.6 | 3,000.4 | 23 ms | 1 ms |
| NON_BATCH | 10,000 | 9,995.3 | 10,001.8 | 5 ms | 1 ms |

同源码另做一次单规划线程突发对照（只设置运行参数 `flexlb.queue.planner.threads=1`，未更改生产默认）：client/master QPS **7,162.4 / 7,325.7**，Master P99 **713 ms**；route submit P99 **676 ms**、batch wait P99 **69 ms**。仍未通过 250 ms 门槛。该一次对照显示降低并行度减少了 batch wait，但路由提交仍是显著等待来源；不足以据此修改生产线程数或认定原因已完全定位。

**本快照功能回归和三方 review 通过；25,000 行目标及性能验收仍未完成。**


## 2026-09-25：Capture 列表构建 A/B，撤回 stream 优化

远端使用用户指定目录及 `luoli_gpu` 容器，Temurin 21.0.12.1，固定堆 1 GiB。复用远端上轮 classpath，仅通过首位 classpath 分别覆盖旧/新 PrefillActiveIndex 及内部类；不是当前整个工作区的重新构建。两侧编译同一 SnapshotBench 和临时 CaptureBench，10 个真实 PrefillEndpoint，真实账本成员变动、捕获与共享列表物化，mock 监控等依赖；没有网络发送或运行 worker 线程。每个版本 3 个独立 JVM，第二轮倒序运行，各场景每 JVM 2 轮预热 + 3 轮测量（每轮至少 200ms），下列值为 9 个样本中位数。各轮 checksum 校验通过。

| 场景 | 旧 ns/op | stream ns/op | 旧 CPU ns/op | stream CPU ns/op | 旧 B/op | stream B/op |
|---|---:|---:|---:|---:|---:|---:|
| fleet10_depth1024_p1_membership_capture | 32218.36 | 35542.07 | 31254.36 | 33817.74 | 42353.94 | 32580.00 |
| fleet10_depth1024_p64_membership_capture | 972.91 | 1060.26 | 4359.90 | 5580.90 | 690.33 | 540.11 |
| fleet10_depth32_p1_membership_capture | 3816.87 | 3952.09 | 3282.79 | 3408.22 | 2547.00 | 2697.00 |
| fleet10_depth32_p64_membership_capture | 397.88 | 396.69 | 575.33 | 662.51 | 59.56 | 62.00 |

结论：深队列约节省 22%–23% 分配，但两个深队列场景耗时中位数退化约 9%–10%，浅队列分配也增加。没有证据支持当前替换，已恢复原显式 ArrayList + List.copyOf 实现，不为减少 9 行保留此改动。结果不能证明所有负载都退化，也不构成端到端性能验收。原 burst P99 和 1P/2D 等性能失败仍未解决。

原始日志、环境、脚本和临时 harness：`/tmp/flexlb-capture-ab-results.tar.gz`（本地）；容器目录 `/tmp/flexlb-capture-ab-20260925`。旧源 SHA256 `403e147a5bc6b6ddf2bc36ed46352d8c9f6208852c87f434bd6fe5a6027209b6`；stream 源 `ba88a8f184cd202a5b749c26e88f3eb888a0c152c513828b752c3284af54c167`，均与本地版本匹配。


## 2026-09-25: Current canonical-state API performance gate (27,779 lines)

- Ran in luoli_gpu at /home/luoli.hn/work/rtp_llm_4/github-opensource, Temurin 21.0.12.1. Fixed profile heap 2 GiB; planner and SLO defaults unchanged. No competing Maven/Surefire/benchmark process found before launch. No environment blocker.
- Backed up remote module sources and POMs to host /tmp/flexlb-before-canonical-state-perf-20260925.tar.gz. Synced six module src trees and POMs plus parent POM; 486-file source verification passed. Manifest /tmp/flexlb-canonical-state-sources.sha256, SHA256 3cf8ea3ec8446d0cb98c1ea72435c79e71add997b988c073409db2550f811080.
- Command: ./mvnw -B -P opensource,!internal,api-performance-regression -pl flexlb-api -am clean test. 16 tests: 14 passed, 2 assertion failures, no errors/skips. Job exited 1 and has completed.
- Burst 8192: client/master QPS 5458.5/5540.9 (pass 5000 floor), Master P99 797ms (FAIL 250ms). Stage P99: grpc queue 8ms, route submit 536ms, batch wait 360ms, ACK 20ms, response 1ms. Stage quantiles are not additive.
- Second failure was 16P/32D at 10k: delivery P99 82ms > 50ms. Previous failing 1P/2D at 10k passes this run (delivery P99 14ms). A single run does not establish a causal regression or improvement between snapshots. Burst remains the reproducible unresolved bottleneck; do not describe the two failures as unchanged topology.

| P | D | Target QPS | Client QPS | Master QPS | Master P99 | Delivery P99 |
|---|---|---|---|---|---|---|
| 1 | 2 | 1000 | 993.3 | 1006.4 | 14ms | 13ms |
| 1 | 2 | 2000 | 1980.7 | 2027.8 | 29ms | 29ms |
| 1 | 2 | 5000 | 4900.0 | 4964.0 | 9ms | 8ms |
| 1 | 2 | 10000 | 9809.2 | 9926.2 | 16ms | 14ms |
| 2 | 4 | 2000 | 1960.8 | 2005.6 | 12ms | 12ms |
| 4 | 8 | 1000 | 990.8 | 1001.4 | 11ms | 11ms |
| 4 | 8 | 2000 | 1974.9 | 2019.7 | 12ms | 11ms |
| 4 | 8 | 5000 | 4900.9 | 5014.7 | 34ms | 33ms |
| 4 | 8 | 10000 | 9788.1 | 9950.1 | 18ms | 12ms |
| 8 | 16 | 2000 | 1961.8 | 2005.1 | 11ms | 11ms |
| 16 | 32 | 1000 | 990.7 | 1001.0 | 11ms | 11ms |
| 16 | 32 | 2000 | 1961.5 | 2003.6 | 19ms | 18ms |
| 16 | 32 | 5000 | 4897.5 | 5023.7 | 20ms | 18ms |
| 16 | 32 | 10000 | 9439.4 | 9700.0 | 91ms | 82ms |

Artifacts: local /tmp/flexlb-canonical-state-results.tar.gz; container /tmp/flexlb-canonical-state-perf-20260925; script /tmp/flexlb-canonical-state-perf.sh. This reran the API performance gate only, not decode/sync microbenchmarks or the 750-engine matrix. No production edits in this turn; sync production remains 27,779 lines, 2,779 above target.


## 2026-09-25：统一路由提交版本（sync 27,589 行）

- 用户指定主机 `luoli.hn@11.163.39.110`、目录
  `/home/luoli.hn/work/rtp_llm_4/github-opensource`、容器 `luoli_gpu`。
- 容器运行中，启动前没有其他 Maven/Surefire；JDK21 路径
  `/opt/flexlb-jdk21-20260924`，沿用 2g profile heap 和默认 planner 配置。
- 六模块源码与 pom 共 486 文件全部 hash 校验通过。manifest
  `/tmp/flexlb-unified-placement-sources.sha256`，SHA256
  `e63051f68c13801f5d8d85fa889a2abe5fd0cce05fe953b80c95ca7398dd4fbf`。
- 远端备份 `/tmp/flexlb-before-unified-placement-perf.tar.gz`；运行脚本
  `/tmp/flexlb-unified-placement-perf.sh`；容器输出
  `/tmp/flexlb-unified-placement-perf-20260925`；本地归档
  `/tmp/flexlb-unified-placement-results.tar.gz`。
- 命令：`./mvnw -B -P 'opensource,!internal,api-performance-regression' -pl flexlb-api -am clean test`。
- 结果：16 tests、15 passed、1 failure、0 errors、0 skipped。无环境阻碍。
- 唯一失败：8192 请求 burst，client/master QPS 5465.5/5571.0，吞吐门槛通过；
  Master P99 **705ms > 250ms**。阶段 P99：grpc queue 5ms、route submit 452ms、
  batch wait 333ms、dispatch ACK 11ms、ACK response 2ms。
- 本次规模矩阵全部通过，含此前失败的 16P/32D 10k（delivery P99 37ms）。
  不将单次波动归因于本轮重构，也不将其视为整个性能目标通过。
- 本轮没有重跑 Decode/sync microbenchmark 或 750P/750D gate。

| P | D | 目标QPS | client QPS | master QPS | master P99 | delivery P99 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 1000 | 996.8 | 1011.0 | 14ms | 13ms |
| 1 | 2 | 2000 | 1972.2 | 2018.1 | 29ms | 11ms |
| 1 | 2 | 5000 | 4895.9 | 4960.8 | 6ms | 5ms |
| 1 | 2 | 10000 | 9498.0 | 9627.7 | 51ms | 49ms |
| 2 | 4 | 2000 | 1977.4 | 2023.9 | 12ms | 12ms |
| 4 | 8 | 1000 | 991.7 | 1003.0 | 11ms | 11ms |
| 4 | 8 | 2000 | 1966.9 | 2011.4 | 13ms | 11ms |
| 4 | 8 | 5000 | 4923.5 | 5032.2 | 32ms | 29ms |
| 4 | 8 | 10000 | 9778.9 | 9942.3 | 23ms | 22ms |
| 8 | 16 | 2000 | 1965.2 | 2005.5 | 11ms | 11ms |
| 16 | 32 | 1000 | 990.9 | 1001.4 | 11ms | 10ms |
| 16 | 32 | 2000 | 1961.7 | 2004.3 | 22ms | 21ms |
| 16 | 32 | 5000 | 4906.3 | 5029.9 | 30ms | 28ms |
| 16 | 32 | 10000 | 9782.6 | 10074.2 | 96ms | 37ms |

## 2026-09-25：删除投递中间所有者后的远端回归

- 当前 sync 生产 Java 27,463 行；同步指定主机目录，在 `luoli_gpu` 中执行。
- 相同 JDK21、默认 2 GiB heap、默认 planner 并发；未调整测试门槛。
- 命令：`./mvnw -B -P 'opensource,!internal,api-performance-regression' -pl flexlb-api -am clean test`。
- 486 个 source/pom 文件逐个 SHA256 校验通过；manifest
  `/tmp/flexlb-transaction-ownership-sources.sha256`，manifest SHA256：
  `6eee83482469eb7569160c2d889c57d6cdca4cdccf6f4cfc2a2dd3854c96087d`。
- 16 项：15 通过、1 失败、0 错误/跳过。唯一失败仍是 8192 burst 的 Master P99。
- burst：client/master QPS 5,578.7 / 5,667.8（吞吐门槛通过），Master P99 **735 ms > 250 ms**。
  阶段 P99：grpc_queue 1 ms、route_submit 460 ms、batch_wait 386 ms、dispatch_ack 18 ms、ack_response 1 ms。
- engine scale 各档通过，包括 1P/2D 10k 的 delivery P99 13 ms、Master P99 15 ms，
  16P/32D 10k 的 delivery/Master P99 均 27 ms。
- 前一次 27,589 行版本 burst P99 为 705 ms；两次都是单次运行，且间隔包含其他本地重构，
  不能把差值归因于本轮删类。性能目标仍未通过，没有环境阻碍。
- 本轮未重跑 Decode 微基准和 750-worker fleet；不将 API profile 结果扩大为全部性能验证。
- 归档：`/tmp/flexlb-transaction-ownership-results.tar.gz`；远端工作目录
  `/tmp/flexlb-transaction-ownership-perf-20260925`；本地运行日志
  `/tmp/flexlb-transaction-ownership-remote-run.log`。


## 2026-09-25：当前所有权重构与路由判空物化对比

环境仍为指定11.163.39.110主机、github-opensource目录与luoli_gpu，JDK21、默认2GiB堆与
默认planner配置。两次均执行相同API性能profile的clean test，未改阈值。无环境阻碍。

| 版本 | 总行数 | burst client/master QPS | Master P99 | 结果 |
| --- | ---: | --- | --- | --- |
| 当前所有权重构，判空改动前 | 27,198 | 4136.4 / 4191.7 | 1223ms | 16项中1失败：burst吞吐；延迟也超250ms |
| 删除锁内判空的请求列表物化 | 27,194 | 5158.3 / 5228.7 | 862ms | 16项中2失败：burst延迟、1P/2D 10k吞吐 |

- 改前阶段P99：grpc4、route_submit1027、batch_wait735、dispatch_ack35、response2ms。
- 改后阶段P99：grpc10、route_submit484、batch_wait378、dispatch_ack21、response1ms。
- 改后1P/2D 10k：client8476.1/master8637.0 QPS，client低于8500门槛；Master P99 137ms。
  改前该档client9790.7/master9919.9 QPS、Master P99 14ms。
- 改后16P/32D 10k：client9697.8/master9947.5 QPS，Master P99 36ms；该档通过。
- 单次运行不能把差值归因于判空优化。正以相同源码/配置复跑，核对新增矩阵失败的稳定性。
- 两次均486文件SHA256校验通过。改前manifest SHA256：
  `201f227160a3c89cecf94f2e18103b865a7bce3c2426c7587b130579839e879a`；
  改后manifest SHA256：`0ee609d47127c20af29c129838c7a21f5dd3408eeacc71c925257a3785b112f6`。
- 本地归档 `/tmp/flexlb-current-ownership-results.tar.gz`、`/tmp/flexlb-projection-empty-results.tar.gz`，
  包含完整日志、环境信息、源码校验和API XML报告。未运行750-worker fleet或Decode微基准。


相同源码和配置复跑结果：16项15通过1失败，唯一失败为burst Master P99 **771ms>250ms**。
client/master QPS 5275.3/5359.0；阶段P99 grpc3、route_submit419、batch_wait400、
dispatch_ack17、response1ms。1P/2D 10k恢复通过：client9767.8/master9879.1 QPS，
Master P99 17ms。矩阵吞吐失败未稳定复现，但保留首次失败，不能据此宣称不存在性能波动。
复跑也完成486文件SHA256核验，源码与前次改后一致。归档
`/tmp/flexlb-projection-empty-repeat-results.tar.gz`。当前远端源码为27,194行版。
性能目标仍不通过；没有环境阻碍。判空改动的收益仅从源码证明减少物化，不能把单次
前后P99差异当作因果收益。

## 2026-09-25：27,115行版本的 planner 数量定位实验

当前源文件同步到用户指定目录，在 luoli_gpu 中执行；JDK Temurin21.0.12.1，256 CPU，默认2GiB堆。486个源码/资源/pom文件逐一SHA256通过；manifest为 `/tmp/flexlb-planner-probe.sha256`，其SHA256为 `caba9e886b9fe7b1edbfb00a0be33ae373027748f94ec049473f57feb1f1edf2`。远端备份 `/tmp/flexlb-before-planner-probe.tar.gz`。

本轮仅执行 `MasterBatchEndToEndPerformanceTest#batchScheduleRemainsFastAcrossRealGrpcBoundaries`，8192请求。没有执行完整16项性能矩阵，不代表完整性能验收。默认、32、8依次运行；后两次通过现有 `flexlb.queue.planner.threads` 系统属性覆盖，XML已确认参数生效。没有修改生产默认值或验收阈值。

| planner | client QPS | Master QPS | Master P99 ms | grpc queue P99 | route submit P99 | batch wait P99 | dispatch ACK P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 默认256 | 5141.4 | 5223.2 | 889 | 8 | 505 | 426 | 23 |
| 32 | 5524.8 | 5613.8 | 723 | 2 | 405 | 349 | 18 |
| 8 | 6426.0 | 6541.8 | 643 | 6 | 435 | 312 | 17 |

三个测试均失败于Master P99超过250ms，吞吐均超过5000门槛。三个样本各一次，按固定顺序运行，不能据此确定最优线程数或归因全部延迟。较少planner值得进一步分析，但即使8线程仍明显未达标；不以参数覆盖结果冒充默认性能。

只读代码复核得到两个后续可证伪候选：每个planner各自对全舰队执行投影；completedPlans在单decision线程等待期间可能变旧，再因blockedEndpointChanged触发REPLAN。需要每请求规划次数、REPLAN比率、投影耗时和线程/锁剖面区分，而不能仅凭双向依赖或CPU数修改结构。

日志与逐次XML：`/tmp/flexlb-planner-probe-20260925`；完整归档 `/tmp/flexlb-planner-probe-results.tar.gz`；执行脚本 `/tmp/flexlb-planner-probe.sh`；SSH运行日志 `/tmp/flexlb-planner-probe-run.log`。本轮远端无环境阻碍。

同版本本地完整API依赖回归：1634项记录、1633通过、1跳过，无失败/错误（common225/cache37/grpc15/sync1182/api175）。日志 `/tmp/flexlb-27115-full-tests.log`。该回归不替代性能门槛。

## 2026-09-25：默认配置JFR与Worker前缀快照

基线27,115行、默认256 planner、2GiB堆，同一burst方法通过JAVA_TOOL_OPTIONS开启JFR profile、stackdepth128、dumponexit；没有改源码、阈值或planner默认值。采样运行Master QPS5505.4/P99 849ms，仍失败；采样本身有开销，不与未采样运行直接作性能结论。

记录包含Maven、测试fork和ByteBuddy attach三个JVM，分析选择测试fork89710。554个ExecutionSample来自整个测试进程（含启动和预热，未严格切出8192请求区间）；planner线程208个、dispatch executor74个、batcher12个。按128帧展开后，包含projectWithPredictions的132个、ProjectedQueue.create的56个、Capture.projectedItems的24个；包含计数互相重叠，不能相加为百分比。它们支持优先分析投影构造，不支持把全部延迟归因于锁竞争。

原始归档 `/tmp/flexlb-default-jfr-results.tar.gz`；录制目录 `/tmp/flexlb-default-jfr-20260925/recordings`；脚本 `/tmp/flexlb-default-jfr.sh`；展开事件 `/tmp/flexlb-default-jfr-cpu.json`；摘要 `/tmp/flexlb-default-jfr-summary.txt`。首次默认jfr print深度仅5帧，已改用--stack-depth128，本文按完整展开数据记录。

随后Worker快照改为最多一个批次，并删除失去生产使用的Capture.items缓存，生产降至27,108行。默认配置、不开JFR复测同一burst：client QPS5331.5/Master QPS5412.8，Master P99 836ms；grpc_queue/route_submit/batch_wait/dispatch_ack/ack_response的P99分别6/474/419/21/2ms。失败于250ms门槛。与此前默认889ms均为单样本，不能据此声称稳定提升。

486个源文件校验通过，manifest `/tmp/flexlb-worker-prefix.sha256` 的SHA256为 `19837cab09c82f0feb5c50d5903fd880754bb9f644c335c340004d905840baff`。原始日志/XML归档 `/tmp/flexlb-worker-prefix-results.tar.gz`；目录 `/tmp/flexlb-worker-prefix-perf-20260925`；脚本 `/tmp/flexlb-worker-prefix-perf.sh`。远端备份 `/tmp/flexlb-before-worker-prefix-remote.tar.gz`。本轮仅burst，不是完整矩阵，无环境阻碍。


### Service cursor删除后的完整profile（26,914行）

远端luoli_gpu，默认256 planner/2GiB/JDK21；486文件manifest SHA256
`81de392ca3231a6f4a646450ffbe18dbd0e6169a1cf8fe1309c96c27a49c560c`。
16项15通过1失败。burst8192 client/Master QPS 4208.0/4262.4，
Master P99 1182ms；阶段P99为grpc3、route_submit804、batch_wait591、
dispatchACK21、ackresponse2ms。吞吐与延迟均未达到门槛。
原始结果`/tmp/flexlb-service-cursor-results.tar.gz`。
先前836ms来自不同版本单项运行，不能据此归因当前改动。原计划的精确基线/当前
burst对照尚未执行；基线文件曾同步后已恢复当前service-cursor源文件，
容器内486文件SHA复核通过。该对照仍待完成，不计为已验证性能。


### Service cursor精确源码对照：第一组A→B

A为删除前26,997行，B为删除后26,914行；本地当前26,891行的后续两轮
改动不在这次比较内。两次均执行同一个8192 burst方法、默认planner配置、
同容器JDK21；各自486项源码校验全部OK。

| 版本 | client QPS | Master QPS | Master P99 | route_submit P99 | batch_wait P99 | 门槛 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A baseline | 4105.5 | 4176.5 | 1340ms | 949ms | 724ms | 吞吐与延迟失败 |
| B 删除service cursor | 5151.5 | 5243.1 | 809ms | 415ms | 437ms | 吞吐通过，延迟失败 |

各阶段P99不是同一请求的分解，不能相加。单组不证明稳定改善，反向B→A复测
用于检查顺序影响。原始归档`/tmp/flexlb-cursor-comparison-results.tar.gz`。
基线manifest SHA256为`bcd4a968e46f542ac2837810d1919077a6ad3b2e43c29b2e0bf03923cf94fe5d`；
B manifest为`81de392ca3231a6f4a646450ffbe18dbd0e6169a1cf8fe1309c96c27a49c560c`。


### 反向B→A复测及结论

| 执行顺序 | offered QPS | client QPS | Master QPS | Master P99 | route_submit P99 | batch_wait P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 第一组A | 12608.2 | 4105.5 | 4176.5 | 1340ms | 949ms | 724ms |
| 第一组B | 10526.3 | 5151.5 | 5243.1 | 809ms | 415ms | 437ms |
| 第二组B | 12181.4 | 5700.5 | 5791.5 | 761ms | 425ms | 336ms |
| 第二组A | 10872.8 | 5199.6 | 5277.6 | 820ms | 504ms | 397ms |

四次XML均1项测试、1项失败、0错误；test.exit均1。第一组A吞吐与P99均不达标，
其余吞吐通过但P99未达到250ms。脚本末尾cp/tail导致shell可能返回0，因此验收
以test.exit与XML为准，不能以SSH退出码判断。第二组两侧486文件校验全部OK。
反向原始归档`/tmp/flexlb-cursor-reverse-results.tar.gz`。结束后恢复B源码，容器内
完整SHA复核通过，日志`/tmp/flexlb-cursor-final-restoration.log`；没有环境阻碍。

两组未观察到B相对A的回退，保留有明确结构依据的服务游标删除；样本量小且
非限速burst实际offered QPS不同，不能声明稳定提速或因果归因。两组都不覆盖
后续Worker提交标记删除和Route规划标量化。

三个review核对源码范围、失败门槛及阶段统计。route_submit包含全局队列等待和
选路；batch_wait含Worker成组等待与dispatcher排队，并非纯Worker CPU耗时。
下一项性能定位应固定输入速率并区分排队与计算；原burst验收门槛仍保留，不用
限速实验替代验收，也不继续重复同一burst来挑较好结果。

## 2026-09-25：PrefillActiveIndex 单类化对照

此次 A/B 的全部源码为当前工作区版本，仅 PrefillActiveIndex 不同；前一轮服务游标比较不能作为此次的精确基线。两份清单仅此一个路径的 SHA 不同，A 为 26,820 行生产代码，B 为 26,740 行。

环境：指定主机 `luoli.hn@11.163.39.110`、指定仓库目录及 `luoli_gpu` 容器，JDK Temurin 21.0.12.1、256 CPU。每侧 `mvnw -P opensource,!internal,api-performance-regression -pl flexlb-api -am clean test`，指定 `MasterBatchEndToEndPerformanceTest#batchScheduleRemainsFastAcrossRealGrpcBoundaries`。保留原 8192 请求、BATCH/FIXED_WINDOW 和原断言。

| 指标 | A：接口与两实现 | B：单一类 |
| --- | ---: | ---: |
| offered QPS | 10723.9 | 11019.4 |
| client QPS | 5530.2 | 5610.4 |
| Master QPS | 5615.1 | 5687.5 |
| Master P50 ms | 426 | 489 |
| Master P99 ms | 712 | 712 |
| route_submit P99 ms | 355 | 366 |
| batch_wait P99 ms | 357 | 354 |
| dispatch_ack P99 ms | 20 | 19 |

两侧均 test.exit=1，XML 各 tests=1/failures=1/errors=0/skipped=0，失败为 Master P99 超过 250 ms。两侧及最终恢复均 473/473 Java 源码 SHA 通过。一次顺序对照、不同 offered QPS 不能证明稳定性能改善；相同 P99 也不足以排除小幅回退。此次保留结构删除，性能门槛仍未通过。

原始日志、XML、环境和源码校验归档：`/tmp/flexlb-single-index-perf-results.tar.gz`；两份源清单 `/tmp/flexlb-single-index-{before,after}.sha256`；执行脚本 `/tmp/flexlb-single-index-perf.sh`。after 清单 SHA256 为 `e3a090e2b3db3451cd230a6e63ce895c23afa9041aca865663fe06d4f42be5eb`。

没有编译或测试执行环境阻碍。原 SSH 客户端在远端脚本结束后未退出；通过独立连接取回完整报告、确认没有性能脚本/Maven/Surefire 进程，并核对最终恢复 SHA 后，仅终止该滞留 SSH 客户端。没有重启或重复运行测试；测试结果以 XML 和 test.exit 为准。

## 2026-09-25：投影物化 JFR 与同源对照

按指定主机、目录和 luoli_gpu 容器执行，Temurin 21.0.12.1、256 CPU。先同步当前 473 份 Java 源码并校验，再以原性能 profile、原单 worker 8192 请求 burst 采集 JFR。没有放宽 QPS/P99 门槛。

### 定位证据

采样时版本为 26,709 行，`JAVA_TOOL_OPTIONS=-XX:StartFlightRecording=settings=profile,dumponexit=true` 继承到 Maven、Surefire 和 ByteBuddy attacher。分析使用 PID 106256 的 Surefire 记录，排除另外两个 JVM。

- 此次带采样运行 Master QPS 4340.0、P99 1042 ms；route_submit/batch_wait P99 为 608/456 ms，test.exit=1。采样运行不能与无采样结果直接作回归比较。
- 524 个 `flexlb-` 前缀线程的 ExecutionSample 中，265 个包含 `PrefillActiveIndex.Capture.projectedItems`，262 个包含 `ProjectionSource.materialize`；两组有重叠，不能相加或当作精确 CPU 占比。
- 多个 planner 线程在 `ProjectionSource.materialize` 的 monitor 上等待。各线程等待时长可重叠，不能累加为请求延迟。
- ArrayList.grow 出现在内联采样栈中，但当前代码已预分配列表容量；不据此推断真实扩容次数，JIT 栈归因有局限。
- 原始 JFR/日志/XML/源码校验：`/tmp/flexlb-state-profile-results.tar.gz`；本地分析脚本 `/tmp/analyze-flexlb-state-profile.py`，摘要 `/tmp/flexlb-state-profile-summary.txt`。开始和结束源码均 473/473 校验通过。

### 代码与无采样 A/B

唯一差异是 Capture.projectedItems 删除 ArrayList 中间容器及 List.copyOf，改为有序 `stream().map(Entry::projectedItem).toList()`。两层缓存、同步和异常重试边界不变。版本 A/B 分别为 26,709/26,704 行。

| 项目 | A：中间列表 | B：直接生成不可变列表 |
| --- | ---: | ---: |
| offered QPS | 10209.2 | 11074.8 |
| client QPS | 4157.4 | 5336.8 |
| Master QPS | 4214.1 | 5424.0 |
| Master P50 ms | 693 | 552 |
| Master P99 ms | 1165 | 791 |
| Master mean ms | 687.181 | 518.474 |
| route_submit P99 ms | 908 | 491 |
| batch_wait P99 ms | 761 | 362 |
| dispatch_ack P99 ms | 21 | 20 |

两侧均为 8192 请求、513 个 batch、平均 15.97、最大 16；均 clean build。两侧 test.exit=1、XML tests=1/failures=1/errors=0/skipped=0，完整门槛仍未通过。单次顺序对照、不同 offered QPS 不足以宣称稳定提升或排除回退；本轮保留语义等价的中间容器删除。

两侧和最终恢复均 473/473 SHA 通过。归档 `/tmp/flexlb-materialization-results.tar.gz`，脚本 `/tmp/flexlb-materialization-perf.sh`，清单 `/tmp/flexlb-materialization-{before,after}.sha256`；after 清单 SHA256 为 `8d377a6fe80912fc85d27430a0c1a4c5899d1774ac219a6cfa9e3351926b3c3f`。本轮没有环境阻碍，也没有遗留的测试进程。

### 32 planner 线程诊断

同一 B 版本、同一原测试，额外传入现有参数 `-Dflexlb.queue.planner.threads=32`，XML 确认参数生效。此项是诊断配置，不修改源码或默认值；未放宽验收阈值。

- offered/client/Master QPS：12436.8/5732.5/5825.3。
- Master P50/P99/mean：459/765/443.885 ms。
- route_submit/batch_wait/dispatch_ack P99：448/324/14 ms。
- 8192 请求、513 batch、平均 15.97、最大 16；test.exit=1，XML tests=1/failures=1/errors=0/skipped=0，P99 门槛仍失败。
- 开始及结束源码各 473/473 校验通过；归档 `/tmp/flexlb-planner32-results.tar.gz`，脚本 `/tmp/flexlb-planner32-diagnostic.sh`。

限制并发的单次结果仍远离目标，不能据此认为争锁是唯一瓶颈，也不足以修改默认线程数。下一步应检验队列投影反复扫描、成组模拟及快照物化的工作量；不继续通过选择有利样本宣称性能通过。

### 当前源码的默认并发 / 单 planner 对照（2026-09-25）

源码为 sync 生产 26,604 行版本，已同步至用户指定远端目录并在 luoli_gpu 内执行。
Temurin 21.0.12.1，容器 nproc=256。两侧均 clean build，运行原
MasterBatchEndToEndPerformanceTest#batchScheduleRemainsFastAcrossRealGrpcBoundaries，
保持 8192 请求、QPS 下限及 250 ms Master P99 门槛。单 planner 仅额外传入
`-Dflexlb.queue.planner.threads=1`，XML 确認该属性；生产默认配置未修改。

| 指标 | 默认并发 | 单 planner |
| --- | ---: | ---: |
| offered QPS | 11811.1 | 19915.9 |
| client QPS | 5424.9 | 7239.9 |
| Master QPS | 5511.9 | 7392.4 |
| Master P50 ms | 558 | 459 |
| Master P99 ms | 813 | 717 |
| Master mean ms | 521.896 | 447.728 |
| route_submit P99 ms | 466 | 698 |
| batch_wait P99 ms | 347 | 51 |
| dispatch_ack P99 ms | 21 | 15 |
| ack_response P99 ms | 2 | 1 |
| batches | 513 | 514 |
| average batch size | 15.97 | 15.94 |

两侧 test.exit=1，XML 均 tests=1/failures=1/errors=0/skipped=0，失败原因均为 Master P99
超过 250 ms。单 planner 减少了 Worker 队列等待，但全局选路提交等待更长，整体仍未达标。
阶段 P99 不能相加；单次顺序对照且 offered QPS 不同，不据此声称稳定提升或直接修改默认值。
下一步应在低并发下检查全局选路的每请求工作量及交接成本，不能把问题仅归于高并发物化争锁。

两次开始及最终源码清单均 474/474 SHA 校验通过；清单
`/tmp/flexlb-planner1-source.sha256` 的 SHA256 为
`92909e1adf76caf5a642ff2386435d1de4145e0fa69f8a8f7d9be0d89337bd0f`。
脚本 `/tmp/flexlb-planner1-diagnostic.sh`，归档 `/tmp/flexlb-planner1-results.tar.gz`，
解包日志 `/tmp/flexlb-planner1-diagnostic-20260925/`。远端脚本及本地 SSH 会话均已终结，
无环境阻碍。本轮没有生产改动，不计减行成果。


## 2026-09-25 输入/资源拆分后默认配置验证

源码：sync 生产 26,583 行；包含上一轮 Route 原子交接及本轮 RequestRequirements 拆分。
同步到用户指定 `/home/luoli.hn/work/rtp_llm_4/github-opensource`，在 `luoli_gpu` 执行。
Temurin 21.0.12.1，256 CPU；未设置 planner override、未开启 JFR。
475 个 Java 文件运行前后均 SHA256 校验通过；当前本地源码也匹配同一清单。
清单 `/tmp/flexlb-requirements-source.sha256`，自身 SHA256：
`014b2214f8673c5cc5ec9d794fbac34b3e70a9f929e52b3e2538ccf0feb6da34`。

命令：
```sh
./mvnw -B -P 'opensource,!internal,api-performance-regression' -pl flexlb-api -am clean test \
  '-Dtest=MasterBatchEndToEndPerformanceTest#batchScheduleRemainsFastAcrossRealGrpcBoundaries'
```

| 指标 | 结果 |
| --- | ---: |
| 请求数 | 8192 |
| offered QPS | 10939.2 |
| client QPS | 5293.1 |
| Master QPS | 5362.1 |
| Master P50 | 554 ms |
| Master P99 | 796 ms |
| Master mean | 519.430 ms |
| route_submit P99 | 443 ms |
| batch_wait P99 | 363 ms |
| dispatch_ack P99 | 19 ms |
| ack_response P99 | 1 ms |
| Engine batches | 513 |
| 平均 / 最大 batch | 15.97 / 16 |
| 平均输入 tokens | 6373.6 |

Maven exit 1；XML tests=1/failures=1/errors=0/skipped=0，失败为 P99 796 ms 超过 250 ms。
环境正常，无环境阻碍。与此前默认配置 813 ms 属于不同时间单次运行且 offered QPS 不同，
不能据此断言性能改善。各阶段 P99 不能相加解释总 P99。

产物：`/tmp/flexlb-requirements-perf-20260925/`；
归档 `/tmp/flexlb-requirements-results.tar.gz`；脚本 `/tmp/flexlb-requirements-perf.sh`。


## 2026-09-25 投影列表复用实验：未保留

此前单 planner JFR（26,604 行源码，不能当作当前源码延迟基线）的补充分线程统计：
planner 21 个 CPU 样本，7 个含 RouteTimelineProjector.projectWithPredictions、6 个含 List.copyOf；
dispatcher 71 个样本，32 个含 GenerateInputPB.Builder.mergeFrom、25 个含 writeTo；
completion publisher 51 个样本，包含测试 CompletionCoverageRecorder 和 JSON 序列化。
样本计数相互重叠，不是独占 CPU 时间。尤其 planner 仅 21 样本，不足以断言唯一瓶颈。
分析文件 `/tmp/flexlb-single-planner-thread-breakdown.txt`。
源码核对 DefaultBatchDispatcher.buildInput 仅一次 mergeFrom，然后修改 roleAddrs/priority/trace，
并非已证明重复解码；不能据 protobuf 热点直接删除协议转换。

实验改动仅 PrefillActiveIndex.Capture.projectedItems 的 Stream.toList 改成
Collectors.toUnmodifiableList，另加 import。当前 JDK 实测 List.copyOf 不复用前者但复用后者。
QueueSnapshot 的 defensive copy、缓存锁与失败重试契约不变。
首次 collect 自身可能增加分配，故不能由“后续复用列表”推出端到端更快。

本地专项 72/72，通过日志 `/tmp/flexlb-projection-list-tests.log`；三个 reviewer 无功能阻断。
指定远端 luoli_gpu，默认 planner、8192 burst、无 JFR，顺序跑 before/after 两次 clean test，
沿用 MasterBatchEndToEndPerformanceTest#batchScheduleRemainsFastAcrossRealGrpcBoundaries，门槛未修改。
两组各自 475 源码 SHA 全部通过；Maven 均 exit 1，XML 均 1 test / 1 failure / 0 error。

| 指标 | before | after |
| --- | ---: | ---: |
| offered QPS | 11733.0 | 11588.7 |
| Master QPS | 5515.6 | 5141.0 |
| Master P50 | 524 ms | 608 ms |
| Master P99 | 812 ms | 905 ms |
| Master mean | 524.179 ms | 581.488 ms |
| route_submit P99 | 582 ms | 614 ms |
| batch_wait P99 | 381 ms | 411 ms |
| dispatch_ack P99 | 19 ms | 23 ms |
| ack_response P99 | 2 ms | 2 ms |

两组均不满足 250 ms P99；环境正常。单次顺序对照不能证明性能回退的因果，
但不足以支持保留候选，已撤回单文件实验并同步恢复远端。生产仍 26,583 行。
产物 `/tmp/flexlb-list-perf-20260925/`，归档 `/tmp/flexlb-list-results.tar.gz`，
脚本 `/tmp/flexlb-list-perf.sh`；恢复后的源码核验 `/tmp/flexlb-list-restored-source.log`。


## 2026-09-25 Worker 单次队列捕获对照

在指定主机目录与 luoli_gpu 容器执行默认 clean 性能测试，无 planner 参数覆盖。
改前为 26,565 行，改后 26,533 行；每轮运行前后 478 个 Java 文件 SHA 全部一致。
478 包含 tools Java；仅新增本地快照测试未包含在远端清单内，生产源码与最终本地一致。

| 指标 | 改前 | 改后 |
| --- | ---: | ---: |
| offered QPS | 10141.3 | 11965.8 |
| Master QPS | 4842.6 | 6304.5 |
| Master P50 ms | 545 | 426 |
| Master P99 ms | 910 | 630 |
| Master 平均 ms | 540.204 | 399.666 |
| route submit P99 ms | 443 | 320 |
| batch wait P99 ms | 476 | 343 |
| dispatch ACK P99 ms | 20 | 16 |

两轮测试退出码均 1，XML 均为 1 failure / 0 error，P99 超过 250 ms。
环境正常，无环境阻碍。单次顺序对照、offered 不同，不能证明因果收益；
未观察到本轮退化，但没有专门证明窗口等待时新增列表分配的成本。
保留删除重复捕获和统一快照时点的结构改进，不宣称性能达标。
脚本 `/tmp/flexlb-capture-perf.sh`；结果 `/tmp/flexlb-capture-perf/`；
归档 `/tmp/flexlb-capture-results.tar.gz`；源清单 `/tmp/flexlb-capture-{before,after}.sha256`。


## 2026-09-25 14:13：当前 26,446 行源码重新验证与剖析

在指定主机目录与 luoli_gpu 容器执行。485 个 Java/POM 文件前后全部 SHA256 一致；
默认正常测量执行 clean test；随后相同源码开启 JFR，再做 planner=16 的诊断运行。
环境 Temurin 21.0.12.1、256 CPU，没有环境阻碍。默认 8192 burst、BATCH、FIXED_WINDOW，
未放宽 250 ms P99 或 5000 QPS 门槛。

| 指标 | 默认无 JFR | 默认含 JFR | 16 planner 诊断 |
| --- | ---: | ---: | ---: |
| offered QPS | 10865.1 | 10636.9 | 16676.0 |
| client QPS | 5280.2 | 5103.6 | 6113.3 |
| Master QPS | 5360.7 | 5219.7 | 6210.1 |
| Master P99 ms | 792 | 831 | 846 |
| route_submit P99 ms | 425 | 416 | 447 |
| batch_wait P99 ms | 419 | 416 | 438 |
| dispatch_ack P99 ms | 21 | 18 | 19 |
| ack_response P99 ms | 1 | 2 | 1 |

三次 Maven exit 均为 1，XML 均 1 test / 1 failure / 0 errors / 0 skipped，P99 门槛失败。
吞吐达标不能替代延迟门槛。16 线程运行的 offered load 不同，单次顺序对照不足以证明
线程数量对尾延迟的因果影响；生产默认配置未改。

JFR 明确选 Surefire JVM PID 120570（其余 PID 120389 为 Maven，120702 为 ByteBuddy）。
565 个 ExecutionSample 中，planner 220、dispatcher 53、global decision 8；planner 中
107 个样本含 RouteTimelineProjector.projectWithPredictions，74 个含 GroupPlanner.plan/select，
38 个含 ProjectedQueue.create，22 个含 ImmutableCollections.listCopy。
这些是重叠的调用栈样本，不是独占耗时，也不能由全程 ThreadPark 数量推断锁瓶颈。
结合正常运行的分段等待，下一步优先核对每次选路的队列投影/分组重复计算。
不据此删除既有投影缓存或修改 planner 默认值。

脚本 `/tmp/flexlb-current-perf.sh`、`/tmp/flexlb-current-planner16.sh`；
源清单 `/tmp/flexlb-current-perf.sha256`；归档 `/tmp/flexlb-current-perf-results.tar.gz`、
`/tmp/flexlb-current-planner16-results.tar.gz`；本地解包 `/tmp/flexlb-current-perf-20260925/`，
其中 test/profile/planner16.log 与对应 XML/exit，events.json、thread-samples.txt 可复核。
所有远端执行及下载会话已正常结束；脚本完成不代表其中 Maven 测试通过。


## 2026-09-25 14:22：无到期窗口复用计划

生产 26,445 行。RouteTimelineProjector 在推进收集窗口且严格早于全快照最早到期时，
复用同一选择和预测；可能有到期时仍 prune/replan。指定远端默认 clean test，无线程覆盖/JFR。
485 个 Java/POM 文件运行前后 SHA256 全部通过。

- offered 11339.6 QPS；client 5561.1 QPS；Master 5654.5 QPS。
- Master P50 447 ms、P99 746 ms、mean 438.728 ms。
- route_submit P99 433 ms、batch_wait 318 ms、dispatch_ack 22 ms、ack_response 1 ms。
- 513 批，平均 15.97，最大 16；Maven exit 1，XML 1 failure / 0 errors / 0 skipped。

仍未通过 250 ms 门槛。此前相邻默认运行 P99 792 ms，但 offered 10865.1 QPS，单次顺序
对照不能证明这次优化导致端到端改善。保留的依据是无到期分支的重复规划确实被删除，
且冻结快照的选择结果等价；不是性能已经达标。
脚本 `/tmp/flexlb-window-perf.sh`；清单 `/tmp/flexlb-window-perf.sha256`；
归档 `/tmp/flexlb-window-perf-results.tar.gz`；解包 `/tmp/flexlb-window-perf-20260925/`。
运行和下载均已终结，无环境阻碍。
