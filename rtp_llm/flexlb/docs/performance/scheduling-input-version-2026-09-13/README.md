# schedulingInputVersion 整理：验证记录

基线：`dfd4b75d40`；实现：`fb0cb2bb79`。本次生产改动仅在 `WorkerBatcher.java`。
实现说明见[调度输入版本与快照缓存](../../worker-batcher-projection-cache.zh-CN.md)。

## 改动

- 输入版本从 `AtomicLong` 改为 `volatile long`：写入在同一把锁下，锁外检查通过 volatile 读取。
- 版本递增集中在 `schedulingInputsChangedUnderLock()`，是否唤醒线程由各事件处理方法决定。
- 将三个已有版本组合成 `ProjectionVersion`，消除重复比较。
- `ProjectionSource` 携带一次锁内读取的原料及版本；排序和快照生成保持在锁外。
- 缓存校验和发布在同一临界区内完成，保持旧快照不能重新命中缓存的约束。

没有新增版本计数器、线程、队列或模式开关。缓存命中不创建新对象；未命中增加两个小的原料/版本对象，
并在发布时重新获取锁。这里保留性能检查，避免仅根据代码外观判断运行开销。

## 验证方法

源码从基线归档到独立目录，覆盖唯一修改的生产文件；所有已跟踪 Java 文件逐一与工作区比较，
一致后记录在 `source-sha256.json`。排除了工作区其他未跟踪重构文件。

最终验证均串行执行：

```sh
./mvnw -B -P 'opensource,!internal' -pl flexlb-api,flexlb-mock-engine -am test

./mvnw -B -P 'opensource,!internal,sync-performance-regression' \
  -pl flexlb-sync -am -Dtest=WorkerBatcherPerformanceTest \
  -DfailIfNoTests=false -Dsurefire.failIfNoSpecifiedTests=false test

./tools/run_queue_performance.sh /tmp/flexlb-input-version-e2e
```

缓存读取性能分别在旧、新源码目录运行。端到端沿用 750P/750D、3000/10000 offered QPS、
BATCH/FIXED_WINDOW 和 NON_BATCH/SINGLE、8 GiB heap、10 秒预热和 10 秒测量，SLO 没有放宽。
归档目录脚本只替换 git 元数据读取，不改变性能测试逻辑或参数。

完整矩阵按新版→旧版各运行一轮；因首轮新版 BATCH 3000 出现吞吐回落，再针对该档按新版→旧版各复测一次。
机器仍有桌面进程，两次也未交换版本顺序，不能据此宣称稳定提升或排除所有性能回退。
BATCH 10K 的吞吐和 RT 门槛仍未通过。

## 最终结果

完整 UT：184 个测试类、1497 项测试，0 失败、0 错误、1 跳过；耗时 5 分 30 秒。
日志：`unit-tests-serial.log.gz`；逐类计数：[unit-tests.json](unit-tests.json)。

缓存读取微基准（每档预热后，5 轮各 500 次，取轮次中位数）：

| 队列深度 | 旧版 ns/op | 新版 ns/op | 旧/新分配 bytes/op |
| ---: | ---: | ---: | ---: |
| 0 | 200 | 234 | 0 / 0 |
| 1 | 69 | 72 | 0 / 0 |
| 32 | 66 | 79 | 0 / 0 |
| 128 | 67 | 80 | 0 / 0 |
| 512 | 197 | 79 | 0 / 0 |

两版均通过原有微基准门槛。这是缓存命中测试，不覆盖并发未命中时新增的原料对象和发布锁开销，
不能用这些数字代替端到端测试或宣称普遍提速。

| 轮次 | 版本 | 模式 | 目标 QPS | client QPS | client P50 ms | client P99 ms | 性能门槛 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 矩阵 | 新 | BATCH | 3000 | 2801.7 | 13.542 | 704.511 | 未通过 |
| 矩阵 | 新 | BATCH | 10000 | 5845.7 | 2220.213 | 9369.093 | 未通过 |
| 矩阵 | 新 | NON_BATCH | 3000 | 2999.9 | 0.396 | 5.384 | 通过 |
| 矩阵 | 新 | NON_BATCH | 10000 | 9998.1 | 0.833 | 157.506 | 通过 |
| 矩阵 | 旧 | BATCH | 3000 | 2996.5 | 12.660 | 274.351 | 未通过 |
| 矩阵 | 旧 | BATCH | 10000 | 5796.3 | 2903.429 | 9051.499 | 未通过 |
| 矩阵 | 旧 | NON_BATCH | 3000 | 2999.8 | 0.753 | 20.391 | 通过 |
| 矩阵 | 旧 | NON_BATCH | 10000 | 9998.0 | 0.864 | 130.670 | 通过 |
| 3K 复测 | 新 | BATCH | 3000 | 2996.8 | 11.165 | 126.446 | 通过 |
| 3K 复测 | 旧 | BATCH | 3000 | 2996.8 | 11.752 | 406.925 | 未通过 |

BATCH 包含客户端到 Master 再到 mock Prefill RPC 的调度交互；NON_BATCH 测到 Master 的路由响应。
这些 RT 不包含 GPU 推理和完整输出流，两个模式只各自比较前后版本。

结论：NON_BATCH 两档前后均通过。BATCH 3000 首轮新版吞吐下降，复测恢复约 2997 QPS，P99 为 126ms；
旧版两次吞吐达标但 P99 均超标。首轮新版的回落未在复测重现，仍保留该失败样本，不能排除所有回退。
BATCH 10000 前后吞吐均约 5800 QPS、P99 约 9 秒，既有性能门槛仍未通过。

本次有依据的结论是：代码同步规则收拢，完整功能回归通过，缓存命中仍不分配对象。
端到端数据波动明显，没有证明稳定提升，也不能声称性能验收全部通过。

全部原始指标：[results.json](results.json)。完整日志以 `.log.gz` 保存，首轮及复测均保留。
表中门槛按打印的吞吐/延迟核对；原始构建失败断言见日志。


## 未采用的验证

第一次编译发现嵌套 `WorkerBatcher.QueueSnapshot` 与投影 `QueueSnapshot` 同名；已恢复使用完整类名。
随后我在旧 JVM 尚未退出时重启测试，发生运行重叠，该轮 `WorkerOfflineTest` 收到 HTTP/1 响应而使 gRPC 失败。
该次运行不能作为本次改动的可靠验证，已停止全部本任务 JVM 后重新串行执行；没有修改该测试或放宽断言。
对应日志保留在本轮目录，最终结论只使用最终串行运行。
