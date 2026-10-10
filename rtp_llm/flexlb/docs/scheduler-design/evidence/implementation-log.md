# 实施记录

## 记录范围与当前交付状态

本文记录 EngineLocalView 批次：只保留线程池生命周期修复，计算算法逐字保持原实现；新增 13 个功能测试、1 个性能测试，并接入现有 sync-performance-regression profile。原有 241 个测试文件未修改或删除。**该类及依赖模块 UT 通过，整体功能/性能验收未全部通过，不代表完整类关系重构已完成。**

随后完成的 Decode `ReleaseReason` 删除和取消命名统一单独记录在 [枚举治理第一批实现](enum-implementation-round1.md)。其功能用例通过，API 性能门槛有 1 项未通过；这两批都不代表核心实体重构已完成；原七状态 RequestLifecycle 与 DeliveryBatch 路线已由 [最终方案](../final-design.md) 撤回。

## 执行约束

保留已有功能 case，不删除测试、不降低性能阈值。先建立旧实现基线，再增加行为测试证明需要改变的约束；逐个生产文件修改，每个文件修改后运行对应模块 UT，通过后才开始下一个生产文件。每批运行相关功能和独立性能回归。本文记录实际证据，未执行项不标为完成。

完整类关系迁移按 verification.md 分批推进。本轮从一个有直接代码证据、可独立验证的内部类开始，不同时变更调度线程模型和跨请求事务。

## 批次 01：EngineLocalView

目标：先验证执行边界是否值得改变，再修复有证据的资源生命周期问题。最终仅增加线程池关闭，保留全部差异算法、统计语义和上层代际锁。

证据：calculateDiff 对新增/移除集合分别提交 ForkJoinTask，内部又调用 parallelStream，随后 join；GrpcCacheStatusCheckRunner 在 worker 锁下调用此路径；类创建 customPool 但没有关闭路径。差异计算没有 IO，同步调用方只需要最终结果。

设计约束：

- 最终计算路径与 HEAD 完全相同，不新增分支、线程池或业务锁；现有线程池增加明确关闭边界。
- 只为新增和移除成员建立结果集合，不复制两个完整输入集合；输出不随输入随后变化。
- 不改变现有并发 engineViews 的读写 API，也不声称其裸 API 支持同 worker 并发事务快照。
- 保留每次有效差异计算的指标与动态间隔统计；非法输入仍返回空结果。
- 实现前后独立运行相同规模/变更比例的性能用例，测量值与功能覆盖分开报告。

新增测试：EngineLocalViewTest（输入/结果隔离、空值、无变化、初始化/清空、固定种子差异对照、关闭幂等、拒绝新计算及等待在途读取）；EngineLocalViewPerformanceTest（32/1024/16384/65536 个 key，无变化/50% 替换，真实实现，无 Mockito 热路径）。性能测试 caller_bytes 只统计调用线程，不包含旧实现池内分配，不能据此计算总分配改善百分比。

测试日志目录：`/tmp/flexlb-implementation-baseline/`。最终执行结果见本文验收表。

### 实测导致的修订

全串行试验没有保留：虽然小集合明显变快，65536 个 key 的无变化和替换场景出现性能回退。减少重复扫描后仍不能覆盖所有大集合场景；单任务依次执行两个并行扫描也没有稳定达到原有表现。随后尝试保留大集合双任务并行、仅小集合直接扫描；此候选也因下列对照结果被撤回。候选试验日志 cache-perf-after-*、cache-perf-v2/v4 不代表最终实现；首轮候选的后续测试链已主动终止，不能计作最终验收。

兼容性自检额外发现 HashSet 会接受原并发集合拒绝的 null key：先增加失败测试，再保留原拒绝行为。最终新增 13 个功能用例，并为无变化报告增加调用线程分配小于 1024 B 的性能检查；大集合仍有池内分配，该指标不代表进程总分配。

### 改动前已有的性能失败

原有全量功能基线通过：1837 个用例，0 failure / 0 error，1 个已有 skip。sync 性能基线通过。API 性能基线 16 个用例中 15 个通过，8192 请求突发用例 P99 为 964 ms，超过原有 250 ms 门槛；未改生产代码单独复测为 955 ms。首轮阶段指标 route_submit_p99=763 ms、batch_wait_p99=254 ms，只能定位耗时区间，尚不能确定根因。门槛未修改，不能将此基线问题描述为随机抖动或已解决。

### 已撤回候选的缓存微基准（不代表最终实现）

同一已编译基准，先用 HEAD 的 EngineLocalView 源文件编译旧类并仅替换该 classpath 项；其余依赖一致。旧/新实现各 3 个独立 JVM，交替顺序，固定 -Xms512m/-Xmx512m；每种输入预热 3000 次，5 轮各 300 次测量，取每 JVM 中位数后再取三次中位数。CPU 为本机环境，本结果不等同于线上端到端收益。

| key 数 | 输入变化 | 原实现 µs/op | 撤回候选 µs/op | 耗时变化 |
| --- | --- | ---: | ---: | ---: |
| 32 | 无变化 | 42.162 | 0.702 | -98.3% |
| 32 | 50% 替换 | 39.136 | 2.011 | -94.9% |
| 1024 | 无变化 | 43.315 | 19.984 | -53.9% |
| 1024 | 50% 替换 | 55.415 | 25.609 | -53.8% |
| 16384 | 无变化 | 87.461 | 87.222 | -0.3% |
| 16384 | 50% 替换 | 377.512 | 398.821 | +5.6% |
| 65536 | 无变化 | 269.715 | 258.000 | -4.3% |
| 65536 | 50% 替换 | 941.015 | 1012.299 | +7.6% |

16384 key 替换场景原实现范围 376.851—382.698 µs，新实现范围 377.100—409.507 µs，中位数约慢 5.6%；65536 key 替换场景中位数约慢 7.6%。不能据此宣称所有大集合无回退，需要在稳定性能环境继续确认。没有通过删去该规模或放宽已有阈值掩盖这一结果。因此未保留小集合优化和大小分支，避免把未证明的性能取舍混入生命周期修复。

基准原始日志：`/tmp/flexlb-implementation-baseline/comparison-{baseline,candidate}-{1,2,3}.log`；对照脚本和编译输出保留在同目录。当前版本可通过以下命令复跑：

```sh
./mvnw test -P '!internal'
./mvnw test -P '!internal,sync-performance-regression' -pl flexlb-sync -am
./mvnw test -P '!internal,api-performance-regression' -pl flexlb-api -am
# 仅缓存微基准；复制相同测试文件到旧版本也可执行此命令。
./mvnw test -P '!internal,sync-performance-regression' -pl flexlb-cache -am -Dtest=EngineLocalViewPerformanceTest
```

### 最终保留的生产变更

仅 EngineLocalView：实现 AutoCloseable，并增加带 @PreDestroy 的 close()，调用原有 customPool.close() 停止接收任务并等待已提交计算退出。原有 calculateDiff、成员索引和统计调用逐字保持 HEAD 实现；无线程阈值、HashSet 替换、新状态机或额外回调。新增关闭等待测试允许 close 与第二个任务提交竞争时返回 RejectedExecutionException，但验证已提交读取结束之前 close 不返回。

原有 241 个测试文件的 SHA-256 对照无改写/删除。新增测试中的“强制调用线程读取”属于已撤回方案的约束，也随方案撤回，替换为实际保留的关闭等待行为测试；没有删除原有基础 case。

最终前一轮全量验证中，common/cache/grpc/sync/API 均通过；mock-engine 的 ProductionCaliberDecodeTest.lowBatchDrainRateMatchesProductionAnchor 测得 410 tok/s，低于 519±15% 下限。源码路径直接调用 Mock decode，不经过 EngineLocalView。隔离复测仍失败，不标为已解决。

## 最终验收表（2026-09-23，JDK 21，关闭 internal profile）

| 验证 | 实际结果 | 原始日志 |
| --- | --- | --- |
| 最终类及上游 UT | common 225 + cache 50 全部通过，含 13 个新增用例 | cache-lifecycle-final.log |
| Maven profile 修改后 UT | common/cache 通过，性能测试保持独立执行 | pom-ut.log |
| 最终全量功能 | 1850 个用例：1 failure、1 error、1 个已有 skip；common/cache/grpc/sync/API 无 failure/error，mock-engine 失败如下 | functional-accepted.log |
| Mock 计时吞吐 | 全量 392 tok/s，低于 519±15%；独立复测 403 tok/s 仍失败 | functional-accepted.log；mock-anchor-isolated.log |
| Mock 端口绑定 | 全量绑定 64640 报 Address already in use；检查时已无占用，隔离复测该类 5/5 通过；未归因到特定进程 | functional-accepted.log；mock-port-isolated.log |
| cache/sync 性能 | 4 个用例全部通过，覆盖缓存差异、分配、队列捕获及诊断读取 | sync-perf-accepted.log |
| API 性能 | 16 个用例，3 个失败：突发 P99=983 ms >250 ms；目标10000 QPS 时实测6119.7 <8500；另一规模的 delivery wait P99=57 ms >50 ms | api-perf-accepted.log |

日志位于 `/tmp/flexlb-implementation-baseline/`。API 性能夹具的 createRouter 使用 mock CacheAwareService，不经过本次 EngineLocalView；Mock 计时用例直接构造 decode 服务。API 突发门槛在生产代码修改前已连续两次失败。额外规模场景失败的原因尚未确定，不能宣布为偶发、已解决或通过降低阈值消除。现有 skip 来自 FollowerAsyncForwardingNettyTest，未新增跳过项。

### 交付自检

- **代码清晰**：最终生产差异仅 AutoCloseable、@PreDestroy 和 close 方法，无算法阈值或新状态。
- **流程清晰**：同步计算路径保留；容器销毁或调用方关闭时，等待原池已提交工作退出。
- **类内聚**：创建线程池的类负责关闭，不把释放责任推给调用者的业务回调。
- **验证边界**：原计算方法及其余原成员逐字对照 HEAD；文档链接、代码块和 diff whitespace 检查通过。没有宣称整个重构或全部性能门槛已完成。

下一批前先定位上述基础性能失败并获得可信基线，再按 verification.md 的类职责顺序继续迁移；不将撤回的算法试验带入后续实现。
