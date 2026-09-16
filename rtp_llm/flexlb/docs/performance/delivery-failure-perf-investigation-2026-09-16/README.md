# 750P/750D 性能失败原因与最终验证

## 定位结论

原测试混入了两项可以复现的测试开销。定位过程没有修改生产逻辑、增加 heap、延长预热或放宽性能阈值。

### 1. 10k：ConfigService mock 保存调用历史

`FlexLBMockTestBase` 使用普通 Mockito mock 提供配置。每次 `loadBalanceConfig()` 调用都会保存调用记录；这些集成测试只需要返回配置，并不验证调用次数。本次 Dispatcher 新增逐批读取 `fetchAttachTimeoutMs`，让测试额外记录大量调用。真实 `ConfigService.loadBalanceConfig()` 直接返回配置字段，没有这个记录行为。

先对旧 reviewed 快照和 HEAD 各复制一份，**只把该 mock 改成 `stubOnly()`**，按基线、改后、改后、基线顺序测试：

| 10k BATCH | Client avg ms | Client P50 ms | Client P99 ms | Delivery wait P99 ms |
| --- | ---: | ---: | ---: | ---: |
| 基线 1 | 12.702 | 11.019 | 40.176 | 33 |
| 改后 1 | 12.510 | 10.991 | 37.790 | 32 |
| 改后 2 | 13.262 | 11.004 | 44.804 | 32 |
| 基线 2 | 12.998 | 10.997 | 41.856 | 36 |

10k 差距消失，四次都满足原阈值；3k 仍然失败。因此这一步不能宣告整个性能测试通过。原始单次 32.512ms 对 14.706ms 的差距不能直接当成生产路径回退。

证据：[完整摘要](stub-summary.txt)、[结构化结果](stub-results.json)。共享此 fixture 的六项 API 集成测试也已在本地重跑通过。

### 2. 3k：预热和测量切换了 request_id 的 protobuf 编码宽度

750P/750D、3k 时，旧预热 ID 从 `57,503,000` 开始，使用四字节 varint；正式测量 ID 从 `760,300,000` 开始，使用五字节 varint。

分段计时与 JFR 对齐发现：

- 测量开始约 13ms，`CodedInputStream.ArrayDecoder.readRawVarint64()` 出现 `unstable_if / reinterpret` 反优化，包含 `flexlb-dispatch-executor` 线程中的 GenerateInputPB 解析调用。
- 测量第一秒发生 829 次超过 50ms 的派发执行器排队、75 次超过 50ms 的请求构建；最长构建约 383ms。后续仅另有一次 51ms 构建。
- CPU 采样集中于 `DefaultBatchDispatcher.buildInput()` 下的 protobuf token 列表解析。该慢尾发生在组批完成之后、真正调用 RPC 之前。
- 这解释了 3k 的 delivery wait P99 约 11ms，batch wait P99 却达数百毫秒；它们的结束计时点不同。
- 后续 10k 阶段已经执行过两种解码路径，没有同样的测量起始慢尾。

第二个单因素实验只改变 ID 范围，使预热和测量都使用八字节 varint；预热仍为十秒，负载、请求 token、SLO 均不变。改后、基线、基线、改后的四轮 BATCH 全部通过。前两轮 3k Client P99 分别为 24.586ms 和 32.973ms。

最终 fixture 使用 `(1L << 60) + 原 ID`，预热和测量都走九字节 int64 解码路径，与现场长 request_id 的编码宽度一致。实际发送的字段没有减少。

证据：[JFR 事件和调用栈](build-jfr.txt)、[对照摘要](id-summary.txt)、[结构化结果](id-results.json)。分段计时仅加在远端诊断副本，未加入业务代码。

### 其他观察及边界

未修正 fixture 的 JFR 中，10k 存在明显 GC 压力；一次改后运行发生约 256ms Full GC。`EndpointRoundRobin.next()` 也有锁竞争。这些是实测现象，但单凭热点不能认定它们是此次业务改动的回退原因。单因素对照排除了继续修改 Slot 锁或路由逻辑来迎合测试的必要性。

这是 Master/client/mock 同 JVM loopback 测试，不能据此推导真实 GPU wall TPS 或线上 GC 状态。请求 finished 可能先于 ACK，测试保留两类完成计数及完整样本覆盖检查。

## 代码复查

此前复查发现 `RequestRegistry.projectPrefillRetirementItem()` 按请求尚未终态筛选 Slot，会丢掉“已返回失败、资源仍待清理”的 Prefill 退休通知。本次改为定位当前 Slot，再由 Slot 校验精确 ScheduledRequest 和 Endpoint。新增回归覆盖 Decode finished 在退休之前或之后到达；修复前两例失败，修复后通过。

当前快照进一步核对了失败响应与清理责任分离、关闭入口统一检查、清理进度单调合并、抢占完成依据、到期决定保留、准入/抢占退出后的继续处理，以及旧身份和重复回调。当前全量回归包含这些并发路径，未发现其他可复现的阻断问题。

## 当前代码最终验证

用户确认其他修改已结束后，16:41:13 冻结当前生产代码；之后仅将上述 ID 修正加入测试 fixture。同步 475 个文件（472 个入库构建输入，另有 3 个本地生成文件，见提交前核对），使用逐文件 SHA-256 校验，基线为 `c738fd4034efd8c5816a353674e9bc99fa590531`，两边使用相同的修正后测试 fixture。

- 远端：`luoli.hn@11.163.39.110`，容器 `luoli_gpu`，Dragonwell 21.0.11.0.11。
- 目录：`/data0/luoli.hn/work/rtp_llm_4/flexlb-perf-investigation-20260916`。
- 当前源：`frozen-current/source`；最终基线：`baseline-final`。
- 参数：750P/750D，8 GiB heap，3000 / 10000 QPS，每档预热 10 秒、测量 10 秒；BATCH 使用 10ms FIXED_WINDOW / maxRequests=16，NON_BATCH 使用 SINGLE。
- 原门槛：Client/Master QPS ≥目标 98%，Client/Master P99 ≤250ms，Delivery/Batch wait P99 ≤50ms。
- 顺序：当前 BATCH、基线 BATCH、基线 BATCH、当前 BATCH、当前 NON_BATCH。验收运行没有 JFR 或分段计时插桩。

**最终验证通过。** 当前代码 1158 项单元测试、6 项共享 fixture 集成测试全部通过；当前版本两轮 BATCH 和一轮 NON_BATCH、基线两轮 BATCH 的共十档测量全部满足原门槛。远端运行前后 475 个文件的 SHA-256 一致，其中 472 个入库构建输入与 Git 索引一致。

延迟单位 ms；每轮独立 JVM。两轮均保留，没有挑选较快的一轮。

| 版本 / 模式 | QPS | Client QPS | Client avg | Client P50 | Client P99 | Master avg | Master P99 | Delivery / Batch wait P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| current-1 | 3000 | 2996.8 | 11.837 | 10.984ms | 46.477ms | 10.526ms | 16ms | 11ms / 10ms |
| current-1 | 10000 | 9960.9 | 14.811 | 11.049ms | 85.392ms | 11.512ms | 37ms | 35ms / 34ms |
| baseline-final-1 | 3000 | 2996.8 | 11.438 | 10.956ms | 31.159ms | 10.394ms | 13ms | 11ms / 10ms |
| baseline-final-1 | 10000 | 9988.4 | 13.941 | 11.046ms | 46.056ms | 11.835ms | 39ms | 37ms / 36ms |
| baseline-final-2 | 3000 | 2989.2 | 12.167 | 10.987ms | 52.827ms | 10.748ms | 23ms | 11ms / 11ms |
| baseline-final-2 | 10000 | 9988.9 | 13.053 | 11.025ms | 39.353ms | 11.338ms | 34ms | 32ms / 32ms |
| current-2 | 3000 | 2996.8 | 11.571 | 10.961ms | 32.664ms | 10.448ms | 15ms | 11ms / 10ms |
| current-2 | 10000 | 9968.2 | 13.107 | 11.020ms | 42.728ms | 11.261ms | 38ms | 36ms / 36ms |
| current-nonbatch | 3000 | 2999.5 | 1.240 | 0.523ms | 31.076ms | 0.035ms | 1ms | 1ms / N/A |
| current-nonbatch | 10000 | 9994.9 | 0.997 | 0.500ms | 21.956ms | 0.094ms | 3ms | 1ms / N/A |

当前 BATCH 10k 两轮平均延迟 14.811 / 13.107ms，基线为 13.941 / 13.053ms。仍有短窗口波动，但没有重现原来的两倍平均延迟差距，各项门槛均通过。

[最终原始摘要](final-summary.txt) · [全部指标](results.json) · [验证汇总](verification.txt)

完整日志和单元测试 XML 保存在远端 `final-results.tar.gz`，本地同目录也留存该归档。此前失败结果保留在 [初次复查数据](initial-results.json)，不能用本轮成功覆盖此前失败记录。初次 reviewed / baseline 的完整日志分别位于远端 `flexlb-delivery-failure-review-20260916-reviewed` / `flexlb-delivery-failure-review-20260916-baseline` 的 `results.tar.gz`。

[冻结源码清单](source-manifest.json) · [SHA-256](source.sha256) · [最终验证脚本](final-runs.sh)

## 提交前产物核对

同步清单包含一个 `.DS_Store` 和两个位于 `src/main/java` 的旧 `.class`，未把它们加入提交。清单的 `non_source_files_excluded_from_commit` 明确列出这三个文件；其余 472 个输入逐项核对 Git 索引中的内容。

在远端移除三项后执行 `./mvnw -B -P 'opensource,!internal' -pl flexlb-api -am -DskipTests clean test`，重新生成业务和测试字节码，逐文件比较已有验收运行产物。结果见 `clean-build-equivalence.json`。
