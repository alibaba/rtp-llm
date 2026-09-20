# 合并前 Review 与验证

范围：`b2cbe9c416..112c26fba0` 及本次移除 Response 内部诊断字段的修改。

未发现阻塞性代码问题。重点检查了错误码与 reason 的一致性、诊断从 PlacementResult 到 Context/PV 的传递、超时与迟到调度交错、终态释放 Context、队列优先级计数和只在需要时构造的诊断快照。Review 阶段仅展开了两个测试文件的通配符 import；全部生产源码仍与远端性能测试快照逐文件一致。

## 验证结果

- 本机 `./mvnw -B -P 'opensource,!internal' package`：1774 项，1772 通过、1 失败、1 跳过；因测试失败整体退出码为 1，没有将此次完整执行报告为全绿。
- 失败为 `ProductionCaliberDecodeTest.lowBatchDrainRateMatchesProductionAnchor`：396 tok/s，要求 519 ±15%（下限 441.15）。该测试和 Mock Engine 生产实现均未在被 review 的改动中修改。
- 隔离导出改动前 `b2cbe9c416`，本机仅运行同一测试类：低 batch 同样失败，438 tok/s；其余 7 项通过。当前完整测试与基线隔离测试没有在本机并行运行。
- 当前代码在 B300/Dragonwell 21 上复核同一测试类并打包：8 项全部通过，低 batch 510 tok/s、高 batch 7688 tok/s、单流 129 tok/s，退出码 0。这支持本机计时环境敏感的判断，不据此比较基线和当前吞吐优劣。
- 唯一跳过项为 `FollowerAsyncForwardingNettyTest.benchmarkFollowerCapacityAgainstDelayedMaster`，因未设置 opt-in 属性 `flexlb.forwarding.capacity.benchmark`。
- `LeakCanaryLongRunE2ETest` 通过，持续发送约 62 秒，覆盖故障期间的请求资源收敛；整个用例耗时 64.513 秒。这不是长时间堆留存证明。
- 本次修改涉及的 Java 文件通过 Spotless 检查，`git diff HEAD^ --check` 通过。
- 远端 4 项 Sync 性能测试、750P/750D 的 4 档 E2E 性能门禁全部通过，详见 [性能报告](README.md)。

[完整验证日志与关键测试 XML](review-logs.tar.gz)
