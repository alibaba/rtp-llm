# 按选路完成顺序提交：验证记录

实现：`280f2e0a4e`。修改前基线：`4904d00594`。
逐行说明：[代码讲解](../../global-queue-completion-walkthrough.zh-CN.md)。

**功能验证通过，性能验收尚未全部通过。** 不把已出现的性能失败改成成功，也不放宽门槛。

## 功能验证

```sh
./mvnw -B -P 'opensource,!internal' -pl flexlb-api,flexlb-mock-engine -am test
```

完整运行 5 分 30 秒，184 个测试类，1495 个测试，0 失败、0 错误、1 跳过。
保存了完整压缩日志和结构化计数。这是实现 commit 的验证；随后另一会话修改的工作区不包含在此结论内。

新增/更新的契约：慢选路不阻挡其他结果、名额可持续补充、取消不能绕过在途上限、关闭时清理晚到结果、
优先级/FIFO决定下一个名额、唤醒不作废在途工作。严格最终 FIFO 只在单 planner 基线中保留。
10,000 请求积压测试仍然只在一次 Decode 容量释放后新增 2 次尝试（一次成功、一次确认满），
重复心跳不产生新的选路；没有降低原有过载约束。

## 性能参数和测试边界

- 750P/750D；BATCH/FIXED_WINDOW 与 NON_BATCH/SINGLE。
- 3000 和 10000 offered QPS，各 10 秒预热、10 秒测量；实发 30,000 / 100,000 条测量请求。
- Java 21，固定 8 GiB heap，默认 15 planner；client、Master、mock 在同机测试 JVM。
- 完成 QPS 至少为目标的 98%；client/server P99 均不超过 250ms；BATCH 等待 P99 不超过 50ms。
- 保留原始实际长度 payload，未缩短请求、跳过迟到发压时隙或修改生产线程数。
- BATCH 使用 750 个真实 Netty mock Prefill RPC server 和 750 个逻辑 Decode endpoint。
  NON_BATCH 验证 client→Master→路由响应，不发送 EnqueueBatch。
- 不包含 GPU 推理或输出流；模拟状态终止可以先于 ACK，原始 `terminal_without_ack_count` 已保留。
- 两种模式的 E2E 边界不同，不能只拿两行 QPS 宣称 batching 本身造成了某个倍率的差距。

## 独立源目录结果

| 版本 | 模式 | 目标 QPS | client 完成 QPS | Master 完成 QPS | client P99 ms | Master P99 ms | 门槛 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 修改前 | BATCH | 3000 | 2996.8 | 3002.6 | 155.514 | 51 | 通过 |
| 修改后 | BATCH | 3000 | 2989.8 | 2996.8 | 258.808 | 220 | client P99 未通过 |
| 修改前 | BATCH | 10000 | 4934.4 | 4937.6 | 10252.514 | 9950 | 未通过 |
| 修改后 | BATCH | 10000 | 6234.5 | 6241.2 | 6379.151 | 6080 | 未通过 |
| 修改前 | NON_BATCH | 3000 | 2999.7 | 3000.3 | 308.606 | 270 | P99 未通过 |
| 修改后 | NON_BATCH | 3000 | 2999.8 | 3000.2 | 10.098 | 3 | 通过 |
| 修改前 | NON_BATCH | 10000 | 9910.5 | 9911.6 | 138.586 | 137 | 通过 |
| 修改后 | NON_BATCH | 10000 | 9991.4 | 10001.2 | 214.124 | 206 | 通过 |

这些测量请求均通过成功响应/覆盖校验；失败的是相应性能断言。该表不能用作严格的独占机器 A/B：
另一 Codex 会话间歇运行 Maven，已请求协调独占测量窗口，尚未得到确认。源目录隔离解决构建冲突，
不等于隔离了 CPU/内存竞争。保留结果，但不根据单次波动宣称稳定提升或无性能回归。

## 首轮、复测与诊断

- 原工作区首轮 BATCH：3K 为 1730.0 QPS / client P99 8648.772ms，10K 为 4688.1 QPS /
  client P99 11750.351ms，均失败。随后 NON_BATCH 编译遇到另一个 Maven 同时重建 target/classes，
  出现 `NoSuchFileException`。这次 NON_BATCH 没有产生有效性能测量。
  不能由这次构建冲突推断首轮所有 BATCH 退化都由另一个进程造成。
- 将前后两个 commit 分别归档到独立目录，并按相同脚本重测，得到上表。
- 又一次 NON_BATCH 复测：3K 为 2999.7 QPS / client P99 27.692ms；10K 为 9826.3 QPS /
  client P99 240.263ms，两档通过。该 NON_BATCH JVM 没有开启 JFR。
- BATCH 的 JFR 诊断轮：3K 为 2996.3 QPS / client P99 94.128ms，通过；10K 为 4746.6 QPS /
  client P99 11982.890ms，未通过。因为开启了采样，不与无 profiler 的表格混为严格 A/B。
- 70 秒记录中有 3931 个执行样本，1446 个栈包含 `DefaultRouter.selectRole`，1029 个包含
  `CostBasedPrefillStrategy.evaluateCandidates`。栈会重叠，记录也含准备/预热，不能当成纯测量窗口的耗时比例。
- 133 个 GC 事件累计暂停约 4.722 秒，最长暂停约 1.008 秒。这里累加的是 `sumOfPauses`，
  没有把并发老年代周期的整个 duration 当作 stop-the-world 时间。
- 证据指向继续检查候选计算、分配与 GC；尚不足以证明只优化其中一个点就能通过 10K 门槛。
  没有为了某一轮成绩向协调器增加路由捷径、特殊批处理分支或放宽 SLO。

## 复现及原始证据

```sh
./tools/run_queue_performance.sh /tmp/flexlb-queue-perf
```

完整结果：[results.json](results.json)。采样摘要：[profile-summary.json](profile-summary.json)。
所有首轮、独立目录、基线、采样/复测日志都以 `.log.gz` 保留，未删除失败样本。
原始 JFR 含运行环境元数据，仅保存在本机 `/tmp/flexlb-completion-20260913/`，没有提交。

尚待完成：无其他压测竞争时复测，以及 BATCH 10K 吞吐/尾延迟门槛的定位和优化。
本报告不表示用户要求的性能验收已经全部完成。
