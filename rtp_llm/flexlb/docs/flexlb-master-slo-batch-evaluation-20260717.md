# FlexLB Master SLO Batch 性能评估

## 1. 结论

在不修改调度策略、固定 `SCHEDULE_WORKER_SIZE=16` 的条件下，FlexLB Master 的 batch Schedule 吞吐达到目标：

- `REPLAY_SPEED=650`：Master arrival QPS `5173.321`，completion QPS `5340.350`，154771 个请求全部完成，错误数 0。
- `REPLAY_SPEED=1400`：Master arrival QPS `10630.998`，completion QPS `10076.521`，333590 个请求全部完成，错误数 0。

10K 档的 P50 为 161 ms，P90/P95/P99 分别为 1461/2039/2891 ms。长尾主要在 Master gRPC 入口排队，P99 为 2755 ms；batch wait P99 为 177 ms，dispatch ACK P99 为 499 ms。因此结论是“吞吐达到 10K，但 10K 下的尾延迟未受控”，不能写成 10K 下同时满足低延迟 SLO。

## 2. 测试边界

测试链路为：

`trace 到达时间 -> Schedule RPC -> Master 路由 -> fixed-window 凑批 -> EnqueueBatch -> engine ACK -> Schedule 返回`

本测试设置 `SCHEDULE_ONLY=1`，不调用 `FetchResponse`。Master 延迟在服务端打点，从 RPC 到达 Master 开始，到 Schedule response 完成为止，不使用 client RTT 作为正式延迟。

## 3. 固定配置

有效性能轮次的代码基线为 `716a621d5`，加上本文涉及的 mock、打点和测试工具改造。远端在测试完成后新增的 `c92e204e0`、`e74cccd6c` 公式/预测策略变更不在本报告验证范围内；合入这些提交后的容量和延迟需要用相同矩阵重新做 A/B。

| 配置 | 值 |
|---|---:|
| Master worker | `SCHEDULE_WORKER_SIZE=16` |
| Load client worker | 8 |
| Prefill/Decode engine | 750/500 |
| 调度模式 | `batch` |
| Batch 算法 | `fixed_window` |
| `FLEXLB_BATCH_PREDICT_THRESHOLD_MS` | 500 ms |
| `FLEXLB_BATCH_FIXED_WAIT_MS` | 160 ms |
| `FLEXLB_BATCH_SIZE_MAX` | 32 |
| `COST_SLO_MS` | 1000 ms |
| Master/Java mock heap | 32 GiB / 32 GiB |
| Mock 性能模型 | `dsv4_flash_performance.formula_1x.json` |
| 流量 | 原始 trace 毫秒分布按 `REPLAY_SPEED` 等比例压缩 |

Master 配置文件为 `tools/online_eval/data/config/master_fixed_window_slo500_wait160.json`。所有压力档固定 Master 和 load worker 数，避免把 worker 变化误判为算法或 Master 容量变化。

`fixed_window` 中 500 ms 是预测时间触发阈值，不是端到端硬 SLO。批次在以下任一条件满足时 dispatch：预测 prefill 时间达到 500 ms、最老请求等待达到 160 ms、batch size 达到 32。`COST_SLO_MS=1000` 仅用于成本估算和结果校验，该算法不会以它作为硬 deadline。

## 4. 有效压力矩阵

| 目标档 | Replay speed | Client send QPS | Master arrival QPS | Master completion QPS | 请求数 | Error | Pacing P99 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 14 | 107.571 | 107.701 | 108.068 | 6454 | 0 | 2.527 |
| 1K | 130 | 1024.057 | 1025.334 | 1033.229 | 61431 | 0 | 2.365 |
| 5K | 650 | 5159.836 | 5173.321 | 5340.350 | 154771 | 0 | 7.405 |
| 10K | 1400 | 11107.184 | 10630.998 | 10076.521 | 333590 | 0 | 83.112 |

四档均满足：所有计划任务已开始、所有已开始 RPC 有结果、Master arrival/completion count 等于 client success count、错误数为 0、client pacing P99 不超过 100 ms。

arrival QPS 和 completion QPS 使用各自首尾事件窗口，数值可不同。完整性必须比较 count，不能要求两个 QPS 相等。

## 5. Master 服务端延迟

| Master arrival QPS | Avg ms | P50 ms | P90 ms | P95 ms | P99 ms | Max ms |
|---:|---:|---:|---:|---:|---:|---:|
| 107.701 | 162.210 | 162 | 165 | 167 | 173 | 392 |
| 1025.334 | 150.977 | 161 | 166 | 168 | 232 | 1207 |
| 5173.321 | 116.941 | 27 | 166 | 291 | 1459 | 1734 |
| 10630.998 | 455.754 | 161 | 1461 | 2039 | 2891 | 3300 |

10K 档的服务端阶段分解：

| 阶段 | Avg ms | P50 ms | P90 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|---:|
| gRPC queue | 354.693 | 0 | 1308 | 1889 | 2755 |
| Route submit | 7.086 | 1 | 19 | 23 | 37 |
| Batch wait | 77.328 | 61 | 160 | 160 | 177 |
| Dispatch ACK | 15.159 | 1 | 9 | 29 | 499 |
| ACK response | 0.256 | 0 | 0 | 0 | 0 |

总延迟的分阶段值存在重叠采样边界，不能简单相加。阶段数据用于定位长尾来源。

## 6. Batch 决策结果

| Master arrival QPS | Dispatch 数 | Fixed timeout | Predict threshold | Batch full | Batch mean | P50 | P90 | P95 | P99 | Max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 107.701 | 6418 | 6418 | 0 | 0 | 1.006 | 1 | 1 | 1 | 1 | 2 |
| 1025.334 | 54316 | 54289 | 27 | 0 | 1.131 | 1 | 2 | 2 | 2 | 5 |
| 5173.321 | 41619 | 36500 | 5118 | 1 | 3.719 | 3 | 7 | 8 | 12 | 32 |
| 10630.998 | 83850 | 32467* | 10062* | 78* | 3.980 | 4* | 9* | 11* | 15* | 32 |

10K 档的 `83850` 是 Java mock 精确记录的 Enqueue RPC 数，一次 Master dispatch 对应一次 Enqueue RPC。带 `*` 的决策原因和分位数来自 42607 条结构化日志，覆盖约 50.8%，只能作为样本分布。该有效轮次运行时 `critical-only` 尚未保留 dispatch counter，不能从样本反推精确 reason 总数。

测试工具已增加 `master_prometheus_after.prom`，并将 `app.engine.balancing.master.dispatch.reason` 加入关键指标白名单。后续测试的 `slo_batch_analysis.json` 使用 Prometheus counter 给出精确 reason 总量，并单独记录 `log_coverage_ratio`；逐批日志只负责 batch size、wait 和 predicted time 的样本分布。

5K 档有完整逐批日志：batch wait mean/P99 为 156.364/161 ms，predicted time mean/P99 为 360.886/680 ms，`wait + predicted > COST_SLO_MS` 的批次为 53，决策不变量违规数为 0。

## 7. Java Mock 语义

Java mock 的 Enqueue ACK 立即返回，用于测量 Master Schedule，而不是等待理论 prefill 完成后才 ACK。每个 Enqueue 请求的每个 DP slot 被视为一个独立 batch：

1. 收到后按整个 batch 的 task 输入长度、cache 命中和 batch size 计算理论 prefill 时间。
2. 排队状态立即出现在 `running_task_info`，phase 为 `RECEIVED`。
3. 到理论开始时间后，同一 DP batch 的 task 一起切到 `RUNNING`。
4. 到理论结束时间后，一起进入 finished，并启动对应 decode 状态模拟。
5. 不同 DP rank 使用独立时间线，不会被错误地串行化。

模型使用 Master 相同的 `PREFILL_TIME_FORMULA`，`sleep_scale=1.0`。它用于验证批次决策、状态同步和 Master 调度容量，不等价于真实 GPU kernel 或真实 RTP-LLM 端到端性能。

有效 10K 轮次中 mock 堆峰值为 16998 MiB/32768 MiB，无 OOM/Full GC；prefill pending 峰值为 57132，总任务在 drain 后降到 0。dispatch ACK mean/P99 为 15.159/499 ms，说明 mock ACK 不是平均吞吐瓶颈，但在微突发期间参与了尾延迟。

## 8. Profile 结论

10K 有效轮次 JFR 的主要 CPU 热点包括公式计算、`CostBasedPrefillStrategy.applyHardFilters`、`ConcurrentHashMap.get`、available endpoint 遍历和 prefill 等待估算。JFR 未出现 OOM；运行期堆存活约 7.5 GiB，GC 不是持续吞吐上限的证据。

结合服务端阶段打点，当前 10K 尾延迟的首要问题是 gRPC 入口排队，而不是 fixed-window 160 ms 等待或 Java mock 的平均 ACK。下一步优化必须在同一 10K 档保留现场，对 gRPC executor、CPU runnable 和事件循环做 A/B；不能仅依据 `NioEventLoop.select` 的采样占比调参数。

## 9. 无效轮次

- `LOAD_CLIENT_WORKERS=16, REPLAY_SPEED=1400`：client pacing P99 173.6 ms，11074 个错误，Master completion QPS 9197；发压端失真，不能作为 Master 容量数据。
- 精确 counter 复测 `metrics_r3`：共享机上同时运行另一组 8-worker 压测，client pacing P99 1257.4 ms，19720 个错误。
- 精确 counter 复测 `metrics_r4`：另一组测试改用 7201/20000 端口，端口不冲突但共享约 27 个 CPU 核；本轮 Master gRPC executor 的 128 线程和 50000 队列打满，出现 `RejectedExecutionException`，client pacing P99 462.7 ms，21102 个错误。

两个 counter 轮次只验证了 Prometheus 精确计数采集，均不纳入性能表。R4 JFR 的热点中没有 Micrometer，失败现场与共享 CPU 争用和 gRPC executor 队列打满一致，不能归因于 counter 本身。

网络命名空间可以解决端口冲突，但不能隔离 CPU。工具现在默认检查宿主机上的并发 FlexLB/mock/load client 进程；正式轮次仍需确认系统负载稳定，否则即使端口独立，结果也无效。

## 10. 原始产物

有效轮次目录：

```text
tools/online_eval/run/20260717_slo500_wait160_valid_mw16_qps100_calibrated
tools/online_eval/run/20260717_slo500_wait160_valid_mw16_qps1000
tools/online_eval/run/20260717_slo500_wait160_valid_mw16_speed650
tools/online_eval/run/20260717_slo500_wait160_valid_mw16_speed1400
```

每个目录包含 `summary.json`、`server_latency.json`、逐请求数据、Master/Mock 日志和 JFR。复现流程见 [FlexLB Master + Mock Engine 性能测试 Handoff 手册](flexlb-master-mock-engine-performance-handoff.md)。
