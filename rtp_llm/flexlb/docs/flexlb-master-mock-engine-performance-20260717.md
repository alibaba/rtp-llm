# FlexLB Master + Mock Engine Batch 性能报告

## 1. 测试范围

本报告只测试 FlexLB Master 的 batch 调度性能，请求链路为：

`Schedule RPC -> Master 路由 -> fixed-window 凑批 -> EnqueueBatch -> engine ACK -> Schedule 返回`

不测试 `FetchResponse`。Fetch 是 frontend 到 engine 的结果拉取操作，不属于 Master 调度性能。

## 2. 测试配置

- 测试日期：2026-07-17
- 调度模式：`batch`
- 客户端模式：`SCHEDULE_ONLY=1`
- 凑批算法：`fixed_window`
- 固定凑批等待：10 ms
- 最大 batch size：32
- Engine 数量：750 prefill + 500 decode
- Mock engine：Java 公式驱动实现
- 性能模型：`dsv4_flash_performance.fast_ab.json`
- 流量模型：保留原始 trace 的毫秒级到达分布，通过 `REPLAY_SPEED` 等比例加速
- FlexLB JVM 堆：32 GiB
- Master 调度 worker：`SCHEDULE_WORKER_SIZE=16`，所有档位固定不变
- 发压 worker：`LOAD_CLIENT_WORKERS`，分别测试 1、2、4、8
- 单档测试时长：20 秒
- 高压力复测：发压前等待 10 秒，使 endpoint 同步和 gRPC channel 建连稳定
- 有效数据标准：Master 收到的请求全部完成，Schedule 错误数为 0

表中的 Master QPS 使用 Master 服务端实际 arrival QPS。延迟在 Master 内部打点，从 RPC 到达开始，到 Schedule 完成为止，包含 gRPC 排队、路由、凑批等待、EnqueueBatch ACK 和响应完成。

## 3. 性能矩阵

| 发压 worker | Replay speed | 客户端发送 QPS | Master 实际 QPS | Master 完成 QPS | Avg ms | P50 ms | P90 ms | P95 ms | P99 ms | 错误数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 125 | 958.469 | 964.721 | 983.349 | 28.255 | 11 | 14 | 32 | 565 | 0 |
| 1 | 250 | 1948.212 | 1960.527 | 2000.263 | 161.429 | 12 | 370 | 1191 | 2439 | 0 |
| 1 | 375 | 2961.774 | 2978.083 | 3064.703 | 59.754 | 12 | 142 | 392 | 881 | 0 |
| 2 | 250 | 1948.191 | 1950.092 | 1999.785 | 70.429 | 12 | 121 | 359 | 1351 | 0 |
| 2 | 375 | 2978.923 | 2990.264 | 3039.417 | 113.320 | 11 | 351 | 628 | 1870 | 0 |
| 2 | 650 | 5162.882 | 5175.669 | 5326.688 | 78.633 | 11 | 298 | 479 | 1010 | 0 |
| 4 | 375 | 2978.896 | 2990.099 | 3067.141 | 118.352 | 12 | 537 | 578 | 1831 | 0 |
| 4 | 650 | 5162.883 | 5190.102 | 5295.788 | 49.301 | 11 | 114 | 263 | 858 | 0 |
| 4 | 1000 | 7920.602 | 7942.946 | 8165.137 | 283.374 | 11 | 890 | 2444 | 3545 | 0 |
| 8 | 650 | 5162.790 | 5177.057 | 5320.767 | 141.952 | 11 | 558 | 974 | 1989 | 0 |
| 8 | 1000 | 7920.153 | 7954.074 | 8197.324 | 77.109 | 11 | 161 | 670 | 1036 | 0 |
| 8 | 1400 | 11151.363 | 11190.225 | 11464.918 | 450.817 | 11 | 695 | 5207 | 6228 | 0 |

Master completion QPS 可能略高于 arrival QPS，因为两个速率使用各自的首尾事件作为统计窗口。是否完整处理以请求计数为准；表内所有有效档位均满足 arrival count 等于 completion count。

## 4. 高压力阶段耗时

8 个发压 worker、Master 实际 11,190 QPS 时：

| Master 阶段 | Avg ms | P50 ms | P90 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|---:|
| gRPC queue | 276.182 | 0 | 526 | 1716 | 5166 |
| Route submit | 5.119 | 1 | 2 | 16 | 81 |
| Batch wait | 9.997 | 7 | 10 | 12 | 119 |
| Dispatch ACK | 157.176 | 1 | 50 | 191 | 4972 |
| ACK response | 1.091 | 0 | 0 | 0 | 33 |

高压力下 fixed-window 凑批本身仍然稳定：batch wait P50 为 7 ms、P90 为 10 ms、P95 为 12 ms。10K 档长尾主要来自入口 gRPC queue 和下游 engine dispatch ACK，不是凑批算法。

## 5. 结论

1. FlexLB Master batch 调度在 2 或 4 个发压 worker 下可稳定达到 5.1K QPS，Schedule 零错误。当前最好的 5K 档是 4 个发压 worker：Avg 49.301 ms、P50 11 ms、P90 114 ms、P95 263 ms、P99 858 ms。
2. 8 个发压 worker 下，Master 可达到 7.95K QPS，Schedule 零错误：Avg 77.109 ms、P50 11 ms、P90 161 ms、P95 670 ms、P99 1036 ms。
3. 8 个发压 worker 下，Master 吞吐可达到 11.19K QPS，Schedule 零错误，但长尾尚未受控：P95 5207 ms、P99 6228 ms。
4. 表中的 worker 数是发压端 `LOAD_CLIENT_WORKERS`，它改变客户端 channel/timer 并行度和微突发形态，不是 Master 内部调度 worker 数。Master 的 `SCHEDULE_WORKER_SIZE` 在全部测试中固定为 16。
5. 高压力测试前必须预留至少 10 秒做 endpoint/channel 稳定。没有稳定等待时，冷启动 channel callback 会使 `engine-grpc-client-executor` 饱和并触发 `RejectedExecutionException`，属于启动阶段污染，不代表稳态 Master 容量。

## 6. 原始产物

- 初始矩阵：`rtp_llm/flexlb/tools/online_eval/run/20260717_master_batch_matrix_w*_s*/`
- 稳定等待复测：`rtp_llm/flexlb/tools/online_eval/run/20260717_master_batch_warm10_w*_s*/`
- 每个 run 均包含：`load_client/summary.json`、`load_client/server_latency.json`、`flexlb_profile.jfr`、`flexlb.log`、`mock_engine.log`

复现方法、阶梯加压方案、指标口径和瓶颈判定见 [FlexLB Master + Mock Engine 性能测试 Handoff 手册](flexlb-master-mock-engine-performance-handoff.md)。
