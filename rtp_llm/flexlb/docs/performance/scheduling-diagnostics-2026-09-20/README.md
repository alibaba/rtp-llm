# schedulingDiagnostics 移出 Response 后的远端性能验证

2026-09-20，当前工作区快照在 B300 的 `luoli_gpu` 容器运行，所有性能门禁通过。

## 源码与环境

- 原始 HEAD：`112c26fba06781d8279f45a40fcbefa54c8b76cc`，包含本次尚未提交的修改，以 [源码清单](source-manifest.json) 和 [归档中的 workspace.diff](remote-logs.tar.gz) 为准。
- 同步 565 个文件，包含构建配置、源码、测试数据和 proto，不包含本地 target。远端执行前后 SHA-256 校验全部通过；完成时本地构建输入也与快照一致。
- Dragonwell Java 21.0.11.0.11，256 个可见 CPU，8 GiB E2E 测试堆；共享机器，完整信息见 [环境记录](environment.txt)。
- 独立目录：`/data0/luoli.hn/work/rtp_llm_4/flexlb-diagnostics-perf-20260920-231008`。
- 原 10001–11499 端口段有占用，本轮使用测试已有参数改为 30001 起始端口，运行前检查 30001–31499 无监听。

## Sync 性能门禁

使用 `sync-performance-regression` profile，通过 `-Dtest` 指定 `WorkerBatcherPerformanceTest,PrefillAdmissionFailurePerformanceTest`。共 4 项测试全部通过；未修改门槛。

Prefill 失败诊断聚合：

| Worker 数 | ns / failure | bytes / failure |
| ---: | ---: | ---: |
| 1 | 385 | 328 |
| 64 | 6463 | 328 |
| 512 | 48820 | 368 |
| 1024 | 96994 | 368 |

## 端到端性能门禁

执行现有 `tools/run_queue_performance.sh`，750P/750D，BATCH/FIXED_WINDOW 与 NON_BATCH/SINGLE 顺序运行；每档预热 10 秒、测量 10 秒，目标 3000/10000 QPS，测量请求分别为 30000/100000。

| 模式 | 目标 QPS | Client QPS | Master QPS | Client P99 | Master P99 | Delivery wait P99 | Batch wait P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BATCH | 3000 | 2996.8 | 3000.6 | 25.282ms | 12ms | 11ms | 10ms |
| BATCH | 10000 | 9988.9 | 10004.4 | 39.085ms | 33ms | 31ms | 30ms |
| NON_BATCH | 3000 | 2999.5 | 3000.3 | 3.182ms | 1ms | 1ms | N/A |
| NON_BATCH | 10000 | 9995.2 | 10001.7 | 8.386ms | 1ms | 1ms | N/A |

四档全部通过原门槛：Client/Master 吞吐 ≥ 目标的 98%，Client/Master P99 ≤ 250 ms，Delivery wait P99 ≤ 50 ms，BATCH 的 Batch wait P99 ≤ 50 ms。Sync、BATCH、NON_BATCH 退出码均为 0。

这是 loopback 调度性能测试；BATCH 包含 mock engine RPC，NON_BATCH 测量直接路由返回。未进行同机配对基线比较，也未执行长时间堆留存测试，因此不据此声称性能提升或证明所有场景不存在内存泄漏。

[机器可读结果](results.json) · [指标摘要](summary.txt) · [完整远端日志、测试 XML 和运行脚本](remote-logs.tar.gz)

合并前还完成了代码 Review 和完整功能验证；本机计时用例的基线对照、远端复核及测试 import 整理详见 [Review 记录](review.md)。
