# 请求生命周期重构验证（2026-09-14）

实现提交：`5e7765eb2b`；基线：`8911015bcd`。后续提交仅补充注释和验证记录。

## 功能与格式

| 验证 | 结果 |
| --- | --- |
| 本机全量 UT，`opensource,!internal` | 1,505 项，0 失败，0 错误，1 跳过 |
| Linux 容器全量 UT，同 profile | 1,505 项，0 失败，0 错误，1 跳过 |
| 本机 `sync-performance-regression` | 1 项，通过 |
| 核心重构／新增 Java 文件 Spotless 检查 | 8 个文件，通过 |
| `git diff --check` | 通过 |

新增回归验证旧请求被移除、ID 复用后，旧 DeliveryClaim、PreemptionRegistration 和取消入口不能改变新请求。现有准入、发布竞争、投递锁、终止结算断言保留，mock 拦截点迁移到实际拥有事件的对象。

## 750P/750D 参数与顺序

- Linux x86_64；容器可见 256 CPU，无 CPU quota；约 3.9 TiB 物理内存。
- Java 21（Dragonwell 21.0.11.0.11），固定 8 GiB heap；未改变 planner 参数。
- 原有 `tools/run_queue_performance.sh`；BATCH/FIXED_WINDOW 与 NON_BATCH/SINGLE。
- 目标 3,000／10,000 QPS，每档预热 10 秒、测量 10 秒；原始 payload。
- BATCH 为 750 个 Netty Prefill mock RPC server 与 750 个逻辑 Decode endpoint；这是调度/投递性能，不是 GPU 推理吞吐。
- 有效运行按重构 1 → 基线 2 → 重构 2 → 基线 3 → 重构 3 → 基线 retry1 顺序交替，无并行 Maven/性能测量。
- 基线最初的 round 1 在 Git safe.directory 校验处退出 128，未进入测量。保留启动失败记录，补测 retry1；未将启动失败计为 0 QPS。
- 门槛：完成吞吐 ≥目标 98%，client/server P99 ≤250 ms，BATCH wait P99 ≤50 ms。

## 三轮统计

数值为三轮中位数；达标次数同时检查上述全部门槛。

| 版本 | 模式 | 目标 QPS | 完成 QPS | Client P99 ms | Server P99 ms | Batch wait P99 ms | 达标次数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | BATCH | 3000 | 2996.8 | 272.1 | 271.0 | 261.0 | 0/3 |
| baseline | BATCH | 10000 | 9988.8 | 60.8 | 50.0 | 47.0 | 2/3 |
| baseline | NON_BATCH | 3000 | 2999.6 | 2.4 | 1.0 | — | 3/3 |
| baseline | NON_BATCH | 10000 | 9994.9 | 53.0 | 5.0 | — | 3/3 |
| refactor | BATCH | 3000 | 2996.8 | 325.3 | 325.0 | 315.0 | 0/3 |
| refactor | BATCH | 10000 | 9989.1 | 63.7 | 47.0 | 43.0 | 3/3 |
| refactor | NON_BATCH | 3000 | 2999.5 | 7.2 | 1.0 | — | 3/3 |
| refactor | NON_BATCH | 10000 | 9988.4 | 79.7 | 10.0 | — | 3/3 |

## 每轮原始数值

| 运行 | 模式 | 目标 QPS | 完成 QPS | Client P99 ms | Server P99 ms | Batch wait P99 ms | 全部门槛 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline-750-2 | BATCH | 3000 | 2996.8 | 241.5 | 241.0 | 230.0 | 未通过 |
| baseline-750-2 | BATCH | 10000 | 9988.8 | 333.6 | 66.0 | 60.0 | 未通过 |
| baseline-750-2 | NON_BATCH | 3000 | 2998.7 | 1.6 | 1.0 | — | 通过 |
| baseline-750-2 | NON_BATCH | 10000 | 9996.0 | 53.3 | 5.0 | — | 通过 |
| baseline-750-3 | BATCH | 3000 | 2996.8 | 272.1 | 271.0 | 261.0 | 未通过 |
| baseline-750-3 | BATCH | 10000 | 9988.9 | 60.8 | 50.0 | 47.0 | 通过 |
| baseline-750-3 | NON_BATCH | 3000 | 2999.7 | 2.4 | 1.0 | — | 通过 |
| baseline-750-3 | NON_BATCH | 10000 | 9994.9 | 46.5 | 4.0 | — | 通过 |
| baseline-750-retry1 | BATCH | 3000 | 2996.8 | 344.6 | 344.0 | 334.0 | 未通过 |
| baseline-750-retry1 | BATCH | 10000 | 9986.3 | 56.1 | 48.0 | 45.0 | 通过 |
| baseline-750-retry1 | NON_BATCH | 3000 | 2999.6 | 3.5 | 1.0 | — | 通过 |
| baseline-750-retry1 | NON_BATCH | 10000 | 9987.5 | 53.0 | 7.0 | — | 通过 |
| refactor-750-1 | BATCH | 3000 | 2996.8 | 85.1 | 84.0 | 74.0 | 未通过 |
| refactor-750-1 | BATCH | 10000 | 9989.2 | 232.5 | 45.0 | 41.0 | 通过 |
| refactor-750-1 | NON_BATCH | 3000 | 2999.5 | 9.1 | 1.0 | — | 通过 |
| refactor-750-1 | NON_BATCH | 10000 | 9988.4 | 81.1 | 14.0 | — | 通过 |
| refactor-750-2 | BATCH | 3000 | 2996.8 | 471.1 | 470.0 | 458.0 | 未通过 |
| refactor-750-2 | BATCH | 10000 | 9957.0 | 63.7 | 49.0 | 46.0 | 通过 |
| refactor-750-2 | NON_BATCH | 3000 | 2999.5 | 4.8 | 1.0 | — | 通过 |
| refactor-750-2 | NON_BATCH | 10000 | 9986.4 | 79.7 | 8.0 | — | 通过 |
| refactor-750-3 | BATCH | 3000 | 2996.8 | 325.3 | 325.0 | 315.0 | 未通过 |
| refactor-750-3 | BATCH | 10000 | 9989.1 | 53.8 | 47.0 | 43.0 | 通过 |
| refactor-750-3 | NON_BATCH | 3000 | 2999.6 | 7.2 | 1.0 | — | 通过 |
| refactor-750-3 | NON_BATCH | 10000 | 9993.2 | 69.3 | 10.0 | — | 通过 |

## 结论与限制

功能验证通过；全部有效测量的吞吐均达到目标的 98%。吞吐中位数基本持平，但尾延迟不能据此认定无回归：NON_BATCH 的 Client P99 中位数在 3,000 QPS 下由 2.4 ms 升至 7.2 ms，在 10,000 QPS 下由 53.0 ms 升至 79.7 ms；后者三个重构样本均高于三个基线样本。这些档位仍满足绝对 SLO，但变慢现象的原因尚未确定。BATCH 3,000 QPS 的 Client P99 中位数由 272.1 ms 升至 325.3 ms，样本区间有重叠。BATCH 的尾延迟门槛未在所有轮次通过，基线同样存在失败，不能将本次结果写成“性能全部验收通过”，也不能仅据基线失败断定与改动无关。三轮数据用于描述本次观测，不构成统计显著性或稳定提升的证明。

完整测量字段见 [results.json](results.json)。每轮原始日志、退出码、运行前后负载及环境信息均保留在验证归档中；未挑选最佳轮次或放宽门槛。

原始性能日志见 [logs/](logs/)，包括全部失败轮次及最初的 Git 启动失败；退出码见 [exit-statuses.json](exit-statuses.json)，远端 UT 计数见 [ut-counts.json](ut-counts.json)。
