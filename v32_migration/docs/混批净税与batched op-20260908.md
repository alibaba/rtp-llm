# 混批净税与 batched op

日期：2026-09-08

## 结论

同一 C 服务实例内完成了 5 组相邻 `纯短(P) → 混批(M)` 对比。五组配对差 `M-P` 全部为负：

```text
-0.494, -0.194, -0.922, -0.005, -1.007 ms/token
```

配对差均值为 **-0.524ms/token**，0/5 为正。轮内标准差 RMS 为 0.227ms/token，2 倍为 0.454ms/token。

预先约定的实现门槛是：

1. 至少 4/5 对为正；
2. 配对差均值至少 +0.5ms/token；
3. 配对差均值大于 2 倍轮内波动。

三项均未通过。因此，本次没有实现 per-layer batched op，也没有修改 `v32_ctx.cu`、Python hook 或其他运行代码。当前 eager 路径下，没有证据表明让短 row 退出 mixed per-layer 处理能带来值得实现的收益。

负差值不应解释成 offload 会加速短请求。P 和 M 都有随时间变化的运行波动，而实验顺序固定为 P 后 M；本报告只据此排除“稳定、正向且大于噪声的混批净税”。

## 实验设计

部署与 `fastpath_20260907` 一致：

| 项目 | 配置 |
|---|---|
| P | `11.17.131.204`，TP2×DP4，EP8 |
| D | `33.240.36.239`，TP1×DP8，EP8，12GiB KV/卡 |
| 请求路由 | 全部通过 `role_addrs` 固定到 D rank0 |
| 运行模式 | C lossless，`reuse_cache=0`，`enable_cuda_graph=0` |
| 短请求 | 2,220 输入，1,024 输出，8 条，两波提交 |
| 长请求 | 62,830 输入，8,192 输出 |

在同一个 C 实例内按以下顺序运行：

```text
P1 → M1 → P2 → M2 → P3 → M3 → P4 → M4 → P5 → M5
```

每轮短请求提交时间仍为：

```text
70, 72, 74, 76, 100, 102, 104, 106 秒
```

混批轮在 `t=0` 先发长请求。短请求使用修正 TPOT：

```text
(decode_service_us - decode_first_token_us) / (output_len - 1)
```

每个 M 结束后，屏障依次确认：

- rank0 `running_query_len=0`；
- `waiting_query_len=0`；
- `running_task_info` 为空；
- KV 已恢复为 `123200/123200`；
- 再等待 40 秒，跨过 mirror 的 30 秒 deferred-release 窗口；
- P 和全部 8 个 D HTTP health 均正常。

P 轮后也确认 idle 和 KV 全量归还，并等待 5 秒。这样下一轮不会继承仍活跃的请求或 KV/mirror 状态。

## 配对结果

单位均为 ms/token。轮内波动是该轮 8 条短请求 corrected TPOT 的样本标准差。

| Pair | P mean | P 轮内 stdev | M mean | M 轮内 stdev | M-P |
|---:|---:|---:|---:|---:|---:|
| 1 | 117.043 | 0.100 | 116.549 | 0.108 | **-0.494** |
| 2 | 116.464 | 0.286 | 116.270 | 0.248 | **-0.194** |
| 3 | 116.868 | 0.411 | 115.946 | 0.220 | **-0.922** |
| 4 | 115.799 | 0.124 | 115.794 | 0.206 | **-0.005** |
| 5 | 116.842 | 0.250 | 115.835 | 0.117 | **-1.007** |

汇总：

| 指标 | 结果 |
|---|---:|
| P 五轮均值 | 116.603ms |
| M 五轮均值 | 116.079ms |
| 配对差均值 | **-0.524ms** |
| 配对差中位数 | -0.494ms |
| 正差数量 | **0/5** |
| 轮内 stdev RMS | 0.227ms |
| 2×轮内 stdev RMS | 0.454ms |
| 最大单轮 stdev | 0.411ms |

判定：

| 门槛 | 要求 | 实测 | 通过 |
|---|---:|---:|---|
| 符号一致性 | ≥4/5 为正 | 0/5 | 否 |
| 差值量级 | 均值 ≥+0.5ms | -0.524ms | 否 |
| 超过噪声 | 均值 >2×0.227ms | -0.524ms | 否 |

## 正确性与服务状态

- 5 个 P 轮共 40 条短请求，全部成功；
- 5 个 M 轮共 40 条短请求和 5 条长请求，全部成功；
- 80 条短请求输出 hash 均为 `47229f5fe1b12614`；
- 5 条长请求输出 hash 均为 `27887e7c2b6b7b51`；
- M 轮 rank0 峰值均为 9 条请求，确认长短 decode 重叠；
- 所有轮次时间窗内无 allocator、CUDA、RDMA、disconnect、mirror 或 fail-closed 错误；
- 每轮后的释放屏障均通过，KV 都回到 `123200/123200`。

长请求 corrected TPOT 为：

```text
113.960, 113.794, 113.767, 113.639, 113.736 ms/token
```

均值 113.779ms。ring load 均值为 944.757ms；第一轮 mirror 冷分配为 1,708.6ms，后续四轮命中 pool，prepare 约 0.001–0.005ms。admission blocking wait 始终约 0.004–0.006ms。

## 是否实现 batched op

**不实现。**

当前数据没有显示短 row 在 mixed per-layer 处理里承担稳定正向成本。即使按最宽松的点估计，M 也没有比相邻 P 更慢。继续做 batched op 会增加 C++/Python 接口和状态管理复杂度，却没有可验证的 eager 收益。

这不否定 batched op 在其他 workload 上的价值。若以后 batch 中同时存在多个 offloaded row、更大的短请求 batch，或 cuda graph 开启后出现新的 shape/graph 税，应重新测量，而不是沿用本次结论。

## 启动过程说明

有效数据全部来自 attempt5，同一实例连续完成 5 对。

此前无效尝试没有发出实验请求：

- attempt1/2：编排器把 D 和 P 共用同一个 readiness 总预算，D 冷启动后留给 P 的时间不足；
- attempt3/4：P 从源码工作树 cwd 启动，源码空 proto 包遮蔽 runtime 生成的 `predict_v2_pb2`，两次均在启动阶段失败。

修复 harness 后，P 从 `/home/admin` 启动，attempt5 完整通过。这些失败日志保留在数据目录，但不进入统计。

## 限定

本实验全程 `enable_cuda_graph=0`，结论只适用于当前 eager 路径。它不能回答 cuda graph 双版本 capture 是否值得做，也不能代表其他 batch 组成或多条 offloaded row 的成本。

## 证据

原始数据：

```text
v32_migration/data/mixedtax_interleave_20260908/
```

有效轮次前缀：

```text
preop_a5_P1 ... preop_a5_P5
preop_a5_M1 ... preop_a5_M5
```

机器可读判定：

```text
preop_analysis.json
decision.json
```

每轮包含请求 JSONL、client log、P/D GPU、D host memory、D engine/Python log、worker status、health、start/end epoch 和释放屏障。代码 SHA 记录在 `final_code_sha256.txt`。本任务没有新增运行代码改动，也没有提交。
