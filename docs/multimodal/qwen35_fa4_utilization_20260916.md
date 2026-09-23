# 当前 FA4 Tensor Core 利用率检查

2026-09-16，GPU0 / NVIDIA L20D（SM103，148 SM）。仓库 /home/xieshui.yyx/workspace/RTP-LLM/github-opensource，分支 feat/qwen35_vl_omega，HEAD 2679cfed1b2ae4b3d315c34577e9feb31ef6017f。

## 结论

当前生产定长 FA4 路径没有用满 Tensor Core。batch 从 1 增加到 32 后，Tensor 管线活跃率仍约 56%–57%，没有逼近持续满载；HBM 吞吐也没有达到上限。本轮只检查利用率，未改生产实现。

## 本次测量

这是匹配实际 ViT attention 形状和 stride 的随机 Q/K/V 独立 kernel 测量，不是完整 ViT 或服务请求 MFU。

- BF16，非 causal，16 heads，head_dim=72，scale=72**-0.5。
- 每视频 23 段，每段 10032 tokens；各段独立。batch 指打包视频数。
- Q/K stride=(1152,72,1)，V stride=(3456,72,1)，与当前输入布局一致。
- 调用当前生产 _fa4_vision_attention，执行原生公开 FA4 定长接口；默认 tile=128×128。
- 两档实采 kernel 均为 rank-4 dense FA4；grid=(148,1,1)，block=(512,1,1)，对应 persistent 调度。
- 所有输出有限；FP32 dense attention 对首/中/末段各 3 个 query 的所有 heads、完整 K/V 抽查通过，atol=5e-4、rtol=2e-2；重复抽样逐位一致。不是本轮真实权重全量输出验证。

### 无 profiler 正式计时

预热收敛后各 5 轮，batch1 每轮 20 次、batch32 每轮 10 次；分配、随机数、正确性验证不计入。CUDA Event 计时，并同步记录 host 时间。前后 GPU0 无其他计算进程。

| 视频 batch | CUDA 中位 ms | CUDA 平均 ms | 有效 TFLOP/s | 按 2250 归一化的 MFU |
|---|---:|---:|---:|---:|
| 1 | 10.92078 | 10.90901 | 976.70 | 43.41% |
| 32 | 391.26616 | 391.51397 | 872.35 | 38.77% |

有效 FLOPs=4×段数×10032²×16×72，只计 QK 和 PV，有效维度、FMA=2，不计 padding 和 softmax 标量操作。2250 TFLOP/s 是用户给定参考值，本轮未将其验证为厂商 BF16 dense 规格。

### Nsight Compute 硬件计数器

独立 profiler 运行，每档采一次目标 kernel、12 次 replay；仅用于硬件归因。无 clock/cache control，保持运行环境设置；不将 profiler 的 kernel 时间用于上表 MFU。以下是整个 kernel 的平均指标，不能排除部分时段或 warp 局部饱和。

| 指标 | batch1 | batch32 |
|---|---:|---:|
| Tensor 管线活跃率 | 56.47% | 56.74% |
| NCU BF16→FP32 Tensor 运算吞吐/其峰值 | 28.23% | 28.37% |
| XU 管线活跃率 | 45.53% | 45.75% |
| HBM 吞吐/峰值 | 2.88% | 11.92% |
| 调度器发射活跃率 | 45.73% | 45.73% |
| achieved warp occupancy | 23.44% | 23.44% |
| 每调度器每 active cycle 可发射 warp | 0.56 | 0.56 |

Tensor 活跃率、运算吞吐百分比、MFU 是不同口径：有 Tensor 管线活动的周期占比不能直接当作 FLOP 利用率。NCU 吞吐百分比按其硬件计数器峰值归一化，与上表用户给定 2250 TFLOP/s 的分母不同；也不能将两次运行的时间和计数器混算。[NVIDIA 指标说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder)

当前 NCU 实测运算数：batch1=12.1936805888 TFLOP，batch32=390.1977788416 TFLOP；相对有效计算多 14.32%，符合本次 tile/head padding 的额外工作。上表 MFU 没有将 padding 当作有效工作。

发射活跃率仅约 45.7%，可发射 warp 平均 0.56。两档每条指令对应平均 warp 周期约 8.20，其中 barrier 指标约 1.90、long scoreboard 约 3.12，说明依赖/同步等待仍值得追踪。本轮没有重做 PC/SASS 定位，不能直接把早期 varlen 的具体等待位置套到当前 dense 内核，也不能把这些 warp 周期占比等同请求耗时或预计加速收益。

TMEM 容量占满与 Tensor Core 运算满载是两回事；本轮计数器没有测 TMEM 容量占用。XU 活跃率也不等于某一条 MUFU 指令的利用率。

## 原始数据和复现

- [计时及正确性](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-util-20260916-_qm1ml26/bench-result.json)
- [硬件指标](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-util-20260916-_qm1ml26/selected-metrics.json)
- [计数器完整 CSV](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-util-20260916-_qm1ml26/counters.csv)
- [NCU report](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-util-20260916-_qm1ml26/counters.ncu-rep)
- [NCU 日志](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-util-20260916-_qm1ml26/ncu.log)
- [精确 NCU 命令](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-util-20260916-_qm1ml26/ncu-command.json)
- [计时命令及代码哈希](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-util-20260916-_qm1ml26/benchmark-command.json)
- [汇总 JSON](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-util-20260916-_qm1ml26/summary.json)
- [实验脚本](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-util-20260916-_qm1ml26/benchmark.py)

运行均已完成，GPU0 无残留计算进程。生产文件 SHA256 与运行前一致。所有实验脚本、缓存、报告位于用户授权的 RTP-LLM 根目录内。

## 后续原因诊断

[当前 dense FA4 等待位置与布局对照](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/docs/multimodal/qwen35_fa4_bottleneck_20260916.md)补充了 SASS、K/V 供数与同步依赖，以及 72→80/128 的同值控制。
