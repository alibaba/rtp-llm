# Qwen3.5 当前 FA4 路径的 Tensor 活跃率与有效算力

采集日期：2026-09-17。当前 batch=32 单层 FA4 耗时 **356.82 ms**；Tensor pipe 活跃率 **66.52%**；以用户提供的 2250 TFLOP/s 为分母，有效算力利用率 **42.51%**。

本报告沿用对话中的“MTU”称呼，具体指标是 Tensor pipeline 的归一化活跃周期比例，不代表有效模型 FLOPs 占峰值的比例。6 次计数器捕获和所有正确性检查完成。生产源码、安装依赖未修改。

## 测量边界与环境

- 仓库：/home/xieshui.yyx/workspace/RTP-LLM/github-opensource，分支 feat/qwen35_vl_omega，HEAD 2679cfed1b2ae4b3d315c34577e9feb31ef6017f，使用当前含已有未提交改动的工作区；相关源文件 SHA256 见 contract.json，测后验证未变。
- GPU0：NVIDIA L20D，SM103，148 SM，GPU UUID 67f05232-d6d5-7210-1143-8cb1c0a9eed0；driver 580.105.08，功率上限 1100 W，未锁频。
- PyTorch 2.11.0+cu130，CUDA 13.0；实际调用当前生产 helper _fa4_vision_attention，原生 FA4 dense，tile 128×128，split_P_arrive=96。
- BF16，16 heads，有效维度 72；Q/K 物理维度 80，最后 8 维为零；V/output 72，softmax_scale=72**(-0.5)，non-causal。Q/K stride=(1280,80,1)，V stride=(3456,72,1)。
- 每视频 23 个相互独立的 attention 段，每段 10032 tokens；B1 共 230736 tokens，B32 共 7383552 tokens。输入是匹配生产形状、步幅的合成正态数据，未加载视频和真实模型权重。
- 计时包括单次 FA4 helper GPU 执行；输入分配、padding、RoPE、检查、QKV/projection、其他 ViT 层、视频加载/解码、网络/gRPC、LLM 均不在本次计时范围。无 CUDA Graph。
- 各次采样前后检查 GPU0 没有其他计算进程；结果输出后 GPU0 已无计算进程。

## 当前测量结果

| 视频 batch | 无 profiler 单层时间 ms | Tensor pipe 活跃率 | 有效 TFLOP/s | 有效算力利用率 /2250 |
|---:|---:|---:|---:|---:|
| 1 | 10.354673 | 66.028702% | 1030.099 | 45.7822% |
| 32 | 356.817220 | 66.518626% | 956.576 | 42.5145% |

时间：CUDA events，预热至少 5 次且连续 3 次变化小于 3%；每组 6 轮，每轮 B1=20 次、B32=3 次调用，每轮前另预热 5 次。表中取各轮平均每次耗时的中位数。输入 tensor 重用，未主动清缓存，输出由正常 helper 分配。未剔除样本。

B1 各轮 ms：10.378712, 10.365216, 10.290954, 10.348702, 10.295213, 10.360645。
B32 各轮 ms：360.776367, 358.195597, 355.873088, 350.994263, 357.761353, 353.545614。

计数器：另启 Nsight Compute 进程，每种 batch 捕获 3 次，每次 3 replay passes，NVTX 精确匹配 B1_rep0..2、B32_rep0..2 内的 MEASURE 范围与原生 FlashAttentionForwardSm100 kernel。未控制时钟或缓存，B32 replay 备份使用系统内存。计数器取 3 次中位数。

Profiler 中的执行时间与无 profiler 时间不同（见原始表），因此没有把 profiler 时间当成正常耗时，也没有将它与无 profiler 时间混在同一公式中。

## Tensor 活跃率计算

采集以下原始量：

- A = sm__pipe_tensor_cycles_active.avg
- E = sm__cycles_elapsed.avg
- P = sm__pipe_tensor_cycles_active.avg.peak_sustained

公式：**100 × A / (E × P)**。

本设备本指标实测 P=4。A 是该计数器汇总的平均 SM Tensor 活跃周期，不是单个“开/关”的 SM 周期；不能直接用 A/E，必须用工具返回的峰值系数归一化。这与 sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed 一致，全部 6 条复算误差小于 0.00001 个百分点。[NVIDIA 指标定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#metrics-structure)

B32 的中位数代表捕获 B32_rep2：

```text
A = 1287339589.189189
E = 483826735.189189
P = 4
活跃率 = 100 × A / (E × P) = 66.518626%
```

| 捕获 | A：Tensor 周期/SM | E：总周期/SM | P | Tensor 活跃率 | NCU BF16 ops 峰值比例 | profiler 时间 ms |
|---|---:|---:|---:|---:|---:|---:|
| B1_rep0 | 40229362.162162 | 15231770.891892 | 4 | 66.028702% | 33.014351% | 8.648096 |
| B1_rep1 | 40229362.162162 | 15240879.689189 | 4 | 65.989239% | 32.994620% | 8.650688 |
| B1_rep2 | 40229362.162162 | 15201481.540541 | 4 | 66.160265% | 33.080133% | 8.619328 |
| B32_rep0 | 1287339589.189189 | 483466781.364865 | 4 | 66.568151% | 33.284076% | 336.835840 |
| B32_rep1 | 1287339589.189189 | 483948177.256757 | 4 | 66.501934% | 33.250967% | 334.719424 |
| B32_rep2 | 1287339589.189189 | 483826735.189189 | 4 | 66.518626% | 33.259313% | 333.363616 |

NCU 自身还报告 BF16→FP32 Tensor 操作数峰值比例；B32 中位数为 33.259313%。其计算为 100 × sm__ops_path_tensor_src_bf16_dst_fp32.sum / sm__ops_path_tensor_src_bf16_dst_fp32.sum.peak_sustained_elapsed，使用 NCU 的操作吞吐峰值定义，既不是上面的活跃周期比例，也不是用户给定 2250 TFLOP/s 的分母。原始分子、分母均在 analysis.json，不将这三种口径合并。

## 有效 FLOPs 与算力利用率计算

一次乘加计 2 FLOPs。每独立段长度 L，H 个 heads，逻辑维度 d：

- QKᵀ：2 × L² × H × d
- PV：2 × L² × H × d
- 合计：4 × L² × H × d

因此：

```text
F(B) = 4 × B × 23 × 10032² × 16 × 72
F(1) = 10,666,338,287,616 FLOPs = 10.666338287616 TFLOP
F(32) = 341,322,825,203,712 FLOPs = 341.322825203712 TFLOP

B32 时间 = 0.35681722005208 s
有效吞吐 = 341.322825203712 / 时间 = 956.576101 TFLOP/s
有效利用率 = 有效吞吐 / 2250 × 100% = 42.514493%
```

23 段之间不互相做 attention，所以按各段 L² 相加，不能将 23×10032 整体平方。batch 也是独立样本，只线性乘 B。

2250 TFLOP/s 是用户给定的 FP16 峰值参考，本次 BF16 测量沿用它做归一化；本报告不将其声称为独立验证的本卡 BF16 硬件峰值。有效 FLOPs 只计 QK/PV，排除 softmax 标量操作及 padding、tile 边界多做的算术，因此这是单层 attention 主矩阵乘的有效利用率，不是完整 ViT/LLM 的 MFU。

硬件操作数 B1=12,193,680,588,800，B32=390,197,778,841,600，包含多做的填充算术。二者均吻合：

```text
4 × B × 23 × (ceil(10032/256)×256) × (ceil(10032/128)×128) × 16 × 80
```

B32 有效操作 341.323 TFLOP，硬件执行 390.198 TFLOP；有效计算不能用后者冒充。活跃率 66.52% 也不能解释成“66.52% 的标称算力用在有效计算上”。

## 正确性与证据

所有重复运行的完整输出 bitwise equal 且全部 finite；另在首段、中间段、末段各抽取首/中/末 query，与 FP32 dense attention 比较，atol=5e-4、rtol=2e-2，全部通过。该检查验证本次合成输入，不等同于本次重新验证真实 397B 整模型。

原始实验目录：/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-current-util-20260917-h3fnzfj8

- contract.json：环境与源码指纹。
- benchmark.py、entry、launch.py、ncu_launch.py：复现实验入口。
- command.json、ncu-command.json：完整命令与环境。
- bench-result.json：无 profiler 时间和正确性。
- ncu-result.json、ncu.log、counters.ncu-rep：硬件采集与日志。
- counters.csv：NCU 原始宽表（首条记录为单位）。
- analyze.py、analysis.json：公式、重算结果、逐次计数器与汇总。

原始 CSV 导出命令：

```bash
/usr/local/cuda/bin/ncu --import /home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-current-util-20260917-h3fnzfj8/counters.ncu-rep --page raw --csv > /home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-current-util-20260917-h3fnzfj8/counters.csv
/opt/conda310/bin/python -B /home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-current-util-20260917-h3fnzfj8/analyze.py
```
