# Qwen3.5 FA4 独立 kernel 性能和 MFU

> 后续真实权重验证：128×160 导致 27 层最终 embedding 相对 L2 偏差 1.63%，未达到本轮采用门槛，未接入生产。后续改用保持 128×128 的原生定长路径，详见 [优化验证报告](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/docs/multimodal/qwen35_fa4_optimization_20260916.md)。

日期：2026-09-16。GPU0 L20D / SM103，148 SM。这里只测试 FA4 前向：模拟 Q/K/V 已在 GPU 就绪，到 attention 输出在 GPU 可用。不加载模型权重，不包括 QKV 投影、RoPE、输出投影、视频处理、ViT 其他层、RPC 或 LLM；不是完整模型 MFU。

## 最新：128×128 / 128×160 交替顺序复测

在 GPU0 使用同一组 Q/K/V allocation，先预热并分别验证两种 tile，再进行 6 轮 AB/BA 交替计时。每轮每档 batch1 连续 20 次、batch32 连续 10 次；每个计时间隔前先执行 3 次候选预热。输入 shape、stride、dtype、attention 分段与前述生产路径一致。没有 profiler，编译和正确性检查均不计入 RT。

| 视频 batch | 128×128 中位 ms | 128×160 中位 ms | 耗时降低 | 基线 / 候选 MFU |
|---|---:|---:|---:|---:|
| 1 | 12.072 | 11.504 | 4.71% | 39.27% / 41.21% |
| 32 | 410.878 | 390.970 | 4.85% | 36.92% / 38.80% |

表内中位数为六轮均值的中位数，不是单调用 P50。MFU 继续使用用户给定的 2250 TFLOP/s 参考及有效 QK/PV FLOPs。batch1 六轮配对改善 4.62%–4.94%；batch32 为 4.78%–5.00%，每轮均改善。两种 batch 都有约 5% 的独立 FA4 收益，尚未重跑真实 ViT 或服务，不能直接当作端到端提升。

四个配置均通过全量输出 shape/dtype/有限值检查、选定 query 的 FP32 dense 参考抽查，以及重复输出抽样逐位一致检查。最大参考绝对误差 2.07e-4，容差 atol=5e-4、rtol=2e-2。没有进行全量 FP32 参考验证。所有计时前后检查均未发现 GPU0 其他计算进程。

调参通过内部 `_flash_attn_fwd(..., tile_mn=(128,160))` 完成；当前公开 varlen wrapper 未暴露 tile 参数。生产代码和 FA4 依赖文件均未修改。当前布局下 N=160 可以运行；更大的 N 仍受 TMEM/shared-memory 布局约束，见后文资源分析。

原始六轮 CUDA 时间：

- batch=1, tile=[128, 128]: 12.062193, 12.110910, 12.086045, 12.057349, 12.059696, 12.082754 ms。

- batch=1, tile=[128, 160]: 11.505512, 11.517934, 11.511933, 11.500127, 11.502287, 11.485889 ms。

- batch=32, tile=[128, 128]: 410.164062, 410.599902, 410.951221, 411.098437, 410.804248, 411.003857 ms。

- batch=32, tile=[128, 160]: 389.724707, 390.089917, 391.303271, 391.305884, 391.042700, 390.897559 ms。

- [交替对照脚本](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-tile-abba-20260916-t2wnkm7s/benchmark.py)
- [完整结果与正确性检查](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-tile-abba-20260916-t2wnkm7s/bench-result.json)
- [日志](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-tile-abba-20260916-t2wnkm7s/benchmark.log)
- [启动命令](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-tile-abba-20260916-t2wnkm7s/command.json)

## 初始基线结果：不带 profiler

直接调用生产路径同一 `flash_attn.cute.flash_attn_varlen_func`：BF16、non-causal、softmax_scale=72**-0.5，CUDA Graph 关闭。每个调用相当于一层 ViT 的 FA4；batch 指打包的视频数，独立 attention 段不互相注意。

| 视频 batch | attention 段数 | 中位耗时 ms | 平均耗时 ms | 五轮范围 ms | 有效计算量 TFLOP/次 | 有效 TFLOP/s | 按 2250 TFLOP/s 的 MFU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23 | 11.871 | 11.904 | 11.851–12.050 | 10.666338 | 898.5 | 39.93% |
| 32 | 736 | 408.940 | 408.923 | 408.524–409.215 | 341.322825 | 834.7 | 37.10% |

batch32 耗时为 batch1 的 34.449 倍，每视频耗时增加 7.65%。合批没有提高这个 kernel 的有效算力利用率。

每档五个完整重复，每轮 batch1 连续 20 次、batch32 连续 10 次，以 CUDA events 测整个重复后除以次数。上表中位数是五轮均值的中位数，不是单次调用 P50。同步 host 时间中位数分别为 11.880 / 408.960 ms。

原始五轮归一化 CUDA 时间：

- batch=1：11.884380, 11.865076, 11.871026, 11.851155, 12.050048 ms。
- batch=32：409.215039, 408.524170, 408.871069, 409.064868, 408.940015 ms。

预热至少五次，要求连续三次相邻变化小于 3%；编译、分配、随机数生成、正确性检查均不在计时内。正确性检查后又连续五次重新预热。正式样本 GPU0 无其他进程，采样 SM/显存频率 2032/3996 MHz；没有修改时钟和功耗上限。

## 输入和计算口径

- 种子 20260916，Q/K/V 独立标准正态随机 BF16。
- batch1 shape `[230736,16,72]`；batch32 shape `[7383552,16,72]`。
- Q/K stride `(1152,72,1)`，V stride `(3456,72,1)`，保留生产路径非连续 V 布局。
- 每视频 23 段，每段 10032 tokens；batch32 共 736 段，段长不变。
- cu_seqlens 为 CUDA int32，按 10032 累加；输出与 Q 同 shape、BF16。

有效 FLOPs 只计 QK^T 和 P@V 两次矩阵乘法，FMA=2 FLOPs：

```text
F = 4 × sum(segment_length²) × heads × head_dim
  = 4 × (23 × video_batch) × 10032² × 16 × 72
TFLOP/s = F / elapsed_seconds / 1e12
MFU = F / elapsed_seconds / (2250 × 1e12)
```

没有把所有视频算作一个超长 dense attention。分子不计 padding 和 softmax 等标量操作，分母包括完整 FA4 kernel 时间。这里的 MFU 指 FA4 有效矩阵计算量的归一化利用率。

2250 TFLOP/s 沿用用户给定的峰值参考，本实验未独立验证为设备 BF16 dense 官方规格。若采用另一峰值 P，表中 MFU 乘以 `2250/P`。按此参考，batch1/32 理想纯矩阵计算下限为 4.741/151.699 ms；这不是可直接实现的优化目标。

## 正确性

两个规模的全量输出 shape/dtype/有限值检查通过。每档选第一、中间、最后三个 segment，每段首、中、末三个 query，对所有 16 个头和完整长度 K/V 做 FP32 matmul → softmax → matmul 参考，关闭 TF32。容差 atol=5e-4、rtol=2e-2，最大绝对误差均小于 1.94e-4。重复调用的固定输出抽样逐位相等；没有进行全量 FP32 数值参考比较。

## Nsight 核对

Nsys 中每个 FA4 范围恰好只有一个 fused CUDA kernel，与真实 ViT trace 核名及 launch 配置相同：

```text
kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__tensor000o111012_tensor000o111012_tensor000o101112_tensorptrbf16gmemalign16oi64div81i64div8_None_t_0
```

| batch | grid | block | registers/thread | dynamic shared memory/block | profiler 中 kernel ms |
|---|---|---|---:|---:|---:|
| 1 | (14784,1,1) | (512,1,1) | 128 | 228352 B | 12.137 |
| 32 | (473200,1,1) | (512,1,1) | 128 | 228352 B | 414.236 |

Profiler 时间只用于路径归因，正式性能采用无 profiler 五轮结果。

## NCU 指标与 MFU 的区别

另一次独立采集；NCU replay 耗时不作为正式性能：

| batch | Tensor op 计数 / 1e12 | NCU BF16 Tensor 指标 | SM throughput | DRAM throughput | achieved occupancy |
|---|---:|---:|---:|---:|---:|
| 1 | 12.193681 | 24.10% | 51.91% | 2.53% | 21.67% |
| 32 | 390.197779 | 23.02% | 49.58% | 2.24% | 21.76% |

硬件 Tensor op 计数是有效 FLOPs 的 1.143192749 倍。源码把 head_dim 从 72 对齐为 80，且计数与下面的 tile 补齐公式严格吻合：

```text
hardware tensor ops = 4 × (23 × batch) × 10240 × 10112 × 16 × 80
useful matrix FLOPs  = 4 × (23 × batch) × 10032 × 10032 × 16 × 72
```

硬件因此执行约 14.32% 的额外 padding 矩阵工作，MFU 没有把它算为有效模型计算。

NCU 百分比使用自身硬件/时钟归一化基准，并非固定的 2250 TFLOP/s。用同一报告的 `ops.per_second / (pct/100)` 反推约 4.61/4.23 P op/s；这些是多次 replay 派生值，时钟及采样可以不同，不是设备规格或新的峰值测量。所以之前约 23% 的 NCU 指标不能替代本次按有效 FLOPs、无插桩耗时、2250 参考算出的 37–40% MFU。

寄存器与 shared memory 的 occupancy limit 均为 1 block/SM，DRAM 吞吐约 2.2–2.5%。HBM 带宽没有打满；后续补充了 batch1 的 stall 与 SASS 采样，见文末。低 occupancy 本身不足以解释全部性能损失。

## 环境与复现

- 仓库：/home/xieshui.yyx/workspace/RTP-LLM/github-opensource
- 分支：feat/qwen35_vl_omega
- HEAD：2679cfed1b2ae4b3d315c34577e9feb31ef6017f
- PyTorch：2.11.0+cu130；CUDA：13.0；flash-attn-4：4.0.0b21。
- GPU0 UUID：GPU-67f05232-d6d5-7210-1143-8cb1c0a9eed0。
- 本轮未改生产代码，ViT 源码哈希未变，原有修改保留。
- 本实验按用户要求使用模拟输入，不加载权重；没有重新验证完整模型或服务性能。

运行以下命令会更新同目录的 bench-result.json：

```bash
MM_TEST_GPU=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /opt/conda310/bin/python -B /home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-isolated-20260916-3xhwx2nn/launch.py \
  --exec /home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-isolated-20260916-3xhwx2nn/entry --mode bench --peak-tflops 2250
```

- [独立测试脚本](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-isolated-20260916-3xhwx2nn/benchmark.py)
- [运行环境入口](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-isolated-20260916-3xhwx2nn/launch.py)
- [正式结果](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-isolated-20260916-3xhwx2nn/bench-result.json)
- [正式运行日志](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-isolated-20260916-3xhwx2nn/benchmark.log)
- [Nsys trace](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-isolated-20260916-3xhwx2nn/fa4.nsys-rep)
- [kernel 归属摘要](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-isolated-20260916-3xhwx2nn/trace-summary.json)
- [NCU report](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-isolated-20260916-3xhwx2nn/counters.ncu-rep)
- [NCU 原始 CSV](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-isolated-20260916-3xhwx2nn/counters.csv)
- [NCU 指标摘要](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-isolated-20260916-3xhwx2nn/ncu-summary.json)
- [生产 FA4 来源](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-isolated-20260916-3xhwx2nn/entry.runfiles/pip_gpu_cuda13_torch_flash_attn_4/site-packages/flash_attn/cute/interface.py)

benchmark-command.json、trace-command.json、ncu-command.json 保存精确调用；dependency-manifest.json 保存依赖版本和源码哈希。产物均位于用户 RTP-LLM 目录，测试和监控已退出，未提交或推送。

## 补充：为什么合批没有提高 MFU（batch1 stall / SASS）

FA4 支持合批，前面的 batch32 已是 736 段在一次 kernel 调用中执行；Nsight 证实只有一个 fused kernel。batch1 本身就有 14784 blocks，约 99.89 waves/SM；将工作量扩大不能改变单个 block 的内部流水线。

本次只对 batch1 增采 SpeedOfLight、ComputeWorkloadAnalysis、SchedulerStats、WarpStateStats、MemoryWorkloadAnalysis、Occupancy、SourceCounters，共 22 次 replay。没有重测正式 RT，没有改生产代码。batch32 的相同 launch 资源配置已经由上一节验证，未将本次 batch1 的 stall 百分比声称为 batch32 实测。

| 指标 | 实测 |
|---|---:|
| scheduler 无 eligible warp 的周期 | 60.49% |
| active warps / scheduler | 3.47 |
| eligible warps / scheduler / cycle | 0.43 |
| long scoreboard 在平均 warp 指令间周期中的占比 | 41.58% |
| barrier 在平均 warp 指令间周期中的占比 | 24.79% |
| wait 在平均 warp 指令间周期中的占比 | 13.85% |
| math pipe throttle 在平均 warp 指令间周期中的占比 | 0.26% |
| L1/TEX / L2 throughput | 55.67% / 58.64% |
| DRAM throughput | 2.56% |
| Tensor pipe active（时间占比，不是有效 FLOPs MFU） | 48.21% |
| 每个 SM 的寄存器 / shared memory 驻留上限 | 1 / 1 block |

这些 warp 周期占比不是请求 wall time 的分解，也不是可以直接回收的加速比例。GPU 的异步单元可以在 warp 等待时继续执行；不能将 no eligible 60.5% 直接当作 GPU 60.5% 完全空闲。

### SASS 对热点的进一步归因

1. 最大 long-scoreboard 热点在 PC 0x7fe9cb5c1d30，207877 个样本，约占该指标全部 309294 个样本的 67.21%。它是消费 P0 的条件跳转，前面生成 P0 的指令为：
   `SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR5+0x100], R2`。
   因此不能把这个 long-scoreboard 标签简单解读成等待 HBM 读取；该热点对应异步 barrier 检查/阶段同步依赖。

2. 两个最大 barrier 热点分别为 PC 0x7fe9cb5c3b70 / 0x7fe9cb5c3f90，共 177545 个样本，占全部 barrier 采样的 95.05%。两处都紧跟 `BAR.SYNC.DEFER_BLOCKING ..., 0x40`，随后读取 shared memory。证据指向 block 内部 warp 之间的同步等待。

3. 源码将 softmax 分给 warp 0–7、结果 correction 分给 warp 8–11、MMA 给 warp 12、加载给 warp 14。可确认不同工作阶段依靠同步衔接；后续结合 SharedStorage 偏移与 SASS 前后序列进一步关联到 softmax/correction 交接，见文末；没有据此断言某个阶段的实际计算是唯一根因。

同时记录到约 879 万 local spilling requests、约 33% excessive global sectors。这些是后续优化线索，尚未做控制变量实验确定各自的可回收时间；不能把所有等待都归因于 spilling 或非连续 V。

### 当前结论

瓶颈更明确地表现为 FA4 block 内部阶段同步/数据依赖，以及单 block 占用较多寄存器和 shared memory 后的延迟隐藏能力不足。它已合批、已有足够多待执行 blocks；更大的 batch 主要增加执行轮数。HBM 带宽未饱和，也没有“数学流水线排队占主导”的证据。

优先考虑 kernel 内部 tile、pipeline stage、warp 间生产消费同步及寄存器使用的控制变量调优，并保持数值正确性。单独增大 batch 或 stream 的收益已由前面的实测否定；tile 调优已由文首交替顺序实验验证约 5% 收益，其他调优尚未验证。

原始证据：

- [完整计数器报告](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-stall-20260916-ezujr5v1/stalls.ncu-rep)
- [NCU 文字明细](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-stall-20260916-ezujr5v1/details.txt)
- [所有原始指标](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-stall-20260916-ezujr5v1/stalls.csv)
- [SASS 与 stall 采样](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-stall-20260916-ezujr5v1/source.csv)
- [热点 PC 及上下文](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-stall-20260916-ezujr5v1/top-stall-pcs.json)
- [归因摘要](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-stall-20260916-ezujr5v1/bottleneck-summary.json)

本次采集已结束，GPU0 没有残留测试进程。

## 补充：扩大 tile 的初步验证

只在独立 harness 中通过内部 _flash_attn_fwd(tile_mn=...) 调参，未改生产模块或 FA4 依赖文件。沿用 batch1、同一随机种子及完整 shape/stride；每档预热并做 FP32 抽查后，五轮、每轮 20 次。两个配置均通过全量有限值、FP32 抽查和重复抽样一致性检查。

| tile M×N | 中位 CUDA ms | 有效计算 MFU（2250 TFLOP/s） |
|---|---:|---:|
| 128×128 | 11.908550 | 39.81% |
| 128×160 | 11.285136 | 42.01% |

128×160 在本次 batch1 顺序 A/B 初测中降低耗时 5.24%，速度比 1.0552。这是可运行且数值检查通过的候选，该初测之后已完成交替顺序及 batch32 复测，见文首；真实 ViT 回归仍未执行，生产默认值未修改。

### 为什么不能直接设 2048

当前 varlen/head_dim72 路径 q_stage=2，head_dim_v_padded=80。源码 Tensor Memory 列布局是 2 * tile_N + q_stage * 80：

- N=128：416 列。
- N=160：480 列，本次编译及运行通过。
- N=192：544 列。
- N=2048：4256 列，超过当前每 CTA 512 列上限，不能直接套用此布局。

如果用户指 M=2048，则 Q 和 O 的 shared memory 仅两者就需要 2 stages * 2048 rows * 80 dims * 2 bytes * 2 buffers = 1310720 bytes = 1.25 MiB，不含 K/V 等，已经超过当前约 227 KiB/block 的可用 shared memory。这里是当前源码布局的资源计算，未执行 2048 的 GPU 测试。

若想保留 2048 的逻辑大块，必须在内部继续分成可驻留的小 tile 或重写分阶段/跨 CTA 算法；这已不是直接修改一个 tile 参数。

- [Tile 实验脚本](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-tile-20260916-9cl19h__/benchmark.py)
- [Tile 原始结果](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-tile-20260916-9cl19h__/bench-result.json)
- [Tile 日志](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-tile-20260916-9cl19h__/benchmark.log)

## 当前瓶颈的进一步归因：softmax / correction 交接

本节复用已有 NCU/SASS 证据，只做源码分析，没有新增 GPU 测试或修改生产代码。

### 已定位的两个主要等待点

当前 q_stage=2、kv_stage=7。按 SharedStorage 的 Int64 barrier 数组及字段排列：

| 字段 | shared-memory 字节偏移 |
|---|---:|
| mbar_S_full_P_full_O_rescaled | 0x90 |
| mbar_P_full_lastsplit | 0xb0 |
| mbar_O_full | 0xd0 |
| mbar_softmax_stats | 0xf0 |
| mbar_softmax_stats 的 empty stage0 / stage1 | 0x100 / 0x108 |
| sScale stage0 / stage1 | 0x15c / 0x35c |

CUTLASS PipelineAsync 将 empty barrier 放在 full barrier 后 num_stages 个 Int64 位置，所以 softmax_stats empty stage0 为 0xf0 + 2*8 = 0x100。

**热点一：softmax 等待 correction 释放槽位。**
最大的 long-scoreboard PC 0x7fe9cb5c1d30 在等待 0x100 的 TRYWAIT 结果。它前面依次是 exp2、BF16 转换、STTM 写概率、0xb0 的 P-last-split 通知，后面是大量 FADD2 行求和。这与 [softmax_step](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-isolated-20260916-3xhwx2nn/entry.runfiles/pip_gpu_cuda13_torch_flash_attn_4/site-packages/flash_attn/cute/flash_fwd_sm100.py:2360) 的 producer_acquire → update_row_sum 顺序相符。因此该热点可关联到 softmax 已写出 P 后，等待 pipeline_sm_stats 的 empty stage0，而不是 Q/K/V 的 HBM 数据读取。

**热点二：correction 等待 softmax 的缩放系数。**
两个最大的 barrier PC 紧跟 64-thread named barrier，随后 LDS 分别读取 0x15c / 0x35c，检查 scale < 1，并进入 TMEM 读、FMUL2、TMEM 写。这与 [correction_loop](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-isolated-20260916-3xhwx2nn/entry.runfiles/pip_gpu_cuda13_torch_flash_attn_4/site-packages/flash_attn/cute/flash_fwd_sm100.py:2485) 的等待缩放系数、读取 sScale、条件性 correction_rescale 完整序列一致。

[correction_loop 的释放顺序](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-isolated-20260916-3xhwx2nn/entry.runfiles/pip_gpu_cuda13_torch_flash_attn_4/site-packages/flash_attn/cute/flash_fwd_sm100.py:2500) 是 q_stage - 1 - stage：处理 stage0 后释放 empty stage1，处理 stage1 后释放 empty stage0。因此两个 Q stage 的推进存在交叉依赖。最大的 softmax stage0 等待，需要 correction 推进到 stage1 才释放。

上述归因由指令顺序和三个 storage 偏移共同支持，属于静态源码/SASS 对应；没有编译器逐行映射或对照实验来证明重排这条链能节省多少时间。等待者慢不等于释放者的算术计算慢，等待可能沿 QK/softmax/correction/PV 链传播。

### 瓶颈排序与优化依据

1. **优先分析 softmax ↔ correction ↔ MMA 的依赖与同步安排。** 两大采样热点对应这条交接链；scheduler 60.5% active cycles 没有 eligible warp，Tensor pipe 活跃约 48.2%。存在内部流水衔接不足的证据，不能将 60.5% 当成可直接恢复的算力或 wall time。
2. **资源占用限制了隐藏延迟的能力。** 512 threads × 128 regs 已用满单 SM 寄存器数，shared memory 约 223 KiB，当前实现还固定申请 512 列 TMEM。不能仅靠增加 batch/stream 再驻留一个同类 CTA 来填等待；寄存器预算和 shared-memory stage 深度要联合考虑。
3. **padding 是已量化的额外工作。** d72→d80 和边界 tile 补齐共增加 14.32% Tensor 运算量，这是运算量增幅，不是测得的 wall-time 占比。它不足以独自解释全部 MFU 差距。
4. **spilling/访存合并是次级候选。** 已观察到约 879 万 local spilling requests 和约 33% excessive global sectors；L1/L2 throughput 约 55.7%/58.6%，DRAM 仅 2.56%。不能根据 long-scoreboard 名称把主因误判为 HBM 带宽，也尚未测出各类 spilling/访问的独立成本。

目前唯一做过性能对照的改变是 N tile 从 128 增到 160：batch1 时间 11.909→11.285 ms，约改善 5.24%。K/V 内层迭代从 ceil(10032/128)=79 次降为 ceil(10032/160)=63 次；每轮工作量也变大，stage 深度等资源配置随之变化，所以不能把这 5.24% 全部归因于“同步次数变少”。这仍说明调整 tile/流水线比继续放大服务 batch 更有实测依据。

128×160 的交替顺序和 batch32 复测已完成，见文首。进一步实验可分别控制 KV stage 深度、softmax/correction 释放时序、寄存器分配，观察热点等待是否下降并验证数值一致性。直接放大到 2048 或只增加线程数没有针对已定位的依赖链。

- [源码与热点对应摘要](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-stall-20260916-ezujr5v1/producer-consumer-analysis.json)
- [SASS 热点上下文](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-stall-20260916-ezujr5v1/top-stall-pcs.json)
- [完整 SASS 采样](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-stall-20260916-ezujr5v1/source.csv)
