# 纯 FA4 测量边界与当前小 head 路径分析

2026-09-17。结论：此前测量已经是 warmed FA4 forward API 的独立性能。当前代码及依赖 SHA256 与采样时一致。本轮复核源码和已有硬件记录，没有新增 GPU 采样；GPU0 上有另一个 ViT 阶段测试，本轮未干预它。当前结论仅针对 Q/K80、V72、SM103 的这一条路径，不泛化为所有 FA4 形状。

## 1. “纯 FA4”具体包含什么

计时从 GPU 输入已准备好之后开始，到 FA4 输出完成为止。生产 helper 做 CPU 条件检查、Q/K/V view、调用 flash_attn.cute.flash_attn_func、输出 reshape。三者最后一维 stride 都为 1，FA4 maybe_contiguous 不触发拷贝；output 是 FA4 内部 torch.empty 创建的连续 tensor，reshape 不需搬数据。输入分配、补零、RoPE、视频处理、QKV projection、ViT MLP/projection 均在测量边界外。

常规接口内部的输出分配、Python/CUDA dispatch 属于这个 API 测量；并未用预分配输出加 CUDA Graph 将 host launch 间隙全部剥除，因此不声称 helper 时间严格为零。编译和预热在正式计时外，requires_grad=False，return_lse=False，num_splits=1，没有 LSE 或 SplitKV combine kernel。

| 视频 batch | 无 profiler FA4 API 时间 | 有效 FLOPs / 2250 TFLOP/s 的 MFU |
|---:|---:|---:|
| 1 | 10.354673 ms | 45.7822% |
| 32 | 356.817220 ms | 42.5145% |

单独捕获的 B32 FlashAttentionForwardSm100 kernel 在 NCU 中耗时 333.36–336.84 ms，说明 kernel 本身就是数百毫秒量级；这不是把其他 ViT 阶段混进 FA4。Profiler 改变运行环境，该时间只用于归因，不能从 356.82 ms 减去它来算 helper 开销，也不能替代无 profiler 的正式耗时。

## 2. 当前 shape 为什么难以达到 GEMM 峰值

每视频 23 段，每段 10032 tokens，16 heads，有效维度 72。每个 score 元素的 QK+PV 有效计算为 4×72=288 FLOPs。固定段长和 heads 时，矩阵工作量随 head_dim 变化，而 softmax 的 score 数并不随之变化，因此小 head 对 exp、归约、修正和阶段同步的摊销更弱。这是形状层面的分析，不是已测出的 softmax 耗时百分比。

当前 tile=128×128，q_stage=2，每个 work tile 对应 256 个 query rows；每个段/head 有 ceil(10032/256)=40 个 work tile，每个 work tile 要处理 ceil(10032/128)=79 个 K/V 块。B1 有 14720 个 work tile，B32 有 471040 个，已有大量可调度任务。

接口的 2-CTA 条件要求 Q/K padded head 处于 128/192 且 V padded head=128；当前 Q/K80、V72（内部 V 对齐80）走 1-CTA。当前调优表没有 (False,False,80,True) 项，采用小 head 的通用寄存器分配：softmax=200、correction=64、other=48。这只能证明配置选择，没有证明“没有表项”本身就是全部损失来源。

## 3. 真正的依赖链

MMA 循环的关键顺序是：

1. 等 V 的 TMA 搬入完成。
2. 等 P 可用、旧 O 已完成重缩放。
3. 执行 PV。
4. 等下一个 K 的 TMA 完成。
5. 执行 QK，通知 softmax。
6. softmax 做 row-max/scale、exp、BF16 转换、写 P；correction 按 scale 修正 O，释放下一次 PV 的依赖。

这些阶段可以重叠一部分，但还需要反复交接。最新 B32 计数器：Tensor pipe active 66.52%，issue active 53.58%，eligible warp/scheduler 0.668；此前同一 Q/K80、V72 的 B1 捕获也有 barrier=1.5794、long scoreboard=2.2760 的 warp 周期/issued instruction。结合源码，证据支持阶段依赖和有限延迟隐藏；不能将 warp 指标直接换算成 B32 各阶段耗时。

旧 native72 dense 的 PC/SASS 曾把等待定位到 K/V ready、P/O ready、softmax/correction stats 交接。当前源码仍有这些依赖，但旧 PC 样本分布不能冒充当前 Q/K80 B32 的重新采样。

## 4. 为什么 batch=32 仍掩盖不了等待

当前 launch 是 148 个 persistent CTA，512 threads/CTA，128 allocated registers/thread。每个 CTA 共占 shared memory 224 KiB，配置容量228 KiB；寄存器和 shared memory 均只允许 1 CTA/SM。

理论 occupancy=25%，实测23.44%，模板包含一个空 warp。工作分配是8个 softmax warp、4个 correction warp、1个 MMA warp、1个加载warp、1个输出warp、1个空warp。MMA warp 发出的是异步 Tensor 指令，不能将“1个MMA warp”直接解释为只使用1/16算力。

B1→B32 的 Tensor active 66.03%→66.52%，说明增加任务主要延长同一组 persistent CTA 的循环，并未显著改善内部阶段重叠。低 occupancy 单独不证明瓶颈，但与发射不足、就绪 warp 少和依赖链结合，支持延迟隐藏不足。[NVIDIA 调度与 occupancy 指标](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

DRAM throughput=6.5%–8.5%，没有 HBM 带宽饱和证据；仍可能受片上请求、TMA/共享内存延迟、生产者交付时间限制。不能把所有 long scoreboard 都解释为 HBM load。

## 5. 已经做过的对照如何支持分析

| 控制 | 结果 | 能说明什么 |
|---|---|---|
| 早期 native72 → padded80 | 同次实验纯 FA4 11.37681→10.04858 ms，实际 Tensor FLOPs 相同，L2 tag 请求减少37.02% | 物理布局/供数效率可显著影响性能；已吸收到当前 Q/K padding 改动 |
| 早期 padded80 → padded128 | Tensor active 68.18%→96.02%，耗时10.04858→11.35627 ms | 更高活跃率未带来更快有效计算；实际 Tensor 工作量多60%，且2-CTA也改变 |
| 当前 split_P_arrive 96→64 | B32 356.34210→365.24872 ms，慢2.50% | 仅提前通知阈值没有改善整体阶段重叠；该候选已否决 |

各控制仅在同次实验内比较，不把不同日期的绝对时间混算收益。padded80 历史控制与当前仅补 Q/K 的生产输入有区别，不能将其全部计数器套到当前路径。

当前 softmax 在 flash_fwd_sm100.py:2333 先调用整块 apply_exp2_convert，再于2344行按片段写P并通知MMA。因此把首批通知从96列改成64列，只改变写P阶段的通知时刻，没有显式将前64列的exp/转换提前执行。这解释了该改动没有真正实现预想中的“前半计算先交付”；它不是所有性能损失的定量解释。

## 6. MFU 的确定分解和下一步

B32 有效计算341.323 TFLOP，硬件实际执行390.198 TFLOP，时间0.356817 s：

- 实际执行吞吐1093.55 TFLOP/s，除以用户给定2250得到48.60%。
- 有效/实际执行操作=87.4743%，剩余12.5257%为维度与tile边界填充。
- MFU=48.6023%×87.4743%=42.5145%。

最大差距在实际执行吞吐相对参考峰值的不足，padding 再降低有效比例。2250 是用户给定参考，尚未用同机大 BF16 GEMM 建立实际持续峰值，也未采无 profiler 执行中的频率/功耗曲线，因此不能把全部差距定量分摊给同步、softmax、供数或频率。

下一步最值得对照的方向是：为 SM103 head80/72 单独调整 softmax/correction 配比、K/V stage 与寄存器预算；研究将 P 的 exp/转换/写入真正分段，提前释放前半块 PV，而不只是改变通知阈值。各候选都需完整数值检查和同形状无 profiler 对照。该方向目前是待验证优化假设，不承诺提升比例。

本轮完成纯 FA4 测量边界、已有计数器和当前源码的核对；未修改生产代码、未新增受背景负载影响的性能数字，未运行完整 ViT/LLM。

## 证据

- 原始测量与逐次计数器：/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-current-util-20260917-h3fnzfj8/analysis.json
- 本轮边界和源码核对：/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-current-util-20260917-h3fnzfj8/pure-fa4-audit.json
- FA4源码、helper、每处依赖的绝对路径和行号见 pure-fa4-audit.json。
- 历史布局控制：qwen35_fa4_bottleneck_20260916.md。
- 最近96→64对照：qwen35_fa4_split_p_20260917.md。
