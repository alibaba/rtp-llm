# FA4 未满载原因：当前 dense 路径诊断

日期：2026-09-16。GPU0 / L20D、SM103、148 SM；仓库 /home/xieshui.yyx/workspace/RTP-LLM/github-opensource，分支 feat/qwen35_vl_omega，HEAD 2679cfed1b2ae4b3d315c34577e9feb31ef6017f。Python 3.10、Torch 2.11.0+cu130、NCU 2026.1.1。生产文件和依赖源码哈希未变化。

## 结论与证据强度

**当前 head_dim=72 路径受 K/V 供数、softmax/correction/MMA 阶段依赖及有限驻留并行度共同限制。** 最大的 warp stall 不能简单等同于唯一根因：8 个 softmax warp、4 个 correction warp 的等待会放大全体 warp 样本，而真正发射 MMA 的 warp 还明显等待 K/V 就绪。

目前最明确的可改善因素是 72 维物理布局：同值补零到 80 后，Tensor 运算数保持相同，L2 tag 请求减少 37.02%，纯 attention 中位耗时降低 11.67%，输出逐位一致。该证据支持片上访存/供数效率影响性能；尚未把收益进一步精确拆分为 TMA 内部请求切分、边界处理、cache 请求合并各多少。

## 范围与状态

| 项目 | 本轮状态 | 证据/限制 |
|---|---|---|
| 当前 production FA4 路径 | 已核对 | 调用当前 helper，dense rank-4 kernel，148 个 persistent CTA |
| 未加 profiler 的单层计时 | 完成 | 每种控制 6 轮正反交替，每轮 20 次 |
| 数值正确性 | 完成到单层输出 | 正常/均匀输入 FP32 抽查；布局控制所有元素与原路径逐位一致 |
| SASS/PC、occupancy、内存计数器 | 完成 | 当前 dense B1 代表 kernel；不是沿用旧 varlen 采样 |
| batch 扩大 | 引用上一轮 | B1/B32 Tensor 活跃约 56.5%/56.7%；本轮 PC 只采 B1 |
| 真实权重 27 层、完整 MM/LLM 服务 | 本轮未重跑 | 新控制尚未接入生产，不能声称端到端收益 |
| ViT→LLM 实际传输 | 本轮未测 | 不在本次单层 kernel 诊断边界内 |

输入保持每视频 23 段、每段 10032 tokens、16 heads、BF16、非 causal、scale=72**-0.5。原 Q/K stride=(1152,72,1)，V stride=(3456,72,1)。随机 Q/K/V 匹配形状及布局，不是本轮真实模型激活。

计时排除分配、补零、compact、正确性检查。数值全量检查 shape/dtype/finite；控制输出保留前 72 维，额外维度输出为零。布局控制均逐位一致。生产代码未更改。

## 1. 72 维布局增加片上访存请求

| 控制 | 无 profiler 中位 ms | Tensor 活跃率 | L2 tag 请求/次 | 实际 Tensor TFLOP/次 |
|---|---:|---:|---:|---:|
| native72 | 11.37681 | 56.43% | 2,297,884,034 | 12.19368 |
| compactV72 | 11.31938 | 56.43% | 2,293,920,676 | 12.19368 |
| padded80 | 10.04858 | 68.18% | 1,447,096,736 | 12.19368 |
| padded128 | 11.35627 | 96.02% | 354,065,814 | 19.50989 |

- native72 和 padded80 的实际 Tensor 运算都是 12.1936805888 TFLOP；差异不是少算了 attention。
- TMA 指令请求计数均为 11,953,232，但 L2 tag 请求减少 37.02%，L2 sector 数减少 35.92%。
- DRAM 实际访问从 2.122 GB 增至 2.352 GB，耗时反而下降：仅看 HBM 流量不能解释该收益。
- 只 compact V 的改善约 0.50%，接近样本波动；其 Tensor 活跃率未提升，L2 tag 数也几乎不变。
- BF16 每个 head 从 144 B 间隔变为 160 B 间隔，32 B 对齐条件改善。这与实测请求减少相吻合，但这一步同时改变了全局布局和维度边界条件，不能把全部收益单独归给某一种 TMA 事务机制。
- 两档都使用 1-CTA 路径，128 registers/thread、228352 B dynamic shared memory、148 blocks；padded80 的收益没有来自提高 CTA 驻留数。

L2 请求压力指标从 66.73% 降至 50.79%；调度器发射活跃从 45.72% 升至 55.22%；long scoreboard 每 issued instruction 的 warp 周期指标从 3.12 降至 2.17，barrier 从 1.90 降至 1.54。均为 profiler 归因指标，不与无 profiler 时间混算。

## 2. MMA 实际等待什么

根据当前 dense SASS 的操作顺序和 shared-storage 偏移，对应关系如下。这是静态相关定位，不是编译器 line-table 精确映射。

| 等待者 | 当前 PC / barrier | 含义 | FA4 源码 |
|---|---|---|---|
| MMA | 0x7fd6ff5b83a0、0x7fd6ff5b8950；full barrier 基址 0x20 | 等 V/K 的 TMA 搬入完成 | [1725/1764](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-util-20260916-_qm1ml26/entry.runfiles/pip_gpu_cuda13_torch_flash_attn_4/site-packages/flash_attn/cute/flash_fwd_sm100.py:1725) |
| MMA | 0x7fd6ff5b8430、0x7fd6ff5b8c70；0xa0/0xa8 | 等 P 概率块和已修正的 O | [1734](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-util-20260916-_qm1ml26/entry.runfiles/pip_gpu_cuda13_torch_flash_attn_4/site-packages/flash_attn/cute/flash_fwd_sm100.py:1734) |
| softmax | 0x7fd6ff5c0540；0x100 | 等 correction 释放 stats 槽位 | [2360](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-util-20260916-_qm1ml26/entry.runfiles/pip_gpu_cuda13_torch_flash_attn_4/site-packages/flash_attn/cute/flash_fwd_sm100.py:2360) |
| correction | 0x7fd6ff5c0fb0、0x7fd6ff5c13b0；64-thread named barrier | 等 softmax 的 scale，再读 0x15c/0x35c | [2485](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-util-20260916-_qm1ml26/entry.runfiles/pip_gpu_cuda13_torch_flash_attn_4/site-packages/flash_attn/cute/flash_fwd_sm100.py:2485) |
| epilogue | 0x7fd6ff5c2750；0x110 | 等最终 O 就绪 | [2907](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-util-20260916-_qm1ml26/entry.runfiles/pip_gpu_cuda13_torch_flash_attn_4/site-packages/flash_attn/cute/flash_fwd_sm100.py:2907) |

softmax/correction 的反馈顺序是：QK 结果 → row-max/scale → correction 重缩放 O → MMA 可执行 PV；同时 softmax 需要完成 exp/转换/写 P。双 stage 的 stats 槽位释放使用 q_stage-1-stage，存在跨 stage 依赖。[释放位置](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-util-20260916-_qm1ml26/entry.runfiles/pip_gpu_cuda13_torch_flash_attn_4/site-packages/flash_attn/cute/flash_fwd_sm100.py:2500)

全体 warp 平均周期指标：long scoreboard 约 38.0%、barrier 约 23.2%、固定依赖等待约 14.1%、math-pipe throttle 约 0.7%。这些是 warp 周期口径，**不是 kernel/请求耗时占比，也不是可直接回收的加速比例**。最大 long-scoreboard PC 是屏障返回值的依赖；不能直接解读为 HBM load 慢。epilogue 等待完整结果也不代表应优先优化 epilogue。

上述 MMA 循环中，两个 K/V 就绪等待分支各采到 3202、8358 个 long-scoreboard 样本；两个 P/O 就绪等待分支各 2464、4089 个。仅作位置证据，未把局部 PC 子集当作整个 MMA 时间分解。

## 3. 驻留资源使等待难以隐藏

NCU 当前 dense 测量：
- 512 threads/CTA，128 registers/thread（分配口径），寄存器 block limit=1。
- dynamic shared memory=228352 B，含 driver 分配共 229376 B；shared-memory block limit=1。
- 理论 occupancy=25%，实测=23.44%，约 15 个 active warp/SM；模板本就包含一个 empty warp。
- 每个调度器平均只有 0.56 个 eligible warp；54.27% 周期没有可发射 warp。

这解释了为什么增大视频 batch 没有显著提高管线活跃率：persistent CTA 仍是 148 个，增加的是每个 CTA 循环处理的 tile 数，不能多驻留另一个 CTA 去覆盖当前等待。低 occupancy 本身不是结论，须与实际依赖和对照计数器一起看。

## 4. 补到 128 为什么“更满”却没更快

padded128 Tensor 活跃率为 96.02%，但耗时仍约 11.36 ms。它与 72/80 不只是 D 不同：FA4 的选择条件让 padded128 自动启用 2-CTA，NCU 的 cluster_dim_x=2，72/80 都是普通 1-CTA。[选择条件](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-util-20260916-_qm1ml26/entry.runfiles/pip_gpu_cuda13_torch_flash_attn_4/site-packages/flash_attn/cute/interface.py:587)

实际 Tensor 运算由 12.19 增至 19.51 TFLOP（多 60%），额外 head 维是零填充，有效模型计算没有增加。2-CTA、布局和矩阵工作量同时变化，因此不能将该对照解释成单一 head_dim 效应，也不能将 96% 活跃率当成用户请求提速。

## 5. correction 计算的诊断控制

本轮还保持相同形状，比较正常随机 Q 与 Q=0（均匀 softmax），并在实验副本里强制对 scale=1 也执行 O 重缩放。没有改依赖或生产文件。

| 控制 | 中位 ms |
|---|---:|
| production_normal | 11.33010 |
| production_uniform | 9.91193 |
| always_normal | 11.80959 |
| always_uniform | 10.53688 |

强制重缩放相对各自 native，正常输入慢 4.23%，均匀输入慢 6.31%。对应输出与 native 全量逐位相等，FP32 抽查通过。

这一控制说明 correction 的实际工作量会影响 kernel，但修改输入也改变了数据活动、分支和可能的频率/功耗行为；不能把正常与均匀输入之间全部时间差归给 correction，更不能把 Q=0 视为生产优化。未锁 GPU 时钟；所有正式计时均在同进程中正反交替，保留样本，不跨实验绝对时间计算收益。

## 下一步方向与限制

最值得验证的是保留 72 个有效维度、在生产路径以低开销布局准备到 80，并计入 padding/恢复成本，同时重跑真实权重完整 embedding 和完整 ViT 路径。若净收益不足，则继续针对 K/V TMA 请求组织和软最大值/重缩放的阶段交接优化。直接增加 batch、只 compact V 或为了满载补到 128，本轮均未显示有吸引力的净方向。

该建议尚未接入生产。本文诊断完成到代表性 kernel 及模拟输入层；没有宣称完整服务优化已经完成。

## 产物

- [诊断汇总和源码哈希](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-dense-cause-20260916-yq7yn4nj/analysis.json)
- [当前 dense 的 NCU report](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-dense-cause-20260916-yq7yn4nj/stalls.ncu-rep)
- [当前 dense 的 SASS/PC](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-dense-cause-20260916-yq7yn4nj/source.csv)
- [热点 PC 摘要](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-dense-cause-20260916-yq7yn4nj/top-stall-pcs.json)
- [dense 精确采集命令](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-dense-cause-20260916-yq7yn4nj/ncu-command.json)
- [布局控制无 profiler 6 轮原始样本及正确性](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-shape-control-20260916-ceyc8l_b/bench-result.json)
- [布局控制脚本](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-shape-control-20260916-ceyc8l_b/benchmark.py)
- [布局控制 NCU report](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-shape-control-20260916-ceyc8l_b/shape.ncu-rep)
- [布局控制 NCU 命令](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-shape-control-20260916-ceyc8l_b/ncu-command.json)
- [绝对请求数和运算数 report](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-shape-control-20260916-ceyc8l_b/counts.ncu-rep)
- [绝对计数汇总](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-shape-control-20260916-ceyc8l_b/counts-summary.json)
- [绝对计数命令](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-shape-control-20260916-ceyc8l_b/counts-command.json)
- [correction 控制原始样本](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-correction-control-20260916-7q7nw439/bench-result.json)
- [correction 实验副本](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-correction-control-20260916-7q7nw439/always_rescale.py)

所有本轮 GPU 进程均已退出，GPU0 无残留计算进程。只在用户授权的 RTP-LLM 根目录下创建实验和文档，未提交或推送。


## 后续实现验证

本文诊断后，已将只补 Q/K 到 80 并入生产 RoPE 转换，完成真实权重 batch1/32 全 embedding 逐位一致检查及正式性能复测。代表 kernel Tensor 活跃率实测 56.42%→66.01%。见 [Q/K 布局优化与最终结果](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/docs/multimodal/qwen35_fa4_padding_20260916.md)。本文前述“尚未接入”描述诊断时的状态。
