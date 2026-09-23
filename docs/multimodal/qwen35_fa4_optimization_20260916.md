# Qwen3.5 ViT FA4 优化验证

日期：2026-09-16。仓库：/home/xieshui.yyx/workspace/RTP-LLM/github-opensource，分支 feat/qwen35_vl_omega，HEAD 2679cfed1b2ae4b3d315c34577e9feb31ef6017f。仅 GPU0，L20D / SM103。保留原有工作区修改，未提交或推送。

## 最终代码方案

对 SM103、BF16、16 heads、head_dim=72、每段长度至少 1024 且所有段等长的 Qwen3.5 ViT attention，调用现有 FA4 公开定长接口 flash_attn_func。将原 [总 tokens, heads, dim] 张量零复制 view 为 [段数, 每段 tokens, heads, dim]，每个视频帧段保持独立，计算后恢复原 shape。混合段长、较短段、其他 dtype/head shape/GPU 继续使用原 varlen 接口。

使用 FA4 默认 128×128 tile；未更换精度、权重或 FA4 wheel，未修改依赖源码。该定长路径采用现有 persistent scheduler 和 TMA 输出路径。相同计算和分段下减少部分调度/数据交接开销，尚不能声称消除了全部同步等待或达到 Tensor Core 满载。

实现：
- [attention 选择与执行](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_moe_vit.py)
- [段隔离、回退与 CUDA Graph 测试](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/test/qwen35_vision_kernels_test.py)

## 当前验证状态

10 个针对性测试全部通过，无跳过。包括非连续 V、段间隔离、混合长短段回退、FP16/其他 head_dim 回退、真实 attention CUDA Graph 重放及原有 RoPE/视觉图测试。
完整多模态路径 batch1/32 的最终交替对照已通过。四轮中，每档每轮的全部 embedding 和位置编码均与同 batch 基线逐位一致。

## 最终修改后的代码：无 profiler 正式结果

| 视频 batch | 基线中位 ms | 优化中位 ms | 中位耗时降低 | 基线平均 ms | 优化平均 ms |
|---|---:|---:|---:|---:|---:|
| 1 | 746.955 | 717.002 | 4.01% | 747.017 | 717.398 |
| 32 | 22894.304 | 22342.836 | 2.41% | 22907.417 | 22406.488 |

每档每种路径四个完整请求；表中中位数是四次完整请求的中位数。样本量不足以报告 P99。原始请求耗时：

- batch=1, baseline: 744.064, 749.846, 740.330, 753.826 ms。

- batch=1, production: 716.512, 719.113, 716.474, 717.491 ms。

- batch=32, baseline: 22971.983, 22890.221, 22869.076, 22898.387 ms。

- batch=32, production: 22294.876, 22373.696, 22645.406, 22311.976 ms。

路径计数：{"baseline": 486, "production": 405}；定长接口调用：{"production": 405}。基线未调用定长接口，优化调用确实执行。

- batch=1, baseline，该轮流程峰值 allocated：8.433 GiB。

- batch=1, production，该轮流程峰值 allocated：8.433 GiB。

- batch=32, baseline，该轮流程峰值 allocated：238.038 GiB。

- batch=32, production，该轮流程峰值 allocated：238.038 GiB。

正式样本的前后检查均没有 GPU0 其他计算进程。该方案缓解了部分 attention 调度/交接开销，没有修改同步协议或提高数值误差容限；不声称剩余 kernel 等待已被全部解决。

## 测量范围与输入

起点 MMProcessEngine 收到请求，终点全部 embedding/hash 在 GPU 完成。包括视频文件加载、metadata、NVDEC、GPU resize/预处理、27 层 ViT、merger/token assembly；不包括 gRPC 序列化、网络/RDMA、LLM prefill/decode。这是多模态子系统测量，不是完整 LLM 请求端到端结果。

- 真实权重：/mnt/nas1/hf/Qwen3.5-397B-A17B-FP8；ViT 实际 BF16。
- 媒体：/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-request1-nvdec.JBWUyk/request1.mp4。
- fps=6，min_pixels=2500000，max_pixels=73728000，max_frames=180。
- 每视频 23 个 attention 段，每段 10032 tokens；输出 [57868,4096] BF16，位置 [57868,3]。
- batch 指单次打包的视频数，不是独立 HTTP 并发；本轮 batch1/32。
- ViT concurrency=64，GPU max batch size=32，GPU max batch images=256，batch wait=10 ms；NVDEC workers=32。
- embedding/URL/字节缓存关闭，CUDA Graph 关闭；CPU/GPU cache max bytes=0。
- 正式结果无 profiler；每档双方分别预热至相邻耗时稳定，再交替 AB/BA 四轮。每轮完整请求计时，正确性比较和监测不计入请求耗时。
- 最终正确性门槛：同 batch 基线与优化的每一份完整 embedding 和位置编码逐位相等。

## 候选实验与取舍

之前的随机 Q/K/V、单层 FP32 抽查不足以保证真实 27 层输出一致。128×160 随机单层约改善 5%，但真实视频最终 embedding 与基线相对 L2 偏差为 0.016315（1.63%），最大绝对差 0.8203125，超过实验前设定的 relative L2<=1% / cosine>=0.9999 门槛，因此拒绝接入。没有据此扩大容差。

保持 128×128 时，提前执行 softmax 行求和、寄存器重新分配、两者组合、定长调度均通过了真实视频完整首份 embedding 的逐位一致检查。六轮完整多模态路径 batch1 初筛如下，非最终生产代码对照：

| 候选 | 中位 ms |
|---|---:|
| 默认 varlen | 746.531 |
| 行求和提前到槽位等待之前 | 737.820 |
| softmax/correction/other 寄存器 192/80/48 | 740.204 |
| 上述两项组合 | 727.116 |
| 原生定长调度 | 724.360 |

随后对同一模拟输入做四轮正反顺序的单层 kernel 实验：

| 候选 | 中位 ms |
|---|---:|
| 默认 varlen | 12.148381 |
| 原生定长 | 11.324815 |
| 定长 + 行求和提前 | 11.360335 |
| 定长 + 行求和提前 + 寄存器调整 | 11.318397 |
| varlen + 行求和提前 + 寄存器调整 | 11.600832 |

原生定长 kernel 相对默认降低约 6.78%；定长再叠加自定义 kernel 仅多约 0.057%，不足以支持维护自定义 kernel。最终采用原生公开接口。随机 kernel 的数值门槛：全量有限值、指定 query 的 FP32 dense attention 参考 atol=5e-4/rtol=2e-2、重复抽样一致性。保留真实 Q/K/V shape、stride、BF16 和分段，不计分配/随机数/验证时间。内核实验不是完整多模态路径数据。

## Nsight 路径核对

单独采集的 batch1 模拟输入 trace 确認两条路径均只有一个 fused attention kernel；以下 launch 参数来自实测：

| 路径 | grid blocks | threads/block | registers/thread | dynamic shared memory/block |
|---|---:|---:|---:|---:|
| baseline | 14784 | 512 | 128 | 228352 B |
| production | 148 | 512 | 128 | 228352 B |

定长路径使用 148 个 persistent CTA 循环处理原有全部段和 tile，替代 varlen 的 14784 个 CTA。寄存器及 shared-memory 驻留预算仍相同，TMEM 容量没有扩大。这确认了执行路径改变；不能据此把全部收益定量归因于某一种同步或初始化开销。两条路径 trace 输出逐位相等。此次没有重采 NCU，不将早期 varlen 的 stall 百分比当作优化后的测量值。

Profiler 时间仅用于路径归因，不替代前面的无 profiler 正式 RT。这里只采 batch1，不将其 launch 数据标为 batch32。

- [Nsys trace](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-dispatch-trace-20260916-8cpzgxa7/fa4.nsys-rep)
- [关联 kernel 与 launch 参数](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-dispatch-trace-20260916-8cpzgxa7/trace-summary.json)
- [精确采集命令](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-dispatch-trace-20260916-8cpzgxa7/command.json)

## 原始产物

- [8 项 kernel 初筛](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-tune-20260916-hsbhbf9h/bench-result.json)
- [128×160 真实权重偏差](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit-fa4-tune-20260916-kdkh2ymh/benchmark-result.json)
- [保持 128×128 的真实视频候选对照](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit-fa4-preserve-20260916-qnodxok4/benchmark-result.json)
- [定长与自定义 kernel 组合对照](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-adapter-20260916-8pi88jj9/bench-result.json)
- [10 项测试结果](/home/xieshui.yyx/workspace/RTP-LLM/.t/fa4-dense-tests-20260916-p6tp5gc7/result.json)
- [最终生产代码对照](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit-fa4-final-20260916-cy03e0w2/benchmark-result.json)
- [最终运行日志](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit-fa4-final-20260916-cy03e0w2/experiment.log)
- [最终精确命令与源码哈希](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit-fa4-final-20260916-cy03e0w2/command.json)

本轮测试与 profiler 均已结束。未测试完整 LLM 请求或 ViT→LLM 的网络/RDMA/gRPC 传输。

## 后续利用率检查

本报告完成后，另对当前 dense 路径重采了 batch1/32 的 NCU 计数器；参见 [当前 FA4 利用率](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/docs/multimodal/qwen35_fa4_utilization_20260916.md)。本报告前述“此次没有重采 NCU”指原优化验证阶段。
