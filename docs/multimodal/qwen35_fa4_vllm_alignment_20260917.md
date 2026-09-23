# Qwen3.5 ViT：对齐 vLLM FA4 的性能对照

日期：2026-09-17。设备：GPU0，NVIDIA L20D / SM103，BF16。

## 结论

在本轮等长序列的纯 FA4 前向测试中，完整对齐 vLLM 的 kernel、依赖和 varlen 调用后，B1 耗时增加 **13.55%**，B32 增加 **10.12%**。保持当前 dense D80 布局、只更换为 vLLM kernel 和依赖时，B1 基本持平，B32 中位数降低 **1.80%**；这一幅度小于本轮基线波动，不能认定为稳定收益。

所有参与正式比较的输出逐元素一致，FP32 抽样参考检查通过。这次没有获得值得替换当前 BF16 attention 路径的性能收益。实验代码、依赖及报告均写在用户 workspace/RTP-LLM 下。

## 对齐范围与版本

- RTP-LLM：/home/xieshui.yyx/workspace/RTP-LLM/github-opensource，分支 feat/qwen35_vl_omega，HEAD 2679cfed1b2ae4b3d315c34577e9feb31ef6017f；使用工作区现有 attention 实现。
- vLLM：/home/xieshui.yyx/vllm，HEAD bb233626caa31602728f7ee4625f3d2a4d1a3ad5。
- vLLM 固定 FA4 提交：617264c1c7955c9e84817654ebeedff069f3c5f1，依据本机 [CMake 配置](/home/xieshui.yyx/vllm/cmake/external_projects/vllm_flash_attn.cmake:42)。
- 原环境：flash-attn-4 4.0.0b21、CUTLASS DSL 4.6.0.dev0、QuACK 0.5.3。
- 对齐环境：上述固定 FA4 源码，CUTLASS DSL 4.6.2（base、core、cu13 同版本），QuACK 0.6.4。
- 公共运行时保持 Torch 2.11.0+cu130、CUDA 13.0、Python 3.10；没有安装完整 vLLM 服务。
- 本机 vLLM 的 flash_attn_interface.py wrapper 按字节复制；FA4 源码仅按 vLLM CMake 的方式替换 namespace 为 vllm.vllm_flash_attn.cute，未修改算法。
- 完整对齐组调用 flash_attn_varlen_func，fa_version=4，Q/K/V 有效维度 72，num_splits 使用 vLLM 默认值 0。实际 kernel 未启用 SplitKV。

## 输入与计时

沿用当前诊断形状，用固定种子生成正态随机 Q/K/V：

- B=1、32；每个样本有 23 个独立 attention 段，每段长度 10032。
- 16 个 head，有效 head dimension=72，非 causal，softmax scale=72**-0.5。
- Q72/K72 连续，V72 stride=(3456,72,1)；dense 组的 Q80/K80 是相同有效数据的零填充版本，V 保持 D72。
- 所有进程对相同 batch 使用相同种子、相同生成顺序；交叉检查 Q 输入哈希和完整输出哈希。

计时范围是**预热后的纯 FA4 前向 API**，包含正常输出分配与 launch。输入生成、padding、RoPE、视频读取、解码、其余 ViT、传输、正确性检查和哈希均在计时区间外。这里的 batch 不是服务并发数。

使用 CUDA Events，每组 B1 连续调用 20 次、B32 连续调用 3 次，取组内平均值。每种方案 6 组采样；原版本在新版本前后各测 3 组并合并，新环境内三种方案以正序/逆序交错采样。正式比较使用六组的中位数。

GPU0 测试前、采样前后均检查其他 compute PID。未人为锁频；前后遥测保存在原始 JSON，不将微小差异认定为稳定收益。

## 正式结果

耗时为采样中位数。MFU 按用户提供的 BF16/FP16 峰值 2250 TFLOP/s 归一化。

| 方案 | B1 耗时 ms | B1 MFU | B32 耗时 ms | B32 MFU |
|---|---:|---:|---:|---:|
| 当前 FA4 + 原依赖 + dense D80 | 10.396 | 45.60% | 362.270 | 41.87% |
| 当前 FA4 + 新依赖 + dense D80 | 10.460 | 45.32% | 359.979 | 42.14% |
| vLLM FA4 + 新依赖 + dense D80 | 10.409 | 45.54% | 355.757 | 42.64% |
| vLLM FA4 + 新依赖 + vLLM varlen D72 | 11.804 | 40.16% | 398.934 | 38.03% |

| Batch | 方案 | 均值 ms | 最小–最大 ms |
|---:|---|---:|---:|
| 1 | 当前 FA4 + 原依赖 + dense D80 | 10.374 | 10.309–10.422 |
| 32 | 当前 FA4 + 原依赖 + dense D80 | 363.773 | 355.371–382.772 |
| 1 | 当前 FA4 + 新依赖 + dense D80 | 10.462 | 10.429–10.500 |
| 32 | 当前 FA4 + 新依赖 + dense D80 | 367.755 | 353.295–415.910 |
| 1 | vLLM FA4 + 新依赖 + dense D80 | 10.390 | 10.283–10.483 |
| 32 | vLLM FA4 + 新依赖 + dense D80 | 357.203 | 354.388–363.618 |
| 1 | vLLM FA4 + 新依赖 + vLLM varlen D72 | 11.802 | 11.773–11.831 |
| 32 | vLLM FA4 + 新依赖 + vLLM varlen D72 | 398.893 | 398.583–399.006 |

原版本 B1 前后两轮中位数为 10.386 / 10.408 ms；B32 为 362.984 / 356.276 ms。B32 存在约几个百分点的环境/执行波动。

完整 vLLM varlen 与同一新依赖环境中的 vLLM dense 直接相比，B1 慢 **13.41%**，B32 慢 **12.14%**。退化方向不依赖只选前一轮或后一轮基线。

## MFU 计算

只计有效 QKᵀ 和 PV 两个矩阵乘法的算法 FLOPs：

    F(B) = 4 × B × 23 × 10032² × 16 × 72
    F(1)  = 10.666338287616 TFLOP
    F(32) = 341.322825203712 TFLOP
    有效 TFLOP/s = F(B) / 耗时秒
    MFU = 有效 TFLOP/s / 2250 × 100%

完整 vLLM B32：341.322825203712 / 0.3989339599609375 / 2250 = **38.0261%**。

2250 是用户指定的归一化峰值。此 MFU 不等于 Tensor Core pipeline 活跃率；本轮没有重新采集 NCU 流水线计数器。

## 实际 kernel 与调度

单独使用 Torch profiler 做短 trace，正式性能数据不采用该次 profiling 的耗时。通过 kernel 构造参数和实际 CUDA launch 同时确认：

| 属性 | vLLM kernel + 当前 dense 布局 | 完整 vLLM varlen 布局 |
|---|---|---|
| 内核 | FlashAttentionForwardSm100 | FlashAttentionForwardSm100 |
| 内部 Q/K 和 V 计算维度 | 80 / 80 | 80 / 80 |
| tile M×N / Q stages / KV stages | 128×128 / 2 / 7 | 128×128 / 2 / 7 |
| Scheduler | StaticPersistentTileScheduler | SingleTileVarlenScheduler |
| Persistent | 开 | 关 |
| TMA Q/K/V load | 开 | 开 |
| TMA output store | 开 | 关 |
| 2CTA / SplitKV | 关 / 关 | 关 / 关 |
| B1 launch grid | (148,1,1) | (14784,1,1) |
| B32 launch grid | (148,1,1) | (473200,1,1) |
| Threads/block，registers/thread | 512，128 | 512，128 |
| Shared memory/block | 228352 bytes | 228352 bytes |

vLLM 的 D72 输入在 kernel 内仍按 D80 计算，没有减少矩阵乘法维度；varlen 使用不同的任务调度和输出写回路径。其 tile_scheduler.py 与当前安装版本逐字节相同，BF16 路径未带来明显收益。

完整对齐组同时改变调用接口与 Q/K 物理布局，本轮尚未单独量化两项改变各自的耗时贡献。trace 证实执行路径差异，不将全部性能差额归因于某一个开关。

参考：[当前 dense 入口](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_moe_vit.py:67)、[vLLM wrapper](/home/xieshui.yyx/vllm/vllm/vllm_flash_attn/flash_attn_interface.py:423)、[FA4 scheduler 选择](/data2/xieshui.yyx/workspace/RTP-LLM/.t/fa4-vllm-align-20260917-sd4az4wc/aligned/vllm/vllm_flash_attn/cute/flash_fwd_sm100.py:246)、[TMA 输出开关](/data2/xieshui.yyx/workspace/RTP-LLM/.t/fa4-vllm-align-20260917-sd4az4wc/aligned/vllm/vllm_flash_attn/cute/flash_fwd_sm100.py:190)。

## 正确性、预热与结束状态

- B1 完整输出 265,807,872 个元素，B32 完整输出 8,505,851,904 个元素；各版本逐元素一致、无非有限值。
- 完整输出 SHA256 在前后基线和新环境全部匹配，输入 Q72 SHA256 也一致。
- 首/中/末三个 attention 段，各抽首/中/末查询位置、全部 head，与 FP32 dense attention 参考比较通过。
- 数值容差 atol=5e-4、rtol=2e-2；跨版本完整输出实际达到 bitwise equal。
- 第一次新环境测试完成 B1 后，在 B32 单次预热稳定性检查中止。保留已完成且通过首尾校验的 B1 六组正式样本；该次 B32 无正式计时样本，不纳入结果。
- B32 随后使用每组三次调用平均值检查预热稳定性：至少五组预热、连续三次变化小于 3% 后开始正式采样，完成六组结果；后置原版本 B32 使用相同预热方式。
- 正式计时离群值没有删除，均值、范围和所有原始样本可查。
- 原有 tracked 源文件与实验期间快照哈希一致。所有实验进程已退出，结束时 GPU0 没有 compute PID。

结论仅覆盖当前 BF16、等长独立 attention 段形状，不代表异长混合输入、FP8 ViT、CUDA Graph 或完整服务性能。

## 数据与复现入口

实验目录：/data2/xieshui.yyx/workspace/RTP-LLM/.t/fa4-vllm-align-20260917-sd4az4wc

- [汇总数据](/data2/xieshui.yyx/workspace/RTP-LLM/.t/fa4-vllm-align-20260917-sd4az4wc/comparison-summary.json)
- [基准脚本](/data2/xieshui.yyx/workspace/RTP-LLM/.t/fa4-vllm-align-20260917-sd4az4wc/benchmark.py)；原始脚本保存在 benchmark-original.py
- [前置原版本](/data2/xieshui.yyx/workspace/RTP-LLM/.t/fa4-vllm-align-20260917-sd4az4wc/old-before-result.json)、[后置原版本](/data2/xieshui.yyx/workspace/RTP-LLM/.t/fa4-vllm-align-20260917-sd4az4wc/old-after-result.json)
- [新环境 B1，含中止的 B32 预热记录](/data2/xieshui.yyx/workspace/RTP-LLM/.t/fa4-vllm-align-20260917-sd4az4wc/new-main-result.json)、[新环境 B32](/data2/xieshui.yyx/workspace/RTP-LLM/.t/fa4-vllm-align-20260917-sd4az4wc/new-main32-result.json)
- [实际 kernel/调度与 trace 摘要](/data2/xieshui.yyx/workspace/RTP-LLM/.t/fa4-vllm-align-20260917-sd4az4wc/dispatch-summary.json)
- [源码来源和 SHA256](/data2/xieshui.yyx/workspace/RTP-LLM/.t/fa4-vllm-align-20260917-sd4az4wc/source-manifest.json)、[kernel 源码哈希比较](/data2/xieshui.yyx/workspace/RTP-LLM/.t/fa4-vllm-align-20260917-sd4az4wc/kernel-source-comparison.json)
- [源文件与 GPU 结束状态检查](/data2/xieshui.yyx/workspace/RTP-LLM/.t/fa4-vllm-align-20260917-sd4az4wc/final-audit.json)

每次实际执行命令与 PID 记录在对应的 command.json 文件，日志为同名前缀的 .log；独立 profiling 产生 dispatch-B 开头的 Chrome trace。
