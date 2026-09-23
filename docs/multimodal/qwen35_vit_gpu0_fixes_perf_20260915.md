Qwen3.5-397B 独立 ViT 修复与性能记录（2026-09-15）

本轮已修复大 batch 下的 RoPE 索引溢出、LayerNorm 数值错误和 patch embedding 卷积数值差异。最终在物理 GPU 0、GPU batch 上限 64、缓存关闭的配置下，135 条完整输出校验与历史基准逐元素一致，1,147 条正式性能请求全部通过。

本记录仅整理当前配置及其性能数据。128 请求并发下，三轮合并吞吐为 3.6947 请求/s，折合约 74,782 个输出视觉 token/s。

1. **环境与模型**

| 项目 | 实际值 |
|---|---|
| 机器 / GPU | b300_docker / 物理 GPU 0，NVIDIA L20D，SM 10.3 |
| GPU 可见性 | CUDA_VISIBLE_DEVICES=0 |
| 工作目录 | /home/xieshui.yyx/workspace/RTP-LLM/github-opensource |
| 测试分支 | feat/qwen35_vl_omega |
| 测试基准提交 | de75ab34d3bbfb49302643c2af1c8bbc10f35d6d，叠加本轮两份源码文件的修复 |
| 模型路径 | /mnt/nas1/hf/Qwen3.5-397B-A17B-FP8 |
| 加载范围 | 仅 ViT，456,010,480 个参数，不加载语言模型 |
| ViT 精度 / attention | BF16 / FA4；rotary inv_freq 保持 FP32 |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |

Checkpoint 名称中的 FP8 不表示本轮 ViT 使用 FP8；本轮 ViT 权重与计算为 BF16。测试开始与结束时，记录的 cache、engine 和两份 Qwen3.5 计算源码哈希均一致。

2. **当前配置与输入**

以下是本次测试设置到 VitConfig 的配置：

```yaml
vit_concurrency: 64
vit_max_queue_size: 64
gpu_max_batch_size: 64
gpu_max_batch_images: 256
gpu_batch_wait_ms: 10
mm_timeout_ms: 600000
use_local_preprocess: true
use_igraph_cache: false
mm_cache_item_num: 0
url_cache_item_num: 0
mm_cache_gpu_max_bytes: 0
mm_cache_cpu_max_bytes: 0
mm_hash_key_cache_max_bytes: 0
```

请求并发分别为 1、2、4、128。vit_concurrency=64 控制异步线程池；本次同步 RPC 入口使用生产配置的 200 个工作线程，不能将它理解为本次 RPC 的 64 并发硬上限。gpu_max_batch_size=64 是调度器组批上限，gpu_max_batch_images=256 是工作量预算；本视频的预算允许实际组成 64 个媒体的 batch。

视频使用 NVDEC 解码，输入为此前提供的“第1条请求.mp4”：

[/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-request1-nvdec.JBWUyk/request1.mp4](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-request1-nvdec.JBWUyk/request1.mp4)

```json
{
  "fps": 6,
  "min_pixels": 2500000,
  "max_pixels": 73728000,
  "max_frames": 180
}
```

未指定 width、height、min_frames，测试请求中均使用 -1。实际预处理结果：

| 项目 | 数量 / 形状 |
|---|---|
| 采样帧数 | 46 |
| 调整后的帧尺寸 | 704 × 1280 |
| grid_thw | [23, 44, 80] |
| 输入 patch 数 | 23 × 44 × 80 = 80,960 |
| 输出视觉 token 数 | 80,960 / 4 = 20,240 |
| embedding | BF16，[20,240, 4,096]，165,806,080 字节 |
| position | [20,240, 3] |

3. **修复的问题与证据**

| 问题 | 触发与表现 | 修复方式 | 验证证据 |
|---|---|---|---|
| RoPE 索引溢出 | 大 batch 的 packed QKV stride / 输出偏移超过 int32 范围，触发 CUDA illegal memory access | 在 Triton 中先将 program_id 转为 int64，再进行乘法，避免乘完才转换 | 原版大 stride 边界复现失败；修复版返回预期的 6 个数值，逐元素一致 |
| LayerNorm 大张量数值错误 | 实际 batch 64 产生 [5,181,440, 1,152] 隐状态；逐层定位到第一个 block 的 norm1 已出现明显错误 | 增加 Qwen3_5MoeVisionLayerNorm，按完整归一化行分块，每次调用不超过 2^31−1 个元素；应用于 block 的 norm1/norm2 及 patch merger norm | GPU 0 常量输入边界测试中，原生 LayerNorm 末尾输出 0，预期为 3；分块后检查位置均为 3；完整 ViT 的大幅误差消失 |
| Patch Conv3d 大 batch 数值差异 | LayerNorm 修复后，差异仍从 patch embedding 开始，初始最大差异 0.0009765625，最终 embedding 最大差异 0.875 | 按单个 patch 的输入与输出元素数，限制每次独立 Conv3d 投影的规模；预分配输出后分块写回 | 每块 1,398,101 个 patch 的验证中，64 个视频的完整 patch embedding 都与单视频结果逐元素一致；接入完整 ViT 后全部输出校验通过 |

LayerNorm 的最小边界复现使用 3,730,318 × 1,152 = 4,297,326,336 个元素。输入全 1、weight=1、bias=3，预期每个输出都是 3；它与逐层诊断中首尾媒体受影响的分布共同支持 flattened 32-bit 索引回绕的判断。修复采用保守的 2^31−1 元素上限。

卷积差异已通过输入完全一致的逐层对照定位到 Conv3d；本轮未进一步确定 cuDNN 内部具体算法切换或舍入机制，不将这一点写成已经确认的底层根因。

分块仅作用于 LayerNorm 和相互独立的 patch 投影；attention 仍处理调度器组成的完整 batch。保留原参数及 state_dict 名称，本轮修复范围为两份 Qwen3.5 源码文件。

- [vision_kernels.py：RoPE int64 索引](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/vision_kernels.py:91)
- [qwen3_5_moe_vit.py：patch 投影](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_moe_vit.py:143)
- [qwen3_5_moe_vit.py：LayerNorm](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_moe_vit.py:190)

4. **测试口径与正确性**

通过生产权重加载器、同步 RemoteMultimodalEmbedding handler、MMProcessEngine 和 MMScheduler 执行。计时从服务端 handler 入口开始，直到输出 embedding 的生产 CUDA event 完成。包括媒体读取、NVDEC 解码、预处理、排队、组批、ViT、projector、结果组装和生产 feature hash。

测试专用输出 transport 只返回空确认，因此不执行 embedding D2H、输出序列化和传输；不包含客户端网络往返、真实 RDMA、语言模型 prefill/decode。它测量的是请求到 embedding 就绪的服务端路径，不是单独的 ViT forward 时间，也不是完整 PD 推理。

先执行 3 条单请求预热，再对并发 1、2、4、128 共 135 条请求全量对比 embedding 和 position。全部 finite，max_abs_diff=0，逐元素一致。128 并发校验实际组成 4、64、60 三个 batch。

正式性能阶段每档执行 3 轮闭环请求，每轮至少持续 10 秒且至少发送 2×并发数条请求，之后排空队列。统计总计 1,147 条请求，错误数为 0；每条检查 shape、dtype 以及与基准一致的生产 per-token feature hash。正式性能阶段未重复执行额外的全量 embedding 对比。

最终 cache miss=1,285（3 条预热 + 135 条校验 + 1,147 条正式请求），hit=0、inflight_dedup=0、resident_entries=0；实际计算媒体数与请求数相等。

5. **当前配置的性能数据**

平均 RT、P50、P99 从三轮所有正式请求合并计算；QPS=三轮总请求数 / 各轮服务端首请求开始到末请求 embedding 就绪的跨度之和。视觉 token/s 使用固定输入每请求 20,240 个输出视觉 token 换算。

| 请求并发 | 正式样本数 | 平均 RT（ms） | P50（ms） | P99（ms） | QPS | 视觉 token/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 114 | 265.53 | 261.02 | 304.34 | 3.7493 | 75,887 |
| 2 | 129 | 472.36 | 475.44 | 510.90 | 4.1733 | 84,469 |
| 4 | 136 | 917.62 | 930.24 | 971.29 | 4.2682 | 86,389 |
| 128 | 768 | 30118.80 | 31656.82 | 47618.63 | 3.6947 | 74,782 |

128 请求并发的三轮原始结果：

| 轮次 | 请求数 | 平均 RT（s） | P99（s） | QPS | 实际 batch 分布 |
|---:|---:|---:|---:|---:|---|
| 1 | 256 | 34.149 | 47.639 | 3.3202 | 8×1次，64×3次，56×1次 |
| 2 | 256 | 27.016 | 34.507 | 4.0416 | 12×1次，64×3次，52×1次 |
| 3 | 256 | 29.191 | 35.926 | 3.7972 | 8×1次，64×3次，56×1次 |

128 并发正式阶段共计算 768 个媒体，实际 batch 64 共执行 9 次。PyTorch 活跃 tensor 分配峰值为 167.735 GiB，allocator reserved 峰值为 240.137 GiB。reserved 以及 NVIDIA 整卡显存读数包含大 batch 预热后保留的 allocator 缓存，不能用来推断低并发的独立显存需求。

这是共享主机上的诊断性测试。128 并发三轮 QPS 为 3.3202～4.0416，存在轮间波动。NVML 部分区间采样超时，未把采样缺失当作 0，也未将 GPU 利用率作为完整覆盖的汇总指标。

6. **QPS 与 TPS 的换算**

```text
输出视觉 token 吞吐 = 3.6947394868 请求/s × 20,240 token/请求
                   = 74,781.53 token/s
                   ≈ 7.48 万视觉 token/s
```

因此，“3.69 QPS 大约等于 7.5 万 TPS”在本条固定视频、每请求输出 20,240 个视觉 token、上述计时边界下成立。这里的 TPS 指输出视觉 embedding 的 token 行数，不是输入 patch/s，也不是 LLM 生成文本 token/s。对不同视频，应该用实测总输出视觉 token 数除以服务端总时间跨度，不能直接套用 20,240。

7. **MFU 估算**

按用户确认的 L20D FP16 峰值 2,250 TFLOPS，并采用相同的 BF16 dense 峰值，本轮 128 请求并发的服务端口径 MFU 估算为 **17.07%**。NVIDIA 的 B300 非稀疏 BF16 峰值表也给出 2,250 TFLOPS：[NVIDIA dgxc-benchmarking](https://github.com/NVIDIA/dgxc-benchmarking#peak-theoretical-throughput)。

计算量从实际 ViT 结构解析计算，按一次乘加计 2 FLOPs。每请求 N=80,960 个输入 patch token，hidden_size D=1,152，intermediate_size I=4,304，共 L=27 层。Attention 按 23 个长度 S=3,520 的独立非因果段计算；输出 merger 在 N/4=20,240 个 token 上运行。

| 组成 | 每请求计算公式 | TFLOPs/请求 |
|---|---|---:|
| Patch Conv3d | 2 × N × (3×2×16×16) × D | 0.2865 |
| 27 层 QKV 与 output projection | L × 8ND² | 23.2075 |
| 27 层 attention QKᵀ 与 AV | L × 4×23×S²×D | 35.4560 |
| 27 层 MLP | L × 4NDI | 43.3530 |
| 输出 merger 两层 linear | 2 × 20,240 × (4,608² + 4,608×4,096) | 1.6236 |
| 合计 |  | 103.9266 |

有效计算吞吐 = 103.9266 TFLOPs/请求 × 3.6947394868 请求/s = **383.9817 TFLOPS**。

MFU = 383.9817 / 2,250 = **17.0659%**。

这是主要矩阵乘法及卷积的解析估算，未计 LayerNorm、GELU、softmax、RoPE 等非矩阵操作的 FLOPs；总计算时间仍采用完整的既定服务端测量跨度，包含读取、解码、预处理、排队和组装。此值不是 NCU 测得的 Tensor Core 利用率，也不是仅 GPU forward 区间的 MFU。

7.48 万视觉 token/s 使用的是 merger 后的输出 token 数；计算 MFU 时，27 层 ViT 主体运行在 merger 前的 80,960 个 patch token 上，同时需计入分段 attention 的二次项。

8. **MFU 偏低的分段诊断**

在 GPU 0 补充了一次生产 MMProcessEngine 路径的分段计时：先预热两次单视频，再测 1 个和 64 个视频。两次输出均与基准逐元素一致。使用 CPU 单调时钟与 CUDA event，未修改生产源码；这次在一个 engine 请求中放入 64 个媒体，用于阶段归因，不替代前面的 128 独立 RPC 并发性能结果。它不是新的三轮稳定态 benchmark，也不是 NCU kernel 计数结果。

| 阶段 | 本次观测 | 计时含义 |
|---|---:|---|
| 64 个媒体的完整 engine 调用 | 21.149 s | host wall time |
| 解码与预处理合计 | 7.492 s | 64 次 preprocess_video_cuda 的 host 调用总和；内部包含 NVDEC |
| 其中 NVDEC | 5.242 s | 64 次解码的 host 调用总和 |
| ViT visual_total | 12.115 s | CUDA event 区间；包含下列嵌套阶段及间隙 |
| 其中 QKV / output projection / MLP 两层 linear | 2.815 s | 27 层对应 CUDA event 区间之和 |
| 其中 FA4 | 3.765 s | 27 层 attention CUDA event 区间之和 |
| 其中 norm1/norm2、GELU、RoPE | 3.291 s | 相应 CUDA event 区间之和 |
| 其中 patch embedding + merger | 0.911 s | CUDA event 区间 |
| ViT 区间内未由上述子区间覆盖的剩余部分 | 1.333 s | 包括位置处理、残差、未单独包裹的操作及间隙，尚未做 kernel 级拆分 |

各行有父子包含关系，host 与 CUDA 也可能重叠，不能把整张表相加。CUDA event 记录的是流上区间，可能包含内部拷贝和提交间隙，不应当等同于 NCU 的 Tensor Core 活跃时间。

本次证据支持三个判断：

- 视频解码/预处理是显著开销。当前 batched_embedding 先用 Python 列表循环，对每个 GpuVideoInput 执行完整 preprocess_video_cuda，再拼接 pixels、运行 ViT。64 个媒体在这段路径逐个处理，调整 vit_concurrency 不会自动让该循环并行。
- 大 batch 有显著非矩阵工作。Norm/GELU/RoPE 合计约 3.29 s；RoPE 每层显式生成两份完整 FP32 Q/K 中间结果，再转换 BF16。batch 64 时这两份中间结果合计约 44.48 GiB，造成额外物化和内存读写。它们的耗时进入服务端时间，但不贡献与 dense Tensor Core 峰值相对应的大量矩阵 FLOPs。
- 矩阵 linear 本身的效率明显高于 17%。用相同解析 FLOPs / 对应 CUDA event 区间估算，QKV、output projection 与 MLP linear 合计约 1513 TFLOPS（67.3% 峰值）；FA4 区间约 603 TFLOPS（26.8%）；整个 visual 区间约 24.4%。因此 17.1% 的服务端 MFU 不能解释成“矩阵乘法只跑到 17%”。

Feature hash 的 host 区间合计约 1.505 s，但其 CUDA event 区间合计仅 15.62 ms。实现会同步当前 CUDA stream，因此 host 等待包含此前尚未完成的工作；不能把约 1.5 s 全部归咎于 hash kernel。本轮沿用的 native 库包含 CUDA invokeFeatureHash，未将此路径判断为 CPU 全 embedding hash。

优先优化方向是验证解码/预处理与 ViT 的流水重叠、降低大张量物化及分配开销，再针对 FA4 和 patch Conv3d 的实际形状做 kernel 级 profile。现有数据还不能把 FA4 的差距具体归因于寄存器、访存或 head_dim padding；这些需要进一步 Nsys/NCU 证据。

分段原始记录：[stages.json](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit397b-stage64-gpu0-ayuhb4ct/stages.json)，[计算明细](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit397b-stage64-gpu0-ayuhb4ct/analysis.json)，[诊断脚本](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit397b-stage64-gpu0-ayuhb4ct/stages.py)。

9. **原始数据与复现材料**

- [本轮原始统计 report.json](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit397b-concurrency64-batch64-fixed-gpu0-jviwkcwi/report.json)
- [汇总及源码哈希核对 summary.json](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit397b-concurrency64-batch64-fixed-gpu0-jviwkcwi/summary.json)
- [配置与基准 manifest.json](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit397b-concurrency64-batch64-fixed-gpu0-jviwkcwi/manifest.json)
- [完整日志 service.log](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit397b-concurrency64-batch64-fixed-gpu0-jviwkcwi/service.log)
- [每条请求 all-requests.json](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit397b-concurrency64-batch64-fixed-gpu0-jviwkcwi/all-requests.json)
- [实际 forward / 组批 forwards.json](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit397b-concurrency64-batch64-fixed-gpu0-jviwkcwi/forwards.json)
- [性能脚本 bench.py](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit397b-concurrency64-batch64-fixed-gpu0-jviwkcwi/bench.py)
- [测试时源码差异 source.diff](/home/xieshui.yyx/workspace/RTP-LLM/.t/vit397b-concurrency64-batch64-fixed-gpu0-jviwkcwi/source.diff)
- [RoPE 边界复现](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-rope-index-64od7lsf/boundary.py)
- [GPU 0 LayerNorm 边界结果](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-layernorm-boundary-ly5fun35/boundary.log)
- [GPU 0 修复 LayerNorm 后逐层诊断](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-batch64-normfix-gpu0-kx7gfy29/diagnostic.json)
- [GPU 0 patch 卷积完整对照结果](/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-patch-conv-boundary-gpu0-jlvfbagc/conv.log)

本轮沿用已有 native runtime，验证当前 Python/Triton ViT 计算；排除的 C++/protobuf 输出传输路径不在本次验证范围内。
