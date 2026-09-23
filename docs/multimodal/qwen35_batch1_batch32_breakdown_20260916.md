# Qwen3.5 batch=1 / batch=32 耗时归因（2026-09-16）

主要耗时在 ViT，尤其是 FA4 attention。batch32 的一次暖态采样共 23.012 s，ViT 占 21.027 s（91.4%），FA4 核心占 11.293 s（整请求 49.1%）。单视频自身已有充足的 GPU block；FA4 的寄存器和 shared memory 又把同一 SM 的驻留上限限制为 1 个 block，因此增加 stream 不能把同类 block 叠加驻留。这不是“显存带宽打满”，也不等于 Tensor Core 算力已完全用满。

## 正式耗时：无 profiler、无分段计时插桩

范围为实际 MMProcessEngine 请求入口，包含本地文件、元数据、NVDEC、GPU 预处理、真实 397B ViT、时间戳/位置拼装和 hash，到 CUDA 输出完成。应用缓存全关。排除 gRPC 序列化、网络、RDMA 和 LLM。每个请求含 1 或 32 份相同原视频；不是 32 个独立请求。

| batch | 三轮耗时（s） | 平均 RT（s） | 中位 RT（s） | 视频/s | Torch 峰值（GiB） |
|---|---|---:|---:|---:|---:|
| 1 | 0.755, 0.737, 0.732 | 0.741 | 0.737 | 1.3570 | 8.433 |
| 32 | 22.802, 22.803, 22.905 | 22.837 | 22.803 | 1.4033 | 238.038 |

batch32 的中位 RT 是 batch1 的 30.94 倍；每视频摊销 0.713 s，相对单视频 0.737 s 的吞吐收益仅 3.41%。这种摊销时间不是独立请求的响应延迟。

预热要求连续两次相邻耗时变化小于 5%，达到后才测三轮。无插桩 batch32 前两轮预热为 75.930 / 66.846 s，随后 23.136 / 22.830 / 22.768 s。正式三轮已稳定在约 22.8 s，冷分配耗时没有混入平均值。采样前后 GPU0 无其他可见 PID，外部显存差值 22–67 MiB。

## 分阶段：另一次暖态 profiling

下面对应带 NVTX 和预创建 CUDA events 的独立采样，总耗时 batch1=0.750 s、batch32=23.012 s。用于归因，不替代上面的正式 RT。两个阶段使用 CUDA 区间，CPU 准备使用 host 区间；最后一行是扣除前三项的剩余临界路径，包含实际调度、拼接、组装/hash 及小间隙。

| 阶段 | batch=1（s） | batch=32（s） | batch32 占采样请求 |
|---|---:|---:|---:|
| 读取和 CPU 元数据准备 | 0.024 | 0.710 | 3.09% |
| NVDEC + GPU resize/processor（包含等待） | 0.097 | 1.195 | 5.19% |
| ViT | 0.614 | 21.027 | 91.37% |
| 调度、concat、token 拼装、hash 和剩余间隙 | 0.016 | 0.080 | 0.35% |

文件读取本身只有约 1.21 ms / 33.34 ms（batch1 / 32，总 host 时间）；CPU 准备主要是视频元数据/采样规则等，不应把 0.710 s 全部称为磁盘 I/O。

NVDEC 的 32 个 worker host 时间求和约 19.06 s，但这些调用重叠。整个 NVDEC + GPU 预处理临界区只有 1.195 s，不能把 worker 时间相加当请求耗时。batch32 resize CUDA 区间求和约 320 ms，processor 约 155 ms。

## ViT 内部

| ViT 子模块（CUDA 区间） | batch=1（s） | batch=32（s） |
|---|---:|---:|
| Attention 总计（包含 QKV、RoPE、FA4、输出投影） | 0.412 | 13.756 |
| 其中：FA4 核心 | 0.339 | 11.293 |
| 其中：QKV + 输出投影 | 0.041 | 1.404 |
| 其中：RoPE | 0.031 | 1.058 |
| MLP（FC1 + GELU + FC2） | 0.107 | 3.804 |
| 两次 LayerNorm，含分块 copy | 0.038 | 1.670 |
| Patch embedding | 0.036 | 1.161 |
| 位置/旋转等 ViT 元数据 | 0.003 | 0.088 |
| Merger | 0.004 | 0.138 |

Attention 的“其中”行与 Attention 总计重复，不能相加。27 层总 CUDA 区间约 0.570 / 19.633 s。Nsys 相关联的每层平均 GPU span 为 21.111 / 727.131 ms，27 层持续重复这份工作；不是某一层偶发慢。

batch32 LayerNorm 为避免 32 位索引溢出走分块路径，额外 copy 共约 918.6 GB（逻辑字节）、286.36 ms。它会放大大 batch 的开销，但只占请求约 1.24%，不是 11.29 s attention 的主因。Patch embedding 也分块，但其总体时间基本按工作量增长。

Nsys 显示 batch32 GPU kernel + memcpy 的活动 union 为 21.556 s（请求约 93.7%）；27 层内部合计真正空隙仅约 8.3 ms。GPU 活动时间不等于 Tensor Core 利用率，不能据此认定算力满载。

## 为什么多 stream 无明显收益

真实 FA4 启动配置：

- batch1：14,784 blocks；batch32：473,200 blocks；设备共有 148 个 SM。
- 单视频已经约 99.89 个 block waves/SM。batch32 增加的是要执行的轮数，GPU 的 SM 数量没有增加。
- 每 block 512 threads、每线程 128 个寄存器，总计 65,536 个寄存器，等于该 SM 的寄存器容量。
- 动态 shared memory=228,352 B（223 KiB），加 driver 占用后每 block 分配 229,376 B（224 KiB）；该 SM 只有 233,472 B（228 KiB）。
- NCU 的寄存器和 shared memory 两项都明确给出 occupancy_limit=1 block/SM。另一个 stream 的同类 block 无法与它同时驻留。

因此，即使 Tensor Core 指标并不高，也不能把其余空间简单理解为“再塞一个请求就能利用”。需要改变内核内部的 tile、寄存器/shared memory 布局和流水效率，释放可驻留资源，而不是仅增加线程或 stream。

## 硬件计数器：匹配形状/stride 的探针

从真实 trace 读取 Q/K/V 的 shape、stride、segment 数和最大长度，在同一 GPU 上对 FA4 和 FC1 做匹配形状的 NCU 探针。真实权重/数据的耗时以上面的 Nsys 为准；这里的输入为合成值，不把 probe 耗时当请求结果。探针核名、grid、block 与真实 trace 匹配。

| 匹配形状探针 | SM 吞吐指标 | DRAM 吞吐指标 | BF16 Tensor 指标 | 实际 occupancy | blocks/SM 上限（寄存器 / shared） |
|---|---:|---:|---:|---:|---|
| FA4_B1 | 51.90% | 2.53% | 24.10% | 21.67% | 1 / 1 |
| FA4_B32 | 49.56% | 2.44% | 23.01% | 21.76% | 1 / 1 |
| FC1_B1 | 96.97% | 32.59% | 48.26% | 9.37% | 1 / 1 |
| FC1_B32 | 97.72% | 34.00% | 48.59% | 9.37% | 1 / 1 |

FA4 的 DRAM 只有约 2.4–2.5%，排除了“这份 attention 核把 HBM 带宽用满”的解释。SM 指标约 50%、低驻留率说明仍有内核效率优化空间；这里没有采集 stall 明细，不能进一步指定为某一种 SFU、barrier 或依赖停顿。FC1 的 SM 吞吐指标约 97%，增加并发也不会增加该内核可用的同卡执行资源。

BF16 Tensor 一列使用 Nsight 自身的归一化定义，不能直接当作整个模型的 MFU。NCU 未强制改时钟或清空缓存，计数器结果用于内核分类和驻留限制，避免将合成探针的绝对耗时与真实前向混为一谈。

## 时间线定位

以下时间以 Nsys SQLite session 的纳秒时钟为基准换算成 ms；保留完整原始数值于 trace-summary.json，可在对应 nsys-rep 中定位。每层的 gap 已排除 kernel 和 memcpy，避免把分块 copy 误认为 GPU 空闲。

| 采样范围 | Nsys 起始（ms） | 结束（ms） | 说明 |
|---|---:|---:|---|
| MM_REQUEST_B1 | 140.515 | 890.581 | GPU 活动 union 0.627 s |
| MM_REQUEST_B1 / L00 | 313.583 | 332.499 | GPU span 18.916 ms，真正空隙 0.087 ms |
| MM_REQUEST_B1 / L13 | 582.851 | 604.035 | GPU span 21.183 ms，真正空隙 0.087 ms |
| MM_REQUEST_B1 / L26 | 862.368 | 883.871 | GPU span 21.503 ms，真正空隙 0.089 ms |
| MM_REQUEST_B32 | 1076.863 | 24089.288 | GPU 活动 union 21.556 s |
| MM_REQUEST_B32 / L00 | 4259.093 | 4963.424 | GPU span 704.331 ms，真正空隙 0.305 ms |
| MM_REQUEST_B32 / L13 | 13729.201 | 14477.268 | GPU span 748.067 ms，真正空隙 0.316 ms |
| MM_REQUEST_B32 / L26 | 23159.843 | 23891.921 | GPU span 732.078 ms，真正空隙 0.310 ms |

## 正确性、配置和边界

- batch1 / batch32 全量有限值、输出数量和形状通过；所有视频的固定抽样值与 batch1 一致，profiling 与无插桩运行的抽样值也一致。没有将抽样检查声称为全量逐位比较。
- 每视频输出 [57868,4096] BF16、位置 [57868,3]；46 帧，1216×2112，grid=[23,76,132]，230736 输入 patches；每帧组 attention 长度 10032，单视频 23 段，batch32 736 段。视频之间没有做跨视频 attention。
- 原请求参数：fps=6，min_pixels=2500000，max_pixels=73728000，max_frames=180。
- 模型 /mnt/nas1/hf/Qwen3.5-397B-A17B-FP8，真实 BF16 ViT（27 层，hidden=1152，16 heads，head_dim=72，MLP=4304，输出=4096），FA4。
- NVDEC workers=32；vit_concurrency=64；gpu_max_batch_size=32；gpu_max_batch_images=256；gpu_batch_wait_ms=10；URL/GPU/CPU/hash cache=0；CUDA Graph 关闭；allocator expandable_segments=True。
- GPU0 L20D / SM103，UUID GPU-67f05232-d6d5-7210-1143-8cb1c0a9eed0；torch 2.11.0+cu130，CUDA13，transformers 5.2.0。
- 仓库 /home/xieshui.yyx/workspace/RTP-LLM/github-opensource，feat/qwen35_vl_omega，HEAD 2679cfed1b2ae4b3d315c34577e9feb31ef6017f；本轮没有改生产代码，已有修改的 SHA256 校验未变。
- 这是多模态模块诊断。按请求范围未运行 LLM、真实 RDMA/gRPC 传输或完整服务质量测试；不能称作整个 RTP-LLM 请求性能。

## 代码与证据

- [合批及视频预处理](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_moe_mixin.py:247)
- [FA4 调用](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_moe_vit.py:390)
- [大 batch LayerNorm 分块](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_moe_vit.py:190)

实验目录：/home/xieshui.yyx/workspace/RTP-LLM/.t/vit-b1-b32-profile-20260916-692s7gte

benchmark-result.json / benchmark.log：无插桩三轮结果与预热。
trace-result.json：host/CUDA 分段；stages.nsys-rep / stages.sqlite / trace-summary.json：完整时间线、层和算子归属。
counters.ncu-rep / counters.csv / ncu-summary.json / ncu-probe.json：计数器及探针输入。
diagnosis.json：本报告的机器可读数据。manifest.json：源码哈希。
experiment.py / ncu_probe.py / parse_trace.py：复现脚本；trace-command.json / ncu-command.json：精确命令。

本轮所有输出位于允许的 RTP-LLM 目录下；未提交、未推送。优先优化对象是 FA4 核心（11.293 s），其次是 MLP/GELU 和归一化/旋转相关读写；CPU 元数据准备整个阶段最多约 0.710 s，继续单独优化加载不能带来数量级改善。
