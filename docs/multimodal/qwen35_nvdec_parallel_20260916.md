# Qwen3.5 视频并发解码：实现与验证

修改位置：/home/xieshui.yyx/workspace/RTP-LLM/github-opensource。分支 feat/qwen35_vl_omega，基于 2679cfed1b2ae4b3d315c34577e9feb31ef6017f，本次未提交。

## 实现

原实现由 mm_scheduler 组成 batch 后，在同一线程内逐个完成视频解码和预处理。现在在 Qwen3.5 视频路径增加有界 NVDEC 预取：多个独立线程各自持有 decoder 和 CUDA stream，完成 RGB 后按输入顺序交回原预处理及 ViT batch。

- 并发范围是同一个 scheduler batch。预取窗口最多为配置的解码路数，不把整个大 batch 的 RGB 全部提前展开。
- RGB 完成后通过 record_stream 保护消费端的张量生命周期；异常或提前结束时取消排队任务、等待在执行任务结束。
- 每个视频仍新建 demuxer、解码到 EOS；只复用硬件 decoder session，不缓存视频帧、embedding，不合并重复视频任务。
- 保留原来的 CPU uint8 resize、processor 和输出装配语义。单视频 batch 保持直接解码；图像和已经处理好的张量沿原路径处理。
- 现有 mm_scheduler 负责 ViT 凑批，本次解码线程池不改变 vit_concurrency 或 gpu_max_batch_size。

## 配置

QWEN35_VIDEO_DECODE_WORKERS 默认值为 2。设为 1 使用串行路径；设为 4 可启用 4 路。需要在新启动的服务进程中设置。

选择默认 2 路的依据：完整 engine 路径中，2 路和 4 路性能接近，2 路占用更少的解码会话和预取帧。

## 正确性

- 24 项 GPU 视频与 Qwen3.5 视频 batch 单元测试通过，包括 NVDEC session 复用/异常丢弃、顺序保持、有界预取、异常排空与恢复、已有帧不重复解码、混合图像/视频组批和 FA4。
- 原视频与倒序抽帧变体在 1/2/4 路下的完整预处理像素逐元素完全一致，且两个变体内容不同，检查了输入顺序和串流隔离。
- 加载真实 Qwen3.5-397B-A17B-FP8 的 BF16 ViT 与 CPU 文本词嵌入。2 视频 packed forward 在 1/2/4 路下输出 embedding 和位置编码逐元素完全一致。
- 现有 MMProcessEngine 的 4 视频请求在 1/2/4 路预热验证共 6 次，全部一致；每条视频 embedding 形状为 [57868,4096]。
- 验证范围到多模态 embedding；没有运行 LLM prefill/decode 或完整生成 smoke。

## 完整多模态 engine 对比

设备固定物理 GPU0，NVIDIA L20D/SM103；三轮轮换执行顺序，每模式每轮至少 10 秒、至少 2 个请求，取吞吐中位数对应的完整轮次。每个请求含 4 份原视频，实际 ViT batch 均为 4。

计时从 MMProcessEngine.mm_embedding_rpc 入口到所有 GPU embedding 完成，包括文件读取、元数据、调度、NVDEC、CPU resize、processor、ViT、装配、feature hash。排除 gRPC 序列化、网络、输出 embedding D2H 和 LLM。此表是固定多媒体 batch 的模块耗时，不是请求并发压测或整体 RTP-LLM 性能。

视频：request1.mp4，5996056 bytes；fps=6，min_pixels=2500000，max_pixels=73728000，max_frames=180。原片 233 帧、720×1280，抽取 46 帧，resize 1216×2112，grid=[23,76,132]。

URL、GPU/CPU embedding、hash key 缓存均关闭；系统文件页缓存保留。ViT BF16，CUDA Graph 关闭，vit_concurrency=64，gpu_max_batch_size=4，gpu_max_batch_images=256，gpu_batch_wait_ms=10。

| 解码线程数 | 4 视频请求平均耗时 ms | 视频吞吐 /s | 峰值 torch allocated GiB |
|---|---:|---:|---:|
| 1 | 7724.033 | 0.517863 | 31.714 |
| 2 | 7409.063 | 0.539878 | 31.714 |
| 4 | 7443.197 | 0.537403 | 31.714 |

2 路相对串行：请求耗时下降 4.08%，视频吞吐提升 4.25%。4 路未显示额外的完整路径收益。

## 阶段短测（诊断）

以下均为充分预热后 3 次短测的中位数，单次长度短于正式 engine 测量，用于理解阶段变化。解码终点为选中 RGB 已完成；完整预处理包含原来的 resize、归一化、patch 排列和 BF16 转换。

| 解码线程数 | 8 视频解码 ms | 4 视频完整预处理 ms |
|---|---:|---:|
| 1 | 630.065 | 2689.994 |
| 2 | 349.732 | 2443.686 |
| 4 | 206.201 | 2397.976 |

解码阶段能明显并行，但完整路径还包含 resize/processor 和 ViT；不能把解码阶段加速倍数直接套到服务吞吐。

## 原始文件

- /home/xieshui.yyx/workspace/RTP-LLM/.t/nvdec-parallel-20260916-w8umjlda/engine-benchmark.json：正式三轮 engine 数据。
- /home/xieshui.yyx/workspace/RTP-LLM/.t/nvdec-parallel-20260916-w8umjlda/experiment.json：阶段诊断和真实权重正确性。
- /home/xieshui.yyx/workspace/RTP-LLM/.t/nvdec-parallel-20260916-w8umjlda/unit.log：24 项测试。
- /home/xieshui.yyx/workspace/RTP-LLM/.t/nvdec-parallel-20260916-w8umjlda/engine_benchmark.py：可复现 engine 测量脚本。
- /home/xieshui.yyx/workspace/RTP-LLM/.t/nvdec-parallel-20260916-w8umjlda/source.patch：本次源码与测试改动。
- /home/xieshui.yyx/workspace/RTP-LLM/.t/nvdec-parallel-20260916-w8umjlda/before：修改前的三个文件。

全部测试进程已结束，未修改或停止其他服务。

## 32 路 NVDEC 试跑（2026-09-16）

本轮把解码线程上限设为 32，并使用真实的 32 视频 batch，避免 batch 只有 4 个视频时实际只运行少量解码线程。GPU 固定为物理 0；视频、fps/min_pixels/max_pixels/max_frames 与上文一致，grid=[23,76,132]。实际观察到同时执行的解码函数数为 32、线程数为 32。

| 解码线程数 | 32 视频纯解码 ms，三次中位数 | 完整 32 视频请求 s，单轮 | 视频吞吐 /s，单轮 |
|---|---:|---:|---:|
| 2 | 1380.440 | 51.145 | 0.625677 |
| 4 | 804.411 | 56.543 | 0.565937 |
| 32 | 570.669 | 58.248 | 0.549372 |

纯解码边界：已读入内存的压缩视频到选中 46 帧的 RGB 在 GPU 完成；三次稳定调用取中位数，输出与串行 RGB 基线完全相同。另一次独立的线程区间诊断用于验证实际并发，不混入上述计时。

完整请求边界：mm_embedding_rpc 入口到所有 GPU embedding 完成，包含读取、元数据、调度、解码、resize、processor、397B BF16 ViT、装配及 feature hash，排除 gRPC 序列化/网络、输出 embedding D2H 和 LLM。

完整路径每档仅一个计时样本，属于诊断，不能据此判断稳定的性能回退或提升。每档测量前已预热 decoder，模型已做 32 视频 shape 预热；计时前清理 allocator 的空闲缓存。各档均用相同 allocator 配置，实际形成的 ViT batch 均为 32。

32 路纯解码更快，但这轮完整多模态路径没有看到收益；生产代码的默认值继续保持 2，32 只设置在本轮测试进程环境中。

### 显存与 allocator

默认 allocator 下，32 视频 ViT batch 在 2 路解码基线上即发生 OOM：需要分配 59.19 GiB，而可用 52.03 GiB，同时有 44.18 GiB PyTorch reserved-but-unallocated。与 32 路解码器无关。

使用 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 后，2/4/32 路的完整 32 视频 batch 全部跑通。此设置仅用于测试启动环境，没有修改仓库的生产默认配置。

三档完整路径峰值 torch allocated 均为 238.480 GiB；整个重试进程采样到的显卡内存占用峰值为 271163 MiB（264.81 GiB，包含 allocator 保留空间和非 PyTorch 分配）。

纯解码阶段的 torch allocated 峰值：2 路 0.596 GiB, 4 路 0.829 GiB, 32 路 5.196 GiB。该阶段未加载 ViT；与完整路径峰值是不同范围。

### 正确性与记录

2/4/32 路均验证 32 份 embedding 和位置编码逐元素完全一致、embedding 全部有限，单视频 embedding 为 [57868,4096]。本轮没有改动生产源码，源文件哈希与启动时一致。测试与显存监控进程已结束。

- 纯解码及首次 OOM：/home/xieshui.yyx/workspace/RTP-LLM/.t/nvdec32-gpu0-20260916-dvclzuvo/result.json
- 32 路线程区间 Chrome trace：/home/xieshui.yyx/workspace/RTP-LLM/.t/nvdec32-gpu0-20260916-dvclzuvo/decode-overlap-32.chrome.json（CPU 函数区间，终点等待 GPU RGB 完成；不是 NVDEC 硬件 kernel 时间线）。
- 可扩展 allocator 完整结果：/home/xieshui.yyx/workspace/RTP-LLM/.t/nvdec32-expandable-gpu0-20260916-53tpvhrj/result.json
- 显存采样：/home/xieshui.yyx/workspace/RTP-LLM/.t/nvdec32-expandable-gpu0-20260916-53tpvhrj/gpu-samples.jsonl
- 复现脚本：/home/xieshui.yyx/workspace/RTP-LLM/.t/nvdec32-expandable-gpu0-20260916-53tpvhrj/experiment.py


## GPU resize 后续验证

实际 batch32 平均 RT：41.772 s → 22.829 s。详情和数值差异见 [GPU resize 验证](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/docs/multimodal/qwen35_gpu_resize_20260916.md)。
