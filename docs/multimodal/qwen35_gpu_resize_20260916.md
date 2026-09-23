# Qwen3.5 GPU resize 验证（2026-09-16）

已去掉 Qwen3.5 resize 前后的 CPU 往返拷贝，复用现有 torchvision CUDA resize；保留 bicubic、antialias、uint8 输出和原有帧采样/尺寸规则。CPU 和 GPU 的运算及舍入不同，结果不逐位等价。

## 真实 32 视频 batch

范围：实际 MMProcessEngine 请求入口，包括文件读取、元数据、NVDEC、resize、processor、真实 397B ViT、时间戳 embedding 拼接及 hash，到 CUDA 输出完成。不含 gRPC 序列化、网络、RDMA 或 LLM 推理。一个请求含 32 份原视频，实际 batch=32；不是 32 个独立请求的并发测试。

GPU0 开始前空闲，运行时 GPU 进程只有本实验。两条路径预热后交替各测 3 次，输出校验在计时外，没有在每次请求前 empty_cache。

| 路径 | 三轮耗时（s） | 平均 RT（s） | 中位 RT（s） | 视频/s（按中位 RT） |
|---|---|---:|---:|---:|
| CPU resize | 41.877, 41.733, 41.708 | 41.772 | 41.733 | 0.7668 |
| GPU resize | 22.802, 22.898, 22.786 | 22.829 | 22.802 | 1.4034 |

中位 RT 降低 45.36%，吞吐提升 83.03%。两条路径 Torch peak allocated 均为 238.038 GiB；全程 GPU 采样峰值 273931 MiB（267.511 GiB，包含缓存与 CUDA/NVDEC 开销）。

冷启动记录单列：CPU 首次预热 144.222 s，随后 GPU 预热 24.071 s。分配状态不同，不用这两次做加速比。此前 workers32 的 58.248 s 单次记录在请求前清空 allocator cache，也不直接作为本表基线。

## 环境与配置

- 仓库：/home/xieshui.yyx/workspace/RTP-LLM/github-opensource；分支 feat/qwen35_vl_omega；HEAD 2679cfed1b2ae4b3d315c34577e9feb31ef6017f，另加本轮未提交的 GPU resize 和此前 NVDEC pool 修改。
- GPU0：NVIDIA L20D / SM103；UUID GPU-67f05232-d6d5-7210-1143-8cb1c0a9eed0。
- 运行时：{"python": "3.10.9", "torch": "2.11.0+cu130", "cuda": "13.0", "transformers": "5.2.0", "torchvision": "0.26.0+cu130", "device": "NVIDIA L20D"}。
- 权重：/mnt/nas1/hf/Qwen3.5-397B-A17B-FP8；只加载真实 BF16 ViT 与所需文字 embedding，SM103 FA4 attention。
- 视频：/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-request1-nvdec.JBWUyk/request1.mp4。
- fps=6，min_pixels=2500000，max_pixels=73728000，max_frames=180；233 原始帧，30 FPS，720×1280；选择 46 帧，resize 为 1216×2112。
- grid=[23,76,132]；输入 230736 patches/视频；输出 [57868,4096] BF16/视频，包含时间戳/标记。
- NVDEC workers=32；vit_concurrency=64；vit_max_queue_size=64；gpu_max_batch_size=32；gpu_max_batch_images=256；gpu_batch_wait_ms=10；timeout=600000 ms。
- URL/GPU/CPU/hash cache 均为 0；use_igraph_cache=False；PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。
- 仅本轮测试显式设 workers=32，没有顺带改变生产默认 worker 数。当前尺寸不同于早先 704×1280 的实验，不混用其吞吐/MFU。

## 单视频模块诊断

以下分段在其他任务仍保留显存时运行，仅作为诊断；各预热后 5 次取中位。resize 输入已就绪，不含解码；预处理包含 NVDEC 到 processor 输出完成。

| 范围 | CPU resize 路径 | GPU resize 路径 |
|---|---:|---:|
| resize 本身 | 452.660 ms | 9.944 ms |
| NVDEC + resize + processor | 630.972 ms | 95.896 ms |
| 单视频 MM 模块请求（3 次，中位） | 1357.089 ms | 735.582 ms |

旧 CPU resize 在 1/4/8/16 intra-op threads 下约为 460/460/470/468 ms，进程 CPU 时间接近墙钟时间。NVDEC pool 没有并行这段逐视频 resize；本次去掉了这段 CPU 串行处理及帧数据往返。

## 正确性与限制

- 两组 Qwen3.5 视频/批处理测试共 24 项通过，覆盖 GPU resize 放大/缩小且禁止 Tensor.cpu、帧/patch 顺序、奇数帧 padding、混合媒体映射、NVDEC pool 异常恢复及 SM103 FA4。
- 单视频和 batch32 输出全部有限；每条路径 3 次重跑逐元素一致。batch32 的所有媒体均检查了形状、位置、有限值、重复一致性。
- resize 后 354410496 个 uint8 通道值中 17.3355% 不同；绝大多数差 1，83003 个（0.02342%）差值大于 1。MAE=0.17366，最大绝对差=12。
- 最终 embedding（单视频，以及 batch32 第一个完整视频）余弦相似度=0.99944176，相对 L2 差异=3.34145%，MAE=0.00223873，最大绝对差=3.3125；全部位置编码与 CPU 路径一致。
- 本次验证执行、有限值、形状和稳定性，不证明与 CPU 数值等价；尚未验证最终 LLM 文本/视频理解质量，没有运行完整 LLM smoke。

## 时间线

在 batch32 退出后单独采集一个预热后的单视频 Nsight Systems 区间：没有 H2D/D2H memcpy 事件。旧区间存在 127180800 字节 D2H 和 354410496 字节 H2D。GPU resize 内核确实执行：

- NVDEC_RGB：host enqueue 79.047 ms，CUDA kernels 求和 0.583 ms。
- GPU_RESIZE：host enqueue 0.558 ms，CUDA kernels 求和 9.894 ms。
- VIDEO_PROCESSOR：host enqueue 0.747 ms，CUDA kernels 求和 4.191 ms。

Trace 仅作归因，不使用其耗时作为性能表基准。host enqueue、CUDA event 区间、内核求和不能直接相加。未采集 NCU 硬件计数器，不进一步断言 resize 是 SM-bound/HBM-bound。

## 代码与原始记录

- [生产代码](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/gpu_video.py:353)
- [测试](/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/test/qwen35_gpu_video_test.py:351)
- 实验目录：/home/xieshui.yyx/workspace/RTP-LLM/.t/gpu-resize-20260916-eysp4eha
- batch32.json / batch32.log / batch32-gpu.csv：三轮正式结果、日志与 GPU 采样。
- experiment.json：单视频诊断和真实 ViT 数值比较。
- preprocess.nsys-rep / preprocess.sqlite / trace-summary.json：新时间线。
- unit.log：24 项测试；source.patch / before/ / manifest.json：本轮 diff、修改前快照和源码哈希。
- experiment.*.attempt1：统计脚本参数命名冲突的早期失败记录，未计入结果。
- 旧 CPU trace：/home/xieshui.yyx/workspace/RTP-LLM/.t/vit32-bottleneck-20260916-vfy17wd6/preprocess.nsys-rep

复现完整 batch32：GPU0 需空闲，本轮峰值约 267.5 GiB 显存，接近整卡容量。

```bash
MM_TEST_GPU=0 MM_NATIVE_LIB_DIR=/home/xieshui.yyx/workspace/RTP-LLM/.t/mm-migration-20260915.j048qksg/bazel-root/cf2c4cb7d73003e2fc6eda3fe2c305e2/execroot/rtp_llm/bazel-out/k8-opt/bin/rtp_llm/libs PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /opt/conda310/bin/python -B /home/xieshui.yyx/workspace/RTP-LLM/.t/gpu-resize-20260916-eysp4eha/launch.py --exec /home/xieshui.yyx/workspace/RTP-LLM/.t/gpu-resize-20260916-eysp4eha/batch32_entry
```

只修改 RTP-LLM 目录中的代码和实验记录，保留原有工作区改动；未提交、未推送。
