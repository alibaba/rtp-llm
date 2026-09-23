# Qwen3.5 视频并发实验（2026-09-16）

已完成 GPU0 上的两路、四路 CUDA stream 并发和单路小批次对照，使用真实 397B ViT。实验实现保留在 .t 目录；本轮没有修改生产代码、通用 MMScheduler 或默认配置。此前 NVDEC pool / GPU resize 改动保留。

本次固定负载下，单路 microbatch=4 的中位耗时 21.816 s，两路 22.131 s、四路 21.846 s，没有观察到多 stream 的明确加速。单路小批次只需 43.638 GiB Torch 峰值，相比并发的 71.420 / 105.153 GiB 更省显存。没有把并发实验启用为生产默认。

## 性能结果

同一个请求含 32 份原视频，文件读取、NVDEC、GPU resize/processor、ViT、时间戳拼装/hash 均计入，终点为 CUDA embedding 完成。不含 gRPC 序列化、网络、RDMA、LLM。这是多模态模块性能，不是完整 LLM 请求性能；32 份媒体不是 32 个独立请求并发。

| 执行方式 | 三轮耗时（s） | 平均（s） | 中位（s） | 视频/s（中位） | Torch peak allocated（GiB） |
|---|---|---:|---:|---:|---:|
| 原 packed batch32 | 27.898, 23.570, 22.857 | 24.775 | 23.570 | 1.3577 | 238.092 |
| 每批 4 个，单路顺序执行 | 21.726, 21.816, 21.852 | 21.798 | 21.816 | 1.4668 | 43.638 |
| 每批 4 个，两路并发 | 22.636, 21.252, 22.131 | 22.007 | 22.131 | 1.4459 | 71.420 |
| 每批 4 个，四路并发 | 22.199, 21.846, 21.735 | 21.927 | 21.846 | 1.4648 | 105.153 |

原 batch32 的第一轮 27.898 s 伴随 GPU allocation 从 246343 MiB 增到 270443 MiB；后两轮为 23.570 / 22.857 s。平均值受 allocator 扩容影响，不能用平均值夸大加速比。表中保留全部样本，主要比较中位数；三轮样本不足以给出稳定的细小百分比差异。

两路、四路相对原 batch32 的中位耗时变化分别为 -6.10% / -7.32%。
相对相同 microbatch=4 的单路对照，分别为 1.44% / 0.13%。

运行前需连续三次空闲检查通过。12 个正式样本前后无其他可见 GPU PID，整卡显存减本进程显存处于 26–65 MiB。正式计时没有分段 CUDA event 或 profiler 插桩。每种模式预热后测三次；单路控制额外预热一次。只有切换拓扑时清空 allocator cache，不在每次样本前清空。

## 为什么加并发没有成倍提速

当前 NVDEC pool 已并发解码；原路径在一个 stream 上逐视频提交 resize/processor，然后把整个 batch 拼成一次 packed ViT forward。并不是 32 个完整 ViT forward 依次计算。

本实验把 32 个视频拆成 8 个 microbatch=4，在 1 / 2 / 4 条 CUDA stream 上执行。它改变了排队和中间张量的生命周期，没有减少每个视频的 27 层 ViT 运算。实测拆批显著降低中间张量显存，而多 stream 的耗时收益有限。不能仅凭 GPU utilization 断言是 SM 或 HBM 饱和。

本次每个视频从 720×1280 resize 到 1216×2112（由 min_pixels=2500000 决定），选择 46 帧，grid=[23,76,132]。每视频 230736 个输入 patch、57868 个输出 token；32 份共 7383552 个 patch、1851776 个输出 token。输入工作量没有为并发实验而缩小。

## 正确性与执行实现

实验适配器只替换 Qwen3.5 batched_embedding：共享只读 ViT 权重，有限线程和独立 CUDA stream，每路按顺序执行小批次；返回前等待该路完成，恢复媒体顺序并记录消费 stream。图捕获关闭，URL/GPU/CPU/hash cache 全部为 0。

- 实际 32 视频：两路与四路、原 batch32 的所有 embedding 逐元素相同，max_abs=0，位置完全相同，形状 [57868,4096] BF16，输出均有限。
- 另一次 8 视频检查：两路×4、四路×2 与原 batch8 也逐元素一致。
- 单路 microbatch=4 控制检查了数量、形状和有限值；它单独启动，不宣称与上一进程进行了数值对照。
- 这里只覆盖同一原视频的重复媒体。不同媒体混合、动态形状、错误恢复和完整 LLM 文本质量尚未全面验证，因此实验实现没有直接落入生产默认路径。

## 冷启动编译与时间线

两条线程同时首次进入 FA4 时，CUTLASS DSL AST 预处理失败：AttributeError: 'ClassDef' object has no attribute '_new_value'。堆栈指向 flash_attn/cute/interface.py 的 compile_cache 未命中分支调用 cute.compile；首次编译没有互斥保护。加入相同键的串行预热后，固定输入并发运行通过。任意形状的生产并发还需保护新编译键初始化，不能只靠一次预热。

早期 Nsight Systems 在四条计算 stream（85/89/93/97）记录到实际 GPU kernel；两条以上 kernel 同时执行约 50 ms，单条约 9521 ms。不过当时有外部 GPU 负载，且分段 CUDA event 创建出现 host 阻塞，不能用这个受扰动区间估算稳定加速或判定硬件饱和。正式样本移除了这些 event 计时。

源代码还有同步点：ViT 提交后时间戳文字 embedding 的 CPU→GPU 拷贝、视频位置数据的 CPU→GPU 拷贝。后续可提前准备、减少等待并验证流水重叠；本轮没有修改这部分。

## 环境与原始记录

- 仓库：/home/xieshui.yyx/workspace/RTP-LLM/github-opensource
- 分支 feat/qwen35_vl_omega，HEAD 2679cfed1b2ae4b3d315c34577e9feb31ef6017f，加此前未提交改动。
- 物理 GPU0，L20D / SM103，CUDA13，torch 2.11.0+cu130，transformers 5.2.0。
- /mnt/nas1/hf/Qwen3.5-397B-A17B-FP8，真实 BF16 ViT、FA4。
- 原视频 /home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-request1-nvdec.JBWUyk/request1.mp4。
- fps=6，min_pixels=2500000，max_pixels=73728000，max_frames=180。
- NVDEC workers=32，vit_concurrency=64，gpu_max_batch_size=32，gpu_max_batch_images=256，gpu_batch_wait_ms=10，allocator expandable_segments=True。

实验目录：/data2/xieshui.yyx/workspace/RTP-LLM/.t/vit-concurrent-20260916-mxq87ig9

- clean_experiment.py / clean-result.json / clean.log / clean-gpu.csv：正式两路、四路、batch32 数据。
- serial_experiment.py / serial-result.json / serial.log / serial-gpu.csv：单路 microbatch=4 控制。
- correctness.py / correctness-result.json：早期真实 ViT 数值检查；其中时间不用于性能比较。
- parallel4.nsys-rep / parallel4.sqlite / parallel4-summary.json：受扰动的诊断时间线。
- clean.*.attempt1：首次并发编译失败记录。
- result.json / vit-result.json / perf-validity.json：早期被外部 GPU 活动干扰、已判无效的性能记录。
- concurrency-manifest.json：源码和脚本哈希、比较值、验证状态。

开始阶段曾出现容器内不可见的 GPU0 负载，停掉本实验进程后仍约 5.3 GiB / 持续计算。其负载消失后才重测正式数据；没有停止其他任务或改用其他 GPU。此前一次 OOM 同时有大量外部分配，不代表本配置自身不能执行。

复现（GPU0 需空闲，原 batch32 接近整卡显存）：

    /opt/conda310/bin/python -B /data2/xieshui.yyx/workspace/RTP-LLM/.t/vit-concurrent-20260916-mxq87ig9/launch.py --exec /data2/xieshui.yyx/workspace/RTP-LLM/.t/vit-concurrent-20260916-mxq87ig9/clean_entry
    /opt/conda310/bin/python -B /data2/xieshui.yyx/workspace/RTP-LLM/.t/vit-concurrent-20260916-mxq87ig9/launch.py --exec /data2/xieshui.yyx/workspace/RTP-LLM/.t/vit-concurrent-20260916-mxq87ig9/serial_entry

本轮未提交、未推送。没有采集 NCU 硬件计数器或真实 ViT→LLM 传输。
