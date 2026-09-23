> 口径纠正（2026-09-17）：本文约 1.5 QPS 使用 46×1216×2112 / 57,684 视觉 token，历史约 4 QPS 使用 46×704×1280 / 20,240 视觉 token，两者计算量不同；入口也由同步改为异步。已补充完全无 gRPC 的旧尺寸同步引擎对照，见 [同尺寸复测](qwen35_nogrpc_matched_perf_20260917.md)，不能把本文数据直接解释为相同负载下性能回退。

# Qwen3.5 ViT engine 性能重测（2026-09-17）

按 /home/xieshui.yyx/skill/inference-load-test/SKILL.md 执行。并发 1/2/4 的性能轮次均成功；并发 128 第一轮出现 4 次 admission 队列拒绝，因此按 skill 默认失败策略停止后续高并发轮次。不能将该档标为零错误容量。

## 当前结果

边界为 **engine**：实际引擎入口 → embedding 在 GPU 上可读。包含生产 admission/等待队列、文件读取、NVDEC、GPU resize/normalize、组批、27 层 ViT、merger、视频时间戳/token assembly 和生产 feature hash；排除 gRPC 序列化、网络/RDMA 及 LLM。未运行完整 397B 语言模型。

| 并发 | 数据轮次（0起） | 成功 QPS | Mean ms | P50 ms | P95 ms | P99 ms | 成功/失败 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 60s_low_concurrency / 2 | 1.425 | 701.8 | 699.7 | 717.3 | 723.5 | 86/0 |
| 2 | 60s_low_concurrency / 2 | 1.518 | 1310.1 | 1339.5 | 1352.9 | 1370.6 | 93/0 |
| 4 | 30s_initial / 1 | 1.563 | 2517.4 | 2557.2 | 2572.1 | 2584.3 | 48/0 |
| 128 | 30s_initial / 0 | 1.506 | 58826.7 | 64994.8 | 85958.7 | 86017.4 | 157/4 |

C1/C2 使用追加的 60 秒 × 3 轮；C4 使用 30 秒 × 3 轮。每档取中位 QPS 对应的完整轮次，全部延迟来自同一轮。C128 仅一轮失败诊断，不能视为三轮稳定统计。所有轮次均包含最后请求排空。

## 版本与当前配置

- 仓库：/home/xieshui.yyx/workspace/RTP-LLM/github-opensource；分支 feat/qwen35_vl_omega；HEAD 2679cfed1b2ae4b3d315c34577e9feb31ef6017f，含原有未提交改动。
- 完整 diff、相关源码和实际导入文件均做前后校验，两次运行版本一致。生产源码未改动、未提交。
- 权重：/mnt/nas1/hf/Qwen3.5-397B-A17B-FP8；真实 ViT 参数量 456,010,480，BF16，RoPE FP32；词表权重用于视频标签。模型配置/index 哈希与 shard 文件元数据已保存。
- GPU0：L20D/SM103；PyTorch 2.11.0+cu130，CUDA 13.0；未锁频。
- 原视频 SHA256：d8b10b3fd5669c6d4db8cab1d0b2cd970d4e9db09a2ce09059b3a3e3a76987b0；1280×720、233 帧、30 fps。
- 请求配置：fps=6、min_pixels=2500000、max_pixels=73728000、max_frames=180。
- 实际输入：采样 46 帧，resize 2112×1216，grid=[23,76,132]，230736 patches。
- 输出：57684 纯视觉 tokens；包含时间戳等为 [57868,4096] BF16 embedding，position=[57868,3]。不能与早期 20240 tokens 的旧分辨率结果直接对比。
- vit_concurrency=64，vit_max_queue_size=64，admission capacity=128；gpu_max_batch_size=32，gpu_max_batch_images=256，gpu_batch_wait_ms=10，mm_max_queue_size=1024。
- NVDEC decode workers=32，本地预处理；CUDA Graph 关闭；GPU/CPU embedding、hash、URL、旧式缓存和 igraph 均关闭，最终 hit/inflight_dedup/resident_entries 均为 0。
- PyTorch expandable_segments=True。不主动清 OS page cache，文件读取走生产代码，OS 缓存可能命中。
- 使用最近高分辨率完整路径验证可运行的 batch 上限 32；历史 batch64 是较小输入。本次没有声称高分辨率 B64 可运行。

## 并发 128 的失败原因

该轮 161 次尝试，157 次成功、4 次失败，失败率 2.484%；错误全为：

    ViT queue is full: admitted=128, capacity=128, requested=1

成功请求 Mean=58.827 秒、P99=86.017 秒。该轮遇到首个错误后提前停止发新请求，并排空已在途请求。成功 QPS=1.506120，包含 104.241 秒的完整执行和排空。失败请求不计成功 QPS，原始失败 RT 与异常全部保留。

当前代码中 _async_compute 先调用 complete_cache 使结果 ready（mm_process_engine.py:1262），名额在 _on_async_compute_done 完成回调里释放（:1144）。结果可读与 admission 释放之间存在窗口；客户端立即提交新请求时可能仍看到 admitted=128。这个代码顺序与本次现象一致，本次未增加内部时序探针进一步验证因果，也未修改实现或悄悄提高 queue 参数。

## 正确性与预检查

主运行在并发 1/2/4/128 下分别全量检查 1/2/4/128 个请求。使用原视频与颜色反转、尺寸/帧数相同的 H.264 High 对照视频，先建立各自串行参考。两个输入的 embedding 不同；全部并发输出 finite、与各自参考逐 bit 相同、position 相同。压测只使用原视频，每请求检查 shape/dtype/device 与完整生产 feature hash；无 embedding 校验失败、无超时、无 OOM。

初始合成对照视频采用 H.264 Constrained Baseline；在原视频 High → Baseline 的复用解码会话中返回 0 帧，而 CPU 解码为 233 帧。该预检查失败已保留；改为同属 High profile 的对照素材后交替解码通过。这个 profile 切换限制尚未修复，不属于原视频吞吐结果。证据：/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-skill-retest-20260917-i6lnp2n6/probe.log 与 probe_high.log。

## 波动处理与全部原始轮次

初次 C1 首轮含一次 7.603 秒长尾，该轮主机 CPU 采样峰值达到 78.6%；C2 的短轮次也存在波动。保留所有短轮次，另追加相同版本/输入/引擎参数的 C1/C2 各 60 秒 × 3 轮，主表采用追加组的中位完整轮次，没有挑选最快轮。

| 运行组 | 并发 | 轮次 | 含排空秒数 | 尝试 | 成功 | 失败 | 成功 QPS | Mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30s_initial | 1 | 0 | 30.099 | 32 | 32 | 0 | 1.063165 | 940.341 |
| 30s_initial | 1 | 1 | 30.256 | 43 | 43 | 0 | 1.421221 | 703.445 |
| 30s_initial | 1 | 2 | 30.241 | 43 | 43 | 0 | 1.421917 | 703.133 |
| 30s_initial | 2 | 0 | 30.949 | 42 | 42 | 0 | 1.357085 | 1457.495 |
| 30s_initial | 2 | 1 | 30.922 | 46 | 46 | 0 | 1.487617 | 1329.555 |
| 30s_initial | 2 | 2 | 30.738 | 40 | 40 | 0 | 1.301308 | 1520.057 |
| 30s_initial | 4 | 0 | 30.699 | 48 | 48 | 0 | 1.563569 | 2516.305 |
| 30s_initial | 4 | 1 | 30.714 | 48 | 48 | 0 | 1.562808 | 2517.447 |
| 30s_initial | 4 | 2 | 30.763 | 48 | 48 | 0 | 1.560298 | 2521.782 |
| 30s_initial | 128 | 0 | 104.241 | 161 | 157 | 4 | 1.506120 | 58826.735 |
| 60s_low_concurrency | 1 | 0 | 60.403 | 86 | 86 | 0 | 1.423766 | 702.198 |
| 60s_low_concurrency | 1 | 1 | 60.314 | 86 | 86 | 0 | 1.425877 | 701.144 |
| 60s_low_concurrency | 1 | 2 | 60.370 | 86 | 86 | 0 | 1.424549 | 701.804 |
| 60s_low_concurrency | 2 | 0 | 60.903 | 93 | 93 | 0 | 1.527015 | 1302.356 |
| 60s_low_concurrency | 2 | 1 | 60.999 | 92 | 92 | 0 | 1.508222 | 1318.627 |
| 60s_low_concurrency | 2 | 2 | 61.266 | 93 | 93 | 0 | 1.517981 | 1310.102 |

## 资源

| 并发 | PyTorch allocated 峰值 GiB | reserved 峰值 GiB | 驱动显存采样峰值 GiB | GPU0 util 采样均值 % | 主机 CPU 采样均值 % | GPU0 样本数 | 实际 batch 直方图 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8.2 | 10.0 | 11.0 | 83.0 | 26.6 | 36 | {'1': 86} |
| 2 | 16.0 | 17.3 | 18.3 | 87.8 | 25.2 | 37 | {'2': 22, '1': 49} |
| 4 | 30.2 | 264.8 | 266.9 | 81.8 | 26.3 | 8 | {'4': 1, '3': 11, '1': 11} |
| 128 | 229.7 | 265.1 | 267.3 | 92.1 | 27.2 | 35 | {'1': 1, '32': 4, '28': 1} |

PyTorch 峰值包含该轮 worker 预热；驱动显存只取正式轮次采样。正确性检查的两份 GPU embedding 参考约 0.883 GiB 仍在进程中。主运行先做 C128 正确性，PyTorch allocator 的大块 reserved 会保留到后续低并发；静态 reserved/驱动显存不是每请求必要显存。

原始 skill 采样器全卡查询曾超时，已保留 available=false，未当成零负载。随后补充 GPU0 独立采样；主运行部分早期轮次仍有全卡采样缺口。单卡补查其余 GPU 当时利用率均为 0；静态显存占用和共享主机 CPU 负载保留在资源日志中。因此这是共享主机结果，不能宣称全程独占整机。独立 engine 未初始化外部 kmonitor，日志含未注册指标 warning；请求计数、batch 和资源由测试独立记录，相关日志开销仍包含在执行中。

## 计算方法与边界

- success QPS = 校验成功数 / 从统一放行到最后请求输出可读的墙钟时间。
- RT = invoke 开始到 embedding 可读，使用生产者 stream 的 CUDA event 等待完成，未用全设备同步串行化不同请求。
- Mean 是请求 RT 算术平均；P50/P95/P99 对同一选中轮次的成功 RT 样本线性插值。
- embedding tokens/s = QPS × 57868；纯视觉 tokens/s = QPS × 57684。它们不是 LLM 生成 TPS。
- 每轮各 worker 预热 1 请求，排除加载、编译、预热；轻量校验在 RT 之外，但仍占闭环客户端容量。
- 这是闭环固定在途量测试。C128 发请求窗口结束后仍有较长排空；不证明固定到达率容量或长期显存稳定性。

## 文件

主运行：/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-skill-retest-validated-20260917-pi3wwxl0

追加长轮次：/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-skill-long-lowc-20260917-9a30x8dr

两个目录中的 load/requests.jsonl、load/results.json、load/summary.md 是 skill runner 的原始输出；engine-state.json 记录生效配置、正确性、缓存和每轮内存，forwards.json 记录实际组批与预处理尺寸；resources.jsonl 和 gpu0-resources.jsonl 记录资源。source-before.patch/source-after.patch、version-before.json/version-after.json、checkpoint-manifest.json 保存版本证据。

独立复算：/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-skill-retest-validated-20260917-pi3wwxl0/summarize_all.py；汇总：/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-skill-retest-validated-20260917-pi3wwxl0/combined-analysis.json。退出与资源释放证据见 cleanup.json。
