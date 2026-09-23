# Qwen3.5 ViT：4 QPS 与 1.5 QPS 的口径核对及无 gRPC 复测

旧结果与上一轮使用了不同的有效视频尺寸；不能把请求吞吐差直接解释为性能回退。此次补充固定旧尺寸、同步引擎入口的对照。

## 已核实的差异

| 项目 | 历史约 4 QPS | 上一轮约 1.5 QPS | 本次对照 |
|---|---|---|---|
| 帧数 / 尺寸 H×W | 46 / 704×1280 | 46 / 1216×2112 | 46 / 704×1280 |
| grid | [23,44,80] | [23,76,132] | [23,44,80] |
| patches | 80,960 | 230,736 | 80,960 |
| 视觉 token | 20,240 | 57,684 | 20,240 |
| 含时间戳/标记的输出行数 | 20,240 | 57,868 | 20,424 |
| gpu_max_batch_size | 64 | 32 | 64 |
| 引擎入口 | 同步 mm_embedding_impl | 异步 get_embedding_result | 同步 mm_embedding_impl |
| gRPC 参与实测 | handler 入口计时，排除输出序列化/网络 | 无 | 无 |

历史 video_processing.py 把请求 min_pixels/max_pixels 直接写入全视频的 shortest_edge/longest_edge 总预算。当前 qwen3_vl_video.py:102-107 把请求 min_pixels/max_pixels 视为每帧限制，乘以采样帧数后再传 smart_resize。此逻辑来自 1e658084a99d7adade85bf969c406c3e9395f99b。

因此同样 min_pixels=2500000 时，旧代码不会把 720p 视频升至每帧 250 万像素；当前代码会。patch/token 数增至 2.85 倍，按每段全注意力序列计算的 QK/PV 二次项增至 8.1225 倍；后者并非整个 ViT 的总 FLOPs 倍数。

本次仅在测试请求中指定 height=704,width=1280 以固定历史计算量。保留当前代码的时间戳组装；该对照不代表当前不指定尺寸、min_pixels=2500000 的性能。未修改生产默认值或源码。

## 本次无 gRPC 结果

单张物理 GPU 0 / L20D，Qwen3.5-397B-A17B-FP8 的真实 ViT 权重，BF16。直接调用 MMProcessEngine.mm_embedding_impl；不创建 gRPC server/client，也不调用输出 transport。计时包含原文件读取、NVDEC、GPU resize、生产调度/组批、ViT、projector、时间戳组装、生产 feature hash，并等待输出生产 stream 的 CUDA event。排除模型加载、预热、网络、embedding 序列化、RDMA 和 LLM。

缓存全部关闭；vit_concurrency=64、vit_max_queue_size=64（同步入口不经过异步 admission）、gpu_max_batch_size=64、gpu_max_batch_images=256、gpu_batch_wait_ms=10、NVDEC workers=32；graph 关闭。

每档三轮，发请求阶段至少 15 秒后排空，预热不计入。选择中位 QPS 对应的完整轮次，RT 百分位来自同一轮。QPS=成功数/统一开始至最后请求完成时间。

| 并发 | 三轮 QPS | 选中 QPS | Mean ms | P50 ms | P95 ms | P99 ms | 三轮成功/失败 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 3.867/3.836/3.836 | 3.836 | 260.550 | 257.846 | 280.515 | 289.317 | 175/0 |
| 2 | 4.397/4.450/4.593 | 4.450 | 445.887 | 462.906 | 470.091 | 479.579 | 205/0 |
| 4 | 5.110/4.978/4.964 | 4.978 | 793.260 | 799.282 | 861.642 | 977.075 | 233/0 |
| 128 | 5.469/5.150/4.596 | 5.150 | 20605.913 | 24642.842 | 26149.900 | 26219.907 | 601/0 |

## 与历史结果并列

| 并发 | 历史 QPS（中位完整轮） | 当前大尺寸 QPS | 当前同尺寸同步 QPS |
|---:|---:|---:|---:|
| 1 | 3.751 | 1.425 | 3.836 |
| 2 | 4.171 | 1.518 | 4.450 |
| 4 | 4.276 | 1.563 | 4.978 |
| 128 | 3.797 | 1.506 | 5.150 |

历史记录曾报告三轮合并 QPS=3.749/4.173/4.268/3.695；上表改用完整中位轮保持统计一致。历史与当前源码、视频时间戳组装及停止发请求策略仍有差别（历史还要求至少 2×并发数条请求），因此不据此量化某条优化带来的净收益。
上一轮 C128 的 1.506 QPS 含 4 个 admission queue-full 失败，仅跑了一轮；异步入口与历史同步入口不同，该错误不能推导为历史路径的容量退化。

## 校验、资源与复现

两个可区分视频分别建立串行参考，1/2/4/128 并发共 135 次完整校验；embedding 有限、与各自串行参考逐 bit 相同，位置完全相同，两个视频输出不同。计时阶段保留形状/设备/dtype/生产 feature hash 校验。成功请求数与实际计算媒体数一致，缓存 hit/inflight_dedup/resident_entries 均为 0。
源码 HEAD=2679cfed1b2ae4b3d315c34577e9feb31ef6017f，前后源码一致=True，实际导入文件一致=True。保留已有工作树改动。

| 并发 | 活跃 tensor 峰值 GiB | 组批分布（选中轮） | 主机 CPU 平均/峰值 % |
|---:|---:|---|---|
| 1 | 3.44 | {'1': 58} | 24.725/28.3 |
| 2 | 6.17 | {'2': 12, '1': 44} | 32.86/37.0 |
| 4 | 11.17 | {'4': 4, '1': 16, '3': 15} | 53.46666666666666/71.1 |
| 128 | 161.15 | {'11': 1, '64': 3} | 51.65714285714286/56.5 |

tensor 峰值包括保留的两份正确性参考输出；reserved 可能因先进行高并发预热而偏高，不代表低并发独立内存需求。共享主机 CPU/其他 GPU 的逐轮观察、采样失败数见 comparison-analysis.json；不把 unavailable 当作空闲。独立实例未初始化 kmonitor，日志的 no metric named 不用于性能判定。

原始数据目录：/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-matched-sync-nogrpc-20260917-2g2u1duz
历史目录：/home/xieshui.yyx/workspace/RTP-LLM/.t/vit397b-concurrency64-batch64-fixed-gpu0-jviwkcwi
上一轮目录：/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-skill-retest-validated-20260917-pi3wwxl0

复现：config.json + adapter.py + driver.py + launch.py；load/requests.jsonl 为请求原始样本，load/results.json 为所有轮次；comparison-analysis.json 重新验证成功数/QPS/RT 百分位。资源见 resources.jsonl 和 gpu0-resources.jsonl，输出校验见 engine-state.json，实际 batch/grid 见 forwards.json，版本见 version-before/after.json 与 source-before/after.patch，收尾审计见 post-run-audit.json / cleanup.json（关闭阻塞，未生成正常 exit.json）。

测试请求全部完成且无错误，但退出时卡在 NvdecVideoPool.close 的 ThreadPoolExecutor.shutdown(wait=True)。等待后采集 shutdown-stack.txt，再向本次引擎及采样器发送 SIGTERM 清理；未修改生产代码。此问题独立于计时结果，不能把本轮称为正常退出的完整验收。

C128 三轮吞吐有约 19% 的最大/最小差距，本轮用于确认历史尺寸下吞吐量级，未扩展为稳定容量测试或源码优化收益测量。
清理确认：本次引擎与采样器已退出，GPU0 无 compute process，显存恢复为 117 MiB，利用率 0%。
