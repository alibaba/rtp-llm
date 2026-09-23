# Qwen3.5 vLLM ViT 移植后端到端压测（FA4 / 并发 1–16）

范围：直接进入生产 MMProcessEngine.mm_embedding_impl，从视频文件请求到 CUDA embedding 可用。
包含本地文件读取、元数据与抽帧、NVDEC、GPU 预处理、生产调度与组批、ViT、merger、帧时间戳/标记组装和 feature hash。
不建立 gRPC server/client，不计算 embedding 格式化、序列化、网络、RDMA 或 LLM prefill/decode。结果是 ViT 子系统端到端性能。

## 配置

- 仓库：/home/xieshui.yyx/workspace/RTP-LLM/github-opensource；分支：feat/qwen35_vl_omega；HEAD：2679cfed1b2ae4b3d315c34577e9feb31ef6017f
- GPU：物理 GPU0，NVIDIA L20D，capability=[10, 3]。
- Torch / CUDA：2.11.0+cu130 / 13.0；ViT dtype：torch.bfloat16。
- 权重：/mnt/nas1/hf/Qwen3.5-397B-A17B-FP8；参数：456,010,480；实测加载：18.107 秒。
- 实际模型：/home/xieshui.yyx/workspace/RTP-LLM/github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_vllm_vit.py。
- QKV / 参数类型：QKVParallelLinear / ModelWeightParameter。
- FA4：/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-vllm-e2e-20k-c16-20260917-4hpvfu9z/entry.runfiles/pip_gpu_cuda13_torch_flash_attn_4/site-packages/flash_attn/cute/interface.py；每个 batch 都检查 27 层使用 FA4。
- 视频：/home/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-request1-nvdec.JBWUyk/request1.mp4。
- fps=6、min_pixels=2500000、max_pixels=73728000、max_frames=180；固定 width=1280、height=704，恢复之前压测的有效尺寸。
- 实际 46 帧，H×W=704×1280，grid=[23,44,80]，80,960 个输入 patches，20,240 个纯视觉 token。
- 返回 embedding=[20,424,4096]，BF16、CUDA；其中 184 行是时间戳及 vision_start/vision_end。
- vit_concurrency=16、vit_max_queue_size=16、gpu_max_batch_size=16、gpu_max_batch_images=256、gpu_batch_wait_ms=10。
- NVDEC workers=32；GPU/CPU embedding cache、hash cache、URL cache 全关闭；CUDA Graph 关闭。
- 同步引擎入口由客户端控制在途并发，不经过异步服务 admission 队列。

## 无 profiler 压测结果

每档 3 轮，每轮持续发请求 20 秒后等待排空；预热不计时。QPS=成功数/统一开始至最后请求 CUDA 完成时间。
取中位 QPS 对应完整轮次，RT 与百分位均来自该轮。

| 并发 | 三轮 QPS | 选中 QPS | 平均 RT ms | P50 ms | P99 ms | 三轮成功/失败 | 活跃 tensor 峰值 GiB | 实际 batch 分布 |
|---:|---|---:|---:|---:|---:|---|---:|---|
| 1 | 3.951 / 3.939 / 3.921 | 3.939 | 253.748 | 253.183 | 263.248 | 238 / 0 | 3.498 | {'1': 79} |
| 2 | 4.743 / 4.499 / 4.842 | 4.743 | 419.164 | 412.295 | 467.410 | 286 / 0 | 6.289 | {'2': 41, '1': 14} |
| 4 | 5.495 / 5.211 / 5.512 | 5.495 | 727.636 | 721.564 | 757.276 | 332 / 0 | 11.406 | {'4': 28} |
| 8 | 5.617 / 5.743 / 5.621 | 5.621 | 1409.357 | 1420.578 | 1500.220 | 360 / 0 | 21.641 | {'8': 3, '7': 12, '1': 12} |
| 16 | 3.277 / 5.767 / 5.222 | 5.222 | 2979.027 | 2739.135 | 5193.271 | 320 / 0 | 42.111 | {'16': 1, '6': 7, '10': 6} |

活跃 tensor 峰值包括两份完整正确性参考 embedding。CUDA reserved/驱动显存会保留高并发预热分配，不能据此推算低并发独立内存。

## 并发 16 的波动

三轮 QPS 分别为 3.277、5.767、5.222；最终选用中位 QPS 的第三轮，平均 RT=2979.027 ms、P99=5193.271 ms。
第一轮最长请求为 12238.217 ms，其中一个实际 B15 的 batched_embedding 主机区间约 11936 ms。第三轮还有一个 B16 区间约 5124 ms。
本轮没有新增 profiler，未定位这些长尾的根因，不把它们直接归为 JIT 或 NVDEC；原始样本全部保留，因此本表是这次短时压测的观测值，不宣称无长尾稳态容量。

此前误用不固定尺寸配置的 1216×2112 测试已中止；其 57868 行输出结果未混入本报告。

## 正确性与审计

- 移植相关单测：{'passed': True, 'tests': 6, 'failures': 0, 'errors': 0}。
- 两个不同视频各有串行参考。并发 1、2、4、8、16 共 31 次完整检查：embedding 有限、与各自参考逐元素相同、位置相同；两个视频输出不同。
- 计时阶段逐请求检查 shape、dtype、device 和生产 feature hash。完整数值比较放在计时区间外。
- 实际计算媒体数等于成功数。缓存全部关闭；无 cache hit、in-flight 去重或驻留 embedding。
- version_unchanged=True；未消费输出事件=0。
- 退出码=0；NVDEC 关闭等待超时=True。
- 本次 NVDEC 线程池关闭等待 10 秒仍超时，harness 保存结果后退出进程；生产关闭问题未在本轮修复。

## 原始证据

- 目录：/data2/xieshui.yyx/workspace/RTP-LLM/.t/qwen35-vllm-e2e-20k-c16-20260917-4hpvfu9z
- config.json：配置；unit-result.json / unit.log：单测；engine.log：日志。
- load/requests.jsonl：逐请求数据；load/results.json：全部轮次；summary.json：汇总。
- forwards.json：实际组批与 grid；engine-state.json：输出校验、缓存和运行时。
- resources.jsonl：GPU/CPU；version-before/after.json：源码审计；cleanup.json：资源释放。

本轮未新增 Nsys/NCU，不根据这些数据推断 MTU/MFU。

## 资源采样

| 并发 | GPU0 平均利用率 % | 采样数 | 含不可用读数的采样数 |
|---:|---:|---:|---:|
| 1 | 64.2 | 11 | 0 |
| 2 | 59.4 | 10 | 0 |
| 4 | 61.4 | 7 | 0 |
| 8 | 94.7 | 11 | 0 |
| 16 | 76.6 | 8 | 0 |

GPU 利用率来自 nvidia-smi，不能当作 Tensor Core 利用率。有效采样未见 GPU0 外部计算进程；采样不可用不视为空闲。

收尾实测：本次引擎、采样器与控制进程均已退出；GPU0 无 compute process，显存 117 MiB，利用率 0%。
