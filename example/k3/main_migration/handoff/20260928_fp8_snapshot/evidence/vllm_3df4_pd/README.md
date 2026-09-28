# vLLM 四层 64K 历史 trace

来源：vLLM `3df4` 的四层 FP8 K3 NIXL PD Prefill，TP8/EP8、PYNCCL，65,536 输入 token。输入 token ID SHA256 为 `97a53100491426d80436747b477dbe592ea1106eed6308a99ab83ba1bd3863ee`。vLLM 只用于 target Prefill 比较，不要求 Native MTP。

八个 `*.json.gz` 是原始八 rank trace 的无损压缩。`requests.tar.gz` 包含输入、十四次预热、三次采集响应、原始 trace 摘要及 SHA256 清单；`manifest.json` 保存逐文件字节数和哈希。原件曾位于 115 的 `timeline-64k-vllm-3df4-fourlayer-nixl-pd-pynccl-profile-20260928/`。

原始摘要报告三次 GPU Prefill annotation 的最慢 rank 中位数为 113.652 ms、最大偏差 3.861%。完整 HTTP 预热耗时未收敛，启动时 FlashInfer 自动调优因超过五分钟被禁用；这两个限制应随任何比较保留。通用 NVJet GEMM 的模块归属尚未从 trace 单独证明。这是历史候选 trace，不是当前集成版的最终对照。
