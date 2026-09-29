# feat/k3_dev 四层 64K 历史 trace

来源：固定 `feat/k3_dev@55641e09` 的 r3 双机 PD 运行，113 Prefill、112 Decode，TP8/EP8，65,536 输入 token。`same-request-8rank-manifest.json` 记录了十次预热、八 rank 同一请求的 GPU 窗口以及 Native MTP 执行证据。输入 token ID SHA256 为 `97a53100491426d80436747b477dbe592ea1106eed6308a99ab83ba1bd3863ee`。

八个 `*.json.gz` 是原始八 rank trace 的无损压缩。`requests.tar.gz` 包含根目录下的输入、预热、采集响应和当时的核对清单；`manifest.json` 保存每个原件及归档的字节数与 SHA256。原件曾位于 115 的 `timeline-64k-feat-3fs-fourlayer-r3/`。

该 feat 版本使用 FP8 AllGather，与集成版要求的 BF16 NCCL 不同。这里保存的是候选实现的历史证据，不代表当前集成源码的最终性能，也不能单凭内核名称给通用 GEMM 归属模块。
