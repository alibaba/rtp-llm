# K3 FP8 Prefill 性能锚点（2026-09-30）

本锚点固定 `872879f77` 的模型实现。四层 TP8/EP8、PD、Native MTP、BF16 NCCL、FP8 E4M3 attention 与原生 MXFP4 MoE 均按现有启动配置运行；固定 65,536 个输入 token，Prefill 端报告 `prefill_total_reuse_len=0`。四层输出不用于判断语义准确性。

先前在 114/115 的同机 A/B 中，集成版与固定 `feat/k3_dev@a9bf762e` 的 target GPU span 中位分别为 71.611 和 70.946 ms。差距为 0.665 ms；两个实现的通信格式和算子边界并不完全相同。`feat` 的融合 AG/GEMM、GEMM/RS 使用本任务不接受的通信格式，因此没有整块迁入。vLLM 固定版 `3df4ae153` 的 NIXL PD target-only 三次全 rank GPU span 为 112.793、111.386、111.276 ms；它没有 Native MTP，FlashInfer 的长输入自动调优当时关闭。这些记录在 `20260929_3fs_prefill_ab/README.md` 与本目录 `README.md`。

这次在 115 Prefill / 111 Decode 上检验了 shared expert 与 routed MoE 的双 CUDA stream 候选。两次运行都经过十次同路径 64K 预热，且正式请求的八 rank trace 中有七条能按 target 时间戳严格配对。16 条正式请求均返回 HTTP 200，`pd_sep=true`，每条有七轮 MTP draft，Prefill cache reuse 为零。候选启动后的首条预热请求曾因 TokenSpeed CuteDSL 首次编译触发 PD keepalive 超时；缓存建立后重新预热、采集，超时请求不计入正式样本。

| 115/111 指标，ms | 基线 | 双 stream 候选 |
| --- | ---: | ---: |
| target：每请求最慢 rank GPU span 中位 | 69.986 | 70.239 |
| draft：每请求最慢 rank GPU span 中位 | 63.306 | 63.209 |
| target：最慢 rank routed expert 三层累计内核时长 | 约 13.50 | 约 13.59 |

候选 trace 的 shared expert kernel 在 CUDA stream 31，routed expert 在 stream 7；基线两者均在 stream 7。双 stream 确实生效，但 target 关键路径没有缩短，且候选占用更多显存。已撤回候选代码；补丁和原始 trace 保留在个人 artifact 目录 `k3-fp8-opt-20260927/`，文件名以 `rejected-moe-shared-overlap-20260930` 和 `timeline-64k-moe-overlap-115111-{0,1}-20260930` 开头。GPU span 由 CUDA launch correlation 归到同一请求的 target/draft 范围，模块内核累计时间允许跨 stream 重叠，不能当成墙钟时间。

同一模型实现先前完成 93 层 FP8 双机 PD 限时 smoke：121 条发送并独立复核通过，六条按规则跳过，其中含原清单最后三条长输出 repeat；结果在个人 artifact 目录 `smoke-93layer-fp8-integrated-f67-114115-20260929/{result.json,independent-final-audit.json}`。这不是原完整 suite 全通过。后续峰值显存修改会改变最终代码，仍须按计划重新编译并跑最终 93 层 smoke 与三方全层热态对照。
