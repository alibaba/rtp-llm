# FP8 MLA 峰值激活锚点：缩短 BF16 投影的存活时间

本提交从性能锚点 `34d6d58d9ed90bc3260a7e39d503bb13ca2fd9fd` 出发。普通无历史 prefix 的 MLA Prefill 先用融合 epilogue 生成独立的 E4M3 Q/K/V 并写入 target KV cache，随后由 TokenSpeed 消费这些 FP8 operand。此前 BF16 KV-up 投影及其 view 一直活到 TokenSpeed 返回；现在 cache 写入完成后先释放这些引用。带历史 prefix 的回退路径和分片路径也在 FP8 operand 形成后释放各自的 BF16 K/V。三个路径都不改变计算核、通信格式或 cache 内容。

四层验证使用 114 Prefill、115 Decode、TP8/EP8、FP8 target KV、原生 MTP 和 TokenSpeed。新增生命周期测试先在融合路径上按预期失败，修改后三个路径均通过；Bazel 测试和两端 `--config=cuda13 --config=sm10x` 构建通过。小预算 `RTP_MLA_PREFILL_EXPANDED_KV_BUDGET_GIB=0.05` 强制历史 prefix 分片的双机 flow 为 **11/11**，独立审计 `checked=11, errors=[]`；请求与源码哈希保存在个人 artifact `peak-fourlayer-flow-mla-fused-114115-20261001-r8/`。

115 独占 GPU 的 CUDA 生命周期测量在 64K、每 rank 12 头的形状下，每组预热 10 次、测量 5 次：融合 cache-insert 路径在 TokenSpeed 入口少保留 **402,653,184 字节（384 MiB）** BF16 投影；非融合 FP8 回退路径少保留 **503,316,480 字节（480 MiB）** BF16 K/V。原始数据分别为个人 artifact `mla-fused-projection-lifetime-cuda-115-20261001.json` 和 `mla-bf16-lifetime-cuda-115-20261001.json`。两项测量的 attention 均使用替身，因此证明的是张量存活量，不是完整服务的峰值显存或 Prefill 耗时。

一次四层 64K 服务 NVML A/B 曾显示 rank 0 高水位差约 646 MiB，但复核调用点后发现该次候选尚未修改无 prefix 的**融合**路径。这个差值不能归因于本提交，不作为峰值收益证据；原始记录仍保留在 `mla-64k-{candidate,baseline}-nvml-114-20261001.summary.json`。本提交后的真实服务峰值和完整 93 层热态性能仍需按独占规则重新测量。

其余峰值候选按实际路径处理：`skip-head-mid` 与每 rank 默认 6 GiB 的历史 FP8 KV 分片已经在祖先 `4dcde8045e3f1c143cb1328179ebb1bcb151b329` 中，分片的多片数值检查和四层 flow 已通过。shared expert 已是一份 gate/up GEMM 输出加两个 view；本次额外 `del` 原型没有可验证收益，已撤回。新版 `model_factory` 把本次调度器的 MoE Prefill 容量定为 65,536 个全局 token，TP8 后每 rank 8,192 个；融合投影 workspace 同样按 65,536 个 token 配置。AttnRes bank 原型会在集成版收集整段 logits 和 MTP hidden 时额外驻留，尚无峰值收益证据，已撤回。feat 的模型自有 MLA scratch 和 MTP hidden 释放接口与当前 collector 的所有权不同，未直接移植；整模型 Chunk Prefill 也未移植。

这份锚点只确认四层功能和局部激活生命周期。最终验收还需要固定代码后的完整 93 层 FP8 双机 PD 限时 smoke、独立答案复核，以及三方完整模型 64K 热态 timeline；在这些结果完成前不宣称完整模型峰值或 Prefill 性能胜出。
