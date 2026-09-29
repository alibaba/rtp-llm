# 四层 FP8 + MTP PD 的同机 64K 算子筛选

这次比较在 111 Prefill、112 Decode 上轮流启动两版服务。集成版二进制对应 `0a36d6d24829f79e06e229ed53feeefee8913172`，固定 `feat/k3_dev` 对应 `55641e09bc09cdafcf8f31b28aa55b18bc66d24b`。两版都用四层、TP8/EP8、FP8 target、BF16 MTP draft、相同的 65,536 个输入 token（ID SHA256 为 `97a53100491426d80436747b477dbe592ea1106eed6308a99ab83ba1bd3863ee`），从 3FS 经 FastSafetensors 直接读取。每版完成十次同路径预热；八 rank trace 分别匹配出集成版五次、feat 六次完整请求。请求和 trace 原件保存在本目录的 `evidence/integrated_0a36_r44c/` 与 `evidence/feat_55641_r6/`，两份 `run-summary.json` 记录条件和各次结果。

按 CUDA correlation 将 kernel 归到发起它的 target 或 draft CPU 范围，再对每次请求取八 rank 最慢 GPU span：集成版 Prefill/target/draft 中位数为 **138.279/74.493/63.848 ms**，feat 为 **134.873/71.134/62.830 ms**。集成版 target 慢 3.359 ms。这个差值不能直接分摊给 kernel 族，因为不同 CUDA stream 可以重叠，且两版通信格式并不相同。

| target 内核族，最慢 rank 的累计 GPU 时间中位数 | 集成版 | feat | 当前判断 |
| --- | ---: | ---: | --- |
| NCCL | 14.445 ms | 8.828 ms | feat 部分 AllGather 传 FP8 value/scale；它的融合 GEMM/ReduceScatter 还使用对称内存。固定 BF16 NCCL 约束下不能照搬。 |
| DeepGEMM，尚未归属投影 | 7.245 ms | 12.224 ms | 同名内核可承担不同投影，先补模块标签和形状再择优。 |
| MegaMoE 主核 | 14.404 ms | 16.101 ms | 集成版主核模板 `M=9600`，feat 为 `M=8448`；观察值不等于同形状实现优劣。 |
| MLA TokenSpeed | 5.376 ms | 5.333 ms | 两版已近似持平；保留 TokenSpeed Prefill。 |
| KDA cuLA delta | 3.138 ms | 3.094 ms | 差距太小，暂不迁移。 |
| KDA short conv | 1.424 ms | 1.409 ms | 集成版已接入融合分页卷积；既有同卡、同输入 A/B 证明其完整整理路径更快。 |
| KDA 其他内核 | 3.541 ms | 3.243 ms | 需核对每个子内核的输入和状态写回，不能按族迁移。 |
| FP8 producer/quant | 1.800 ms | 1.341 ms | feat 另有 AttnRes FP8 producer 0.534 ms；合并等价功能后再测。 |

逐 kernel 检查确认，集成版 target 的 `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` 约 5.30 ms，feat 约 1.37 ms，但 feat 另有 `sm100_bf16_gemm_rs_*`。feat 源码的 `all_gather_gemm.py` 在 FP8 路径集合 value 和 scale，`gemm_reduce_scatter.py` 可走融合通信；不能把这两个时长差写成纯 NCCL 优化收益。集成版 MoE 使用固定 vLLM DeepGEMM backend，feat 则使用其分支的 DeepGEMM；现有两条 trace 的路由和模板行数不同。

目前可以保留的候选是 **KDA 其他子内核、等价范围内的 FP8 producer，以及归属明确后的投影 GEMM**。需要先记录每层、每个投影的 CPU range 和输入形状，在四层用相同输入做算子 A/B、数值检查，再看完整 target 关键路径。现有 RTP trace 没有足够的模块标签，NVJet 和 DeepGEMM 仍为 `[unattributed]`。vLLM `3df4` 的历史四层 NIXL PD trace 有完整八 rank 原件，但当次 HTTP 预热未收敛，FlashInfer 64K 自动调优因超过五分钟关闭；它目前只能作为待复测的 target 候选，不能和上述两版直接宣称胜负。

这些是性能锚点前的筛选记录。完整 93 层对照、最终 FP8 双机 smoke 和峰值激活锚点尚未完成。

后续集成版增加了默认关闭的 `RTP_LLM_PROFILE_MODEL_MODULES=1` 标记，并在同一 111/112 组合上用提交 `c7479de2` 重建四层 PD 服务。`evidence/integrated_c747_r45/` 保存了 10 次热态预热、16 次相同输入的请求及八 rank trace；其中 6 次可在八 rank 完整匹配。Prefill/target/draft 最慢 rank GPU span 中位数为 **138.230/74.344/63.907 ms**，与上面的旧版结果接近。所有匹配请求都能归属 target KDA 投影与 MoE 路由、draft MLA 投影与 MoE 路由。具体方法和原始记录见该目录的 `README.md`。这验证了集成版的模块标签；固定 feat 版仍需用相同归因口径重测，当前不能据此判定任何模块迁移收益。
