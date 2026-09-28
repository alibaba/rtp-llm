# 四层 FP8 producer 候选核对

这份记录只筛选 `feat/k3_dev` 中与当前集成版有关的 LLM producer。比较数据来自 111 Prefill、112 Decode 上的四层 65,536-token FP8 + MTP PD trace：集成版 `c7479de2` 的 `evidence/integrated_c747_r45/module-target.json`，以及固定 feat 版诊断提交 `a9bf762e` 的 `evidence/feat_profile_a9bf_r7/module-target.json`。两版各自完成了同路径预热；表中是每次请求八 rank 最慢累计 GPU 时间的中位数，单位为 ms。scope 的边界不同，且跨 stream 的累计时间可能重叠，以下数值只用于找候选。

| 路径 | 集成版 | feat 版 | 源码核对后的判断 |
| --- | ---: | ---: | --- |
| KDA output norm + gate + FP8 quant，L0–L2 | 0.165/0.165/0.165 | 0.259/0.259/0.259 | 集成版已用通用融合 producer，单看这个等价功能没有迁入 feat Triton kernel 的性能理由。仍需用相同张量核对两种 scale wire 和数值误差。 |
| MLA output gate + quant，L3 | 0.133 | 0.132 | 近似持平，保留现有实现。 |
| AttnRes + output norm，L1–L3 | 每层约 0.067 | feat `residual_fp8_producer` 每层约 0.160 | **不可直接比**：集成版 scope 输出 BF16；feat scope 输出 FP8，省掉后续量化，但与其 FP8 AllGather 连用。 |
| MLA input RMSNorm，L3 | `qkv_norm` 0.173 | `norm_fp8_producer` 0.212 | scope 和投影入口不同，尚不能判断哪版更快。 |
| MoE router，L1–L3 | 0.139/0.143/0.138 | 0.164/0.165/0.166 | 当前 trace 没有显示 feat 的收益；数值、路由分布仍以单独用例为准。 |
| MLA core，L3 | 6.308 | feat `native_mla_and_cache_pipeline` 6.087 | 两个范围的 cache 工作尚未逐项拆齐；约 0.22 ms 是后续同形状 A/B 候选，不能先迁。 |

集成版的 `KimiK3AttentionResidual.forward` 调用从 vLLM 适配的 native CUDA AttnRes，并在同一 kernel 中做输出 RMSNorm，返回 BF16。feat 的 `Fp8AttentionResidual.forward` 调用 `kimi_k3_attn_res_fp8`，把残差混合、可选输出 RMSNorm 和每 128 元素一组的 E4M3 量化放到单个 Triton kernel；它还显式维护 BF16 舍入点。两条路径可能有数值差异，不能只按 kernel 个数替换。

更关键的是数据流：feat 的 `_project_tp_sp_inputs` 接收 producer 的 `QuantizedActivation`，`all_gather_gemm.py` 随后调用 `all_gather_fp8` 或分别 AllGather FP8 value 与 scale，再让 GEMM 消费量化输入。这个通信格式不满足本任务固定 **BF16 NCCL** 的要求。若只移植 producer，却在发送前反量化成 BF16，就失去其直接喂给融合 FP8 GEMM 的收益，还新增变换。因此暂不移植 AttnRes/MLA 输入 FP8 producer 或 feat 的 FP8 AllGather/GEMM 融合路径；这只是当前约束下的筛选结论，不是否定它们在 feat 原配置里的性能。

当前可继续做的同条件实验是拆开 MLA core 的 cache/attention 子范围，记录相同张量形状与 FP8 KV 格式，再测等价 operator 的数值和热态耗时。KDA cuLA direct-output 的等形状 A/B 与四层 PD 验证已另存于 `evidence/kda_direct_candidate/`；它没有证明完整 93 层模型比 feat 或 vLLM 更快。vLLM `3df4` 历史 NIXL PD trace 的 HTTP 预热未收敛且 FlashInfer 64K 自动调优超时，不能纳入最终热态三方排序。

本次是源码与既有 trace 核对，没有启动新 GPU 测试。性能锚点、93 层 FP8 smoke 和峰值激活锚点仍待完成。
