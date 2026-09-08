# GLM-5.3-Flash 数值路径修复（2026-09-09）

GLM-5.3-Flash 的 routed MoE 使用发布的 FP8 权重路径，router 投影使用 FP32，KDA 四 tap 卷积在 FP32 乘加和 SiLU 后输出 BF16。主模型测试关闭 MTP，保留 GLM 的 top8、sigmoid routing、correction bias、2.5 缩放、SwiGLU clamp=10 和 mHC 语义。

## 加载与执行约束

- 启动时同时设置 `MOE_STRATEGY=mega_moe_fp8` 和 `--moe_strategy mega_moe_fp8`。GLM 加载器拒绝将发布权重转 FP4 的 `mega_moe`、`mega_moe_se`、`mega_moe_fused`。
- Router 的 checkpoint BF16 数值在加载时提升为 FP32。`Glm53FP32Router` 在投影前提升输入，直接产生 FP32 logits；要求 `torch.backends.cuda.matmul.allow_tf32=False`。8192 行分块用于限制临时输入空间，不改变 top-k 规则。
- KDA 卷积权重按 FP32 加载；conv history 为 BF16，SSM 为 FP32。Prefill 在 kernel 内提升所有 history 分支及循环寄存器，避免整个激活/输出的 FP32 临时副本。Decode 的连续 Q/K/V 输出、页表映射与 Graph padding 规则保持不变。
- Prefill routed token 的 TP/SP 切分、shared expert 的 TP 计算及归并沿用当前 GLM 实现。Decode TP1/DP8/EP8 下 indexer 也是 DP。

## FP8 的精度和容量边界

固定 SM103 DeepGEMM `bd515de` 需要 UE8M0 scales，因而会将发布的 FP8+FP32 block scales 转成 FP8+UE8M0。这仍然涉及 FP8 重量化，并非原始 bytes/scales 的无损重排。[SGLang 同版本 FP8 MoE 加载路径](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/layers/quantization/fp8.py#L1721) 对 DeepGEMM 也进行该转换。这里修复的是 FP8→FP4，不声称完整模型与官方跨框架逐 token 一致。

该 DeepGEMM 版本的 FP8 `BLOCK_M=224` 特化在 SM103 上存在重复输出不一致的问题，在旧库中也已复现。GLM wrapper 暂时限制单次逻辑输入，使特化停留在已验证的 `BLOCK_M≤192`：EP8/top8/288专家时每次最多2880 token。物理 buffer 向上对齐到3072，调用方不能将物理 padding 当作逻辑预算。修复底层 kernel 并独立验收前不能解除此限制。

B8×131072、TP8下，每卡131072 routed token 分46块，42个MoE层共1932次主kernel调用；Decode每卡B48仍为每层一次。该容量限制会增加Prefill分块开销，总batch与序列长度不变。

注意：dense shared expert 的 `sm100_fp8_fp4_gemm_1d1d_impl` 是支持不同类型的通用模板名。实际 A/B 实例化类型均为 `cutlass::float_e4m3_t` 时，它执行 FP8×FP8，不能仅凭名字中的“fp4”推断权重类型。

## 验证

在 CUDA Runtime 确认的 SM103 上，57项精度/配置回归和3项通用Prefill卷积回归通过，无跳过。涉及模块：

```text
rtp_llm.models_py.modules.hybrid.test.glm53_precision_contract_test
rtp_llm.models_py.triton_kernels.kimi_kda.test.glm53_decode_fusion_test
rtp_llm.models_py.triton_kernels.kimi_kda.test.glm53_reference_numerics_test
rtp_llm.test.glm5_3_flash_config_test
rtp_llm.models_py.modules.glm5_mega_moe.test_fused_moe_wrapper_layout
rtp_llm.models_py.triton_kernels.causal_conv1d.test.test_casual_conv1d_prefill
```

真实 checkpoint 合约测试需设置 `GLM5_CKPT_PATH`，GPU测试需使用正确的构建依赖。覆盖 FP32 router 的 B1/7/48/384/8193、Graph replay、真实权重 logits/top8、混合 prefix=0/128/256、长度1/3/127/128/129/4097、乱序物理页、跨页、BF16/FP32 history、32步SSM和非整除MoE分块。

另以两层各两个真实专家交错复制到288 slots，在EP8上执行80例均衡路由和48例集中到rank0的路由。每例五次重复输出一致，对独立同量化FP32参考的最大输出相对L2分别为0.000566654和0.000532039。六个真实矩阵的UE8M0重量化相对L2约2.37%–2.41%；包含FP8激活量化的算子输出相对发布权重解量化+BF16激活参考约5.87%–9.17%。这些不是任务准确率变化，也不是全checkpoint的穷举验证。

真实双机PD混长24/24、Prefill两波B8×128K的16/16请求、Decode B384×128K的384/384请求均通过；D输出256 token，MTP0，无重试。长文本检索验收不能代替完整模型质量评测。
