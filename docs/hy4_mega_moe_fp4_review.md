# HY4 / DeepSeek-V4 MegaMoE FP4 与 SwiGLU clamp 代码审查

本文记录 2026-09-03 对 RTP-LLM `feat/hy4_cu13`、DeepSeek-V4 工作树
`4e9926957e11745a8a334d29609cccc000da413b`、vLLM 和两份公开 checkpoint
的核对结果。

## 结论

- `tencent/Hy4-preview-FP8` 的 routed experts 确实是 MXFP8，而不是已经量化好的
  FP4；配置 `MOE_STRATEGY=mega_moe` 时，需要在加载阶段把 routed expert 从
  MXFP8 在线转换成 MegaMoE 使用的 MXFP4。当前实现的格式、scale 和 gate/up
  布局是匹配的。
- HY4 的 shared expert 必须保持独立执行且不做 SwiGLU clamp。因此支持普通
  `mega_moe`，不支持会把 shared expert 融入同一个 kernel 的 `mega_moe_se`、
  `mega_moe_fused` 和 `mega_moe_fp8_se`。
- `deepseek-ai/DeepSeek-V4-Pro-0813` 不能仅凭顶层
  `quantization_config.quant_method=fp8` 判断 routed experts 是 FP8。它同时明确
  配置了顶层 `expert_dtype=fp4`，实际 safetensors 中 routed expert 是 I8 打包的
  FP4，scale 是 `F8_E8M0`。它应直接加载 FP4，不应再做 FP8→FP4 转换。

推荐 HY4-FP8 配置：

```bash
QUANTIZATION=MXFP8
MOE_STRATEGY=mega_moe
```

不要为 HY4 设置 `MOE_STRATEGY=mega_moe_se`。

## 1. 为什么 DSV4 可以融合 shared expert，而 HY4 不可以

### 1.1 模型语义的直接证据

DeepSeek-V4 官方 checkpoint 同时给 routed 和 shared expert 传入同一个
`swiglu_limit`：

- 官方 `inference/model.py` 中，routed `Expert` 使用 `args.swiglu_limit`，shared
  `Expert` 也使用同一个值；其 forward 对 gate/up 执行相同的 clamp。
  [DeepSeek-V4-Pro-0813 inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/blob/main/inference/model.py)
- vLLM 的 `DeepseekV4MLP` 在 `swiglu_limit` 非空时选择带 clamp 的激活；构造
  `shared_experts` 时传入 `self.swiglu_limit`。MegaMoE 融合 shared expert 时也只
  生成一个 `activation_clamp` 传给统一 kernel。
  [vLLM DeepSeek-V4 实现（固定版本）](https://github.com/vllm-project/vllm/blob/1356635d837c4ef002ec98c1a0296e7ff60be3c1/vllm/models/deepseek_v4/nvidia/model.py#L100-L157)

RTP-LLM 的 DSV4 工作树也保持了相同语义：

```python
# routed expert: moe/expert.py
x = require_silu_mul_split()(
    gate.contiguous(), up.contiguous(), clamp_limit=self.swiglu_limit
)

# shared expert: moe/shared_expert.py
hidden = require_silu_mul_split()(
    gate.contiguous(), up.contiguous(), clamp_limit=self.swiglu_limit
)
```

`moe_layer.py` 创建两者时传的是同一个构造参数 `swiglu_limit`；
`MegaMoEStrategySE` 因而可以把这个单值作为 unified DeepGEMM kernel 的
`activation_clamp`。这就是 `DSV4_USE_MEGA_MOE_SE=1` 语义正确的原因。

HY4 相反。vLLM 的实现明确区分两条路径：

- `HYV4FeedForward` 用于 dense/shared expert，固定使用普通 `SiluAndMul()`；
- 只有 routed `FusedMoEFactory` 收到 `swiglu_limit`；文件注释也明确说明
  dense/shared 不 clamp。

证据见 [vLLM HY4 MoE 实现（固定版本）](https://github.com/vllm-project/vllm/blob/b2f685834a6456197e7033966fdef52a23f1abcd/vllm/models/hy_v4/nvidia/moe.py#L23-L80)
和同文件的 [shared/routed 构造代码](https://github.com/vllm-project/vllm/blob/b2f685834a6456197e7033966fdef52a23f1abcd/vllm/models/hy_v4/nvidia/moe.py#L131-L170)。

### 1.2 DeepSeek 与 HY4 routed clamp 公式是否相同

相同。两者对 routed expert 使用的都不是普通的对称全量 clamp，而是同一种
非对称 SwiGLU clamp：

```python
gate = clamp(gate, max=L)       # 只限制上界，负值不截断
up = clamp(up, min=-L, max=L)  # 同时限制上下界
output = silu(gate) * up
```

证据链如下：

1. DeepSeek 官方 `inference/model.py` 先计算 `gate=w1(x)`、`up=w3(x)`，随后对
   `up` 做 `[-L, L]` clamp、对 `gate` 只做上界 clamp，再计算
   `F.silu(gate) * up`。
   [DeepSeek-V4-Pro-0813 reference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/blob/main/inference/model.py#L592-L611)
2. HY4 的 vLLM 实现明确给出了同样的三行公式，仅将它用于 routed experts。
   [vLLM HY4 formula](https://github.com/vllm-project/vllm/blob/b2f685834a6456197e7033966fdef52a23f1abcd/vllm/models/hy_v4/nvidia/moe.py#L69-L80)
3. 本仓库实际构建 DeepGEMM 时使用的
   `3rdparty/deep_gemm/0002-mega-moe-runtime-activation-clamp.patch` 也是：
   `bf16_gate=min(gate,L)`，然后依次执行 `bf16_up=max(up,-L)` 和
   `bf16_up=min(up,L)`。kernel 后续计算的是 `silu(gate) * up`。

权重槽位也没有颠倒：HY4 loader 先把 checkpoint `[gate, up]` 转成 RTP 的
`[up, gate]`；`_interleave_stacked_up_gate` 再把它变为 DeepGEMM 要求的
`[gate(0:8), up(0:8), ...]`。DeepGEMM epilogue 把每对的第一个值读作 gate、
第二个值读作 up。测试
`test_fp4_up_gate_interleave_matches_deepgemm_layout` 对这个逐块顺序做了断言。

所以 HY4 routed expert 可以直接复用当前 MegaMoE FP4 kernel 的 clamp 实现，
不需要新增一种 clamp 模式。需要禁止的仍然只是 HY4 shared expert 融合。

数值精度方面有一个非语义差异：DeepSeek 的朴素 Python reference 把 GEMM 输出
转成 FP32 后 clamp；DeepGEMM epilogue 先把 FP32 accumulator 舍入 BF16，再做
clamp/SwiGLU。DeepSeek 与 HY4 走该 kernel 时都具有这一 kernel 固有差异，且当前
调用设置 `fast_math=False`。它不是两种模型 clamp 定义不同，也不要求为 HY4
修改公式；最终精度仍应通过端到端 logits/生成一致性验证。

### 1.3 `mega_moe_se` 为什么无法表达 HY4

`mega_moe_se` 调用一次 `deep_gemm.fp8_fp4_mega_moe`，同时传入 routed 和
shared 权重，但接口只有一个 `activation_clamp`：

```python
deep_gemm.fp8_fp4_mega_moe(
    ...,
    shared_l1_weights=(...),
    shared_l2_weights=(...),
    activation_clamp=one_shared_value,
)
```

这个接口无法表达“routed=10、shared=None”。把参数设为 `10` 会错误 clamp
HY4 shared expert；设为 `None` 又会漏掉 routed expert 的 clamp。数值恰好都是
`10.0` 并不能消除差异，关键在于这个值是否被应用到 shared expert。

普通 `mega_moe` 没有这个问题：它只运行 routed experts，随后
`GenericMoeLayer` 单独调用 HY4 shared `DenseMLP`。因此 routed kernel 可以使用
`activation_clamp=10.0`，shared 路径仍不 clamp。

## 2. 两份 checkpoint 的真实权重格式

| checkpoint | 顶层量化信息 | routed expert 实际格式 | 应走的路径 |
|---|---|---|---|
| `DeepSeek-V4-Pro-0813` | `quant_method=fp8`, `fmt=e4m3`, `scale_fmt=ue8m0`, 同时 `expert_dtype=fp4` | I8 packed FP4 + F8_E8M0 scale | 直接加载 FP4 |
| `Hy4-preview-FP8` | ModelOpt `quant_algo=MXFP8`, `swiglu_limit=10.0` | F8_E4M3 + U8 UE8M0 exponent，1x32 scale | routed 在线转 FP4 |

公开配置证据：

- [DeepSeek-V4-Pro-0813 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/blob/main/config.json)
  同时包含 `expert_dtype: fp4` 与 FP8 的全局量化配置；官方 inference 配置也明确
  区分 `dtype: fp8` 和 `expert_dtype: fp4`。
  [inference/config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/blob/main/inference/config.json)
- [Hy4-preview-FP8 config.json](https://huggingface.co/tencent/Hy4-preview-FP8/blob/main/config.json)
  配置的是 ModelOpt MXFP8，没有 `expert_dtype=fp4`。

另外直接检查 safetensors header 得到：

```text
HY4 layer 1 routed gate_up_proj: F8_E4M3 [256, 4096, 6144]
HY4 layer 1 routed gate_up_proj_scale: U8 [256, 4096, 192]
DSV4 layer 1 expert 0 w1.weight: I8 [3072, 3584]
DSV4 layer 1 expert 0 w1.scale: F8_E8M0 [3072, 224]
```

这里 DeepSeek 的最后一维 `3584 = 7168 / 2`，正是两个 FP4 值打包进一个 I8；
HY4 的 scale 最后一维 `192 = 6144 / 32`，与 MXFP8 1x32 scale 一致。

## 3. HY4 MXFP8 → MegaMoE FP4 转换 review

加载链路如下：

```text
checkpoint F8_E4M3 + U8 exponent scale
  -> Mxfp8Weight
  -> OnlineMegaMoeFp4FromFp8Weight
  -> scale = 2 ** (U8 - 127)
  -> 按 expert 反量化到 BF16
  -> deep_gemm.utils.per_token_cast_to_fp4(gran_k=32, use_ue8m0=True)
  -> packed int8 FP4 + float32 UE8M0 scale
  -> MegaMoeWrapper.setup_weights_from_fp4
  -> deep_gemm.fp8_fp4_mega_moe
```

逐项检查：

1. **格式识别正确。** `wrap_moe_for_mega_moe` 优先识别 `Mxfp8Weight`，使用
   `source_block_size=32` 和 `scale_is_ue8m0_exponent=True`，不会误走普通
   128x128 FP8 scale 分支。
2. **scale 数学正确。** ModelOpt 把 UE8M0 scale 保存为原始 U8 指数字节；转换
   使用 `2 ** (value - 127)` 后再反量化 FP8，与 MXFP8 定义一致。
3. **目标格式正确。** 转换调用 DeepGEMM 自身的
   `per_token_cast_to_fp4(..., gran_k=32, use_ue8m0=True)`，与 MegaMoE kernel 的
   FP4 recipe 一致。
4. **gate/up 顺序正确。** HY4 checkpoint 的 packed `gate_up_proj` 在
   `models/hy_v4.py::_transpose_stacked_gate_up` 中从 `[gate, up]` 调整为 RTP 的
   `[up, gate]`；在线转换只改变数值格式，不再次交换布局。wrapper 以
   `w1_layout="up_gate"` 安装权重。
5. **shared expert 不会被转换。** 普通 `mega_moe` 的 loader 只包装 `moe_w1/w2`；
   shared expert 保持正常 MXFP8 FFN loader 和独立 forward，符合 HY4 语义。
6. **内存峰值受控但不可忽略。** 实现逐 expert 创建 BF16 临时张量，而不是同时
   展开所有 experts；仍会产生单个 expert 的 BF16 临时内存和一次 FP8→FP4
   重量化开销，只发生在模型加载阶段。
7. **会有二次量化误差。** HY4 原始权重已经量化为 FP8，再转成 FP4 必然比原始
   FP8 路径多一层误差；这是选择 W4A8 MegaMoE 的精度/性能取舍，不是无损转换。

对应实现：

- `rtp_llm/model_loader/online_modelopt_fp4_quant_weight.py`
  的 `convert_fp8_moe_to_fp4_ue8m0`、`OnlineMegaMoeFp4FromFp8Weight` 和
  `wrap_moe_for_mega_moe`；
- `rtp_llm/models_py/modules/glm5_mega_moe/mega_moe_wrapper.py` 中普通
  `MegaMoeWrapper` 只把 routed clamp 传入 kernel；
- `rtp_llm/models_py/model_desc/generic_moe.py` 中 HY4 策略校验及独立 shared
  expert 加和路径。

## 4. 本次代码调整

- 普通 `mega_moe` 现在把 HY4 的 `swiglu_limit` 传给 FP4 MegaMoE routed kernel。
- HY4 MXFP8 checkpoint 选择普通 `mega_moe` 时，routed weights 在加载阶段在线
  转为 FP4。
- HY4 明确拒绝所有 fused-shared MegaMoE 策略，避免得到形状正确但数值语义错误
  的结果；这些策略仍为 DSV4/GLM 等语义匹配的模型保留。
- `664173772ee352b812f38aca3dfea2ef26031d2b` 引入的 ModelOpt
  `MIXED_PRECISION`、HY4 原生 MXFP4 映射、MXFP8 shared-SE recipe 等改动已整体
  撤回；本次不扩展 fused/shared 路径。

相关回归测试覆盖：HY4 策略允许/拒绝矩阵、普通 FP4 wrapper 的 clamp 透传、
DeepGEMM 调用参数、MXFP8 1x32 UE8M0 转换与参考结果、gate/up 布局、EP/TP
切分。
