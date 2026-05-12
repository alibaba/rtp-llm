# 模型与算子接入

> **Workspace:** c:\WorkingSpace\Code\vllm-ascend

---

## User
模型、算子接入

## User
Qwen3模型，调用相关算子，在vllm-ascend仓内的定义

## Assistant
我来帮你查找 vllm-ascend 仓库中 Qwen3 模型相关的

现在我已经收集了足够的信息，下面为你详细梳理 Qwen3 系列模型在 vllm-ascend 仓库中调用的所有算子及其定义。

---

## Qwen3 模型在 vllm-ascend 中的算子接入全景

vllm-ascend 中涉及 **4 个 Qwen3 相关 patch 文件**，对应不同模型变体和硬件平台：

| Patch 文件 | 目标模型 | 硬件平台 |
|---|---|---|
| [patch_qwen3_5.py](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py) | Qwen3.5 (GatedDeltaNet + Attention) | 910B/C |
| [patch_qwen3_next.py](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_next.py) | Qwen3Next (GatedDeltaNet) | 910B/C |
| [patch_qwen3_5_310.py](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5_310.py) | Qwen3.5 (GatedDeltaNet) | 310P |
| [patch_qwen3_c8.py](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_c8.py) | Qwen3 W8A8C8 量化 | 通用 |

---

### 一、核心算子总览

Qwen3 系列模型主要涉及 **6 大类算子**：

#### 1. `npu_causal_conv1d_custom` — 因果卷积（AscendC 自定义算子）

**调用位置**：[patch_qwen3_5.py:182](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py#L182)、[patch_qwen3_next.py:168](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_next.py#L168)

**场景**：Prefill 阶段的 non-spec 部分，对 mixed_qkv 做因果卷积变换

**C++ 注册定义**：[torch_binding.cpp:880-895](file:///c:/WorkingSpace/Code/vllm-ascend/csrc/torch_binding.cpp#L880-L895)

```cpp
ops.def(
    "npu_causal_conv1d_custom(Tensor x, "
    "                         Tensor weight, "
    "                         Tensor conv_state, "
    "                         Tensor? bias_opt, "
    "                         int[] query_start_loc_opt, "
    "                         int[] cache_indices_opt, "
    "                         int[] initial_state_mode_opt, "
    "                         int[] num_accepted_tokens_opt, "
    "                         int activation_mode, "
    "                         int pad_slot_id, "
    "                         int run_mode) -> (Tensor output)");
ops.impl("npu_causal_conv1d_custom", torch::kPrivateUse1, &vllm_ascend::npu_causal_conv1d_custom);
```

**Python 调用方式**：
```python
mixed_qkv_non_spec = torch.ops._C_ascend.npu_causal_conv1d_custom(
    mixed_qkv_non_spec,
    conv_weights_T,
    conv_state=self_kv_cache[0],
    bias_opt=self.conv1d.bias,
    query_start_loc_opt=to_int64_tuple(non_spec_query_start_loc),
    cache_indices_opt=to_int64_tuple(non_spec_state_indices_tensor),
    initial_state_mode_opt=to_int64_tuple(has_initial_state),
    num_accepted_tokens_opt=[],
    activation_mode=activation_num,
    pad_slot_id=PAD_SLOT_ID,
    run_mode=0,  # 0=prefill mode
)
```

> **310P 平台**使用独立算子 `npu_causal_conv1d_310`，定义在 [torch_binding.cpp:711-719](file:///c:/WorkingSpace/Code/vllm-ascend/csrc/torch_binding.cpp#L711-L719)

---

#### 2. `causal_conv1d_update` — 因果卷积更新（Triton 算子）

**调用位置**：Decode 阶段和 spec decoding 阶段

**实现**：[vllm_ascend/ops/triton/mamba/causal_conv1d.py](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/ops/triton/mamba/causal_conv1d.py)

**Patch 替换**：[patch_triton.py:11](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_triton.py#L11) 将上游 vLLM 的 `causal_conv1d_update` 替换为 NPU 优化版本：

```python
vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update = causal_conv1d_update_npu
vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn = causal_conv1d_fn
```

核心 Triton kernel 为 `_causal_conv1d_update_kernel_npu_tiled`，支持：
- Varlen（变长序列）
- Speculative decoding（投机解码）
- APC（Accepted token count）
- 分块 tiling 策略

---

#### 3. `npu_recurrent_gated_delta_rule` — 门控递归 Delta 规则（torch_npu 算子）

**调用位置**：[patch_qwen3_next.py:237-249](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_next.py#L237-L249)（Qwen3Next Decode + Spec Decode）

```python
core_attn_out_spec = torch_npu.npu_recurrent_gated_delta_rule(
    query=query_spec.squeeze(0),
    key=key_spec.squeeze(0),
    value=value_spec.squeeze(0),
    g=g_spec.squeeze(0),
    beta=beta_spec.squeeze(0),
    state=ssm_state,
    scale=key_spec.shape[-1] ** -0.5,
    actual_seq_lengths=actual_seq_lengths,
    ssm_state_indices=spec_state_indices_tensor.flatten(),
    num_accepted_tokens=num_accepted_tokens.to(torch.int32),
)
```

> **注意**：此算子来自 `torch_npu`（华为 PTA 库），不在 vllm-ascend 仓内定义。Qwen3.5 不使用此算子，因为其 `ssm_state` 为 float32 格式，该算子暂不支持。

---

#### 4. `fused_recurrent_gated_delta_rule` — 融合递归门控 Delta 规则（Triton 算子）

**调用位置**：[patch_qwen3_5.py:221-234](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py#L221-L234)（Qwen3.5 的 Spec Decode + Decode 阶段）

**实现**：[vllm_ascend/ops/triton/fla/sigmoid_gating.py](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/ops/triton/fla/sigmoid_gating.py)

核心 kernel `fused_recurrent_gated_delta_rule_fwd_kernel` 支持：
- `USE_INITIAL_STATE`：是否使用初始 SSM 状态
- `IS_VARLEN`：变长序列
- `IS_CONTINUOUS_BATCHING`：连续批处理
- `IS_SPEC_DECODING`：投机解码
- `USE_QK_L2NORM_IN_KERNEL`：QK L2 归一化

**Patch 替换**：[patch_triton.py:7-8](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_triton.py#L7-L8)
```python
vllm.model_executor.layers.fla.ops.fused_recurrent.fused_recurrent_gated_delta_rule_fwd_kernel = (
    fused_recurrent_gated_delta_rule_fwd_kernel
)
```

---

#### 5. `fused_sigmoid_gating_delta_rule_update` — 融合 Sigmoid 门控 Delta 规则更新（Triton 算子）

**调用位置**：[patch_qwen3_5.py:254-268](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py#L254-L268)（Qwen3.5 纯 Decode 阶段的快速路径）

**实现**：[sigmoid_gating.py:239-393](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/ops/triton/fla/sigmoid_gating.py#L239-L393)

这是一个**融合 kernel**，将 sigmoid gating 计算与递归 delta rule 更新合并为单次 kernel launch，减少 NPU 同步开销。

```python
core_attn_out_non_spec = fused_sigmoid_gating_delta_rule_update(
    A_log=self.A_log.contiguous(),
    dt_bias=self.dt_bias.contiguous(),
    q=query_non_spec.contiguous(),
    k=key_non_spec.contiguous(),
    v=value_non_spec.contiguous(),
    a=a.contiguous(),
    b=b.contiguous(),
    initial_state_source=ssm_state,
    initial_state_indices=non_spec_state_indices_tensor,
    cu_seqlens=non_spec_query_start_loc,
    use_qk_l2norm_in_kernel=True,
    softplus_beta=1.0,
    softplus_threshold=20.0,
)
```

---

#### 6. `chunk_gated_delta_rule` — 分块门控 Delta 规则（Triton 算子）

**调用位置**：[patch_qwen3_5.py:241-253](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py#L241-L253)（Prefill 阶段 non-spec 部分）

**实现**：[vllm_ascend/ops/triton/fla/chunk.py](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/ops/triton/fla/chunk.py)

内部调用了多个子 kernel：
- `chunk_local_cumsum` — 局部累积求和
- `recompute_w_u_fwd` — 重计算 W 和 U
- `chunk_scaled_dot_kkt_fwd` — 缩放点积
- `chunk_gated_delta_rule_fwd_h` — 前向 h 状态
- `chunk_fwd_o` — 前向输出

**Patch 替换**：[patch_triton.py:15](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_triton.py#L15)
```python
vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule = chunk_gated_delta_rule
```

---

### 二、辅助算子

#### `fused_gdn_gating_patch` — 融合 GDN 门控计算

**调用位置**：[patch_qwen3_5.py:207](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py#L207)、[patch_qwen3_next.py:197](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_next.py#L197)

**实现**：[vllm_ascend/ops/triton/fused_gdn_gating.py](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/ops/triton/fused_gdn_gating.py)

计算 `g = -exp(A_log) * softplus(a + dt_bias)` 和 `beta = sigmoid(b)`，用于 GatedDeltaNet 的门控机制。

---

#### `fused_qkvzba_split_reshape_cat` — 融合 QKVZ+BA 拆分重排

**调用位置**：[patch_qwen3_next.py:56-63](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_next.py#L56-L63)（仅 Qwen3Next）

**实现**：[vllm_ascend/ops/triton/fla/fused_qkvzba_split_reshape.py](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/ops/triton/fla/fused_qkvzba_split_reshape.py)

将投影后的 `mixed_qkvz` 和 `mixed_ba` 融合拆分为 `mixed_qkv, z, b, a` 四个张量，避免多次独立 kernel launch。

---

#### `torch.ops.vllm.triton_split_qkv_rmsnorm_mrope` — 融合 QKV 拆分 + RMSNorm + mRoPE

**调用位置**：[patch_qwen3_5.py:339-352](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py#L339-L352)（Qwen3.5 的 full_attention 层）

用于 Qwen3.5 的 Attention 层，将 QKV 拆分、RMSNorm 归一化、mRoPE 旋转位置编码融合为单次操作。

---

#### `torch.ops.vllm.gdn_attention_core` — GDN 注意力核心操作

**调用位置**：[patch_qwen3_5.py:93-98](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py#L93-L98)、[patch_qwen3_next.py:82-87](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_next.py#L82-L87)

这是 vLLM 上游注册的 custom op，用于将 GDN 核心注意力计算封装为可被 CUDA Graph 捕获的操作。其内部回调 `_forward_core` 方法。

---

### 三、算子调度逻辑总结

以 Qwen3.5 GatedDeltaNet 为例，算子在不同推理阶段的调度如下：

```
┌─────────────────────────────────────────────────────────────┐
│                   GatedDeltaNet._forward_core               │
├─────────────┬──────────────────┬────────────────────────────┤
│   阶段       │  Spec 部分       │  Non-Spec 部分             │
├─────────────┼──────────────────┼────────────────────────────┤
│ 1.卷积变换   │ causal_conv1d_   │ Prefill:                   │
│             │ update (triton)  │   npu_causal_conv1d_custom │
│             │                  │   (AscendC自定义算子)       │
│             │                  │ Decode:                    │
│             │                  │   causal_conv1d_update     │
│             │                  │   (triton)                 │
├─────────────┼──────────────────┼────────────────────────────┤
│ 2.门控计算   │ fused_gdn_gating_patch (triton)              │
│             │ g = -exp(A_log)*softplus(a+dt_bias)           │
│             │ beta = sigmoid(b)                             │
├─────────────┼──────────────────┼────────────────────────────┤
│ 3.递归注意力 │ fused_recurrent_ │ Prefill:                   │
│             │ gated_delta_rule │   chunk_gated_delta_rule   │
│             │ (triton)         │   (triton)                 │
│             │                  │ Decode(有prefill):          │
│             │                  │   fused_recurrent_gated_   │
│             │                  │   delta_rule (triton)      │
│             │                  │ Decode(纯decode):           │
│             │                  │   fused_sigmoid_gating_    │
│             │                  │   delta_rule_update        │
│             │                  │   (融合triton kernel)       │
└─────────────┴──────────────────┴────────────────────────────┘
```

### 四、310P 平台差异

310P 平台不使用 AscendC 自定义算子和 `torch_npu` 算子，而是全部使用 **PyTorch 原生算子组合**实现：

| 算子功能 | 910B/C 实现 | 310P 实现 |
|---|---|---|
| Prefill 卷积 | `npu_causal_conv1d_custom` | `causal_conv1d_fn` (PyTorch) |
| Decode 卷积 | `causal_conv1d_update` (triton) | `causal_conv1d_update` (PyTorch) |
| 门控计算 | `fused_gdn_gating_patch` (triton) | `fused_gdn_gating_pytorch` |
| Prefill 递归 | `chunk_gated_delta_rule` (triton) | `chunk_gated_delta_rule_pytorch` |
| Decode 递归 | `fused_recurrent_gated_delta_rule` / `fused_sigmoid_gating_delta_rule_update` | `fused_recurrent_gated_delta_rule_pytorch` |

---

### 五、Patch 注册机制

所有 patch 通过 [patch/worker/\_\_init\_\_.py](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/__init__.py#L37-L44) 统一加载：

```python
if is_310p():
    import vllm_ascend.patch.worker.patch_qwen3_5_310  # noqa
else:
    import vllm_ascend.patch.worker.patch_qwen3_next  # noqa
    import vllm_ascend.patch.worker.patch_qwen3_5  # noqa
```

而 Triton 算子的全局替换在 [patch_triton.py](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/patch/worker/patch_triton.py) 中完成，将上游 vLLM 的 triton 实现替换为 NPU 优化版本。

## User
Qwen3-4B embedding算子调用的什么接口

## Assistant
现在我已经完整梳理了 Qwen3-4B 的 Embedding 算子调用链路。下面是详细说明：

---

## Qwen3-4B Embedding 算子调用接口

Qwen3-4B 的 Embedding 层（`embed_tokens`）在 vllm-ascend 中通过 **算子替换机制** 使用 Ascend 定制实现，而非直接 patch。

### 一、替换注册机制

在 [utils.py:647](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/utils.py#L647) 中，通过 vLLM 的 `CustomOp` 注册机制将上游的 `VocabParallelEmbedding` 替换为 Ascend 版本：

```python
REGISTERED_ASCEND_OPS = {
    ...
    "VocabParallelEmbedding": AscendVocabParallelEmbedding,   # ← Embedding 层
    "ParallelLMHead": AscendParallelLMHead,                   # ← LM Head 层
    "LogitsProcessor": AscendLogitsProcessor,                 # ← Logits 处理
    ...
}

for name, op_cls in REGISTERED_ASCEND_OPS.items():
    CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)
```

> 这意味着当 Qwen3-4B 模型中 `embed_tokens = VocabParallelEmbedding(...)` 被创建时，实际实例化的是 `AscendVocabParallelEmbedding`。

---

### 二、Embedding 前向调用链

Qwen3-4B 的 `embed_tokens` 调用 `AscendVocabParallelEmbedding.forward()`，其内部根据并行策略分为两条路径：

#### 路径 1：Embedding TP 模式（`forward_type == "embed_tp"`）

当启用 `embedding_tp_enable()` 且 prefix 包含 `"embed_tokens"` 时触发，定义在 [vocab_parallel_embedding.py:163-182](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py#L163-L182)：

```python
def _forward_embed_tp(self, input_):
    # 1. AllGather: 收集所有 TP rank 的 input_ids
    complete_input = self.comm_group.all_gather(input_, dim=0)
    
    # 2. 计算掩码输入: 将 vocab index 映射到本地分片
    masked_input, input_mask = self._get_masked_input_and_mask(
        complete_input,
        self.shard_indices.org_vocab_start_index,
        self.shard_indices.org_vocab_end_index,
        ...
    )
    
    # 3. 核心 Embedding 查表: 调用 quant_method.embedding()
    output_parallel = self.quant_method.embedding(self, masked_input.long())
    
    # 4. 掩码清零: 无效位置填 0
    output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
    
    # 5. ReduceScatter: 将结果按 token 分发回各 TP rank
    output = self.comm_group.reduce_scatter(output_parallel, dim=0)
    output = output.view(input_.shape[0], -1)
    return output
```

**调用的算子接口**：
| 步骤 | 接口 | 说明 |
|---|---|---|
| AllGather | `get_embed_tp_group().all_gather()` | HCCL 通信原语 |
| 掩码计算 | `_get_masked_input_and_mask()` | 纯 PyTorch 张量操作 |
| **Embedding 查表** | **`self.quant_method.embedding()`** | **核心算子** |
| 清零 | `masked_fill_()` | PyTorch 原地操作 |
| ReduceScatter | `get_embed_tp_group().reduce_scatter()` | HCCL 通信原语 |

#### 路径 2：普通 TP 模式（默认路径）

定义在 [vocab_parallel_embedding.py:184-204](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py#L184-L204)：

```python
def _forward_origin(self, input_):
    if self.tp_size > 1:
        # 1. 计算掩码输入
        masked_input, input_mask = self._get_masked_input_and_mask(...)
    else:
        masked_input = input_
    
    # 2. 核心 Embedding 查表
    output_parallel = self.quant_method.embedding(self, masked_input.long())
    
    # 3. 掩码清零
    if self.tp_size > 1:
        output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
    
    # 4. AllReduce: 跨 TP rank 归约
    output = torch.ops.vllm.maybe_pad_and_reduce(output_parallel)
    return output
```

**调用的算子接口**：
| 步骤 | 接口 | 说明 |
|---|---|---|
| 掩码计算 | `_get_masked_input_and_mask()` | 纯 PyTorch 张量操作 |
| **Embedding 查表** | **`self.quant_method.embedding()`** | **核心算子** |
| 清零 | `masked_fill_()` | PyTorch 原地操作 |
| AllReduce | `torch.ops.vllm.maybe_pad_and_reduce` | vLLM 自定义 op |

---

### 三、核心 Embedding 查表算子

`self.quant_method.embedding()` 的具体实现取决于量化配置：

#### 非量化场景（Qwen3-4B 默认）

使用 `UnquantizedEmbeddingMethod`，其 `embedding()` 方法来自上游 vLLM，底层调用 **`torch.nn.functional.embedding()`**，即 PyTorch 标准的 Embedding 查表操作：

```python
# vllm 上游 UnquantizedEmbeddingMethod.embedding()
def embedding(self, layer, input_):
    return torch.nn.functional.embedding(input_, layer.weight)
```

在 NPU 上，`torch.nn.functional.embedding()` 会被 torch_npu 自动映射到 **Ascend 的 aclnnEmbedding** 算子。

#### 310P 平台

310P 使用 `AscendUnquantizedEmbeddingMethod310`，定义在 [310p/vocab_parallel_embedding.py:30-42](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/_310p/ops/vocab_parallel_embedding.py#L30-L42)：

```python
class AscendUnquantizedEmbeddingMethod310(UnquantizedEmbeddingMethod):
    def process_weights_after_loading(self, layer):
        layer.weight_nz = maybe_trans_nz(layer.weight)  # 转 NZ 格式

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight_nz, bias)  # LM Head 用 F.linear
```

> 注意：310P 的 `embedding()` 方法仍继承自 `UnquantizedEmbeddingMethod`，只是在权重加载后做了 NZ 格式转换（`maybe_trans_nz`），以适配 310P 的存储格式。

---

### 四、`_get_masked_input_and_mask` — Ascend 定制优化

这是 vllm-ascend 对上游的重要优化点，定义在 [vocab_parallel_embedding.py:135-164](file:///c:/WorkingSpace/Code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py#L135-L164)：

```python
def _get_masked_input_and_mask(self, input_, ...):
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    
    # Adapt: 当 added_vocab 为空时，避免创建无用的 added_vocab_mask
    if added_vocab_start_index == added_vocab_end_index:
        valid_offset = org_vocab_start_index * org_vocab_mask
        vocab_mask = org_vocab_mask
    else:
        added_vocab_mask = (input_ >= added_vocab_start_index) & (input_ < added_vocab_end_index)
        added_offset = (added_vocab_start_index - (org_vocab_end_index - org_vocab_start_index) 
                        - num_org_vocab_padding)
        valid_offset = (org_vocab_start_index * org_vocab_mask) + (added_offset * added_vocab_mask)
        vocab_mask = org_vocab_mask | added_vocab_mask
    
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask
```

**优化点**：当模型没有 added vocabulary（如 Qwen3-4B 默认配置），跳过 `added_vocab_mask` 的创建，减少不必要的张量运算。

---

### 五、LM Head（`AscendParallelLMHead`）

Qwen3-4B 的 `lm_head` 层使用 `AscendParallelLMHead`，其核心调用是 `quant_method.apply()`：

- **非量化**：`F.linear(hidden_states, weight, bias)` → 映射到 aclnnMatmul
- **lmhead TP 模式**：先 AllGather hidden_states → 本地 `F.linear` → AllToAll logits
- **普通模式**：本地 `F.linear` → `_gather_logits` (AllGather)

---

### 六、总结调用链

```
Qwen3-4B forward
  └─ model.embed_tokens(input_ids)
       └─ AscendVocabParallelEmbedding.forward(input_ids)
            ├─ [Embedding TP 模式]
            │    ├─ all_gather(input_ids)           ← HCCL
            │    ├─ _get_masked_input_and_mask()    ← PyTorch 张量操作
            │    ├─ quant_method.embedding()        ← torch.nn.functional.embedding → aclnnEmbedding
            │    ├─ masked_fill_()                  ← PyTorch
            │    └─ reduce_scatter()                ← HCCL
            │
            └─ [普通 TP 模式]
                 ├─ _get_masked_input_and_mask()    ← PyTorch 张量操作
                 ├─ quant_method.embedding()        ← torch.nn.functional.embedding → aclnnEmbedding
                 ├─ masked_fill_()                  ← PyTorch
                 └─ torch.ops.vllm.maybe_pad_and_reduce()  ← vLLM custom op → HCCL AllReduce
```

**核心结论**：Qwen3-4B 的 Embedding 算子本质上调用的是 **`torch.nn.functional.embedding()`**，在 NPU 上由 torch_npu 自动下发到 **aclnnEmbedding** 算子。vllm-ascend 的定制工作主要集中在 **TP 并行通信策略**（Embedding TP / lmhead TP）和 **掩码计算优化**，而非替换底层 Embedding 查表算子本身。

