# Ascend NPU 算子接入实施方案

## 1. 概述

本文档描述在 rtp-llm 代码仓中接入 Ascend NPU 算子的完整实施方案。核心原则是：**优先使用 `torch`、`torch_npu` 提供的原生 API**，避免引入自定义 kernel 编译。方案参考了现有 ROCm 后端的实现模式。

---

## 2. 现有架构分析

### 2.1 三层架构

```
base/   → 基础算子，按设备类型静态分发
factory/ → 工厂模式，按设备和配置动态选择策略
hybrid/ → 高层组合，无设备特异性
```

### 2.2 设备分发机制

| 层 | 分发方式 | 判断依据 |
|---|---|---|
| `base/__init__.py` | import-time `if/else` | `DeviceType.ROCm` vs CUDA（else 分支） |
| `factory/attention/__init__.py` | import-time `if/elif` | `DeviceType.ROCm` vs `DeviceType.Cuda` |
| `factory/linear/__init__.py` | import-time `if/else` | `DeviceType.ROCm` vs CUDA/其他 |
| `factory/fused_moe/__init__.py` | import-time `if/else` | `DeviceType.ROCm` vs CUDA/其他 |

**当前问题**: `base/__init__.py` 只有 ROCm 和 else（CUDA）两个分支。Ascend 设备会走到 else 分支加载 CUDA 代码，这会导致 `ImportError` 或运行时错误。

### 2.3 设备检测现状

`rtp_llm/device/device_type.py` **已经支持 Ascend 检测**：

```python
class DeviceType(IntEnum):
    Ascend = 6

def get_device_type() -> DeviceType:
    if torch.cuda.is_available():
        # ... CUDA / ROCm / PPU ...
        return DeviceType.Cuda
    try:
        import torch_npu
        if torch.npu.is_available():
            return DeviceType.Ascend
    except ImportError:
        pass
    return DeviceType.Cpu
```

`rtp_llm/device/__init__.py` 已支持 `AscendImpl`。
`rtp_llm/device/device_impl.py` 已有 `AscendImpl(GpuImpl)` 基础实现。

### 2.4 当前 CUDA/ROCm 后端对比

以 ROCm 作为参考模板，每个算子有三种实现：

| 算子 | CUDA | ROCm |
|---|---|---|
| `FusedSiluAndMul` | `rtp_llm_ops.silu_and_mul` | `aiter.silu_and_mul` |
| `RMSNorm` | `rtp_llm_ops.rmsnorm` | `aiter.rms_norm` |
| `RMSResNorm` | `rtp_llm_ops.fused_add_rmsnorm` | `aiter.rmsnorm2d_fwd_with_add` |
| `AddBiasResLayerNorm` | `rtp_llm_ops.fused_add_layernorm` | `aiter.layernorm2d_fwd` 或 `rtp_llm_ops` |
| `SigmoidGateScaleAdd` | Triton kernel | Pure PyTorch |
| `SelectTopk` | `compute_ops.SelectTopkOp` | `aiter.topk_softmax` |
| `GroupTopK` | `compute_ops.GroupTopKOp` | **NotImplementedOp** |
| `FakeBalanceExpert` | `compute_ops.FakeBalanceExpertOp` | **NotImplementedOp** |
| `IndexerOp` | 完整 CUDA 类（deep_gemm, flashinfer） | **NotImplementedOp** |

---

## 3. 实现路线图

### Phase 1: 基础架构改造（设备分发层）

#### 3.1 `base/__init__.py` 增加 Ascend 分支

```python
if device_type == DeviceType.ROCm:
    # ... existing ROCm imports ...
elif device_type == DeviceType.Ascend:
    from rtp_llm.models_py.modules.base.ascend.activation import FusedSiluAndMul
    from rtp_llm.models_py.modules.base.ascend.moe_gating import SigmoidGateScaleAdd
    from rtp_llm.models_py.modules.base.ascend.norm import (
        AddBiasResLayerNorm, FusedQKRMSNorm, QKRMSNorm, RMSNorm, RMSResNorm,
    )
    from rtp_llm.models_py.modules.base.ascend.not_implemented_ops import (
        FakeBalanceExpert, GroupTopK, IndexerOp,
    )
    from rtp_llm.models_py.modules.base.ascend.select_topk import SelectTopk
else:
    # ... existing CUDA imports ...
```

#### 3.2 工厂层分发修改

对以下三个文件增加 `DeviceType.Ascend` 分支：

| 文件 | 现有分支 |
|---|---|
| `factory/attention/__init__.py` | ROCm / CUDA 两分支 |
| `factory/linear/__init__.py` | ROCm / else 两分支 |
| `factory/fused_moe/__init__.py` | ROCm / else 两分支 |

### Phase 2: base/ascend/ 核心算子实现

在 `modules/base/ascend/` 目录下创建 Ascend 专用算子包。**所有算子优先使用 PyTorch 原生 API 和 torch_npu API，不引入自定义 C++ kernel。**

#### 2.1 目录结构

```
base/ascend/
├── __init__.py
├── activation.py         # FusedSiluAndMul
├── norm.py               # RMSNorm, RMSResNorm, AddBiasResLayerNorm, QKRMSNorm, FusedQKRMSNorm, LayerNorm
├── moe_gating.py         # SigmoidGateScaleAdd
├── select_topk.py        # SelectTopk
├── not_implemented_ops.py # GroupTopK, FakeBalanceExpert, IndexerOp stub
└── test/
    └── ... (与 cuda/test/ 和 rocm/test/ 结构一致)
```

#### 2.2 各算子的推荐实现方案

| 算子 | 实现方案 | 优先级 | 说明 |
|---|---|---|---|
| **FusedSiluAndMul** | `F.silu(gate) * up` 或 `torch_npu.npu_silu` | P0 | 核心 gated MLP 算子 |
| **RMSNorm** | `torch_npu.npu_rms_norm` 或手动计算 | P0 | 核心 normalize 算子 |
| **RMSResNorm** | `hidden + residual` → `RMSNorm` | P0 | 融合残差 + RMSNorm |
| **AddBiasResLayerNorm** | `hidden + bias + residual` → `F.layer_norm` | P0 | 融合加法 + LayerNorm |
| **QKRMSNorm** | 组合 `RMSNorm` | P0 | Q/K 分别做 RMSNorm |
| **FusedQKRMSNorm** | `torch_npu.npu_rms_norm` 逐 head 计算 | P0 | 融合 Q/K RMSNorm |
| **LayerNorm** | `F.layer_norm` | P0 | 标准 LayerNorm |
| **SigmoidGateScaleAdd** | `experts += torch.sigmoid(gate) * shared` | P0 | MoE gating，纯 PyTorch |
| **SelectTopk** | `torch.topk` + `F.softmax` | P0 | MoE TopK 选择 |
| **GroupTopK** | NotImplementedOp | P2 | 高级 MoE 特性 |
| **FakeBalanceExpert** | NotImplementedOp | P2 | 负载均衡 |
| **IndexerOp** | NotImplementedOp | P2 | DeepSeek DSA |

以下为各算子的参考实现：

##### `activation.py` - FusedSiluAndMul

```python
class FusedSiluAndMul(SiluAndMulBase):
    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up
```

##### `norm.py` - RMSNorm

```python
class RMSNorm(BaseNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.data * hidden_states).to(hidden_states.dtype)
```

或者使用 `torch_npu` 的优化实现（性能更优，需验证可用性）：

```python
class RMSNorm(BaseNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from torch_npu.npu import rms_norm
        return rms_norm(hidden_states, self.weight.data, self.variance_epsilon)[0]
```

##### `norm.py` - RMSResNorm

```python
class RMSResNorm(BaseResNorm):
    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        output = hidden_states + residual
        residual = output
        output = self.rms_norm(output)
        return output
```

##### `norm.py` - AddBiasResLayerNorm

```python
class AddBiasResLayerNorm(BaseAddBiasResLayerNorm):
    def forward(self, hidden_states, residual, bias):
        hidden_states = hidden_states + bias + residual
        return torch.nn.functional.layer_norm(
            hidden_states, hidden_states.shape[-1:], self.weight.data, self.beta, self.variance_epsilon)
```

##### `moe_gating.py` - SigmoidGateScaleAdd

```python
class SigmoidGateScaleAdd(nn.Module):
    def forward(self, gate, shared, experts):
        experts.add_(torch.sigmoid(gate) * shared)
        return experts
```

##### `select_topk.py` - SelectTopk

```python
class SelectTopk(nn.Module):
    def __init__(self, config):
        self.config = config

    def forward(self, router_logits_fp32, topk_ids, topk_weights):
        topk_weights[:] = torch.softmax(router_logits_fp32.float(), dim=-1)
        topk_ids[:] = torch.topk(router_logits_fp32, self.config.moe_k, dim=-1).indices
```

##### `not_implemented_ops.py`

```python
from rtp_llm.models_py.modules.base.not_implemented import NotImplementedOp

class GroupTopK(NotImplementedOp):
    def __init__(self, *args, **kwargs):
        super().__init__(op_name="GroupTopK", device_type="Ascend")

class FakeBalanceExpert(NotImplementedOp):
    def __init__(self, *args, **kwargs):
        super().__init__(op_name="FakeBalanceExpert", device_type="Ascend")

class IndexerOp(NotImplementedOp):
    def __init__(self, *args, **kwargs):
        super().__init__(op_name="IndexerOp", device_type="Ascend")
```

### Phase 3: 工厂层适配

#### 3.1 Linear 工厂

**`factory/linear/__init__.py`** 增加 Ascend 分支：

```python
if device_type == DeviceType.ROCm:
    import rtp_llm.models_py.modules.factory.linear.impl.rocm
elif device_type == DeviceType.Ascend:
    import rtp_llm.models_py.modules.factory.linear.impl.ascend
else:
    import rtp_llm.models_py.modules.factory.linear.impl.cuda
```

**`factory/linear/impl/ascend/`** 需要实现的内容：

| Strategy 类 | 用途 | 推荐实现 |
|---|---|---|
| `AscendF16Linear` | BF16/FP16 非量化线性 | `torch.nn.functional.linear` |
| `AscendFp8PTPCLinear` | FP8 量化线性 | `torch_npu.npu_fp8_gemm` 或 NotImplementedOp |

**`impl/ascend/f16_linear.py`** 参考实现：

```python
class AscendF16Linear(LinearBase):
    @classmethod
    def can_handle(cls, quant_config, weight, weight_scales, **kwargs):
        return weight_scales is None  # 非量化

    def __init__(self, weight, **kwargs):
        super().__init__()
        self.weight = weight

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight)
```

#### 3.2 Attention 工厂

**`factory/attention/__init__.py`** 增加 Ascend 分支：

```python
if device_type == DeviceType.ROCm:
    # ... existing ROCm attention impls ...
elif device_type == DeviceType.Ascend:
    # Ascend 使用 PyTorch 原生 flash attention 或 torch_npu 实现
    PREFILL_MHA_IMPS.append(...)
    DECODE_MHA_IMPS.append(...)
else:
    # ... existing CUDA attention impls ...
```

推荐方案：

| 模式 | 推荐实现 | 说明 |
|---|---|---|
| Prefill MHA | `torch.nn.functional.scaled_dot_product_attention` | PyTorch 2.0+ 原生支持，Ascend NPU 已适配 |
| Decode MHA | Paged attention via torch_npu API | 需 `torch_npu` 支持 paged attention |
| MLA | NotImplementedOp（初始阶段） | 高级特性，后续阶段适配 |

**`attention/ascend_impl/torch_sdpa.py`** 参考实现：

```python
class AscendSDPAPrefillImpl(FMHAImplBase):
    @staticmethod
    def support(attn_configs, attn_inputs):
        return attn_inputs.is_prefill

    def forward(self, qkv, kv_cache, layer_idx=0):
        q, k, v = qkv.chunk(3, dim=-1)
        # 根据 position 信息构建 attention mask
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return output
```

#### 3.3 FusedMoE 工厂

**`factory/fused_moe/__init__.py`** 增加 Ascend 分支：

```python
if device_type == DeviceType.ROCm:
    # ... existing ROCm MoE strategies ...
elif device_type == DeviceType.Ascend:
    # Ascend MoE: 纯 PyTorch 实现 + BatchedTritonStrategy
    registry = StrategyRegistry()
    registry.register(batched_triton_or_pytorch_strategy)
    FusedMoeFactory.set_registry(registry)
else:
    # ... existing CUDA MoE strategies ...
```

初始阶段推荐注册一个基于纯 PyTorch 的 fallback MoE strategy。

### Phase 4: Hybrid 层适配

Hybrid 层（`causal_attention.py`, `dense_mlp.py`, `mla_attention.py`）本身没有设备特异性代码。但其中 `causal_attention.py` 有一处设备判断：

```python
device_type = get_device_type()
if device_type == DeviceType.ROCm:
    from rtp_llm.models_py.modules.base.rocm.norm import FusedQKRMSNorm
else:
    from rtp_llm.models_py.modules.base.cuda.norm import FusedQKRMSNorm
```

需要修改为：

```python
if device_type == DeviceType.ROCm:
    from rtp_llm.models_py.modules.base.rocm.norm import FusedQKRMSNorm
elif device_type == DeviceType.Ascend:
    from rtp_llm.models_py.modules.base.ascend.norm import FusedQKRMSNorm
else:
    from rtp_llm.models_py.modules.base.cuda.norm import FusedQKRMSNorm
```

---

## 4. 工作分解与排期

| Phase | 工作项 | 涉及文件 | 工作量评估 |
|---|---|---|---|
| **P1** | 基础架构改造 | `base/__init__.py`, `factory/attention/__init__.py`, `factory/linear/__init__.py`, `factory/fused_moe/__init__.py` | 小 |
| **P1** | FusedSiluAndMul 实现 | `base/ascend/activation.py` | 小 |
| **P1** | RMSNorm, LayerNorm 实现 | `base/ascend/norm.py` | 小 |
| **P1** | RMSResNorm 实现 | `base/ascend/norm.py` | 小 |
| **P1** | AddBiasResLayerNorm 实现 | `base/ascend/norm.py` | 小 |
| **P1** | SigmoidGateScaleAdd 实现 | `base/ascend/moe_gating.py` | 极小 |
| **P1** | SelectTopk 实现 | `base/ascend/select_topk.py` | 小 |
| **P1** | NotImplementedOp 存根 | `base/ascend/not_implemented_ops.py` | 极小 |
| **P1** | Linear 工厂 Ascend 策略 | `factory/linear/impl/ascend/` | 中 |
| **P2** | CMakeLists/BUILD 添加 ascend 后端编译 | 构建系统 | 中 |
| **P2** | Attention 工厂 Ascend 实现 | `factory/attention/ascend_impl/` | 大 |
| **P3** | FusedMoE 工厂 Ascend 策略 | `factory/fused_moe/impl/ascend/` | 大 |
| **P3** | Attention hybrid 层设备分发 | `hybrid/causal_attention.py` | 极小 |
| **P4** | QKRMSNorm, FusedQKRMSNorm | `base/ascend/norm.py` | 小 |
| **P4** | Qwen3 gate 适配 | `hybrid/causal_attention.py` | 小 |
| **P4** | FP8 量化支持 | `factory/linear/impl/ascend/fp8_linear.py` | 大 |
| **P5** | GroupTopK, FakeBalanceExpert, IndexerOp | 高级特性 | 待评估 |
| **P5** | Paged Attention | 推理优化 | 待评估 |

---

## 5. 关键技术决策

### 5.1 为什么优先使用 torch/torch_npu API

- **零编译依赖**：不需要为 Ascend 编写自定义 CUDA kernel 或 C++ extension
- **天然兼容**：`torch_npu` 已经实现 Ascend NPU 上 `torch.*` API 的语义兼容
- **维护成本低**：纯 Python 实现，无需跨 C++/Python 调试
- **与 ROCm 策略一致**：ROCm 已经证明了 "第三方库 + 纯 PyTorch fallback" 模式可行

### 5.2 第一阶段目标

**能跑通基础推理链路**。对于非核心算子（GroupTopK, IndexerOp 等），使用 `NotImplementedOp` 暂时跳过。以下模型可作为验证目标：

1. **Qwen2.5**（标准 MHA + Gated MLP + RMSNorm）— 最基础的 transformer 结构
2. **Qwen2.5 MoE**（路由 MoE，需 SelectTopk + FusedSiluAndMul）

### 5.3 性能敏感算子的优化路径

当纯 PyTorch 实现成为性能瓶颈时，按以下优先级优化：

| 优先级 | 优化方案 | 示例 |
|---|---|---|
| 1 | `torch_npu` 融合算子 API | `torch_npu.npu_rms_norm`, `torch_npu.npu_fusion_attention` |
| 2 | `torch.compile` | 对计算图整体编译优化 |
| 3 | Triton on Ascend | 如果平台支持 Triton 前端 |
| 4 | 自定义算子（CANN Ascend C库） | 终极优化手段 |

---

## 6. 修改文件清单

### 6.1 需修改的现有文件

| 文件 | 修改内容 |
|---|---|
| `modules/base/__init__.py` | 增加 `elif device_type == DeviceType.Ascend` 分支 |
| `modules/factory/attention/__init__.py` | 增加 Ascend 分支注册 attention impl |
| `modules/factory/linear/__init__.py` | 增加 Ascend 分支注册 linear strategy |
| `modules/factory/fused_moe/__init__.py` | 增加 Ascend 分支注册 MoE strategy |
| `hybrid/causal_attention.py` | Ascend 分支引入 `FusedQKRMSNorm` |
| `DEVELOP.md` | 更新目录结构和架构说明 |

### 6.2 需新建的文件

```
modules/base/ascend/
├── __init__.py                    # package marker
├── activation.py                  # FusedSiluAndMul
├── norm.py                        # RMSNorm, RMSResNorm, AddBiasResLayerNorm, QKRMSNorm, FusedQKRMSNorm, LayerNorm
├── moe_gating.py                  # SigmoidGateScaleAdd
├── select_topk.py                 # SelectTopk
├── not_implemented_ops.py         # GroupTopK, FakeBalanceExpert, IndexerOp
└── test/
    ├── __init__.py
    ├── norm_test.py
    ├── rmsnorm_test.py
    └── activation_test.py

modules/factory/linear/impl/ascend/
├── __init__.py                    # 注册 Ascend Linear strategies
└── f16_linear.py                  # AscendF16Linear

modules/factory/attention/ascend_impl/
├── __init__.py
└── torch_sdpa.py                  # AscendSDPAPrefillImpl (if needed)

modules/factory/fused_moe/impl/ascend/strategy/
├── __init__.py
└── pytorch_fallback.py            # Ascend fallback MoE strategy
```

---

## 7. 验证方案

### 7.1 单元测试

对每个算子，编写与 CUDA/ROCm 对应的单元测试：

| 测试文件 | 测试内容 |
|---|---|
| `test/ascend_norm_test.py` | RMSNorm vs RMSNormTorch，allclose 验证 |
| `test/ascend_activation_test.py` | FusedSiluAndMul vs 纯 PyTorch 实现 |
| `test/ascend_select_topk_test.py` | SelectTopk vs torch.topk + softmax |
| `test/ascend_moe_gating_test.py` | SigmoidGateScaleAdd vs 参考实现 |

### 7.2 集成测试

- 使用 `Qwen2.5-1.5B` 模型验证完整推理链路
- 对比 Ascend NPU 与 CUDA 输出的 logits 差异（允许数值误差）
- 验证 TP 通信（all-reduce）在 Ascend NPU 上的正确性

---

## 8. 风险与注意事项

1. **`torch_npu` API 覆盖度**：需要确认 `torch_npu` 版本支持 `scaled_dot_product_attention`、`rms_norm` 等关键 API。
2. **数值精度**：Ascend NPU 的 FP16/BF16 数值行为可能与其他平台不同，需要关注 allclose 的 atol/rtol 阈值设置。
3. **Page Attention**：当前主流推理框架在 Ascend 上使用 PagedAttention 的实现需要额外适配。初始阶段可先用 contiguous 注意力。
4. **FP8 支持**：Ascend NPU 的 FP8 格式可能与 NVIDIA 不同（Ascend 使用 `float8_e4m3fn`），量化工具链需要确认兼容性。
5. **`compute_ops` 依赖**：部分 common 层的算子（如 `Embedding`, `LayerNorm` 在 `common/norm.py` 中）会调用 `rtp_llm_ops.*`，这在 Ascend 上需要对应实现绑定或降级为 PyTorch。
