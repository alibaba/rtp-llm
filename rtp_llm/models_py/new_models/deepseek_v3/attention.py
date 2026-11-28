"""
MLA attention for DeepSeek V3.2, new-loader style.

Key design decisions:
  - __init__ creates independent nn.Module submodules named with HF ckpt keys
    (q_a_proj, kv_a_proj_with_mqa, q_a_layernorm, kv_a_layernorm, q_b_proj, o_proj).
    HF weights flow through RtpModule.load_weights directly into nn.Parameter.
  - process_weights_after_loading() fuses q_a_proj + kv_a_proj_with_mqa into
    _fused_qkv_a_w and splits kv_b_proj into the absorb/decode views.
  - _build_mla_kernel_weights() exposes only the kv_b/kc/vc tensors consumed
    by the MLA backend; Q and O projections execute through this module.
  - forward() mirrors MlaAttention.forward() exactly; the Indexer is a
    separate submodule built by the DecoderLayer when is_sparse is True.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.fp8 import (
    Fp8LinearMethod,
    Fp8PerChannelLinearMethod,
    _resolve_fp8_gemm_nt,
    _resolve_requant_weight_ue8m0,
    _resolve_sgl_per_token_group_quant,
    is_deep_gemm_e8m0_used,
)

if TYPE_CHECKING:
    from rtp_llm.models_py.modules.factory.attention.attn_factory import MlaImplBase
    from rtp_llm.ops.compute_ops import LayerKVCache


class _CudaRuntimeFusedFp8Linear(nn.Module):
    """CUDA-only fused view over independently loaded FP8 projections."""

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        block_size: tuple[int, int],
    ):
        super().__init__()
        if len(block_size) != 2 or any(size <= 0 for size in block_size):
            raise ValueError(f"invalid FP8 block size {block_size!r}")
        self.register_buffer("weight", weight.contiguous(), persistent=False)
        self.register_buffer(
            # DeepGEMM packs UE8M0 into an MN-major, TMA-aligned view.
            # Row-major contiguous() would destroy its column stride.
            "weight_scale",
            (
                weight_scale
                if weight_scale.dtype == torch.int32
                else weight_scale.contiguous()
            ),
            persistent=False,
        )
        self.block_size = block_size
        self._fp8_logical_output_size = weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_2d = x.view(-1, x.shape[-1]).contiguous()
        scale_ue8m0 = self.weight_scale.dtype == torch.int32
        qinput, input_scales = _resolve_sgl_per_token_group_quant()(
            input_2d,
            group_size=self.block_size[1],
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=scale_ue8m0,
        )
        output = torch.empty(
            input_2d.shape[0],
            self.weight.shape[0],
            dtype=torch.bfloat16,
            device=input_2d.device,
        )
        _resolve_fp8_gemm_nt()(
            (qinput, input_scales),
            (self.weight, self.weight_scale),
            output,
            c=None,
            disable_ue8m0_cast=not scale_ue8m0,
        )
        return output.view(*x.shape[:-1], self.weight.shape[0])


class _MlaInt8Projection(nn.Module):
    """A runtime projection owns the same packed layout as other W8A16 linears."""

    def __init__(self, weight):
        super().__init__()
        from rtp_llm.models_py.quant_methods.weight_only_int8 import (
            pack_weight_only_int8,
        )

        self.output_size = weight.shape[0]
        packed, scale = pack_weight_only_int8(weight)
        self.register_buffer("weight", packed, persistent=False)
        self.register_buffer("weight_scale", scale, persistent=False)

    def forward(self, x):
        from rtp_llm.models_py.quant_methods.weight_only_int8 import (
            apply_packed_weight_only_int8,
        )

        return apply_packed_weight_only_int8(
            x, self.weight, self.weight_scale, self.output_size
        )


class _MixedMlaAProjection(nn.Module):
    """Honor a quantization exclusion that only applies to one A projection."""

    def __init__(self, q, kv, quantize_q):
        super().__init__()
        self.int8 = _MlaInt8Projection(q if quantize_q else kv)
        self.register_buffer("floating", kv if quantize_q else q, persistent=False)
        self.quantize_q = quantize_q

    def forward(self, x):
        integer = self.int8(x)
        floating = torch.nn.functional.linear(x, self.floating)
        return torch.cat(
            (integer, floating) if self.quantize_q else (floating, integer), dim=-1
        )


class _MlaKernelWeights(dict):
    """Tensor view for MLA kernels, with an optional module-owned prefill projection."""

    def __init__(self, kv_b_projection):
        super().__init__()
        self.kv_b_projection = kv_b_projection


def _dequant_fp8_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Materialize a floating MLA projection in the model's compute dtype."""
    n, k = weight.shape
    # The legacy loader rounds both operands through the compute dtype before
    # multiplying in FP32. This matters even when the final dtype is FP16.
    w = weight.to(dtype).float()
    s = scale.to(dtype).float()
    if s.numel() == 1:
        s = s.reshape(1, 1)
    elif tuple(s.shape) in ((n,), (n, 1), (1, n)):
        s = s.reshape(n, 1)
    elif s.dim() == 2:
        block_n, block_k = block_size
        expected = (math.ceil(n / block_n), math.ceil(k / block_k))
        if tuple(s.shape) != expected:
            raise ValueError(
                f"FP8 block scale shape must be {expected} for weight "
                f"{tuple(weight.shape)}, got {tuple(scale.shape)}"
            )
        s = s.repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)
        s = s[:n, :k]
    else:
        raise ValueError(
            f"Unsupported FP8 scale shape {tuple(scale.shape)} for "
            f"weight {tuple(weight.shape)}"
        )
    return (w * s).to(dtype)


def _is_fp8_block_scale(linear: LinearBase, scale: torch.Tensor) -> bool:
    if scale.dim() != 2:
        return False
    block_n, block_k = linear.fp8_scale_block_size()
    expected = (
        math.ceil(linear.weight.shape[0] / block_n),
        math.ceil(linear.weight.shape[1] / block_k),
    )
    return tuple(scale.shape) == expected


def _floating_linear_weight(linear: LinearBase) -> torch.Tensor:
    """Return a linear's compute-dtype weight, dequantizing it when it is FP8.

    The floating fused A projection and the decode KC/VC matrices must
    match their activation dtype for F.linear and torch.bmm. Quantized
    projection kernels retain their own FP8 weights and full-precision scales.
    """
    w = linear.weight.data
    fp8_dtypes = (torch.float8_e4m3fn,)
    if hasattr(torch, "float8_e4m3fnuz"):
        fp8_dtypes += (torch.float8_e4m3fnuz,)
    if w.dtype not in fp8_dtypes:
        return w
    scale = _fp8_scale_parameter(linear)
    if scale is None:
        raise RuntimeError(f"fp8 linear {type(linear).__name__} is missing its scale")
    return _dequant_fp8_weight(
        w, scale.data, linear.fp8_scale_block_size(), linear.params_dtype
    )


def _fp8_scale_parameter(linear: LinearBase) -> Optional[nn.Parameter]:
    """Return the registered pre- or post-hook FP8 block scale.

    Fp8BlockLinearMethod deliberately renames the registered parameter from
    ``weight_scale_inv`` to ``weight_scale`` in its post-load hook. The MLA
    parent hook runs before child hooks, while kernel views are built after
    them, so both lifecycle states are part of the quant-method contract.
    Keep that dynamic protocol isolated here instead of scattering
    hasattr/getattr control flow across the attention implementation.
    """
    scale = linear._parameters.get("weight_scale")
    if scale is None:
        scale = linear._parameters.get("weight_scale_inv")
    return scale


def _release_parameter_storage(
    module: nn.Module,
    parameter_names: tuple[str, ...],
) -> None:
    """Replace checkpoint-only parameters with zero-sized device tensors."""
    parameters = dict(module.named_parameters(recurse=False))
    for name in parameter_names:
        parameter = parameters.get(name)
        if parameter is None:
            continue
        module.register_parameter(
            name,
            nn.Parameter(
                parameter.detach().new_empty(0),
                requires_grad=False,
            ),
        )


def _kernel_fp8_weight_and_scale(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the layout consumed by the legacy LinearFactory.

    FP32 block scales use the historical flattened-transpose view. DeepGEMM
    UE8M0 postprocessing instead stores an already runtime-ready ``[N, K]``
    weight and packed int32 ``[N, ceil(K / 512)]`` scales; reshaping either
    tensor would corrupt that contract.
    """
    if scale.dtype == torch.int32:
        return weight, scale
    if scale.dim() == 2:
        return (
            weight.reshape(weight.shape[1], weight.shape[0]),
            scale.reshape(scale.shape[1], scale.shape[0]),
        )
    return weight, scale


def _prepare_fused_fp8_runtime_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    ft_owner=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same UE8M0 conversion as the child FP8 linear hook.

    RtpModule post-load hooks visit this attention module before its child
    projections.  The fused QKV-A runtime view must therefore perform the
    device-specific conversion itself instead of caching the checkpoint's
    pre-requantized FP32 block scales.
    """
    if is_deep_gemm_e8m0_used(weight.device):
        if ft_owner is not None and getattr(ft_owner, "_packed_block_fp8_ft", None):
            from rtp_llm.models_py.quant_methods.block_fp8_ft import block_ft_runtime_pair

            return block_ft_runtime_pair(ft_owner, weight, scale, "weight_scale_inv", _resolve_requant_weight_ue8m0())
        return _resolve_requant_weight_ue8m0()(weight, scale)
    return weight, scale


def _quantize_online_mla_weight(source):
    """Quantize the floating MLA source once, directly into the chosen format."""
    from rtp_llm.models_py.kernels.cuda.fp8_quant import (
        per_block_cast_to_fp8,
        quant_weight_ue8m0_packed,
    )

    if is_deep_gemm_e8m0_used(source.device):
        return quant_weight_ue8m0_packed(source.contiguous())
    return per_block_cast_to_fp8(source, use_ue8m0=False)


class DeepSeekV32MlaAttention(RtpModule):
    """MLA attention for DeepSeek V3.2, new-loader style.

    HF ckpt keys consumed (per layer):
      model.layers.{i}.self_attn.q_a_proj.weight          → q_a_proj
      model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight → kv_a_proj_with_mqa
      model.layers.{i}.self_attn.q_a_layernorm.weight      → q_a_layernorm
      model.layers.{i}.self_attn.kv_a_layernorm.weight     → kv_a_layernorm
      model.layers.{i}.self_attn.q_b_proj.weight           → q_b_proj
      model.layers.{i}.self_attn.kv_b_proj.weight          → kv_b_proj
      model.layers.{i}.self_attn.o_proj.weight             → o_proj

    process_weights_after_loading fuses q_a_proj + kv_a_proj_with_mqa into
    _fused_qkv_a_w and splits kv_b_proj into _kc_w (nope half) and _vc_w
    (v half) using the same transpose+slice formula as the legacy loader.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        nope_head_dim: int,
        rope_head_dim: int,
        v_head_dim: int,
        layer_idx: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: torch.dtype = torch.bfloat16,
        layernorm_eps: float = 1e-6,
        prefix: str = "self_attn",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads // tp_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.nope_head_dim = nope_head_dim
        self.rope_head_dim = rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_head_dim = nope_head_dim + rope_head_dim
        self.layer_idx = layer_idx
        self.tp_size = tp_size
        self.quant_config = quant_config
        self._online_fp8 = (
            quant_config is not None and quant_config.quant_type == "fp8_block_online"
        )
        self._online_int8 = (
            quant_config is not None and quant_config.quant_type == "INT8"
        )
        self._checkpoint_weights_released = False

        # The checkpoint stores the A projections as FP8 per-block tensors.
        # Preserve their scale parameters during streaming load. Postprocessing
        # builds BF16 kernel views while forward keeps the fused FP8 execution.
        # An unquantized allocation would silently drop weight_scale_inv and
        # cast the raw FP8 codes to BF16 without applying their scales.
        # Legacy MLA keeps A projections in BF16 for runtime/on-the-fly FP8.
        # Pre-quantized checkpoints still need their stored FP8 scales preserved.
        a_proj_quant_config = (
            quant_config
            if quant_config is not None
            and not self._online_int8
            and not quant_config.quant_type.endswith("_online")
            # The online FT writer quantizes compressed A, but its direct-Q
            # (no LoRA) fused record stays in the original compute dtype.
            and not (
                not q_lora_rank
                and getattr(quant_config, "ft_encoded_online_source", None)
                == "fp8_block_online"
            )
            else QuantizationConfig(quant_type="none")
        )

        # --- Independent submodules matching HF ckpt names ---
        # q_a_proj is either the LoRA down-projection (hidden -> q_lora_rank)
        # or, when q_lora_rank == 0, the direct query projection
        # (hidden -> num_heads * q_head_dim). The LoRA down-projection is
        # replicated, while direct Q is sharded by head across TP ranks.
        q_a_output_size = (
            q_lora_rank if q_lora_rank > 0 else num_heads * self.q_head_dim
        )
        q_a_tp_size = 1 if q_lora_rank > 0 else tp_size
        q_a_tp_rank = 0 if q_lora_rank > 0 else tp_rank
        self.q_a_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=q_a_output_size,
            tp_size=q_a_tp_size,
            tp_rank=q_a_tp_rank,
            quant_config=a_proj_quant_config,
            prefix=f"{prefix}.q_a_proj" if q_lora_rank > 0 else f"{prefix}.q_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        # kv_a_proj_with_mqa: hidden → (kv_lora_rank + rope_head_dim).
        # MQA shared kv latent + k_pe — REPLICATED across TP ranks (tp_size=1).
        self.kv_a_proj_with_mqa = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=kv_lora_rank + rope_head_dim,
            tp_size=1,
            tp_rank=0,
            quant_config=a_proj_quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
            bias=False,
            params_dtype=params_dtype,
        )
        self.q_a_layernorm = (
            RMSNorm(q_lora_rank, eps=layernorm_eps, params_dtype=params_dtype)
            if q_lora_rank > 0
            else None
        )
        self.kv_a_layernorm = RMSNorm(
            kv_lora_rank, eps=layernorm_eps, params_dtype=params_dtype
        )
        # q_b_proj exists only for Q-LoRA checkpoints.
        self.q_b_proj = (
            ColumnParallelLinear(
                input_size=q_lora_rank,
                output_size=num_heads * self.q_head_dim,
                tp_size=tp_size,
                tp_rank=tp_rank,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
                bias=False,
                params_dtype=params_dtype,
            )
            if q_lora_rank > 0
            else None
        )
        # kv_b_proj: kv_lora_rank → num_heads * (nope_head_dim + v_head_dim).
        # Per-head k_nope / v up-projection — SHARDED by head along the output
        # dim (column parallel), matching the legacy loader's head_num split.
        # Each head contributes (nope+v) contiguous rows, so a plain column
        # split lands whole heads on each rank; self.num_heads (= num_heads //
        # tp_size) then matches the loaded weight in the kc/vc derivation below.
        # Online FP8 leaves are quantized one at a time before parent post-load
        # hooks run. Keep kv_b in the checkpoint dtype until this attention
        # module has derived the exact kc/vc decode views; process_weights then
        # creates a separate FP8 runtime view for prefill on CUDA.
        kv_b_quant_config = (
            QuantizationConfig(quant_type="none")
            if self._online_fp8 or self._online_int8
            else quant_config
        )
        self.kv_b_proj = ColumnParallelLinear(
            input_size=kv_lora_rank,
            output_size=num_heads * (nope_head_dim + v_head_dim),
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=kv_b_quant_config,
            prefix=f"{prefix}.kv_b_proj",
            bias=False,
            params_dtype=params_dtype,
        )
        # These three linears are sources for attention-owned fused QKV-A and
        # KV-B/KC/VC. Replacing one source alone cannot update all consumers;
        # initialize() may discard its storage after building kernel views.
        # Do not publish them as independently mutable linears (or retain them
        # in the update manager). Q-B, O and the norms remain live owners.
        for source in (self.q_a_proj, self.kv_a_proj_with_mqa, self.kv_b_proj):
            source.supports_independent_weight_updates = False
        # o_proj: num_heads * v_head_dim → hidden
        self.o_proj = RowParallelLinear(
            input_size=num_heads * v_head_dim,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            bias=False,
            reduce_output=True,
            params_dtype=params_dtype,
        )

        # --- Fused weights (built after loading) ---
        self.register_buffer("_fused_qkv_a_w", None, persistent=False)
        self._fused_qkv_a_runtime: Optional[nn.Module] = None
        self._kv_b_int8: Optional[nn.Module] = None
        self.register_buffer("_kv_b_w", None, persistent=False)
        self.register_buffer("_kv_b_runtime_w", None, persistent=False)
        self.register_buffer("_kv_b_runtime_s", None, persistent=False)
        self.register_buffer("_kc_w", None, persistent=False)
        self.register_buffer("_vc_w", None, persistent=False)
        self.indexer: Optional[nn.Module] = None
        self.reuse_topk_indices = False

    def _projection_name(self, name: str) -> str:
        if self.q_lora_rank == 0 and name.startswith("q_proj."):
            return "q_a_proj." + name[len("q_proj.") :]
        return name

    def checkpoint_read_spec(self, name, shape):
        # This module owns a custom load_weights hook, so NewModelLoader stops
        # traversal here. Delegate to the actual leaf with the same alias the
        # writer uses (V2-Lite q_proj is TP-sharded, unlike replicated q_a).
        name = self._projection_name(name)
        first, sep, rest = name.partition(".")
        child = self._modules.get(first)
        if sep and isinstance(child, RtpModule):
            return child.resolve_checkpoint_read_spec(rest, shape)
        return None

    def load_weights(self, weights):
        if self._checkpoint_weights_released:
            raise RuntimeError(
                "MLA checkpoint weights were released after runtime layout "
                "construction; rebuild the model before loading new weights"
            )
        if self.q_lora_rank == 0:
            items = weights.items() if isinstance(weights, dict) else weights
            weights = ((self._projection_name(name), tensor) for name, tensor in items)
        return super().load_weights(weights)

    _derived_buffers = ("_fused_qkv_a_w", "_kv_b_w", "_kv_b_runtime_w",
                        "_kv_b_runtime_s", "_kc_w", "_vc_w")

    def prepare_resident_projection(self):
        # Match the ordinary parent-before-child hook: KC/VC and fused A must
        # be derived from the pre-quantized projections, including FT absorb.
        self.process_weights_after_loading()
        self._resident_runtime_buffers = self._derived_buffers
        for name in self._derived_buffers:
            value = getattr(self, name)
            if value is not None:
                self._buffers[name] = value.detach()
        for root in (self._fused_qkv_a_runtime, self._kv_b_int8):
            if root is not None:
                for module in root.modules():
                    module._resident_runtime_buffers = tuple(module._buffers)
        self._resident_mla_ready = True

    def resident_projection_facts(self):
        def spec(module):
            if module is None:
                return None
            if isinstance(module, _MixedMlaAProjection):
                return ["mixed", module.quantize_q, spec(module.int8)]
            if isinstance(module, _MlaInt8Projection):
                return ["w8", module.output_size]
            if isinstance(module, _CudaRuntimeFusedFp8Linear):
                return ["fp8", list(module.block_size)]
            raise TypeError(f"Unknown resident MLA projection {type(module)}")
        return [spec(self._fused_qkv_a_runtime), spec(self._kv_b_int8),
                bool(getattr(self, "_ft_absorb_weights", False))]

    def restore_resident_projection(self, state, values, prefix):
        def projection(spec, path):
            if spec is None:
                return None
            kind = spec[0]
            if kind == "fp8":
                module = _CudaRuntimeFusedFp8Linear(values[path+"weight"],
                    values[path+"weight_scale"], tuple(spec[1]))
            elif kind == "w8":
                # Already packed. The cold constructor quantizes a BF16 tensor.
                module = _MlaInt8Projection.__new__(_MlaInt8Projection)
                nn.Module.__init__(module)
                module.output_size = spec[1]
                for name in ("weight", "weight_scale"):
                    module.register_buffer(name, values[path+name], persistent=False)
            elif kind == "mixed":
                module = _MixedMlaAProjection.__new__(_MixedMlaAProjection)
                nn.Module.__init__(module)
                module.quantize_q = spec[1]
                module.int8 = projection(spec[2], path+"int8.")
                module.register_buffer("floating", values[path+"floating"], persistent=False)
            else:
                raise ValueError(f"Invalid resident MLA projection {kind!r}")
            module._resident_runtime_buffers = tuple(module._buffers)
            return module
        a, kv_b, ft_absorb = state
        self._fused_qkv_a_runtime = projection(a, prefix+"_fused_qkv_a_runtime.")
        self._kv_b_int8 = projection(kv_b, prefix+"_kv_b_int8.")
        self._ft_absorb_weights = ft_absorb
        self._resident_runtime_buffers = self._derived_buffers
        for name in self._derived_buffers:
            self._buffers[name] = values.get(prefix+name)
        if self._kc_w is None or self._vc_w is None or (
            self._fused_qkv_a_runtime is None and self._fused_qkv_a_w is None
        ):
            raise ValueError("Incomplete resident MLA projections")
        self._resident_mla_ready = True

    def process_weights_after_loading(self):
        """Fuse q_a_proj + kv_a_proj_with_mqa into a single _fused_qkv_a_w.

        Also split kv_b_proj into _kc_w (nope) and _vc_w (v) using the same
        formula as the legacy loader's transpose_slice_k / transpose_slice_v
        (utils/model_weight.py). q_b_proj stays in its quantized linear module;
        _build_mla_kernel_weights creates only a view over that storage.
        """
        if getattr(self, "_resident_mla_ready", False):
            return
        q_a_scale = _fp8_scale_parameter(self.q_a_proj)
        kv_a_scale = _fp8_scale_parameter(self.kv_a_proj_with_mqa)
        is_hip = torch.version.hip is not None
        block_n, _ = self.q_a_proj.fp8_scale_block_size()
        fused_a_block_aligned = self.q_a_proj.weight.shape[0] % block_n == 0
        if (
            self.q_lora_rank > 0
            and q_a_scale is not None
            and kv_a_scale is not None
            and not is_hip
            and fused_a_block_aligned
            and _is_fp8_block_scale(self.q_a_proj, q_a_scale)
            and _is_fp8_block_scale(self.kv_a_proj_with_mqa, kv_a_scale)
        ):
            fused_a_weight, fused_a_scale = _prepare_fused_fp8_runtime_weight(
                torch.cat(
                    [
                        self.q_a_proj.weight.detach(),
                        self.kv_a_proj_with_mqa.weight.detach(),
                    ],
                    dim=0,
                ),
                torch.cat(
                    [q_a_scale.detach(), kv_a_scale.detach()],
                    dim=0,
                ),
                ft_owner=self.q_a_proj,
            )
            self._fused_qkv_a_runtime = _CudaRuntimeFusedFp8Linear(
                fused_a_weight,
                fused_a_scale,
                self.q_a_proj.fp8_scale_block_size(),
            )
            self._fused_qkv_a_w = fused_a_weight
        else:
            q_a_w = _floating_linear_weight(self.q_a_proj)
            kv_a_w = _floating_linear_weight(self.kv_a_proj_with_mqa)
            quantize_q = self._online_int8 and not self.quant_config.is_layer_ignored(
                self.q_a_proj.prefix
            )
            quantize_kv = self._online_int8 and not self.quant_config.is_layer_ignored(
                self.kv_a_proj_with_mqa.prefix
            )
            fused_a = (
                torch.cat((q_a_w, kv_a_w)) if quantize_q == quantize_kv else None
            )
            if fused_a is None:
                self._fused_qkv_a_runtime = _MixedMlaAProjection(
                    q_a_w, kv_a_w, quantize_q
                )
            elif quantize_q:
                self._fused_qkv_a_runtime = _MlaInt8Projection(fused_a)
            elif self.q_lora_rank > 0 and not is_hip and self._online_fp8:
                fused_a_weight, fused_a_scale = _quantize_online_mla_weight(fused_a)
                self._fused_qkv_a_runtime = _CudaRuntimeFusedFp8Linear(
                    fused_a_weight,
                    fused_a_scale,
                    self.q_a_proj.fp8_scale_block_size(),
                )
                self._fused_qkv_a_w = fused_a_weight
            else:
                self._fused_qkv_a_w = fused_a

        # kv_b_proj weight: [num_heads * (nope + v_head), kv_lora_rank].
        # Reshape to [kv_lora_rank, num_heads, nope+v_head] then slice.
        kv_b_w = _floating_linear_weight(self.kv_b_proj)
        head_num = self.num_heads
        nope = self.nope_head_dim
        v_head = self.v_head_dim
        t = kv_b_w.transpose(0, 1).contiguous().view(self.kv_lora_rank, head_num, nope + v_head)
        if not getattr(self, "_ft_absorb_weights", False):
            self._kc_w = t[:, :, :nope].permute(1, 2, 0).contiguous()
            self._vc_w = t[:, :, nope:].transpose(0, 1).contiguous()
        # _kv_b_w: [kv_lora_rank, head_num * (nope + v_head)] — transposed kv_b
        # for floating FlashInfer prefill. Decode/absorb consumes the floating
        # checkpoint-derived kc/vc views above. CUDA online-FP8 builds a
        # separate quantized prefill view only after those derivations.
        kv_b_scale = _fp8_scale_parameter(self.kv_b_proj)
        if self._online_int8 and not self.quant_config.is_layer_ignored(
            self.kv_b_proj.prefix
        ):
            # KC/VC were derived from the checkpoint before this quantization.
            self._kv_b_int8 = _MlaInt8Projection(kv_b_w)
        elif self._online_fp8 and not is_hip:
            (
                self._kv_b_runtime_w,
                self._kv_b_runtime_s,
            ) = _quantize_online_mla_weight(kv_b_w)
            self._kv_b_w = None
        else:
            self._kv_b_w = (
                None
                if kv_b_scale is not None
                else t.view(self.kv_lora_rank, head_num * (nope + v_head)).contiguous()
            )

    def _build_mla_kernel_weights(self) -> Dict[str, torch.Tensor]:
        """Expose only the kv_b/kc/vc tensors consumed by MLA factories."""
        from rtp_llm.utils.model_weight import W

        if self._fused_qkv_a_w is None and self._fused_qkv_a_runtime is None:
            raise RuntimeError(
                "process_weights_after_loading() must be called before "
                "_build_mla_kernel_weights()"
            )
        weights = _MlaKernelWeights(self._kv_b_int8)
        # Prefill consumes kv_b through the attention factory. Preserve the
        # FP8 weight/scale pair so it selects the same block-quantized linear
        # as the legacy loader; kc/vc remain dequantized derived views for the
        # absorb/decode paths.
        kv_b_weight = self.kv_b_proj.weight.data
        kv_b_scale = _fp8_scale_parameter(self.kv_b_proj)
        if self._kv_b_int8 is not None:
            weights[W.mla_kv_b_w] = self._kv_b_int8.weight
            weights[W.mla_kv_b_s] = self._kv_b_int8.weight_scale
        elif self._kv_b_runtime_w is not None:
            if self._kv_b_runtime_s is None:
                raise RuntimeError("online-FP8 MLA kv_b runtime scale is missing")
            (
                weights[W.mla_kv_b_w],
                weights[W.mla_kv_b_s],
            ) = _kernel_fp8_weight_and_scale(
                self._kv_b_runtime_w,
                self._kv_b_runtime_s,
            )
        elif kv_b_scale is not None:
            if isinstance(
                self.kv_b_proj.quant_method, (Fp8LinearMethod, Fp8PerChannelLinearMethod)
            ):
                # Keep the calibrated input scale and backend-specific row
                # layout with the child that owns them. The legacy factory's
                # block layout cannot represent those contracts.
                weights.kv_b_projection = self.kv_b_proj
                weights[W.mla_kv_b_w] = kv_b_weight
                weights[W.mla_kv_b_s] = kv_b_scale.data
            else:
                (
                    weights[W.mla_kv_b_w],
                    weights[W.mla_kv_b_s],
                ) = _kernel_fp8_weight_and_scale(
                    kv_b_weight,
                    kv_b_scale.data,
                )
        else:
            if self._kv_b_w is None:
                raise RuntimeError("floating MLA kv_b runtime view was not constructed")
            weights[W.mla_kv_b_w] = self._kv_b_w
        weights[W.mla_kc] = self._kc_w
        weights[W.mla_vc] = self._vc_w
        return weights

    def release_checkpoint_only_weights(self) -> None:
        """Free projection parameters superseded by MLA runtime views.

        NewModelLoader validation and every child quantization post-load hook
        must finish before this method runs. The top-level model therefore
        calls it only after `_build_mla_kernel_weights()` has captured the
        runtime tensors consumed by attention factories. This transition is
        intentionally irreversible: weight refresh requires rebuilding the
        module so checkpoint-shaped parameters and quantization state exist.
        """
        if self._fused_qkv_a_w is None and self._fused_qkv_a_runtime is None:
            raise RuntimeError("MLA runtime weights must be built before release")
        parameter_names = ("weight", "weight_scale", "weight_scale_inv")
        _release_parameter_storage(self.q_a_proj, parameter_names)
        _release_parameter_storage(self.kv_a_proj_with_mqa, parameter_names)
        # BF16 and online-FP8 prefill consume derived runtime copies.
        # Pre-quantized FP8 consumes kv_b_proj weight/scale directly through
        # the kernel layout, so only that checkpoint storage must remain live.
        if _fp8_scale_parameter(self.kv_b_proj) is None:
            _release_parameter_storage(self.kv_b_proj, parameter_names)
        self._checkpoint_weights_released = True

    def _run_sparse_indexer(
        self,
        hidden_states: torch.Tensor,
        q_c: Optional[torch.Tensor],
        q_view: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        fmha_impl: MlaImplBase,
        previous_topk: Optional[torch.Tensor] = None,
        force_reuse_topk_indices: bool = False,
    ) -> Optional[torch.Tensor]:
        """Compute sparse top-k indices via the Indexer submodule.

        Mirrors legacy MlaAttention._run_sparse_indexer
        (modules/hybrid/mla_attention.py).  Returns None for dense layers
        (indexer not attached) so fmha_impl.forward gets a None and dense
        backends short-circuit; sparse backends require non-None.
        """
        from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention

        return MlaAttention._run_sparse_indexer(
            self,
            hidden_states,
            q_c,
            q_view,
            kv_cache,
            fmha_impl,
            prev_topk_indices=previous_topk,
            force_reuse_topk_indices=force_reuse_topk_indices,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: MlaImplBase,
        kv_cache: Optional[LayerKVCache] = None,
        *,
        previous_topk: Optional[torch.Tensor] = None,
        return_topk: bool = False,
        force_reuse_topk_indices: bool = False,
    ):
        input_shape = hidden_states.shape[:-1]
        q_c = None

        if self._fused_qkv_a_runtime is not None:
            fused_qkv_a = self._fused_qkv_a_runtime(hidden_states)
        elif self._fused_qkv_a_w is not None:
            fused_qkv_a = torch.nn.functional.linear(hidden_states, self._fused_qkv_a_w)
        else:
            raise RuntimeError("process_weights_after_loading() must run first")
        if self.q_lora_rank > 0:
            q, kv_a = torch.split(
                fused_qkv_a,
                [self.q_lora_rank, self.kv_lora_rank + self.rope_head_dim],
                dim=-1,
            )
            # split: q_a, then kv_a+rope
            compressed_kv = kv_a[..., : self.kv_lora_rank]
            k_pe = kv_a[..., self.kv_lora_rank :]
            # q_a layernorm
            if self.q_a_layernorm is None or self.q_b_proj is None:
                raise RuntimeError("incomplete MLA Q-LoRA modules")
            q_c = self.q_a_layernorm(q.contiguous())
            # q_b projection
            q = self.q_b_proj(q_c)
        else:
            q_offset = self.num_heads * self.q_head_dim
            q_output, kv_output = torch.split(
                fused_qkv_a,
                [q_offset, self.kv_lora_rank + self.rope_head_dim],
                dim=-1,
            )
            compressed_kv = kv_output[..., : self.kv_lora_rank]
            k_pe = kv_output[..., self.kv_lora_rank :]
            q = q_output

        q_view = q.reshape(-1, self.num_heads, self.q_head_dim)

        # kv_a layernorm
        compressed_kv = self.kv_a_layernorm(compressed_kv.contiguous())

        # Sparse Indexer (DSA) — runs only when self.indexer is attached
        # (DecoderLayer sets self.indexer when is_sparse=True).
        from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention

        default_impl, default_cache, indexer_impl, indexer_cache = (
            MlaAttention.resolve_layer_routes(
                self.indexer is not None or self.reuse_topk_indices,
                self.reuse_topk_indices,
                fmha_impl,
                kv_cache,
            )
        )
        topk_indices = self._run_sparse_indexer(
            hidden_states, q_c, q_view, indexer_cache, indexer_impl, previous_topk,
            force_reuse_topk_indices,
        )
        attn_output = default_impl.forward(
            q_view, compressed_kv, k_pe, default_cache, self.layer_idx, topk_indices
        )

        if attn_output is not None and attn_output.numel() != 0:
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        else:
            attn_output = torch.zeros(
                (*input_shape, self.num_heads * self.v_head_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        attn_output = self.o_proj(attn_output)
        return (attn_output, topk_indices) if return_topk else attn_output
