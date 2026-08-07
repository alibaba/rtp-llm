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

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.factory.attention.attn_factory import MlaImplBase
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.fp8 import (
    _resolve_fp8_gemm_nt,
    _resolve_requant_weight_ue8m0,
    _resolve_sgl_per_token_group_quant,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.ops.compute_ops import LayerKVCache
from rtp_llm.utils.model_weight import W


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
            "weight_scale", weight_scale.contiguous(), persistent=False
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


def _rounded_fp8_values(weight: torch.Tensor) -> torch.Tensor:
    # The legacy MLA kc/vc derivation loads both the FP8 codes and block
    # scales through the model compute dtype before dequantization.  Preserve
    # that rounding only for these kernel-facing BF16 views; the projection
    # modules themselves keep the original FP8 weight and FP32 scale.
    return weight.to(torch.bfloat16).to(torch.float32)


def _dequant_fp8_to_bf16(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int],
) -> torch.Tensor:
    """Dequantize per-tensor, per-channel, or per-block FP8 to BF16."""
    n, k = weight.shape
    w = _rounded_fp8_values(weight)
    s = scale.to(torch.bfloat16).to(torch.float32)
    if s.numel() == 1:
        return (w * s.reshape(1, 1)).to(torch.bfloat16)
    if s.dim() == 1 and s.shape[0] == n:
        return (w * s.reshape(n, 1)).to(torch.bfloat16)
    if s.dim() == 2 and tuple(s.shape) == (n, 1):
        return (w * s).to(torch.bfloat16)
    if s.dim() == 2 and tuple(s.shape) == (1, n):
        return (w * s.t()).to(torch.bfloat16)
    if s.dim() != 2:
        raise ValueError(
            f"Unsupported FP8 scale shape {tuple(scale.shape)} for "
            f"weight {tuple(weight.shape)}"
        )

    block_n, block_k = block_size
    expected = (math.ceil(n / block_n), math.ceil(k / block_k))
    if tuple(s.shape) != expected:
        raise ValueError(
            f"FP8 block scale shape must be {expected} for weight "
            f"{tuple(weight.shape)}, got {tuple(scale.shape)}"
        )
    s = s.repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)
    s = s[:n, :k]
    return (w * s).to(torch.bfloat16)


def _is_fp8_block_scale(linear: LinearBase, scale: torch.Tensor) -> bool:
    if scale.dim() != 2:
        return False
    block_n, block_k = linear.fp8_scale_block_size()
    expected = (
        math.ceil(linear.weight.shape[0] / block_n),
        math.ceil(linear.weight.shape[1] / block_k),
    )
    return tuple(scale.shape) == expected


def _linear_weight_bf16(linear: LinearBase) -> torch.Tensor:
    """Return a linear's weight as bf16, dequantizing it when it is FP8.

    Used by the MLA post-load derivations (fused qkv-a, kc/vc) which run
    through torch.cat / torch.bmm — neither supports fp8 — even though the
    forward projections themselves keep running the fp8 weights via DeepGEMM.
    Called from the attention module's post-load hook, which fires before the
    child linears' own hook (parent-before-child), so the scale is still under
    `weight_scale_inv`; `weight_scale` is checked too for robustness.
    """
    w = linear.weight.data
    fp8_dtypes = (torch.float8_e4m3fn,)
    if hasattr(torch, "float8_e4m3fnuz"):
        fp8_dtypes += (torch.float8_e4m3fnuz,)
    if w.dtype not in fp8_dtypes:
        return w
    scale = _fp8_scale_parameter(linear)
    if scale is None:
        raise RuntimeError(f"fp8 linear {type(linear).__name__} is missing block scale")
    return _dequant_fp8_to_bf16(w, scale.data, linear.fp8_scale_block_size())


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same UE8M0 conversion as the child FP8 linear hook.

    RtpModule post-load hooks visit this attention module before its child
    projections.  The fused QKV-A runtime view must therefore perform the
    device-specific conversion itself instead of caching the checkpoint's
    pre-requantized FP32 block scales.
    """
    if is_deep_gemm_e8m0_used(weight.device):
        return _resolve_requant_weight_ue8m0()(weight, scale)
    return weight, scale


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
            and not quant_config.quant_type.endswith("_online")
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
            QuantizationConfig(quant_type="none") if self._online_fp8 else quant_config
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
        self.register_buffer("_kv_b_w", None, persistent=False)
        self.register_buffer("_kv_b_runtime_w", None, persistent=False)
        self.register_buffer("_kv_b_runtime_s", None, persistent=False)
        self.register_buffer("_kc_w", None, persistent=False)
        self.register_buffer("_vc_w", None, persistent=False)
        self.indexer: Optional[nn.Module] = None

    def load_weights(self, weights):
        if self._checkpoint_weights_released:
            raise RuntimeError(
                "MLA checkpoint weights were released after runtime layout "
                "construction; rebuild the model before loading new weights"
            )
        if self.q_lora_rank == 0:
            items = weights.items() if isinstance(weights, dict) else weights
            weights = {
                (
                    "q_a_proj." + name[len("q_proj.") :]
                    if name.startswith("q_proj.")
                    else name
                ): tensor
                for name, tensor in items
            }
        return super().load_weights(weights)

    def process_weights_after_loading(self):
        """Fuse q_a_proj + kv_a_proj_with_mqa into a single _fused_qkv_a_w.

        Also split kv_b_proj into _kc_w (nope) and _vc_w (v) using the same
        formula as the legacy loader's transpose_slice_k / transpose_slice_v
        (utils/model_weight.py). q_b_proj stays in its quantized linear module;
        _build_mla_kernel_weights creates only a view over that storage.
        """
        # These derivations run through torch.cat / torch.bmm, which do not
        # support fp8, so dequantize the (possibly fp8-per-block) weights to
        # bf16 here. The forward projections still execute the fp8 weights via
        # the linear's DeepGEMM apply — this only affects the kc/vc + fused
        # views consumed by the MLA kernel.
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
            )
            self._fused_qkv_a_runtime = _CudaRuntimeFusedFp8Linear(
                fused_a_weight,
                fused_a_scale,
                self.q_a_proj.fp8_scale_block_size(),
            )
            self._fused_qkv_a_w = fused_a_weight
        else:
            q_a_w = _linear_weight_bf16(self.q_a_proj)
            kv_a_w = _linear_weight_bf16(self.kv_a_proj_with_mqa)
            fused_a_bf16 = torch.cat([q_a_w, kv_a_w], dim=0).contiguous()
            if (
                self.q_lora_rank > 0
                and not is_hip
                and self.quant_config is not None
                and self.quant_config.quant_type == "fp8_block_online"
            ):
                from rtp_llm.models_py.kernels.cuda.fp8_quant import (
                    per_block_cast_to_fp8,
                )

                fused_a_weight, fused_a_scale = per_block_cast_to_fp8(
                    fused_a_bf16,
                    use_ue8m0=False,
                )
                fused_a_weight, fused_a_scale = _prepare_fused_fp8_runtime_weight(
                    fused_a_weight, fused_a_scale
                )
                self._fused_qkv_a_runtime = _CudaRuntimeFusedFp8Linear(
                    fused_a_weight,
                    fused_a_scale,
                    self.q_a_proj.fp8_scale_block_size(),
                )
                self._fused_qkv_a_w = fused_a_weight
            else:
                self._fused_qkv_a_w = fused_a_bf16

        # kv_b_proj weight: [num_heads * (nope + v_head), kv_lora_rank].
        # Reshape to [kv_lora_rank, num_heads, nope+v_head] then slice.
        kv_b_w = _linear_weight_bf16(self.kv_b_proj)
        head_num = self.num_heads
        nope = self.nope_head_dim
        v_head = self.v_head_dim
        t = (
            kv_b_w.transpose(0, 1)
            .contiguous()
            .view(self.kv_lora_rank, head_num, nope + v_head)
        )
        # _kc_w shape: [head_num, nope, kv_lora_rank]
        self._kc_w = t[:, :, :nope].permute(1, 2, 0).contiguous()
        # _vc_w shape: [head_num, kv_lora_rank, v_head]
        self._vc_w = t[:, :, nope:].transpose(0, 1).contiguous()
        # _kv_b_w: [kv_lora_rank, head_num * (nope + v_head)] — transposed kv_b
        # for BF16 FlashInfer prefill. Decode/absorb always consumes the exact
        # BF16 checkpoint-derived kc/vc views above. CUDA online-FP8 builds a
        # separate quantized prefill view only after those derivations.
        kv_b_scale = _fp8_scale_parameter(self.kv_b_proj)
        if self._online_fp8 and not is_hip:
            from rtp_llm.models_py.kernels.cuda.fp8_quant import per_block_cast_to_fp8

            runtime_weight, runtime_scale = per_block_cast_to_fp8(
                kv_b_w,
                use_ue8m0=False,
            )
            (
                self._kv_b_runtime_w,
                self._kv_b_runtime_s,
            ) = _prepare_fused_fp8_runtime_weight(runtime_weight, runtime_scale)
            self._kv_b_w = None
        else:
            self._kv_b_w = (
                None
                if kv_b_scale is not None
                else t.view(self.kv_lora_rank, head_num * (nope + v_head)).contiguous()
            )

    def _build_mla_kernel_weights(self) -> Dict[str, torch.Tensor]:
        """Expose only the kv_b/kc/vc tensors consumed by MLA factories."""
        if self._fused_qkv_a_w is None:
            raise RuntimeError(
                "process_weights_after_loading() must be called before "
                "_build_mla_kernel_weights()"
            )
        weights: Dict[str, torch.Tensor] = {}
        # Prefill consumes kv_b through the attention factory. Preserve the
        # FP8 weight/scale pair so it selects the same block-quantized linear
        # as the legacy loader; kc/vc remain dequantized derived views for the
        # absorb/decode paths.
        kv_b_weight = self.kv_b_proj.weight.data
        kv_b_scale = _fp8_scale_parameter(self.kv_b_proj)
        if self._kv_b_runtime_w is not None:
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
            (
                weights[W.mla_kv_b_w],
                weights[W.mla_kv_b_s],
            ) = _kernel_fp8_weight_and_scale(
                kv_b_weight,
                kv_b_scale.data,
            )
        else:
            if self._kv_b_w is None:
                raise RuntimeError("BF16 MLA kv_b runtime view was not constructed")
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
        if self._fused_qkv_a_w is None:
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
    ) -> Optional[torch.Tensor]:
        """Compute sparse top-k indices via the Indexer submodule.

        Mirrors legacy MlaAttention._run_sparse_indexer
        (modules/hybrid/mla_attention.py).  Returns None for dense layers
        (indexer not attached) so fmha_impl.forward gets a None and dense
        backends short-circuit; sparse backends require non-None.
        """
        if self.indexer is None:
            return None
        q_for_indexer = q_c if self.q_lora_rank > 0 else q_view
        return self.indexer(
            hidden_states,
            q_for_indexer,
            kv_cache,
            fmha_impl.fmha_params,
            fmha_impl.attn_inputs,
            use_fast_path=not fmha_impl.is_sparse(),
            cp_params=fmha_impl.cp_params,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: MlaImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        q_c = None

        if self.q_lora_rank > 0:
            # The legacy path executes q_a + kv_a as one FP8 GEMM. Keep the
            # checkpoint-facing modules independent, but use their fused
            # runtime view so activation quantization and accumulation match.
            if self._fused_qkv_a_runtime is not None:
                fused_qkv_a = self._fused_qkv_a_runtime(hidden_states)
            else:
                if self._fused_qkv_a_w is None:
                    raise RuntimeError("process_weights_after_loading() must run first")
                fused_qkv_a = torch.nn.functional.linear(
                    hidden_states, self._fused_qkv_a_w
                )
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
            # Match the legacy no-LoRA path's single fused BF16 projection.
            if self._fused_qkv_a_w is None:
                raise RuntimeError("process_weights_after_loading() must run first")
            fused_qkv = torch.nn.functional.linear(hidden_states, self._fused_qkv_a_w)
            q_offset = self.num_heads * self.q_head_dim
            q_output, kv_output = torch.split(
                fused_qkv,
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
        topk_indices = self._run_sparse_indexer(
            hidden_states, q_c, q_view, kv_cache, fmha_impl
        )
        attn_output = fmha_impl.forward(
            q_view, compressed_kv, k_pe, kv_cache, self.layer_idx, topk_indices
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
        return attn_output
