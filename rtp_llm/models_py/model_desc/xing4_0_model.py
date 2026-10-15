"""Xing4.0 python execution model: GenericMoe MLA/MoE stack with mHC residual.

Layer choreography mirrors the HF reference (modeling_xing4_0.py
Xing4_0DecoderLayer.forward):

    residual = hidden                       # [T, hc_mult, H]
    x_pre, post, comb = attn_hc.pre(residual)
    x_pre = input_layernorm(x_pre)          # plain RMSNorm on the collapsed stream
    attn_out = MLA(x_pre)
    hidden = attn_hc.post(attn_out, residual, post, comb)
    ... same for ffn_hc / post_attention_layernorm / mlp ...

The model expands the embedding to [T, hc_mult, H] and reduces the streams
with mean(dim=hc) before the final norm (the reference has no learned hc head).
The mHC mixer math is Xing4-specific and differs from the DeepSeek-V4 units:
no +eps on pre, exp(clamp(logits, +-h_res) - rowmax) + (row+eps, col+eps)
sinkhorn rounds, comb applied untransposed, clamps read from the model config.
"""

import logging
import os
from typing import Any, Optional, Tuple

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_fmha_impl_for_layer
from rtp_llm.models_py.model_desc.generic_moe import (
    GenericMoeDecoderLayer,
    GenericMoeLayer,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    DenseMLP,
    Embedding,
    LinearFactory,
    MlaAttention,
    RMSNorm,
)
from rtp_llm.models_py.modules.dsv4.hc.base import HCUnitBase
from rtp_llm.models_py.modules.dsv4.hc.utils import maybe_squeeze_hc_1d
from rtp_llm.models_py.modules.hybrid.fused_norm_quant import (
    fused_rmsnorm_fp8_quant_ue8m0,
    linear_supports_quantized_input,
    ue8m0_deepgemm_active,
)
from rtp_llm.models_py.modules.xing4.hc import (
    xing4_hc_triton_entries,
    xing4_hc_use_fused,
)
from rtp_llm.models_py.modules.xing4.hc.mean_norm_triton import mean_rmsnorm
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


def _xing4_fused_norm_quant_enabled() -> bool:
    # Fused RMSNorm + ue8m0 FP8 quant for input/post_attention layernorm
    # consumers. Per-call resolution, same discipline as XING4_HC_IMPL.
    # Default ON since the register-resident kernel (single read, whole row
    # in registers, tl.reshape group scales) replaced the DSV4 double-read
    # layout: 2.31us/launch in-graph vs the 4.74us norm+quant pair it
    # replaces, decode TPOT -2.1%..-2.6% E2E (see
    # fusion_workflow/p1_4_norm_quant_fusion.md).
    return os.environ.get("XING4_FUSED_NORM_QUANT", "1") != "0"


def _xing4_hc_chunk_tokens() -> int:
    raw = os.environ.get("XING4_HC_FALLBACK_CHUNK_TOKENS", "16384")
    try:
        return max(int(raw), 1)
    except (TypeError, ValueError):
        return 16384


class Xing4HCUnit(HCUnitBase):
    """Xing4.0-exact hyper-connection unit.

    Weight layout is shared with the DeepSeek-V4 mHC units
    (fn[(hc+2)*hc, hc*dim], base[(hc+2)*hc], scale[3]); the mixer math follows
    modeling_xing4_0.py verbatim:

      * flat = UnweightedRMSNorm(x.flatten(-2).float()); GEMM in stream dtype;
      * pre  = sigmoid(pre_w * pre_scale + pre_b)             (no +eps)
      * post = 2 * sigmoid(post_w * post_scale + post_b)
      * comb = exp(clamp(comb_w*s + b, +-h_res) - rowmax) then
        hc_sinkhorn_iters rounds of (row +eps, col +eps) normalization;
      * post-out = post (x) x + comb @ residual                (comb untransposed).

    Execution backends (factory: ``XING4_HC_IMPL``, see
    ``rtp_llm/models_py/modules/xing4/hc/__init__.py``; default "triton" on a
    CUDA bf16 stream, everything else — including CPU tensors, other dtypes
    and import failures — automatically uses "torch" and logs one line):

      * fused: K1 row RMSNorm -> F.linear GEMM (untouched, fp32 ``mixes``,
        ``_fn_for`` dtype cache) -> K2 gates/sinkhorn -> K3 collapse, and K4
        for the post combination; no chunk branch (no [T, hc*dim] fp32
        intermediate exists to bound).
      * fused K1+GEMM (fusion round-2): when the flattened row count is at
        most ``XING4_HC_FN_GEMV_MAX_TOKENS`` (default 32; the
        decode/small-batch regime), the K1 norm and the hc_fn ``F.linear``
        collapse into ONE Triton GEMV kernel that emits the bf16 ``mixes``
        directly from the stream — no ``flat_normed`` materialization and no
        cuBLAS split-K pair (measured 11-19 us per call at those sizes on
        SM103 — the largest mHC pre-chain stage). Numerics keep the
        reference rounding structure (bf16 norm rounding + fp32 accumulation
        + one bf16 round); only the fp32 summation order differs, the same
        tolerance class the UTs gate at rtol = atol = 2**-7. Larger row
        counts keep K1 + cuBLAS (bandwidth-bound there; the fused form
        re-reads fn per row and loses from ~64 rows on).
      * torch: the eager chain below, kept verbatim as the reference and the
        fallback; it retains the ``XING4_HC_FALLBACK_CHUNK_TOKENS`` chunking.

    Both backends return identical shapes/dtypes: ``y [..., dim]``,
    ``post [..., hc_mult, 1]``, ``comb [..., hc_mult, hc_mult]`` in the stream
    dtype, with all hyperparameters (norm_eps/hc_eps/sinkhorn_iters/clamp)
    read from this instance's attributes.
    """

    def __init__(
        self,
        fn: torch.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
        *,
        h_res_clamp_min: float = -30.0,
        h_res_clamp_max: float = 30.0,
        **kwargs,
    ) -> None:
        super().__init__(fn, base, scale, **kwargs)
        self.h_res_clamp_min = float(h_res_clamp_min)
        self.h_res_clamp_max = float(h_res_clamp_max)
        self._scale = maybe_squeeze_hc_1d(self.scale)
        self._fn_stream_dtype: Optional[torch.Tensor] = None
        self._fn_stream_dtype_ref = None
        self._mix_constants_fp32: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _fn_for(self, dtype: torch.dtype) -> torch.Tensor:
        if (
            self._fn_stream_dtype is None
            or self._fn_stream_dtype.dtype != dtype
            or self._fn_stream_dtype.device != self.fn.device
            or self._fn_stream_dtype.shape != self.fn.shape
        ):
            self._fn_stream_dtype = self.fn.to(dtype)
            self._fn_stream_dtype_ref = self.fn
        return self._fn_stream_dtype

    def _mix_constants_for(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """fp32 CUDA (base, scale) copies for the fused mix kernel (K2).

        The casts are value-exact (fp32 stays fp32; bf16 widens losslessly),
        so the cached tensors are numerically identical to the eager path's
        type promotion of ``self.base`` / ``self._scale`` against the fp32
        mixes. Mirrors the ``_fn_for`` caching discipline.
        """
        cached = self._mix_constants_fp32
        if (
            cached is None
            or cached[0].device != device
            or cached[0].shape != self.base.shape
            or cached[1].shape != self._scale.shape
        ):
            cached = (
                self.base.to(device=device, dtype=torch.float32).contiguous(),
                self._scale.to(device=device, dtype=torch.float32).contiguous(),
            )
            self._mix_constants_fp32 = cached
        return cached

    def _use_fused_pre(self, x: torch.Tensor) -> bool:
        # K2 is compiled for the production hc=4 (24-wide) mixer and K3 takes
        # 3D strides or contiguous higher-rank streams; anything else stays
        # on the eager reference path.
        return (
            self.hc_mult == 4
            and xing4_hc_use_fused(x)
            and (x.is_contiguous() or x.dim() == 3)
        )

    def _pre_fused(
        self, x: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused _pre_impl: K1 norm -> F.linear GEMM -> K2 mix -> K3 collapse."""
        hc = self.hc_mult
        entries = xing4_hc_triton_entries()
        rows = x.numel() // (hc * self.dim)
        if rows <= entries.fn_gemv_max_tokens():
            # K1+GEMM fusion (round-2): one kernel produces the hc_fn mixes
            # directly from the stream — no flat_normed materialization, no
            # cuBLAS split-K pair. The rows gate exists because the fused
            # form re-reads fn per row while cuBLAS wins from ~64 rows on
            # (measured crossover; fn_gemv_triton.py documents the numbers).
            # Numerics: the bf16 norm rounding is reproduced in-kernel and
            # the dot accumulates fp32 like the GEMM — same structure, only
            # the summation order differs (the 1-ulp class the UTs gate).
            mixes = entries.norm_fn_gemv(
                x, self._fn_for(dtype), self.norm_eps
            )
        else:
            # K1: x[..., hc, H] -> flat_normed with x.flatten(-2) shape, one
            # launch, fp32 square accumulation in registers (no [T, hc*H]
            # fp32 intermediate, hence no chunking on this path).
            flat_normed = entries.row_rms_norm(x, self.norm_eps)
            # GEMM preserved verbatim (the hc_fn F.linear share stays out of
            # the fusion numerator); the bf16 GEMM output feeds K2 directly
            # (widened losslessly in-kernel — bitwise identical to the eager
            # .float() path, minus the separate widen launch).
            mixes = torch.nn.functional.linear(flat_normed, self._fn_for(dtype))
        mixes = mixes.reshape(-1, (hc + 2) * hc)
        # K2: gates + clamp/exp + sinkhorn comb in one launch; its single exit
        # cast lands exactly where the reference casts pre/post/comb.
        base_fp32, scale_fp32 = self._mix_constants_for(mixes.device)
        pre, post, comb = entries.mix(
            mixes,
            base_fp32,
            scale_fp32,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            eps=self.hc_eps,
            clamp_min=self.h_res_clamp_min,
            clamp_max=self.h_res_clamp_max,
            out_dtype=dtype,
        )
        # K3: y[..., H] = sum_i pre[..., i] * x[..., i, H]; K2 already cast
        # pre to the stream dtype, mirroring the reference's pre.to(dtype).
        y = entries.collapse(pre, x)
        leading = x.shape[:-2]
        return y, post.view(*leading, hc, 1), comb.view(*leading, hc, hc)

    def _mixes(self, x_flat: torch.Tensor, rsqrt: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        flat_normed = (x_flat.float() * rsqrt).to(dtype)
        return torch.nn.functional.linear(flat_normed, self._fn_for(dtype)).float()

    def _pre_impl(self, x: torch.Tensor, dbg_tag: Optional[str] = None):
        if self._use_fused_pre(x):
            return self._pre_fused(x, x.dtype)
        shape, dtype = x.size(), x.dtype
        hc = self.hc_mult
        x_flat = x.flatten(-2)
        flat = x_flat.dim() == 2
        T = x_flat.shape[0] if flat else 0
        chunk = _xing4_hc_chunk_tokens()

        if flat and T > chunk:
            mixes_list = []
            for s in range(0, T, chunk):
                e = min(s + chunk, T)
                rsqrt = torch.rsqrt(
                    x_flat[s:e].float().square().mean(-1, keepdim=True)
                    + self.norm_eps
                )
                mixes_list.append(self._mixes(x_flat[s:e], rsqrt, dtype))
            mixes = torch.cat(mixes_list, dim=0).contiguous()
        else:
            rsqrt = torch.rsqrt(
                x_flat.float().square().mean(-1, keepdim=True) + self.norm_eps
            )
            mixes = self._mixes(x_flat, rsqrt, dtype)

        pre_w, post_w, comb_w = mixes.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self._scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b)
        post = 2.0 * torch.sigmoid(post_w * post_scale + post_b)

        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb_logits = comb_logits.clamp(
            min=self.h_res_clamp_min, max=self.h_res_clamp_max
        )
        comb = torch.exp(comb_logits - comb_logits.amax(dim=-1, keepdim=True))
        for _ in range(self.hc_sinkhorn_iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)

        x_view = x.view(*shape)
        if flat and T > chunk:
            y = torch.empty((T, shape[-1]), dtype=dtype, device=x.device)
            pre_dt = pre.to(dtype)
            for s in range(0, T, chunk):
                e = min(s + chunk, T)
                y[s:e] = torch.sum(
                    pre_dt[s:e].unsqueeze(-1) * x_view[s:e], dim=-2
                )
            y = y.view(*shape[:-2], shape[-1])
        else:
            y = torch.sum(pre.to(dtype).unsqueeze(-1) * x_view, dim=-2)
        return y.to(dtype), post.to(dtype).unsqueeze(-1), comb.to(dtype)

    def _use_fused_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> bool:
        # K4 (xing4_hc_post) takes contiguous CUDA bf16 tensors and casts
        # post/comb itself; anything else stays on the eager reference path.
        return (
            self.hc_mult == 4
            and xing4_hc_use_fused(x)
            and residual.dtype == x.dtype
            and residual.device == x.device
            and all(t.is_contiguous() for t in (x, residual, post, comb))
        )

    def _post_impl(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        if self._use_fused_post(x, residual, post, comb):
            # K4: first + comb @ residual in one launch, with the reference's
            # dtype casts and the post[..., 1] squeeze handled by the wrapper.
            return xing4_hc_triton_entries().post(x, residual, post, comb)
        post_b = post.squeeze(-1) if post.dim() == residual.dim() else post
        first = post_b.to(x.dtype).unsqueeze(-1) * x.unsqueeze(-2)
        second = torch.matmul(comb.to(x.dtype), residual.to(x.dtype))
        return (first + second).to(x.dtype)


class Xing4DecoderLayer(nn.Module):
    """GenericMoe decoder layer with the fused RMSResNorm residual replaced by
    Xing4 hyper-connection units; attention and MLP modules are the production
    MlaAttention / DenseMLP / GenericMoeLayer components."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights,
        global_weights,
        layer_idx: int,
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        quant_config = config.quant_config

        if not config.attn_config.use_mla:
            raise ValueError(
                "Xing4DecoderLayer requires MLA attention (use_mla=True)"
            )
        self.self_attn = MlaAttention(
            config.attn_config,
            parallelism_config,
            weights,
            layer_idx,
            config.layernorm_eps,
            quant_config,
            hw_kernel_config,
            global_weights=global_weights,
            enable_fused_norm_quant=_xing4_fused_norm_quant_enabled(),
        )

        if layer_idx not in config.moe_layer_index:
            self.mlp = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                quant_config,
                hw_kernel_config=hw_kernel_config,
            )
        else:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
            )

        # Plain RMSNorm on the collapsed single stream, mirroring the HF
        # reference (input_layernorm applies to attn_hc's collapsed output).
        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

        hc_kwargs = dict(
            dim=config.hidden_size,
            hc_mult=config.hc_mult,
            hc_sinkhorn_iters=config.hc_sinkhorn_iters,
            norm_eps=config.layernorm_eps,
            hc_eps=config.hc_eps,
            h_res_clamp_min=config.mhc_h_res_clamp_min,
            h_res_clamp_max=config.mhc_h_res_clamp_max,
        )
        self.attn_hc = Xing4HCUnit(
            weights[W.v4_hc_attn_fn],
            weights[W.v4_hc_attn_base],
            weights[W.v4_hc_attn_scale],
            layer_id=layer_idx,
            name="attn",
            **hc_kwargs,
        )
        self.ffn_hc = Xing4HCUnit(
            weights[W.v4_hc_ffn_fn],
            weights[W.v4_hc_ffn_base],
            weights[W.v4_hc_ffn_scale],
            layer_id=layer_idx,
            name="ffn",
            **hc_kwargs,
        )

    def _can_fuse_norm_quant(
        self,
        x: torch.Tensor,
        norm: RMSNorm,
        linear_consumers: tuple,
        require_ue8m0_router: bool = False,
    ) -> bool:
        """Fused RMSNorm + ue8m0 FP8 quant gate shared by both norm sites.

        Requires the collapsed [T, H] stream, a contiguous bf16 norm weight,
        H % 128 == 0, and every linear consumer to accept the same
        (fp8, packed ue8m0 scale) pair the fused kernel emits. m-row aware:
        ``CudaFp8GEMMLinear`` wrappers dispatch m < 32 to a FlashInfer
        backend that must keep reading the bf16 normed tensor.

        ``require_ue8m0_router`` additionally gates on the device's MoE
        router quant flavor being packed ue8m0 (SM100/SM120); SM90 routers
        quantize with fp32 scales, so the MoE site stays unfused there.
        """
        if not _xing4_fused_norm_quant_enabled():
            return False
        if require_ue8m0_router and not ue8m0_deepgemm_active():
            return False
        if (
            x.dim() != 2
            or x.dtype != torch.bfloat16
            or not x.is_cuda
            or not x.is_contiguous()
        ):
            return False
        # Decode/graph regime only (m <= 64 = the capture buckets): the
        # fused kernel wins in-graph (2.31us vs the 4.74us pair, TPOT
        # -2.1~-2.6% E2E), but eager prefill is host-bound and the per-call
        # wrapper overhead measured +6~24% TTFT at m=1024/4096 — larger
        # batches take the plain two-kernel chain.
        if x.shape[0] > 64:
            return False
        n = x.shape[1]
        if n % 128 != 0:
            return False
        weight = norm.weight
        if (
            weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
            or weight.shape != (n,)
        ):
            return False
        return all(
            linear_supports_quantized_input(linear, n, m_rows=x.shape[0])
            for linear in linear_consumers
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        """hidden_states: [T, hc_mult, H]; returns the same shape."""
        residual = hidden_states
        x_pre, post, comb = self.attn_hc.pre(hidden_states)
        attn_quant = None
        attn_consumers = ()
        if self.self_attn.q_lora_rank > 0 and hasattr(
            self.self_attn, "fused_qkv_a_proj"
        ):
            attn_consumers = (self.self_attn.fused_qkv_a_proj,)
        if self._can_fuse_norm_quant(
            x_pre, self.input_layernorm, attn_consumers
        ):
            # Norm in place over the dead pre-norm stream; the (fp8, scale)
            # pair feeds fused_qkv_a_proj without the separate quant kernel.
            x_pre, attn_fp8, attn_scales = fused_rmsnorm_fp8_quant_ue8m0(
                x_pre,
                self.input_layernorm.weight,
                eps=self.input_layernorm.variance_epsilon,
                out_norm=x_pre,
            )
            attn_quant = (attn_fp8, attn_scales)
        else:
            x_pre = self.input_layernorm(x_pre)
        attn_out = self.self_attn(
            hidden_states=x_pre,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
            quantized_input=attn_quant,
        )
        hidden_states = self.attn_hc.post(attn_out, residual, post, comb)

        residual = hidden_states
        x_pre, post, comb = self.ffn_hc.pre(hidden_states)
        ffn_quant = None
        if isinstance(self.mlp, (GenericMoeLayer, DenseMLP)) and self._can_fuse_norm_quant(
            x_pre, self.post_attention_layernorm, (), require_ue8m0_router=True
        ):
            # No linear gate here: the MoE router and shared expert consume
            # the quantized pair through GenericMoeLayer/DenseMLP opt-in
            # paths (each re-checks its own linears); the routed gate GEMM
            # keeps reading the bf16 normed tensor.
            x_pre, ffn_fp8, ffn_scales = fused_rmsnorm_fp8_quant_ue8m0(
                x_pre,
                self.post_attention_layernorm.weight,
                eps=self.post_attention_layernorm.variance_epsilon,
                out_norm=x_pre,
            )
            ffn_quant = (ffn_fp8, ffn_scales)
        else:
            x_pre = self.post_attention_layernorm(x_pre)
        mlp_out = self.mlp(x_pre, quantized_input=ffn_quant)
        hidden_states = self.ffn_hc.post(mlp_out, residual, post, comb)
        return hidden_states


class Xing4Model(GptModelBase):
    """Xing4.0 model: embed -> hc expand -> N mHC decoder layers -> mean(dim=hc)
    -> final RMSNorm. Returns the [T, H] hidden states for the C++ lm_head."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.hc_mult = int(model_config.hc_mult)
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                Xing4DecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    weights.global_weights,
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )
        self._init_capture_context(
            self._capture_canonical_layer,
            self._capture_canonical_final,
        )

    def _capture_canonical_layer(
        self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # Xing4 carries one hc residual stream set; the canonical layer
        # boundary hidden is the mean over the hc axis (reference: mean(dim=2)).
        return hidden_states.mean(dim=-2)

    def _capture_canonical_final(
        self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # Fused mean+norm tail (XING4_MEAN_NORM_IMPL, torch fallback):
        # mathematically self.norm(hidden_states.mean(dim=-2)) with one
        # single-read kernel and no [T, H] intermediate.
        return mean_rmsnorm(
            hidden_states, self.norm.weight, self.norm.variance_epsilon
        )

    def embedding(self, inputs: PyModelInputs) -> torch.Tensor:
        return self.embed_tokens(inputs.input_ids)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        hidden_states = self.embedding(inputs)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(
                inputs
            )  # pyright: ignore[reportUnreachable]
        capture = self.capture_context(inputs.capture_hidden_states)
        h = hidden_states.unsqueeze(-2).expand(-1, self.hc_mult, -1).contiguous()
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            layer_fmha_impl = select_fmha_impl_for_layer(
                fmha_impl, self.kv_cache, i
            )
            layer_kv_cache = (
                self.kv_cache.get_layer_cache(i) if self.kv_cache else None
            )
            h = decoder_layer(h, layer_fmha_impl, kv_cache=layer_kv_cache)
            capture.capture_layer(i, h)
        return capture.finalize(h)


class Xing4MtpModel(GptModelBase):
    """Xing4.0 MTP (nextn) draft model: python-side eh_proj fusion.

    Follows the Qwen2MtpModel contract: the engine feeds the target's final
    hidden via ``inputs.input_hiddens`` and the draft tokens via
    ``input_ids``; the draft embeds, applies enorm/hnorm, fuses with a single
    eh_proj, runs the one standard-residual decoder layer (layer 40, MLA +
    MoE, no mHC), and applies its own final norm (shared_head.norm) before
    the C++ side applies the shared head.

    The concat order is Xing4.0-specific: [enorm(e), hnorm(h)] (e-first),
    verified numerically against the dequantized reference (the reversed
    order scores 0/16 next-token matches; see RUN mtp_order_diag.py).
    """

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        moe_config: MoeConfig = None,
        quant_config: Optional[object] = None,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.eh_proj = LinearFactory.create_linear_from_weights(
            weights.weights[0],
            W.multi_tokens_predict_eh_proj,
            quant_config=quant_config,
            hw_kernel_config=py_hw_kernel_config,
        )
        self.e_norm = RMSNorm(
            weights.weights[0][W.multi_tokens_predict_enorm],
            eps=config.layernorm_eps,
        )
        self.h_norm = RMSNorm(
            weights.weights[0][W.multi_tokens_predict_hnorm],
            eps=config.layernorm_eps,
        )
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                GenericMoeDecoderLayer(
                    config,
                    parallelism_config,
                    weights.weights[idx],
                    weights.global_weights,
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.weights[0][W.multi_tokens_predict_final_ln_gamma],
            eps=config.layernorm_eps,
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(
                inputs
            )  # pyright: ignore[reportUnreachable]
        e_n = self.e_norm(self.embed_tokens(inputs.input_ids))
        h_n = self.h_norm(inputs.input_hiddens)
        hidden_states = self.eh_proj(torch.cat([e_n, h_n], dim=-1))
        residual = torch.zeros_like(hidden_states)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            layer_fmha_impl = select_fmha_impl_for_layer(
                fmha_impl, self.kv_cache, i
            )
            output = decoder_layer(
                hidden_states,
                residual,
                layer_fmha_impl,
                kv_cache=(
                    self.kv_cache.get_layer_cache(i) if self.kv_cache else None
                ),
            )
            hidden_states = output.hidden_states
            residual = output.residual
        hidden_states = self.norm(hidden_states + residual)
        return PyModelOutputs(hidden_states)


__all__ = ["Xing4Model", "Xing4DecoderLayer", "Xing4HCUnit", "Xing4MtpModel"]
