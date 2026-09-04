from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.factory.attention.attn_factory import MlaImplBase
from rtp_llm.models_py.modules.hybrid.indexer import Indexer
from rtp_llm.ops import AttentionConfigs, HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache
from rtp_llm.utils.model_weight import W

# CUDA-only fused strided RMSNorm (replaces .contiguous() + RMSNorm). When
# q_b_proj is fp8, we additionally emit fp8+scale to skip a separate
# per-token-group fp8 quant launch. ROCm path falls back to the unfused chain.
_DEVICE_TYPE = get_device_type()
if _DEVICE_TYPE == DeviceType.Cuda:
    from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_quant_act_packed
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
        CudaFp8GEMMLinear,
    )
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.mxfp8_linear import (
        CudaMxfp8Linear,
    )
    from rtp_llm.models_py.modules.hy_v4.gated_mla_triton import (
        maybe_fused_gated_mla_proj_mxfp8,
    )
    from rtp_llm.models_py.triton_kernels.common.attn_output_gate import (
        sigmoid_mul_fp8_quant_fwd,
    )
    from rtp_llm.models_py.triton_kernels.common.fused_strided_rmsnorm import (
        fused_strided_rmsnorm,
        fused_strided_rmsnorm_per_token_fp8_quant,
        fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output,
    )
else:
    CudaFp8GEMMLinear = None  # type: ignore
    CudaMxfp8Linear = None  # type: ignore
    maybe_fused_gated_mla_proj_mxfp8 = None  # type: ignore
    mxfp8_quant_act_packed = None  # type: ignore
    sigmoid_mul_fp8_quant_fwd = None  # type: ignore
    fused_strided_rmsnorm = None  # type: ignore
    fused_strided_rmsnorm_per_token_fp8_quant = None  # type: ignore
    fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output = None  # type: ignore


def _infer_gated_mla_type(
    gate_weight: torch.Tensor,
    num_heads: int,
    v_head_dim: int,
) -> str:
    """Infer the gate kind from BF16-transposed or FP8 checkpoint layout."""
    if gate_weight.dim() != 2:
        raise ValueError(
            f"gated MLA weight must be 2D, got {tuple(gate_weight.shape)}"
        )
    shape = tuple(gate_weight.shape)
    elementwise_width = num_heads * v_head_dim
    elementwise = elementwise_width in shape
    headwise = num_heads in shape
    if elementwise and not headwise:
        return "elementwise"
    if headwise and not elementwise:
        return "headwise"
    raise ValueError(
        f"invalid or ambiguous gated MLA shape {shape}; expected one dimension "
        f"to equal {num_heads} (headwise) or {elementwise_width} (elementwise)"
    )


class MlaAttention(nn.Module):
    """MLA attention. Supports both dense and sparse (indexer/top-k) modes.
    Whether to use Indexer is determined by attn_config.is_sparse.
    """

    def __init__(
        self,
        attn_config: AttentionConfigs,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        layernorm_eps: float,
        quant_config: object,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        global_weights: Optional[Dict[str, torch.Tensor]] = None,
        has_indexer: Optional[bool] = None,
        reuse_topk_indices: bool = False,
        indexer_layernorm_eps: Optional[float] = None,
        indexer_scale_fmt: Optional[str] = None,
        indexer_use_hadamard: bool = True,
    ):
        super().__init__()
        self.attn_config = attn_config
        self.parallelism_config = parallelism_config
        self.num_heads = (
            attn_config.head_num // self.parallelism_config.get_attn_tp_size()
        )
        self.qk_nope_head_dim = attn_config.nope_head_dim
        self.qk_rope_head_dim = attn_config.rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_lora_rank = attn_config.kv_lora_rank
        self.v_head_dim = attn_config.v_head_dim
        self.q_lora_rank = attn_config.q_lora_rank
        self.softmax_scale = self.q_head_dim ** (-0.5)
        self.layer_idx = layer_idx
        self.token_per_block = attn_config.kernel_tokens_per_block
        self.has_indexer = (
            bool(attn_config.is_sparse) if has_indexer is None else has_indexer
        )
        self.reuse_topk_indices = bool(reuse_topk_indices)

        if self.has_indexer:
            self.indexer = Indexer(
                attn_config,
                weights,
                global_weights,
                layer_idx,
                (
                    layernorm_eps
                    if indexer_layernorm_eps is None
                    else indexer_layernorm_eps
                ),
                quant_config,
                hw_kernel_config,
                parallelism_config,
                scale_fmt=(
                    "none" if indexer_scale_fmt is None else indexer_scale_fmt
                ),
                use_hadamard=indexer_use_hadamard,
            )
        else:
            self.indexer = None

        if self.q_lora_rank > 0:
            self.fused_qkv_a_proj = LinearFactory.create_linear_from_weights(
                weights,
                W.mla_fusedqkrope_w,
                W.mla_fusedqkrope_s,
                None,
                quant_config=quant_config,
                hw_kernel_config=hw_kernel_config,
            )
            self.q_a_layernorm = RMSNorm(
                weights.get(W.mla_q_a_ln_gamma, None), eps=layernorm_eps
            )
            self.q_b_proj = LinearFactory.create_linear_from_weights(
                weights,
                W.mla_q_b_w,
                W.mla_q_b_s,
                None,
                quant_config=quant_config,
                hw_kernel_config=hw_kernel_config,
            )
        else:
            self.fused_qkv_proj = LinearFactory.create_linear_from_weights(
                weights,
                W.mla_fusedqkrope_no_lora_w,
                W.mla_fusedqkrope_no_lora_s,
                None,
                quant_config=quant_config,
                hw_kernel_config=hw_kernel_config,
            )

        self.kv_a_layernorm = RMSNorm(
            weights.get(W.mla_kv_a_ln_gamma, None), eps=layernorm_eps
        )

        self.o_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.attn_o_w,
            W.attn_o_s,
            W.attn_o_b,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        # HY V4 opt-in extensions.  Presence of the HY-specific loader keys is
        # the runtime contract, so other MLA models keep the exact old path.
        self.gate_proj = None
        self.gating_type = None
        gate_weight = weights.get(W.attn_gate_w)
        if gate_weight is not None:
            self.gate_proj = LinearFactory.create_linear_from_weights(
                weights,
                W.attn_gate_w,
                W.attn_gate_s,
                None,
                quant_config=quant_config,
                hw_kernel_config=hw_kernel_config,
            )
            self.gating_type = _infer_gated_mla_type(
                gate_weight, self.num_heads, self.v_head_dim
            )

        self.attn_sink = weights.get(W.hy4_attn_sink)
        if self.attn_sink is not None:
            if self.attn_sink.dtype != torch.float32:
                raise TypeError(
                    f"HY V4 attention sink must be fp32, got {self.attn_sink.dtype}"
                )
            if self.attn_sink.dim() != 1 or self.attn_sink.numel() != self.num_heads:
                raise ValueError(
                    f"HY V4 attention sink at layer {layer_idx} must have shape "
                    f"({self.num_heads},), got {tuple(self.attn_sink.shape)}"
                )

        # ------------------------------------------------------------------
        # Fusion detection (DSV3.2 MLA path).
        #
        # F2  : kv_a_layernorm receives a strided slice from torch.split. We
        #       always try the fused_strided_rmsnorm path; the wrapper falls
        #       back to .contiguous() + flashinfer.norm.rmsnorm when the input
        #       isn't compatible (H>8192 or last-dim stride != 1).
        # F1a : q_a_layernorm with bf16 q_b_proj — same as F2 (bf16 output).
        # F1b : q_a_layernorm with fp8 q_b_proj — produces dual output
        #       (bf16 for the indexer wq_b consumer, fp8 + scale for q_b_proj).
        # ------------------------------------------------------------------
        from rtp_llm.models_py.utils.fuse_config import fuse_kernels_enabled

        _fuse_on = fuse_kernels_enabled(hw_kernel_config)
        self._fuse_kv_a_norm = (
            _fuse_on
            and _DEVICE_TYPE == DeviceType.Cuda
            and fused_strided_rmsnorm is not None
        )

        # q-path fusion mode: "fp8_dual" (F1b), "bf16" (F1a), or "off" (fallback)
        self._fuse_q_a_norm_mode = "off"
        if _fuse_on and self.q_lora_rank > 0 and _DEVICE_TYPE == DeviceType.Cuda:
            q_b_is_fp8 = (
                CudaFp8GEMMLinear is not None
                and isinstance(self.q_b_proj, CudaFp8GEMMLinear)
                and self.q_lora_rank % 128 == 0
            )
            if q_b_is_fp8:
                # UE8M0 needs num_groups % 4 == 0
                if self.q_b_proj.scale_ue8m0:
                    if (self.q_lora_rank // 128) % 4 == 0:
                        self._fuse_q_a_norm_mode = "fp8_dual"
                else:
                    self._fuse_q_a_norm_mode = "fp8_dual"
            elif fused_strided_rmsnorm is not None:
                self._fuse_q_a_norm_mode = "bf16"

        # MXFP8 projections use the same deterministic per-(row, 32-column)
        # activation quantization.  When the sparse Indexer consumes the same
        # hidden/q_c tensor as MLA, quantize once and pass the exact same FP8
        # bytes and packed scales to both GEMMs.
        main_input_proj = (
            self.fused_qkv_a_proj
            if self.q_lora_rank > 0
            else self.fused_qkv_proj
        )
        self.accepts_mxfp8_input = bool(
            CudaMxfp8Linear is not None
            and isinstance(main_input_proj, CudaMxfp8Linear)
        )
        self._reuse_mxfp8_hidden_quant = bool(
            mxfp8_quant_act_packed is not None
            and CudaMxfp8Linear is not None
            and self.indexer is not None
            and self.accepts_mxfp8_input
            and isinstance(self.indexer.wk, CudaMxfp8Linear)
        )
        self._reuse_mxfp8_q_c_quant = bool(
            mxfp8_quant_act_packed is not None
            and CudaMxfp8Linear is not None
            and self.q_lora_rank > 0
            and self.indexer is not None
            and isinstance(self.q_b_proj, CudaMxfp8Linear)
            and isinstance(self.indexer.wq_b, CudaMxfp8Linear)
        )
        if (
            _fuse_on
            and CudaMxfp8Linear is not None
            and fused_strided_rmsnorm_per_token_fp8_quant is not None
            and isinstance(getattr(self, "q_b_proj", None), CudaMxfp8Linear)
        ):
            if self.indexer is None:
                self._fuse_q_a_norm_mode = "mxfp8"
            elif (
                self._reuse_mxfp8_q_c_quant
                and fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output
                is not None
            ):
                self._fuse_q_a_norm_mode = "mxfp8_dual"

        # HY4 Gated MLA epilogue: fuse elementwise sigmoid, multiply, and the
        # activation quantization consumed by the quantized output projection.
        self._fuse_gated_mla_quant = False
        self._gated_mla_quant_group_size = 128
        self._gated_mla_scale_ue8m0 = False
        self._gated_mla_round_scale_to_pow2 = False
        self._fuse_gated_mla_proj_quant = False
        if (
            _fuse_on
            and self.gating_type == "elementwise"
            and sigmoid_mul_fp8_quant_fwd is not None
        ):
            if CudaMxfp8Linear is not None and isinstance(
                self.o_proj, CudaMxfp8Linear
            ):
                self._fuse_gated_mla_quant = True
                self._gated_mla_quant_group_size = (
                    self.o_proj.input_quant_group_size
                )
                # Request DeepGEMM's packed UE8M0/TMA layout directly so the
                # output projection skips its standalone scale-pack kernel.
                self._gated_mla_scale_ue8m0 = (
                    self.o_proj.input_quant_scale_ue8m0
                )
                self._gated_mla_round_scale_to_pow2 = (
                    self.o_proj.input_quant_round_to_pow2
                )
                self._fuse_gated_mla_proj_quant = bool(
                    maybe_fused_gated_mla_proj_mxfp8 is not None
                    and getattr(self.gate_proj, "bias", None) is None
                    and self._gated_mla_quant_group_size == 32
                    and self._gated_mla_scale_ue8m0
                    and self._gated_mla_round_scale_to_pow2
                )
            elif CudaFp8GEMMLinear is not None and isinstance(
                self.o_proj, CudaFp8GEMMLinear
            ):
                self._gated_mla_scale_ue8m0 = self.o_proj.scale_ue8m0
                self._fuse_gated_mla_quant = self.o_proj.K % 128 == 0 and (
                    not self._gated_mla_scale_ue8m0
                    or (self.o_proj.K // 128) % 4 == 0
                )

    def _run_sparse_indexer(
        self,
        hidden_states: torch.Tensor,
        q_c: Optional[torch.Tensor],
        q_view: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        fmha_impl: MlaImplBase,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
        q_c_fp8: Optional[torch.Tensor] = None,
        q_c_scale: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
        force_reuse_topk_indices: bool = False,
    ) -> Optional[torch.Tensor]:
        if self.reuse_topk_indices or force_reuse_topk_indices:
            if not fmha_impl.is_sparse():
                return None
            if prev_topk_indices is None:
                raise RuntimeError(
                    f"DSA shared layer {self.layer_idx} needs previous top-k "
                    "indices, but none were provided"
                )
            return prev_topk_indices

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
            x_fp8=x_fp8,
            x_scale=x_scale,
            q_c_fp8=q_c_fp8,
            q_c_scale=q_c_scale,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: MlaImplBase,
        kv_cache: Optional[LayerKVCache] = None,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
        force_reuse_topk_indices: bool = False,
        return_topk: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        if (
            x_fp8 is None
            and x_scale is None
            and self._reuse_mxfp8_hidden_quant
            and hidden_states.dim() == 2
            and hidden_states.is_contiguous()
        ):
            x_fp8, x_scale = mxfp8_quant_act_packed(hidden_states)
        q_c = None
        q_c_fp8 = None
        q_c_scale = None
        if self.q_lora_rank > 0:
            if x_fp8 is not None and x_scale is not None:
                fused_qkv = self.fused_qkv_a_proj(x_fp8, input_scales=x_scale)
            else:
                fused_qkv = self.fused_qkv_a_proj(hidden_states)
            kv_offset = self.q_lora_rank
            q, compressed_kv = torch.split(
                fused_qkv,
                [
                    kv_offset,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ],
                dim=-1,
            )
            # F1a/F1b: fused strided RMSNorm (skip .contiguous() copy). When
            # q_b_proj is fp8 we additionally emit fp8+scale (F1b dual output)
            # so q_b_proj can use input_scales= and skip its internal quant.
            if self._fuse_q_a_norm_mode == "mxfp8_dual":
                q_c, q_c_fp8, q_c_scale = (
                    fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output(
                        q,
                        self.q_a_layernorm.weight.data,
                        self.q_a_layernorm.variance_epsilon,
                        group_size=32,
                        scale_ue8m0=True,
                        mxfp8_semantics=True,
                    )
                )
                q = self.q_b_proj(q_c_fp8, input_scales=q_c_scale)
            elif self._fuse_q_a_norm_mode == "mxfp8":
                q_fp8, q_scale = fused_strided_rmsnorm_per_token_fp8_quant(
                    q,
                    self.q_a_layernorm.weight.data,
                    self.q_a_layernorm.variance_epsilon,
                    group_size=32,
                    scale_ue8m0=True,
                    mxfp8_semantics=True,
                )
                q = self.q_b_proj(q_fp8, input_scales=q_scale)
            elif self._fuse_q_a_norm_mode == "fp8_dual":
                q_c, q_c_fp8, q_c_scale = (
                    fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output(
                        q,
                        self.q_a_layernorm.weight.data,
                        self.q_a_layernorm.variance_epsilon,
                        group_size=128,
                        scale_ue8m0=self.q_b_proj.scale_ue8m0,
                    )
                )
                q = self.q_b_proj(q_c_fp8, input_scales=q_c_scale)
            elif self._fuse_q_a_norm_mode == "bf16":
                q_c = fused_strided_rmsnorm(
                    q,
                    self.q_a_layernorm.weight.data,
                    self.q_a_layernorm.variance_epsilon,
                )
                q = self.q_b_proj(q_c)
            else:
                q_c = self.q_a_layernorm(q.contiguous())
                q = self.q_b_proj(q_c)
        else:
            if x_fp8 is not None and x_scale is not None:
                fused_qkv = self.fused_qkv_proj(x_fp8, input_scales=x_scale)
            else:
                fused_qkv = self.fused_qkv_proj(hidden_states)
            kv_offset = self.num_heads * self.attn_config.size_per_head
            q, compressed_kv = torch.split(
                fused_qkv,
                [
                    kv_offset,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ],
                dim=-1,
            )
        q_view = q.reshape(-1, self.num_heads, self.q_head_dim)

        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # F2: fused strided RMSNorm on compressed_kv (skip .contiguous() copy)
        if self._fuse_kv_a_norm:
            compressed_kv = fused_strided_rmsnorm(
                compressed_kv,
                self.kv_a_layernorm.weight.data,
                self.kv_a_layernorm.variance_epsilon,
            )
        else:
            compressed_kv = self.kv_a_layernorm(compressed_kv.contiguous())

        topk_indices = self._run_sparse_indexer(
            hidden_states,
            q_c,
            q_view,
            kv_cache,
            fmha_impl,
            x_fp8 if self._reuse_mxfp8_hidden_quant else None,
            x_scale if self._reuse_mxfp8_hidden_quant else None,
            q_c_fp8,
            q_c_scale,
            prev_topk_indices,
            force_reuse_topk_indices,
        )
        # q_c and its quantized representation are Indexer-only. Releasing
        # the local references here lets SparseMLA reuse their blocks;
        # q_view, compressed_kv and k_pe must stay live through attention.
        del q_c, q_c_fp8, q_c_scale
        if self.attn_sink is None:
            attn_output = fmha_impl.forward(
                q_view, compressed_kv, k_pe, kv_cache, self.layer_idx, topk_indices
            )
        else:
            if not fmha_impl.is_sparse():
                raise RuntimeError(
                    "HY V4 learnable attention sink requires a sparse MLA backend"
                )
            attn_output = fmha_impl.forward(
                q_view,
                compressed_kv,
                k_pe,
                kv_cache,
                self.layer_idx,
                topk_indices,
                attn_sink=self.attn_sink,
            )

        # The sparse-attention launch has consumed these projections. PyTorch's
        # stream-aware allocator delays physical reuse until the launch is safe.
        del q_view, compressed_kv, k_pe, fused_qkv, q

        if attn_output is not None:
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        else:
            attn_output = torch.zeros(
                (*input_shape, self.num_heads * self.v_head_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        output = None
        if self.gate_proj is not None:
            fused_gate_quant = None
            if self._fuse_gated_mla_proj_quant and attn_output.dim() == 2:
                fused_gate_quant = maybe_fused_gated_mla_proj_mxfp8(
                    hidden_states,
                    self.gate_proj.weight,
                    attn_output,
                )
            if fused_gate_quant is not None:
                fp8_output, fp8_scale = fused_gate_quant
                output = self.o_proj(fp8_output, input_scales=fp8_scale)
            else:
                gate = self.gate_proj(hidden_states)
            if output is None and (
                self._fuse_gated_mla_quant
                and self.gating_type == "elementwise"
                and attn_output.dim() == 2
            ):
                fp8_output, fp8_scale = sigmoid_mul_fp8_quant_fwd(
                    attn_output,
                    gate,
                    quant_group_size=self._gated_mla_quant_group_size,
                    scale_ue8m0=self._gated_mla_scale_ue8m0,
                    round_scale_to_pow2=self._gated_mla_round_scale_to_pow2,
                    column_major_scales=True,
                )
                output = self.o_proj(fp8_output, input_scales=fp8_scale)
            elif output is None:
                gate = torch.sigmoid(gate)
                if self.gating_type == "headwise":
                    attn_output = attn_output.reshape(
                        *input_shape, self.num_heads, self.v_head_dim
                    )
                    attn_output = attn_output * gate.unsqueeze(-1)
                    attn_output = attn_output.reshape(*input_shape, -1)
                else:
                    attn_output = attn_output * gate
        if output is None:
            output = self.o_proj(attn_output)
        if self.parallelism_config.get_attn_tp_size() > 1:
            output = all_reduce(output, group=Group.TP)
        if return_topk:
            return output, topk_indices
        return output
