from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple, cast

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.factory.attention.attn_factory import MlaImplBase
from rtp_llm.models_py.modules.hybrid.fused_norm_quant import (
    deepgemm_forward_quantized,
    fused_rmsnorm_fp8_quant_ue8m0,
    linear_supports_quantized_input,
)
from rtp_llm.models_py.modules.hybrid.indexer import Indexer
from rtp_llm.ops import AttentionConfigs, HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache
from rtp_llm.utils.model_weight import W


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
        enable_fused_norm_quant: bool = False,
    ):
        super().__init__()
        self.attn_config = attn_config
        self.parallelism_config = parallelism_config
        # Opt-in fused RMSNorm+FP8 quant paths (q_a_layernorm -> q_b_proj and
        # caller-provided input_norm -> fused_qkv_a_proj quantized input).
        # Off by default: only models that verified the ue8m0 DeepGEMM layout
        # (e.g. Xing4 on SM100/SM120) turn it on.
        self.enable_fused_norm_quant = enable_fused_norm_quant
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

        if attn_config.is_sparse:
            self.indexer = Indexer(
                attn_config,
                weights,
                global_weights,
                layer_idx,
                layernorm_eps,
                quant_config,
                hw_kernel_config,
                parallelism_config,
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

    def _run_sparse_indexer(
        self,
        hidden_states: torch.Tensor,
        q_c: Optional[torch.Tensor],
        q_view: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        fmha_impl: MlaImplBase,
    ) -> Optional[torch.Tensor]:
        if self.indexer is None or kv_cache is None:
            return None
        q_for_indexer = q_c if self.q_lora_rank > 0 else q_view
        topk_indices = self.indexer(
            hidden_states,
            q_for_indexer,
            kv_cache,
            fmha_impl.fmha_params,
            fmha_impl.attn_inputs,
            use_fast_path=not fmha_impl.is_sparse(),
            cp_params=fmha_impl.cp_params,
        )
        attn_inputs = fmha_impl.attn_inputs
        if (
            attn_inputs.is_prefill
            and attn_inputs.cache_store_inputs is not None
            and attn_inputs.cache_store_writer is not None
        ):
            attn_inputs.cache_store_writer.write(
                attn_inputs.cache_store_inputs, kv_cache
            )
        return topk_indices

    def _can_fuse_qa_norm_quant(self, q: torch.Tensor) -> bool:
        """q_a_layernorm -> q_b_proj fusion gate (opt-in per model)."""
        if not self.enable_fused_norm_quant:
            return False
        if self.q_lora_rank <= 0 or self.q_lora_rank % 128 != 0:
            return False
        if (
            q.dim() != 2
            or q.dtype != torch.bfloat16
            or not q.is_cuda
            or q.stride(1) != 1
            or q.stride(0) < q.shape[1]
        ):
            return False
        weight = self.q_a_layernorm.weight
        if (
            weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
            or weight.shape != (self.q_lora_rank,)
        ):
            return False
        # m-aware: a CudaFp8GEMMLinear wrapper dispatches m < 32 to its
        # FlashInfer backend, which must keep reading the bf16 normed input;
        # large eager batches (prefill) are host-bound and lose to the plain
        # two-kernel chain, so cap the fusion to the decode/graph regime.
        if q.shape[0] > 64:
            return False
        return linear_supports_quantized_input(
            self.q_b_proj, self.q_lora_rank, m_rows=q.shape[0]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: MlaImplBase | Mapping[str, MlaImplBase],
        kv_cache: Optional[LayerKVCache] | Mapping[str, Optional[LayerKVCache]] = None,
        quantized_input: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.indexer is not None:
            required_tags = {"default", "indexer_kv"}
            if not isinstance(fmha_impl, Mapping) or set(fmha_impl) != required_tags:
                raise RuntimeError(
                    "sparse MLA requires exactly the default and indexer_kv FMHA routes"
                )
            default_fmha_impl = fmha_impl["default"]
            indexer_fmha_impl = fmha_impl["indexer_kv"]
            if not isinstance(default_fmha_impl, MlaImplBase) or not isinstance(
                indexer_fmha_impl, MlaImplBase
            ):
                raise RuntimeError("sparse MLA FMHA routes must contain MlaImplBase")
            if kv_cache is None:
                default_kv_cache = None
                indexer_kv_cache = None
            elif isinstance(kv_cache, Mapping) and set(kv_cache) == required_tags:
                default_kv_cache = kv_cache["default"]
                indexer_kv_cache = kv_cache["indexer_kv"]
                if not isinstance(default_kv_cache, LayerKVCache) or not isinstance(
                    indexer_kv_cache, LayerKVCache
                ):
                    raise RuntimeError(
                        "sparse MLA cache routes must contain LayerKVCache"
                    )
            else:
                raise RuntimeError(
                    "sparse MLA requires exactly the default and indexer_kv "
                    "KV-cache routes when cache is initialized"
                )
        else:
            if isinstance(fmha_impl, Mapping) or isinstance(kv_cache, Mapping):
                raise RuntimeError("dense MLA does not accept tagged cache routes")
            default_fmha_impl = cast(MlaImplBase, fmha_impl)
            default_kv_cache = cast(Optional[LayerKVCache], kv_cache)

        input_shape = hidden_states.shape[:-1]
        q_c = None
        if self.q_lora_rank > 0:
            if quantized_input is not None and linear_supports_quantized_input(
                self.fused_qkv_a_proj,
                hidden_states.shape[-1],
                m_rows=hidden_states.shape[0],
            ):
                fused_qkv = deepgemm_forward_quantized(
                    self.fused_qkv_a_proj, *quantized_input
                )
            else:
                # Plain path: hidden_states is the normed bf16 tensor the
                # fused kernel wrote even when the quant pair is unusable
                # (e.g. the wrapper dispatches small-M to FlashInfer).
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
            if self._can_fuse_qa_norm_quant(q):
                # Fused q_a RMSNorm + ue8m0 quant in one kernel; the strided
                # split view is accepted directly (kernel reads rows through
                # x_stride_m) and q_c lands in a fresh contiguous tensor,
                # matching the flashinfer strided-input contract below.
                q_c, q_fp8, q_scales = fused_rmsnorm_fp8_quant_ue8m0(
                    q,
                    self.q_a_layernorm.weight,
                    eps=self.q_a_layernorm.variance_epsilon,
                )
                q = deepgemm_forward_quantized(self.q_b_proj, q_fp8, q_scales)
            else:
                # Pass the strided split view straight into the norm when the
                # backend accepts strided rows; this removes the .contiguous()
                # copy while keeping the norm math identical (the kernel reads
                # the same row elements via their stride and writes a fresh
                # contiguous output).
                q_c = self.q_a_layernorm(
                    q if self.q_a_layernorm.supports_strided_input else q.contiguous()
                )
                q = self.q_b_proj(q_c)
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

        # Strided split view into the norm when supported; see the q-side
        # comment above for the contiguity-copy removal rationale.
        compressed_kv = self.kv_a_layernorm(
            compressed_kv
            if self.kv_a_layernorm.supports_strided_input
            else compressed_kv.contiguous()
        )

        topk_indices = None
        if self.indexer is not None:
            topk_indices = self._run_sparse_indexer(
                hidden_states,
                q_c,
                q_view,
                indexer_kv_cache,
                indexer_fmha_impl,
            )
        attn_output = default_fmha_impl.forward(
            q_view,
            compressed_kv,
            k_pe,
            default_kv_cache,
            self.layer_idx,
            topk_indices,
        )

        if attn_output is not None:
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        else:
            attn_output = torch.zeros(
                (*input_shape, self.num_heads * self.v_head_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        attn_output = self.o_proj(attn_output)
        if self.parallelism_config.get_attn_tp_size() > 1:
            attn_output = all_reduce(attn_output, group=Group.TP)
        return attn_output
