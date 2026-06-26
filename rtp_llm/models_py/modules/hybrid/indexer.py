import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.modules import IndexerOp, LayerNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import AttentionConfigs, HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import KVCache
from rtp_llm.utils.model_weight import W

_INDEXER_QUANT_DTYPE_ENV = "RTP_LLM_INDEXER_QUANT_DTYPE"
_SUPPORTED_INDEXER_QUANT_DTYPES = ("fp8", "fp4")
_BLACKWELL_MIN_CC = (10, 0)

# One-time PD-consistency banner. The indexer KV cache stride is decided at
# config time on each instance; if the prefill instance resolves "fp4" and the
# decode instance resolves "fp8" (or vice versa), the RDMA-transferred blocks
# carry mismatched bytes-per-token and the decode side reads silently
# corrupt logits. Until the GetPeerInfo RPC (model_rpc_service.proto:531)
# carries this field for a real handshake check, log it loudly so an operator
# can grep PD pairs for a mismatch:
#   grep -h 'INDEXER_QUANT_BANNER' prefill.log decode.log
_PD_BANNER_EMITTED = False


def _emit_pd_consistency_banner(quant_dtype: str) -> None:
    """Log resolved indexer dtype with role_type so prefill/decode logs can
    be grep-compared. Only logs once per process."""
    global _PD_BANNER_EMITTED
    if _PD_BANNER_EMITTED:
        return
    _PD_BANNER_EMITTED = True
    role = os.environ.get("ROLE_TYPE", "<unset>")
    bytes_per_token = 68 if quant_dtype == "fp4" else 132  # HD=128
    logging.info(
        "INDEXER_QUANT_BANNER role=%s indexer_quant_dtype=%s "
        "kv_stride_bytes_per_token=%d (must match peer in PD pair)",
        role,
        quant_dtype,
        bytes_per_token,
    )


def _resolve_indexer_quant_dtype() -> str:
    raw = os.environ.get(_INDEXER_QUANT_DTYPE_ENV, "fp8").strip().lower()
    if raw not in _SUPPORTED_INDEXER_QUANT_DTYPES:
        raise ValueError(
            f"{_INDEXER_QUANT_DTYPE_ENV}={raw!r} not supported; "
            f"expected one of {_SUPPORTED_INDEXER_QUANT_DTYPES}"
        )
    if raw == "fp4":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{_INDEXER_QUANT_DTYPE_ENV}=fp4 requires CUDA; no CUDA device available"
            )
        cc = torch.cuda.get_device_capability()
        if cc < _BLACKWELL_MIN_CC:
            raise RuntimeError(
                f"{_INDEXER_QUANT_DTYPE_ENV}=fp4 requires Blackwell (SM>=10.0); "
                f"current device capability {cc}"
            )
    _emit_pd_consistency_banner(raw)
    return raw


_DEVICE_TYPE = get_device_type()
if _DEVICE_TYPE == DeviceType.Cuda:
    from rtp_llm.models_py.triton_kernels.sparse_mla.fused_logits_head_gate import (
        fused_logits_head_gate,
    )
else:
    fused_logits_head_gate = None  # type: ignore


class Indexer(nn.Module):
    """
    Indexer for DeepSeek-V3.2 DSA (DeepSeek Sparse Attention) mechanism.
    Adapted from sglang's Indexer implementation.
    """

    def __init__(
        self,
        attn_config: AttentionConfigs,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        layernorm_eps: float,
        quant_config: object,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        parallelism_config: Optional[ParallelismConfig] = None,
        scale_fmt: Optional[str] = "none",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        from rtp_llm.models_py.utils.fuse_config import fuse_kernels_enabled

        # Resolve once at init: HWKernelConfig.enable_fuse_kernels (or env
        # ``ENABLE_FUSE_KERNELS``) → ``self._fuse_logits_head_gate``. Keep it
        # out of the forward path so it's free at decode (no env / config
        # lookup per token).
        self._quant_dtype = _resolve_indexer_quant_dtype()

        # Fused-logits-head-gate is FP8-only (its kernel folds q_scale into
        # the gate, which FP4 doesn't have as fp32). The fused QK path has
        # separate FP8 (``fused_rope_quant_qk``) and FP4
        # (``fused_rope_quant_qk_fp4``) implementations — both run K's
        # RoPE+Hadamard and Q's RoPE+Hadamard+quant in one Triton launch.
        # Both honor the same ``enable_fuse_kernels`` master switch.
        _fuse_kernels = fuse_kernels_enabled(hw_kernel_config)
        self._fuse_logits_head_gate = (
            self._quant_dtype == "fp8"
            and _fuse_kernels
            and fused_logits_head_gate is not None
        )
        # FP4 decode-path fused QK kernel (one launch): K(RoPE+Had→bf16) +
        # Q(RoPE+Had+FP4 e2m1+UE8M0/32). Mirrors the FP8 path's structure;
        # K is then cache-written via ``quant_k_only``.
        self._fuse_q_rope_quant_fp4 = self._quant_dtype == "fp4" and _fuse_kernels

        self.index_n_heads = attn_config.indexer_head_num
        self.index_head_dim = attn_config.indexer_head_dim
        self.index_topk = attn_config.indexer_topk

        self.rope_head_dim = attn_config.rope_head_dim
        self.block_size = 128  # quantization block size (128)
        self.head_kv = 1
        self.scale_fmt = scale_fmt  # FP8 quantization format
        self.softmax_scale = self.index_head_dim**-0.5
        self.weights_scale = self.index_n_heads**-0.5
        self.blocksize = attn_config.kernel_tokens_per_block  # page size, typically 64
        # Owner (physical) block size used by the C++ KVCacheAllocator / CPSlotMapper
        # to decide page-RR ownership. bpk = owner_tpb / kernel_tpb >= 1. Mirrors the
        # DSV4 indexer's _kv_owner_tokens_per_block contract; threaded into
        # _get_topk_ragged_cp so build_indexer_cp_chunk_plan computes per-rank padded
        # local KV lens and restore indices at the owner granularity that matches
        # how prefill writes were laid out via cp_params.sharded_slot_mapping.
        kernel_tpb = int(attn_config.kernel_tokens_per_block)
        owner_tpb = int(getattr(attn_config, "tokens_per_block", kernel_tpb))
        if owner_tpb <= 0 or kernel_tpb <= 0 or owner_tpb % kernel_tpb != 0:
            owner_tpb = kernel_tpb
        self._kv_owner_tokens_per_block = owner_tpb
        self.indexer_size = self.index_head_dim / 2 + self.index_head_dim / 128 * 2
        self.is_neox_style = attn_config.rope_config.indexer_is_neox_style
        self.parallelism_config = parallelism_config

        self.wq_b = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_qb_w,
            W.mla_indexer_qb_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        self.wk = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_k_w,
            W.mla_indexer_k_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        self.k_norm = LayerNorm(
            weights[W.mla_indexer_k_norm_w],
            weights[W.mla_indexer_k_norm_b],
            eps=layernorm_eps,
        )

        self.weights_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_weights_proj_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        # Pre-contiguify weight for the fused Triton kernel (one-time init
        # copy). Production weight is often a transposed view [N, K] of
        # underlying [K, N] storage; the small-T per-(t,n) kernel needs
        # contiguous weight for coalesced 1D loads. Use plain attribute
        # reassignment (not `.data = ...`) — the latter does an in-place
        # `set_` of the underlying storage that leaves the tensor in a
        # state where `F.linear` under cuda-graph capture + inference_mode
        # trips PyTorch's version-counter check.
        if (
            self._fuse_logits_head_gate
            and hasattr(self.weights_proj, "weight")
            and not self.weights_proj.weight.is_contiguous()
        ):
            self.weights_proj.weight = self.weights_proj.weight.contiguous()
        self.cos_sin_cache = global_weights[W.rope_cos_sin_cache]

        self.indexer_op = IndexerOp(
            index_n_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
            index_topk=self.index_topk,
            rope_head_dim=self.rope_head_dim,
            cos_sin_cache=self.cos_sin_cache,
            blocksize=self.blocksize,
            block_size=self.block_size,
            scale_fmt=self.scale_fmt,
            is_neox_style=self.is_neox_style,
            quant_dtype=self._quant_dtype,
        )

    def _prefill_cp_enabled(self) -> bool:
        if self.parallelism_config is None:
            return False
        return self.parallelism_config.prefill_cp_config.is_enabled()

    def _is_sparse_prefill_cp(self, attention_inputs: Any) -> bool:
        return bool(attention_inputs.is_prefill) and self._prefill_cp_enabled()

    def _get_logits_head_gate(
        self, x: torch.Tensor, q_scale: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # F3: fused (cast + GEMV + 2 elementwise muls) into one Triton kernel.
        # ``self._fuse_logits_head_gate`` is resolved at __init__ from
        # ``HWKernelConfig.enable_fuse_kernels`` and FP8 mode.
        scale = self.softmax_scale * self.weights_scale
        if self._fuse_logits_head_gate and x.is_contiguous() and q_scale is not None:
            return fused_logits_head_gate(
                x,
                q_scale,
                self.weights_proj.weight,
                scale,
                fallback_proj=self.weights_proj,
            )
        x = x.float()
        weights = self.weights_proj(x)
        weights = weights.float()
        if q_scale is None:
            # FP4 path: per-token/per-group Q scales are consumed inside
            # deep_gemm.fp8_fp4_*mqa_logits via the q tuple, so do not fold
            # them into ``weights`` here.
            weights = weights * scale
        else:
            weights = weights.unsqueeze(-1) * q_scale * scale
        return weights

    def _fused_forward_decode(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
        q_c_fp8: Optional[torch.Tensor] = None,
        q_c_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused decode: QK kernel does K(RoPE+Had→bf16) + Q(RoPE+Had+FP8).

        Returns (q_fp8, q_scale).
        """
        if q_c_fp8 is not None and q_c_scale is not None:
            q = self.wq_b(q_c_fp8, input_scales=q_c_scale)
        else:
            q = self.wq_b(q_lora)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)

        if x_fp8 is not None and x_scale is not None:
            k = self.wk(x_fp8, input_scales=x_scale)
        else:
            k = self.wk(x)
        k = self.k_norm(k)

        q_fp8, q_scale, key = self.indexer_op.fused_rope_quant_qk(
            q, k, fmha_params.positions_d
        )
        self.indexer_op.quant_k_only(key, kv_cache, fmha_params.slot_mapping)

        return q_fp8, q_scale

    def _fused_forward_decode_fp4(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
        q_c_fp8: Optional[torch.Tensor] = None,
        q_c_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """FP4 fused decode: same shape as ``_fused_forward_decode`` —
        one fused QK kernel + one K-cache write. The only difference vs
        FP8 is Q's final quantization (FP4 e2m1 + UE8M0/32 instead of
        FP8 e4m3 + UE8M0/HD scalar).

        Returns ``(q_fp4, q_scale_fp4)``:
          q_fp4:       int8  [N, n_heads, HD//2]   (2 FP4 nibbles/byte)
          q_scale_fp4: int32 [N, n_heads]          (4 packed UE8M0 bytes)
        """
        if q_c_fp8 is not None and q_c_scale is not None:
            q = self.wq_b(q_c_fp8, input_scales=q_c_scale)
        else:
            q = self.wq_b(q_lora)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)

        if x_fp8 is not None and x_scale is not None:
            k = self.wk(x_fp8, input_scales=x_scale)
        else:
            k = self.wk(x)
        k = self.k_norm(k)

        q_fp4, q_scale_fp4, key = self.indexer_op.fused_rope_quant_qk_fp4(
            q, k, fmha_params.positions_d
        )
        self.indexer_op.quant_k_only(key, kv_cache, fmha_params.slot_mapping)

        return q_fp4, q_scale_fp4

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        flashmla_params: Any,
        cp_params: Optional[Any],
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
        q_c_fp8: Optional[torch.Tensor] = None,
        q_c_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q_c_fp8 is not None and q_c_scale is not None:
            q = self.wq_b(q_c_fp8, input_scales=q_c_scale)
        else:
            q = self.wq_b(q_lora)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)

        if x_fp8 is not None and x_scale is not None:
            k = self.wk(x_fp8, input_scales=x_scale)
        else:
            k = self.wk(x)
        k = self.k_norm(k)

        if self._prefill_cp_enabled():
            assert cp_params is not None
            query, key = self.indexer_op.apply_rope_and_rotate_q_k_cp(
                q,
                k,
                cp_params.full_rope_pos_ids,
            )
        else:
            positions = flashmla_params.positions_d
            query, key = self.indexer_op.apply_rope_and_rotate_q_k(q, k, positions)

        return query, key

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        flashmla_params: Any,
    ) -> torch.Tensor:
        k = self.wk(x)
        k = self.k_norm(k)
        return self.indexer_op.apply_rope_and_rotate_k(k, flashmla_params.positions_d)

    def _quantize_q_k(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
        cp_params: Optional[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._is_sparse_prefill_cp(attention_inputs):
            assert cp_params is not None
            slot_mapping = (
                cp_params.sharded_slot_mapping
                if bool(getattr(cp_params, "kv_cache_sharded", False))
                else fmha_params.slot_mapping
            )
            return self.indexer_op.quant_q_k_cp(
                query,
                key,
                kv_cache,
                slot_mapping,
                cp_params.kv_restore_unpad_indices,
            )
        return self.indexer_op.quant_q_k(query, key, kv_cache, fmha_params.slot_mapping)

    def _compute_topk(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
        cp_params: Optional[Any],
        q_scale_fp4: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not attention_inputs.is_prefill or bool(
            getattr(attention_inputs, "is_target_verify", False)
        ):
            return self.indexer_op._get_topk_paged(
                q_fp8,
                weights,
                kv_cache,
                fmha_params,
                attention_inputs,
                q_scale_fp4=q_scale_fp4,
            )
        if self._prefill_cp_enabled():
            assert cp_params is not None
            return self.indexer_op._get_topk_ragged_cp(
                q_fp8,
                weights,
                kv_cache,
                fmha_params,
                attention_inputs,
                cp_params.total_local_ids,
                cp_params.cu_kv_seqlens_global,
                cp_params.total_kv_len,
                cp_params.precomputed_ks,
                cp_params.precomputed_ke,
                cp_params.precomputed_lengths,
                cp_params.precomputed_topk_off,
                bool(getattr(cp_params, "kv_cache_sharded", False)),
                int(getattr(cp_params, "cp_size", 1)),
                int(getattr(cp_params, "cp_rank", 0)),
                kv_owner_tokens_per_block=self._kv_owner_tokens_per_block,
                q_scale_fp4=q_scale_fp4,
            )
        return self.indexer_op._get_topk_ragged(
            q_fp8,
            weights,
            kv_cache,
            fmha_params,
            attention_inputs,
            q_scale_fp4=q_scale_fp4,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
        use_fast_path: bool,
        cp_params: Any = None,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
        q_c_fp8: Optional[torch.Tensor] = None,
        q_c_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_fast_path:
            key = self._get_k_bf16(hidden_states, fmha_params)
            self.indexer_op.quant_k_only(key, kv_cache, fmha_params.slot_mapping)
            return None

        if self._is_sparse_prefill_cp(attention_inputs):
            assert cp_params is not None, "cp_params is required for sparse prefill CP"

        # Decode-only fused QK paths. FP8 uses ``fused_rope_quant_qk``
        # (returns FP8 e4m3 Q + fp32 scale); FP4 uses
        # ``fused_rope_quant_qk_fp4`` (returns FP4 e2m1 Q + packed UE8M0/32
        # int32 scale). Both produce a bf16 K and bypass the slow unfused
        # ``apply_rope_and_rotate_q_k`` + Python quant path.
        _decode_eligible = not attention_inputs.is_prefill and cp_params is None
        if _decode_eligible and self._fuse_logits_head_gate:
            q_fp8, q_scale = self._fused_forward_decode(
                q_lora,
                hidden_states,
                kv_cache,
                fmha_params,
                x_fp8,
                x_scale,
                q_c_fp8,
                q_c_scale,
            )
        elif _decode_eligible and self._fuse_q_rope_quant_fp4:
            q_fp8, q_scale = self._fused_forward_decode_fp4(
                q_lora,
                hidden_states,
                kv_cache,
                fmha_params,
                x_fp8,
                x_scale,
                q_c_fp8,
                q_c_scale,
            )
        else:
            query, key = self._get_q_k_bf16(
                q_lora,
                hidden_states,
                fmha_params,
                cp_params,
                x_fp8,
                x_scale,
                q_c_fp8,
                q_c_scale,
            )
            q_fp8, q_scale = self._quantize_q_k(
                query, key, kv_cache, fmha_params, attention_inputs, cp_params
            )

        # FP4 mode: q_scale is packed UE8M0 int32 [N, n_heads] for the
        # deep_gemm fp8_fp4_*mqa_logits q tuple; weights are computed without
        # q_scale absorption. FP8 mode: q_scale is float32 [N, n_heads, 1]
        # and is folded into weights via the gate.
        if self._quant_dtype == "fp4":
            weights = self._get_logits_head_gate(hidden_states, None)
            return self._compute_topk(
                q_fp8,
                weights,
                kv_cache,
                fmha_params,
                attention_inputs,
                cp_params,
                q_scale_fp4=q_scale,
            )
        weights = self._get_logits_head_gate(hidden_states, q_scale)
        return self._compute_topk(
            q_fp8, weights, kv_cache, fmha_params, attention_inputs, cp_params
        )
