"""Torch Naive Attention Backend - Fallback implementation using PyTorch + flash_attn.

Provides Prefill / Decode base impls used as fallbacks (and as parents for the
PQ-accelerated impls in torch_naive_pq.py).

Reference: SGLang's torch_native_backend.py
"""

import logging
from typing import Optional

import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOpQKVOut,
    KVCache,
    PyAttentionInputs,
)

# ============================================================================
# Dummy FMHA Params for Interface Compatibility
# ============================================================================


class DummyFMHAParams:
    """Dummy FMHA params for TorchNaive — satisfies PyModelOutputs.fmha_params."""

    def fill_params(
        self,
        sequence_lengths,
        input_lengths,
        kv_cache_block_id_host,
        batch_size,
        seq_size_per_block,
    ):
        pass


# ============================================================================
# FP4 E2M1 Simulated Quantization (used by Decode forward)
# ============================================================================

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FP4_BLOCK_SIZE = 16


def _round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """就近舍入到 FP4 E2M1 的 16 个离散值。"""
    sign = torch.sign(x)
    a = torch.abs(x)
    out = torch.zeros_like(a)
    out[a > 0.25] = 0.5
    out[a >= 0.75] = 1.0
    out[a > 1.25] = 1.5
    out[a >= 1.75] = 2.0
    out[a > 2.5] = 3.0
    out[a >= 3.5] = 4.0
    out[a > 5.0] = 6.0
    return out * sign


def _fp4_simulate_quant(x: torch.Tensor) -> torch.Tensor:
    """FP4 E2M1 模拟量化：quantize → dequantize，保持原始 dtype。"""
    orig_shape = x.shape
    orig_dtype = x.dtype
    x_2d = x.reshape(-1, orig_shape[-1])
    m, n = x_2d.shape

    tensor_amax = torch.abs(x_2d).max()
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax.clamp(min=1e-12)

    x_blocked = x_2d.reshape(m, n // FP4_BLOCK_SIZE, FP4_BLOCK_SIZE)
    vec_max = x_blocked.abs().amax(dim=-1, keepdim=True)

    scale = global_scale * (vec_max / FLOAT4_E2M1_MAX)
    scale = scale.to(torch.float8_e4m3fn).to(orig_dtype)

    output_scale = 1.0 / (scale / global_scale).clamp(min=1e-12)
    scaled_x = x_blocked * output_scale
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0)

    quantized = _round_to_e2m1(clipped_x)
    dequantized = quantized / output_scale

    return dequantized.reshape(orig_shape)


# ============================================================================
# Prefill
# ============================================================================


class TorchNaivePrefillImpl(FMHAImplBase):
    """Torch Naive Prefill — flash_attn_varlen + RoPE + write cache."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs

        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scaling = 1.0 / (self.head_dim**0.5)
        self.enable_gqa = self.num_heads != self.num_kv_heads

        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQKVOut(attn_configs)

        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)

        self.fmha_params = DummyFMHAParams()

        logging.debug(
            f"TorchNaivePrefillImpl initialized: heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, gqa={self.enable_gqa}"
        )

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        self.attn_inputs = attn_inputs
        batch_size = attn_inputs.input_lengths.size(0)
        cu_seqlens = attn_inputs.cu_seqlens[: batch_size + 1]
        self._max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        new_offset = new_rope_params.kv_cache_offset
        old_offset = self.rope_params.kv_cache_offset
        common.copy_kv_cache_offset(old_offset, new_offset)

    @classmethod
    def support(
        cls,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ) -> bool:
        if attn_configs.use_mla:
            return False
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if self.need_rope_kv_cache:
            qkv = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)

        q, k, v = self._split_qkv(qkv)

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        output = self._run_attention_extend(q, k, v)
        output = output.reshape(output.shape[0], -1)
        return output

    def _split_qkv(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = qkv.reshape(qkv.shape[0], -1)

        q, k, v = torch.split(
            qkv,
            [
                self.head_dim * self.num_heads,
                self.head_dim * self.num_kv_heads,
                self.head_dim * self.num_kv_heads,
            ],
            dim=-1,
        )

        q = q.reshape(q.shape[0], self.num_heads, self.head_dim)
        k = k.reshape(k.shape[0], self.num_kv_heads, self.head_dim)
        v = v.reshape(v.shape[0], self.num_kv_heads, self.head_dim)

        return q, k, v

    def _run_attention_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = self.attn_inputs.input_lengths.size(0)
        cu_seqlens = self.attn_inputs.cu_seqlens[: batch_size + 1].to(torch.int32)
        max_seqlen = getattr(self, "_max_seqlen", None)
        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        if not (q.dtype == k.dtype == v.dtype):
            k = k.to(q.dtype)
            v = v.to(q.dtype)

        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
            softmax_scale=self.scaling,
        )

        return output


# ============================================================================
# Decode
# ============================================================================


class TorchNaiveDecodeImpl(FMHAImplBase):
    """Torch Naive Decode — RoPE + read cache (+ FP4 sim quant) + flash_attn."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs

        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scaling = 1.0 / (self.head_dim**0.5)
        self.enable_gqa = self.num_heads != self.num_kv_heads
        self.tokens_per_block = attn_configs.kernel_tokens_per_block

        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOp(attn_configs)

        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)

        self.fmha_params = DummyFMHAParams()

        logging.debug(
            f"TorchNaiveDecodeImpl initialized: heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, gqa={self.enable_gqa}"
        )

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        self.attn_inputs = attn_inputs
        from rtp_llm.ops.compute_ops import fill_mla_params

        batch_size = attn_inputs.input_lengths.size(0)
        params = fill_mla_params(
            (
                attn_inputs.prefix_lengths
                if getattr(attn_inputs, "prefix_lengths", None) is not None
                else torch.tensor([], dtype=torch.int32)
            ),
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            (
                attn_inputs.kv_cache_block_id_host
                if attn_inputs.kv_cache_block_id_host is not None
                else torch.tensor([], dtype=torch.int32)
            ),
            self.tokens_per_block,
        )
        self._max_seq_len_decode = params.kvlen_h[:batch_size].max().item()
        new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        new_offset = new_rope_params.kv_cache_offset
        old_offset = self.rope_params.kv_cache_offset
        common.copy_kv_cache_offset(old_offset, new_offset)

    @classmethod
    def support(
        cls,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ) -> bool:
        if attn_configs.use_mla:
            return False
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        # Decode RoPE writes K,V to cache directly and only returns Q
        if self.need_rope_kv_cache:
            q = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)

            if q.ndim == 2:
                q = q.reshape(q.shape[0], self.num_heads, self.head_dim)
            elif q.ndim == 3:
                pass
            elif q.ndim == 4:
                pass
            else:
                raise ValueError(f"Unexpected Q shape from RoPE: {q.shape}")
        else:
            q, k, v = self._split_qkv(qkv)

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        k_full, v_full = self._read_kv_from_cache(kv_cache)

        k_full = _fp4_simulate_quant(k_full)
        v_full = _fp4_simulate_quant(v_full)

        output = self._run_attention_decode(q, k_full, v_full)
        output = output.reshape(output.shape[0], -1)
        return output

    def _split_qkv(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = qkv.reshape(qkv.shape[0], -1)

        expected_size = self.head_dim * (self.num_heads + 2 * self.num_kv_heads)
        actual_size = qkv.shape[-1]
        if expected_size != actual_size:
            logging.error(
                f"QKV size mismatch in {self.__class__.__name__}: "
                f"expected {expected_size}, got {actual_size}"
            )
            actual_qkv_heads = actual_size // self.head_dim
            if actual_qkv_heads < 2 * self.num_kv_heads:
                raise ValueError(
                    f"QKV size {actual_size} too small for kv_heads={self.num_kv_heads}"
                )

        q, k, v = torch.split(
            qkv,
            [
                self.head_dim * self.num_heads,
                self.head_dim * self.num_kv_heads,
                self.head_dim * self.num_kv_heads,
            ],
            dim=-1,
        )

        q = q.reshape(q.shape[0], self.num_heads, self.head_dim)
        k = k.reshape(k.shape[0], self.num_kv_heads, self.head_dim)
        v = v.reshape(v.shape[0], self.num_kv_heads, self.head_dim)

        return q, k, v

    def _read_kv_from_cache(
        self, kv_cache: KVCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read complete K, V from paged KV cache (vectorized, CUDA-graph safe)."""
        batch_size = self.attn_inputs.input_lengths.size(0)

        from rtp_llm.ops.compute_ops import fill_mla_params

        params = fill_mla_params(
            (
                self.attn_inputs.prefix_lengths
                if getattr(self.attn_inputs, "prefix_lengths", None) is not None
                else torch.tensor([], dtype=torch.int32)
            ),
            self.attn_inputs.sequence_lengths,
            self.attn_inputs.input_lengths,
            (
                self.attn_inputs.kv_cache_block_id_host
                if self.attn_inputs.kv_cache_block_id_host is not None
                else torch.tensor([], dtype=torch.int32)
            ),
            self.tokens_per_block,
        )

        sequence_lengths = params.kvlen_h[:batch_size]
        max_seq_len = getattr(self, "_max_seq_len_decode", None)
        if max_seq_len is None:
            max_seq_len = sequence_lengths.max().item()

        kv_cache_base = kv_cache.kv_cache_base

        tpb = self.tokens_per_block
        if kv_cache_base.ndim == 2:
            block_num = kv_cache_base.shape[0]
            expected_elems = 2 * self.num_kv_heads * tpb * self.head_dim
            kv_cache_tensor = kv_cache_base[:, :expected_elems].reshape(
                block_num,
                2,
                self.num_kv_heads,
                tpb,
                self.head_dim,
            )
        else:
            kv_cache_tensor = kv_cache_base

        if self.attn_inputs.kv_cache_block_id_host is None:
            k_full = torch.zeros(
                batch_size,
                max_seq_len,
                self.num_kv_heads,
                self.head_dim,
                dtype=kv_cache_tensor.dtype,
                device=kv_cache_tensor.device,
            )
            return k_full, torch.zeros_like(k_full)

        block_indices = self.attn_inputs.kv_cache_block_id_host[0][:batch_size, :]
        max_blocks_per_seq = block_indices.shape[1]

        num_blocks_per_seq = (sequence_lengths + tpb - 1) // tpb
        block_range = torch.arange(max_blocks_per_seq, device=block_indices.device)
        valid_mask = block_range.unsqueeze(0) < num_blocks_per_seq.unsqueeze(1)

        flat_block_ids = block_indices[valid_mask]

        gathered = kv_cache_tensor[flat_block_ids.to(kv_cache_tensor.device)]

        batch_idx_per_block = (
            torch.arange(batch_size, device=block_indices.device)
            .unsqueeze(1)
            .expand_as(valid_mask)[valid_mask]
        )
        block_idx_per_block = block_range.unsqueeze(0).expand_as(valid_mask)[valid_mask]

        gathered = gathered.permute(0, 1, 3, 2, 4)

        k_padded = torch.zeros(
            batch_size,
            max_blocks_per_seq,
            tpb,
            self.num_kv_heads,
            self.head_dim,
            dtype=kv_cache_tensor.dtype,
            device=kv_cache_tensor.device,
        )
        v_padded = torch.zeros_like(k_padded)

        batch_idx_gpu = batch_idx_per_block.to(kv_cache_tensor.device)
        block_idx_gpu = block_idx_per_block.to(kv_cache_tensor.device)
        k_padded[batch_idx_gpu, block_idx_gpu] = gathered[:, 0]
        v_padded[batch_idx_gpu, block_idx_gpu] = gathered[:, 1]

        k_full = k_padded.reshape(
            batch_size, max_blocks_per_seq * tpb, self.num_kv_heads, self.head_dim
        )[:, :max_seq_len]
        v_full = v_padded.reshape(
            batch_size, max_blocks_per_seq * tpb, self.num_kv_heads, self.head_dim
        )[:, :max_seq_len]

        return k_full, v_full

    def _run_attention_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if q.ndim == 3:
            q = q.unsqueeze(1)

        if not (q.dtype == k.dtype == v.dtype):
            k = k.to(q.dtype)
            v = v.to(q.dtype)

        output = flash_attn_func(
            q,
            k,
            v,
            causal=False,
            softmax_scale=self.scaling,
        )

        return output.squeeze(1)
