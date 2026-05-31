from typing import Any

import torch

try:
    import flashinfer

    _has_flashinfer = True
except ImportError:
    _has_flashinfer = False

from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, rtp_llm_ops


def _has_fill_mla_params():
    return hasattr(rtp_llm_ops, "fill_mla_params")


class _TorchAppendParams:
    """Pure-Python params for append_paged_kv_cache when FlashInfer is unavailable.

    Handles both prefill and decode steps with correct RTP-LLM semantics:
    - Prefill: input_lengths = number of new tokens, prefix_lengths = write start
    - Decode:  1 new token per sequence, written at position sequence_lengths
    """

    def __init__(
        self,
        n_new_tokens: torch.Tensor,
        write_positions: torch.Tensor,
        kernel_block_id_host: torch.Tensor,
        token_per_block: int,
    ):
        device = torch.device("cuda")
        batch_size = n_new_tokens.size(0)

        # Vectorized construction of batch_indice_d and positions_d
        # avoiding per-element .item() calls that cause CPU-GPU sync
        batch_indices = torch.arange(batch_size, dtype=torch.int32).repeat_interleave(
            n_new_tokens.int()
        )
        offsets = torch.arange(n_new_tokens.max().item(), dtype=torch.int32).unsqueeze(
            0
        )  # [1, max_tokens]
        mask = offsets < n_new_tokens.int().unsqueeze(1)  # [batch, max_tokens]
        positions = (write_positions.int().unsqueeze(1) + offsets).masked_select(mask)

        self.batch_indice_d = batch_indices.to(device, non_blocking=True)
        self.positions_d = positions.to(device, non_blocking=True)

        # page_indice_d: flatten kernel block table
        self.page_indice_d = kernel_block_id_host.to(
            device=device, dtype=torch.int32, non_blocking=True
        ).reshape(batch_size, -1)

        # page_indptr and last_page_len
        kv_lengths = n_new_tokens + write_positions
        cum_kv = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        cum_kv[1:].copy_(torch.cumsum(kv_lengths, 0), non_blocking=True)
        self.prefill_ragged_kv_len_indptr_d = cum_kv
        self.decode_page_indptr_d = cum_kv
        self.paged_kv_last_page_len_d = ((kv_lengths - 1) % token_per_block + 1).to(
            dtype=torch.int32, device=device, non_blocking=True
        )


def _append_paged_kv_cache_vectorized(
    k: torch.Tensor,
    v: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    kv_cache_base: torch.Tensor,
    page_table: torch.Tensor,
):
    """Vectorized torch fallback for flashinfer.append_paged_kv_cache.

    Writes KV in vectorized layout matching _reshape_kv_cache_vectorized,
    so that mha_batch_prefill_func can read directly from the paged cache.

    kv_cache_base shape: [num_pages, 2, num_heads, page_size, head_dim]
    k, v shape: [total_tokens, num_heads, head_dim]
    page_table shape: [batch, max_pages_per_seq]
    """
    num_pages = kv_cache_base.shape[0]
    num_heads = kv_cache_base.shape[2]
    page_size = kv_cache_base.shape[3]
    head_dim = kv_cache_base.shape[4]
    vs = 16 // kv_cache_base.element_size()

    page_idx = positions // page_size
    in_page_offset = positions % page_size
    page_ids = page_table[batch_indices, page_idx]

    cache_flat = kv_cache_base.view(num_pages, 2, num_heads, page_size * head_dim)

    # K vectorized write: k_cache[page, head, :, offset, :] = k_vec
    k_cache_vec = cache_flat[:, 0].view(
        num_pages, num_heads, head_dim // vs, page_size, vs
    )
    k_vec = k.view(k.shape[0], num_heads, head_dim // vs, vs)
    k_cache_vec[page_ids, :, :, in_page_offset, :] = k_vec

    # V vectorized write: v_cache[page, head, offset//vs, :, offset%vs] = v
    v_cache_vec = cache_flat[:, 1].view(
        num_pages, num_heads, page_size // vs, head_dim, vs
    )
    v_cache_vec[page_ids, :, in_page_offset // vs, :, in_page_offset % vs] = v


class AppendKVCacheOpBase:
    def __init__(self, config: AttentionConfigs):
        self.token_per_block = config.kernel_tokens_per_block
        self._use_flashinfer = _has_flashinfer and _has_fill_mla_params()

    @staticmethod
    def _get_append_params(attn_inputs: PyAttentionInputs):
        """Derive (n_new_tokens, write_positions) from attn_inputs.

        - Prefill: input_lengths = new token count, prefix_lengths = write start
        - Decode:  input_lengths = total context length (NOT new token count),
                   sequence_lengths = current KV length, always 1 new token
        """
        if attn_inputs.is_prefill:
            return attn_inputs.input_lengths, attn_inputs.prefix_lengths
        batch_size = attn_inputs.sequence_lengths.size(0)
        n_new_tokens = torch.ones(batch_size, dtype=torch.int32)
        write_positions = attn_inputs.sequence_lengths
        return n_new_tokens, write_positions

    def create_params(self, attn_inputs: PyAttentionInputs) -> Any:
        # AppendKVCacheOp writes K/V into the paged cache via a GPU kernel and
        # therefore must consult the kernel-granularity block table. The
        # physical kv_cache_block_id_* table is reserved for cache-store ops
        # that run outside attention kernels (and outside the CUDA graph) and
        # is not populated by CudaGraphRunner during capture/replay.
        n_new_tokens, write_positions = self._get_append_params(attn_inputs)
        kernel_block_id_host = attn_inputs.kv_cache_kernel_block_id
        if self._use_flashinfer:
            params = rtp_llm_ops.fill_mla_params(
                write_positions,
                attn_inputs.sequence_lengths,
                n_new_tokens,
                kernel_block_id_host,
                self.token_per_block,
            )
            return params
        else:
            return _TorchAppendParams(
                n_new_tokens,
                write_positions,
                kernel_block_id_host,
                self.token_per_block,
            )

    def prepare(self, params: Any, attn_inputs: PyAttentionInputs):
        new_params = self.create_params(attn_inputs)
        params.batch_indice_d.copy_(new_params.batch_indice_d, non_blocking=True)
        params.positions_d.copy_(new_params.positions_d, non_blocking=True)
        params.page_indice_d.copy_(new_params.page_indice_d, non_blocking=True)
        if attn_inputs.is_prefill:
            params.prefill_ragged_kv_len_indptr_d.copy_(
                new_params.prefill_ragged_kv_len_indptr_d, non_blocking=True
            )
        else:
            params.decode_page_indptr_d.copy_(
                new_params.decode_page_indptr_d, non_blocking=True
            )
        params.paged_kv_last_page_len_d.copy_(
            new_params.paged_kv_last_page_len_d, non_blocking=True
        )

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache_base: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        params: Any,
    ):
        if self._use_flashinfer:
            flashinfer.append_paged_kv_cache(
                k,
                v,
                params.batch_indice_d,
                params.positions_d,
                kv_cache_base,
                params.page_indice_d,
                (
                    params.prefill_ragged_kv_len_indptr_d
                    if attn_inputs.is_prefill
                    else params.decode_page_indptr_d
                ),
                params.paged_kv_last_page_len_d,
                "HND",
            )
        else:
            _append_paged_kv_cache_vectorized(
                k,
                v,
                params.batch_indice_d,
                params.positions_d,
                kv_cache_base,
                params.page_indice_d,
            )


class AppendKVCacheOp(AppendKVCacheOpBase):
    def __init__(self, config: AttentionConfigs):
        super().__init__(config)

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: LayerKVCache,
        attn_inputs: PyAttentionInputs,
        params: Any,
    ):
        super().forward(k, v, kv_cache.kv_cache_base, attn_inputs, params)
