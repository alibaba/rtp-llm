"""Shared building blocks for absorbed paged MLA decode backends."""

import os
from typing import Any, Dict, List, Optional

import torch

from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W

MLA_DECODE_KERNEL_ENV = "RTP_MLA_DECODE_KERNEL"
MLA_DECODE_KERNELS = ("auto", "flashinfer", "tokenspeed_mla", "trtllm_gen")


def get_mla_decode_kernel() -> str:
    """Return the exact requested decode backend or reject a misspelling."""
    kernel = os.environ.get(MLA_DECODE_KERNEL_ENV, "auto")
    if kernel not in MLA_DECODE_KERNELS:
        supported = ", ".join(MLA_DECODE_KERNELS)
        raise RuntimeError(
            f"invalid {MLA_DECODE_KERNEL_ENV}={kernel!r}; expected one of: {supported}"
        )
    return kernel


class PagedMlaDecodeMetadata:
    """Stable block-table and sequence-length buffers for paged MLA kernels."""

    def __init__(
        self,
        token_per_block: int,
        block_alignment_tokens: int,
        max_bs: int,
        max_context_len: int,
        use_cuda_graph: bool,
        device: torch.device,
    ) -> None:
        if token_per_block <= 0:
            raise ValueError(f"token_per_block must be positive, got {token_per_block}")
        if block_alignment_tokens <= 0:
            raise ValueError(
                "block_alignment_tokens must be positive, got "
                f"{block_alignment_tokens}"
            )
        if block_alignment_tokens % token_per_block != 0:
            raise ValueError(
                f"block alignment {block_alignment_tokens} is not divisible by "
                f"page size {token_per_block}"
            )

        self.token_per_block = token_per_block
        self.blocks_per_alignment = block_alignment_tokens // token_per_block
        self.max_context_len = max_context_len
        self.use_cuda_graph = use_cuda_graph
        self.device = device

        self.block_tables: Optional[torch.Tensor] = None
        self.seq_lens: Optional[torch.Tensor] = None
        self.column_indices: Optional[torch.Tensor] = None
        self.batch_size = 0
        self.padded_blocks = 0
        self.max_seq_len = 0

        if use_cuda_graph and max_bs > 0:
            max_blocks = max(
                1,
                (max_context_len + token_per_block - 1) // token_per_block,
            )
            self.ensure_capacity(max_bs, self.align_blocks(max_blocks))

    def align_blocks(self, num_blocks: int) -> int:
        align = self.blocks_per_alignment
        return (num_blocks + align - 1) // align * align

    def ensure_capacity(self, batch_size: int, padded_blocks: int) -> None:
        if (
            self.block_tables is None
            or self.block_tables.size(0) < batch_size
            or self.block_tables.size(1) < padded_blocks
        ):
            if self.use_cuda_graph and self.block_tables is not None:
                raise ValueError(
                    "paged MLA decode metadata cannot grow under CUDA graph: "
                    f"need ({batch_size}, {padded_blocks}), have "
                    f"{tuple(self.block_tables.shape)}"
                )
            self.block_tables = torch.zeros(
                (batch_size, padded_blocks), dtype=torch.int32, device=self.device
            )
            self.seq_lens = torch.zeros(
                batch_size, dtype=torch.int32, device=self.device
            )
        if (
            self.column_indices is None
            or self.column_indices.numel() < padded_blocks
            or self.column_indices.device != self.device
        ):
            if self.use_cuda_graph and self.column_indices is not None:
                raise ValueError(
                    "paged MLA decode column metadata cannot grow under CUDA graph"
                )
            self.column_indices = torch.arange(padded_blocks, device=self.device)

    def plan(self, fmha_params: Any) -> None:
        """Materialize dense block tables from RTP's compact FlashInfer metadata."""
        batch_size = fmha_params.qo_indptr_h.numel() - 1
        kv_lens = fmha_params.kvlen_h.tolist()
        max_seq_len = max(kv_lens) if kv_lens else 0
        needed_blocks = self.align_blocks(
            max(1, (max_seq_len + self.token_per_block - 1) // self.token_per_block)
        )

        if self.use_cuda_graph:
            if self.block_tables is None or self.block_tables.size(0) < batch_size:
                raise ValueError(
                    f"paged MLA graph metadata is too small for batch {batch_size}"
                )
            width = self.block_tables.size(1)
            if width < needed_blocks:
                raise ValueError(
                    f"paged MLA graph metadata needs {needed_blocks} blocks, has {width}"
                )
        else:
            self.ensure_capacity(batch_size, needed_blocks)
            width = needed_blocks

        assert self.block_tables is not None
        assert self.seq_lens is not None
        assert self.column_indices is not None
        page_indices = fmha_params.page_indice_d
        page_indptr = fmha_params.decode_page_indptr_d
        row_starts = page_indptr[:batch_size].view(-1, 1)
        row_sizes = (page_indptr[1 : batch_size + 1] - page_indptr[:batch_size]).view(
            -1, 1
        )
        columns = self.column_indices[:width].view(1, -1)
        source_indices = row_starts + columns
        dense_tables = self.block_tables[:batch_size, :width]
        if page_indices.numel() == 0:
            dense_tables.zero_()
        else:
            source_indices = source_indices.clamp_max(page_indices.numel() - 1)
            dense_tables.copy_(page_indices[source_indices])
            dense_tables.masked_fill_(columns >= row_sizes, 0)
        self.seq_lens[:batch_size].copy_(fmha_params.kvlen_d)
        self.batch_size = batch_size
        self.padded_blocks = width
        self.max_seq_len = max_seq_len

    def refresh_cuda_graph(
        self, block_table: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> None:
        """Refresh captured metadata in place from the selected cache group."""
        if not self.use_cuda_graph:
            raise RuntimeError("paged MLA graph metadata refresh requires CUDA graph")
        assert self.block_tables is not None
        assert self.seq_lens is not None
        assert self.column_indices is not None
        batch_size = self.batch_size
        width = self.padded_blocks
        src = block_table[:batch_size]
        if src.dim() != 2 or src.size(1) < width:
            raise RuntimeError(
                "paged MLA group refresh needs a block table of width "
                f">= {width}, got {tuple(src.shape)}"
            )
        kv_lens = sequence_lengths[:batch_size].to(torch.int32)
        live_blocks = (kv_lens + self.token_per_block - 1) // self.token_per_block
        dense_tables = self.block_tables[:batch_size, :width]
        dense_tables.copy_(src[:, :width])
        dense_tables.masked_fill_(
            self.column_indices[:width].view(1, -1) >= live_blocks.view(-1, 1),
            0,
        )
        self.seq_lens[:batch_size].copy_(kv_lens)


class AbsorbedPagedMlaDecodeOp:
    """Common absorbed-query and output projections around a paged MLA kernel."""

    def __init__(
        self,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        token_per_block: int,
        softmax_extra_scale: float,
        weights: List[Dict[str, torch.Tensor]],
        max_bs: int = 0,
        max_q_len: int = 1,
        is_cuda_graph: bool = False,
    ) -> None:
        if weights is None:
            raise RuntimeError("absorbed paged MLA decode requires projection weights")
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.token_per_block = token_per_block
        self.scale = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5
        self.bmm1_scale = self.scale * softmax_extra_scale
        self.weights = weights
        self.use_cuda_graph = is_cuda_graph
        self._q_absorbed: Optional[torch.Tensor] = None

        if is_cuda_graph and max_bs > 0:
            device = torch.device("cuda", torch.cuda.current_device())
            self._q_absorbed = torch.empty(
                (
                    max_bs * max_q_len,
                    num_heads,
                    kv_lora_rank + qk_rope_head_dim,
                ),
                dtype=torch.bfloat16,
                device=device,
            )

    def _absorb_query(
        self, q_nope: torch.Tensor, q_pe: torch.Tensor, layer_id: int
    ) -> torch.Tensor:
        q_nope = q_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        q_pe = q_pe.view(-1, self.num_heads, self.qk_rope_head_dim)
        num_tokens = q_nope.size(0)
        if (
            self._q_absorbed is None
            or self._q_absorbed.size(0) < num_tokens
            or self._q_absorbed.dtype != q_nope.dtype
            or self._q_absorbed.device != q_nope.device
        ):
            if self.use_cuda_graph and self._q_absorbed is not None:
                raise RuntimeError(
                    "absorbed MLA query buffer cannot grow or change dtype/device "
                    "under CUDA graph"
                )
            self._q_absorbed = torch.empty(
                (
                    num_tokens,
                    self.num_heads,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ),
                dtype=q_nope.dtype,
                device=q_nope.device,
            )
        q_absorbed = self._q_absorbed[:num_tokens]
        q_absorbed[..., self.kv_lora_rank :].copy_(q_pe)
        torch.bmm(
            q_nope.transpose(0, 1),
            self.weights[layer_id][W.mla_kc],
            out=q_absorbed[..., : self.kv_lora_rank].transpose(0, 1),
        )
        return q_absorbed

    def _view_paged_kv(self, kv_cache: Optional[LayerKVCache]) -> torch.Tensor:
        if kv_cache is None:
            raise RuntimeError("absorbed paged MLA decode requires KV cache")
        return kv_cache.kv_cache_base.view(
            -1,
            self.token_per_block,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )

    def _project_output(self, attn_output: torch.Tensor, layer_id: int) -> torch.Tensor:
        attn_output = attn_output.view(-1, self.num_heads, self.kv_lora_rank)
        output = torch.bmm(
            attn_output.transpose(0, 1), self.weights[layer_id][W.mla_vc]
        )
        return output.transpose(0, 1)


class PagedMlaDecodeImplMixin:
    """Shared CUDA Graph lifecycle for dense paged MLA decode implementations."""

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        self.prepare(attn_inputs, forbid_realloc=True)

    def prepare_cuda_graph_group(self, attn_inputs: PyAttentionInputs) -> None:
        assert self.fmha_impl is not None
        assert self.fmha_params is not None
        self.attn_inputs = attn_inputs
        sequence_lengths = getattr(attn_inputs, "sequence_lengths_plus_1_d", None)
        block_table = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
        backend_name = getattr(
            self.fmha_impl, "backend_name", type(self.fmha_impl).__name__
        )
        if sequence_lengths is None or sequence_lengths.numel() == 0:
            raise RuntimeError(
                f"{backend_name} MLA group refresh requires "
                "sequence_lengths_plus_1_d"
            )
        if block_table is None or block_table.numel() == 0:
            raise RuntimeError(
                f"{backend_name} MLA group refresh requires a device block table"
            )
        # Refresh the framework-owned decode tensors before the backend-specific
        # page metadata. In particular, this clears the host-planned
        # slot_mapping and records live positions/batch indices so KV writes use
        # the selected HybridCache group during graph replay.
        self.fmha_params.fill_decode_cuda_graph_params(
            sequence_lengths,
            block_table,
            self.seq_size_per_block,
        )
        self.fmha_impl.refresh_cuda_graph_metadata(
            self.fmha_params,
            block_table,
            sequence_lengths,
            self.seq_size_per_block,
        )
