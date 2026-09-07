"""Live Decode positions, owner write slots and local causal bounds for Page-RR."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.factory.attention.common import mla_cache_block_table
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    decode_query_length,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_kernels import (
    _page_rr_coordinates,
)
from rtp_llm.ops.compute_ops import PyAttentionInputs


@triton.jit
def _prepare_decode_metadata(
    sequence_base,
    capture_lengths,
    table,
    positions,
    slots,
    local_lengths,
    rows,
    table_width,
    table_stride,
    queries: tl.constexpr,
    page_size: tl.constexpr,
    kernel_page_size: tl.constexpr,
    cp_size: tl.constexpr,
    cp_rank: tl.constexpr,
    multi_query_update: tl.constexpr,
    graph_decode: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0) * block + tl.arange(0, block)
    valid = row < rows
    batch = row // queries
    base = tl.load(sequence_base + batch, valid, other=0)
    if graph_decode:
        # The runner's synthetic q1 capture has plus_1=0, while sequence_lengths
        # describes its maximum request. Live replay, including padding, has >=1.
        capture_base = tl.load(capture_lengths + batch, valid, other=0) + 1
        base = tl.where(base == 0, capture_base, base)
    position = base + row % queries if multi_query_update else base - 1
    tl.store(positions + row, position, valid)

    # A full P*C interval contributes P keys to every owner. The remainder
    # contributes only its intersection with this rank's P-token page.
    end = tl.maximum(position + 1, 0)
    interval = page_size * cp_size
    length = end // interval * page_size + tl.minimum(
        tl.maximum(end % interval - cp_rank * page_size, 0), page_size
    )
    tl.store(local_lengths + row, length, valid)

    local_page, offset, owned = _page_rr_coordinates(
        position, page_size, kernel_page_size, cp_size, cp_rank
    )
    owned = valid & owned & (local_page >= 0) & (local_page < table_width)
    physical_page = tl.load(table + batch * table_stride + local_page, owned, other=-1)
    slot = physical_page.to(tl.int64) * kernel_page_size + offset
    tl.store(slots + row, tl.where(owned & (physical_page >= 0), slot, -1), valid)


class PageRRMlaDecodeMetadata:
    """Share prepared positions, write slots and causal bounds across MLA layers.

    Prepared buffers retain their addresses after capture. The original page
    table belongs to the runner; query_block_tables is the compact kernel input.
    """

    def __init__(
        self, page_size: int, kernel_page_size: int, cp_size: int, cp_rank: int
    ):
        if page_size <= 0 or kernel_page_size <= 0 or page_size % kernel_page_size:
            raise ValueError(
                "Page-RR ownership page must be a multiple of kernel page size"
            )
        if cp_size <= 1 or not 0 <= cp_rank < cp_size:
            raise ValueError(
                "Page-RR requires a valid rank in a CP group larger than one"
            )
        self.page_size = page_size
        self.kernel_page_size = kernel_page_size
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.positions_d = None
        self.slot_mapping = None
        self.local_causal_lens = None
        self.block_tables = None
        self.query_block_tables = None

    def prepare(
        self,
        inputs: PyAttentionInputs,
        forbid_realloc: bool = False,
        cache_group_id: int = 0,
    ) -> None:
        queries = decode_query_length(inputs)
        multi = inputs.is_target_verify or inputs.is_mtp_draft_update
        base = inputs.prefix_lengths if multi else inputs.sequence_lengths_plus_1_d
        table = mla_cache_block_table(inputs, cache_group_id)
        if (
            base is None
            or not base.is_cuda
            or base.dtype != torch.int32
            or base.ndim != 1
        ):
            raise ValueError("Page-RR Decode requires CUDA int32 per-request lengths")
        if (
            table is None
            or table.device != base.device
            or table.dtype != torch.int32
            or table.ndim != 2
            or table.shape[0] != base.numel()
            or table.stride(1) != 1
        ):
            raise ValueError(
                "Page-RR Decode requires a group-current CUDA int32 page table"
            )
        shape = (base.numel(), queries)
        if self.local_causal_lens is None or self.local_causal_lens.shape != shape:
            if forbid_realloc:
                raise ValueError(
                    "Page-RR Decode metadata shape cannot change during replay"
                )
            self.local_causal_lens = torch.empty(
                shape, dtype=torch.int32, device=base.device
            )
            self.positions_d = torch.empty(
                base.numel() * queries, dtype=torch.int32, device=base.device
            )
            self.slot_mapping = torch.empty_like(self.positions_d, dtype=torch.int64)
        self.block_tables = table
        rows = self.positions_d.numel()
        query_table_shape = (rows, table.shape[1])
        if (
            self.query_block_tables is None
            or self.query_block_tables.shape != query_table_shape
        ):
            if forbid_realloc:
                raise ValueError(
                    "Page-RR Decode query page table shape cannot change during replay"
                )
            self.query_block_tables = torch.empty(
                query_table_shape, dtype=torch.int32, device=base.device
            )
        self.query_block_tables.view(base.numel(), queries, table.shape[1]).copy_(
            table[:, None, :]
        )
        _prepare_decode_metadata[(triton.cdiv(rows, 128),)](
            base,
            inputs.sequence_lengths if inputs.is_cuda_graph and not multi else base,
            table,
            self.positions_d,
            self.slot_mapping,
            self.local_causal_lens,
            rows,
            table.shape[1],
            table.stride(0),
            queries=queries,
            page_size=self.page_size,
            kernel_page_size=self.kernel_page_size,
            cp_size=self.cp_size,
            cp_rank=self.cp_rank,
            multi_query_update=multi,
            graph_decode=inputs.is_cuda_graph and not multi,
            block=128,
        )
