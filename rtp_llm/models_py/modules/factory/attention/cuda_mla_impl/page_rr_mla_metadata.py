"""Backend-neutral Page-RR metadata prepared through MLA attention hooks."""

from itertools import accumulate

import torch
import triton

from rtp_llm.models_py.triton_kernels.mla.page_rr import (
    _prepare_decode_metadata,
    _prepare_prefill_metadata,
)


class PageRRMlaMetadata:
    """Shared Page-RR layout with eager Prefill and graph-stable Decode entry points.

    Both FlashMLA BF16 and TokenSpeed FP8 consume the Prefill ragged offsets;
    backend-independent cache slots use the same ownership rule in every phase.
    """

    def __init__(
        self, page_size: int, kernel_page_size: int, cp_size: int, cp_rank: int,
    ):
        if page_size <= 0 or kernel_page_size <= 0 or page_size % kernel_page_size:
            raise ValueError("Page-RR ownership page must be a multiple of kernel page size")
        if cp_size <= 1 or not 0 <= cp_rank < cp_size:
            raise ValueError("Page-RR requires a valid rank in a multi-rank TP group")
        self.page_size = page_size
        self.kernel_page_size = kernel_page_size
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.positions_d = None
        self.slot_mapping = None
        self.local_causal_lens = None
        self.block_tables = None
        self.query_block_tables = None

    def prepare_prefill(self, inputs) -> None:
        """Reuse framework request metadata; add Page-RR owner-only cache slots."""
        if inputs.prefix_lengths is None:
            raise ValueError("MLA Page-RR Prefill requires prefix lengths")
        q_lens = [int(value) for value in inputs.input_lengths.tolist()]
        prefix_lens = [int(value) for value in inputs.prefix_lengths.tolist()]
        if not q_lens or len(q_lens) != len(prefix_lens):
            raise ValueError("MLA Page-RR requires matched nonempty Q/prefix lengths")
        if any(length < 0 for length in q_lens + prefix_lens):
            raise ValueError("MLA Page-RR token lengths must be non-negative")
        kv_lens = [q + prefix for q, prefix in zip(q_lens, prefix_lens)]
        q_offsets = (0, *accumulate(q_lens))
        kv_offsets = (0, *accumulate(kv_lens))
        if max(q_offsets[-1], kv_offsets[-1]) > torch.iinfo(torch.int32).max:
            raise ValueError("MLA Page-RR token offsets exceed int32 storage")
        lengths = inputs.input_lengths_device
        prefixes = inputs.prefix_lengths_device
        qo_indptr = inputs.cu_seqlens_device
        table = inputs.kv_cache_kernel_block_id_device
        if lengths is None or not lengths.is_cuda:
            raise ValueError("MLA Page-RR requires CUDA attention metadata")
        device = lengths.device
        if (prefixes is None or prefixes.device != device or prefixes.dtype != torch.int32
                or prefixes.shape != (len(q_lens),) or not prefixes.is_contiguous()
                or qo_indptr is None or qo_indptr.device != device
                or qo_indptr.shape != (len(q_lens) + 1,)
                or qo_indptr.dtype not in (torch.int32, torch.int64)):
            raise ValueError("MLA Page-RR requires matching CUDA integer request metadata")
        if (table is None or table.device != device or table.dtype != torch.int32
                or table.ndim != 2 or table.shape[0] != len(q_lens) or table.stride(1) != 1):
            raise ValueError("MLA Page-RR requires a group-current CUDA int32 page table")
        max_pages = (max(kv_lens) + self.page_size - 1) // self.page_size
        required_pages = (max_pages + self.cp_size - 1 - self.cp_rank) // self.cp_size
        required_pages *= self.page_size // self.kernel_page_size
        if required_pages > table.shape[1]:
            raise RuntimeError(
                "MLA page-RR sequence exceeds the rank-local block table: "
                f"required_kernel_pages={required_pages} width={table.shape[1]} "
                f"shard_size={self.cp_size} shard_rank={self.cp_rank}"
            )
        total_q = q_offsets[-1]
        logical = int(inputs.logical_token_count)
        valid_tokens = logical if inputs.physical_token_count > logical else total_q
        if not 0 <= valid_tokens <= total_q:
            raise ValueError("valid_token_count must be within the slot-mapping extent")
        self.attn_inputs = inputs
        self.q_lens_host = q_lens
        self.kv_lens_host = kv_lens
        self.prefix_lens_host = prefix_lens
        self.q_offsets_host = q_offsets
        self.max_q = max(q_lens)
        self.max_k = max(kv_lens)
        self.q_tokens = total_q
        self.kv_tokens = kv_offsets[-1]
        self.qo_indptr_d = qo_indptr.to(torch.int32).contiguous()
        # Cache's mutable cu_kv_seqlens_device can be stale across producer and
        # attention streams during MTP Prefill. Keep the eager plan's own small
        # indptr from the same host lengths used by the CPU chunk planner.
        self.prefill_ragged_kv_len_indptr_d = torch.tensor(kv_offsets, dtype=torch.int32, device=device)
        self.kv_lens_d = self.prefill_ragged_kv_len_indptr_d.diff()
        self.block_tables = table
        fixed_q_len = q_lens[0] if all(q == q_lens[0] for q in q_lens) else 0
        padding = inputs.padding_offset if not fixed_q_len and total_q else self.qo_indptr_d
        if (padding is None or padding.device != device or padding.dtype != torch.int32
                or (not fixed_q_len and total_q and (
                    padding.shape != (total_q,) or not padding.is_contiguous()))):
            raise ValueError("MLA Page-RR requires CUDA int32 packed-to-padded offsets")
        self.slot_mapping = torch.empty(total_q, dtype=torch.int64, device=device)
        if total_q:
            _prepare_prefill_metadata[(triton.cdiv(total_q, 128),)](
                padding, prefixes, table, self.slot_mapping, total_q, valid_tokens,
                table.shape[1], table.stride(0), request_count=len(q_lens),
                fixed_q_len=fixed_q_len, max_q_len=self.max_q, page_size=self.page_size,
                kernel_page_size=self.kernel_page_size, cp_size=self.cp_size,
                cp_rank=self.cp_rank, block=128, debug=True,
            )
        stream = torch.cuda.current_stream(device)
        for tensor in (lengths, prefixes, self.qo_indptr_d,
                       self.prefill_ragged_kv_len_indptr_d, self.kv_lens_d,
                       table, padding, self.slot_mapping):
            tensor.record_stream(stream)

    def prepare_decode(self, inputs, forbid_realloc: bool = False) -> None:
        multi = inputs.is_target_verify or inputs.is_mtp_draft_update
        mtp_update = inputs.is_mtp_draft_update
        batch = inputs.input_lengths.numel()
        tokens = inputs.physical_token_count
        if batch <= 0 or tokens <= 0 or (not mtp_update and tokens % batch):
            raise ValueError(
                "Page-RR Decode requires rectangular physical queries: "
                f"batch={batch} physical_tokens={tokens} "
                f"verify={inputs.is_target_verify} "
                f"draft_update={inputs.is_mtp_draft_update} "
                f"graph={inputs.is_cuda_graph}"
            )
        queries = 1 if mtp_update else tokens // batch
        if not multi and queries != 1:
            raise ValueError("Ordinary Page-RR Decode requires one query per request")
        base = (
            inputs.prefix_lengths_device
            if multi else inputs.sequence_lengths_plus_1_device
        )
        # Main passes one tagged, group-local attention view at a time.
        table = inputs.kv_cache_kernel_block_id_device
        if base is None or not base.is_cuda or base.dtype != torch.int32 or base.ndim != 1:
            raise ValueError("Page-RR Decode requires CUDA int32 per-request lengths")
        if (
            table is None or table.device != base.device or table.dtype != torch.int32
            or table.ndim != 2 or table.shape[0] != base.numel() or table.stride(1) != 1
        ):
            raise ValueError("Page-RR Decode requires a group-current CUDA int32 page table")
        cu_seqlens = inputs.cu_seqlens_device if mtp_update else None
        valid_mask = inputs.valid_token_mask if mtp_update else None
        if mtp_update:
            if (
                base.numel() != batch or cu_seqlens is None
                or cu_seqlens.device != base.device or cu_seqlens.dtype != torch.int32
                or cu_seqlens.ndim != 1 or cu_seqlens.numel() != batch + 1
                or not cu_seqlens.is_contiguous()
            ):
                raise ValueError("MTP update requires CUDA int32 per-request query offsets")
            if valid_mask is not None and (
                valid_mask.device != base.device or valid_mask.dtype != torch.bool
                or valid_mask.ndim != 1 or valid_mask.numel() != tokens
                or not valid_mask.is_contiguous()
            ):
                raise ValueError("MTP update requires one valid-row mask entry per physical query")
        # MTP consumes the original compact query layout; metadata capacity
        # does not determine its physical query count or per-request width.
        shape = (tokens, 1) if mtp_update else (base.numel(), queries)
        if self.local_causal_lens is None or self.local_causal_lens.shape != shape:
            if forbid_realloc:
                raise ValueError("Page-RR Decode metadata shape cannot change during replay")
            self.local_causal_lens = torch.empty(shape, dtype=torch.int32, device=base.device)
            self.positions_d = torch.empty(shape[0] * shape[1], dtype=torch.int32, device=base.device)
            self.slot_mapping = torch.empty_like(self.positions_d, dtype=torch.int64)
        self.block_tables = table
        rows = self.positions_d.numel()
        # A completely empty owner still needs a valid kernel table descriptor.
        # Its zero lengths prevent KV reads; no dummy page becomes a real key.
        width = max(table.shape[1], 1)
        table_shape = (rows, width)
        if self.query_block_tables is None or self.query_block_tables.shape != table_shape:
            if forbid_realloc:
                raise ValueError("Page-RR Decode query page table shape cannot change during replay")
            self.query_block_tables = torch.zeros(table_shape, dtype=torch.int32, device=base.device)
        if table.shape[1] and not mtp_update:
            self.query_block_tables.view(base.numel(), queries, width).copy_(table[:, None, :])
        grid = (rows,) if mtp_update else (triton.cdiv(rows, 128),)
        _prepare_decode_metadata[grid](
            base,
            # Main's sequence_lengths is a CPU mirror. The device length is
            # the graph-stable source for ordinary Decode capture and replay.
            inputs.sequence_lengths_plus_1_device if inputs.is_cuda_graph and not multi else base,
            cu_seqlens if mtp_update else base,
            valid_mask if valid_mask is not None else base,
            table,
            self.query_block_tables,
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
            mtp_update=mtp_update,
            request_count=batch,
            has_valid_mask=valid_mask is not None,
            block=128,
        )
