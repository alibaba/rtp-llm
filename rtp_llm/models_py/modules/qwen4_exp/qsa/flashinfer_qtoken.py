"""FlashInfer 0.7.4 QToken-KvBlock paged G8 attention for Qwen4-Max."""

from dataclasses import replace

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.qsa.qtoken_gate import apply_qtoken_gate


@triton.jit(do_not_specialize=["T", "B", "ROUTES"])
def _build_qtoken_routes(Cu, Out, T, B, ROUTES, B_TILE: tl.constexpr):
    request = tl.program_id(1)
    row = tl.program_id(0) * 128 + tl.arange(0, 128)
    requests = tl.arange(0, B_TILE)
    first = tl.load(Cu + requests, requests < B, other=T)
    last = tl.load(Cu + requests + 1, requests < B, other=T)
    base = tl.sum(tl.where(requests < request, tl.cdiv(last - first, 8), 0), 0)
    start = tl.load(Cu + request)
    end = tl.load(Cu + request + 1)
    token = start + row * 8
    tl.store(Out + base + row, token, token < end)
    if request == 0 and tl.program_id(0) == 0:
        tl.store(Out + ROUTES, T)


class PagedQTokenG8:
    """One forward's routes and workspace, shared by the attention layers."""

    def __init__(self, inputs, q_shape, pool, *, split_kv):
        from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path

        setup_cutlass_import_path()
        from flashinfer.attention.prims_ts.decode import (
            _prepare_prims_ts_batch_decode_plan,
        )
        from flashinfer.attention.prims_ts.q_token_kv_block_sparse_metadata import (
            _get_prims_ts_q_token_kv_block_sparse_workspace_layout,
        )

        tokens = q_shape[0]
        self.prefill = bool(inputs.is_prefill)
        self.draft_padding = (
            self.prefill and not inputs.is_target_verify and inputs.is_cuda_graph
        )
        table = inputs.kv_cache_kernel_block_id_device
        self.table = table
        self.cu = (
            inputs.cu_seqlens_device
            if self.prefill
            else inputs.decode_cu_seqlens_device
        )
        self.prefix = (
            inputs.prefix_lengths_device
            if self.prefill
            else inputs.sequence_lengths_plus_1_device
        )
        batch = table.shape[0]
        self.max_kv = table.shape[1] * 128
        self.routes = self.cu
        max_route_q = 8
        if self.draft_padding:
            # Speculative draft replays may leave a suffix of unused packed rows.
            # Singleton routes keep their storage and number stable on replay.
            self.routes = torch.arange(
                tokens + 1, dtype=torch.int32, device=pool.device
            )
            max_route_q = 1
        elif not self.prefill:
            max_route_q = 1
        elif self.prefill and not inputs.is_cuda_graph:
            lengths = inputs.input_lengths.tolist()
            self.max_kv = max(
                p + n for p, n in zip(inputs.prefix_lengths.tolist(), lengths)
            )
            max_q = max(lengths)
            max_route_q = min(max_q, 8)
            if max_q > 8:
                routes = sum(triton.cdiv(n, 8) for n in lengths)
                self.routes = torch.empty(
                    routes + 1, dtype=torch.int32, device=pool.device
                )
                _build_qtoken_routes[(triton.cdiv(max_q, 1024), batch)](
                    self.cu,
                    self.routes,
                    tokens,
                    batch,
                    routes,
                    triton.next_power_of_2(batch),
                    num_warps=4,
                )

        self.query = torch.empty(
            (tokens, 64, 256), dtype=pool.dtype, device=pool.device
        )
        # The attention planner sees compacted KV, whose capacity is bounded
        # by selected fragments per route even when the logical context is long.
        compact_capacity = min(self.max_kv, max_route_q * (512 + 1) * 4)
        layout = _get_prims_ts_q_token_kv_block_sparse_workspace_layout(
            tokens,
            512,
            table.shape[1],
            128,
            8,
            max_seq_len_kv=compact_capacity,
            num_qo_heads=64,
            num_kv_heads=4,
            head_dim=256,
            q_dtype=pool.dtype,
            device=pool.device,
            num_query_groups=self.routes.numel() - 1,
            packed_query=True,
            split_kv=split_kv,
        )
        self.metadata = layout.bind(
            torch.empty(layout.total_bytes, dtype=torch.uint8, device=pool.device)
        )
        self.plan, _ = _prepare_prims_ts_batch_decode_plan(
            self.query,
            self._cache_views(pool),
            self.metadata.attention_workspace_buffer,
            self.metadata.q_token_kv_block_sparse_page_indices,
            self.metadata.seq_lens,
            layout.max_seq_len,
            seq_len_q=8,
            qo_indptr=self.routes,
            max_q_len=8,
            out=self.query,
            out_dtype=pool.dtype,
            mask_type="causal",
            window_left=-1,
            kv_layout="HND",
            page_size=4,
            q_token_kv_block_sparse_page_memberships=self.metadata.q_token_kv_block_sparse_page_memberships,
            use_q_token_kv_block_sparse_route=True,
            split_kv=split_kv,
        )

    @staticmethod
    def _cache_views(pool):
        if pool.ndim == 2:
            pool = pool.as_strided(
                (pool.shape[0], 2, 4, 128, 256), (pool.stride(0), 131072, 32768, 256, 1)
            )
        return pool.unbind(1)

    def prepare(
        self, qkv, selected, pool, positions=None, norm=None, *, zero_tail=False
    ):
        from rtp_llm.models_py.kernels.cuda.qsa.qtoken_prepare import prepare_qtoken

        prepare_qtoken(
            qkv,
            selected,
            positions,
            None if norm is None else norm.q_weight,
            None if norm is None else norm.k_weight,
            pool,
            self.table,
            self.cu,
            self.prefix,
            self.routes,
            self.query,
            self.metadata.q_token_kv_block_sparse_page_indices,
            self.metadata.q_token_kv_block_sparse_page_memberships,
            self.metadata.seq_lens,
            max_kv_len=self.max_kv,
            eps=0.0 if norm is None else norm.eps,
            prefill=self.prefill,
            locator_heads=self.plan._flat_locator_heads,
            decode_q1=not self.prefill,
            q_only=norm is None,
            padded_rows=self.draft_padding,
            zero_tail=zero_tail,
        )

    def attend(self, pool, gate):
        plan = self.plan
        k, v = self._cache_views(pool)
        if plan._cache.k_cache.data_ptr() != k.data_ptr():
            # This pool's layers share the same physical page geometry. The
            # compiled FlashInfer call receives the layer's actual addresses.
            if (
                k.shape != plan._cache.k_cache.shape
                or k.stride() != plan._cache.k_cache.stride()
            ):
                raise RuntimeError("QToken layer paged KV geometry changed")
            plan = replace(plan, _cache=replace(plan._cache, k_cache=k, v_cache=v))
        out = torch.empty_like(self.query)
        plan._run_unchecked(self.query, out, 0.0625, 1.0)
        if gate is not None:
            apply_qtoken_gate(
                out, gate, live_tokens=self.cu[-1:] if self.draft_padding else None
            )
        return out.view(-1, 64 * 256)
