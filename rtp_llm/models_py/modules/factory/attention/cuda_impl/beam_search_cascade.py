"""Decode shared KV pages once per beam group, then merge private-tail attention.

Grouping is derived from the host page table, not from a fixed beam width. Only
identical, complete historical pages are shared; the page containing the new KV
always belongs to the tail. Graphs retain fixed sequence and request capacities.
"""

import numpy as np
import torch
from flashinfer.cascade import merge_state_in_place
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferDecodeImpl,
    attn_kv_dtype,
    get_py_flashinfer_workspace_buffer,
    release_py_flashinfer_workspace_buffer,
)
from rtp_llm.ops.compute_ops import FusedRopeKVCacheDecodeOp, rtp_llm_ops


class BeamSearchCascadeAttention:
    MIN_BEAMS = 32
    MIN_SHARED_TOKENS = 128

    def __init__(
        self,
        batch_size,
        heads,
        kv_heads,
        head_dim,
        page_size,
        max_requests,
        max_pages,
        dtype,
        use_cuda_graph,
        workspace,
    ):
        self.batch_size = batch_size
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.page_size, self.max_pages, self.dtype = page_size, max_pages, dtype
        # The extra slot absorbs rounded/padded query rows without another bucket axis.
        self.slots = min(max_requests, batch_size) + 1
        self.use_cuda_graph = use_cuda_graph
        self.shared_tokens = 0
        self._signature = None
        # FlashInfer Decode requires zero-initialized workspace on first use.
        # The engine pool may return storage used by a different attention op.
        workspace.zero_()

        def host(n):
            return torch.zeros(n, dtype=torch.int32, pin_memory=True)

        def device(n):
            return torch.zeros(n, dtype=torch.int32, device=workspace.device)

        self.qo = host(self.slots + 1)
        self.prefix_ptr = host(self.slots + 1)
        self.prefix_idx = host(self.slots * max_pages)
        self.prefix_last = host(self.slots)
        self.tail_ptr = host(batch_size + 1)
        self.tail_idx = host(batch_size * max_pages)
        self.tail_last = host(batch_size)
        self.columns = np.arange(max_pages)[None, :]
        self.prefix = BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            "HND",
            backend="fa2",
            use_cuda_graph=use_cuda_graph,
            qo_indptr_buf=device(self.slots + 1),
            paged_kv_indptr_buf=device(self.slots + 1),
            paged_kv_indices_buf=device(self.slots * max_pages),
            paged_kv_last_page_len_buf=device(self.slots),
        )
        self.tail = BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            "HND",
            use_cuda_graph=use_cuda_graph,
            use_tensor_cores=False,
            paged_kv_indptr_buffer=device(batch_size + 1),
            paged_kv_indices_buffer=device(batch_size * max_pages),
            paged_kv_last_page_len_buffer=device(batch_size),
        )

    def plan(self, block_table, sequence_lengths):
        if block_table.is_cuda or sequence_lengths.is_cuda:
            raise ValueError(
                "Cascade planning requires host KV page tables and lengths"
            )
        table = block_table.numpy()
        lengths = sequence_lengths.numpy().astype(np.int64, copy=False)
        if len(lengths) != self.batch_size or table.shape[0] != self.batch_size:
            raise ValueError(
                "Cascade sequence count must match its fixed batch capacity"
            )
        if np.any(lengths < 0):
            raise ValueError("Cascade sequence lengths must be nonnegative")
        pages = (lengths + self.page_size) // self.page_size
        width = int(pages.max())
        if width > self.max_pages or width > table.shape[1]:
            raise ValueError("Cascade KV page table exceeds its captured capacity")
        table = table[:, :width]
        # Adjacent beams of a request share their first physical page. Verifying
        # every subsequent page makes this safe even when separate requests share
        # a cached prefix, or beam ancestry changes after candidate selection.
        bounds = np.r_[
            0, np.flatnonzero(table[1:, 0] != table[:-1, 0]) + 1, self.batch_size
        ]
        if len(bounds) - 1 > self.slots:
            bounds = np.array([0, self.batch_size])
            allow_sharing = False
        else:
            allow_sharing = True
        cuts = np.zeros(self.batch_size, dtype=np.int32)
        qo, ptr, last = (
            self.qo.numpy(),
            self.prefix_ptr.numpy(),
            self.prefix_last.numpy(),
        )
        qo.fill(self.batch_size)
        qo[0] = 0
        ptr.fill(0)
        last.fill(0)
        cursor = 0
        self.shared_tokens = 0
        for slot, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            shared = 0
            if allow_sharing and end - start >= self.MIN_BEAMS:
                limit = int(lengths[start:end].min() // self.page_size)
                same = np.all(table[start:end, :limit] == table[start, :limit], axis=0)
                mismatch = np.flatnonzero(~same)
                shared = int(mismatch[0]) if len(mismatch) else limit
                if shared * self.page_size < self.MIN_SHARED_TOKENS:
                    shared = 0
            cuts[start:end] = shared
            qo[slot + 1] = end
            self.prefix_idx.numpy()[cursor : cursor + shared] = table[start, :shared]
            cursor += shared
            ptr[slot + 1] = cursor
            last[slot] = self.page_size if shared else 0
            self.shared_tokens += (end - start) * shared * self.page_size
        ptr[len(bounds) :] = cursor
        tail_counts = pages - cuts
        self.tail_ptr.numpy()[0] = 0
        np.cumsum(tail_counts, out=self.tail_ptr.numpy()[1:])
        mask = (self.columns[:, :width] >= cuts[:, None]) & (
            self.columns[:, :width] < pages[:, None]
        )
        tail_count = int(self.tail_ptr[-1])
        self.tail_idx.numpy()[:tail_count] = table[mask]
        self.tail_last.numpy()[:] = lengths % self.page_size + 1
        # Graph mode pads the prefix launch and masks unused work tiles. Keep
        # FlashInfer's default split planning: in 0.6.9, disable_split_kv omits
        # that mask while still launching padded tiles. Request/length changes
        # update the plan buffers without changing the captured launch shape.
        self.prefix.plan(
            self.qo,
            self.prefix_ptr,
            self.prefix_idx[:cursor],
            self.prefix_last,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.page_size,
            causal=False,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            non_blocking=True,
        )
        self.tail.plan(
            self.tail_ptr,
            self.tail_idx[:tail_count],
            self.tail_last,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.page_size,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            non_blocking=True,
        )
        signature = (tuple(self.prefix._plan_info), tuple(self.tail._plan_info))
        if (
            self.use_cuda_graph
            and self._signature is not None
            and signature != self._signature
        ):
            raise RuntimeError("Cascade plan changed captured launch geometry")
        self._signature = signature

    def run(self, q, kv_cache):
        prefix_output, prefix_lse = self.prefix.run(q, kv_cache, return_lse=True)
        tail_output, tail_lse = self.tail.run(q, kv_cache, return_lse=True)
        # Empty prefixes have LSE=-inf: merging selects the full tail unchanged.
        merge_state_in_place(prefix_output, prefix_lse, tail_output, tail_lse)
        return prefix_output


class PyFlashinferCascadeDecodeAttnOp:
    def __init__(self, configs, inputs, max_requests, max_seq_len):
        self.configs = configs
        self.workspace = get_py_flashinfer_workspace_buffer()
        self.cascade = BeamSearchCascadeAttention(
            inputs.input_lengths.numel(),
            configs.head_num,
            configs.kv_head_num,
            configs.size_per_head,
            configs.kernel_tokens_per_block,
            max_requests,
            (max_seq_len + configs.kernel_tokens_per_block - 1)
            // configs.kernel_tokens_per_block,
            configs.dtype,
            inputs.is_cuda_graph,
            self.workspace,
        )

    def __del__(self):
        if hasattr(self, "workspace"):
            release_py_flashinfer_workspace_buffer(self.workspace)

    def prepare_for_cuda_graph_replay(self, inputs):
        self.cascade.plan(inputs.kv_cache_kernel_block_id, inputs.sequence_lengths)

    def forward(self, q, kv_cache, params):
        q = q.reshape(q.shape[0], self.configs.head_num, self.configs.size_per_head)
        kv = kv_cache.kv_cache_base
        if kv.dim() == 2:
            kv = common.reshape_paged_kv_cache(
                kv,
                self.configs.kv_head_num,
                self.configs.kernel_tokens_per_block,
                self.configs.size_per_head,
            )
        return self.cascade.run(q, kv)


class PyFlashinferCascadeDecodeImpl(PyFlashinferDecodeImpl):
    @classmethod
    def support(cls, configs, inputs):
        return (
            torch.version.hip is None
            and not configs.use_mla
            and not configs.is_sparse
            and configs.is_causal
            and configs.q_scaling == 1.0
            and configs.softmax_extra_scale == 1.0
            and not inputs.is_prefill
            and not inputs.is_target_verify
            and inputs.input_lengths.numel() >= BeamSearchCascadeAttention.MIN_BEAMS
            and not inputs.sequence_lengths.is_cuda
            and inputs.kv_cache_kernel_block_id is not None
            and not inputs.kv_cache_kernel_block_id.is_cuda
            and configs.dtype in (torch.float16, torch.bfloat16)
            and attn_kv_dtype(configs) == configs.dtype
            and configs.head_num % configs.kv_head_num == 0
            and configs.head_num // configs.kv_head_num in (1, 2, 4, 8, 16, 32)
            and configs.size_per_head in (64, 128, 256)
        )

    def __init__(self, configs, inputs, parallelism_config, max_requests, max_seq_len):
        self.need_rope_kv_cache = configs.need_rope_kv_cache
        self.attn_configs, self.attn_inputs = configs, inputs
        self.fmha_impl = PyFlashinferCascadeDecodeAttnOp(
            configs, inputs, max_requests, max_seq_len
        )
        self.fmha_impl.prepare_for_cuda_graph_replay(inputs)
        self.rope_impl = FusedRopeKVCacheDecodeOp(configs)
        self.rope_params = self.rope_impl.prepare(inputs)
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.write_cache_store_impl = common.create_write_cache_store_impl(inputs)
