"""Native K3 MLA with RTP cache metadata and CacheStore publication."""

import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import MlaImplBase
from rtp_llm.models_py.modules.kimi_k3.native_mla_decode import NativeMlaDecode
from rtp_llm.ops.compute_ops import rtp_llm_ops
from rtp_llm.utils.model_weight import W


_workspaces = {}


class KimiK3MlaVerifyImpl(MlaImplBase):
    """Decode and compact native-MTP queries, with fixed replay metadata."""

    def __init__(self, config, parallelism, weights, inputs, fmha_config, is_cuda_graph):
        attention = config.getAttentionConfigs(parallelism.get_attn_tp_size())
        super().__init__(
            attention, inputs, weights.weights, None, fmha_config,
            max_seq_len=config.max_seq_len, is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism,
        )
        self.graph_mode = is_cuda_graph
        self.batch = inputs.input_lengths.numel()
        self.tokens = inputs.physical_token_count
        if self.batch <= 0 or self.tokens <= 0:
            raise ValueError("K3 MLA requires nonempty physical rows")
        if inputs.is_target_verify and self.tokens % self.batch:
            raise ValueError("K3 target verification requires rectangular physical rows")
        device = next(w[W.mla_vc].device for w in self.weights if W.mla_vc in w)
        if device not in _workspaces:
            _workspaces[device] = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.native = NativeMlaDecode(
            num_heads=attention.head_num, kv_lora_rank=attention.kv_lora_rank,
            nope_dim=attention.nope_head_dim, pe_dim=attention.rope_head_dim,
            page_size=attention.kernel_tokens_per_block,
            softmax_extra_scale=attention.softmax_extra_scale,
            workspace=_workspaces[device], max_batch=self.batch,
        )
        columns = (self.max_seq_len + self.native.page_size - 1) // self.native.page_size
        # RTP reserves additional logical blocks for speculative candidates.
        # Preserve that physical metadata capacity without extending valid KV length.
        initial_table = inputs.kv_cache_kernel_block_id
        if initial_table.ndim != 2 or initial_table.shape[0] != self.batch:
            raise ValueError("K3 MLA requires one block-table row per physical request")
        columns = max(columns, initial_table.shape[1])
        self.block_tables_h = torch.empty((self.batch, columns), dtype=torch.int32, pin_memory=True)
        self.block_tables = torch.empty((self.batch, columns), dtype=torch.int32, device=device)
        self.slot_mapping_h = torch.empty(self.tokens, dtype=torch.int64, pin_memory=True)
        self.slot_mapping = torch.empty(self.tokens, dtype=torch.int64, device=device)
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.write_cache_store_impl = common.create_write_cache_store_impl(inputs)
        # Draft-update graphs bucket total tokens. The capture dummy spreads
        # them over rows, but a replay may place them in one request.
        self.max_query_len = (
            self.tokens if is_cuda_graph and inputs.is_mtp_draft_update else None
        )
        # The shared planner otherwise reserves reuse-page metadata only for
        # the capture prefix. A later page crossing must not reallocate it.
        self.block_tables_h.zero_()
        reserve_lengths = inputs.input_lengths if inputs.prefix_lengths is not None and inputs.prefix_lengths.numel() else torch.ones_like(inputs.input_lengths)
        reserve_prefix = torch.full_like(reserve_lengths, self.max_seq_len) - reserve_lengths
        self.fmha_params.fill_params(
            reserve_prefix, torch.empty(0, dtype=torch.int32), reserve_lengths,
            self.block_tables_h, self.native.page_size, False,
        )
        self.prepare(inputs)

    def prepare(self, inputs, forbid_realloc=False):
        if inputs.input_lengths.numel() != self.batch or inputs.physical_token_count != self.tokens:
            raise ValueError("K3 MLA replay requires the captured physical shape")
        table = inputs.kv_cache_kernel_block_id
        if table.is_cuda or table.ndim != 2 or table.shape[0] != self.batch:
            raise ValueError("K3 MLA requires RTP's host kernel-page table")
        if table.shape[1] > self.block_tables.shape[1]:
            raise ValueError("K3 MLA block table exceeds captured metadata capacity")
        self.attn_inputs = inputs
        self.fmha_params.fill_params(
            inputs.prefix_lengths, inputs.sequence_lengths, inputs.input_lengths,
            table, self.native.page_size, forbid_realloc,
        )
        offsets = self.fmha_params.qo_indptr_h
        query_bound = int((offsets[1:] - offsets[:-1]).max())
        if int(offsets[-1]) != self.tokens:
            raise ValueError("K3 MLA metadata does not cover every physical query")
        actual_max_seq_len = int(self.fmha_params.kvlen_h.max())
        if actual_max_seq_len > self.max_seq_len:
            raise ValueError("K3 MLA sequence exceeds configured capacity")
        # Graph launch bounds are captured constants and must cover later replay.
        # Eager execution can use the current host metadata, as vLLM does.
        self.kernel_max_seq_len = self.max_seq_len if self.graph_mode else max(1, actual_max_seq_len)
        if self.max_query_len is None:
            self.max_query_len = max(1, query_bound)
        elif query_bound > self.max_query_len:
            raise ValueError("K3 MLA query width exceeds the capture bound; select a new bucket")
        # CPU padding and one H2D copy outside capture avoid per-layer metadata
        # kernels and preserve the address captured by the attention backend.
        self.block_tables_h.zero_()
        self.block_tables_h[:, :table.shape[1]].copy_(table)
        self.block_tables.copy_(self.block_tables_h, non_blocking=True)
        # RTP reserves page zero for SP dummy requests. vLLM uses slot -1 for
        # those rows; passing RTP's unmodified zero-page slot would overwrite it.
        rows = torch.repeat_interleave(torch.arange(self.batch), offsets[1:] - offsets[:-1])
        positions = self.fmha_params.positions_h.to(torch.int64)
        pages = table[rows, positions // self.native.page_size].to(torch.int64)
        self.slot_mapping_h.copy_(torch.where(
            pages > 0, pages * self.native.page_size + positions % self.native.page_size, -1
        ))
        self.slot_mapping.copy_(self.slot_mapping_h, non_blocking=True)

    def prepare_cuda_graph(self, inputs):
        self.prepare(inputs, forbid_realloc=True)

    def forward(self, q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=None):
        if topk_indices is not None or kv_cache is None:
            raise ValueError("K3 native MLA requires a dense paged KV cache")
        weights = self.weights[layer_id]
        query = self.native.write_cache(
            q, compressed_kv, k_pe, kv_cache.kv_cache_base,
            self.slot_mapping, weights[W.mla_kc],
        )
        common.apply_write_cache_store(self.write_cache_store_impl, self.attn_inputs, kv_cache)
        return self.native.attend(
            query, kv_cache.kv_cache_base, weights[W.mla_vc],
            block_tables=self.block_tables, seq_lens=self.fmha_params.kvlen_d,
            cu_query_lens=self.fmha_params.qo_indptr_d,
            max_query_len=self.max_query_len, max_seq_len=self.kernel_max_seq_len,
        )
