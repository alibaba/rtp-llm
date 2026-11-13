import torch
import torch_npu

from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_attn_params import (
    AscendAttnParams,
    _blocks_per_phys_from_config,
    compute_ascend_attn_params,
)

from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_kv_cache_write_op import AscendKVCacheWriteOp
from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_rope_emb import AscendRotaryEmbeddingOp
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.modules.factory.attention import common


class AscendPrefillImpl(FMHAImplBase):
    """Ascend MHA Prefill using npu_fused_infer_attention_score.

    Composes RoPE -> KVCacheWrite -> write_cache_store -> FMHA.
    """

    def __init__(self, attn_configs, attn_inputs, parallelism_config):
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.fmha_params = None

        self.fmha_impl = AscendPrefillAttnOp(attn_configs, attn_inputs)
        self.rope_impl = self._create_rope_impl(attn_configs)
        self.kv_cache_write_op = AscendKVCacheWriteOp(
            num_kv_heads=attn_configs.kv_head_num,
            head_size=attn_configs.size_per_head,
            token_per_block=attn_inputs.kv_cache.seq_size_per_block if attn_inputs.kv_cache else 128,
        )

        self.params = AscendAttnParams() # Only used by rope and KV cache write
        if self.rope_impl is not None:
            self.rope_impl.set_params(self.params)
        self.kv_cache_write_op.set_params(self.params)

        self.fmha_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def _create_rope_impl(self, attn_configs):
        from rtp_llm.ops import RopeStyle
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return AscendRotaryEmbeddingOp(attn_configs)

    def _split_qkv(self, qkv):
        qkv = qkv.reshape(qkv.shape[0], -1)
        num_heads = self.attn_configs.head_num
        num_kv_heads = self.attn_configs.kv_head_num
        head_dim = self.attn_configs.size_per_head
        q, k, v = torch.split(qkv, [
            head_dim * num_heads,
            head_dim * num_kv_heads,
            head_dim * num_kv_heads,
        ], dim=-1)
        query = q.reshape(q.shape[0], num_heads, head_dim)
        key = k.reshape(k.shape[0], num_kv_heads, head_dim).contiguous()
        value = v.reshape(v.shape[0], num_kv_heads, head_dim).contiguous()
        return query, key, value

    def _update_rope_kv_write_params(self, device, kv_cache, num_tokens: int, layer_idx: int = 0):
        # The kernel block granularity comes from the per-layer cache view; the
        # physical block size it maps onto drives slot_mapping and the writes.
        blocks_per_phys = _blocks_per_phys_from_config(self.attn_configs, self.attn_inputs)
        kernel_page = kv_cache.seq_size_per_block if kv_cache is not None else 0
        self.params.blocks_per_phys = blocks_per_phys
        if getattr(self.attn_inputs, "is_cuda_graph", False):
            # Graph mode: derive positions/slot_mapping by device ops from the
            # capture buffers so replay updates are picked up without re-capture.
            self._update_rope_kv_write_params_device(device, kernel_page, num_tokens)
            return
        positions, slot_mapping, mrope_positions = compute_ascend_attn_params(
            self.attn_inputs, layer_idx, kernel_page * blocks_per_phys
        )
        self.params.positions_d = positions.to(device, non_blocking=True)
        self.params.slot_mapping = slot_mapping.to(device, non_blocking=True)
        self.params.mrope_positions_d = (
            mrope_positions.to(device, non_blocking=True) if mrope_positions is not None else None
        )

    def _update_rope_kv_write_params_device(self, device, kernel_page, num_tokens: int):
        attn_inputs = self.attn_inputs
        if not kernel_page:
            kernel_page = (
                attn_inputs.kv_cache.seq_size_per_block
                if attn_inputs.kv_cache is not None
                else 128
            )
        # Verify feeds a uniform tokens-per-stream count; the TND layout is
        # stream-major, so token t belongs to stream t // apsk.
        seq_plus_1 = attn_inputs.sequence_lengths_plus_1_device  # [B] == prefix + 1
        num_streams = int(seq_plus_1.numel())
        tokens_per_stream = num_tokens // num_streams if num_streams > 0 else 0

        arange = torch.arange(tokens_per_stream, dtype=torch.int32, device=device)
        positions_d = (
            (seq_plus_1.to(torch.int32) - 1).unsqueeze(1) + arange.unsqueeze(0)
        ).reshape(-1)

        block_table = attn_inputs.kv_cache_kernel_block_id_device
        if (
            block_table is not None
            and block_table.numel() > 0
            and positions_d.numel() > 0
        ):
            if block_table.ndim != 2:
                block_table = block_table.reshape(-1, block_table.shape[-1])
            max_blocks = block_table.size(1)
            pos_long = positions_d.long()
            block_index = (pos_long // kernel_page).clamp(max=max_blocks - 1)
            block_offset = pos_long % kernel_page
            batch_ids = (
                torch.arange(num_streams, dtype=torch.long, device=device)
                .unsqueeze(1)
                .expand(num_streams, tokens_per_stream)
                .reshape(-1)
            )
            slot_block_numbers = block_table[batch_ids, block_index].long().clamp(min=0)
            slot_mapping = (slot_block_numbers * kernel_page + block_offset).to(torch.int64)
        else:
            slot_mapping = torch.empty(0, dtype=torch.int64, device=device)

        self.params.positions_d = positions_d
        self.params.slot_mapping = slot_mapping
        # Graph-mode MRoPE: read the three-axis positions from the captured
        # combo_position_ids buffer ([T*3] flat on device) when present —
        # same contract as the decode path (AscendGraphRunner allocates the
        # buffer at capture and refreshes it at replay via copyTensorSlice).
        combo = getattr(self.attn_inputs, "combo_position_ids", None)
        if combo is not None and combo.numel() > 0:
            if combo.dim() == 1:
                combo = combo.reshape(-1, 3)
            if combo.dim() == 2 and combo.size(1) == 3:
                self.params.mrope_positions_d = combo.to(torch.int64)
                return
        self.params.mrope_positions_d = None

    def prepare(self, attn_inputs):
        self.fmha_impl.prepare(attn_inputs)
        self.attn_inputs = attn_inputs

    def prepare_cuda_graph(self, attn_inputs, graph_bs=None):
        """Called by AscendGraphRunner::prepareInputs() before each replay."""
        self.attn_inputs = attn_inputs
        self.fmha_impl.prepare(attn_inputs)
        # The FIA task update must carry exactly graph_bs context lengths
        # (padded rows contribute ctx=1).
        lens = attn_inputs.input_lengths
        valid = max(int((lens > 0).sum().item()) if lens.numel() else 0, 1)
        graph_bs = graph_bs if graph_bs is not None else valid
        kv_lens = (attn_inputs.prefix_lengths[:valid] + lens[:valid]).to(torch.int32).tolist()
        kv_lens = kv_lens[:graph_bs] + [1] * max(0, graph_bs - len(kv_lens))
        self.fmha_impl.update_graph_fia(kv_lens, graph_bs)

    def signal_graph_events(self, graph_bs):
        """Called by AscendGraphRunner before the post-capture sanity replay."""
        self.fmha_impl.signal_graph_events(graph_bs)

    def forward(self, qkv, kv_cache, layer_idx=0):
        if self.need_rope_kv_cache:
            self._update_rope_kv_write_params(qkv.device, kv_cache, qkv.shape[0], layer_idx)

            if self.rope_impl is not None:
                query, key, value = self.rope_impl.forward(qkv)
            else:
                query, key, value = self._split_qkv(qkv)

            self.kv_cache_write_op.forward(key, value, kv_cache)
            q = query
        else:
            q = qkv.chunk(3, dim=-1)[0]

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(q, kv_cache)

    @staticmethod
    def support(attn_configs, attn_inputs):
        from rtp_llm.ops import RopeStyle
        # MRoPE variants are applied by AscendRotaryEmbeddingOp.
        return attn_inputs.is_prefill and \
               not attn_configs.use_mla and \
               torch.npu.is_available()


class AscendPrefillAttnOp:
    """NPU prefill attention op, reads cache only.

    Eager: FIA v2 with explicit actual_seq tensors; graph (target verify /
    dense draft re-prop): FIA v2 + graph_task_group/update — replay refreshes
    the TND kv lengths through graph_task_update instead of re-capturing (same
    pattern as AscendDecodeAttnOp).
    """

    _causal_mask = None
    _shared_workspace = None
    _shared_update_stream = None

    @classmethod
    def _get_causal_mask(cls, device):
        if cls._causal_mask is None or cls._causal_mask.device.type != device.type:
            cls._causal_mask = torch.triu(
                torch.ones(2048, 2048, dtype=torch.int8), diagonal=1
            ).to(device)
        return cls._causal_mask

    def __init__(self, attn_configs, attn_inputs):
        self.attn_configs = attn_configs
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scale = attn_configs.q_scaling * (self.head_dim ** -0.5)
        self.page_size = attn_inputs.kv_cache.seq_size_per_block if \
                         attn_inputs.kv_cache else 128
        self.block_table = None
        self.actual_seq_q = None
        self.actual_seq_kv = None
        self.blocks_per_phys = _blocks_per_phys_from_config(attn_configs, attn_inputs)

        # FIA graph tasks grouped by captured stream count (mirrors decode).
        self._graph_handles = {}  # streams -> [task handles]
        self._graph_refs = {}     # streams -> [(q, k, v, table, mask, out, lse, page)]
        self._graph_events = {}   # streams -> [ExternalEvent]

    def set_params(self, params):
        self.params = params

    def prepare(self, attn_inputs):
        self.attn_inputs = attn_inputs
        self.blocks_per_phys = _blocks_per_phys_from_config(self.attn_configs, attn_inputs)
        if getattr(attn_inputs, "is_cuda_graph", False):
            # Graph mode: read the block table from the capture buffer (device)
            # so prepareInputs' in-place updates reach the captured FIA task.
            self.block_table = attn_inputs.kv_cache_kernel_block_id_device
            if self.block_table is not None and self.block_table.ndim != 2:
                self.block_table = self.block_table.reshape(-1, self.block_table.shape[-1])
            self.actual_seq_q = None
            self.actual_seq_kv = None
            return
        self.block_table = attn_inputs.kv_cache_kernel_block_id
        if self.block_table is not None:
            self.block_table = self.block_table.clamp(min=0)
            if self.block_table.ndim != 2:
                self.block_table = self.block_table.reshape(-1, self.block_table.shape[-1])

        seq_lens_q = attn_inputs.input_lengths
        seq_lens_kv = attn_inputs.prefix_lengths + attn_inputs.input_lengths
        self.actual_seq_q = torch.cumsum(seq_lens_q, dim=0)
        self.actual_seq_kv = seq_lens_kv

    def _split_kv(self, kv_cache):
        # Zero-copy K/V half views (base[:, 0/1]), matching the write op and
        # decode read addressing.
        base = kv_cache.kv_cache_base
        k_cache = base[:, 0].reshape(base.shape[0], base.shape[2], -1)
        v_cache = base[:, 1].reshape(base.shape[0], base.shape[2], -1)
        return k_cache, v_cache, base.shape[2]

    def _tnd_seq_lens(self, num_tokens, num_streams):
        """TND cumulative query lengths for a uniform tokens-per-stream graph."""
        apsk = num_tokens // num_streams if num_streams > 0 else num_tokens
        return [apsk * (i + 1) for i in range(num_streams)]

    def _to_device(self, t, device):
        return t.to(device) if t is not None and t.device.type != device.type else t

    def _fia_kwargs(self, q, k_cache, v_cache, block_table, atten_mask, page_size,
                    actual_seq_qlen, actual_seq_kvlen):
        """Shared FIA v2 arguments for the eager call and its .out variant."""
        return {
            "query": q, "key": k_cache, "value": v_cache,
            "atten_mask": atten_mask, "block_table": block_table,
            "input_layout": "TND", "block_size": page_size,
            "actual_seq_qlen": actual_seq_qlen, "actual_seq_kvlen": actual_seq_kvlen,
            "num_key_value_heads": self.num_kv_heads,
            "num_query_heads": self.num_heads,
            "softmax_scale": self.scale, "sparse_mode": 3,
        }

    @classmethod
    def _shared_workspace_tensor(cls, kw, q):
        """Allocate the max FIA v2 workspace once, shared by all captures."""
        if cls._shared_workspace is None:
            ws = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(**kw)
            ws_bytes = ws.numel() * ws.element_size()
            cls._shared_workspace = torch.empty(
                ws_bytes // q.element_size(), dtype=q.dtype, device=q.device
            )
        return cls._shared_workspace

    def forward(self, q, kv_cache):
        if (
            getattr(self.attn_inputs, "is_cuda_graph", False)
            and torch.npu.is_current_stream_capturing()
        ):
            # Verify graph path: fixed K+1 tokens per stream.
            return self._forward_fia_graph(q, kv_cache)

        k_cache, v_cache, page_size = self._split_kv(kv_cache)
        actual_seq_q = self.actual_seq_q
        actual_seq_kv = self.actual_seq_kv
        if actual_seq_q is None:
            # Graph-mode prepare leaves actual_seq unset; only the
            # non-capturing warm-up lands here.
            lens = self.attn_inputs.input_lengths
            actual_seq_q = torch.cumsum(lens, dim=0)
            actual_seq_kv = self.attn_inputs.prefix_lengths + lens
        kw = self._fia_kwargs(
            q, k_cache, v_cache,
            self._to_device(self.block_table, q.device),
            self._get_causal_mask(q.device), page_size,
            self._to_device(actual_seq_q, q.device),
            self._to_device(actual_seq_kv, q.device),
        )
        attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(**kw)
        return attn_output

    def _forward_fia_graph(self, q, kv_cache):
        k_cache, v_cache, page_size = self._split_kv(kv_cache)
        num_tokens = q.shape[0]
        num_streams = int(self.attn_inputs.input_lengths.numel())
        # Real kv lengths are refreshed by update_graph_fia() at replay; the
        # capture-time placeholder only needs the right shape (stream count).
        kw = self._fia_kwargs(
            q, k_cache, v_cache,
            self._to_device(self.block_table, q.device),
            self._get_causal_mask(q.device), page_size,
            self._tnd_seq_lens(num_tokens, num_streams),
            [page_size] * max(num_streams, 1),
        )
        out = torch.empty(
            num_tokens, self.num_heads, self.head_dim, dtype=q.dtype, device=q.device
        )
        lse = torch.empty(1, dtype=q.dtype, device=q.device)
        workspace = self._shared_workspace_tensor(kw, q)

        stream = torch.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score_v2.out(
            **kw, workspace=workspace, out=[out, lse]
        )
        handle = torch.npu.graph_task_group_end(stream)

        self._graph_handles.setdefault(num_streams, []).append(handle)
        self._graph_refs.setdefault(num_streams, []).append(
            (q, k_cache, v_cache, kw["block_table"], kw["atten_mask"], out, lse, page_size)
        )
        self._graph_events.setdefault(num_streams, []).append(event)
        return out

    def signal_graph_events(self, num_streams):
        for ev in self._graph_events.get(num_streams, []):
            ev.record(torch.npu.current_stream())

    def update_graph_fia(self, ctx_list, num_streams):
        handles = self._graph_handles.get(num_streams)
        if not handles:
            return
        events = self._graph_events.get(num_streams, [])
        refs = self._graph_refs.get(num_streams, [])
        if AscendPrefillAttnOp._shared_update_stream is None:
            AscendPrefillAttnOp._shared_update_stream = torch.npu.Stream()
        us = AscendPrefillAttnOp._shared_update_stream
        with torch.npu.stream(us):
            for i, handle in enumerate(handles):
                q, k_cache, v_cache, block_table, atten_mask, out, lse, page_size = refs[i]
                if q is None or k_cache is None or out is None:
                    continue
                kw = self._fia_kwargs(
                    q, k_cache, v_cache, block_table, atten_mask, page_size,
                    self._tnd_seq_lens(q.shape[0], num_streams), ctx_list,
                )
                torch.npu.graph_task_update_begin(us, handle)
                torch_npu.npu_fused_infer_attention_score_v2.out(
                    **kw, workspace=self._shared_workspace_tensor(kw, q), out=[out, lse]
                )
                torch.npu.graph_task_update_end(us)
                events[i].record(us)