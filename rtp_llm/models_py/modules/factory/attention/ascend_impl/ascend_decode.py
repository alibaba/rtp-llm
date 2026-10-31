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




class AscendDecodeImpl(FMHAImplBase):
    """Ascend MHA Decode using FIA v2.

    Eager: FIA v2 + .reshape() (contiguous copy).
    Graph: FIA v2 + graph_task_group + graph_task_update.
    Hybrid KV cache (Qwen3.5): per-layer views are at kernel-block granularity;
    slot_mapping addresses the physical storage (blocks_per_phys > 1).
    """

    def __init__(self, attn_configs, attn_inputs, parallelism_config):
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.fmha_params = None

        self.fmha_impl = AscendDecodeAttnOp(attn_configs, attn_inputs)
        self.rope_impl = self._create_rope_impl(attn_configs)
        self.kv_cache_write_op = AscendKVCacheWriteOp(
            num_kv_heads=attn_configs.kv_head_num,
            head_size=attn_configs.size_per_head,
            token_per_block=attn_inputs.kv_cache.seq_size_per_block if attn_inputs.kv_cache else 128,
        )

        self.params = AscendAttnParams()
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

    def _update_rope_kv_write_params(self, device, kv_cache, layer_idx: int = 0):
        # The kernel block granularity comes from the cache config; the
        # physical block size it maps onto drives slot_mapping and the writes.
        blocks_per_phys = _blocks_per_phys_from_config(self.attn_configs, self.attn_inputs)
        kernel_page = kv_cache.seq_size_per_block if kv_cache is not None else 0
        self.params.blocks_per_phys = blocks_per_phys
        self.fmha_impl.blocks_per_phys = blocks_per_phys
        if getattr(self.attn_inputs, "is_cuda_graph", False):
            self._update_rope_kv_write_params_device(device, kernel_page, blocks_per_phys)
            return
        positions, slot_mapping, mrope_positions = compute_ascend_attn_params(
            self.attn_inputs, layer_idx, kernel_page * blocks_per_phys
        )
        self.params.positions_d = positions.to(device, non_blocking=True)
        self.params.slot_mapping = slot_mapping.to(device, non_blocking=True)
        self.params.mrope_positions_d = (
            mrope_positions.to(device, non_blocking=True) if mrope_positions is not None else None
        )

    def _update_rope_kv_write_params_device(self, device, kernel_page, blocks_per_phys):
        seq_lens_plus_1 = self.attn_inputs.sequence_lengths_plus_1_device
        positions_d = seq_lens_plus_1 - 1

        block_table = self.attn_inputs.kv_cache_kernel_block_id_device
        if not kernel_page:
            kernel_page = (self.attn_inputs.kv_cache.seq_size_per_block
                           if self.attn_inputs.kv_cache is not None else 128)
        # kernel-flat slots (kb * kernel_page + offset), paired with the
        # kernel-block views in the KV write op / FIA reads
        if block_table is not None and block_table.numel() > 0 and positions_d.numel() > 0:
            if block_table.ndim != 2:
                block_table = block_table.reshape(-1, block_table.shape[-1])
            max_blocks = block_table.size(1)
            pos_long = positions_d.long()
            block_index = (pos_long // kernel_page).clamp(max=max_blocks - 1)
            block_offset = pos_long % kernel_page
            slot_block_numbers = torch.gather(
                block_table, 1,
                block_index.unsqueeze(1).to(block_table.dtype)
            ).squeeze(1).long().clamp(min=0)
            slot_mapping = (slot_block_numbers * kernel_page + block_offset).to(torch.int64)
        else:
            slot_mapping = torch.empty(0, dtype=torch.int64, device=device)

        self.params.positions_d = positions_d
        self.params.slot_mapping = slot_mapping
        # Graph-mode MRoPE: read the three-axis positions from the captured
        # combo_position_ids buffer ([T*3] flat on device) when present.
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

    def signal_graph_events(self, graph_bs):
        """Called by AscendGraphRunner before the post-capture sanity replay:
        release the FIA ExternalEvent wait points once so the check replay
        does not deadlock (real replays are released by update_graph_fia)."""
        self.fmha_impl.signal_graph_events(graph_bs)

    def prepare_cuda_graph(self, attn_inputs, graph_bs=None):
        """Called by AscendGraphRunner::prepareInputs() before each replay."""
        self.attn_inputs = attn_inputs
        self.fmha_impl.prepare(attn_inputs)
        # The capture buffer's sequence_lengths is sized [max_bs]; padded rows
        # were zeroed by the C++ runner. The FIA graph-task update must carry
        # exactly as many context lengths as the captured batch size of the
        # replayed graph instance, otherwise the update is dropped and FIA
        # keeps the capture-time kvlen.
        seq = attn_inputs.sequence_lengths
        valid = int((seq != 0).sum().item()) if seq.numel() else 0
        valid = max(valid, 1)
        if graph_bs is None:
            graph_bs = valid
        seq_lens = seq[:valid]
        ctx_list = (seq_lens.to(torch.int32) + 1).tolist()
        # pad rows up to the captured graph batch size (ctx=1 is a no-op read)
        ctx_list = ctx_list[:graph_bs] + [1] * max(0, graph_bs - len(ctx_list))
        self.fmha_impl.update_graph_fia(ctx_list, graph_bs)

    def forward(self, qkv, kv_cache, layer_idx=0):
        is_graph = getattr(self.attn_inputs, "is_cuda_graph", False)

        if self.need_rope_kv_cache:
            self._update_rope_kv_write_params(qkv.device, kv_cache, layer_idx)
            if self.rope_impl is not None:
                query, key, value = self.rope_impl.forward(qkv)
            else:
                query, key, value = self._split_qkv(qkv)
            self.kv_cache_write_op.forward(key, value, kv_cache)
            q = query
        else:
            q = qkv.chunk(3, dim=-1)[0]

        if is_graph:
            self.fmha_impl.context_lens = self.attn_inputs.sequence_lengths_plus_1_device
        else:
            self.fmha_impl.context_lens = self.attn_inputs.sequence_lengths + 1

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(q, kv_cache, is_graph)

    @staticmethod
    def support(attn_configs, attn_inputs):
        from rtp_llm.ops import RopeStyle
        # MRoPE variants are applied by AscendRotaryEmbeddingOp.
        return not attn_inputs.is_prefill and \
               not attn_configs.use_mla and \
               torch.npu.is_available()


class AscendDecodeAttnOp:
    """NPU decode attention: FIA v2 (eager) or FIA v2 + graph_task_group (graph).

    Graph mode uses the vllm-ascend pattern:
    - Capture: graph_task_group_begin/end wraps FIA v2 .out() with pre-computed workspace
    - Replay:  graph_task_update_begin/end updates context_lens dynamically

    Hybrid KV cache: the per-layer view interleaves K and V once a physical
    block is subdivided into kernel blocks, so K/V are split explicitly and the
    FIA block_size follows the kernel page of the layer's cache group.
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
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scale = attn_configs.q_scaling * self.head_dim ** -0.5
        self.page_size = attn_inputs.kv_cache.seq_size_per_block if \
                         attn_inputs.kv_cache else 128
        self.block_table = None
        self.context_lens = None
        self.blocks_per_phys = _blocks_per_phys_from_config(attn_configs, attn_inputs)

        # FIA graph tasks grouped by captured batch size: each capture bs
        # bucket registers its own handles; a replay of one bucket must only
        # update that bucket's tasks (ctx list length must match the captured
        # batch size or the update is dropped).
        self._graph_handles = {}  # bs -> [task handles]
        self._graph_refs = {}     # bs -> [(q, k, v, table, mask, out, lse, page)]
        self._graph_events = {}   # bs -> [ExternalEvent] for update/replay ordering

    def set_params(self, params):
        self.params = params

    def prepare(self, attn_inputs):
        if self.blocks_per_phys <= 0:
            self.blocks_per_phys = infer_blocks_per_phys(attn_inputs)
        if getattr(attn_inputs, "is_cuda_graph", False):
            self.block_table = attn_inputs.kv_cache_kernel_block_id_device
            if self.block_table is not None:
                if self.block_table.ndim != 2:
                    self.block_table = self.block_table.reshape(-1, self.block_table.shape[-1])
            seq_plus1 = attn_inputs.sequence_lengths_plus_1_device
            self.context_lens = seq_plus1 if (seq_plus1 is not None and seq_plus1.numel() > 0) else None
            return
        self.block_table = attn_inputs.kv_cache_kernel_block_id
        if self.block_table is not None:
            self.block_table = self.block_table.clamp(min=0)
            if self.block_table.ndim != 2:
                self.block_table = self.block_table.reshape(-1, self.block_table.shape[-1])
        seq = attn_inputs.sequence_lengths
        self.context_lens = (seq + 1) if (seq is not None and seq.numel() > 0) else None

    def forward(self, q, kv_cache, use_graph=False):
        block_table = self.block_table
        if block_table is not None and block_table.device.type != q.device.type:
            block_table = block_table.to(q.device)
        context_lens = self.context_lens
        if context_lens is not None and context_lens.device.type != q.device.type:
            context_lens = context_lens.to(q.device)
        if use_graph and torch.npu.is_current_stream_capturing():
            return self._forward_fia_graph(q, kv_cache, block_table, context_lens)
        return self._forward_fia(q, kv_cache, block_table, context_lens)

    def _split_kv(self, kv_cache):
        # Zero-copy kernel-block K/V views in the 3-D form FIA requires
        # (the old split_kv_kernel_blocks route materialised a copy).
        base = kv_cache.kv_cache_base
        k_cache = base[:, 0].reshape(base.shape[0], base.shape[2], -1)
        v_cache = base[:, 1].reshape(base.shape[0], base.shape[2], -1)
        return k_cache, v_cache, base.shape[2]

    def _forward_fia_graph(self, q, kv_cache, block_table, context_lens):
        k_cache, v_cache, page_size = self._split_kv(kv_cache)
        batch_size = q.shape[0]
        out = torch.empty(batch_size, self.num_heads, self.head_dim,
                          dtype=q.dtype, device=q.device)
        lse = torch.empty(1, dtype=q.dtype, device=q.device)
        atten_mask = self._get_causal_mask(q.device)

        if AscendDecodeAttnOp._shared_workspace is None:
            AscendDecodeAttnOp._shared_workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                query=q, key=k_cache, value=v_cache,
                atten_mask=atten_mask, block_table=block_table,
                input_layout="TND", block_size=page_size,
                actual_seq_qlen=list(range(1, batch_size + 1)),
                actual_seq_kvlen=[page_size] * batch_size,
                num_key_value_heads=self.num_kv_heads,
                num_query_heads=self.num_heads,
                softmax_scale=self.scale, sparse_mode=3)
            ws_bytes = AscendDecodeAttnOp._shared_workspace.numel() * AscendDecodeAttnOp._shared_workspace.element_size()
            AscendDecodeAttnOp._shared_workspace = torch.empty(ws_bytes // q.element_size(),
                                                dtype=q.dtype, device=q.device)

        if not torch.npu.is_current_stream_capturing():
            actual_seq_q = torch.arange(1, batch_size + 1, dtype=torch.int32, device=q.device)
            actual_seq_kv = context_lens.to(torch.int32)
            attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query=q, key=k_cache, value=v_cache,
                atten_mask=atten_mask, block_table=block_table,
                input_layout="TND", block_size=page_size,
                actual_seq_qlen=actual_seq_q, actual_seq_kvlen=actual_seq_kv,
                num_key_value_heads=self.num_kv_heads,
                num_query_heads=self.num_heads,
                softmax_scale=self.scale, sparse_mode=3)
            return attn_output

        actual_seq_q = list(range(1, batch_size + 1))
        actual_seq_kv = [page_size] * batch_size
        stream = torch.npu.current_stream()
        # ExternalEvent: the graph records a wait point before this FIA task;
        # each replay blocks there (device-side, no host stall) until the
        # update stream records the event after refreshing ctx_lens.
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score_v2.out(
            query=q, key=k_cache, value=v_cache,
            atten_mask=atten_mask, block_table=block_table,
            input_layout="TND", block_size=page_size,
            actual_seq_qlen=actual_seq_q, actual_seq_kvlen=actual_seq_kv,
            num_key_value_heads=self.num_kv_heads,
            num_query_heads=self.num_heads,
            sparse_mode=3,
            softmax_scale=self.scale,
            workspace=AscendDecodeAttnOp._shared_workspace,
            out=[out, lse])
        handle = torch.npu.graph_task_group_end(stream)
        # NOTE: no record here — anything issued inside the graph context is
        # recorded as a graph op. The sanity replay gets its signal from
        # signal_graph_events() invoked by the C++ runner (outside the graph).

        self._graph_handles.setdefault(batch_size, []).append(handle)
        self._graph_refs.setdefault(batch_size, []).append((
            q, k_cache, v_cache,
            block_table, atten_mask,
            out, lse, page_size))
        self._graph_events.setdefault(batch_size, []).append(event)
        return out

    def signal_graph_events(self, batch_size):
        for ev in self._graph_events.get(batch_size, []):
            ev.record(torch.npu.current_stream())

    def update_graph_fia(self, ctx_list, batch_size):
        import os as _os
        handles = self._graph_handles.get(batch_size)
        if not handles:
            return
        if _os.environ.get("RTP_LLM_SKIP_FIA_UPDATE"):
            stream = torch.npu.current_stream()
            for ev in self._graph_events.get(batch_size, []):
                ev.record(stream)
            return
        events = self._graph_events.get(batch_size, [])
        if AscendDecodeAttnOp._shared_update_stream is None:
            AscendDecodeAttnOp._shared_update_stream = torch.npu.Stream()
        us = AscendDecodeAttnOp._shared_update_stream
        actual_seq_q = list(range(1, batch_size + 1))
        refs = self._graph_refs.get(batch_size, [])
        with torch.npu.stream(us):
            for i, handle in enumerate(handles):
                q, k_cache, v_cache, block_table, atten_mask, out, lse, page_size = refs[i]
                if q is None or k_cache is None or out is None:
                    continue
                torch.npu.graph_task_update_begin(us, handle)
                torch_npu.npu_fused_infer_attention_score_v2.out(
                    query=q, key=k_cache, value=v_cache,
                    atten_mask=atten_mask, block_table=block_table,
                    input_layout="TND", block_size=page_size,
                    actual_seq_qlen=actual_seq_q, actual_seq_kvlen=ctx_list,
                    num_key_value_heads=self.num_kv_heads,
                    num_query_heads=self.num_heads,
                    sparse_mode=3,
                    softmax_scale=self.scale,
                    workspace=AscendDecodeAttnOp._shared_workspace,
                    out=[out, lse])
                torch.npu.graph_task_update_end(us)
                events[i].record(us)
        # No host synchronize: the graph's recorded ExternalEvent wait point
        # orders the FIA task after this update on-device; replay can be
        # launched immediately from C++ without host stall.

    def _forward_fia(self, q, kv_cache, block_table, context_lens):
        k_cache, v_cache, page_size = self._split_kv(kv_cache)
        batch_size = q.shape[0]
        actual_seq_q = torch.arange(1, batch_size + 1, dtype=torch.int32, device=q.device)
        actual_seq_kv = context_lens.to(torch.int32)
        if actual_seq_kv.device.type != q.device.type:
            actual_seq_kv = actual_seq_kv.to(q.device)
        atten_mask = self._get_causal_mask(q.device)
        attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query=q, key=k_cache, value=v_cache,
            atten_mask=atten_mask, block_table=block_table,
            input_layout="TND", block_size=page_size,
            actual_seq_qlen=actual_seq_q, actual_seq_kvlen=actual_seq_kv,
            num_key_value_heads=self.num_kv_heads,
            num_query_heads=self.num_heads,
            softmax_scale=self.scale, sparse_mode=3,
        )
        return attn_output
