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

    def _update_rope_kv_write_params(self, device, kv_cache, layer_idx: int = 0):
        # The kernel block granularity comes from the per-layer cache view; the
        # physical block size it maps onto drives slot_mapping and the writes.
        blocks_per_phys = _blocks_per_phys_from_config(self.attn_configs, self.attn_inputs)
        kernel_page = kv_cache.seq_size_per_block if kv_cache is not None else 0
        self.params.blocks_per_phys = blocks_per_phys
        positions, slot_mapping, mrope_positions = compute_ascend_attn_params(
            self.attn_inputs, layer_idx, kernel_page * blocks_per_phys
        )
        self.params.positions_d = positions.to(device, non_blocking=True)
        self.params.slot_mapping = slot_mapping.to(device, non_blocking=True)
        self.params.mrope_positions_d = (
            mrope_positions.to(device, non_blocking=True) if mrope_positions is not None else None
        )

    def prepare(self, attn_inputs):
        self.fmha_impl.prepare(attn_inputs)
        self.attn_inputs = attn_inputs
        # TODO: Ascend Is not called outside, will be called in graph mode

    def forward(self, qkv, kv_cache, layer_idx=0):
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
    """Encapsulate NPU prefill attention op, reads cache only."""

    _causal_mask = None

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

    def set_params(self, params):
        self.params = params

    def prepare(self, attn_inputs):
        self.block_table = attn_inputs.kv_cache_kernel_block_id
        self.blocks_per_phys = _blocks_per_phys_from_config(self.attn_configs, attn_inputs)
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

    def forward(self, q, kv_cache):
        k_cache, v_cache, page_size = self._split_kv(kv_cache)
        block_table = self.block_table
        if block_table is not None and block_table.device.type != q.device.type:
            block_table = block_table.to(q.device)
        actual_seq_q = self.actual_seq_q
        if actual_seq_q is not None and actual_seq_q.device.type != q.device.type:
            actual_seq_q = actual_seq_q.to(q.device)
        actual_seq_kv = self.actual_seq_kv
        if actual_seq_kv is not None and actual_seq_kv.device.type != q.device.type:
            actual_seq_kv = actual_seq_kv.to(q.device)
        atten_mask = self._get_causal_mask(q.device)
        attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query=q, key=k_cache, value=v_cache,
            atten_mask=atten_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=page_size,
            actual_seq_qlen=actual_seq_q,
            actual_seq_kvlen=actual_seq_kv,
            num_key_value_heads=self.num_kv_heads,
            num_query_heads=self.num_heads,
            softmax_scale=self.scale,
            sparse_mode=3,
        )
        return attn_output