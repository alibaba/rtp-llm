import torch
import torch_npu

from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_attn_params import (
    AscendAttnParams,
    compute_ascend_attn_params,
)
from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_kv_cache_write_op import AscendKVCacheWriteOp
from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_rope_emb import AscendRotaryEmbeddingOp
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.modules.factory.attention import common


class AscendDecodeImpl(FMHAImplBase):
    """Ascend MHA Decode using torch_npu._npu_paged_attention.

    Composes RoPE -> KVCacheWrite -> write_cache_store -> paged_attention.
    """

    def __init__(self, attn_configs, attn_inputs, parallelism_config):
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs

        self.fmha_impl = AscendDecodeAttnOp(attn_configs, attn_inputs)
        self.rope_impl = self._create_rope_impl(attn_configs)
        self.kv_cache_write_op = AscendKVCacheWriteOp(
            num_kv_heads=attn_configs.kv_head_num,
            head_size=attn_configs.size_per_head,
            token_per_block=attn_inputs.kv_cache.seq_size_per_block,
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
        key = k.reshape(k.shape[0], num_kv_heads, head_dim)
        value = v.reshape(v.shape[0], num_kv_heads, head_dim)
        return query, key, value

    def _update_rope_kv_write_params(self, device):
        positions, slot_mapping = compute_ascend_attn_params(self.attn_inputs)
        self.params.positions_d = positions.to(device, non_blocking=True)
        self.params.slot_mapping = slot_mapping.to(device, non_blocking=True)

    def prepare(self, attn_inputs):
        self.fmha_impl.prepare(attn_inputs)
        self.attn_inputs = attn_inputs
        # TODO: Ascend Is not called outside, will be called in graph mode

    def forward(self, qkv, kv_cache, layer_idx=0):
        if self.need_rope_kv_cache:
            self._update_rope_kv_write_params(qkv.device)

            if self.rope_impl is not None:
                query, key, value = self.rope_impl.forward(qkv)
            else:
                query, key, value = self._split_qkv(qkv)

            self.kv_cache_write_op.forward(key, value, kv_cache)
            q = query
        else:
            q = qkv.chunk(3, dim=-1)[0]

        self.fmha_impl.context_lens = (
            self.attn_inputs.prefix_lengths + self.attn_inputs.input_lengths
        )

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(q, kv_cache)

    @staticmethod
    def support(attn_configs, attn_inputs):
        return not attn_inputs.is_prefill and \
               not attn_configs.use_mla and \
               attn_inputs.kv_cache is not None and \
               attn_inputs.kv_cache.separate_kv_cache


class AscendDecodeAttnOp:
    """Encapsulate NPU decode paged attention op, reads cache only."""

    def __init__(self, attn_configs, attn_inputs):
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scale = attn_configs.scale if attn_configs.scale else \
                     self.head_dim ** -0.5
        self.block_table = None
        self.context_lens = None

    def set_params(self, params):
        self.params = params

    def prepare(self, attn_inputs):
        self.block_table = attn_inputs.kv_cache_block_id_host
        if self.block_table is not None:
            self.block_table = self.block_table.clamp(min=0)
        self.context_lens = attn_inputs.prefix_lengths + attn_inputs.input_lengths

    def forward(self, q, kv_cache):
        output = torch.empty_like(q)
        torch_npu._npu_paged_attention(
            query=q,
            key_cache=kv_cache.k_cache_base,
            value_cache=kv_cache.v_cache_base,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=self.block_table,
            context_lens=self.context_lens,
            out=output,
        )
        return output
