"""Decode MLA over the existing replicated, physical-page SWA cache."""

from math import gcd
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
import triton
from flash_mla import flash_mla_sparse_fwd

from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferImplBase,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.sliding_window_mla_kernels import (
    build_swa_indices,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.rope_emb_new import (
    NewMlaRotaryEmbeddingOp,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W


class SlidingWindowMlaDecodeOp:
    def __init__(self, config: AttentionConfigs, weights: List[Dict[str, torch.Tensor]]):
        if config.is_sparse or config.kv_cache_dtype != KvCacheDataType.BASE:
            raise ValueError("SWA MLA requires dense queries and BASE KV cache")
        if not 0 < config.head_num <= 128 or config.sliding_window <= 0:
            raise ValueError("SWA MLA requires 1–128 query heads and a positive window")
        self.num_heads = config.head_num
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.rope_head_dim
        self.qk_nope_head_dim = config.nope_head_dim
        self.window = config.sliding_window
        self.weights = weights
        self.page_size = config.tokens_per_block
        self.scale = (config.nope_head_dim + config.rope_head_dim) ** -0.5
        self.scale *= config.softmax_extra_scale

    def plan(self, params):
        self.params = params

    def forward(self, q_nope, q_pe, kv_cache: LayerKVCache, layer_id: int):
        params = self.params
        cache = kv_cache.kv_cache_base
        rows = q_nope.shape[0]
        dim = self.kv_lora_rank + self.qk_rope_head_dim
        pad_window = triton.cdiv(self.window, 128) * 128
        heads = 128
        query = torch.zeros((rows, heads, dim), dtype=q_nope.dtype, device=q_nope.device)
        torch.bmm(
            q_nope.transpose(0, 1), self.weights[layer_id][W.mla_kc],
            out=query[:, :self.num_heads, :self.kv_lora_rank].transpose(0, 1),
        )
        query[:, :self.num_heads, self.kv_lora_rank:] = q_pe
        table = params.block_table
        geometry = (
            table.stride(0), table.stride(1), self.page_size, self.window,
            pad_window, cache.shape[1], cache.stride(0), cache.stride(1),
        )
        # A padded K-page tensor cannot be flattened with view(). Index a
        # strided storage lattice instead: every real token start is exactly
        # divisible by this stride, without copying or addressing padding.
        index_stride = gcd(cache.stride(0), cache.stride(1))
        last_start = (cache.shape[0] - 1) * cache.stride(0)
        last_start += (cache.shape[1] - 1) * cache.stride(1)
        lattice_rows = last_start // index_stride + 1
        if lattice_rows > 2**31 - 1:
            raise ValueError("SWA cache exceeds the indexed MLA int32 row range")
        cache_view = cache.as_strided(
            (lattice_rows, 1, dim), (index_stride, dim, 1)
        )
        indices = torch.empty(
            (rows, 1, pad_window), dtype=torch.int32, device=cache.device
        )
        build_swa_indices[(rows, triton.cdiv(pad_window, 256))](
            table, params.positions_d, params.batch_indice_d,
            params.valid_queries, indices, *geometry, index_stride, 256,
        )
        attended = flash_mla_sparse_fwd(
            query, cache_view, indices, self.scale, d_v=self.kv_lora_rank
        )[0][:, :self.num_heads]
        attended = torch.where(params.valid_queries[:, None, None], attended, 0)
        return torch.bmm(
            attended.transpose(0, 1), self.weights[layer_id][W.mla_vc]
        ).transpose(0, 1)


class SlidingWindowMlaDecodeImpl(MlaFlashInferImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
        parallelism_config: Optional[ParallelismConfig] = None,
    ):
        super().__init__(
            SlidingWindowMlaDecodeOp(attn_configs, weights),
            NewMlaRotaryEmbeddingOp(
                cos_sin_cache, attn_configs.rope_config.is_neox_style
            ),
            MlaKVCacheWriteOp(attn_configs.kv_cache_dtype),
            attn_inputs,
            # SWA tables name physical pages even when the tensor is a K-page view.
            attn_configs.tokens_per_block,
            attn_configs,
            weights,
            cos_sin_cache,
            fmha_config,
            quant_config=quant_config,
            max_seq_len=max_seq_len,
            is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism_config,
            warmup_flashinfer=False,
        )

    def create_params(self, attn_inputs: PyAttentionInputs):
        self.fmha_params = self.rope_params = SimpleNamespace(
            positions_d=None, batch_indice_d=None, slot_mapping=None
        )
        self.prepare(attn_inputs)

    def prepare(self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False):
        self.attn_inputs = attn_inputs
        self.fmha_impl.plan(self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        self.prepare(attn_inputs)

    def forward(self, q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=None):
        attn_inputs = self.attn_inputs
        select_block_map_for_layer(attn_inputs, layer_id)
        table = attn_inputs.kv_cache_kernel_block_id_device
        self.fmha_params.block_table = table
        multi = attn_inputs.is_target_verify or attn_inputs.is_mtp_draft_update
        # Infer rectangular rows from the actual producer, including native
        # warmup/padding. These device operations are captured with their users.
        queries = q.shape[0] // table.shape[0]
        offsets = torch.arange(queries, device=table.device, dtype=torch.int32)
        if multi:
            base = attn_inputs.prefix_lengths
            valid = offsets[None, :] < attn_inputs.input_lengths[:, None]
        else:
            plus_one = attn_inputs.sequence_lengths_plus_1_d
            # Native q1 capture uses a zero plus-one placeholder but has a real
            # maximum sequence descriptor; eager legacy callers may omit it.
            base = attn_inputs.sequence_lengths
            if plus_one is not None and plus_one.numel():
                base = torch.where(plus_one > 0, plus_one - 1, base)
            valid = base[:, None] >= 0
        positions = torch.where(valid, base[:, None] + offsets, 0).flatten()
        requests = torch.arange(table.shape[0], device=table.device, dtype=torch.int32)
        requests = requests.repeat_interleave(queries)
        page = self.seq_size_per_block
        physical = table[requests.long(), (positions // page).long()]
        slots = physical.long() * page + positions % page
        slots = torch.where(valid.flatten() & (physical > 0), slots, -1)
        params = self.fmha_params
        params.valid_queries = valid.expand(-1, queries).flatten() & (physical > 0)
        params.positions_d = positions
        params.batch_indice_d = requests
        params.slot_mapping = slots
        return super().forward(q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices)
