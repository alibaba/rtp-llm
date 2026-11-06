"""K3.1 FP8 cache/score policy over the common Indexer KV/TopK/DCP loop."""

import torch

from rtp_llm.models_py.kernels.cuda.fp8_indexer_cache import (
    canonical_indexer_views, pack_indexer_pages64, prepare_direct_indexer_cache,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.base.cuda.indexer_kv_op import IndexerKVOp
from rtp_llm.models_py.modules.dsv4.indexer_topk import NativeExactIndexerTopKBackend
from rtp_llm.models_py.modules.base.cuda.fp8_indexer_score import NativeFp8IndexerScore
from rtp_llm.models_py.modules.base.cuda.indexer_op import _unpack_ue8m0_scale
from rtp_llm.ops.compute_ops import rtp_llm_ops


class Fp8IndexerOp(IndexerKVOp):
    # Prefill consumes the adapter's owned, sanitized output directly.
    # Decode writes sanitized scores directly to the DCP input.
    scores_are_sanitized = True

    def __init__(self, config, *, device="cuda"):
        if (not config.indexer_use_fp8 or config.index_rope_dim
                or config.index_num_heads != 32 or config.index_head_dim != 128):
            raise ValueError("K3.1 FP8 Indexer requires NoRoPE and all32 group128 heads")
        super().__init__(config, device=device)
        self.native_score = NativeFp8IndexerScore()
        self._bounded_selection = isinstance(self.selector, NativeExactIndexerTopKBackend)

    def cache_view(self, cache, page_size):
        raw = cache.kv_cache_base
        if raw.dtype != torch.uint8 or not raw.is_contiguous():
            raise ValueError("FP8 Indexer requires opaque UINT8 cache storage")
        return raw.view(-1, page_size, 132)

    def encode_q_k(self, query, key, cache, params, inputs, cp_params):
        if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
            raise ValueError("FP8 Indexer projections and K LayerNorm must output BF16")
        self.write_key(key, cache, params)
        return self.encode_query(query)

    def write_key(self, key, cache, params):
        if key.dtype != torch.bfloat16:
            raise ValueError("FP8 Indexer K LayerNorm must output BF16")

        def write(values, slots):
            rtp_llm_ops.indexer_k_quant_and_cache(
                values, self.cache_view(cache, params.kernel_page_size), slots, 128, "ue8m0"
            )

        if params.is_prefill and getattr(params, "query_cp", False):
            # Producer-to-owner transport retains the established BF16 ingress;
            # only the owner quantizes/writes its persistent K and FP32 scale.
            params.ingress_plans["indexer"].run(key, write)
        else:
            write(key, params.slot_mapping)

    def encode_query(self, query):
        if query.dtype != torch.bfloat16:
            raise ValueError("FP8 Indexer query projection must output BF16")
        q, scales = sgl_per_token_group_quant_fp8(
            query.reshape(-1, 128).contiguous(), group_size=128, eps=1e-4,
            column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True,
        )
        return q.view(query.shape), _unpack_ue8m0_scale(scales).reshape(*query.shape[:-1], 1)

    def _score_prefill(self, query, gates, q_positions, keys, params,
                       execution, block, workspace, score):
        raw = params.cache_adapter.read_prefix_chunk(
            keys, params.block_table, block.descriptor,
            workspace=workspace, indexer_fp8=True,
            allow_reserved_block_zero=params.is_warmup,
        )
        key, scales = canonical_indexer_views(raw)
        return self.native_score.dense(
            query[0], key, scales, gates[0], block.indexer_starts,
            block.indexer_ends, block.key_positions,
            sanitize=not self._bounded_selection,
        )

    def _score_decode(self, query, gates, q_positions, keys, params,
                      execution, block, workspace, wire):
        if execution.direct_native_page_size is not None:
            prepare_direct_indexer_cache(keys, block.native_table, block.native_context)
            native = keys.unsqueeze(2)
        else:
            native = workspace("cache.native64",
                               (1 + execution.batch * (block.width // 64), 64, 1, 132),
                               torch.uint8)
            pack_indexer_pages64(keys, params.block_table, block.start, native)
        return self.native_score.paged(
            query, native, gates.reshape(execution.rows, 32), block.native_context,
            block.native_table, block.native_schedule, block.width,
            block.key_positions, wire=wire,
        )
