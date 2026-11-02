"""BF16 Indexer policy reserved for the synthetic DSA draft."""

import torch
from rtp_llm.models_py.modules.base.cuda.indexer_kv_op import IndexerKVOp
from rtp_llm.models_py.modules.dsv4._indexer_score_triton import v4_indexer_score
from rtp_llm.ops import compute_ops


class Bf16DraftIndexerOp(IndexerKVOp):
    def __init__(self, config, *, device="cuda"):
        if config.indexer_use_fp8 or config.index_rope_dim:
            raise ValueError("BF16 NoPE Indexer policy requires BF16 keys and no RoPE")
        super().__init__(config, device=device)

    def cache_view(self, cache, page_size):
        return cache.kv_cache_base.view(torch.bfloat16).view(-1, page_size, self.dim)

    def encode_q_k(self, query, key, cache, params, inputs, cp_params):
        if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
            raise ValueError("BF16 Indexer does not quantize Q or K")

        def write(value, slots):
            compute_ops.concat_and_cache_mla(
                value, value.new_empty((value.shape[0], 0)),
                self.cache_view(cache, params.kernel_page_size), slots, "auto", self.scale,
            )

        if params.is_prefill and getattr(params, "query_cp", False):
            params.ingress_plans["indexer"].run(key, write)
        else:
            write(key, params.slot_mapping)
        return query, query.new_ones((*query.shape[:-1], 1), dtype=torch.float32)

    def _score_prefill(self, query, gates, q_positions, keys, params,
                       execution, block, workspace, score):
        history = params.cache_adapter.read_prefix_chunk(
            keys, params.block_table, block.descriptor, workspace=workspace,
            allow_reserved_block_zero=params.is_warmup,
        )
        v4_indexer_score(
            query, history.unsqueeze(0), gates, q_positions,
            kv_positions=block.key_positions.unsqueeze(0),
            query_requests=params.query_requests.unsqueeze(0).to(torch.int32),
            key_requests=block.key_requests.unsqueeze(0), out=score.unsqueeze(0),
        )
        return score

    def _score_decode(self, query, gates, q_positions, keys, params,
                      execution, block, workspace, score):
        width = block.width
        history = workspace("cache.restore", (execution.batch, width, self.dim), torch.bfloat16)
        first = block.start // params.kernel_page_size
        count = width // params.kernel_page_size
        ids = execution.block_ids[:, first:first + count].reshape(-1)
        torch.index_select(keys, 0, ids,
                           out=history.view(-1, params.kernel_page_size, self.dim))
        v4_indexer_score(
            query, history, gates, q_positions,
            kv_positions=block.key_positions.unsqueeze(0).expand(execution.batch, -1),
            out=score.view(execution.batch, execution.queries, width),
        )
        return score
