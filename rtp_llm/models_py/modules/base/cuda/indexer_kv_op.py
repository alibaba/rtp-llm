"""Precision-neutral Indexer KV execution, deterministic selection and DCP."""

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather_into,
    all_to_all_single,
)
from rtp_llm.models_py.modules.dsv4.indexer_topk import (
    NativeExactIndexerTopKBackend, TorchIndexerTopKBackend, native_exact_topk_enabled,
)
from rtp_llm.models_py.utils.module_profiler import profile_scope


class IndexerKVOp(nn.Module):
    """One KV loop, selector and DCP path for independently bound precisions."""

    scores_are_sanitized = False

    def __init__(self, config, *, device="cuda"):
        super().__init__()
        self.dim = config.index_head_dim
        self.topk = config.index_topk
        self.selector = (
            NativeExactIndexerTopKBackend(stable_ties=True)
            if native_exact_topk_enabled() else TorchIndexerTopKBackend(stable_ties=True)
        )
        self.register_buffer(
            "scale", torch.ones((), dtype=torch.float32, device=device)
        )

    def _select(self, score, positions):
        # Encode the complete tie rule before direct selection, including
        # PageRR owner order. No history-wide stable argsort is required.
        return self.selector.select_candidates(score, positions, self.topk)[1]

    def compute_topk(self, query, weights, cache, params, inputs, cp_params):
        execution = params.indexer_execution
        keys = self.cache_view(cache, params.kernel_page_size)
        gates = weights.squeeze(-1)
        if params.is_prefill:
            query = (
                query if getattr(params, "query_cp", False)
                else query.index_select(0, params.query_indices)
            ).unsqueeze(0).contiguous()
            gates = (
                gates if getattr(params, "query_cp", False)
                else gates.index_select(0, params.query_indices)
            ).unsqueeze(0).contiguous()
            q_positions = params.query_positions.unsqueeze(0)
        else:
            query = query.view(
                execution.batch, execution.queries, *query.shape[1:]
            ).contiguous()
            gates = gates.view(
                execution.batch, execution.queries, gates.shape[-1]
            ).contiguous()
            q_positions = params.positions_d.view(execution.batch, execution.queries)
        if not execution.blocks:
            return torch.full(
                (execution.rows, self.topk), -1, dtype=torch.int32, device=query.device
            )
        workspace = execution.begin()
        running = (
            workspace(
                "running.scores", (execution.selection_rows, self.topk), torch.float32
            ),
            workspace(
                "running.positions", (execution.selection_rows, self.topk), torch.int32
            ),
        )
        for number, block in enumerate(execution.blocks):
            width = block.width
            score = (None if execution.fp8_indexer and params.is_prefill else
                     workspace("score", (execution.rows, width), torch.float32))
            if params.is_prefill:
                score = self._score_prefill(query, gates, q_positions, keys, params,
                                           execution, block, workspace, score)
                positions, complete = block.key_positions, score
            else:
                score = self._score_decode(query, gates, q_positions, keys, params,
                                          execution, block, workspace, score)
                padded = execution.selection_rows * params.shard_size
                wire = workspace("wire", (padded, width), torch.float32)
                wire[: execution.rows].copy_(score)
                if padded > execution.rows:
                    wire[execution.rows :].fill_(-float("inf"))
                received = workspace("received", (padded, width), torch.float32)
                received = all_to_all_single(wire, Group.TP, output=received)
                complete = workspace(
                    "complete",
                    (execution.selection_rows, width * params.shard_size),
                    torch.float32,
                )
                complete.view(execution.selection_rows, params.shard_size, width).copy_(
                    received.view(
                        params.shard_size, execution.selection_rows, width
                    ).transpose(0, 1)
                )
                positions = block.dcp_positions
            with profile_scope("RTP::attention.dsa.indexer.chunk_topk"):
                bounds = (
                    {"starts": block.indexer_starts, "ends": block.indexer_ends}
                    if params.is_prefill and execution.fp8_indexer
                    and isinstance(self.selector, NativeExactIndexerTopKBackend)
                    else {}
                )
                current = self.selector.select_candidates(
                    complete, positions, self.topk, workspace=workspace,
                    scores_are_sanitized=self.scores_are_sanitized and params.is_prefill,
                    out=running if number == 0 else None,
                    **bounds,
                )
            if number != 0:
                with profile_scope("RTP::attention.dsa.indexer.running_topk_merge"):
                    self.selector.merge_candidates(
                        running, current, workspace=workspace
                    )
            # MQA owns its output allocation. Release it before the next
            # KV block is launched, including the reference held by complete.
            del score, complete
        if params.is_prefill:
            output = running[1]
        else:
            output = workspace("gathered_indices", (padded, self.topk), torch.int32)
            all_gather_into(running[1], output, Group.TP)
            output = output[: execution.rows]
        from rtp_llm.models_py.utils.smoke_evidence import completed_dsa_chunks
        completed_dsa_chunks("indexer", params, len(execution.blocks), execution.rows)
        execution.finish()
        return output
