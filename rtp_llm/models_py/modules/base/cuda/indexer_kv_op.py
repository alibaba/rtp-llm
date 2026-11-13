"""Precision-neutral Indexer KV execution, deterministic selection and DCP."""

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather_into,
    all_to_all_single,
)
from rtp_llm.models_py.modules.dsv4.indexer_topk import (
    NativeExactIndexerTopKBackend,
    TorchIndexerTopKBackend,
    native_exact_topk_enabled,
    native_page_rr_selection_supported,
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
            if native_exact_topk_enabled()
            else TorchIndexerTopKBackend(stable_ties=True)
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
                (
                    query
                    if getattr(params, "query_cp", False)
                    else query.index_select(0, params.query_indices)
                )
                .unsqueeze(0)
                .contiguous()
            )
            gates = (
                (
                    gates
                    if getattr(params, "query_cp", False)
                    else gates.index_select(0, params.query_indices)
                )
                .unsqueeze(0)
                .contiguous()
            )
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
        if execution.mega_topk:
            return self._mega_topk_decode(
                query, gates, q_positions, keys, params, execution, workspace
            )
        running = (
            workspace(
                "running.scores", (execution.selection_rows, self.topk), torch.float32
            ),
            workspace(
                "running.positions", (execution.selection_rows, self.topk), torch.int32
            ),
        )
        native_fp8 = execution.fp8_indexer and isinstance(
            self.selector, NativeExactIndexerTopKBackend
        )
        for number, block in enumerate(execution.blocks):
            bounds = {}
            page_rr = False
            width = block.width
            score = (
                None
                if execution.fp8_indexer
                else workspace("score", (execution.rows, width), torch.float32)
            )
            if params.is_prefill:
                score = self._score_prefill(
                    query,
                    gates,
                    q_positions,
                    keys,
                    params,
                    execution,
                    block,
                    workspace,
                    score,
                )
                positions, complete = block.key_positions, score
            else:
                padded = execution.selection_rows * params.shard_size
                wire = workspace("wire", (padded, width), torch.float32)
                score = self._score_decode(
                    query,
                    gates,
                    q_positions,
                    keys,
                    params,
                    execution,
                    block,
                    workspace,
                    wire if execution.fp8_indexer else score,
                )
                if not execution.fp8_indexer:
                    wire[: execution.rows].copy_(score)
                    if padded > execution.rows:
                        wire[execution.rows :].fill_(-float("inf"))
                received = workspace("received", (padded, width), torch.float32)
                received = all_to_all_single(wire, Group.TP, output=received)
                page_rr = native_fp8 and native_page_rr_selection_supported(
                    width,
                    self.topk,
                    shards=params.shard_size,
                    page=params.page_size,
                    start=block.start,
                )
                if page_rr:
                    complete = received
                else:
                    complete = workspace(
                        "complete",
                        (execution.selection_rows, width * params.shard_size),
                        torch.float32,
                    )
                    if native_fp8:
                        from rtp_llm.models_py.kernels.cuda.dsa_decode_compact import (
                            restore_visible_decode_scores,
                        )

                        ends = workspace(
                            "decode.ends", (execution.selection_rows,), torch.int32
                        )
                        with profile_scope(
                            "RTP::attention.dsa.indexer.decode_visible_restore"
                        ):
                            restore_visible_decode_scores(
                                received,
                                params.positions_d,
                                complete,
                                ends,
                                shard_rank=params.shard_rank,
                                shards=params.shard_size,
                                page=params.page_size,
                                start=block.start,
                            )
                        positions = block.indexer_decode_positions
                        bounds = {"ends": ends}
                    else:
                        complete.view(
                            execution.selection_rows, params.shard_size, width
                        ).copy_(
                            received.view(
                                params.shard_size, execution.selection_rows, width
                            ).transpose(0, 1)
                        )
                        positions = block.dcp_positions
            with profile_scope("RTP::attention.dsa.indexer.chunk_topk"):
                if page_rr:
                    current = self.selector.select_page_rr_candidates(
                        complete,
                        params.positions_d,
                        self.topk,
                        workspace=workspace,
                        out=running if number == 0 else None,
                        shard_rank=params.shard_rank,
                        shards=params.shard_size,
                        page=params.page_size,
                        start=block.start,
                    )
                else:
                    if params.is_prefill and native_fp8:
                        bounds = {
                            "starts": block.indexer_starts,
                            "ends": block.indexer_ends,
                        }
                    current = self.selector.select_candidates(
                        complete,
                        positions,
                        self.topk,
                        workspace=workspace,
                        scores_are_sanitized=self.scores_are_sanitized
                        and params.is_prefill,
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
        execution.finish()
        return output

    def _mega_topk_decode(
        self, query, gates, q_positions, keys, params, execution, workspace
    ):
        """Decode selection by MegaTopk: no score exchange, no all-gather of positions.

        Scores this rank's whole local history (KV blocks side by side), then selects each row's
        exact global top-k across the DCP group in one kernel. Returns [rows, topk] holding the
        selected global positions this rank owns, then -1: sparse attention reads only its own
        positions, and the DCP combine treats a rank without keys as empty.
        """
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mega_topk import (
            get_page_rr_mega_topk,
        )

        page, rows = params.kernel_page_size, execution.rows
        whole = len(execution.blocks) == 1
        logits = (
            None
            if whole
            else workspace(
                "mega_topk.logits",
                (rows, params.block_table.shape[1] * page),
                torch.float32,
            )
        )
        for block in execution.blocks:
            score = self._score_decode(
                query,
                gates,
                q_positions,
                keys,
                params,
                execution,
                block,
                workspace,
                None,
            )
            if whole:
                logits = score
            else:
                logits[:, block.start : block.start + block.width].copy_(score)
            del score
        selector = get_page_rr_mega_topk(
            self.topk,
            (params.max_owner_columns or params.block_table.shape[1]) * page,
            page,
            params.page_size,
            query.device,
        )
        output = workspace("mega_topk.positions", (rows, self.topk), torch.int32)
        count = workspace("mega_topk.count", (rows,), torch.int32)
        with profile_scope("RTP::attention.dsa.indexer.mega_topk"):
            selector(
                logits,
                params.decode.local_causal_lens.view(-1),
                params.positions_d,
                params.block_table,
                execution.queries,
                output,
                count,
            )
        execution.finish()
        return output
