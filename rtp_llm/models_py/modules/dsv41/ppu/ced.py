"""PPU bounded query metadata; public CED owns transport and restoration.

The public executor selects its own exact plan. Compact140 enters through
the existing PPU Block boundary at L21, without replacing that executor.
"""

from dataclasses import replace
import os

import torch

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.dsv41.prefill.ced import (
    CEDPlan,
    _RowExchange,
    _bounded_replay_selected,
    _query_layout,
)


class _LocalRowExchange(_RowExchange):
    """Transport CED rows within one attention owner in DP Prefill."""

    def compact(self, value):
        selected = value.index_select(0, self.send_indices)
        if self.receive_is_identity:
            return selected
        output = value.new_zeros((self.compact_rows, *value.shape[1:]))
        return output.index_copy_(0, self.receive_positions, selected)

    def restore(self, value, out=None):
        selected = value.index_select(0, self.receive_positions)
        if out is None:
            out = value.new_empty((self.original_rows, *value.shape[1:]))
        return out.zero_().index_copy_(0, self.send_indices, selected)


def _bounded_query_layout(context, selected):
    group = collective_torch._get_group(Group.TP) if context.cp_size > 1 else None
    query, exchange, groups = _query_layout(context, selected, group, keep_candidate_rows=True)
    if context.cp_size == 1:
        exchange = _LocalRowExchange(**vars(exchange))
    return query, exchange, groups


def bounded_plan(v4, context, attn_inputs, kv_cache, policy):
    """Reuse public admission, then replace only the bounded query layout.

    The temporary public128 plan is never executed. In particular, its
    padding is not counted as retained rows and no global is patched.
    """
    # Keep admission and actual inverse-map validation in the public planner.
    # B1/CP1 can directly build the final layout instead of allocating an
    # unconsumed public128 transport before the identical final140 transport.
    single_layout = (
        os.environ.get("RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA", "0") == "1"
        and context.cp_size == 1 and policy.keep_tokens == 140
        and len(context.input_lengths_global_host) == 1
        and context.input_lengths_global_host[0] > policy.keep_tokens
    )
    options = {"bounded_replay_keep_tokens": policy.keep_tokens} if single_layout else {}
    admitted = CEDPlan.create(
        v4, context, attn_inputs, kv_cache, bounded_replay=True, **options
    )
    if admitted is None:
        return None, "public_admission_or_no_compaction"
    if single_layout:
        exchange = _LocalRowExchange(**{**vars(admitted.exchange), "group": None})
        return replace(admitted, exchange=exchange), "compact140"
    lengths = context.input_lengths_global_host
    starts = tuple(max(n - policy.keep_tokens, 0) for n in lengths)
    selected = _bounded_replay_selected(lengths, starts)
    if selected.numel() == context.seq_len_full:
        return None, "all_available_rows_retained"
    query, exchange, groups = _bounded_query_layout(context, selected)
    if len(starts) == 1:
        query.swa_replay_start = starts[0]
    else:
        query.swa_replay_starts_host = starts
    cumulative = [0]
    for chunk in query.chunk_lengths_per_req:
        cumulative.append(cumulative[-1] + chunk)
    if context.cp_size == 8 and context.kv_cache_sharded:
        from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata import _upload

        cu = _upload(cumulative, context.global_positions.device, torch.int32)
    else:
        cu = torch.tensor(cumulative, dtype=torch.int32, device=context.global_positions.device)
    return replace(
        admitted,
        context=query,
        exchange=exchange,
        indexer_groups=groups,
        cu_seqlens=cu,
    ), "compact140"
