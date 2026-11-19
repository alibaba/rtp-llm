"""Request-scoped GLM53 MTP indexer seeds, in expanded raw-token coordinates."""

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.indexer_grouping import (
    append_incomplete_tail_indices,
    expand_indexer_group_indices,
    fused_expand_indexer_groups_with_tail,
    pad_empty_indexer_history,
)

MTP_INDEXER_NORMAL = 0
MTP_INDEXER_SEED = 1
MTP_INDEXER_REUSE = 2


def mtp_indexer_share_active(config, parallelism, layer_count, topk):
    cp = getattr(parallelism, "prefill_cp_config", None)
    return (
        bool(config.index_share_for_mtp_iteration)
        and parallelism.pp_size == 1
        and not (cp is not None and cp.is_enabled())
        and layer_count == 1
        and topk > 0
    )


class Glm53MtpIndexerSeeds:
    """Fixed-address storage shared by the three graph roles.

    The executor loads request-owned rows in the current batch order before
    replay. Reuse freezes the whole raw selection, including the seed's live
    tail; it must not append a later draft position.
    """

    def __init__(self, capacity, pooled_topk, group_size, device):
        if capacity <= 0 or pooled_topk <= 0 or group_size != 4:
            raise ValueError("invalid GLM53 MTP indexer seed geometry")
        self.pooled_topk = pooled_topk
        self.group_size = group_size
        self.indices = torch.zeros(
            (capacity, pooled_topk * group_size + group_size - 1),
            dtype=torch.int32,
            device=device,
        )

    def snapshot(self, batch_size):
        if batch_size < 0 or batch_size > self.indices.shape[0]:
            raise ValueError("MTP indexer seed batch exceeds fixed capacity")
        return self.indices[:batch_size]

    def load(self, indices):
        if (
            indices.ndim != 2
            or indices.dtype != torch.int32
            or indices.device != self.indices.device
            or indices.shape[1] != self.indices.shape[1]
        ):
            raise ValueError("invalid expanded raw-token MTP indexer seed")
        self.snapshot(indices.shape[0]).copy_(indices)

    def store_prefill_cp(self, pooled_indices, layout, local_ids, prefix_lengths):
        """Collect only each request's final query from padded CP shards."""
        batch_size = len(layout.q_lens)
        self.snapshot(batch_size)
        device = self.indices.device
        raw_lengths = prefix_lengths.to(
            device=device, dtype=torch.int32
        ) + torch.tensor(layout.q_lens, device=device, dtype=torch.int32)
        selected = None
        if pooled_indices is not None:
            if (
                pooled_indices.ndim != 2
                or pooled_indices.dtype != torch.int32
                or pooled_indices.device != device
                or pooled_indices.shape[0] != local_ids.numel()
                or pooled_indices.shape[1] not in (0, self.pooled_topk)
            ):
                raise ValueError("invalid CP-local pooled MTP indexer seed output")
            # The indexer omits padded CP queries. Restore their CP offsets,
            # then reduce B seed rows rather than gathering the whole prompt.
            local = torch.full(
                (layout.local_cp_tokens, pooled_indices.shape[1]),
                -1,
                dtype=torch.int32,
                device=device,
            )
            local.index_copy_(
                0, local_ids.to(device=device, dtype=torch.int64), pooled_indices
            )
            selected = torch.zeros(
                (batch_size, local.shape[1]), dtype=torch.int32, device=device
            )
            end = 0
            for request, length in enumerate(layout.q_lens):
                end += length
                for fragment in layout.fragments:
                    if (
                        fragment.cp_rank == layout.rank
                        and fragment.canonical_offset
                        <= end - 1
                        < fragment.canonical_offset + fragment.length
                    ):
                        row = fragment.cp_offset + end - 1 - fragment.canonical_offset
                        selected[request].copy_(local[row])
                        break
            selected = all_reduce(selected, Group.TP)
        rows = torch.arange(batch_size, dtype=torch.int32, device=device)
        self.store(selected, rows, raw_lengths - 1)

    def store(self, pooled_indices, seed_rows, positions):
        if (
            seed_rows.ndim != 1
            or seed_rows.dtype != torch.int32
            or seed_rows.device != self.indices.device
            or positions.ndim != 1
            or positions.dtype not in (torch.int32, torch.int64)
            or positions.device != self.indices.device
        ):
            raise ValueError("MTP seed requires device int32 accepted-row indices")
        destination = self.snapshot(seed_rows.numel())
        rows = seed_rows.to(torch.int64)
        raw_lengths = positions.index_select(0, rows) + 1
        if pooled_indices is None:
            # The short-context dense path selects every causal raw token.
            selected = torch.arange(
                self.pooled_topk, dtype=torch.int32, device=self.indices.device
            ).expand(seed_rows.numel(), -1)
        else:
            if (
                pooled_indices.ndim != 2
                or pooled_indices.dtype != torch.int32
                or pooled_indices.device != self.indices.device
                or pooled_indices.shape[0] != positions.numel()
                or pooled_indices.shape[1] not in (0, self.pooled_topk)
            ):
                raise ValueError(
                    "invalid pooled MTP indexer seed output: "
                    f"shape={tuple(pooled_indices.shape)}, dtype={pooled_indices.dtype}, "
                    f"positions={positions.numel()}, expected_topk={self.pooled_topk}"
                )
            selected = pad_empty_indexer_history(
                pooled_indices, self.pooled_topk
            ).index_select(0, rows)
        expanded = fused_expand_indexer_groups_with_tail(
            selected, raw_lengths, self.group_size
        )
        if expanded is None:
            expanded = append_incomplete_tail_indices(
                expand_indexer_group_indices(
                    selected,
                    self.group_size,
                    raw_sequence_lengths=raw_lengths,
                ),
                raw_lengths,
                self.group_size,
            )
        destination.copy_(expanded)
