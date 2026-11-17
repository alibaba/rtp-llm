"""Request-scoped GLM53 MTP indexer seeds, in expanded raw-token coordinates."""

import os

import torch

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
        os.environ.get("RTP_LLM_ENABLE_MTP_INDEXER_SHARE") == "1"
        and bool(config.index_share_for_mtp_iteration)
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
                raise ValueError("invalid pooled MTP indexer seed output")
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
