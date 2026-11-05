"""PPU CED query layout; source KV remains in the original absolute domain.

The first version leaves the last global producer complete (L20 in Flash)
and compacts only its consumers. CP transport uses the existing equal-sized
all-gather: only selected contributions are padded/sent, never full hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
from rtp_llm.models_py.modules.dsv41.ppu.cp import CPPrefillContext, build_zigzag_plan


def permits_ced(inputs):
    return (
        getattr(inputs, "need_all_logits", None) is False
        and getattr(inputs, "need_all_hidden_states", None) is False
    )


def decoder_replay_starts(requests, lengths, prefixes, window):
    return {
        int(rid): int(prefix) + max(int(length) - int(window), 0)
        for rid, length, prefix in zip(requests, lengths, prefixes)
    }


@dataclass
class PPUCEDPlan:
    selected: torch.Tensor
    original_rows: int
    original_local_rows: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    request_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    context: CPPrefillContext | None
    starts: dict
    info: dict
    _original_context: CPPrefillContext | None = None
    _send_indices: torch.Tensor | None = None
    _receive_indices: torch.Tensor | None = None
    _local_indices: torch.Tensor | None = None
    _contribution_rows: int = 0

    @classmethod
    def create(
        cls, geometry, policy, input_ids, positions, request_ids, cu, cp_context
    ):
        start_layer = policy.decoder_start(geometry)
        if not policy.ced or start_layer >= geometry.num_layers:
            return None
        if any(layer >= start_layer for layer in geometry.engram_layer_ids):
            raise ValueError("PPU CED requires Engram layers before the query boundary")
        if cp_context is not None:
            full_ids = cp_context.plan.full_input_ids
            lengths = [split.length for split in cp_context.plan.requests]
            prefixes = [split.prefix for split in cp_context.plan.requests]
            requests = [split.request_id for split in cp_context.plan.requests]
        else:
            full_ids = input_ids
            boundaries = cu.cpu().tolist()
            lengths = [b - a for a, b in zip(boundaries, boundaries[1:])]
            requests = [int(request_ids[int(a)]) for a in boundaries[:-1]]
            prefixes = [int(positions[int(a)]) for a in boundaries[:-1]]
        # Exact CED retains a conservative dependency cone for every output
        # tail exported by NativeCacheBridge. Bounded mode deliberately starts
        # a fresh, approximate SWA domain at the selected tail.
        keep = (
            geometry.window
            if policy.bounded_replay
            else (geometry.num_layers - start_layer) * (geometry.window - 1)
            + geometry.window
        )
        kept = [min(length, keep) for length in lengths]
        selected, offset = [], 0
        for length, count in zip(lengths, kept):
            selected.extend(range(offset + length - count, offset + length))
            offset += length
        if len(selected) == offset:
            return None
        selected = torch.tensor(selected, dtype=torch.int64)
        absolute_starts = [
            prefix + length - count
            for prefix, length, count in zip(prefixes, lengths, kept)
        ]
        starts = dict(zip(requests, absolute_starts))
        compact_cu = torch.tensor([0] + list(torch.tensor(kept).cumsum(0).tolist()))
        compact_ids = full_ids.index_select(0, selected.to(full_ids.device))
        info = {
            "first_decoder_layer": start_layer,
            "bounded_replay": policy.bounded_replay,
            "original_rows": offset,
            "query_rows": len(selected),
            "kept_lengths": kept,
            "absolute_starts": absolute_starts,
            "cp_size": 1 if cp_context is None else cp_context.cp_size,
        }
        if cp_context is None:
            selected_device = selected.to(input_ids.device)
            return cls(
                selected,
                offset,
                input_ids.numel(),
                compact_ids,
                positions.index_select(0, selected_device),
                request_ids.index_select(0, selected_device),
                compact_cu.to(cu.device),
                None,
                starts,
                info,
            )
        original = cp_context.plan
        plan = build_zigzag_plan(
            compact_ids.cpu(),
            compact_cu,
            cp_size=original.cp_size,
            cp_rank=original.cp_rank,
            prefix_lengths=absolute_starts,
        )
        context = CPPrefillContext(plan, cp_context.group, cp_context.simulate_moe)
        rank_major = original.unpad_restore.cpu().index_select(0, selected)
        owners = rank_major // original.chunk_length
        send = rank_major[owners == original.cp_rank] % original.chunk_length
        counts = torch.bincount(owners, minlength=original.cp_size)
        max_count = int(counts.max())
        seen = [0] * original.cp_size
        receive = []
        for owner in owners.tolist():
            receive.append(owner * max_count + seen[owner])
            seen[owner] += 1
        local = torch.empty(plan.chunk_length, dtype=torch.int64)
        for row, split in enumerate(plan.requests):
            span = slice(split.local_offset, split.local_offset + split.chunk)
            local[span] = compact_cu[row] + plan.local_positions[span] - split.prefix
        return cls(
            selected,
            offset,
            original.chunk_length,
            plan.local_input_ids.to(input_ids.device),
            plan.local_positions.to(positions.device),
            plan.local_request_ids.to(request_ids.device),
            plan.cu_local.to(cu.device),
            context,
            starts,
            info,
            cp_context,
            send,
            torch.tensor(receive),
            local,
            max_count,
        )

    def compact(self, value):
        if value.shape[0] != self.original_local_rows:
            raise ValueError("PPU CED value does not use the original query layout")
        if self._original_context is None:
            return value.index_select(0, self.selected.to(value.device))
        send = value.index_select(0, self._send_indices.to(value.device))
        padded = value.new_zeros((self._contribution_rows, *value.shape[1:]))
        padded[: send.shape[0]].copy_(send)
        gathered = self._original_context.group.all_gather(padded)
        selected = gathered.index_select(0, self._receive_indices.to(value.device))
        return selected.index_select(0, self._local_indices.to(value.device))

    def compact_mix(self, mix):
        if mix is None:
            return None
        updates = {}
        for field in fields(mix):
            value = getattr(mix, field.name)
            if isinstance(value, torch.Tensor):
                updates[field.name] = self.compact(value)
        return replace(mix, **updates)

    def restore_global(self, value):
        if value.shape[0] != self.selected.numel():
            raise ValueError("PPU CED restored hidden has the wrong selected row count")
        restored = value.new_zeros((self.original_rows, *value.shape[1:]))
        return restored.index_copy_(0, self.selected.to(value.device), value)
