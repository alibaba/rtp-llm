"""Exact MRoPE coordinates beside the QSA raw-key ring and page checkpoints."""

from dataclasses import dataclass, field

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import INDEXER_ROPE_POS_TAG
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import build_qsa_rope
from rtp_llm.models_py.modules.qwen4_exp.qsa.validation import DeviceChecks

from .qsa_cache_views import _block_table, _typed_2d_pool


def _tensor_key(tensor):
    if tensor is None:
        return None
    return (
        tensor.data_ptr(),
        tensor.shape,
        tensor.stride(),
        tensor.dtype,
        tensor.device,
    )


@dataclass
class _PositionGeometry:
    positions: torch.Tensor
    table: torch.Tensor
    requests: torch.Tensor
    logical: torch.Tensor
    ends: torch.Tensor
    cu: torch.Tensor
    prefixes: torch.Tensor
    valid: torch.Tensor
    slots: torch.Tensor
    invalid: torch.Tensor | None
    checked_by: DeviceChecks | None = None
    group_lookup: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    undo_slots: torch.Tensor | None = None
    rope_cache: dict = field(default_factory=dict)


class QSAPositionState:
    def __init__(self, context, ratio):
        inputs = context.main_inputs
        cache = context.position_state_cache
        self.page_size = int(cache.seq_size_per_block)
        self.ring_size = 2 * ratio
        self.ratio = ratio
        self.is_graph = bool(inputs.is_cuda_graph)
        self.pool = _typed_2d_pool(
            cache,
            tag=INDEXER_ROPE_POS_TAG,
            storage_dtype=torch.int32,
            payload_dtype=torch.int32,
            entries=self.ring_size,
            width=3,
        )
        device = self.pool.device
        prefill = inputs.is_prefill or inputs.is_target_verify
        if prefill:
            prefix_source = inputs.prefix_lengths_device
            if prefix_source.numel() == 0:
                prefix_source = inputs.prefix_lengths
            cu_source = inputs.cu_seqlens_device
        else:
            prefix_source = (
                inputs.sequence_lengths_plus_1_device
                if self.is_graph
                else inputs.sequence_lengths
            )
            cu_source = None
        batch = prefix_source.numel()
        if not batch:
            raise ValueError("QSA MRoPE state requires a nonempty batch")
        table = _block_table(
            context.position_state_inputs,
            "kv_cache_block_id_device",
            tag=INDEXER_ROPE_POS_TAG,
            batch_size=batch,
            device=device,
        )
        # Native metadata stays immutable for one forward. Share its geometry
        # and validation receipt across layers, never their pools or undo state.
        metadata = getattr(context, "metadata_cache", None)
        key = (
            "qsa_position_geometry",
            self.page_size,
            ratio,
            self.pool.shape[0],
            self.is_graph,
            prefill,
            _tensor_key(inputs.combo_position_ids),
            _tensor_key(prefix_source),
            _tensor_key(cu_source),
            _tensor_key(table),
        )
        self.geometry = metadata.get(key) if metadata is not None else None
        self.staged = False
        if self.geometry is not None:
            return
        positions = inputs.combo_position_ids.to(
            device=device, dtype=torch.int32, non_blocking=True
        ).reshape(-1, 3)
        count = positions.shape[0]
        prefixes = prefix_source.to(device=device, dtype=torch.long, non_blocking=True)
        if not prefill and self.is_graph:
            prefixes = prefixes - 1
        cu = (
            cu_source.to(device=device, dtype=torch.long, non_blocking=True)
            if prefill
            else torch.arange(batch + 1, device=device, dtype=torch.long)
        )
        rows = torch.arange(count, device=device, dtype=torch.long)
        requests = torch.bucketize(rows, cu[1:], right=True).clamp_max(batch - 1)
        logical = prefixes[requests] + rows - cu[requests]
        ends = prefixes[requests] + (cu[1:] - cu[:-1])[requests] - 1
        pages = torch.div(logical.clamp_min(0), self.page_size, rounding_mode="floor")
        blocks = table[requests, pages.clamp(0, table.shape[1] - 1)].long()
        valid = (
            (rows < cu[-1])
            & (logical >= 0)
            & (pages < table.shape[1])
            & (blocks > 0)
            & (blocks < self.pool.shape[0])
        )
        slots = blocks.clamp(
            0, self.pool.shape[0] - 1
        ) * self.ring_size + logical.remainder(self.ring_size)
        invalid = None
        if not self.is_graph:
            active = table.ne(0).any(dim=1)[requests]
            invalid = (active & (rows < cu[-1]) & ~valid).any() | (positions < 0).any()
        self.geometry = _PositionGeometry(
            positions,
            table,
            requests,
            logical,
            ends,
            cu,
            prefixes,
            valid,
            slots,
            invalid,
        )
        if metadata is not None:
            metadata[key] = self.geometry

    def add_checks(self, checks):
        """Append to the caller's existing pre-write acceptance boundary."""
        geometry = self.geometry
        if self.is_graph or (
            geometry.checked_by is not None
            and (geometry.checked_by.verified or geometry.checked_by is checks)
        ):
            return False
        checks.add(
            geometry.invalid,
            "QSA MRoPE state has invalid coordinates or physical pages",
            ValueError,
        )
        geometry.checked_by = checks
        return True

    def group_positions(self):
        g = self.geometry
        if g.group_lookup is None:
            starts = g.logical - g.logical.remainder(self.ratio)
            source = g.cu[g.requests] + starts - g.prefixes[g.requests]
            in_current = starts >= g.prefixes[g.requests]
            pages = torch.div(
                starts.clamp_min(0), self.page_size, rounding_mode="floor"
            )
            blocks = g.table[g.requests, pages.clamp(0, g.table.shape[1] - 1)].long()
            slots = blocks.clamp(
                0, self.pool.shape[0] - 1
            ) * self.ring_size + starts.remainder(self.ring_size)
            current = g.positions[source.clamp(0, g.positions.shape[0] - 1)]
            g.group_lookup = slots, in_current, current
        slots, in_current, current = g.group_lookup
        # Only the lookup is common. Historical coordinates remain layer-local.
        saved = self.pool.flatten(0, 1)[slots]
        return torch.where(in_current[:, None], current, saved)

    def rope(self, config, *, group=False, dtype=torch.bfloat16):
        g = self.geometry
        key = (
            str(config.style),
            config.dim,
            config.base,
            getattr(config, "scale", 1.0),
            config.index_factor,
            getattr(config, "mrope_dim1", 0),
            getattr(config, "mrope_dim2", 0),
            getattr(config, "mrope_dim3", 0),
            getattr(config, "mrope_interleaved", False),
            dtype,
        )
        if not group and key in g.rope_cache:
            return g.rope_cache[key]
        positions = self.group_positions() if group else g.positions
        result = build_qsa_rope(
            positions.reshape(-1),
            config,
            token_count=positions.shape[0],
            dtype=dtype,
            device=self.pool.device,
        )
        if not group:
            g.rope_cache[key] = result
        return result

    def stage(self):
        if self.staged:
            return
        g = self.geometry
        if not self.is_graph and (g.checked_by is None or not g.checked_by.verified):
            raise RuntimeError("QSA coordinate state must pass pre-write checks")
        if g.undo_slots is None:
            page_ends = torch.minimum(
                (torch.div(g.logical, self.page_size, rounding_mode="floor") + 1)
                * self.page_size
                - 1,
                g.ends,
            )
            write = g.valid & (g.logical + self.ring_size > page_ends)
            g.undo_slots = torch.where(write, g.slots, -1)
        self.undo_slots = g.undo_slots
        # Advanced indexing already returns independent storage for undo.
        self.original = self.pool.flatten(0, 1)[g.slots]
        self._store(g.positions)
        self.staged = True

    def _store(self, values):
        if self.pool.is_cuda:
            from rtp_llm.models_py.kernels.cuda.qsa.position_state import (
                store_positions,
            )

            store_positions(self.pool, self.undo_slots, values)
        else:
            valid = self.undo_slots >= 0
            self.pool.flatten(0, 1).index_copy_(
                0, self.undo_slots[valid], values[valid]
            )

    def rollback(self):
        if self.staged:
            self._store(self.original)
            self.staged = False
