"""Exact MRoPE coordinates beside the QSA raw-key ring and page checkpoints."""

from dataclasses import dataclass, field

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import INDEXER_ROPE_POS_TAG
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    build_qsa_rope,
    is_qsa_rope_style,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.validation import DeviceChecks

from .qsa_cache_views import _block_table, _typed_2d_pool
from .step_metadata import rope_config_signature


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
    groups_all_current: bool | None = None
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
        if (
            self.is_graph
            and ratio == 4
            and self.pool.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(device) == (12, 0)
        ):
            from rtp_llm.models_py.kernels.cuda.qsa.position_state import (
                position_geometry,
            )

            packed = position_geometry(
                positions, prefixes, cu, table, self.pool.shape[0], self.page_size
            )
            if packed is not None:
                requests, logical, ends, valid, slots, undo, lookup = packed
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
                    None,
                    group_lookup=lookup,
                    undo_slots=undo,
                )
                if metadata is not None:
                    metadata[key] = self.geometry
                return
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

    def _group_lookup(self):
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
        return g.group_lookup

    def group_positions(self):
        slots, in_current, current = self._group_lookup()
        # Only the lookup is common. Historical coordinates remain layer-local.
        saved = self.pool.flatten(0, 1)[slots]
        return torch.where(in_current[:, None], current, saved)

    def prefix_reuse_group_rope(self, config, checks):
        g = self.geometry
        if g.groups_all_current is None:
            # Use the same device geometry as the sidecar reads. Carry the
            # decision with the existing pre-write verdict on the first layer.
            _, in_current, _ = self._group_lookup()
            (all_current,) = checks.verify(payload=in_current.all().reshape(1))
            g.groups_all_current = bool(all_current)
        return self.rope(config, group=True)

    def group_rope_inputs(self, config, dtype=torch.bfloat16, *, frequency=None):
        g = self.geometry
        if (
            self.is_graph
            and self.pool.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(self.pool.device) == (12, 0)
            and dtype == torch.bfloat16
            and g.positions.shape[0] > 0
            and self.pool.is_contiguous()
            and is_qsa_rope_style(config, "Mrope")
            and config.dim == 64
            and config.index_factor == 3
            and config.mrope_interleaved
            and getattr(config, "indexer_is_neox_style", True)
            and float(getattr(config, "scale", 1.0)) == 1.0
            and config.mrope_dim1 >= 0
            and 0 <= config.mrope_dim2 <= 11
            and 0 <= config.mrope_dim3 <= 10
            and config.mrope_dim1 + config.mrope_dim2 + config.mrope_dim3 == 32
        ):
            # Reuse the exact FP32 frequencies within this forward. Each layer
            # still reads its own historical position ring before any mutation.
            if frequency is None:
                frequency_key = ("group_rope_frequency", config.base)
                frequency = g.rope_cache.get(frequency_key)
                if frequency is None:
                    pairs = torch.arange(
                        32, dtype=torch.float32, device=self.pool.device
                    )
                    frequency = float(config.base) ** (-2.0 * pairs / 64)
                    g.rope_cache[frequency_key] = frequency
            return (
                self.pool,
                *self._group_lookup(),
                frequency,
                config.mrope_dim2,
                config.mrope_dim3,
            )
        return None

    def target_query_rope(self, config, frequency):
        from rtp_llm.models_py.kernels.cuda.qsa.position_state import (
            query_position_rope,
        )

        g = self.geometry
        key = ("target_query_rope", config.base, config.mrope_dim2, config.mrope_dim3)
        if key not in g.rope_cache:
            g.rope_cache[key] = query_position_rope(
                g.positions, frequency, config.mrope_dim2, config.mrope_dim3
            )
        return g.rope_cache[key]

    def rope(self, config, *, group=False, dtype=torch.bfloat16):
        g = self.geometry
        share = not group or g.groups_all_current is True
        key = (group, rope_config_signature(config), dtype, self.pool.device)
        if share and key in g.rope_cache:
            return g.rope_cache[key]
        if group:
            inputs = self.group_rope_inputs(config, dtype)
            if inputs is not None:
                from rtp_llm.models_py.kernels.cuda.qsa.position_state import (
                    group_position_rope,
                )

                result = group_position_rope(*inputs)
                if share:
                    g.rope_cache[key] = result
                return result
        if group and share:
            positions = self._group_lookup()[2]
        else:
            positions = self.group_positions() if group else g.positions
        result = build_qsa_rope(
            positions.reshape(-1),
            config,
            token_count=positions.shape[0],
            dtype=dtype,
            device=self.pool.device,
        )
        if share:
            g.rope_cache[key] = result
        return result

    def staging_slots(self):
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
        return g.undo_slots

    def record_stage(self, slots, original):
        self.undo_slots = slots
        self.original = original
        self.staged = True

    def stage(self):
        if self.staged:
            return
        self.undo_slots = self.staging_slots()
        self.original = self.pool.flatten(0, 1)[self.geometry.slots]
        self._store(self.geometry.positions)
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
