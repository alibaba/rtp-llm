"""Fixed-capacity MTP draft metadata and validation before graph replay."""

import os
from collections.abc import Mapping

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import INDEXER_STATE_TAG

from .state_cache_inputs import prepare_state_cache_inputs


def draft_graph_lanes(inputs, token_capacity):
    batch = inputs.input_lengths_device.numel()
    if not batch or token_capacity % batch or not 1 <= token_capacity // batch <= 4:
        raise RuntimeError(
            "QSA draft graph requires one-to-four-token request capacity"
        )
    return token_capacity // batch


def draft_graph_rows(inputs, token_capacity):
    """Map packed live tokens to fixed request lanes using device metadata."""
    lengths = inputs.input_lengths_device
    prefixes = inputs.prefix_lengths_device
    cu = inputs.cu_seqlens_device
    batch = lengths.numel()
    lanes = draft_graph_lanes(inputs, token_capacity)
    if lengths.is_cuda:
        from rtp_llm.models_py.kernels.cuda.qsa.draft_metadata_triton import draft_rows

        fused = draft_rows(lengths, prefixes, cu, token_capacity)
        if fused is not None:
            return fused
    offsets = torch.arange(lanes, device=cu.device, dtype=torch.int64)
    active = offsets.unsqueeze(0) < lengths.unsqueeze(1)
    sources = cu[:-1].long().unsqueeze(1) + offsets.unsqueeze(0)
    sources = torch.where(active, sources, 0).reshape(-1)
    rows = torch.arange(token_capacity, device=cu.device, dtype=torch.int64)
    request = (rows.unsqueeze(1) >= cu[1:].unsqueeze(0)).sum(1).clamp(max=batch - 1)
    local = rows - cu.long().index_select(0, request)
    packed_valid = rows < cu[-1]
    inverse = torch.where(packed_valid, request * lanes + local, 0)
    positions = prefixes.long().index_select(0, request) + local
    positions = torch.where(packed_valid, positions, 0)
    visible = prefixes.unsqueeze(1) + offsets.unsqueeze(0) + 1
    visible = torch.where(active, visible, 0).to(torch.int32)
    return sources, inverse, packed_valid, positions, visible


def _canonical_draft_positions(
    source_lengths,
    source_prefixes,
    transported,
    *,
    token_capacity,
    position_factor,
):
    """Build draft Base-RoPE coordinates with no host read or validation H2D.

    Only source snapshots choose row ownership. Invalid source values still
    produce bounded gathers; the independent verdict rejects them before replay.
    """
    tensors = (source_lengths, source_prefixes, transported)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        return None
    if not all(
        value.device == source_lengths.device
        and value.dtype == torch.int32
        and value.is_contiguous()
        for value in tensors
    ):
        return None
    batch = source_lengths.numel()
    if (
        source_lengths.ndim != 1
        or not 1 <= batch <= 32
        or source_prefixes.shape != (batch,)
        or token_capacity <= 0
        or token_capacity % batch
        or not 1 <= token_capacity // batch <= 4
        or position_factor not in (1, 3)
        or transported.numel() != token_capacity * position_factor
    ):
        return None
    lengths = source_lengths.to(dtype=torch.int64)
    prefixes = source_prefixes.to(dtype=torch.int64)
    cu = torch.cat((torch.zeros_like(lengths[:1]), lengths.cumsum(0)))
    rows = torch.arange(token_capacity, device=lengths.device, dtype=torch.int64)
    owners = (rows[:, None] >= cu[None, 1:]).sum(1).clamp(0, batch - 1)
    positions = prefixes.index_select(0, owners) + rows - cu.index_select(0, owners)
    positions = torch.where(rows < cu[-1], positions, 0).to(dtype=transported.dtype)
    return positions[:, None].expand(-1, position_factor).reshape_as(transported)


class DraftPrefillGraphReplay(Mapping):
    """Validate all tagged pools before the graph runner launches any writes.

    This intentionally is not a dict: the existing graph runner calls the
    model-wide callback with all cache groups, rather than each FMHA separately.
    """

    cuda_graph_metadata_lease = True
    cuda_graph_device_verdict = True

    def __init__(
        self, impls, bounds, position_factor, token_capacity, state_cache=None
    ):
        self.impls = impls
        self.bounds = bounds
        self.position_factor = position_factor
        self.token_capacity = token_capacity
        self.state_cache = state_cache
        self._device_verdict_kernel = (
            os.environ.get("RTP_LLM_CUDA_GRAPH_DEVICE_VERDICT_KERNEL", "0") == "1"
        )

    def __getitem__(self, key):
        return self.impls[key]

    def __iter__(self):
        return iter(self.impls)

    def __len__(self):
        return len(self.impls)

    def enqueue_cuda_graph_validation(self, groups, source_lengths, source_prefixes):
        if not self._device_verdict_kernel:
            return None
        from rtp_llm.models_py.kernels.cuda.qsa.draft_metadata_triton import (
            replay_invalid_from_sources,
        )

        groups = prepare_state_cache_inputs(groups, self.state_cache)
        anchor = groups[next(iter(self.impls))]
        canonical = _canonical_draft_positions(
            source_lengths,
            source_prefixes,
            anchor.combo_position_ids,
            token_capacity=self.token_capacity,
            position_factor=self.position_factor,
        )
        if canonical is None:
            return None
        copied = set()
        for inputs in groups.values():
            target = inputs.combo_position_ids
            if target.data_ptr() not in copied:
                target.copy_(canonical)
                copied.add(target.data_ptr())
        for tag, pool_size, page_size, _ in self.bounds:
            if tag == INDEXER_STATE_TAG:
                return replay_invalid_from_sources(
                    source_lengths,
                    source_prefixes,
                    anchor.input_lengths_device,
                    anchor.prefix_lengths_device,
                    groups[tag].kv_cache_block_id_device,
                    anchor.combo_position_ids,
                    anchor.cu_seqlens_device,
                    pool_size=pool_size,
                    page_size=page_size,
                    token_capacity=self.token_capacity,
                    position_factor=self.position_factor,
                )
        return None

    def prepare_cuda_graph(self, groups, *, metadata_lease=None):
        original_groups = groups
        if metadata_lease is not None:
            from rtp_llm.ops.compute_ops import GraphReplayMetadataLease

            if type(
                metadata_lease
            ) is not GraphReplayMetadataLease or not metadata_lease.matches(groups):
                raise RuntimeError("QSA draft graph has a stale metadata lease")
        groups = prepare_state_cache_inputs(groups, self.state_cache)
        anchor = groups[next(iter(self.impls))]
        lengths = anchor.input_lengths
        prefixes = anchor.prefix_lengths
        if any(value.device.type != "cpu" for value in (lengths, prefixes)):
            raise RuntimeError("QSA draft graph replay requires host length mirrors")
        lengths = lengths.tolist()
        prefixes = prefixes.tolist()
        if not lengths or len(prefixes) != len(lengths):
            raise RuntimeError("QSA draft graph replay has invalid length geometry")
        expected = [0]
        lanes = draft_graph_lanes(anchor, self.token_capacity)
        padding_started = False
        positions = []
        for length, prefix in zip(lengths, prefixes):
            if length == 0:
                padding_started = True
            elif padding_started or not 1 <= length <= lanes or prefix <= 0:
                raise RuntimeError(
                    "QSA draft graph requires positive live prefixes and lengths within request capacity followed by padding"
                )
            expected.append(expected[-1] + length)
            positions.extend(range(prefix, prefix + length))
        # Derive the host partition from the same current length mirror; a
        # device-only CU tensor must not add a public GraphRunner D2H copy.
        if expected[-1] > self.token_capacity:
            raise RuntimeError(
                "QSA draft graph cu_seqlens do not partition the live token capacity"
            )
        state_geometry = None
        for tag, pool_size, page_size, compressed in self.bounds:
            inputs = groups[tag]
            if not torch.equal(
                inputs.input_lengths, anchor.input_lengths
            ) or not torch.equal(inputs.prefix_lengths, anchor.prefix_lengths):
                raise RuntimeError("QSA draft graph cache groups disagree on lengths")
            if tag == INDEXER_STATE_TAG:
                # The model's state view derives physical pages from the
                # refreshed kernel table; do not use a stale host mirror.
                state_geometry = (inputs.kv_cache_block_id_device, pool_size, page_size)
                continue
            table = inputs.kv_cache_kernel_block_id
            if (
                table.device.type != "cpu"
                or table.dtype != torch.int32
                or table.ndim != 2
                or table.shape[0] != len(lengths)
            ):
                raise RuntimeError(
                    "QSA draft graph replay requires host int32 block tables"
                )
            values = table.numpy()
            for row, (length, prefix) in enumerate(zip(lengths, prefixes)):
                if not length:
                    continue
                if compressed:
                    first = 0
                    end = ((prefix + length) // 4 + page_size - 1) // page_size
                else:
                    first = 0
                    end = (prefix + length + page_size - 1) // page_size
                if end > table.shape[1]:
                    raise RuntimeError(
                        "QSA draft graph block table does not cover the visible tail"
                    )
                if any(
                    not 0 < int(block) < pool_size for block in values[row, first:end]
                ):
                    raise RuntimeError(
                        "QSA draft graph live row resolves to an unallocated or out-of-range physical block"
                    )
        # The draft uses logical Base-RoPE coordinates. The target transports
        # MRoPE coordinates, which differ after an image or timestamped frame;
        # normalize this graph-owned buffer just as eager draft prefill does.
        transported = anchor.combo_position_ids
        if (
            transported is None
            or transported.numel() != self.token_capacity * self.position_factor
        ):
            raise RuntimeError("QSA draft graph transported position capacity changed")
        if state_geometry is None:
            raise RuntimeError("QSA draft graph requires a raw-state cache group")
        cu = getattr(anchor, "cu_seqlens_device", None)
        if (
            not isinstance(cu, torch.Tensor)
            or cu.dtype != torch.int32
            or tuple(cu.shape) != (len(expected),)
        ):
            raise RuntimeError(
                "QSA draft graph requires int32 device cu_seqlens matching the request capacity"
            )
        device_invalid = (
            metadata_lease.device_validation_invalid
            if metadata_lease is not None
            else None
        )
        if device_invalid is None:
            expected_positions = torch.tensor(
                positions,
                dtype=transported.dtype,
                device="cpu",
                pin_memory=transported.is_cuda,
            ).to(device=transported.device, non_blocking=True)
            canonical = torch.zeros_like(transported).reshape(
                self.token_capacity, self.position_factor
            )
            canonical[: len(positions)] = expected_positions[:, None]
            table, pool_size, page_size = state_geometry
            length_device = anchor.input_lengths_device
            prefix_device = anchor.prefix_lengths_device
            invalid_replay = None
            if length_device.is_cuda:
                from rtp_llm.models_py.kernels.cuda.qsa.draft_metadata_triton import (
                    replay_invalid,
                )

                invalid_replay = replay_invalid(
                    length_device,
                    prefix_device,
                    table,
                    canonical.reshape_as(transported),
                    expected_positions,
                    pool_size=pool_size,
                    page_size=page_size,
                    token_capacity=self.token_capacity,
                    position_factor=self.position_factor,
                )
            if invalid_replay is None:
                first = (prefix_device - prefix_device.remainder(4)) // page_size
                last = (prefix_device + length_device - 1) // page_size
                active = length_device > 0
                invalid = active & ((first < 0) | (last >= table.shape[1]))
                columns = (
                    torch.stack((first, last), dim=1)
                    .clamp(0, table.shape[1] - 1)
                    .long()
                )
                physical = table.gather(1, columns)
                invalid_state = (
                    invalid.any()
                    | (
                        active.unsqueeze(1)
                        & ((physical <= 0) | (physical >= pool_size))
                    ).any()
                )
                actual = canonical.reshape(self.token_capacity, self.position_factor)[
                    : len(positions)
                ]
                invalid_positions = (actual != expected_positions.unsqueeze(1)).any()
                invalid_replay = invalid_state | invalid_positions
            expected_cu = torch.tensor(
                expected, device="cpu", dtype=cu.dtype, pin_memory=cu.is_cuda
            )
            expected_cu = expected_cu.to(device=cu.device, non_blocking=True)
            invalid_replay = invalid_replay | torch.any(cu != expected_cu)
            if bool(invalid_replay.item()):
                raise RuntimeError(
                    "QSA draft graph has invalid raw-state pages, cu_seqlens or positions disagree with logical cache positions"
                )
        elif device_invalid:
            raise RuntimeError(
                "QSA draft graph has invalid raw-state pages, cu_seqlens or positions disagree with logical cache positions"
            )
        if metadata_lease is not None and not metadata_lease.consume(original_groups):
            raise RuntimeError("QSA draft graph metadata lease changed before delegate")
        if device_invalid is None:
            copied = set()
            for value in groups.values():
                target = value.combo_position_ids
                if target.data_ptr() not in copied:
                    target.copy_(canonical.reshape_as(target))
                    copied.add(target.data_ptr())
        for tag, impl in self.impls.items():
            impl.prepare_cuda_graph(groups[tag])


_DEVICE_VERDICT_PREPARE = DraftPrefillGraphReplay.prepare_cuda_graph
_DEVICE_VERDICT_ENQUEUE = DraftPrefillGraphReplay.enqueue_cuda_graph_validation
