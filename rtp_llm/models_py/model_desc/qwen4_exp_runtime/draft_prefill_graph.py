"""Fixed-capacity MTP draft metadata and validation before graph replay."""

from collections.abc import Mapping
from threading import Lock

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


class _ReplayWorkspace:
    """One callback/stream's scratch, retained only after successful prepares."""

    def __init__(self, key, transported, cu, capacity, factor):
        self.key = key
        self.positions_host = torch.empty(
            capacity, dtype=transported.dtype, pin_memory=transported.is_cuda
        )
        self.cu_host = torch.empty(cu.shape, dtype=cu.dtype, pin_memory=cu.is_cuda)
        self.positions = (
            torch.empty(capacity, dtype=transported.dtype, device=transported.device)
            if transported.is_cuda
            else self.positions_host
        )
        self.cu = torch.empty_like(cu) if cu.is_cuda else self.cu_host
        self.canonical = torch.empty_like(transported).reshape(capacity, factor)
        self.state_inputs = {}


class DraftPrefillGraphReplay(Mapping):
    """Validate all tagged pools before the graph runner launches any writes.

    This intentionally is not a dict: the existing graph runner calls the
    model-wide callback with all cache groups, rather than each FMHA separately.
    """

    def __init__(
        self, impls, bounds, position_factor, token_capacity, state_cache=None
    ):
        self.impls = impls
        self.bounds = bounds
        self.position_factor = position_factor
        self.token_capacity = token_capacity
        self.state_cache = state_cache
        self._workspace = None
        self._workspace_lock = Lock()

    def __getitem__(self, key):
        return self.impls[key]

    def __iter__(self):
        return iter(self.impls)

    def __len__(self):
        return len(self.impls)

    def _validate_replay_layout(self, anchor):
        """Check bucket metadata before allocating scratch or writing graph inputs."""
        transported = anchor.combo_position_ids
        if not isinstance(transported, torch.Tensor) or transported.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise RuntimeError(
                "QSA draft graph requires int32 or int64 transported positions"
            )
        if transported.numel() != self.token_capacity * self.position_factor:
            raise RuntimeError("QSA draft graph transported position capacity changed")
        cu = getattr(anchor, "cu_seqlens_device", None)
        if (
            not isinstance(cu, torch.Tensor)
            or cu.dtype != torch.int32
            or tuple(cu.shape) != (anchor.input_lengths.numel() + 1,)
        ):
            raise RuntimeError(
                "QSA draft graph requires int32 device cu_seqlens matching the request capacity"
            )
        if transported.device != cu.device:
            raise RuntimeError(
                "QSA draft graph positions and cu_seqlens must use the same device"
            )
        return transported, cu

    def prepare_cuda_graph(self, groups):
        anchor = groups[next(iter(self.impls))]
        transported, cu = self._validate_replay_layout(anchor)
        if not self._workspace_lock.acquire(blocking=False):
            return self._prepare_cuda_graph(groups, transported, cu)
        try:
            stream = (
                torch.cuda.current_stream(transported.device)
                if transported.is_cuda
                else None
            )
            key = (
                self.token_capacity,
                self.position_factor,
                transported.dtype,
                transported.device,
                tuple(cu.shape),
                None if stream is None else stream.cuda_stream,
            )
            if self._workspace is None or self._workspace.key != key:
                self._workspace = _ReplayWorkspace(
                    key, transported, cu, self.token_capacity, self.position_factor
                )
            # The validation .item() completes all scratch reads on this stream
            # before a successful return, so the next prepare can reuse it.
            return self._prepare_cuda_graph(groups, transported, cu, self._workspace)
        except BaseException:
            # A failed prepare may leave H2D copies pending. Drop the cache
            # reference; PyTorch's allocators retain in-flight storage safely.
            self._workspace = None
            raise
        finally:
            self._workspace_lock.release()

    def _prepare_cuda_graph(self, groups, transported, cu, workspace=None):
        groups = prepare_state_cache_inputs(
            groups,
            self.state_cache,
            workspace=None if workspace is None else workspace.state_inputs,
        )
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
                    end = ((prefix + length) // 4 + page_size - 1) // page_size
                else:
                    end = (prefix + length + page_size - 1) // page_size
                if end > table.shape[1]:
                    raise RuntimeError(
                        "QSA draft graph block table does not cover the visible tail"
                    )
                # Check this replay's visible pages in native array code. The
                # graph-owned table may change in place between prepares.
                visible = values[row, :end]
                if end <= 64:
                    # Small rows avoid allocating NumPy comparison masks.
                    pages = visible.tolist()
                    invalid = bool(pages) and (
                        min(pages) <= 0 or max(pages) >= pool_size
                    )
                else:
                    invalid = ((visible <= 0) | (visible >= pool_size)).any()
                if invalid:
                    raise RuntimeError(
                        "QSA draft graph live row resolves to an unallocated or out-of-range physical block"
                    )
        # The draft uses logical Base-RoPE coordinates. The target transports
        # MRoPE coordinates, which differ after an image or timestamped frame;
        # normalize this graph-owned buffer just as eager draft prefill does.
        if workspace is None:
            expected_positions = torch.tensor(
                positions,
                dtype=transported.dtype,
                device="cpu",
                pin_memory=transported.is_cuda,
            ).to(device=transported.device, non_blocking=True)
            canonical = torch.zeros_like(transported).reshape(
                self.token_capacity, self.position_factor
            )
        else:
            host = workspace.positions_host[: len(positions)]
            host.numpy()[:] = positions
            expected_positions = workspace.positions[: len(positions)]
            if transported.is_cuda:
                expected_positions.copy_(host, non_blocking=True)
            canonical = workspace.canonical.zero_()
        canonical[: len(positions)] = expected_positions[:, None]
        copied = set()
        for value in groups.values():
            target = value.combo_position_ids
            if target.data_ptr() not in copied:
                target.copy_(canonical.reshape_as(target))
                copied.add(target.data_ptr())
        if state_geometry is None:
            raise RuntimeError("QSA draft graph requires a raw-state cache group")
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
                transported,
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
                torch.stack((first, last), dim=1).clamp(0, table.shape[1] - 1).long()
            )
            physical = table.gather(1, columns)
            invalid_state = (
                invalid.any()
                | (
                    active.unsqueeze(1) & ((physical <= 0) | (physical >= pool_size))
                ).any()
            )
            actual = transported.reshape(self.token_capacity, self.position_factor)[
                : len(positions)
            ]
            invalid_positions = (actual != expected_positions.unsqueeze(1)).any()
            invalid_replay = invalid_state | invalid_positions
        if workspace is None:
            expected_cu = torch.tensor(
                expected, device="cpu", dtype=cu.dtype, pin_memory=cu.is_cuda
            ).to(device=cu.device, non_blocking=True)
        else:
            workspace.cu_host.numpy()[:] = expected
            expected_cu = workspace.cu
            if cu.is_cuda:
                expected_cu.copy_(workspace.cu_host, non_blocking=True)
        invalid_replay = invalid_replay | torch.any(cu != expected_cu)
        if bool(invalid_replay.item()):
            raise RuntimeError(
                "QSA draft graph has invalid raw-state pages, cu_seqlens or positions disagree with logical cache positions"
            )
        for tag, impl in self.impls.items():
            impl.prepare_cuda_graph(groups[tag])
