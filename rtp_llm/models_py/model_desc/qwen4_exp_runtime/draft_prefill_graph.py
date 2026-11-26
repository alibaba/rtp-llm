"""Fixed-capacity MTP draft metadata and validation before graph replay."""

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


def _prepare_replay_metadata_torch(
    lengths,
    prefixes,
    cu,
    table,
    transported,
    reference,
    *,
    pool_size,
    page_size,
    token_capacity,
    position_factor,
):
    """Prepare unsupported layouts from this replay's validated host mirrors."""
    host_lengths, host_prefixes = reference.unbind()
    expected_cu = torch.nn.functional.pad(host_lengths.cumsum(0), (1, 0))
    active = host_lengths > 0
    invalid = (
        (lengths != host_lengths).any()
        | (active & (prefixes != host_prefixes)).any()
        | (cu != expected_cu).any()
    )

    # A draft can need three preceding raw keys and up to four current keys.
    # A small or non-power-of-two page can put interior columns in that window.
    lanes = token_capacity // lengths.numel()
    first = (host_prefixes - host_prefixes.remainder(4)) // page_size
    last = (host_prefixes + (host_lengths - 1)) // page_size
    max_page_delta = (lanes + 2 + page_size - 1) // page_size
    columns = first[:, None] + torch.arange(max_page_delta + 1, device=cu.device)
    required = active[:, None] & (columns <= last[:, None])
    physical = table.gather(1, columns.clamp(0, table.shape[1] - 1))
    invalid |= (
        required
        & (
            (columns < 0)
            | (columns >= table.shape[1])
            | (physical <= 0)
            | (physical >= pool_size)
        )
    ).any()

    rows = torch.arange(token_capacity, device=cu.device)
    owners = torch.searchsorted(expected_cu[1:], rows, right=True).clamp_max(
        lengths.numel() - 1
    )
    positions = host_prefixes[owners] + (rows - expected_cu[owners])
    live = rows < expected_cu[-1]
    overflow = live & (
        (positions < 0) | (positions > torch.iinfo(transported.dtype).max)
    )
    positions = torch.where(live & ~overflow, positions, 0)
    transported.copy_(
        positions[:, None].expand(-1, position_factor).reshape_as(transported)
    )
    return invalid | overflow.any()


class _ReplayWorkspace:
    """One callback/stream's scratch, retained only after successful prepares."""

    def __init__(self, key):
        self.key = key
        self.reference_host = None
        self.reference = None
        self.invalid = None
        self.state_inputs = {}

    def prepare_positions(
        self,
        anchor,
        lengths,
        prefixes,
        cu,
        transported,
        state_geometry,
        capacity,
        factor,
    ):
        device = transported.device
        if self.reference_host is None:
            self.reference_host = torch.empty(
                (2, len(lengths)),
                dtype=torch.int64,
                device="cpu",
                pin_memory=transported.is_cuda,
            )
            self.reference = (
                torch.empty_like(self.reference_host, device=device)
                if transported.is_cuda
                else self.reference_host
            )
        host = self.reference_host.numpy()
        host[0] = lengths
        host[1] = prefixes
        if transported.is_cuda:
            self.reference.copy_(self.reference_host, non_blocking=True)

        table, pool_size, page_size = state_geometry
        tensors = (
            anchor.input_lengths_device,
            anchor.prefix_lengths_device,
            cu,
            table,
            transported,
        )
        layout = dict(
            pool_size=pool_size,
            page_size=page_size,
            token_capacity=capacity,
            position_factor=factor,
        )
        if transported.is_cuda:
            from rtp_llm.models_py.kernels.cuda.qsa.draft_metadata_triton import (
                prepare_replay_metadata,
            )

            verdict = prepare_replay_metadata(
                *tensors, self.reference, self.invalid, **layout
            )
            if verdict is not None:
                self.invalid = verdict
                return verdict
        return _prepare_replay_metadata_torch(*tensors, self.reference, **layout)


class DraftPrefillGraphReplay(Mapping):
    """Validate all tagged pools before the graph runner launches any writes.

    This intentionally is not a dict: the existing graph runner calls the
    model-wide callback with all cache groups, rather than each FMHA separately.
    The graph runner owns the fixed int32 position and CU buffers. Each replay
    refreshes their contents and shares request lengths and prefixes across
    tags. The caller supplies positive raw-state page sizes and a nonempty
    physical state table. The caller also serializes prepares for each
    captured bucket before reusing its graph buffers and workspace.
    """

    def __init__(
        self,
        impls,
        bounds,
        position_factor,
        token_capacity,
        state_cache=None,
        *,
        capture_groups,
    ):
        anchor = capture_groups[next(iter(impls))]
        positions = anchor.combo_position_ids
        expected_positions = token_capacity * position_factor
        if positions is None or positions.numel() != expected_positions:
            raise RuntimeError(
                "QSA draft graph capture requires transported position capacity "
                f"{expected_positions}"
            )
        cu = anchor.cu_seqlens_device
        expected_cu_shape = (anchor.input_lengths.numel() + 1,)
        if cu is None or cu.dtype != torch.int32 or cu.shape != expected_cu_shape:
            raise RuntimeError(
                f"QSA draft graph capture requires int32 device cu_seqlens {expected_cu_shape}"
            )

        self.impls = impls
        self.bounds = []
        state_bound = None
        for tag, pool_size, page_size, compressed in bounds:
            if tag == INDEXER_STATE_TAG:
                state_bound = (pool_size, page_size)
            else:
                self.bounds.append((tag, pool_size, page_size, compressed))
        if state_bound is None or INDEXER_STATE_TAG not in capture_groups:
            raise RuntimeError(
                "QSA draft graph capture requires a raw-state cache group"
            )
        self.state_bound = state_bound
        self.position_factor = position_factor
        self.token_capacity = token_capacity
        self.state_cache = state_cache
        self._workspace = None

    def __getitem__(self, key):
        return self.impls[key]

    def __iter__(self):
        return iter(self.impls)

    def __len__(self):
        return len(self.impls)

    def prepare_cuda_graph(self, groups):
        anchor = groups[next(iter(self.impls))]
        transported = anchor.combo_position_ids
        cu = anchor.cu_seqlens_device
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
                self._workspace = _ReplayWorkspace(key)
            # The validation .item() completes all scratch reads on this stream
            # before a successful return, so the next prepare can reuse it.
            return self._prepare_cuda_graph(groups, transported, cu, self._workspace)
        except BaseException:
            # A failed prepare may leave H2D copies pending. Drop the cache
            # reference; PyTorch's allocators retain in-flight storage safely.
            self._workspace = None
            raise

    def _prepare_cuda_graph(self, groups, transported, cu, workspace):
        groups = prepare_state_cache_inputs(
            groups, self.state_cache, workspace=workspace.state_inputs
        )
        anchor = groups[next(iter(self.impls))]
        lengths = anchor.input_lengths
        prefixes = anchor.prefix_lengths
        if any(value.device.type != "cpu" for value in (lengths, prefixes)):
            raise RuntimeError("QSA draft graph replay requires host length mirrors")
        lengths = lengths.tolist()
        prefixes = prefixes.tolist()
        if (
            not lengths
            or len(prefixes) != len(lengths)
            or anchor.input_lengths_device.shape != (len(lengths),)
            or anchor.prefix_lengths_device.shape != (len(lengths),)
        ):
            raise RuntimeError("QSA draft graph replay has invalid length geometry")
        lanes = draft_graph_lanes(anchor, self.token_capacity)
        padding_started = False
        for length, prefix in zip(lengths, prefixes):
            if length == 0:
                padding_started = True
            elif padding_started or not 1 <= length <= lanes or prefix <= 0:
                raise RuntimeError(
                    "QSA draft graph requires positive live prefixes and lengths within request capacity followed by padding"
                )
        for tag, pool_size, page_size, compressed in self.bounds:
            inputs = groups[tag]
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
                    invalid = int(visible.min()) <= 0 or int(visible.max()) >= pool_size
                if invalid:
                    raise RuntimeError(
                        "QSA draft graph live row resolves to an unallocated or out-of-range physical block"
                    )
        # Resolve the physical pages from this replay's refreshed kernel table.
        state_table = groups[INDEXER_STATE_TAG].kv_cache_block_id_device
        # Draft Base-RoPE uses logical coordinates even for transported media rows.
        invalid_replay = workspace.prepare_positions(
            anchor,
            lengths,
            prefixes,
            cu,
            transported,
            (state_table, *self.state_bound),
            self.token_capacity,
            self.position_factor,
        )
        copied = {transported.data_ptr()}
        for value in groups.values():
            target = value.combo_position_ids
            if target.data_ptr() not in copied:
                target.copy_(transported.reshape_as(target))
                copied.add(target.data_ptr())
        if bool(invalid_replay.item()):
            raise RuntimeError(
                "QSA draft graph has invalid raw-state pages, cu_seqlens or positions disagree with logical cache positions"
            )
        for tag, impl in self.impls.items():
            impl.prepare_cuda_graph(groups[tag])
