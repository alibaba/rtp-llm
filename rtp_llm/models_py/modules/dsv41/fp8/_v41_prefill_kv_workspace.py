"""Per-forward sparse-attention workspace with an immutable global prefix."""

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _refresh_backend():
    from . import _v41_swa_refresh_triton

    return _v41_swa_refresh_triton


class SwaRows:
    """Current-layer SWA storage plus host row ranges, including ragged batches.

    Request views are created lazily only for the original fallback/initial cat.
    The cross-layer cache never retains this descriptor or its source storage.
    """

    def __init__(self, flat, sizes, row_starts=None):
        sizes = tuple(sizes)
        if (
            not isinstance(flat, torch.Tensor)
            or flat.ndim != 2
            or any(type(size) is not int or size < 0 for size in sizes)
        ):
            raise ValueError("SWA descriptor needs a matrix and nonnegative row counts")
        source_rows = flat.shape[0]
        packed = row_starts is None
        if packed:
            row_starts, offset = [], 0
            for size in sizes:
                row_starts.append(offset)
                offset += size
            if offset != source_rows:
                raise ValueError("Packed SWA row counts must cover the source matrix")
        row_starts = tuple(row_starts)
        if len(row_starts) != len(sizes) or any(
            type(start) is not int or start < 0 or start + size > source_rows
            for start, size in zip(row_starts, sizes)
        ):
            raise ValueError("SWA row ranges exceed source storage")
        self.flat = flat
        self.sizes = sizes
        self.row_starts = row_starts
        self._packed = packed
        self._views = None

    def __len__(self):
        return len(self.sizes)

    def _request_views(self):
        if self._views is None:
            self._views = (
                self.flat.split(self.sizes, dim=0)
                if self._packed
                else tuple(
                    self.flat.as_strided(
                        (size, self.flat.shape[1]),
                        self.flat.stride(),
                        self.flat.storage_offset() + start * self.flat.stride(0),
                    )
                    for start, size in zip(self.row_starts, self.sizes)
                )
            )
        return self._views

    def __iter__(self):
        return iter(self._request_views())

    def __getitem__(self, index):
        return self._request_views()[index]


def split_swa_rows(flat, sizes, *, row_starts=None):
    """Describe contiguous-width CUDA storage without splitting every request."""
    if (
        flat.is_cuda
        and torch.version.hip is None
        and flat.dtype == torch.bfloat16
        and flat.ndim == 2
        and flat.stride(1) == 1
        and flat.stride(0) >= flat.shape[1]
        and not flat.requires_grad
        and not flat.is_conj()
        and not flat.is_neg()
    ):
        return SwaRows(flat, sizes, row_starts)
    descriptor = SwaRows(flat, sizes, row_starts)
    return list(descriptor)


def _row_geometry(swa):
    flat = swa.flat
    return (
        swa.sizes,
        swa.row_starts,
        flat.shape[1],
        flat.stride(0),
        flat.dtype,
        flat.device,
    )


class PreparedGlobalRows(tuple):
    """Immutable producer results owning their interleaved destination plan."""

    def __new__(cls, values, workspace):
        result = super().__new__(cls, values)
        result.workspace = workspace
        return result


class PreparedGlobalWorkspace:
    """One source group's global destination plus current-layer SWA ranges.

    Global aliases own this plan; it never retains a layer's SWA activation.
    Producer kernels write only global ranges, and the consumer refreshes SWA
    on the same forward stream before reading the complete interleaved output.
    """

    def __init__(self, swa, global_sizes, refresh, *, output_stream=None):
        self.global_sizes = global_sizes
        self.swa_capacity = swa.sizes
        self.geometry = _row_geometry(swa)
        starts, layout, offset, tiles = [], [], 0, 0
        for count, start, size in zip(global_sizes, swa.row_starts, swa.sizes):
            starts.append(offset)
            offset += count
            tiles += (
                size * (swa.flat.shape[1] // 4) + refresh._BLOCK - 1
            ) // refresh._BLOCK
            layout.extend((start, offset, size, tiles))
            offset += size
        self.global_starts = tuple(starts)
        if output_stream is None:
            self.output = torch.empty(
                (offset, swa.flat.shape[1]),
                dtype=swa.flat.dtype,
                device=swa.flat.device,
            )
        else:
            producer_stream = torch.cuda.current_stream(swa.flat.device)
            # The caller already ordered the producer after this stream's old
            # work. Only allocation changes stream; layout copies stay on the
            # producer, so there is no new event or cross-stream upload.
            with torch.cuda.stream(output_stream):
                self.output = torch.empty(
                    (offset, swa.flat.shape[1]),
                    dtype=swa.flat.dtype,
                    device=swa.flat.device,
                )
            # Protect even a failure during the remaining metadata preparation.
            self.output.record_stream(producer_stream)
        # One small asynchronous upload supplies both the producer mapping and
        # SWA refresh. Keep its pinned source alive with the global aliases.
        self.pinned_holder = torch.tensor(
            starts + layout, dtype=torch.int64, pin_memory=True
        )
        device = self.pinned_holder.to(device=swa.flat.device, non_blocking=True)
        self.device_offsets = device[: len(starts)]
        self.refresh_layout = device[len(starts) :].view(len(starts), 4)
        self.refresh_tiles = tiles
        self._refresh_metadata = (
            self.refresh_layout,
            self.pinned_holder,
            self.refresh_tiles,
        )

    def try_layout(self, swa):
        """Select physical global offsets and matching SWA destinations together.

        Shorter SWA tails keep the producer's request gaps, including at CED.
        Attention metadata must use global_starts/device_offsets whenever this
        succeeds. Unsupported sources or tails exceeding capacity use the old
        packed layout. Only geometry and a small host/device table are cached;
        no source activation survives the layer that owns it.
        """
        if not isinstance(swa, SwaRows):
            return None
        refresh = _refresh_backend()
        if not refresh.is_supported(swa.flat):
            return None
        geometry = _row_geometry(swa)
        if geometry == self.geometry:
            return self._refresh_metadata
        if (
            len(swa) != len(self.global_sizes)
            or swa.flat.device != self.output.device
            or swa.flat.dtype != self.output.dtype
            or swa.flat.shape[1] != self.output.shape[1]
            or any(
                size > capacity for size, capacity in zip(swa.sizes, self.swa_capacity)
            )
        ):
            return None
        layout, tiles = [], 0
        for count, destination, start, size in zip(
            self.global_sizes, self.global_starts, swa.row_starts, swa.sizes
        ):
            tiles += (
                size * (swa.flat.shape[1] // 4) + refresh._BLOCK - 1
            ) // refresh._BLOCK
            layout.append((start, destination + count, size, tiles))
        host = torch.tensor(layout, dtype=torch.int64, pin_memory=True)
        device = host.to(device=swa.flat.device, non_blocking=True)
        self.geometry = geometry
        self._refresh_metadata = (device, host, tiles)
        return self._refresh_metadata

    def publish(self, globals_by_req):
        """Attach aliases created by the direct producer after all its writes.

        The producer constructs exact aliases from global_starts/global_sizes.
        Rechecking every Tensor's storage here would add per-request CPU work
        to an internal contract already covered by the producer's byte tests.
        """
        values = tuple(globals_by_req)
        if len(values) != len(self.global_sizes):
            raise ValueError("Producer globals must cover every prepared request")
        return PreparedGlobalRows(values, self)


def prepare_global_workspace(swa, global_sizes, *, output_stream=None):
    """Reserve exact request ranges for direct producer output when supported.

    This is independent of batch size or equal request lengths. The caller must
    use the original producer path if this returns None, and must publish the
    returned plan only when the direct producer actually ran successfully.
    When output_stream is supplied, the caller must already have ordered the
    current producer stream after its prior work. Only the final allocation
    uses output_stream; layout uploads stay on the producer, without a new event.
    """
    if not isinstance(swa, SwaRows) or not swa.flat.is_cuda:
        return None
    global_sizes = tuple(global_sizes)
    if (
        not global_sizes
        or len(global_sizes) != len(swa)
        or any(type(size) is not int or size < 0 for size in global_sizes)
    ):
        return None
    refresh = _refresh_backend()
    if not refresh.is_supported(swa.flat):
        return None
    return PreparedGlobalWorkspace(
        swa, global_sizes, refresh, output_stream=output_stream
    )


def _combine_swa_rows(shared, globals_by_req, swa):
    """Refresh request tails through one startup-warmed ragged copy kernel."""
    flat = swa.flat
    if not flat.is_cuda:
        return None
    refresh = _refresh_backend()
    if isinstance(globals_by_req, PreparedGlobalRows):
        plan = globals_by_req.workspace
        layout = plan.try_layout(swa)
        if layout is not None:
            shared.pop("prefill_kv_workspace", None)
            refresh.copy_rows(flat, plan.output, layout[0], layout[2])
            return plan.output
    if not refresh.is_supported(flat):
        return None
    sources = tuple(g for g, _ in globals_by_req)
    if not sources or len(sources) != len(swa):
        return None
    geometry = _row_geometry(swa)
    cached = shared.get("prefill_kv_workspace")
    if cached is not None:
        previous, metadata, output = cached
        if (
            isinstance(previous, tuple)
            and len(previous) == len(sources)
            and all(a is b for a, b in zip(previous, sources))
            and isinstance(metadata, tuple)
            and len(metadata) == 5
            and metadata[0] == "ragged"
            and metadata[1] == geometry
        ):
            refresh.copy_rows(flat, output, metadata[2], metadata[4])
            return output
    if not all(
        g.is_cuda
        and g.device == flat.device
        and g.dtype == flat.dtype
        and g.ndim == 2
        and g.shape[1] == flat.shape[1]
        and g.is_contiguous()
        and not g.requires_grad
        and not g.is_conj()
        and not g.is_neg()
        for g in sources
    ):
        return None
    # Preserve the existing profitability policy, independent of equal lengths.
    global_bytes = sum(g.numel() * g.element_size() for g in sources)
    if (
        global_bytes < len(sources) * 8 * 1024**2
        or global_bytes < sum(swa.sizes) * flat.shape[1] * flat.element_size()
    ):
        return None
    shared.pop("prefill_kv_workspace", None)
    if cached is not None:
        del cached, previous, metadata, output
    output = torch.cat([t for g, sw in zip(sources, swa) for t in (g, sw)])
    # Host dimensions are authoritative even in pool-free warmup. Only this
    # source-group transition uploads a small table, never each consumer layer.
    layout, offset, tiles = [], 0, 0
    for g, start, size in zip(sources, swa.row_starts, swa.sizes):
        offset += g.shape[0]
        tiles += (size * (flat.shape[1] // 4) + refresh._BLOCK - 1) // refresh._BLOCK
        layout.append((start, offset, size, tiles))
        offset += size
    host = torch.tensor(layout, dtype=torch.int64, pin_memory=True)
    device = host.to(device=flat.device, non_blocking=True)
    shared["prefill_kv_workspace"] = (
        sources,
        ("ragged", geometry, device, host, tiles),
        output,
    )
    return output


def _swa_layout(swa):
    return tuple(
        (
            t.shape,
            t.stride(),
            t.dtype,
            t.device,
            t.storage_offset() % 4,
            t.requires_grad,
            t.is_conj(),
            t.is_neg(),
        )
        for t in swa
    )


def _combine_batched_kv(shared, globals_by_req, swa):
    sources = tuple(g for g, _ in globals_by_req)
    signature = None
    cached = shared.get("prefill_kv_workspace")
    if cached is not None:
        previous_sources, metadata, output = cached
        if (
            isinstance(previous_sources, tuple)
            and len(previous_sources) == len(sources)
            and all(a is b for a, b in zip(previous_sources, sources))
        ):
            if metadata is None:
                return torch.cat([t for g, sw in zip(sources, swa) for t in (g, sw)])
            signature = _swa_layout(swa)
            if metadata[0] == signature:
                # Bitwise 8-byte views avoid foreach's 320-block launch limit for
                # the usual B32 SWA tails; no conversion or staging allocation.
                torch._foreach_copy_(metadata[1], [sw.view(torch.int64) for sw in swa])
                return output
        shared.pop("prefill_kv_workspace")
        del cached, previous_sources, metadata, output

    tensors = [t for g, sw in zip(sources, swa) for t in (g, sw)]
    global_bytes = sum(g.numel() * g.element_size() for g in sources)
    # Metadata cost scales with B. Require at least 8 MiB saved per request
    # on average, and no more SWA traffic than the immutable global payload.
    if global_bytes < len(sources) * 8 * 1024**2:
        # Remember only the source decision, not an output or SWA metadata.
        if sources and sources[0].is_cuda:
            shared["prefill_kv_workspace"] = (sources, None, None)
        return torch.cat(tensors)
    if global_bytes < sum(sw.numel() * sw.element_size() for sw in swa):
        return torch.cat(tensors)
    first = sources[0]
    # Contiguous, same-dtype CUDA pairs select foreach's multi-tensor copy.
    # Small globals do not amortize Python metadata checks and a retained buffer.
    eligible = len(sources) == len(swa) and all(
        t.is_cuda
        and t.dtype == torch.bfloat16
        and t.device == first.device
        and t.ndim == 2
        and t.shape[1:] == first.shape[1:]
        and t.shape[1] % 4 == 0
        and t.storage_offset() % 4 == 0
        and t.is_contiguous()
        and not t.requires_grad
        and not t.is_conj()
        and not t.is_neg()
        for t in tensors
    )
    if not eligible:
        return torch.cat(tensors)

    if signature is None:
        signature = _swa_layout(swa)
    output = torch.cat(tensors)
    views = []
    offset = 0
    for g, sw in zip(sources, swa):
        offset += g.shape[0]
        views.append(output[offset : offset + sw.shape[0]].view(torch.int64))
        offset += sw.shape[0]
    shared["prefill_kv_workspace"] = (sources, (signature, views), output)
    return output


def combine_kv(shared, globals_by_req, swa):
    """Only refresh SWA while layers share the same global tensor.

    The caller consumes the result on the forward stream before the next
    layer updates it. The cache is cleared with the per-forward global state.
    SwaRows uses one startup-warmed, length-driven copy for packed or padded
    source storage, including ragged requests. Its per-source-group layout is
    uploaded once and retains the pinned host buffer until cache invalidation.
    General tensor sequences retain the existing foreach/cat fallback. Only
    owned output views and immutable global keys survive across layers; the
    current layer's SWA activation is never retained by this cache.
    """
    if isinstance(swa, SwaRows):
        refreshed = _combine_swa_rows(shared, globals_by_req, swa)
        if refreshed is not None:
            return refreshed
    if len(globals_by_req) != 1:
        return _combine_batched_kv(shared, globals_by_req, swa)
    g, sw = globals_by_req[0][0], swa[0]
    # For small encoder workspaces the extra Python/slice/copy launch cost
    # exceeds the saved traffic. The bounded decoder tail always benefits;
    # long encoder groups also amortize that cost with >=64K global rows.
    if sw.shape[0] > 128 and g.shape[0] < 65536:
        shared.pop("prefill_kv_workspace", None)
        return torch.cat((g, sw))
    signature = (sw.shape, sw.dtype, sw.device)
    cached = shared.get("prefill_kv_workspace")
    if cached is not None:
        source, previous_signature, output = cached
        if source is g and previous_signature == signature:
            output[g.shape[0] :].copy_(sw)
            return output
        # Drop the old buffer before allocating a new source group's layout.
        shared.pop("prefill_kv_workspace")
        del cached, source, output
    output = torch.cat((g, sw))
    shared["prefill_kv_workspace"] = (g, signature, output)
    return output
