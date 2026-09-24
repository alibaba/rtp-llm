"""Per-forward sparse-attention workspace with an immutable global prefix."""

import torch


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
    Multi-request layouts refresh their disjoint SWA slices in one foreach copy.
    """
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
