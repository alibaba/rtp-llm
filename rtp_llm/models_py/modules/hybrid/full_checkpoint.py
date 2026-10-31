"""Copy completed non-CP FULL checkpoints from the runtime kernel-page ABI."""

from rtp_llm.device.device_type import DeviceType, get_device_type


def save_full_checkpoints(cache, inputs, base_tokens):
    """Produce separate physical save blocks after the ordinary attention writer.

    Inputs carry group-local kernel-page sources and physical base-block targets.
    A checkpoint owns only its final base block; preceding immutable blocks are
    managed by the coordinator. No bytes after P or source pages are changed.
    CP uses its owner-aware writer instead. This helper does not publish blocks.
    """
    if (
        cache is None
        or not inputs.is_prefill
        or getattr(inputs, "context_parallel_info", None) is not None
    ):
        return
    positions = getattr(inputs, "cache_save_positions", None)
    destinations = getattr(inputs, "cache_save_block_ids", None)
    if positions is None or destinations is None or not positions.numel():
        return
    if get_device_type() != DeviceType.Cuda:
        raise RuntimeError("selected FULL checkpoint saving requires NVIDIA CUDA")
    kernel = int(cache.seq_size_per_block)
    base = int(base_tokens)
    if kernel <= 0 or base <= 0 or base % kernel:
        raise ValueError("FULL checkpoint base must contain whole kernel pages")
    pages = base // kernel
    physical = cache.kv_cache_base
    scale = cache.kv_scale_base
    if physical.ndim == 5 and physical.shape[3] == kernel:
        # OpDefs.h makeLayerCache: MHA [pages, 2, heads, kernel, dim].
        token_axis = 2  # axis after indexing a single page
        if scale is not None and scale.numel():
            scale = scale.view(physical.shape[0], 2, physical.shape[2], kernel)
    elif physical.ndim == 3 and physical.shape[1] == kernel:
        # MLA (including its scale storage) is [pages, kernel, elements].
        token_axis = 0
        if scale is not None and scale.numel():
            if scale.ndim != 3 or scale.shape[:2] != physical.shape[:2]:
                raise ValueError("unexpected MLA checkpoint scale page layout")
    else:
        raise ValueError("unsupported FULL checkpoint kernel-page layout")
    table = inputs.kv_cache_kernel_block_id
    prefixes = inputs.prefix_lengths.tolist()
    suffixes = inputs.input_lengths.tolist()
    for batch, (prefix, suffix) in enumerate(zip(prefixes, suffixes)):
        for endpoint, destination in zip(
            positions[batch].tolist(), destinations[batch].tolist()
        ):
            # Negative IDs cover absent slots and already-held cached resources.
            if destination <= 0 or endpoint <= prefix or endpoint > prefix + suffix:
                continue
            slot, remainder = divmod(endpoint - 1, base)
            valid = remainder + 1
            for offset in range((valid + kernel - 1) // kernel):
                source = int(table[batch, slot * pages + offset])
                target = destination * pages + offset
                if source < 0:
                    raise ValueError("missing runtime FULL page for checkpoint")
                if source == target:
                    continue
                count = min(kernel, valid - offset * kernel)
                physical[target].narrow(token_axis, 0, count).copy_(
                    physical[source].narrow(token_axis, 0, count)
                )
                if scale is not None and scale.numel():
                    scale[target].narrow(token_axis, 0, count).copy_(
                        scale[source].narrow(token_axis, 0, count)
                    )
