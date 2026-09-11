"""Check the existing opaque DSv4 cache contract at resource binding time."""


def cache_geometry_snapshot(model_config, engine_config, *, speculative):
    """Use the same resolved inputs as HybridPoolConfigCreator, without Torch."""
    config = engine_config.kv_cache_config
    physical = int(config.seq_size_per_block)
    if physical <= 0 or physical == 64:
        physical = int(model_config.attn_config.tokens_per_block)
    kernel = int(config.kernel_seq_size_per_block)
    if kernel <= 0:
        kernel = physical
    return {
        "physical_tokens_per_block": physical,
        "kernel_tokens_per_block": kernel,
        # CacheConfigCreator::createConfig passes zero for the normal engine,
        # even though SpeculativeExecutionConfig defaults this field to one.
        "gen_num_per_cycle": (
            int(engine_config.sp_config.gen_num_per_cycle) if speculative else 0
        ),
    }


def opaque_cache_layouts(metadata):
    """Describe each LayerKVCache view, including native alignment padding.

    Mirrors OpaqueKVCacheSpec entryCount/blockStrideBytes for the CP1 contract.
    Native FULL groups expand physical blocks into kernel blocks at the Python
    boundary; fixed state groups retain physical blocks (OpDefs.h).
    """
    if metadata["cp_enabled"]:
        raise ValueError("Opaque cache validation requires the declared CP1 contract")
    geometry = metadata["cache_geometry"]
    physical = geometry["physical_tokens_per_block"]
    kernel = geometry["kernel_tokens_per_block"]
    generation = geometry["gen_num_per_cycle"]
    if physical <= 0 or kernel <= 0 or physical % kernel or generation < 0:
        raise ValueError("Invalid preflight cache block geometry")
    layers = metadata["cache_descriptions"]
    if len(layers) != metadata["num_layers"]:
        raise ValueError("Cache descriptor layer count differs from model")
    result = []
    dtype_bytes = {"TYPE_UINT8": 1, "TYPE_FP32": 4}
    for layer in layers:
        planned = {}
        for desc in layer:
            tag = desc["tag"]
            dtype = desc["entry_dtype"]
            if tag in planned or dtype not in dtype_bytes:
                raise ValueError(f"Unsupported or duplicate cache descriptor {tag}")
            kind = desc["cache_type"]
            if kind not in ("OPAQUE_KV", "OPAQUE_STATE"):
                raise ValueError(f"Unsupported cache type for {tag}: {kind}")
            mode = desc["entry_count_mode"]
            ratio = desc["compression_ratio"]
            if mode == "KERNEL_BLOCK_COMPRESSED" and ratio > 0 and kernel % ratio == 0:
                entries = kernel // ratio
            elif mode == "STATE_RING" and ratio > 0:
                entries = (1 + desc["state_ring_overlap"]) * ratio
                if desc["state_ring_include_gen_num_per_cycle"]:
                    entries += generation
                entries = (entries + 1) & ~1
            elif mode == "EXPLICIT":
                entries = desc["explicit_entry_count"]
            else:
                raise ValueError(f"Unsupported cache entry geometry for {tag}")
            if entries <= 0 or desc["entry_elems"] <= 0:
                raise ValueError(f"Empty cache entry geometry for {tag}")
            payload = entries * desc["entry_elems"] * dtype_bytes[dtype]
            stride = desc["block_stride_bytes_override"]
            if stride <= 0:
                stride = payload
                alignment = desc["block_stride_bytes_alignment"]
                if (
                    alignment > 0
                    and entries >= desc["block_stride_alignment_min_entries"]
                ):
                    stride = (payload + alignment - 1) // alignment * alignment
            if stride < payload or stride % dtype_bytes[dtype]:
                raise ValueError(f"Invalid cache byte stride for {tag}")
            full = kind == "OPAQUE_KV" and desc["group_type"] in (None, "FULL")
            memory = desc["memory"] or {}
            planned[tag] = {
                "dtype": dtype,
                "entry_elems": desc["entry_elems"],
                "entries_per_view": entries,
                "stride_bytes": stride,
                "seq_size_per_block": kernel if full else physical,
                "physical_tokens_per_block": physical,
                "kernel_tokens_per_block": kernel if full else physical,
                "blocks_per_physical": physical // kernel if full else 1,
                "placement": memory.get("placement") or "DEVICE",
            }
        if not planned:
            raise ValueError("Model layer has no cache descriptors")
        result.append(planned)
    return result


def validate_bound_cache(cache, metadata, *, device):
    """Validate actual tensors without reading, copying or retaining their data."""
    import torch

    planned = opaque_cache_layouts(metadata)
    if cache is None:
        return {"bound": False, "reason": "no_kv_initialization"}
    expected_tags = list(dict.fromkeys(tag for layer in planned for tag in layer))
    if list(cache.group_tags) != expected_tags or cache.layer_count != len(planned):
        raise ValueError("Actual cache topology differs from preflight")
    dtypes = {"TYPE_UINT8": torch.uint8, "TYPE_FP32": torch.float32}
    groups = {}
    layer_groups = []
    for layer_id, expected in enumerate(planned):
        actual = cache.get_layer_cache_groups(layer_id)
        if len(actual) != len(expected) or {view.tag for view in actual} != set(
            expected
        ):
            raise ValueError(
                f"Actual cache group membership differs at layer {layer_id}"
            )
        layer_groups.append([view.tag for view in actual])
        for view in actual:
            spec = expected[view.tag]
            name = f"layer {layer_id} cache {view.tag}"
            if view.layer_id != layer_id or view.group_id != expected_tags.index(
                view.tag
            ):
                raise ValueError(f"Invalid resource identity for {name}")
            if (
                view.seq_size_per_block != spec["seq_size_per_block"]
                or cache.get_seq_size_per_block(view.tag)
                != spec["physical_tokens_per_block"]
                or cache.get_kernel_seq_size_per_block(view.tag)
                != spec["kernel_tokens_per_block"]
            ):
                raise ValueError(f"Actual block geometry differs for {name}")
            tensor = view.kv_cache_base
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dim() != 2
                or tensor.numel() == 0
                or tensor.dtype != dtypes[spec["dtype"]]
                or not tensor.is_contiguous()
                or tensor.stride(1) != 1
                or tensor.shape[1] * tensor.element_size() != spec["stride_bytes"]
                or tensor.stride(0) * tensor.element_size() != spec["stride_bytes"]
                or tensor.shape[0] % spec["blocks_per_physical"]
            ):
                actual = (
                    (str(tensor.dtype), tuple(tensor.shape), tuple(tensor.stride()))
                    if isinstance(tensor, torch.Tensor)
                    else type(tensor).__name__
                )
                raise ValueError(
                    f"Actual cache dtype/shape/stride differs for {name}: "
                    f"actual={actual}, expected={spec}"
                )
            placement = spec["placement"]
            expected_device = (
                torch.device(device) if placement == "DEVICE" else torch.device("cpu")
            )
            if tensor.device != expected_device:
                raise ValueError(f"Actual cache device differs for {name}")
            if placement == "HOST_PINNED" and not tensor.is_pinned():
                raise ValueError(f"Expected pinned host cache for {name}")
            scale = view.kv_scale_base
            if scale is not None and scale.numel() != 0:
                raise ValueError(
                    f"Opaque cache must not have a separate scale buffer: {name}"
                )
            record = {
                **spec,
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "device": str(tensor.device),
            }
            if view.tag in groups and groups[view.tag] != record:
                raise ValueError(
                    f"Cache group layout differs between layers: {view.tag}"
                )
            groups[view.tag] = record
    return {
        "bound": True,
        "layers": len(planned),
        "groups": groups,
        "layer_groups": layer_groups,
    }


def fp8_allocator_inputs():
    return {"indexer_cache_mode": "FP8"}


def fp4_allocator_inputs():
    return {"indexer_cache_mode": "FP4"}
