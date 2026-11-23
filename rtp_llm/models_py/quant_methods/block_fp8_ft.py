"""Decode RTP's legacy block-FP8 FT snapshots into rank-local owner tensors.

PerBlockFp8Weight._postprocess uses two distinct serializations. Unpacked dense
matrices were *reshaped*, not transposed; expert matrices retain three axes.
DeepGEMM snapshots already contain requantized weights plus four exponent bytes
per int32, with one identical scale row for each row in the 128-row block.
Decoding those bytes is lossless. Never dequantize/requantize such weights.
"""

import torch

from rtp_llm.models_py.ft_checkpoint import RuntimeTensorBinding


def _blocks(size, block=128):
    return (size + block - 1) // block


def block_ft_weight(raw, logical_shape, packed, *, rocm_shuffle=False, gate_up=False):
    """Undo only transforms used by the declared producer."""
    shape = tuple(logical_shape)
    if len(shape) not in (2, 3):
        raise ValueError("Expected a dense or expert block-FP8 matrix")
    expected = (shape[1], shape[0]) if not packed and len(shape) == 2 else shape
    if tuple(raw.shape) != expected:
        raise ValueError(f"FT block-FP8 weight shape {tuple(raw.shape)} != {expected}")
    if raw.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        raise TypeError("FT block-FP8 weight must have an explicit E4M3 dtype")
    if packed and raw.dtype != torch.float8_e4m3fn:
        raise TypeError("Packed UE8M0 scales require E4M3FN DeepGEMM weights")
    value = raw.contiguous().reshape(shape)
    if rocm_shuffle:
        if len(shape) != 3 or packed:
            raise ValueError("Invalid AITER expert serializer for block-FP8 tensor")
        from rtp_llm.models_py.quant_methods.moe_layout import (
            undo_rocm_moe_matrix_tiles,
        )

        value = undo_rocm_moe_matrix_tiles(value.view(torch.uint8)).view(raw.dtype)
        if gate_up:
            value = swap_halves(value)
    if value.dtype == torch.float8_e4m3fnuz:
        # Old ROCm producer doubled the scales and retained the FP8 bits
        # (except -0 -> +0). FNUZ 0x80 is a NaN, not FN negative zero.
        if (value.view(torch.uint8) == 128).any():
            raise ValueError("Invalid NaN in ROCm FT block-FP8 weight")
        value = value.view(torch.float8_e4m3fn)
    return value


def swap_halves(value):
    if value.shape[-2] % 2:
        raise ValueError("Cannot split the gate and up projections equally")
    # torch.cat may not implement FP8; the bytes and element layout are identical.
    if value.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        return swap_halves(value.view(torch.uint8)).view(value.dtype)
    first, second = value.chunk(2, dim=-2)
    return torch.cat((second, first), dim=-2)


def block_ft_scale(raw, logical_shape, *, source_dtype, swap_gate_up=False, padding=None):
    shape = tuple(logical_shape)
    *batch, n, k = shape
    nb, kb = _blocks(n), _blocks(k)
    if raw.dtype == torch.int32:
        if source_dtype != torch.float8_e4m3fn or swap_gate_up:
            raise ValueError("Invalid producer for packed UE8M0 scale")
        expected = (*batch, n, _blocks(kb, 4))
        if tuple(raw.shape) != expected:
            raise ValueError(
                f"Packed UE8M0 scale shape {tuple(raw.shape)} != {expected}"
            )
        # Inspect replicated rows without materializing another copy of every
        # byte in the expert tensor. Last N block may be only partially filled.
        complete, tail = divmod(n, 128)
        if complete:
            groups = raw[..., : complete * 128, :].unflatten(-2, (complete, 128))
            if not torch.equal(groups, groups[..., :1, :].expand_as(groups)):
                raise ValueError("Packed UE8M0 contains inconsistent replicated rows")
        if tail:
            end = raw[..., complete * 128 :, :]
            if not torch.equal(end, end[..., :1, :].expand_as(end)):
                raise ValueError("Packed UE8M0 contains inconsistent tail rows")
        packed = raw[..., ::128, :]
        shifts = torch.arange(0, 32, 8, device=raw.device, dtype=torch.int32)
        exps = ((packed.unsqueeze(-1) >> shifts) & 255).flatten(-2)[..., :kb]
        if (exps == 255).any():
            raise ValueError("Invalid UE8M0 NaN exponent in FT scale")
        return torch.exp2((exps - 127).float()).contiguous()
    if raw.dtype != torch.float32:
        raise TypeError("FT block scale must have dtype float32 or packed int32")
    expected = (kb, nb) if len(shape) == 2 else (*batch, nb, kb)
    if tuple(raw.shape) != expected:
        raise ValueError(f"FT block scale shape {tuple(raw.shape)} != {expected}")
    value = raw.contiguous().reshape(*batch, nb, kb)
    if swap_gate_up:
        value = swap_halves(value)
    if source_dtype == torch.float8_e4m3fnuz:
        value = value * 0.5
    if padding is not None:
        # Legacy export fills *whole* added scale blocks with zero. The
        # paired weight decoder verifies the corresponding weights are zero.
        value = value.masked_fill(padding & (value == 0), 1.0)
    if not torch.isfinite(value).all() or (value <= 0).any():
        raise ValueError("Block scales must be positive and finite")
    return value


def block_ft_runtime_pair(layer, weight, scale, scale_key, requant):
    """Keep already-quantized FT payload unchanged when selecting DeepGEMM.

    Register the packed-scale provenance at install time, not on attempted
    validation. The checkpoint canonical scale can also serve other backends.
    """
    if scale_key not in getattr(layer, "_packed_block_fp8_ft", ()):
        return requant(weight, scale)
    from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
        _transform_scale_ue8m0,
    )

    return weight, _transform_scale_ue8m0(scale, weight.shape[-2])


def block_ft_ffn_width(checkpoint, down_key, logical_width, hidden):
    """Keep the producer's TP partition; cropping each rank would lose real rows."""
    if logical_width % 128 == 0:
        # Padding cannot change this width. The normal binding still checks
        # the payload shape; no extra peer file reads are needed.
        return logical_width
    scale_key = down_key.removesuffix(".kernel") + ".weight_only_quant_scale"
    widths = set()
    for peer in range(checkpoint.topology.tp_size):
        packed = (
            checkpoint.read_rank_slice(scale_key, peer, (slice(0, 1),) * 2).dtype
            == torch.int32
        )
        shape = checkpoint.read_rank_slice(down_key, peer)
        if len(shape) != 2:
            raise ValueError("Expected a dense block-FP8 FT down projection")
        n, width = shape if packed else shape[::-1]
        if n != hidden or width not in (logical_width, _blocks(logical_width) * 128):
            raise ValueError(f"Invalid block-FP8 FT FFN shard shape {shape}")
        widths.add(width)
    if len(widths) != 1:
        raise ValueError("Block-FP8 FT FFN ranks disagree on padded width")
    return widths.pop()


def block_ft_bindings(
    layer,
    source,
    scale_source,
    target,
    checkpoint,
    *,
    swizzled=False,
    projection=None,
    ffn_width=None,
):
    """Import a dense rank-local linear matrix while preserving Parameter IDs."""
    peer = checkpoint.topology.tp_rank
    full_shape = tuple(layer.weight.shape)
    param = layer.weight_scale_inv
    if tuple(param.shape) != (_blocks(full_shape[0]), _blocks(full_shape[1])):
        raise ValueError("FT adapter requires an unprocessed 128x128 FP8 owner")
    sources = (source,) if isinstance(source, str) else tuple(source)
    scale_sources = (
        (scale_source,) if isinstance(scale_source, str) else tuple(scale_source)
    )
    if len(sources) != len(scale_sources) or len(sources) not in (1, 2):
        raise ValueError("Expected one linear projection or separate gate/up planes")
    padded = False
    if ffn_width is not None:
        if projection not in ("gate_up", "down"):
            raise ValueError(
                "FT FFN width is only valid for gate/up or down projections"
            )
        logical_width = full_shape[0] // 2 if projection == "gate_up" else full_shape[1]
        if ffn_width not in (logical_width, _blocks(logical_width) * 128):
            raise ValueError("Invalid block-FP8 FT FFN padding")
        padded = ffn_width != logical_width
        full_shape = (
            (2 * ffn_width, full_shape[1])
            if projection == "gate_up"
            else (full_shape[0], ffn_width)
        )
        live = max(
            0,
            min(
                ffn_width,
                logical_width * checkpoint.topology.tp_size - peer * ffn_width,
            ),
        )
    shape = full_shape
    if len(sources) == 2:
        if full_shape[0] % 256:
            raise ValueError(
                "Separate gate and up block scales require an aligned split"
            )
        shape = (full_shape[0] // 2, full_shape[1])

    def dtype(name):
        return checkpoint.read_rank_slice(name, peer, (slice(0, 1),) * 2).dtype

    source_types = {dtype(s) for s in sources}
    scale_types = {dtype(s) for s in scale_sources}
    if len(source_types) != 1 or len(scale_types) != 1:
        raise ValueError("Gate/up FT checkpoint ABIs differ")
    source_type, scale_type = source_types.pop(), scale_types.pop()
    packed = scale_type == torch.int32
    declared = checkpoint.index_metadata.get("ft_export_backend")
    rocm = block_ft_source_is_rocm(declared, source_type, packed)

    def entry(keys):
        return lambda: tuple(checkpoint.read(key) for key in keys)

    def combine(parts):
        if len(parts) == 1:
            return parts[0]
        dt = parts[0].dtype
        if dt == torch.float8_e4m3fn:
            return torch.cat([p.view(torch.uint8) for p in parts]).view(dt)
        return torch.cat(parts)

    def decode_weight(values):
        pieces = []
        for name, val in zip(sources, values):
            weight = block_ft_weight(val, shape, packed)
            # Old ROCm rewrites dense kernels before its final flat reshape.
            # Separate w1 (gate) is specifically absent from the rewrite list.
            if rocm and not name.endswith("ffn_weights.intermediate_weight.kernel"):
                if swizzled:
                    from rtp_llm.models_py.quant_methods.dense_layout import (
                        decode_dense_weight,
                    )

                    weight = decode_dense_weight(weight)
                else:
                    # shuffle_gemm_weight uses the same AITER tile permutation.
                    weight = block_ft_weight(
                        weight.unsqueeze(0), (1, *shape), False, rocm_shuffle=True
                    )[0]
            if padded and live < ffn_width:
                planes = (
                    (weight,)
                    if len(sources) == 2 or projection == "down"
                    else weight.chunk(2)
                )
                for plane in planes:
                    tail = plane[live:] if projection == "gate_up" else plane[:, live:]
                    if torch.count_nonzero(tail.float()).item():
                        raise ValueError(
                            "Block-FP8 FT FFN padding contains nonzero weights"
                        )
            pieces.append(weight)
        return combine(pieces)

    def decode_scale(values):
        padding = None
        if padded and live < ffn_width:
            padding = torch.arange(
                _blocks(ffn_width), device=values[0].device
            ) >= _blocks(live)
            if projection == "gate_up":
                padding = padding.repeat(2 if len(sources) == 1 else 1).unsqueeze(1)
        return combine(
            [
                block_ft_scale(val, shape, source_dtype=source_type, padding=padding)
                for val in values
            ]
        )

    scale_shape = (_blocks(full_shape[0]), _blocks(full_shape[1]))

    def install_weight(values):
        value = decode_weight(values)
        layer.weight.data = layer.weight.new_empty(full_shape)
        layer.weight.copy_(value)
        if projection == "gate_up":
            layer.output_size = layer.output_size_per_partition = full_shape[0]
        else:
            layer.input_size = layer.input_size_per_partition = full_shape[1]

    def install_scale(values):
        if padded and tuple(param.shape) != scale_shape:
            param.data = param.new_empty(scale_shape)
        param.copy_(decode_scale(values))
        layer._packed_block_fp8_ft = {"weight_scale_inv"} if packed else set()

    return (
        RuntimeTensorBinding(
            sources,
            target + ".weight",
            decode_weight,
            entry(sources),
            install_weight if padded else None,
            full_shape if padded else None,
        ),
        RuntimeTensorBinding(
            scale_sources,
            target + ".weight_scale_inv",
            decode_scale,
            entry(scale_sources),
            install_scale,
            scale_shape if padded else None,
        ),
    )


def block_ft_source_is_rocm(declared, source_type, packed):
    if declared not in (None, "cuda", "rocm"):
        raise ValueError(f"Unknown FT producer ABI: {declared!r}")
    if (declared == "rocm" and packed) or (
        declared == "cuda" and source_type == torch.float8_e4m3fnuz
    ):
        raise ValueError("Contradictory FT producer ABI and block-FP8 payload")
    return not packed and (
        declared == "rocm"
        or source_type == torch.float8_e4m3fnuz
        or (declared is None and torch.version.hip is not None)
    )


def block_moe_ft_bindings(layer, source_prefix, target, checkpoint, layer_id):
    """Read the physical rows for this owner, then restore its normal HF staging ABI.

    OldLoader dumps the local execution layout verbatim and reloads it on
    the same runtime. Untagged FN snapshots on ROCm therefore use AITER,
    not CUDA's unshuffled layout. FNUZ and packed scales identify their
    producer independently. A tagged snapshot can also be moved across ABIs.
    """
    if layer._quant_family != "fp8_per_block" or layer._fp8_moe_weight_block_size != [
        128,
        128,
    ]:
        raise ValueError("Expected a block-FP8 expert owner with 128x128 runtime tiles")
    # The runtime layer index need not match the archive's source layer (MTP).
    layer._ft_expert_binding_layout = (block_moe_ft_bindings, source_prefix, layer_id)
    bindings = []
    placement = layer._expert_placement
    declared = checkpoint.index_metadata.get("ft_export_backend")

    def bind(key, tensor_name, convert, projections, scale, packed):
        param = getattr(layer, tensor_name)

        def read():
            return (checkpoint.read_expert_rows(key, layer_id, placement),)

        def install(values):
            param.copy_(convert(values))
            if scale and packed:
                layer._packed_block_fp8_ft = getattr(
                    layer, "_packed_block_fp8_ft", set()
                ) | {tensor_name}
            for slot in range(layer.num_local_experts):
                for proj in projections:
                    if scale:
                        layer._loaded_aux_keys.add((slot, proj, "weight_scale_inv"))
                    else:
                        layer._record_projection_loaded(slot, proj)

        return RuntimeTensorBinding(
            (key,), target + "." + tensor_name, convert, read, install
        )

    for w, stem, projs in (
        ("w13", "intermediate_weight", ("up_proj", "gate_proj")),
        ("w2", "intermediate_weight2", ("down_proj",)),
    ):
        prefix = source_prefix + "partial_moe_weights." + stem
        weight_key = prefix + ".kernel"
        scale_key = prefix + ".weight_only_quant_scale"
        shape = tuple(getattr(layer, w).shape)

        def dtype(key):
            return checkpoint.read_rank_slice(
                key, checkpoint.topology.tp_rank, (slice(0, 1),) * 3
            ).dtype

        source_type, scale_type = dtype(weight_key), dtype(scale_key)
        packed = scale_type == torch.int32
        rocm = block_ft_source_is_rocm(declared, source_type, packed)
        gate_up = w == "w13"

        def convert_weight(values, s=shape, p=packed, r=rocm, g=gate_up):
            return block_ft_weight(values[0], s, p, rocm_shuffle=r, gate_up=g)

        def convert_scale(values, s=shape, dt=source_type, g=gate_up, r=rocm):
            raw = values[0]
            if not g:
                return block_ft_scale(raw, s, source_dtype=dt)
            # HF and old FT use separate gate/up block grids even when each
            # local FFN width ends with a partial block.
            half = s[-2] // 2
            half_rows = half if raw.dtype == torch.int32 else _blocks(half)
            if raw.shape[-2] != 2 * half_rows:
                raise ValueError("Invalid gate/up FT expert scale grid")
            left, right = raw.split(half_rows, dim=-2)
            decoded = [
                block_ft_scale(part.contiguous(), (s[0], half, s[-1]), source_dtype=dt)
                for part in ((right, left) if r else (left, right))
            ]
            return torch.cat(decoded, dim=-2)

        bindings.append(bind(weight_key, w, convert_weight, projs, False, packed))
        bindings.append(
            bind(scale_key, w + "_scale", convert_scale, projs, True, packed)
        )
    return bindings
