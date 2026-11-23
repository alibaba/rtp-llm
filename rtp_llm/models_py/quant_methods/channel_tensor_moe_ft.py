"""Legacy per-expert FP8 FT runtime snapshots, including compressed calibration."""

import torch

from rtp_llm.models_py.ft_checkpoint import RuntimeTensorBinding
from rtp_llm.models_py.quant_methods.block_fp8_ft import (
    block_ft_source_is_rocm,
    block_ft_weight,
)


def nonblock_moe_ft_bindings(owner, prefix, target, checkpoint, layer_id):
    owner._ft_expert_binding_layout = (nonblock_moe_ft_bindings, prefix, layer_id)
    family = owner._quant_family
    if family == "none":
        return _plain_expert_bindings(owner, prefix, target, checkpoint, layer_id)
    if family not in ("fp8_per_channel", "fp8_per_tensor"):
        raise ValueError(f"No non-block FT expert ABI for {family}")
    channel = family == "fp8_per_channel"
    placement = owner._expert_placement
    packed = getattr(owner.quant_method, "_static_activation_metadata", False)
    count = owner.num_local_experts
    out = []

    def validate_scales(x):
        if x.dtype != torch.float32 or not torch.isfinite(x).all() or (x <= 0).any():
            raise ValueError("Legacy FT expert scales must be finite positive float32")
        return x

    def binding(keys, dest, decode, read, *, mark=None, projs=()):
        def install(values):
            getattr(owner, dest).copy_(decode(values))
            if mark:
                for slot in range(count):
                    for proj in projs:
                        if mark == "weight":
                            owner._record_projection_loaded(slot, proj)
                        else:
                            owner._loaded_aux_keys.add((slot, proj, mark))

        out.append(
            RuntimeTensorBinding(keys, target + "." + dest, decode, read, install)
        )

    def rows(key):
        return lambda: (checkpoint.read_expert_rows(key, layer_id, placement),)

    actual_calibration = []
    for w, src, projections in (
        ("w13", "intermediate_weight", ("up_proj", "gate_proj")),
        ("w2", "intermediate_weight2", ("down_proj",)),
    ):
        base = prefix + "partial_moe_weights." + src
        wk, sk = base + ".kernel", base + ".weight_only_quant_scale"
        raw_dt = checkpoint.read_rank_slice(
            wk, checkpoint.topology.tp_rank, (slice(0, 1),) * 3
        ).dtype
        rocm = block_ft_source_is_rocm(
            checkpoint.index_metadata.get("ft_export_backend"), raw_dt, False
        )
        shape = tuple(getattr(owner, w).shape)
        gate_up = w == "w13"

        def kernel(vals, shape=shape, rocm=rocm, gate_up=gate_up):
            return block_ft_weight(
                vals[0], shape, False, rocm_shuffle=rocm, gate_up=gate_up
            )

        binding((wk,), w, kernel, rows(wk), mark="weight", projs=projections)

        def scales(vals, shape=shape, rocm=rocm, gate_up=gate_up, dt=raw_dt):
            x = validate_scales(vals[0])
            expected = (shape[0], shape[1]) if channel else (shape[0],)
            if x.shape[: len(expected)] != expected or any(
                d != 1 for d in x.shape[len(expected) :]
            ):
                raise ValueError(
                    f"Unexpected FT expert scale shape {tuple(x.shape)}, expected {expected}"
                )
            x = x.reshape(expected)
            if channel and rocm and gate_up:
                a, b = x.chunk(2, dim=1)
                x = torch.cat((b, a), dim=1)
            return x * 0.5 if dt == torch.float8_e4m3fnuz else x

        binding(
            (sk,),
            w + "_scale",
            scales,
            rows(sk),
            mark="weight_scale",
            projs=projections,
        )
        for j, proj in enumerate(projections):
            stem = proj.removesuffix("_proj")
            leaf = "_" + stem + ("_ch_scales" if channel else "_scales")

            def part(vals, j=j, fn=scales, gate_up=gate_up):
                x = fn(vals)
                return x.chunk(2, dim=1)[j] if channel and gate_up else x

            binding((sk,), leaf, part, rows(sk))
        if not packed:
            continue
        activation = prefix + (
            "moe_w1_activation." if gate_up else "moe_w2_activation."
        )
        direct, inverse = (
            activation + "static_quant_reciprocal",
            activation + "static_quant",
        )

        # Main's compressed exporter reduces the individual HF input scales
        # to a rank-wide maximum. Preserve precisely the available calibration;
        # it is not a source of the individual HF per-expert observations.
        def read_calib(d=direct, i=inverse, source=wk):
            def by_slot(name):
                if checkpoint.topology.ep_size == 1:
                    scalar = checkpoint.read(name)
                    if scalar.numel() != 1:
                        raise ValueError("Invalid FT rank calibration, expected scalar")
                    return scalar.reshape(()).expand(count)
                return checkpoint.read_expert_rank_scalars(
                    name, source, layer_id, placement
                )

            return (by_slot(d), by_slot(i))

        def calib(vals):
            d, i = (validate_scales(v) for v in vals)
            if not torch.allclose(d * i, torch.ones_like(d), atol=1e-6, rtol=1e-5):
                raise ValueError("Inconsistent reciprocal FT expert calibration")
            return d

        actual_calibration.append((direct, inverse, read_calib, calib))
        binding(
            (direct, inverse),
            w + "_input_scale",
            lambda vals, fn=calib: fn(vals).amax(),
            read_calib,
            mark="input_scale",
            projs=projections,
        )
        binding(
            (direct, inverse),
            w + "_input_scale_inv",
            lambda vals, fn=calib: fn(vals).amax().reciprocal(),
            read_calib,
        )
    if packed:
        gu, down = actual_calibration

        def all_calib(vals):
            a, b = gu[3](vals[:2]), down[3](vals[2:])
            return torch.stack((a, a, b), dim=1)

        binding(
            (*gu[:2], *down[:2]),
            "checkpoint_input_scales",
            all_calib,
            lambda: (*gu[2](), *down[2]()),
        )
    return out


def _plain_expert_bindings(owner, prefix, target, checkpoint, layer_id):
    """A quantized model can exclude entire expert leaves from quantization."""
    from rtp_llm.models_py.quant_methods.moe_layout import undo_rocm_moe_matrix_tiles

    declared = checkpoint.index_metadata.get("ft_export_backend")
    if declared not in (None, "cuda", "rocm"):
        raise ValueError(f"Unknown FT producer ABI: {declared!r}")
    rocm = declared == "rocm" or (declared is None and torch.version.hip is not None)
    result = []
    for name, stem, projs in (
        ("w13", "intermediate_weight", ("up_proj", "gate_proj")),
        ("w2", "intermediate_weight2", ("down_proj",)),
    ):
        key = prefix + "partial_moe_weights." + stem + ".kernel"
        param = getattr(owner, name)

        def read(key=key):
            return (
                checkpoint.read_expert_rows(key, layer_id, owner._expert_placement),
            )

        def decode(vals, gate=name == "w13"):
            x = vals[0]
            if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise TypeError(
                    "Unquantized FT experts require an unquantized source dtype"
                )
            if rocm:
                x = undo_rocm_moe_matrix_tiles(x)
                if gate:
                    a, b = x.chunk(2, dim=1)
                    x = torch.cat((b, a), dim=1)
            return x

        def install(vals, decode=decode, param=param, projs=projs):
            param.copy_(decode(vals))
            for slot in range(owner.num_local_experts):
                for proj in projs:
                    owner._record_projection_loaded(slot, proj)

        result.append(
            RuntimeTensorBinding((key,), target + "." + name, decode, read, install)
        )
    return result
