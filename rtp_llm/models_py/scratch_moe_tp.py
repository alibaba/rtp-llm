"""Opt-in scratch reads for the historical Qwen pure-TP MoE layouts."""

import math

from rtp_llm.models_py.module_base import copy_weight_
from rtp_llm.models_py.read_spec import LoadedCheckpointTensor, make_sharded_read_spec


def make_reader(resolve_target):
    from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts

    def read(key, source):
        target = resolve_target(key)
        if (
            target is None
            or not isinstance(target.module, BaseMoEExperts)
            or not key.endswith(target.name)
        ):
            return None
        return read_parts(target.module, target.name, key, source)

    return read


def _spec(layer, name):
    expert, projection, suffix = name.split(".")
    if not expert.isdigit() or not 0 <= int(expert) < layer.num_experts:
        raise ValueError(f"Invalid pure TP expert {name}")
    if projection not in layer.PROJ_NAMES or layer._quant_family not in (
        "none",
        "fp8_per_block",
    ):
        raise ValueError(f"Invalid pure TP projection {name}")
    h, m = layer.hidden_size, layer.moe_inter
    down = projection == "down_proj"
    shape = (h, m) if down else (m, h)
    axis = int(down)
    if suffix == "weight_scale_inv" and layer._quant_family == "fp8_per_block":
        blocks = layer._fp8_moe_weight_block_size
        if layer.moe_inter_tp % blocks[axis]:
            raise ValueError("Pure TP FP8 expert split is not block aligned")
        shape = tuple(math.ceil(dim / block) for dim, block in zip(shape, blocks))
    elif suffix != "weight":
        raise ValueError(f"Unsupported pure TP tensor {name}")
    return make_sharded_read_spec(
        name,
        shape,
        axis,
        layer.moe_expert_tp_size,
        layer.moe_expert_tp_rank,
        partition="scratch_moe_tp",
    )


def read_parts(layer, name, key, source):
    """Return None for unsupported layouts; the regular loader stays authoritative."""
    if (
        layer._quant_family not in ("none", "fp8_per_block")
        or layer.ep_size != 1
        or layer.moe_expert_tp_size <= 1
    ):
        return None
    shape = tuple(source.get_shape())
    parts = name.split(".")
    if len(parts) == 3 and parts[0].isdigit() and parts[1] in layer.PROJ_NAMES:
        if parts[2] != "weight" and not (
            parts[2] == "weight_scale_inv" and layer._quant_family == "fp8_per_block"
        ):
            return None
        spec = _spec(layer, name)
        if shape != spec.source_shape:
            return None
        return ((key, spec.materialize(source)),)

    # Legacy Qwen3.5 stacked BF16: [E,2M,H] gate/up and [E,H,M] down.
    # Other orientations and quantized stacked files use normal dispatch.
    if (
        parts[0] not in ("gate_up_proj", "down_proj")
        or parts[1:] not in ([], ["weight"])
        or layer._quant_family != "none"
    ):
        return None
    h, m = layer.hidden_size, layer.moe_inter
    down = parts[0] == "down_proj"
    if shape != (layer.num_experts, *((h, m) if down else (2 * m, h))) or h == (
        m if down else 2 * m
    ):
        return None
    prefix = key[: -len(name)]

    def pieces():
        for expert in range(layer.num_experts):
            for projection in ("down_proj",) if down else ("gate_proj", "up_proj"):
                relative = f"{expert}.{projection}.weight"
                spec = _spec(layer, relative)
                rows, cols = spec.slices
                if projection == "up_proj":
                    rows = slice(m + rows.start, m + rows.stop)
                tensor = source[expert : expert + 1, rows, cols][0].contiguous()
                yield prefix + relative, LoadedCheckpointTensor(tensor, spec)

    return pieces()


def install_part(layer, name, value):
    """Install already-local input without slicing the TP dimension a second time."""
    spec = _spec(layer, name)
    if value.read_spec != spec or layer.ep_size != 1:
        raise ValueError(
            f"Pure TP checkpoint shard belongs to a different consumer: {name}"
        )
    expert, proj, suffix = name.split(".")
    tensor = value.tensor
    for slot in layer._remap_expert_ids(int(expert)):
        if suffix == "weight":
            layer._ensure_projection_not_loaded(slot, proj)
            # Preserve the quant method's checkpoint dtype checks.
            layer.quant_method.dispatch_weight(layer, slot, proj, suffix, tensor)
            target = (
                layer.w2[slot]
                if proj == "down_proj"
                else layer.w13[slot].chunk(2, dim=0)[int(proj == "gate_proj")]
            )
            copy_weight_(target, tensor, name)
            layer._record_projection_loaded(slot, proj)
        else:
            key = (slot, proj, suffix)
            if key in layer._loaded_aux_keys:
                raise RuntimeError(f"Duplicate pure TP expert scale: {name}")
            if not tensor.is_floating_point():
                raise TypeError(f"FP8 MoE {suffix} must be floating point")
            target = (
                layer.w2_scale[slot]
                if proj == "down_proj"
                else layer.w13_scale[slot].chunk(2, dim=0)[int(proj == "gate_proj")]
            )
            # The ordinary FP8 loader decodes any floating scale into FP32.
            copy_weight_(target, tensor.float(), name)
            layer._loaded_aux_keys.add(key)
