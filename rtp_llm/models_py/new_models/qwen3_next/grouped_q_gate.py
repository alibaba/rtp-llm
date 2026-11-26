"""Read the interleaved [head, Q/gate, channel] output of a grouped Qwen projection."""

from rtp_llm.models_py.read_spec import (
    LoadedCheckpointTensor,
    ReadSpec,
    unwrap_checkpoint_tensor,
)


def is_grouped_q_gate(cfg, name):
    prefix, _, field = name.rpartition(".")
    return (
        prefix.endswith(".self_attn.q_proj")
        and field in ("qweight", "qzeros", "scales", "g_idx", "bias")
        and getattr(cfg["quant_config"], "quant_type", None) in ("awq", "gptq")
    )


def _layout(cfg, *, outputs=1):
    from rtp_llm.models_py.quant_methods.groupwise import groupwise_settings
    from rtp_llm.models_py.quant_methods.groupwise_checkpoint import (
        GroupwiseCheckpointLayout,
    )

    method, bits, group = groupwise_settings(cfg["quant_config"])
    dim = cfg["head_dim"]
    if dim % (8 if method == "awq" else 32 // bits):
        # Never split a word or AWQ's output-channel permutation across heads.
        raise ValueError(
            f"{method} gated attention head_dim={dim} does not align to its packed output channels"
        )
    return GroupwiseCheckpointLayout(
        method, bits, group, cfg["hidden_size"], outputs * cfg["num_heads"] * dim
    )


def grouped_q_gate_read_spec(cfg, name, shape):
    fused = _layout(cfg, outputs=2)
    field = name.rsplit(".", 1)[-1]
    if field in ("g_idx", "bias"):
        # g_idx maps the INPUT channels of both outputs. A GPTQ all-zero bias
        # on this biasless layer must be validated on every rank in full.
        expected = fused.input_size if field == "g_idx" else fused.output_size
        if tuple(shape) != (expected,):
            raise ValueError(f"{name} expected ({expected},), got {shape}")
        return ReadSpec.full(name, tuple(shape))
    # Heads are contiguous in the physical OUTPUT axis (axis 1 for all
    # three matrix fields). Each rank reads its own heads, including BOTH
    # outputs. No unpack/repack, all-gather or duplicated full source.
    return fused.read_spec(name, shape, 1, cfg["attn_tp_size"], cfg["attn_tp_rank"])


def split_grouped_q_gate(cfg, name, value):
    prefix, field = name.rsplit(".", 1)
    parent = prefix.rsplit(".", 1)[0]
    tensor, spec = unwrap_checkpoint_tensor(value)
    original_shape = tuple(tensor.shape) if spec is None else spec.source_shape
    expected = grouped_q_gate_read_spec(cfg, name, original_shape)
    if spec is not None and spec.for_checkpoint_name(name) != expected:
        raise ValueError(
            f"{name}: packed Q/gate source belongs to another rank or layout"
        )
    if field == "g_idx":
        parts = (tensor, tensor)
    else:
        heads = (
            cfg["num_heads"]
            if spec is None
            else cfg["num_heads"] // cfg["attn_tp_size"]
        )
        # GPTQ qweight packs K, while AWQ qweight and both qzeros pack N.
        # Treating each physical column as an opaque unit preserves both.
        values = tensor.unflatten(-1, (heads, 2, -1))
        parts = tuple(p.flatten(-2).contiguous() for p in values.unbind(-2))
    layout = _layout(cfg) if spec is not None else None
    for dest, part in zip(
        (f"{parent}.qkv_proj.q_proj.{field}", f"{parent}.gate.{field}"), parts
    ):
        if spec is not None:
            logical_spec = layout.read_spec(
                dest, layout.shapes[field], 1, cfg["attn_tp_size"], cfg["attn_tp_rank"]
            )
            part = LoadedCheckpointTensor(part, logical_spec)
        yield dest, part
