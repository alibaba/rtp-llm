"""Test-only output projection topology diagnostic for the official reference."""

from contextlib import contextmanager
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb
from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear


def tp8_output_projection(layer, output, freqs, out=None):
    weight = layer._wo_a_bf16_v41
    if weight is None or not isinstance(layer.wo_b, V41MXFP8Linear):
        raise ValueError("TP8 projection diagnostic requires the SM120 V4.1 owner")
    # Bounded fixtures have fewer groups; the full checkpoint uses eight.
    shards = min(8, layer.n_groups)
    if layer.n_groups % shards:
        raise ValueError("Output groups must divide evenly across diagnostic shards")
    if not hasattr(layer, "_diagnostic_wo_b_shards"):
        linear = layer.wo_b._sm120_linear
        if linear is None:
            raise ValueError("TP8 projection diagnostic requires SM120 dense kernels")
        layer._diagnostic_wo_b_shards = torch.nn.ModuleList(
            [
                V41MXFP8Linear(w.contiguous(), s.contiguous())
                for w, s in zip(
                    linear.weight.chunk(shards, dim=1),
                    linear.scales.view(torch.float8_e8m0fnu).chunk(shards, dim=1),
                )
            ]
        )
    output = output.reshape(-1, layer.n_heads, layer.head_dim)
    apply_rotary_emb(
        output[..., -layer.rope_head_dim :].unsqueeze(0), freqs, inverse=True
    )
    grouped = output.reshape(len(output), layer.n_groups, -1)
    partials = []
    for x, w, linear in zip(
        grouped.chunk(shards, dim=1),
        weight.chunk(shards, dim=0),
        layer._diagnostic_wo_b_shards,
    ):
        projected = torch.einsum("tgd,grd->tgr", x, w)
        partials.append(linear(projected.flatten(1)).float())
    result = torch.stack(partials).sum(0).to(output.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


@contextmanager
def tp8_projection(enabled):
    if enabled:
        with patch.object(AttentionV41FP8, "_project_output", tp8_output_projection):
            yield
    else:
        yield
