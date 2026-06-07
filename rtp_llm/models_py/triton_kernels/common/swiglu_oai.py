"""SwiGLU-OAI activation for MiniMax M3 / GPT-OSS style models.

Reference formula (from sglang
``layers/moe/moe_runner/triton_utils/fused_moe.py:341`` —
``swiglu_no_interleaved_with_alpha_and_limit``):

    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(min=None, max=limit)        # one-sided clamp
    up   = up.clamp(min=-limit, max=limit)        # two-sided clamp
    return gate * sigmoid(gate * alpha) * (up + 1)

Note the +1 on the up branch — this is the OAI variant; without it you get
ordinary SiLU * up.

Layout caveat
-------------
rtp-llm uses two different gate/up orderings depending on the pipeline:

* DenseMLP path: [gate | up] — gate first, up second (``gate_first=True``).
* MoE contiguous path: [up | gate] — up first, gate second (rtp-llm's
  ``stack_moe_w1`` packs ``[up_proj, gate_proj]``, so ``gate_first=False``).

Both paths run the OAI math through ``swiglu_oai_torch``.
"""

from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# Activation dispatch helpers (shared by the MXFP8 MoE executor)
# ---------------------------------------------------------------------------


def is_swiglu_oai(activation: str) -> bool:
    """SwiGLU-OAI dispatch: GPT-OSS / MiniMax-M3 variant
    ``gate * sigmoid(gate*alpha) * (up+1)`` with two-sided clamps.
    """
    return isinstance(activation, str) and activation.lower() == "swiglu_oai"


def swiglu_oai_alpha_limit(extra_expert_args) -> Tuple[float, float]:
    """Pull (alpha, limit) out of ``extra_expert_args`` with sharp errors."""
    if extra_expert_args is None:
        raise ValueError(
            "swiglu_oai activation requires extra_expert_args with "
            "swiglu_alpha and swiglu_limit"
        )
    try:
        return (
            float(extra_expert_args["swiglu_alpha"]),
            float(extra_expert_args["swiglu_limit"]),
        )
    except KeyError as e:
        raise ValueError(f"swiglu_oai missing required extra_expert_arg: {e!s}")


# ---------------------------------------------------------------------------
# Python reference (MoE path + bf16 fallback)
# ---------------------------------------------------------------------------


@torch.compile
def swiglu_oai_torch(
    x: torch.Tensor, alpha: float, limit: float, gate_first: bool = True
) -> torch.Tensor:
    """Pure-PyTorch SwiGLU-OAI reference.

    Matches sglang.swiglu_no_interleaved_with_alpha_and_limit byte-for-byte
    when ``gate_first=True`` (the [gate | up] layout used by DenseMLP and
    sglang). Set ``gate_first=False`` for the MoE [up | gate] layout.
    """
    if gate_first:
        gate, up = x.chunk(2, dim=-1)
    else:
        up, gate = x.chunk(2, dim=-1)
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(gate * alpha) * (up + 1)


__all__ = [
    "is_swiglu_oai",
    "swiglu_oai_alpha_limit",
    "swiglu_oai_torch",
]
