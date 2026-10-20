# SPDX-License-Identifier: Apache-2.0
"""Default DeepGEMM router for the compatible MegaMoE gate-pack path.

DSV4_MEGA_GATE defaults to 1 for prefill and decode at all supported token
counts; set it to 0 to restore the original router. The vLLM-style BF16 GEMM
+ FP32 scoring + TopK kernel does not round scores to BF16 before selection,
so near-tie routes can change. This path requires finite inputs/weights:
DeepGEMM traps on invalid scores instead of RTP's nonfinite fallback.
Hash, visual, CED and debug routing retain the old path.
API and output ownership follow vllm/utils/deep_gemm.py:bf16_mega_gate.
"""

import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _native_gate():
    try:
        import deep_gemm
    except ImportError:
        return None
    return getattr(deep_gemm, "bf16_mega_gate", None)


def can_use_mega_gate(x: torch.Tensor, gate) -> bool:
    # Called only in MoE's gate-pack path, after its visual-token guard.
    if os.environ.get("DSV4_MEGA_GATE", "1") != "1":
        return False
    if (
        x.ndim != 2
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or not 0 < x.shape[0] <= (1 << 20)
        or not x.is_contiguous()
    ):
        return False
    if (
        os.environ.get("DSV4_GATE_FP32", "0") == "1"
        or os.environ.get("DSV4_GATE_FUSED", "1") == "0"
        or os.environ.get("MOEDBG", "0") != "0"
        or gate.hash
        or gate.score_func != "sqrtsoftplus"
        or getattr(gate, "_ced_row_projection", None) is not None
        or getattr(gate, "_dbg_prefix", None) is not None
        or getattr(gate, "_has_visual_tokens", False)
    ):
        return False
    weight, bias = gate.weight, gate.bias
    if weight.ndim != 2 or bias is None:
        return False
    experts, hidden = weight.shape
    # Keep DeepGEMM's shape contract for compatible inputs and routers.
    return (
        x.shape[1] == hidden
        and hidden > 0
        and hidden % 256 == 0
        and 0 < experts <= 512
        and experts % 4 == 0
        and 0 < gate.topk <= min(experts, 32)
        and weight.dtype in (torch.bfloat16, torch.float32)
        and bias.dtype == torch.float32
        and bias.shape == (experts,)
        and x.device == weight.device == bias.device
        and weight.is_contiguous()
        and bias.is_contiguous()
        and torch.cuda.get_device_capability(x.device)[0] == 10
        and _native_gate() is not None
    )


def mega_gate(x: torch.Tensor, gate, ep_rank: int = 0, *, out=None):
    """Return (weights, global IDs); optional ``out`` is (IDs, weights)."""
    kernel = _native_gate()
    if kernel is None:
        raise RuntimeError("DeepGEMM does not provide bf16_mega_gate")
    if out is None:
        indices = torch.empty(
            (x.shape[0], gate.topk), dtype=torch.int64, device=x.device
        )
        weights = torch.empty(
            (x.shape[0], gate.topk), dtype=torch.float32, device=x.device
        )
    else:
        indices, weights = out
    kernel(
        x,
        gate._weight_bf16(),
        gate.topk,
        use_shared_as_routed=False,
        num_shared_experts=0,
        routed_scaling_factor=float(gate.route_scale),
        ep_rank=ep_rank,
        scoring_func=gate.score_func,
        bias=gate.bias,
        out=(indices, weights),
    )
    return weights, indices
