import logging

import torch

logger = logging.getLogger(__name__)
_SILU_FUSED_ENABLED = True


def silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply the fused SwiGLU activation to a ``[gate, up]`` tensor."""
    global _SILU_FUSED_ENABLED

    if gate_up.shape[-1] <= 0 or gate_up.shape[-1] % 2 != 0:
        raise ValueError(
            f"SwiGLU input width must be positive and even, got " f"{gate_up.shape[-1]}"
        )
    is_hip = torch.version.hip is not None
    if (
        gate_up.is_cuda
        and _SILU_FUSED_ENABLED
        and (is_hip or gate_up.dtype in (torch.float16, torch.bfloat16))
    ):
        try:
            if is_hip:
                import aiter

                fused_op = aiter.silu_and_mul
            else:
                from rtp_llm.models_py.modules.base.cuda.activation import (
                    FusedSiluAndMul,
                )

                # Share the serving kernel and its vector-alignment guard. The
                # CUDA compute-ops module does not export a plain silu_and_mul.
                return FusedSiluAndMul.apply_cuda(gate_up)
        except (ImportError, AttributeError) as exc:
            _SILU_FUSED_ENABLED = False
            logger.warning(
                "Fused silu_and_mul is unavailable; disabling it and using eager "
                "fallback: %s",
                exc,
            )
        else:
            output = gate_up.new_empty(gate_up.shape[:-1] + (gate_up.shape[-1] // 2,))
            fused_op(output, gate_up)
            return output

    gate, up = gate_up.chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate.float()) * up.float()).to(gate_up.dtype)
