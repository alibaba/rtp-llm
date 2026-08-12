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
    if gate_up.is_cuda and _SILU_FUSED_ENABLED:
        output = torch.empty(
            gate_up.shape[:-1] + (gate_up.shape[-1] // 2,),
            dtype=gate_up.dtype,
            device=gate_up.device,
        )
        try:
            if getattr(torch.version, "hip", None) is not None:
                import aiter
                fused_op = aiter.silu_and_mul
            else:
                from rtp_llm.ops.compute_ops import rtp_llm_ops

                fused_op = rtp_llm_ops.silu_and_mul
        except (ImportError, AttributeError) as exc:
            _SILU_FUSED_ENABLED = False
            logger.warning(
                "Fused silu_and_mul is unavailable; disabling it and using eager "
                "fallback: %s",
                exc,
            )
        else:
            if getattr(torch.version, "hip", None) is not None:
                fused_op(output, gate_up)
            else:
                stream_id = torch.cuda.current_stream().cuda_stream
                fused_op(output, gate_up, stream_id)
            return output

    gate, up = gate_up.chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate.float()) * up.float()).to(gate_up.dtype)
