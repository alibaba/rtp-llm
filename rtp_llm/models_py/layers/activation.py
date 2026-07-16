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
        try:
            output = torch.empty(
                gate_up.shape[:-1] + (gate_up.shape[-1] // 2,),
                dtype=gate_up.dtype,
                device=gate_up.device,
            )
            if getattr(torch.version, "hip", None) is not None:
                import aiter

                aiter.silu_and_mul(output, gate_up)
                return output

            from rtp_llm.ops.compute_ops import rtp_llm_ops

            stream_id = torch.cuda.current_stream().cuda_stream
            rtp_llm_ops.silu_and_mul(output, gate_up, stream_id)
            return output
        except Exception as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            _SILU_FUSED_ENABLED = False
            logger.warning(
                "Fused silu_and_mul is unavailable; disabling it and using eager "
                "fallback: %s",
                exc,
            )

    gate, up = gate_up.chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate.float()) * up.float()).to(gate_up.dtype)
