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
    is_rocm = torch.version.hip is not None
    # The pinned FlashInfer kernel requires 16-byte aligned half rows. Preserve
    # eager support for widths such as the Qwen2.5-VL intermediate size 3420.
    aligned = (gate_up.shape[-1] // 2 * gate_up.element_size()) % 16 == 0
    if gate_up.is_cuda and _SILU_FUSED_ENABLED and (is_rocm or aligned):
        output = torch.empty(
            gate_up.shape[:-1] + (gate_up.shape[-1] // 2,),
            dtype=gate_up.dtype,
            device=gate_up.device,
        )
        try:
            if is_rocm:
                import aiter

                fused_op = aiter.silu_and_mul
            else:
                from flashinfer.activation import silu_and_mul as fused_op
        except (ImportError, AttributeError, OSError) as exc:
            _SILU_FUSED_ENABLED = False
            logger.warning(
                "Fused silu_and_mul is unavailable; disabling it and using eager "
                "fallback: %s",
                exc,
            )
        else:
            if is_rocm:
                fused_op(output, gate_up)
            else:
                # The caller may keep a different CUDA device current while
                # loading or executing multiple model replicas. FlashInfer must
                # obtain the current stream in the input device context.
                with torch.cuda.device(gate_up.device):
                    fused_op(gate_up, out=output)
            return output

    gate, up = gate_up.chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate.float()) * up.float()).to(gate_up.dtype)
