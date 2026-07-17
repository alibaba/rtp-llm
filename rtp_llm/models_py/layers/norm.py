import logging
from typing import Dict

import torch
import torch.nn as nn

from rtp_llm.models_py.module_base import RtpModule

logger = logging.getLogger(__name__)
_RMSNORM_FUSED_ENABLED = True


class RMSNorm(RtpModule):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=params_dtype), requires_grad=False
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for name, tensor in weights.items():
            if name != "weight":
                raise RuntimeError(f"Unsupported RMSNorm tensor {name!r}")
            if not self._assign_weight(self, "weight", tensor):
                raise RuntimeError("Failed to assign RMSNorm weight")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global _RMSNORM_FUSED_ENABLED

        if x.ndim == 0 or x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"RMSNorm expected last dimension {self.hidden_size}, "
                f"got shape {tuple(x.shape)}"
            )
        if x.is_cuda and _RMSNORM_FUSED_ENABLED:
            try:
                original_shape = x.shape
                input_2d = x.reshape(-1, original_shape[-1]).contiguous()
                if getattr(torch.version, "hip", None) is not None:
                    from aiter import rms_norm

                    output = rms_norm(input_2d, self.weight.data, self.eps)
                else:
                    from rtp_llm.ops.compute_ops import rtp_llm_ops

                    output = torch.empty_like(input_2d)
                    stream_id = torch.cuda.current_stream().cuda_stream
                    rtp_llm_ops.rmsnorm(
                        output,
                        input_2d,
                        self.weight.data,
                        self.eps,
                        stream_id,
                    )
                return output.reshape(original_shape)
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
                _RMSNORM_FUSED_ENABLED = False
                logger.warning(
                    "Fused RMSNorm is unavailable; disabling it and using eager "
                    "fallback: %s",
                    exc,
                )

        input_dtype = x.dtype
        fp32 = x.float()
        variance = fp32.pow(2).mean(-1, keepdim=True)
        normalized = fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight * normalized).to(input_dtype)
