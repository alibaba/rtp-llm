import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
from rtp_llm.models_py.module_base import RtpModule

logger = logging.getLogger(__name__)
_RMSNORM_FUSED_ENABLED = True


def _disable_fused_rmsnorm(exc: ImportError) -> None:
    global _RMSNORM_FUSED_ENABLED

    _RMSNORM_FUSED_ENABLED = False
    logger.error(
        "Fused RMSNorm backend import failed (%s); disabling fused RMSNorm for "
        "this process and using eager fallback: %s",
        type(exc).__name__,
        exc,
        exc_info=(type(exc), exc, exc.__traceback__),
    )


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
        if x.ndim == 0 or x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"RMSNorm expected last dimension {self.hidden_size}, "
                f"got shape {tuple(x.shape)}"
            )
        if self.weight.device != x.device:
            raise ValueError("RMSNorm weight and input must share a device")
        if self.weight.dtype != x.dtype:
            raise TypeError("RMSNorm weight and input must share a dtype")
        if x.is_cuda and _RMSNORM_FUSED_ENABLED:
            original_shape = x.shape
            input_2d = x.reshape(-1, original_shape[-1]).contiguous()
            if getattr(torch.version, "hip", None) is not None:
                try:
                    from aiter import rms_norm
                except ImportError as exc:
                    _disable_fused_rmsnorm(exc)
                else:
                    output = rms_norm(input_2d, self.weight.data, self.eps)
                    return output.reshape(original_shape)
            else:
                try:
                    from rtp_llm.ops.compute_ops import rtp_llm_ops
                except ImportError as exc:
                    _disable_fused_rmsnorm(exc)
                else:
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

        input_dtype = x.dtype
        fp32 = x.float()
        variance = fp32.pow(2).mean(-1, keepdim=True)
        normalized = fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight * normalized).to(input_dtype)


class RMSResNorm(RtpModule):
    """RMSNorm with the residual-add contract used by decoder runtimes."""

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
                raise RuntimeError(f"Unsupported RMSResNorm tensor {name!r}")
            if not self._assign_weight(self, "weight", tensor):
                raise RuntimeError("Failed to assign RMSResNorm weight")

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape != residual.shape:
            raise ValueError(
                "RMSResNorm hidden/residual shape mismatch: "
                f"{tuple(hidden_states.shape)} vs {tuple(residual.shape)}"
            )
        if hidden_states.device != residual.device:
            raise ValueError(
                "RMSResNorm hidden states and residual must share a device"
            )
        if hidden_states.dtype != residual.dtype:
            raise TypeError("RMSResNorm hidden states and residual must share a dtype")
        if self.weight.device != hidden_states.device:
            raise ValueError("RMSResNorm weight and inputs must share a device")
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                f"RMSResNorm expected [tokens, {self.hidden_size}], "
                f"got shape {tuple(hidden_states.shape)}"
            )
        if self.weight.dtype != hidden_states.dtype:
            raise TypeError("RMSResNorm weight and inputs must share a dtype")

        if hidden_states.is_cuda:
            if getattr(torch.version, "hip", None) is not None:
                from aiter import rmsnorm2d_fwd_with_add

                output = torch.empty_like(hidden_states)
                residual_out = torch.empty_like(residual)
                rmsnorm2d_fwd_with_add(
                    output,
                    hidden_states,
                    residual,
                    residual_out,
                    self.weight.data,
                    self.eps,
                    0,
                )
                return output, residual_out

            from rtp_llm.ops.compute_ops import rtp_llm_ops

            stream_id = torch.cuda.current_stream().cuda_stream
            rtp_llm_ops.fused_add_rmsnorm(
                hidden_states,
                residual,
                self.weight.data,
                self.eps,
                stream_id,
            )
            return hidden_states, residual

        residual_out = hidden_states + residual
        input_dtype = residual_out.dtype
        fp32 = residual_out.float()
        variance = fp32.pow(2).mean(-1, keepdim=True)
        normalized = fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight * normalized).to(input_dtype), residual_out
