from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

_RMSNORM_FALLBACK_WARNED = False


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=params_dtype), requires_grad=False
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        for name, tensor in weights.items():
            if "weight" in name:
                self.weight.data.copy_(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 默认走 rtp 融合 rmsnorm（与旧 loader modules.RMSNorm 一致）；
        # CUDA/ROCm 上 is_cuda 均为 True，rtp_llm_ops 按 arch 编译为对应后端 kernel。
        # 非 CUDA / kernel 不可用时回退 eager fp32（可移植）。
        if x.is_cuda:
            try:
                if getattr(torch.version, "hip", None) is not None:
                    from aiter import rms_norm

                    return rms_norm(x, self.weight.data, self.eps)

                from rtp_llm.ops.compute_ops import rtp_llm_ops

                orig_shape = x.shape
                # 融合 kernel 按最后一维归一；reshape 成 2D 以兼容 q_norm/k_norm 的
                # [tokens, heads, head_dim] 三维输入。
                x2d = x.reshape(-1, orig_shape[-1]).contiguous()
                out = torch.empty_like(x2d)
                stream_id = torch.cuda.current_stream().cuda_stream
                rtp_llm_ops.rmsnorm(out, x2d, self.weight.data, self.eps, stream_id)
                return out.reshape(orig_shape)
            except Exception as e:  # 失败回退 eager，且告警一次
                global _RMSNORM_FALLBACK_WARNED
                if not _RMSNORM_FALLBACK_WARNED:
                    _RMSNORM_FALLBACK_WARNED = True
                    import logging

                    logging.getLogger(__name__).warning(
                        "[RMSNorm] 融合 rmsnorm 不可用，回退 eager fp32: %s", e
                    )
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class RMSResNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=params_dtype), requires_grad=False
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        for name, tensor in weights.items():
            if "weight" in name:
                self.weight.data.copy_(tensor)

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.is_cuda:
            try:
                if getattr(torch.version, "hip", None) is not None:
                    from aiter import rmsnorm2d_fwd_with_add as fused_add_rmsnorm

                    output = torch.empty_like(hidden_states)
                    residual_out = torch.empty_like(hidden_states)
                    fused_add_rmsnorm(
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

                orig_shape = hidden_states.shape
                hidden_2d = hidden_states.reshape(-1, orig_shape[-1]).contiguous()
                residual_2d = residual.reshape(-1, orig_shape[-1]).contiguous()
                stream_id = torch.cuda.current_stream().cuda_stream
                rtp_llm_ops.fused_add_rmsnorm(
                    hidden_2d, residual_2d, self.weight.data, self.eps, stream_id
                )
                return hidden_2d.reshape(orig_shape), residual_2d.reshape(orig_shape)
            except Exception as e:
                global _RMSNORM_FALLBACK_WARNED
                if not _RMSNORM_FALLBACK_WARNED:
                    _RMSNORM_FALLBACK_WARNED = True
                    import logging

                    logging.getLogger(__name__).warning(
                        "[RMSResNorm] 融合 fused_add_rmsnorm 不可用，回退 eager fp32: %s",
                        e,
                    )

        residual = hidden_states + residual
        input_dtype = residual.dtype
        x = residual.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype), residual


class LayerNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        params_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=params_dtype), requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.zeros(hidden_size, dtype=params_dtype), requires_grad=False
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        for name, tensor in weights.items():
            if "weight" in name or "gamma" in name:
                self.weight.data.copy_(tensor)
            elif "bias" in name or "beta" in name:
                self.bias.data.copy_(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            x, [self.hidden_size], self.weight, self.bias, self.eps
        )
