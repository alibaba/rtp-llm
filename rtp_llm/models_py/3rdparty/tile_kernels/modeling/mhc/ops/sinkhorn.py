from contextlib import nullcontext
from functools import lru_cache

import torch

from ....mhc.sinkhorn_kernel import _mhc_sinkhorn_bwd, _mhc_sinkhorn_fwd


@lru_cache(maxsize=128)
def _inference_kernel(device: torch.device, hidden_size: int, repeat: int, eps: float):
    # TileLang already caches compiled kernels. Retain the returned callable as
    # well to avoid parsing its specialization arguments on every layer. Tokens
    # are dynamic; no tensors or request-sized storage are retained here.
    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
        return _mhc_sinkhorn_fwd(hidden_size, 1, repeat, eps)


class _SinkhornNormalize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: "_SinkhornNormalize",
        x: torch.Tensor,
        repeat: int,
        eps: float,
    ) -> torch.Tensor:
        hidden_size = x.shape[1]
        output = torch.empty_like(x)
        fwd_kernel = _mhc_sinkhorn_fwd(hidden_size, 1, repeat, eps)
        ctx.save_for_backward(x)
        ctx.backward_args = (hidden_size, 32, repeat, eps)
        fwd_kernel(x, output)
        return output

    @staticmethod
    def backward(
        ctx: "_SinkhornNormalize", grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        x = ctx.saved_tensors[0]
        grad_input = torch.empty_like(x)
        bwd_kernel = _mhc_sinkhorn_bwd(*ctx.backward_args)
        bwd_kernel(grad_output, x, grad_input)
        return grad_input, None, None


def sinkhorn_normalize(
    x: torch.Tensor, repeat: int = 10, eps: float = 1e-6
) -> torch.Tensor:
    if not torch.is_grad_enabled() and not torch.compiler.is_compiling():
        flattened = x.contiguous().view(-1, *x.shape[-2:])
        output = torch.empty_like(flattened)
        _inference_kernel(x.device, flattened.shape[1], repeat, eps)(flattened, output)
        return output.view_as(x)
    return _SinkhornNormalize.apply(
        x.contiguous().view(-1, *x.shape[-2:]), repeat, eps
    ).view_as(x)
