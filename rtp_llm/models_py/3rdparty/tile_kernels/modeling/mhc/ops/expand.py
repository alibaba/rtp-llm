import torch

from ....mhc.expand_kernel import expand_to_mhc_bwd_tl, expand_to_mhc_fwd_tl


class ExpandToMHCFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: "ExpandToMHCFn",
        hidden: torch.Tensor,
        mhc_mult: int,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        if out is None:
            out = hidden.new_empty(*hidden.shape[:-1], mhc_mult, hidden.shape[-1])
        assert hidden.is_contiguous()
        kernel = expand_to_mhc_fwd_tl(hidden.shape[-1], mhc_mult)
        kernel(hidden.flatten(0, -2), out.flatten(0, -3))
        return out

    @staticmethod
    def backward(ctx: "ExpandToMHCFn", out_grad: torch.Tensor) -> torch.Tensor:
        hidden_grad = out_grad.new_empty(*out_grad.shape[:-2], out_grad.shape[-1])
        kernel = expand_to_mhc_bwd_tl(out_grad.shape[-1], out_grad.shape[-2])
        kernel(out_grad.flatten(0, -3), hidden_grad.flatten(0, -2))
        return hidden_grad, None, None


def expand_to_mhc(
    hidden: torch.Tensor,
    mhc_mult: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return ExpandToMHCFn.apply(hidden, mhc_mult, out)
