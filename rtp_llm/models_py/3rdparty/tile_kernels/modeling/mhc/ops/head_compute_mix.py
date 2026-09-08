import torch

from ....config import get_num_sms
from ....mhc.head_compute_mix_kernel import (
    _mhc_head_compute_mix_bwd,
    _mhc_head_compute_mix_fwd,
)


class MHCHeadComputeMix(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: "MHCHeadComputeMix",
        input_mix: torch.Tensor,
        mhc_scale: torch.Tensor,
        mhc_base: torch.Tensor,
        mhc_pre_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert input_mix.ndim == 3
        ctx.tokens_shape = input_mix.shape[:2]
        mhc_mult = input_mix.shape[-1]

        output_mix = torch.empty_like(input_mix)

        fwd_kernel = _mhc_head_compute_mix_fwd(
            mhc_mult, mhc_pre_eps, token_block_size=32
        )
        fwd_kernel(
            input_mix.view(-1, mhc_mult),
            mhc_scale,
            mhc_base,
            output_mix.view(-1, mhc_mult),
        )

        ctx.save_for_backward(input_mix, mhc_scale, mhc_base)
        return output_mix.view_as(input_mix)

    @staticmethod
    def backward(
        ctx: "MHCHeadComputeMix",
        output_mix_grad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        input_mix, mhc_scale, mhc_base = ctx.saved_tensors

        num_sms = get_num_sms()
        input_mix_grad = torch.empty_like(input_mix)
        mhc_scale_grad_partial = torch.empty(
            num_sms,
            *mhc_scale.shape,
            dtype=mhc_scale.dtype,
            device=mhc_scale.device,
        )
        mhc_base_grad_partial = torch.empty(
            num_sms,
            *mhc_base.shape,
            dtype=mhc_base.dtype,
            device=mhc_base.device,
        )

        mhc_mult = input_mix.shape[-1]
        bwd_kernel = _mhc_head_compute_mix_bwd(
            mhc_mult,
            token_block_size=32,
            num_sms=num_sms,
        )
        bwd_kernel(
            output_mix_grad.view(-1, mhc_mult),
            input_mix.view(-1, mhc_mult),
            mhc_scale,
            mhc_base,
            input_mix_grad.view(-1, mhc_mult),
            mhc_scale_grad_partial,
            mhc_base_grad_partial,
        )
        mhc_scale_grad = mhc_scale_grad_partial.sum(0)
        mhc_base_grad = mhc_base_grad_partial.sum(0)

        return input_mix_grad, mhc_scale_grad, mhc_base_grad, None


def mhc_head_compute_mix(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_pre_eps: float,
) -> torch.Tensor:
    return MHCHeadComputeMix.apply(input_mix, mhc_scale, mhc_base, mhc_pre_eps)
