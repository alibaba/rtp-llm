import torch

from ....config import get_num_sms
from ....mhc.pre_split_mixes_kernel import (
    _mhc_pre_split_mixes_bwd,
    _mhc_pre_split_mixes_fwd,
)


class MHCPreSplitMixes(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: "MHCPreSplitMixes",
        input_mixes: torch.Tensor,
        mhc_scale: torch.Tensor,
        mhc_base: torch.Tensor,
        mhc_mult: int,
        mhc_post_mult_value: float,
        mhc_pre_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mhc_mult2 = mhc_mult * mhc_mult
        mhc_mult3 = mhc_mult * 2 + mhc_mult2

        ctx.mhc_scale_main_grad = getattr(mhc_scale, "main_grad", None)
        ctx.mhc_base_main_grad = getattr(mhc_base, "main_grad", None)

        assert input_mixes.ndim == 3
        ctx.tokens_shape = input_mixes.shape[:2]

        input_mixes = input_mixes.view(-1, mhc_mult3)
        num_tokens = input_mixes.shape[0]
        pre_layer_mix = input_mixes.new_empty(num_tokens, mhc_mult)
        post_layer_mix = input_mixes.new_empty(num_tokens, mhc_mult)
        comb_res_mix = input_mixes.new_empty(num_tokens, mhc_mult2)

        ctx.fwd_kernel = _mhc_pre_split_mixes_fwd(
            mhc_mult,
            mhc_post_mult_value,
            mhc_pre_eps,
            token_block_size=32,
        )
        ctx.bwd_kernel = _mhc_pre_split_mixes_bwd(
            mhc_mult,
            mhc_post_mult_value,
            token_block_size=32,
            num_sms=get_num_sms(),
        )
        ctx.num_sms = get_num_sms()

        ctx.fwd_kernel(
            input_mixes,
            mhc_scale,
            mhc_base,
            pre_layer_mix,
            post_layer_mix,
            comb_res_mix,
        )

        ctx.save_for_backward(
            input_mixes, pre_layer_mix, post_layer_mix, mhc_scale, mhc_base
        )
        ctx.mhc_mult = mhc_mult

        pre_layer_mix = pre_layer_mix.view(*ctx.tokens_shape, mhc_mult, 1)
        post_layer_mix = post_layer_mix.view(*ctx.tokens_shape, mhc_mult, 1)
        comb_res_mix = comb_res_mix.view(*ctx.tokens_shape, mhc_mult, mhc_mult)

        return pre_layer_mix, post_layer_mix, comb_res_mix

    @staticmethod
    def backward(
        ctx: "MHCPreSplitMixes",
        pre_layer_mix_grad: torch.Tensor,
        post_layer_mix_grad: torch.Tensor,
        comb_res_mix_grad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        input_mixes, _pre_layer_mix, post_layer_mix, mhc_scale, mhc_base = (
            ctx.saved_tensors
        )

        input_mixes_grad = torch.empty_like(input_mixes)

        mhc_scale_grad_partial = torch.empty(
            ctx.num_sms,
            *mhc_scale.shape,
            dtype=mhc_scale.dtype,
            device=mhc_scale.device,
        )
        mhc_base_grad_partial = torch.empty(
            ctx.num_sms,
            *mhc_base.shape,
            dtype=mhc_base.dtype,
            device=mhc_base.device,
        )

        num_tokens = input_mixes.shape[0]
        mhc_mult = ctx.mhc_mult
        ctx.bwd_kernel(
            # Gradient of output
            pre_layer_mix_grad.view(num_tokens, mhc_mult),
            post_layer_mix_grad.view(num_tokens, mhc_mult),
            comb_res_mix_grad.view(num_tokens, mhc_mult * mhc_mult),
            # Cached activation
            input_mixes,
            post_layer_mix,
            mhc_scale,
            mhc_base,
            # Gradient of input
            input_mixes_grad,
            mhc_scale_grad_partial,
            mhc_base_grad_partial,
        )

        input_mixes_grad = input_mixes_grad.view(
            *ctx.tokens_shape, input_mixes_grad.shape[-1]
        )
        mhc_scale_grad = mhc_scale_grad_partial.sum(0)
        mhc_base_grad = mhc_base_grad_partial.sum(0)

        return input_mixes_grad, mhc_scale_grad, mhc_base_grad, None, None, None


def mhc_pre_split_mixes(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return MHCPreSplitMixes.apply(
        input_mixes,
        mhc_scale,
        mhc_base,
        mhc_mult,
        mhc_post_mult_value,
        mhc_pre_eps,
    )
