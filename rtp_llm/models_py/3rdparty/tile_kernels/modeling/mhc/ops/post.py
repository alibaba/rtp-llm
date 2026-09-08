import torch

from ....mhc.post_kernel import mhc_post_bwd, mhc_post_fwd


class MHCPost(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: "MHCPost",
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        out = mhc_post_fwd(x, residual, post_layer_mix, comb_res_mix, out)
        ctx.save_for_backward(x, residual, post_layer_mix, comb_res_mix)
        return out

    @staticmethod
    def backward(
        ctx: "MHCPost",
        d_o: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        return *mhc_post_bwd(*ctx.saved_tensors, d_o), None


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return MHCPost.apply(x, residual, post_layer_mix, comb_res_mix, out)
