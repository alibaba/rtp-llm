import torch

from ....mhc.head_fuse_kernel import _mhc_head_fuse


def mhc_head_fuse(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    *,
    rms_eps: float,
    mhc_pre_eps: float,
) -> torch.Tensor:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_hidden_size = mhc_mult * hidden_size

    assert fn.shape == (mhc_mult, mhc_hidden_size)
    assert mhc_scale.shape == (1,)
    assert mhc_base.shape == (mhc_mult,)

    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    out = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    _mhc_head_fuse(
        hidden_size,
        rms_eps,
        mhc_pre_eps,
        mhc_mult=mhc_mult,
    )(
        residual_flat,
        fn,
        mhc_scale,
        mhc_base,
        out,
    )
    return out.view(*outer_shape, hidden_size)
