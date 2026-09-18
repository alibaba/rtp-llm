from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def grouped_rms_norm(
    x: torch.Tensor, gamma: torch.Tensor, group_size: int, eps: float
) -> torch.Tensor:
    """RMS taken independently per ``group_size``-wide group, one shared gamma.

    ``gamma`` is the raw checkpoint tensor, which is zero-centred: upstream
    RMSNorm is ``x * (1.0 + weight)``. The ``+1`` is applied here, in fp32 after
    the cast, because that is the order upstream uses -- folding it into the
    stored weight in bf16 instead would round the gain by up to ~4e-3 relative
    and cost bit-exactness. So these norms deliberately do NOT take the loader's
    ``plus_one``, unlike the rest of the repo's RMSNorms.
    """
    out = x.float().unflatten(-1, (x.shape[-1] // group_size, group_size))
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    return (out.flatten(-2) * (1.0 + gamma.float())).type_as(x)


def inject_into_residual(
    hyper_input: torch.Tensor,
    sublayer_out: torch.Tensor,
    inject_weights: torch.Tensor,
) -> torch.Tensor:
    """Write a sublayer output back into every branch of the residual stream.

    ``hyper_input`` is ``[..., hc_mult * hidden]``, ``sublayer_out`` is
    ``[..., hidden]`` and ``inject_weights`` is ``[..., hc_mult]``.
    """
    injection = sublayer_out.unsqueeze(-2) * inject_weights.unsqueeze(-1)
    return hyper_input + injection.flatten(-2)


class Qwen4ExpGatedResidual(nn.Module):
    """Read side of the qwen4_exp gated residual stream.

    Collapses an ``[..., hc_mult * hidden]`` stream down to ``[..., hidden]`` for
    the next sublayer, and produces the per-branch scalar write gate that
    :func:`inject_into_residual` later applies to that sublayer's output.

    Shapes are inferred from the weights: ``mix_down`` is
    ``[lowrank, hc_mult * hidden]``, ``mix_up`` is ``[hc_mult * hidden, lowrank]``
    and ``inject`` is ``[hc_mult, hc_mult * hidden]``. The global mixer is built
    without ``inject``.

    ``norm_gamma`` is the raw checkpoint tensor; the ``+1`` that upstream RMSNorm
    applies is added inside :func:`grouped_rms_norm`, not folded in at load time.
    """

    def __init__(
        self,
        norm_gamma: torch.Tensor,
        mix_down: torch.Tensor,
        mix_up: torch.Tensor,
        inject: Optional[torch.Tensor],
        *,
        hc_mult: int,
        norm_eps: float,
    ):
        super().__init__()
        hc_hidden = norm_gamma.shape[-1]
        if hc_hidden % hc_mult != 0:
            raise ValueError(
                f"gated residual width {hc_hidden} is not divisible by "
                f"hc_mult {hc_mult}"
            )
        if mix_down.shape[-1] != hc_hidden or mix_up.shape[-2] != hc_hidden:
            raise ValueError(
                f"mix weights {tuple(mix_down.shape)} / {tuple(mix_up.shape)} do not "
                f"match the {hc_hidden}-wide stream"
            )
        if mix_down.shape[-2] != mix_up.shape[-1]:
            raise ValueError(
                f"mix_down rank {mix_down.shape[-2]} != mix_up rank {mix_up.shape[-1]}"
            )
        if inject is not None and tuple(inject.shape) != (hc_mult, hc_hidden):
            raise ValueError(
                f"inject weight {tuple(inject.shape)} != {(hc_mult, hc_hidden)}"
            )

        self.hc_mult = hc_mult
        self.hc_hidden_size = hc_hidden
        self.hidden_size = hc_hidden // hc_mult
        self.norm_eps = norm_eps
        self.norm_gamma = norm_gamma
        self.mix_down = mix_down
        self.mix_up = mix_up
        self.inject = inject

    def _norm(self, hyper_input: torch.Tensor) -> torch.Tensor:
        return grouped_rms_norm(
            hyper_input, self.norm_gamma, self.hidden_size, self.norm_eps
        )

    def forward(
        self, hyper_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if hyper_input.shape[-1] != self.hc_hidden_size:
            raise ValueError(
                f"expected {self.hc_hidden_size} gated residual features, got "
                f"{hyper_input.shape[-1]}"
            )
        normed = self._norm(hyper_input)

        mix = F.silu(F.linear(normed, self.mix_down) / self.hc_mult)
        mix = torch.sigmoid(F.linear(mix, self.mix_up))
        branches = (self.hc_mult, self.hidden_size)
        mixed = (mix.unflatten(-1, branches) * normed.unflatten(-1, branches)).mean(-2)

        if self.inject is None:
            return mixed, hyper_input, None
        inject_weights = 2 * torch.sigmoid(F.linear(normed, self.inject) / self.hc_mult)
        return mixed, hyper_input, inject_weights
