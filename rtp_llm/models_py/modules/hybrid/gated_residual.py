from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules import module as module_hooks

from rtp_llm.models_py.modules.base.common.fp8_linear import (
    dense_projection,
    make_projection,
)


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
    if x.is_cuda and torch.version.hip is None:
        from rtp_llm.models_py.triton_kernels.common.gated_residual_norm_triton import (
            grouped_rms_norm_triton,
            is_supported,
        )

        if is_supported(x, gamma, group_size):
            return grouped_rms_norm_triton(x, gamma, group_size, eps)
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
    if hyper_input.is_cuda and torch.version.hip is None:
        from rtp_llm.models_py.triton_kernels.common.gated_residual_inject_triton import (
            fused_inject,
            is_supported,
        )

        if is_supported(hyper_input, sublayer_out, inject_weights):
            return fused_inject(hyper_input, sublayer_out, inject_weights)
    injection = sublayer_out.unsqueeze(-2) * inject_weights.unsqueeze(-1)
    return hyper_input + injection.flatten(-2)


class GatedResidual(nn.Module):
    """Read side of a multi-branch gated residual stream.

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
        weight_scales=None,
        role="PDFUSION",
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
        scales = weight_scales or (None, None, None)
        self.down_projection = make_projection(mix_down, scales[0], role=role)
        self.up_projection = make_projection(mix_up, scales[1], role=role)
        self.bf16_fp8_up = None
        if (
            self.up_projection is not None
            and self.up_projection.use_scaled_mm
            and self.up_projection.bias is None
            and tuple(mix_up.shape) == (10240, 320)
            and mix_up.dtype == torch.float8_e4m3fn
            and mix_up.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(mix_up.device) == (9, 0)
        ):
            # Prepare before graph capture; preserve the original FP8 weight.
            self.bf16_fp8_up = (
                self.up_projection.weight.float() * self.up_projection.scale
            ).to(torch.bfloat16).contiguous()
            # A live update cannot refresh this precomputed graph input safely.
            mix_up._rtp_requires_restart_for_update = True
            scales[1]._rtp_requires_restart_for_update = True
        self.inject_projection = (
            make_projection(inject, scales[2], role=role)
            if inject is not None
            else None
        )
        self.merged_down_inject = None
        self.bf16_fp8_down_inject = None
        if (
            inject is not None
            and hc_mult == 4
            and mix_down.is_cuda
            and torch.version.hip is None
            and mix_down.dtype == inject.dtype == torch.bfloat16
            and mix_down.device == inject.device
        ):
            # Build the read-only projection before any CUDA Graph capture.
            # Padding the output width gives cuBLAS an aligned GEMM shape.
            padding = (-(mix_down.shape[0] + hc_mult)) % 16
            self.merged_down_inject = torch.cat(
                (mix_down, inject, mix_down.new_zeros((padding, hc_hidden))), dim=0
            ).contiguous()

        # This path changes the down projection's numerical rounding.
        # Build the dequantized copy once during model initialization, never
        # during forward or CUDA Graph replay; unsupported cases fall back.
        if (
            inject is not None
            and hc_mult == 4
            and self.down_projection is not None
            and self.down_projection.use_scaled_mm
            and mix_down.dtype == torch.float8_e4m3fn
            and tuple(mix_down.shape) == (320, 10240)
            and mix_down.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(mix_down.device) == (9, 0)
        ):
            inject_bf16 = (
                self.inject_projection.compute_weight
                if self.inject_projection is not None
                else inject
            )
            if (
                inject_bf16 is not None
                and inject_bf16.dtype == torch.bfloat16
                and inject_bf16.device == mix_down.device
            ):
                down_bf16 = (
                    self.down_projection.weight.float() * self.down_projection.scale
                ).to(torch.bfloat16)
                padding = (-(down_bf16.shape[0] + hc_mult)) % 16
                self.bf16_fp8_down_inject = torch.cat(
                    (down_bf16, inject_bf16, down_bf16.new_zeros((padding, hc_hidden))),
                    dim=0,
                ).contiguous()
                for source in (mix_down, scales[0], inject, scales[2]):
                    if source is not None:
                        source._rtp_requires_restart_for_update = True

    def _use_bf16_down_inject(self, x: torch.Tensor) -> bool:
        """Select the prebuilt BF16 projection when its input is supported."""
        merged = self.bf16_fp8_down_inject
        return (
            merged is not None
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.device == merged.device
            and 1 <= x.numel() // self.hc_hidden_size <= 12288
        )

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
        if (
            self.down_projection is not None
            and self.down_projection.use_scaled_mm
            and tuple(self.down_projection.weight.shape) == (320, 10240)
            and not self._use_bf16_down_inject(hyper_input)
            and (
                1 <= hyper_input.numel() // self.hc_hidden_size <= 32
                or 8192 <= hyper_input.numel() // self.hc_hidden_size <= 12288
            )
            and hyper_input.is_cuda
            and torch.version.hip is None
        ):
            from rtp_llm.models_py.triton_kernels.common.gated_residual_norm_triton import (
                try_grouped_rms_norm_with_fp8_scale,
            )

            fused = try_grouped_rms_norm_with_fp8_scale(
                hyper_input, self.norm_gamma, self.hidden_size, self.norm_eps
            )
            if fused is not None:
                normed, scale = fused
                return self._forward_normed(hyper_input, normed, down_input_scale=scale)
        normed = self._norm(hyper_input)

        return self._forward_normed(hyper_input, normed)

    def forward_with_injection(
        self,
        hyper_input: torch.Tensor,
        sublayer_out: torch.Tensor,
        inject_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        fp8_down = (
            self.down_projection is not None
            and self.down_projection.use_scaled_mm
            and tuple(self.down_projection.weight.shape) == (320, 10240)
        )
        rows = hyper_input.numel() // self.hc_hidden_size
        bf16_candidate = (
            (
                (self.hc_hidden_size == 10240 and 4096 <= rows <= 32768)
                or (
                    self.down_projection is None
                    and self.merged_down_inject is not None
                )
            )
            and hyper_input.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(hyper_input.device) == (9, 0)
        )
        if (fp8_down or bf16_candidate) and not (
            self._forward_pre_hooks
            or self._forward_hooks
            or module_hooks._global_forward_pre_hooks
            or module_hooks._global_forward_hooks
        ):
            from rtp_llm.models_py.triton_kernels.common.gated_residual_prefill_inject_norm_triton import (
                try_inject_grouped_norm,
            )

            collect_scale = (
                fp8_down and rows <= 12288 and not self._use_bf16_down_inject(hyper_input)
            )
            fused = try_inject_grouped_norm(
                hyper_input,
                sublayer_out,
                inject_weights,
                self.norm_gamma,
                self.norm_eps,
                collect_scale=collect_scale,
            )
            if fused is not None:
                residual, normed, scale = fused
                return self._forward_normed(residual, normed, down_input_scale=scale)
        return self(inject_into_residual(hyper_input, sublayer_out, inject_weights))

    def _forward_normed(
        self, hyper_input: torch.Tensor, normed: torch.Tensor, *, down_input_scale=None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        inject_weights = None
        bf16_fp8_candidate = self._use_bf16_down_inject(normed)
        merged = self.bf16_fp8_down_inject if bf16_fp8_candidate else self.merged_down_inject
        rows = normed.numel() // self.hc_hidden_size
        # Wide BF16 inputs use separate projections to preserve the model's
        # down/inject rounding.
        if (
            merged is not None
            and normed.is_contiguous()
            and normed.dtype == merged.dtype
            and normed.device == merged.device
            and (1 <= rows <= 32 or bf16_fp8_candidate)
        ):
            from rtp_llm.models_py.triton_kernels.common.gated_residual_down_triton import (
                down_inject_epilogue,
            )

            mix, inject_weights = down_inject_epilogue(
                F.linear(normed, merged), self.mix_down.shape[0], self.hc_mult
            )
        else:
            down = (
                dense_projection(normed, self.mix_down, self.down_projection)
                if down_input_scale is None
                else self.down_projection(normed, input_scale=down_input_scale)
            )
            fused = None
            if (
                self.inject_projection is not None
                and self.inject_projection.compute_weight is not None
            ):
                from rtp_llm.models_py.triton_kernels.common.gated_residual_splitk_inject import (
                    try_splitk_down_inject,
                )

                fused = try_splitk_down_inject(
                    normed, self.inject_projection.compute_weight, down
                )
            if (
                fused is None
                and self.hc_mult == 4
                and down.is_cuda
                and torch.version.hip is None
            ):
                from rtp_llm.models_py.triton_kernels.common.gated_residual_down_triton import (
                    separate_down_inject_epilogue,
                )

                injection = (
                    dense_projection(normed, self.inject, self.inject_projection)
                    if self.inject is not None
                    else None
                )
                fused = separate_down_inject_epilogue(down, injection)
            if fused is None:
                mix = F.silu(down / self.hc_mult)
            else:
                mix, inject_weights = fused
        mixed = None
        up = self.up_projection
        bf16_up = self.bf16_fp8_up
        use_bf16_up = (
            bf16_up is not None
            and mix.dtype == torch.bfloat16
            and mix.device == bf16_up.device
        )
        if (
            not use_bf16_up
            and up is not None
            and up.bias is None
            and up.role in ("DECODE", "PDFUSION")
        ):
            from rtp_llm.models_py.triton_kernels.common.gated_residual_up_mix_triton import (
                try_fp8_up_mix,
            )

            mixed = try_fp8_up_mix(mix, up.weight, up.scale, normed, self.hc_mult)
        if (
            mixed is None
            and not use_bf16_up
            and up is not None
            and up.bias is None
            and up.role in ("PREFILL", "DECODE", "PDFUSION")
        ):
            from rtp_llm.models_py.triton_kernels.common.gated_residual_prefill_up_mix_triton import (
                try_fp8_prefill_up_mix,
            )

            mixed = try_fp8_prefill_up_mix(
                mix, up.weight, up.scale, normed, self.hc_mult
            )
        if mixed is None:
            mix_logits = (
                F.linear(mix, bf16_up)
                if use_bf16_up
                else dense_projection(mix, self.mix_up, self.up_projection)
            )
            branches = (self.hc_mult, self.hidden_size)
            fused_mix = False
            if normed.is_cuda and torch.version.hip is None:
                from rtp_llm.models_py.triton_kernels.common.gated_residual_mix_triton import (
                    fused_mix_reduce,
                    is_supported,
                )

                fused_mix = is_supported(mix_logits, normed, self.hc_mult)
            if fused_mix:
                mixed = fused_mix_reduce(mix_logits, normed, self.hc_mult)
            else:
                mix = torch.sigmoid(mix_logits)
                mixed = (
                    mix.unflatten(-1, branches) * normed.unflatten(-1, branches)
                ).mean(-2)

        if self.inject is None:
            return mixed, hyper_input, None
        if inject_weights is None:
            inject_weights = 2 * torch.sigmoid(
                dense_projection(normed, self.inject, self.inject_projection)
                / self.hc_mult
            )
        return mixed, hyper_input, inject_weights
