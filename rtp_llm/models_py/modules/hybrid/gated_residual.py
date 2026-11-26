from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.base.common.fp8_linear import (
    dense_projection,
    make_projection,
)


def grouped_rms_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    group_size: int,
    eps: float,
    *,
    batch_invariant: bool = False,
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
        from rtp_llm.models_py.triton_kernels.gated_residual.norm import (
            try_grouped_rms_norm,
        )

        result = try_grouped_rms_norm(
            x, gamma, group_size, eps, batch_invariant=batch_invariant
        )
        if result is not None:
            return result
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
        from rtp_llm.models_py.triton_kernels.gated_residual.inject import (
            try_fused_inject,
        )

        result = try_fused_inject(hyper_input, sublayer_out, inject_weights)
        if result is not None:
            return result
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
        batch_invariant: bool = False,
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
        self.batch_invariant = batch_invariant
        self.norm_gamma = norm_gamma
        self.mix_down = mix_down
        self.mix_up = mix_up
        self.inject = inject
        scales = weight_scales or (None, None, None)
        self.down_projection = make_projection(mix_down, scales[0], role=role)
        self.up_projection = make_projection(mix_up, scales[1], role=role)
        self.inject_projection = (
            make_projection(inject, scales[2], role=role)
            if inject is not None
            else None
        )
        self.merged_down_inject = None
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
            # In RL VMM mode the captured GEMM reads this final merged storage.
            # Move it before the Qwen4 model installs its views/consumers; the
            # ordinary path remains the same allocation and performs no copy.
            from rtp_llm.model_loader.rl_vmm import pack_if_rl_vmm

            self.merged_down_inject, vmm_manager = pack_if_rl_vmm(
                self.merged_down_inject, mix_down
            )
            if vmm_manager is not None:
                tail_start = int(mix_down.shape[0]) + int(hc_mult)
                if tail_start < self.merged_down_inject.shape[0]:
                    vmm_manager.register_zero_on_remap(
                        self.merged_down_inject[tail_start:]
                    )

    def _norm(self, hyper_input: torch.Tensor) -> torch.Tensor:
        return grouped_rms_norm(
            hyper_input,
            self.norm_gamma,
            self.hidden_size,
            self.norm_eps,
            batch_invariant=self.batch_invariant,
        )

    def _project(self, input, weight, projection):
        if self.batch_invariant:
            from rtp_llm.models_py.triton_kernels.common.batch_invariant_linear import (
                batch_invariant_linear,
            )

            return batch_invariant_linear(input, weight)
        return dense_projection(input, weight, projection)

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
            and 8192 <= hyper_input.numel() // self.hc_hidden_size <= 12288
            and hyper_input.is_cuda
            and torch.version.hip is None
        ):
            from rtp_llm.models_py.triton_kernels.gated_residual.norm import (
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

    def _has_forward_hooks(self) -> bool:
        module_hooks = torch.nn.modules.module
        return bool(
            self._forward_pre_hooks
            or self._forward_hooks
            or module_hooks._global_forward_pre_hooks
            or module_hooks._global_forward_hooks
        )

    def forward_with_injection(
        self,
        hyper_input: torch.Tensor,
        sublayer_out: torch.Tensor,
        inject_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if (
            not self._has_forward_hooks()
            and self.down_projection is not None
            and self.down_projection.use_scaled_mm
            and tuple(self.down_projection.weight.shape) == (320, 10240)
        ):
            from rtp_llm.models_py.triton_kernels.gated_residual.prefill_inject_norm import (
                try_inject_grouped_norm,
            )

            fused = try_inject_grouped_norm(
                hyper_input,
                sublayer_out,
                inject_weights,
                self.norm_gamma,
                self.norm_eps,
            )
            if fused is not None:
                residual, normed, scale = fused
                return self._forward_normed(residual, normed, down_input_scale=scale)
        return self(inject_into_residual(hyper_input, sublayer_out, inject_weights))

    def _forward_normed(
        self, hyper_input: torch.Tensor, normed: torch.Tensor, *, down_input_scale=None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        inject_weights = None
        merged = self.merged_down_inject
        if (
            merged is not None
            and normed.is_contiguous()
            and normed.dtype == merged.dtype
            and normed.device == merged.device
            and (
                self.batch_invariant or 1 <= normed.numel() // self.hc_hidden_size <= 32
            )
        ):
            from rtp_llm.models_py.triton_kernels.gated_residual.down import (
                down_inject_epilogue,
            )

            mix, inject_weights = down_inject_epilogue(
                self._project(normed, merged, None),
                self.mix_down.shape[0],
                self.hc_mult,
            )
        else:
            down = (
                self._project(normed, self.mix_down, self.down_projection)
                if down_input_scale is None
                else self.down_projection(normed, input_scale=down_input_scale)
            )
            fused = None
            if (
                not self.batch_invariant
                and self.inject_projection is not None
                and self.inject_projection.compute_weight is not None
            ):
                from rtp_llm.models_py.triton_kernels.gated_residual.splitk_inject import (
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
                from rtp_llm.models_py.triton_kernels.gated_residual.down import (
                    separate_down_inject_epilogue,
                )

                injection = (
                    self._project(normed, self.inject, self.inject_projection)
                    if self.inject is not None
                    else None
                )
                if self.batch_invariant and injection is not None:
                    from rtp_llm.models_py.triton_kernels.gated_residual.down import (
                        down_inject_epilogue,
                    )

                    fused = down_inject_epilogue(
                        torch.cat((down, injection), dim=-1),
                        self.mix_down.shape[0],
                        self.hc_mult,
                    )
                else:
                    fused = separate_down_inject_epilogue(down, injection)
            if fused is None:
                mix = F.silu(down / self.hc_mult)
            else:
                mix, inject_weights = fused
        mixed = None
        up = self.up_projection
        if up is not None and up.bias is None and up.role in ("DECODE", "PDFUSION"):
            from rtp_llm.models_py.triton_kernels.gated_residual.up_mix import (
                try_fp8_up_mix,
            )

            mixed = try_fp8_up_mix(mix, up.weight, up.scale, normed, self.hc_mult)
        if (
            mixed is None
            and up is not None
            and up.bias is None
            and up.role in ("PREFILL", "DECODE", "PDFUSION")
        ):
            from rtp_llm.models_py.triton_kernels.gated_residual.prefill_up_mix import (
                try_fp8_prefill_up_mix,
            )

            mixed = try_fp8_prefill_up_mix(
                mix, up.weight, up.scale, normed, self.hc_mult
            )
        if mixed is None:
            mix_logits = self._project(mix, self.mix_up, self.up_projection)
            if normed.is_cuda and torch.version.hip is None:
                from rtp_llm.models_py.triton_kernels.gated_residual.mix import (
                    try_fused_mix_reduce,
                )

                mixed = try_fused_mix_reduce(mix_logits, normed, self.hc_mult)
            if mixed is None:
                branches = (self.hc_mult, self.hidden_size)
                mix = torch.sigmoid(mix_logits)
                mixed = (
                    mix.unflatten(-1, branches) * normed.unflatten(-1, branches)
                ).mean(-2)

        if self.inject is None:
            return mixed, hyper_input, None
        if inject_weights is None:
            inject_weights = 2 * torch.sigmoid(
                self._project(normed, self.inject, self.inject_projection)
                / self.hc_mult
            )
        return mixed, hyper_input, inject_weights
