"""Independent Hyper-Connections used by HY V4.

HY V4 iHC has four residual channels and only pre/post gates.  It is not the
DeepSeek-V4 mHC transform: there is no 4x4 Sinkhorn/combination matrix.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.hy_v4.ihc_triton import (
    maybe_fused_ihc_head,
    maybe_fused_ihc_post,
    maybe_fused_ihc_pre,
    maybe_fused_ihc_pre_normed,
    maybe_fused_ihc_pre_normed_grouped,
)
from rtp_llm.utils.model_weight import W


def _require_fp32(weights: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in weights:
        raise KeyError(f"missing HY V4 iHC weight: {key}")
    weight = weights[key]
    if weight.dtype != torch.float32:
        raise TypeError(f"HY V4 iHC weight {key} must be fp32, got {weight.dtype}")
    return weight


class Hy4IHCUnit(nn.Module):
    """One pre/post iHC boundary around attention or MLP."""

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        hidden_size: int,
        hc_mult: int,
        magnitude: float,
        hc_eps: float,
        norm_eps: float,
        *,
        kind: str,
        chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        if kind not in ("attn", "mlp"):
            raise ValueError(f"invalid HY V4 iHC kind: {kind}")
        if hc_mult <= 0 or hidden_size <= 0:
            raise ValueError(
                f"invalid HY V4 iHC geometry: hc_mult={hc_mult}, hidden={hidden_size}"
            )
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.magnitude = float(magnitude)
        self.hc_eps = float(hc_eps)
        self.norm_eps = float(norm_eps)
        self.chunk_size = max(int(chunk_size), 1)

        prefix = "hy4_ihc_attn" if kind == "attn" else "hy4_ihc_mlp"
        self.fn_weight = _require_fp32(weights, getattr(W, f"{prefix}_fn"))
        self.scale = _require_fp32(weights, getattr(W, f"{prefix}_scale"))
        self.base = _require_fp32(weights, getattr(W, f"{prefix}_base"))

        expected_in = hc_mult * hidden_size
        if tuple(self.fn_weight.shape) != (2 * hc_mult, expected_in):
            raise ValueError(
                f"HY V4 {kind} hc_fn shape must be {(2 * hc_mult, expected_in)}, "
                f"got {tuple(self.fn_weight.shape)}"
            )
        if self.scale.numel() != 2 or self.base.numel() != 2 * hc_mult:
            raise ValueError(
                f"HY V4 {kind} iHC scale/base shapes must be (2,) and "
                f"({2 * hc_mult},), got {tuple(self.scale.shape)} and "
                f"{tuple(self.base.shape)}"
            )

    def prepare_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return ``[tokens, hc_mult, hidden_size]`` without double expansion."""
        if hidden_states.dim() == 3:
            expected = (self.hc_mult, self.hidden_size)
            if tuple(hidden_states.shape[-2:]) != expected:
                raise ValueError(
                    f"HY V4 iHC expected trailing shape {expected}, got "
                    f"{tuple(hidden_states.shape[-2:])}"
                )
            return hidden_states
        if hidden_states.dim() != 2:
            raise ValueError(
                f"HY V4 iHC expects a 2D or 3D tensor, got {hidden_states.shape}"
            )
        if hidden_states.size(-1) == self.hidden_size:
            return (
                hidden_states.unsqueeze(1)
                .expand(-1, self.hc_mult, -1)
                .contiguous()
            )
        if hidden_states.size(-1) == self.hc_mult * self.hidden_size:
            return hidden_states.reshape(-1, self.hc_mult, self.hidden_size)
        raise ValueError(
            "HY V4 iHC input width must be hidden_size or hc_mult*hidden_size, "
            f"got {hidden_states.size(-1)}"
        )

    def pre(self, channels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        channels = self.prepare_input(channels)
        if channels.size(0) == 0:
            return channels.new_empty((0, self.hidden_size)), torch.empty(
                (0, self.hc_mult), dtype=torch.float32, device=channels.device
            )
        reads = []
        post_gates = []
        for chunk in channels.split(self.chunk_size, dim=0):
            fused = maybe_fused_ihc_pre(
                chunk,
                self.fn_weight,
                self.scale,
                self.base,
                magnitude=self.magnitude,
                hc_eps=self.hc_eps,
                norm_eps=self.norm_eps,
            )
            if fused is not None:
                read, post_gate = fused
                reads.append(read)
                post_gates.append(post_gate)
                continue
            flat = chunk.flatten(1).float()
            chunk_fp32 = flat.view(-1, self.hc_mult, self.hidden_size)
            rstd = torch.rsqrt(
                flat.square().mean(dim=-1, keepdim=True) + self.norm_eps
            )
            mixes = F.linear(flat, self.fn_weight) * rstd
            pre_raw, post_raw = mixes.split(self.hc_mult, dim=-1)
            pre_gate = (
                torch.sigmoid(
                    pre_raw * self.scale[0] + self.base[: self.hc_mult]
                )
                + self.hc_eps
            )
            post_gate = (
                self.magnitude
                * torch.sigmoid(
                    post_raw * self.scale[1] + self.base[self.hc_mult :]
                )
                + self.hc_eps
            )
            read = torch.sum(pre_gate.unsqueeze(-1) * chunk_fp32, dim=1)
            reads.append(read.to(dtype=channels.dtype))
            post_gates.append(post_gate)
        if len(reads) == 1:
            return reads[0], post_gates[0]
        return torch.cat(reads, dim=0), torch.cat(post_gates, dim=0)

    def post(
        self,
        block_output: torch.Tensor,
        channels: torch.Tensor,
        post_gate: torch.Tensor,
    ) -> torch.Tensor:
        channels = self.prepare_input(channels)
        if block_output.shape != channels.shape[:1] + channels.shape[2:]:
            raise ValueError(
                f"HY V4 iHC block output shape {block_output.shape} does not match "
                f"channels {channels.shape}"
            )
        if tuple(post_gate.shape) != (channels.size(0), self.hc_mult):
            raise ValueError(
                f"HY V4 iHC post gate shape must be "
                f"{(channels.size(0), self.hc_mult)}, got {tuple(post_gate.shape)}"
            )
        fused = maybe_fused_ihc_post(block_output, channels, post_gate)
        if fused is not None:
            return fused
        output = channels.float() + post_gate.float().unsqueeze(-1) * (
            block_output.float().unsqueeze(1)
        )
        return output.to(dtype=block_output.dtype)

    def pre_normed(
        self, channels: torch.Tensor, norm: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse iHC pre with its immediately following RMSNorm when supported."""
        channels = self.prepare_input(channels)
        if channels.size(0) == 0:
            return self.pre(channels)

        norm_weight = norm.weight.data
        # Coalesce chunks only when their DeepGEMM split-K signature matches.
        # The short tail stays separate to preserve the existing reduction
        # order, while both groups write directly into the final output.
        fused = maybe_fused_ihc_pre_normed_grouped(
            channels,
            self.fn_weight,
            self.scale,
            self.base,
            norm_weight,
            magnitude=self.magnitude,
            hc_eps=self.hc_eps,
            ihc_norm_eps=self.norm_eps,
            read_norm_eps=norm.variance_epsilon,
            chunk_size=self.chunk_size,
        )
        if fused is not None:
            return fused

        reads = []
        post_gates = []
        for chunk in channels.split(self.chunk_size, dim=0):
            fused = maybe_fused_ihc_pre_normed(
                chunk,
                self.fn_weight,
                self.scale,
                self.base,
                norm_weight,
                magnitude=self.magnitude,
                hc_eps=self.hc_eps,
                ihc_norm_eps=self.norm_eps,
                read_norm_eps=norm.variance_epsilon,
            )
            if fused is None:
                read, post_gate = self.pre(channels)
                return norm(read), post_gate
            read, post_gate = fused
            reads.append(read)
            post_gates.append(post_gate)
        if len(reads) == 1:
            return reads[0], post_gates[0]
        return torch.cat(reads, dim=0), torch.cat(post_gates, dim=0)

    def pre_normed_mxfp8(
        self, channels: torch.Tensor, norm: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return BF16 iHC input plus its exact MXFP8 representation.

        The DeepGEMM iHC epilogue emits both representations in one launch.
        Unsupported shapes retain the numerically identical two-launch path.
        """
        channels = self.prepare_input(channels)
        if channels.size(0) > 0:
            result = maybe_fused_ihc_pre_normed_grouped(
                channels,
                self.fn_weight,
                self.scale,
                self.base,
                norm.weight.data,
                magnitude=self.magnitude,
                hc_eps=self.hc_eps,
                ihc_norm_eps=self.norm_eps,
                read_norm_eps=norm.variance_epsilon,
                chunk_size=self.chunk_size,
                emit_mxfp8=True,
            )
            if result is not None:
                read, post_gate, read_fp8, read_scale = result
                return read, post_gate, read_fp8, read_scale

        read, post_gate = self.pre_normed(channels, norm)
        from rtp_llm.models_py.kernels.cuda.mxfp8_ops import (
            mxfp8_quant_act_packed,
        )

        read_fp8, read_scale = mxfp8_quant_act_packed(read.contiguous())
        return read, post_gate, read_fp8, read_scale


class Hy4IHCHead(nn.Module):
    """Merge the four residual channels before the final RMSNorm."""

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        hidden_size: int,
        hc_mult: int,
        hc_eps: float,
        norm_eps: float,
        chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.hc_eps = float(hc_eps)
        self.norm_eps = float(norm_eps)
        self.chunk_size = max(int(chunk_size), 1)
        self.fn_weight = _require_fp32(weights, W.hy4_ihc_head_fn)
        self.scale = _require_fp32(weights, W.hy4_ihc_head_scale)
        self.base = _require_fp32(weights, W.hy4_ihc_head_base)
        expected_in = hc_mult * hidden_size
        if tuple(self.fn_weight.shape) != (hc_mult, expected_in):
            raise ValueError(
                f"HY V4 hc_head_fn shape must be {(hc_mult, expected_in)}, got "
                f"{tuple(self.fn_weight.shape)}"
            )
        if self.scale.numel() != 1 or self.base.numel() != hc_mult:
            raise ValueError(
                "HY V4 iHC head scale/base shapes must be (1,) and "
                f"({hc_mult},), got {tuple(self.scale.shape)} and "
                f"{tuple(self.base.shape)}"
            )

    def forward(self, channels: torch.Tensor) -> torch.Tensor:
        if channels.dim() == 2 and channels.size(-1) == self.hc_mult * self.hidden_size:
            channels = channels.reshape(-1, self.hc_mult, self.hidden_size)
        if channels.dim() != 3 or tuple(channels.shape[-2:]) != (
            self.hc_mult,
            self.hidden_size,
        ):
            raise ValueError(f"invalid HY V4 iHC head input shape: {channels.shape}")

        if channels.size(0) == 0:
            return channels.new_empty((0, self.hidden_size))

        outputs = []
        for chunk in channels.split(self.chunk_size, dim=0):
            fused = maybe_fused_ihc_head(
                chunk,
                self.fn_weight,
                self.scale,
                self.base,
                hc_eps=self.hc_eps,
                norm_eps=self.norm_eps,
            )
            if fused is not None:
                outputs.append(fused)
                continue
            flat = chunk.flatten(1).float()
            chunk_fp32 = flat.view(-1, self.hc_mult, self.hidden_size)
            rstd = torch.rsqrt(
                flat.square().mean(dim=-1, keepdim=True) + self.norm_eps
            )
            mixes = F.linear(flat, self.fn_weight) * rstd
            gates = torch.sigmoid(mixes * self.scale + self.base) + self.hc_eps
            output = torch.sum(gates.unsqueeze(-1) * chunk_fp32, dim=1)
            outputs.append(output.to(dtype=channels.dtype))
        if len(outputs) == 1:
            return outputs[0]
        return torch.cat(outputs, dim=0)
