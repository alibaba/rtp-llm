"""Independent Hyper-Connections used by HY V4.

HY V4 iHC has four residual channels and only pre/post gates.  It is not the
DeepSeek-V4 mHC transform: there is no 4x4 Sinkhorn/combination matrix.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

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
            flat = chunk.flatten(1).float()
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
            read = torch.sum(pre_gate.unsqueeze(-1) * chunk.float(), dim=1)
            reads.append(read.to(dtype=channels.dtype))
            post_gates.append(post_gate)
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
        output = channels.float() + post_gate.float().unsqueeze(-1) * (
            block_output.float().unsqueeze(1)
        )
        return output.to(dtype=block_output.dtype)


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
            flat = chunk.flatten(1).float()
            rstd = torch.rsqrt(
                flat.square().mean(dim=-1, keepdim=True) + self.norm_eps
            )
            mixes = F.linear(flat, self.fn_weight) * rstd
            gates = torch.sigmoid(mixes * self.scale + self.base) + self.hc_eps
            output = torch.sum(gates.unsqueeze(-1) * chunk.float(), dim=1)
            outputs.append(output.to(dtype=channels.dtype))
        return torch.cat(outputs, dim=0)
