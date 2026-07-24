"""Shared Kimi K3 KDA execution types and canonical cache state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


KDAExecutionMode = Literal["prefill", "decode"]


@dataclass(frozen=True)
class KimiKDAState:
    """Canonical KDA cache for one layer.

    Convolution states contain the previous ``kernel_size - 1`` projected
    tokens in chronological order. ``recurrent_state`` uses ``[B,H,K,V]``;
    the RTP cache converter owns any physical-layout transposition.
    """

    q_conv_state: torch.Tensor
    k_conv_state: torch.Tensor
    v_conv_state: torch.Tensor
    recurrent_state: torch.Tensor
