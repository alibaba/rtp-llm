"""Per-forward MoE view of the engine-selected decode execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rtp_llm.ops.compute_ops import (
        DecodeExecutionMode,
        DecodeExecutionPhase,
        PyModelInputs,
    )


@dataclass(frozen=True)
class DecodeMoePlan:
    mode: DecodeExecutionMode
    phase: DecodeExecutionPhase
    bucket: int
    token_width: int
    valid_rows: torch.Tensor

    @classmethod
    def from_inputs(cls, inputs: PyModelInputs) -> DecodeMoePlan | None:
        execution = getattr(inputs, "decode_execution", None)
        if execution is None:
            return None
        from rtp_llm.ops.compute_ops import DecodeExecutionMode

        if execution.mode not in (
            DecodeExecutionMode.EAGER,
            DecodeExecutionMode.FIXED,
            DecodeExecutionMode.GRAPH,
        ):
            raise ValueError(f"Invalid decode execution mode: {execution.mode}")
        return cls(
            execution.mode,
            execution.phase,
            int(execution.bucket),
            int(execution.token_width),
            inputs.decode_row_validity,
        )

    @property
    def fixed(self) -> bool:
        from rtp_llm.ops.compute_ops import DecodeExecutionMode

        return self.mode in (DecodeExecutionMode.FIXED, DecodeExecutionMode.GRAPH)

    def validate_rows(self, x: torch.Tensor) -> None:
        mask = self.valid_rows
        if (
            not isinstance(mask, torch.Tensor)
            or mask.shape != (x.size(0),)
            or mask.dtype != torch.bool
            or mask.device != x.device
        ):
            raise ValueError("Decode validity must be a device bool per row")
        if self.fixed and (
            self.bucket <= 0
            or self.token_width <= 0
            or self.bucket * self.token_width != x.size(0)
        ):
            raise ValueError("Decode rows must match the planned bucket and width")
