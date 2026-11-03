from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class AttentionLayerRecord:
    layer_idx: int
    q: torch.Tensor  # packed [T, head_num, head_dim]
    k: torch.Tensor  # packed [T, kv_head_num, head_dim]
    v: torch.Tensor  # packed [T, kv_head_num, head_dim]
    output: torch.Tensor  # packed [T, head_num * head_dim]
    # Golden-only [batch, head_num] contribution RMS used by the decode gate.
    reference_contribution: Optional[torch.Tensor] = None


@dataclass
class AttentionForwardRecord:
    impl_name: str
    phase: str  # "plain" | "prefix" | "decode_{st}"
    layer_records: Dict[int, AttentionLayerRecord]

    def release(self) -> None:
        for r in self.layer_records.values():
            del r.q, r.k, r.v, r.output, r.reference_contribution
        self.layer_records.clear()
