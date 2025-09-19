from __future__ import annotations

import typing

import torch

__all__ = ["EplbPyWrapperOP"]

class EplbPyWrapperOP:
    def __init__(self) -> None: ...
    def create_balance_plan(
        self, log_stats: torch.Tensor, gpu_loads: torch.Tensor
    ) -> None: ...
    def get_result(
        self,
    ) -> tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]: ...
    def init(self, arg0: typing.Any) -> None: ...
    def load_moe_weight(self, ep_rank: int, ep_size: int) -> None: ...
