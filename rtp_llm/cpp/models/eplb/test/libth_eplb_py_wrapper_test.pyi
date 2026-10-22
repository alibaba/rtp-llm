from __future__ import annotations

from typing import Any, Optional

import torch

__all__ = ["EplbPyWrapperOP", "ExpertBalancerStepTest", "ExpertLoadMonitor"]

class EplbPyWrapperOP:
    def __init__(self) -> None: ...
    def initialize(
        self, model: Any, ep_rank: int = 0, ep_size: int = 1, model_id: int = 0
    ) -> None: ...
    def record_step(self, counts: torch.Tensor) -> None: ...
    def balance(self) -> bool: ...
    def update_config(self, mode: int, update_time: int) -> None: ...
    def destroy(self) -> None: ...

class ExpertBalancerStepTest:
    def __init__(
        self,
        model: Any,
        snapshot_max_files: int,
        interval: int = 2,
        model_id: int = 0,
        moe: bool = True,
        tp_rank: int = 0,
        tp_size: int = 1,
        mode: Optional[int] = None,
    ) -> None: ...
    def step(self, warmup: bool = False) -> None: ...
    def update_config(self, mode: int, update_time: int = 1) -> bool: ...
    def destroy(self) -> None: ...
    @property
    def counts(self) -> Optional[torch.Tensor]: ...

class ExpertLoadMonitor:
    def __init__(
        self,
        layers: int,
        experts: int,
        world_size: int,
        rank: int,
        interval: int,
        model_id: int,
        snapshot_max_files: int,
    ) -> None: ...
    def clear_record_buffer(self) -> None: ...
    def record_step(self) -> None: ...
    @property
    def counts(self) -> torch.Tensor: ...
