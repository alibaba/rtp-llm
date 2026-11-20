"""Select graph MoE execution and workspace sizing as a pair."""

from dataclasses import dataclass
from functools import cache
from typing import Callable

import torch


@dataclass(frozen=True)
class GraphMoeOps:
    workspace_size: Callable[..., int]
    _execute: Callable[..., object]

    def run(self, *, output: torch.Tensor, **kwargs) -> torch.Tensor:
        self._execute(output=output, **kwargs)
        return output


@cache
def _bind(device_index: int) -> GraphMoeOps:
    if torch.cuda.get_device_capability(device_index) != (12, 0):
        raise ValueError("SM120 graph MoE requires an exact SM120 device")

    from .jit import spec
    from .op import cutlass_fused_moe, cutlass_fused_moe_workspace_size

    # Validate the selected provider's source closure. Warmup loads and
    # exercises its native runner before graph capture.
    spec()
    return GraphMoeOps(cutlass_fused_moe_workspace_size, cutlass_fused_moe)


def bind(device: torch.device) -> GraphMoeOps:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("SM120 graph MoE requires a CUDA device")
    index = torch.cuda.current_device() if device.index is None else device.index
    return _bind(index)


def is_available(device: torch.device) -> bool:
    try:
        bind(device)
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError):
        return False
    return True
