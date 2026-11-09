"""Immutable policy for automatically selected tensor-parallel reductions."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CollectiveConfig:
    all_reduce_backend: str = "auto"
    large_prefill: Optional[bool] = None

    def __post_init__(self):
        if self.all_reduce_backend not in ("auto", "nccl", "sm120_ipc", "sm120_p2p"):
            raise ValueError(
                "AllReduce backend must be auto, nccl, sm120_ipc or sm120_p2p"
            )
        if self.large_prefill is not None and type(self.large_prefill) is not bool:
            raise ValueError("large-prefill option must be a bool or None")
        if self.large_prefill and self.all_reduce_backend not in ("auto", "sm120_ipc"):
            raise ValueError("large-prefill requires auto or sm120_ipc")

    def validate_topology(self, parallelism):
        if self.all_reduce_backend in ("auto", "nccl"):
            return
        size = 2 if self.all_reduce_backend == "sm120_ipc" else 8
        cp = parallelism.prefill_cp_config
        if (
            parallelism.tp_size != size
            or parallelism.world_size != size
            or parallelism.local_world_size != size
            or parallelism.dp_size != 1
            or parallelism.pp_size != 1
            or cp.is_enabled()
            or cp.is_prefill_enabled()
        ):
            raise ValueError(
                f"{self.all_reduce_backend} requires single-node TP{size} without DP/PP/CP"
            )

    def to_string(self) -> str:
        return (
            f"all_reduce_backend: {self.all_reduce_backend}\n"
            f"large_prefill: {self.large_prefill}"
        )
