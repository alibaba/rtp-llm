"""Explicit, Python-only options for Qwen4-owned SM120 reductions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen4CollectiveOptions:
    backend: str = "nccl"
    large_prefill: bool = False
    model_type: str = ""

    def __post_init__(self):
        if self.backend not in ("nccl", "sm120_ipc", "sm120_p2p"):
            raise ValueError(
                "Qwen4 AllReduce backend must be nccl, sm120_ipc or sm120_p2p"
            )
        if type(self.large_prefill) is not bool:
            raise ValueError("Qwen4 large-prefill option must be a bool")
        if self.large_prefill and self.backend != "sm120_ipc":
            raise ValueError("Qwen4 large-prefill requires sm120_ipc")

    def validate_topology(self, parallelism):
        if self.backend == "nccl":
            return
        if self.model_type not in ("qwen4_exp", "qwen4_exp_mtp"):
            raise ValueError("SM120 AllReduce backends are owned by Qwen4 models")
        size = 2 if self.backend == "sm120_ipc" else 8
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
                f"{self.backend} requires single-node TP{size} without DP/PP/CP"
            )
