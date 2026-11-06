"""Planned allocation of global decode requests to DP workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from rtp_llm.test.perf_test.errors import PlanningError


def requires_distributed_decode(
    topology: Mapping[str, int], local_world_size: int
) -> bool:
    """Route cross-host DP through each node's local frontend.

    TP-only gangs can use the master frontend. Cross-host DP currently needs
    whole TP groups on each host and no pipeline parallelism.
    """
    tp, dp, pp = (int(topology[name]) for name in ("tp", "dp", "pp"))
    world = tp * dp * pp
    if dp == 1 or world <= local_world_size:
        return False
    if pp != 1:
        raise PlanningError("cross-host DP decode requires PP=1")
    if local_world_size < 1 or local_world_size % tp or world % local_world_size:
        raise PlanningError("cross-host DP decode requires whole local TP groups")
    return True


@dataclass(frozen=True)
class WorkerAllocation:
    worker_index: int
    request_indices: Tuple[int, ...]
    input_lengths: Tuple[int, ...]

    @property
    def local_batch_size(self) -> int:
        return len(self.request_indices)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_index": self.worker_index,
            "assignment": "planned",
            "local_batch_size": self.local_batch_size,
            "request_indices": list(self.request_indices),
            "input_lengths": list(self.input_lengths),
        }


@dataclass(frozen=True)
class DecodeBatchSchedule:
    global_batch_size: int
    dp_size: int
    workers: Tuple[WorkerAllocation, ...]


def supported_decode_batch_sizes(
    max_size: int, candidate_step: int, *, dp_size: int
) -> Tuple[int, ...]:
    """Enumerate search candidates supported by the shared DP scheduler batch."""
    if type(dp_size) is not int or dp_size <= 0:
        raise PlanningError("decode schedule dp_size must be a positive integer")
    values = {1, max_size, *range(candidate_step, max_size + 1, candidate_step)}
    return tuple(
        size for size in sorted(values) if size <= dp_size or size % dp_size == 0
    )


def scheduler_local_batch_size(local_batches: Sequence[int]) -> int:
    """Return the shared scheduler batch, allowing workers without requests."""
    if not local_batches or any(
        type(size) is not int or size < 0 for size in local_batches
    ):
        raise PlanningError("decode local batch sizes must be non-negative integers")
    active = {size for size in local_batches if size > 0}
    if len(active) != 1:
        raise PlanningError(
            "decode scheduler requires uniform non-zero DP local batch sizes; "
            "use a global batch no larger than DP size or divisible by DP size"
        )
    return active.pop()


def schedule_decode_requests(
    input_lengths: Sequence[int], *, dp_size: int
) -> DecodeBatchSchedule:
    """Plan contiguous ranges; transports must enforce supported routing.

    These ranges select nodes in cluster execution. A single HTTP frontend
    cannot bind requests to its local DP workers, so they are not observed
    per-worker assignments.
    """
    lengths = tuple(input_lengths)
    if not lengths:
        raise PlanningError("decode schedule requires at least one request")
    if any(type(item) is not int or item <= 0 for item in lengths):
        raise PlanningError("decode schedule input lengths must be positive integers")
    if type(dp_size) is not int or dp_size <= 0:
        raise PlanningError("decode schedule dp_size must be a positive integer")
    base, extra = divmod(len(lengths), dp_size)
    workers = []
    offset = 0
    for worker_index in range(dp_size):
        count = base + (1 if worker_index < extra else 0)
        indices = tuple(range(offset, offset + count))
        workers.append(
            WorkerAllocation(
                worker_index=worker_index,
                request_indices=indices,
                input_lengths=tuple(lengths[index] for index in indices),
            )
        )
        offset += count
    return DecodeBatchSchedule(
        global_batch_size=len(lengths), dp_size=dp_size, workers=tuple(workers)
    )
