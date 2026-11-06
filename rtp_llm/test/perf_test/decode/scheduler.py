"""Deterministic allocation of global decode requests to DP workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from rtp_llm.test.perf_test.errors import PlanningError


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
            "local_batch_size": self.local_batch_size,
            "request_indices": list(self.request_indices),
            "input_lengths": list(self.input_lengths),
        }


@dataclass(frozen=True)
class DecodeBatchSchedule:
    global_batch_size: int
    dp_size: int
    workers: Tuple[WorkerAllocation, ...]


def schedule_decode_requests(
    input_lengths: Sequence[int], *, dp_size: int
) -> DecodeBatchSchedule:
    """Assign balanced contiguous request ranges, including idle DP workers."""
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
