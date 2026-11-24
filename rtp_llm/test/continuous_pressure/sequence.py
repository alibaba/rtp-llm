"""Cumulative arrival scheduling and deterministic seed-family selection."""

import math
import random
from dataclasses import dataclass


class ArrivalClock:
    """Cumulative offsets from one formal-distribution session origin."""

    def __init__(self, rate: float, poisson: bool, seed: int):
        if isinstance(rate, bool) or not math.isfinite(rate) or rate < 0:
            raise ValueError("arrival rate must be finite and nonnegative")
        self.rate, self.poisson = rate, poisson
        self.rng = random.Random(seed)
        self.offset = 0.0
        self.count = 0

    def next_offset(self) -> float:
        if self.count and self.rate > 0:
            self.offset += (
                self.rng.expovariate(self.rate) if self.poisson else 1.0 / self.rate
            )
        self.count += 1
        return self.offset


class AdaptiveBacklogController:
    """Piecewise-linear offered rate derived from unfinished RPC count."""

    def __init__(self, *, rate_min, rate_max, backlog_low, backlog_high):
        if (
            isinstance(rate_min, bool)
            or isinstance(rate_max, bool)
            or not math.isfinite(rate_min)
            or not math.isfinite(rate_max)
            or rate_min <= 0
            or rate_max < rate_min
        ):
            raise ValueError("adaptive rates must satisfy 0 < min <= max")
        if (
            type(backlog_low) is not int
            or type(backlog_high) is not int
            or backlog_low < 0
            or backlog_high <= backlog_low
        ):
            raise ValueError("adaptive backlog must satisfy 0 <= low < high")
        self.rate_min = float(rate_min)
        self.rate_max = float(rate_max)
        self.backlog_low = backlog_low
        self.backlog_high = backlog_high

    def rate(self, outstanding: int) -> float:
        if type(outstanding) is not int or outstanding < 0:
            raise ValueError("outstanding must be a nonnegative integer")
        if outstanding <= self.backlog_low:
            return self.rate_max
        if outstanding >= self.backlog_high:
            return self.rate_min
        remaining = (self.backlog_high - outstanding) / (
            self.backlog_high - self.backlog_low
        )
        return self.rate_min + (self.rate_max - self.rate_min) * remaining


@dataclass(frozen=True)
class TieredRollingAssignment:
    cohort_id: int
    cohort_lane: int
    family: int
    case_index: int
    expected_cache_tier: str
    promoted_from_cohort: int | None = None


class TieredRollingPlanner:
    """Build fixed cohorts whose memory half becomes the next device half."""

    def __init__(
        self,
        *,
        effective_seed_count: int,
        cohort_size: int,
        memory_reuse_ratio: float,
        case_indices: tuple[int, ...],
    ):
        if type(effective_seed_count) is not int or effective_seed_count <= 0:
            raise ValueError("effective_seed_count must be positive")
        if type(cohort_size) is not int or cohort_size <= 0:
            raise ValueError("cohort_size must be positive")
        if (
            isinstance(memory_reuse_ratio, bool)
            or not math.isfinite(memory_reuse_ratio)
            or not 0 < memory_reuse_ratio < 1
        ):
            raise ValueError("memory_reuse_ratio must be between zero and one")
        memory_count = cohort_size * memory_reuse_ratio
        if not memory_count.is_integer():
            raise ValueError("memory reuse ratio must produce an integral cohort")
        self.memory_count = int(memory_count)
        self.device_count = cohort_size - self.memory_count
        if self.memory_count != self.device_count:
            raise ValueError("tiered rolling currently requires a 50% memory ratio")
        if not case_indices or self.memory_count % len(case_indices):
            raise ValueError("cache cases must divide each tier half evenly")
        self.effective_seed_count = effective_seed_count
        self.cohort_size = cohort_size
        self.case_indices = tuple(case_indices)
        self.hot_start = effective_seed_count - self.device_count
        self.memory_window_end = (
            self.hot_start // self.memory_count
        ) * self.memory_count
        if self.memory_window_end < 2 * self.memory_count:
            raise ValueError("effective seed count must provide two memory windows")
        self.cohort_id = 0

    def _cases(self) -> tuple[int, ...]:
        repeats = self.memory_count // len(self.case_indices)
        return self.case_indices * repeats

    def assignment(self, sequence: int) -> TieredRollingAssignment:
        if type(sequence) is not int or sequence < 0:
            raise ValueError("tiered rolling sequence must be nonnegative")
        cohort_id, cohort_lane = divmod(sequence, self.cohort_size)
        lane = cohort_lane // 2
        cases = self._cases()
        case_index = cases[lane]
        memory_start = (cohort_id * self.memory_count) % self.memory_window_end
        if cohort_lane % 2:
            return TieredRollingAssignment(
                cohort_id=cohort_id,
                cohort_lane=cohort_lane,
                family=memory_start + lane,
                case_index=case_index,
                expected_cache_tier="memory",
            )
        if cohort_id == 0:
            return TieredRollingAssignment(
                cohort_id=cohort_id,
                cohort_lane=cohort_lane,
                family=self.hot_start + lane,
                case_index=case_index,
                expected_cache_tier="device",
            )
        previous_start = ((cohort_id - 1) * self.memory_count) % (
            self.memory_window_end
        )
        return TieredRollingAssignment(
            cohort_id=cohort_id,
            cohort_lane=cohort_lane,
            family=previous_start + lane,
            case_index=case_index,
            expected_cache_tier="device",
            promoted_from_cohort=cohort_id - 1,
        )

    def next_cohort(self) -> tuple[TieredRollingAssignment, ...]:
        start = self.cohort_id * self.cohort_size
        result = tuple(
            self.assignment(start + lane) for lane in range(self.cohort_size)
        )
        self.cohort_id += 1
        return result

    def initial_device_assignments(self) -> tuple[TieredRollingAssignment, ...]:
        return tuple(self.assignment(lane) for lane in range(0, self.cohort_size, 2))


def freeze_seed_families(seed_ids) -> int:
    """Return the populated prefix length, rejecting holes in seed families."""
    effective = len(seed_ids)
    for index, ids in enumerate(seed_ids):
        if ids is None:
            effective = index
            break
    if any(ids is not None for ids in seed_ids[effective:]):
        raise ValueError("completed seed families must form a contiguous prefix")
    return effective
