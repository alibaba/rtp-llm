"""Pure port planning for the currently supported Java mock backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

MASTER_PORT_COUNT = 6
MOCK_CONTROL_OFFSET = -1
VICTIM_OFFSETS = (149, 150, 151)
MOCK_WINDOW_LAST = max(VICTIM_OFFSETS)
MIN_MOCK_STRIDE = MOCK_WINDOW_LAST - MOCK_CONTROL_OFFSET + 1


def port_intervals(
    master_base: int, mock_base: int
) -> tuple[tuple[str, int, int], ...]:
    return (
        ("master", master_base, master_base + MASTER_PORT_COUNT - 1),
        ("mock", mock_base + MOCK_CONTROL_OFFSET, mock_base + MOCK_WINDOW_LAST),
    )


def child_port_env(master_base: int, mock_base: int) -> dict[str, str]:
    return {
        "FLEXLB_FT_MASTER_HTTP_PORT": str(master_base),
        "FLEXLB_FT_MASTER_MANAGEMENT_PORT": str(master_base + 1),
        "FLEXLB_FT_HA_MASTER_A_HTTP_PORT": str(master_base),
        "FLEXLB_FT_HA_MASTER_B_HTTP_PORT": str(master_base + 3),
        "FLEXLB_FT_MOCK_BASE_GRPC_PORT": str(mock_base),
    }


class ResourcePlanError(ValueError):
    """The selected instances cannot be executed within a valid port lease."""


def _integer(value: int, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ResourcePlanError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class JavaMockBudget:
    """Upper bound for one fresh instance, including every possible add.

    DynamicEngineManager allocates max(current ports)+1. Removing a low
    port does not lower that maximum, so peak live workers is insufficient.
    Explicit worker ports and other backends require separate validation.
    """

    initial_workers: int
    dynamic_additions: int
    max_environment_workers: int | None = None

    def __post_init__(self) -> None:
        _integer(self.initial_workers, "initial_workers", 1)
        _integer(self.dynamic_additions, "dynamic_additions", 0)
        if self.max_environment_workers is not None:
            _integer(
                self.max_environment_workers,
                "max_environment_workers",
                self.initial_workers,
            )
        if self.worker_capacity > min(VICTIM_OFFSETS):
            raise ResourcePlanError(
                f"environment worker bound + cumulative dynamic additions = {self.worker_capacity} "
                f"exceeds {min(VICTIM_OFFSETS)}; Java mock reserves offsets "
                f"{VICTIM_OFFSETS} for victim/control ports"
            )

    @property
    def worker_capacity(self) -> int:
        return (
            self.max_environment_workers or self.initial_workers
        ) + self.dynamic_additions

    def to_manifest(self) -> dict:
        return {
            "backend": "java_mock",
            "initial_workers": self.initial_workers,
            **(
                {"max_environment_workers": self.max_environment_workers}
                if self.max_environment_workers is not None
                else {}
            ),
            "dynamic_additions": self.dynamic_additions,
            "worker_capacity": self.worker_capacity,
            "mock_control_offset": MOCK_CONTROL_OFFSET,
            "victim_offsets": list(VICTIM_OFFSETS),
        }

    @classmethod
    def from_metadata(cls, raw: dict) -> JavaMockBudget:
        """Validate compiler resource_budget v1 before opening any lease."""
        fixed = {
            "backend": "java_mock",
            "bounded": True,
            "mock_control_offset": MOCK_CONTROL_OFFSET,
            "victim_control_offset": VICTIM_OFFSETS[0],
            "victim_grpc_offset": VICTIM_OFFSETS[1],
            "reserved_tail_offset": VICTIM_OFFSETS[2],
        }
        required = set(fixed) | {"initial_workers", "max_dynamic_additions"}
        if not isinstance(raw, dict) or set(raw) not in (
            required,
            required | {"max_environment_workers"},
        ):
            raise ResourcePlanError("invalid resource_budget v1 fields")
        for name, expected in fixed.items():
            if type(raw[name]) is not type(expected) or raw[name] != expected:
                raise ResourcePlanError(
                    f"unsupported resource budget {name}={raw[name]!r}"
                )
        if "max_environment_workers" in raw:
            _integer(raw["max_environment_workers"], "max_environment_workers", 1)
        return cls(
            raw["initial_workers"],
            raw["max_dynamic_additions"],
            raw.get("max_environment_workers"),
        )


@dataclass(frozen=True)
class LaneLease:
    """One lane's reservation, shared only by sequential clean instances."""

    lane: int
    master_base: int
    mock_base: int
    worker_capacity: int

    def __post_init__(self) -> None:
        _integer(self.lane, "lane", 0)
        _integer(self.master_base, "master_base", 1024)
        _integer(self.mock_base, "mock_base", 1025)
        _integer(self.worker_capacity, "worker_capacity", 1)
        if self.worker_capacity > min(VICTIM_OFFSETS):
            raise ResourcePlanError("worker capacity overlaps victim/control ports")
        for _, lo, hi in self.intervals():
            if hi > 65535:
                raise ResourcePlanError(f"port interval {lo}..{hi} exceeds 65535")
        (_, m_lo, m_hi), (_, g_lo, g_hi) = self.intervals()
        if m_lo <= g_hi and g_lo <= m_hi:
            raise ResourcePlanError("lane master and mock port intervals overlap")

    def intervals(self) -> tuple[tuple[str, int, int], ...]:
        return port_intervals(self.master_base, self.mock_base)

    def ports(self) -> tuple[int, ...]:
        return tuple(
            port for _, lo, hi in self.intervals() for port in range(lo, hi + 1)
        )

    def lock_names(self) -> tuple[str, ...]:
        return tuple(
            f"{'m' if side == 'master' else 'g'}{lo}_{hi}.lock"
            for side, lo, hi in self.intervals()
        )

    def child_env(self) -> dict[str, str]:
        return child_port_env(self.master_base, self.mock_base)

    def to_manifest(self) -> dict:
        return {
            "lane": self.lane,
            "backend": "java_mock",
            "master_base": self.master_base,
            "mock_base": self.mock_base,
            "worker_capacity": self.worker_capacity,
            "intervals": [
                {"side": side, "first": lo, "last": hi}
                for side, lo, hi in self.intervals()
            ],
            "lock_names": list(self.lock_names()),
            "child_env": self.child_env(),
        }


def plan_lane_leases(
    budgets_by_lane: Sequence[Sequence[JavaMockBudget]],
    *,
    master_base: int,
    mock_base: int,
    mock_stride: int = 500,
) -> list[LaneLease]:
    """Reserve each lane's maximum demand; lanes execute their instances serially."""
    _integer(mock_stride, "mock_stride", MIN_MOCK_STRIDE)
    if not budgets_by_lane or any(not lane for lane in budgets_by_lane):
        raise ResourcePlanError("resource plan must contain nonempty lanes")
    leases = [
        LaneLease(
            i,
            master_base + 10 * i,
            mock_base + mock_stride * i,
            max(budget.worker_capacity for budget in budgets),
        )
        for i, budgets in enumerate(budgets_by_lane)
    ]
    intervals = [
        (lease.lane, side, lo, hi)
        for lease in leases
        for side, lo, hi in lease.intervals()
    ]
    for i, (lane, side, lo, hi) in enumerate(intervals):
        for other, other_side, start, end in intervals[:i]:
            if lo <= end and start <= hi:
                raise ResourcePlanError(
                    f"lane {lane} {side} {lo}..{hi} overlaps lane {other} {other_side} {start}..{end}"
                )
    return leases
