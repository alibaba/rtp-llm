"""Timeout policy for REAPI-backed pytest execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RemoteTimeoutPolicy:
    profile_class: str
    session_budget_seconds: int
    action_timeout_seconds: int
    supervisor_timeout_seconds: int
    pytest_timeout_seconds: int
    queued_timeout_seconds: int
    min_retry_remaining_seconds: int
    heartbeat_stall_seconds: int

    @property
    def rpc_timeout_seconds(self) -> int:
        return self.action_timeout_seconds + 120


_POLICIES = {
    "ut_session": RemoteTimeoutPolicy(
        profile_class="ut_session",
        session_budget_seconds=3000,
        action_timeout_seconds=1500,
        supervisor_timeout_seconds=1440,
        pytest_timeout_seconds=300,
        queued_timeout_seconds=300,
        min_retry_remaining_seconds=360,
        heartbeat_stall_seconds=600,
    ),
    "smoke_session": RemoteTimeoutPolicy(
        profile_class="smoke_session",
        session_budget_seconds=3600,
        action_timeout_seconds=3000,
        supervisor_timeout_seconds=2880,
        pytest_timeout_seconds=600,
        queued_timeout_seconds=300,
        min_retry_remaining_seconds=900,
        heartbeat_stall_seconds=900,
    ),
    "perf_session": RemoteTimeoutPolicy(
        profile_class="perf_session",
        session_budget_seconds=6600,
        action_timeout_seconds=6300,
        supervisor_timeout_seconds=6120,
        pytest_timeout_seconds=1800,
        queued_timeout_seconds=300,
        min_retry_remaining_seconds=1200,
        heartbeat_stall_seconds=1800,
    ),
    "per_test_ut": RemoteTimeoutPolicy(
        profile_class="per_test_ut",
        session_budget_seconds=300,
        action_timeout_seconds=150,
        supervisor_timeout_seconds=130,
        pytest_timeout_seconds=100,
        queued_timeout_seconds=60,
        min_retry_remaining_seconds=60,
        heartbeat_stall_seconds=600,
    ),
    "per_test_smoke": RemoteTimeoutPolicy(
        profile_class="per_test_smoke",
        session_budget_seconds=3300,
        action_timeout_seconds=3000,
        supervisor_timeout_seconds=2880,
        pytest_timeout_seconds=600,
        queued_timeout_seconds=180,
        min_retry_remaining_seconds=360,
        heartbeat_stall_seconds=2400,
    ),
    "per_test_perf": RemoteTimeoutPolicy(
        profile_class="per_test_perf",
        session_budget_seconds=6600,
        action_timeout_seconds=6300,
        supervisor_timeout_seconds=6120,
        pytest_timeout_seconds=1800,
        queued_timeout_seconds=300,
        min_retry_remaining_seconds=1200,
        heartbeat_stall_seconds=1800,
    ),
    "default": RemoteTimeoutPolicy(
        profile_class="default",
        session_budget_seconds=1800,
        action_timeout_seconds=1500,
        supervisor_timeout_seconds=1440,
        pytest_timeout_seconds=300,
        queued_timeout_seconds=300,
        min_retry_remaining_seconds=360,
        heartbeat_stall_seconds=600,
    ),
}


def _profile_class(profile: Optional[str], *, per_test: bool) -> str:
    normalized = (profile or "").strip().lower().replace("-", "_")
    if per_test:
        if normalized.startswith("perf"):
            return "per_test_perf"
        if normalized.startswith("smoke"):
            return "per_test_smoke"
        return "per_test_ut"
    if normalized.startswith("perf"):
        return "perf_session"
    if normalized.startswith("smoke"):
        return "smoke_session"
    if normalized.startswith("ut") or normalized.startswith("py_ut"):
        return "ut_session"
    return "default"


def _cap_int(value: int, cap: int) -> int:
    return max(1, min(int(value), int(cap)))


def _cap_from_session(
    base: RemoteTimeoutPolicy, session_budget_seconds: int
) -> RemoteTimeoutPolicy:
    action_margin = max(
        1, min(300, base.session_budget_seconds - base.action_timeout_seconds)
    )
    supervisor_margin = max(
        1, min(120, base.action_timeout_seconds - base.supervisor_timeout_seconds)
    )
    pytest_margin = max(
        1, min(30, base.supervisor_timeout_seconds - base.pytest_timeout_seconds)
    )

    action_timeout_seconds = _cap_int(
        base.action_timeout_seconds, session_budget_seconds - action_margin
    )
    supervisor_timeout_seconds = _cap_int(
        base.supervisor_timeout_seconds, action_timeout_seconds - supervisor_margin
    )
    pytest_timeout_seconds = _cap_int(
        base.pytest_timeout_seconds, supervisor_timeout_seconds - pytest_margin
    )
    queued_timeout_seconds = _cap_int(
        base.queued_timeout_seconds, action_timeout_seconds
    )
    min_retry_remaining_seconds = _cap_int(
        base.min_retry_remaining_seconds, session_budget_seconds
    )
    heartbeat_stall_seconds = _cap_int(
        base.heartbeat_stall_seconds, supervisor_timeout_seconds
    )
    return RemoteTimeoutPolicy(
        profile_class=base.profile_class,
        session_budget_seconds=session_budget_seconds,
        action_timeout_seconds=action_timeout_seconds,
        supervisor_timeout_seconds=supervisor_timeout_seconds,
        pytest_timeout_seconds=pytest_timeout_seconds,
        queued_timeout_seconds=queued_timeout_seconds,
        min_retry_remaining_seconds=min_retry_remaining_seconds,
        heartbeat_stall_seconds=heartbeat_stall_seconds,
    )


def select_remote_timeout_policy(
    profile: Optional[str],
    *,
    per_test: bool,
    remote_timeout_override: Optional[int] = None,
) -> RemoteTimeoutPolicy:
    """Return the timeout policy for a pytest remote dispatch.

    ``--remote-timeout`` is treated as an override for the total session budget.
    Action, supervisor, pytest and queued watchdog limits are then capped from
    that budget while preserving the policy's usual ordering.
    """

    base = _POLICIES[_profile_class(profile, per_test=per_test)]
    if remote_timeout_override is None:
        return base
    session_budget_seconds = _cap_int(
        remote_timeout_override, base.session_budget_seconds
    )
    if session_budget_seconds == base.session_budget_seconds:
        return base
    return _cap_from_session(base, session_budget_seconds)
