"""Dependency-light cancellation outcome classification for Dash smoke tests."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DashCancelState:
    cancel_requested: bool
    cancel_exercised: bool
    cancelled: bool
    completed_before_cancel: bool


def classify_dash_cancel(
    *, cancel_exercised: bool, terminal_code: Any, cancelled_code: Any, ok_code: Any
) -> DashCancelState:
    """Classify whether the comparer won the cancellation race."""
    if terminal_code == cancelled_code:
        return DashCancelState(
            cancel_requested=True,
            cancel_exercised=cancel_exercised,
            cancelled=cancel_exercised,
            completed_before_cancel=False,
        )
    if terminal_code == ok_code:
        return DashCancelState(
            cancel_requested=True,
            cancel_exercised=cancel_exercised,
            cancelled=False,
            completed_before_cancel=True,
        )
    raise ValueError(f"unexpected cancellation terminal code: {terminal_code}")
