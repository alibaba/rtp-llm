import logging
from typing import Optional


DEFAULT_PRE_STOP_DRAIN_SECONDS = 120.0
AUTO_PRE_STOP_DRAIN_HEADROOM_SECONDS = -1.0


def normalize_non_negative_seconds(
    value: Optional[float],
    default_value: float,
    config_name: str,
) -> float:
    try:
        seconds = float(value) if value is not None else float(default_value)
    except (TypeError, ValueError):
        logging.warning(
            "Invalid %s=%r, falling back to %.3fs",
            config_name,
            value,
            default_value,
        )
        seconds = float(default_value)
    return max(0.0, seconds)


def pre_stop_drain_headroom_seconds(
    configured_headroom_seconds: Optional[float],
    shutdown_timeout: float,
) -> float:
    if configured_headroom_seconds is not None:
        try:
            headroom_seconds = float(configured_headroom_seconds)
        except (TypeError, ValueError):
            headroom_seconds = AUTO_PRE_STOP_DRAIN_HEADROOM_SECONDS
        if headroom_seconds >= 0:
            return headroom_seconds
    return min(60.0, max(1.0, float(shutdown_timeout) * 0.10))


def effective_pre_stop_drain_seconds(
    *,
    configured_drain_seconds: Optional[float],
    shutdown_timeout: Optional[float],
    configured_headroom_seconds: Optional[float],
    component: str,
) -> float:
    drain_seconds = normalize_non_negative_seconds(
        configured_drain_seconds,
        DEFAULT_PRE_STOP_DRAIN_SECONDS,
        f"{component}_pre_stop_drain_seconds",
    )
    if shutdown_timeout is None or shutdown_timeout <= 0:
        return drain_seconds

    headroom_seconds = pre_stop_drain_headroom_seconds(
        configured_headroom_seconds,
        float(shutdown_timeout),
    )
    max_drain_seconds = max(0.0, float(shutdown_timeout) - headroom_seconds)
    if drain_seconds <= max_drain_seconds:
        return drain_seconds

    logging.warning(
        "Clamp %s pre-stop drain %.3fs to %.3fs "
        "(shutdown_timeout=%ss, headroom=%.3fs)",
        component,
        drain_seconds,
        max_drain_seconds,
        shutdown_timeout,
        headroom_seconds,
    )
    return max_drain_seconds
