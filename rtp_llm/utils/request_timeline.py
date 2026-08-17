import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

TIMELINE_LOG_PREFIX = "REQUEST_TIMELINE "
_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
_DURATION_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*$", re.I)
_MAX_DURATION_S = 24 * 60 * 60
_state_lock = threading.Lock()
_override_configured = False
_start_ts_us = 0
_deadline_us = 0


def parse_duration_seconds(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("duration must be a number or a string such as '30s'")
    match = _DURATION_PATTERN.match(str(value))
    if not match:
        raise ValueError(
            "duration must use seconds, minutes, or hours, for example '30s'"
        )
    duration = float(match.group(1))
    multiplier = {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2).lower()]
    duration *= multiplier
    if duration <= 0 or duration > _MAX_DURATION_S:
        raise ValueError(f"duration must be within (0, {_MAX_DURATION_S}] seconds")
    return duration


def configure_request_trace(
    enable: bool,
    *,
    duration: Any = "30s",
    now_us: Optional[int] = None,
) -> dict[str, Any]:
    global _override_configured, _start_ts_us, _deadline_us
    current_us = time.time_ns() // 1_000 if now_us is None else int(now_us)
    with _state_lock:
        _override_configured = True
        if enable:
            duration_s = parse_duration_seconds(duration)
            _start_ts_us = current_us
            _deadline_us = current_us + int(duration_s * 1_000_000)
        else:
            _start_ts_us = 0
            _deadline_us = 0
    return request_trace_status(now_us=current_us)


def set_request_trace_deadline_us(
    deadline_us: int, *, start_ts_us: Optional[int] = None
) -> None:
    """Apply an already-normalized deadline, primarily for control-plane/tests."""
    global _override_configured, _start_ts_us, _deadline_us
    with _state_lock:
        _override_configured = True
        _start_ts_us = (
            (time.time_ns() // 1_000 if start_ts_us is None else int(start_ts_us))
            if deadline_us > 0
            else 0
        )
        _deadline_us = max(0, int(deadline_us))


def reset_request_trace_override() -> None:
    global _override_configured, _start_ts_us, _deadline_us
    with _state_lock:
        _override_configured = False
        _start_ts_us = 0
        _deadline_us = 0


def request_trace_status(*, now_us: Optional[int] = None) -> dict[str, Any]:
    current_us = time.time_ns() // 1_000 if now_us is None else int(now_us)
    with _state_lock:
        override_configured = _override_configured
        start_ts_us = _start_ts_us
        deadline_us = _deadline_us
    env_enabled = (
        os.environ.get("ENABLE_REQUEST_TIMELINE_LOG", "").strip().lower()
        in _TRUE_VALUES
    )
    if override_configured:
        enabled = deadline_us > current_us
        source = "request_trace_api"
    else:
        enabled = env_enabled
        source = "environment" if env_enabled else "disabled"
    return {
        "enabled": enabled,
        "source": source,
        "start_ts_us": start_ts_us if enabled and override_configured else None,
        "expires_ts_us": deadline_us if enabled and override_configured else None,
        "remaining_seconds": (
            max(0.0, (deadline_us - current_us) / 1_000_000)
            if enabled and override_configured
            else None
        ),
    }


def timeline_enabled() -> bool:
    return bool(request_trace_status()["enabled"])


def log_timeline_event(
    component: str,
    event: str,
    *,
    request_id: Optional[int] = None,
    ts_us: Optional[int] = None,
    logger: Any = logging,
    force: bool = False,
    **fields: Any,
) -> None:
    """Emit one compact JSON event that can be merged across PG/backend logs."""
    if not force and not timeline_enabled():
        return

    record = {
        "schema_version": 1,
        "component": component,
        "event": event,
        "ts_us": time.time_ns() // 1_000 if ts_us is None else int(ts_us),
    }
    if request_id is not None:
        record["request_id"] = int(request_id)
    record.update(fields)
    logger.info(
        "%s%s",
        TIMELINE_LOG_PREFIX,
        json.dumps(record, separators=(",", ":"), sort_keys=True),
    )


@contextmanager
def timeline_phase(
    component: str,
    phase: str,
    *,
    request_id: Optional[int] = None,
    logger: Any = logging,
    **fields: Any,
) -> Iterator[None]:
    """Log matching phase_start/phase_end events with wall time and duration."""
    if not timeline_enabled():
        yield
        return

    start_ts_us = time.time_ns() // 1_000
    start_ns = time.perf_counter_ns()
    log_timeline_event(
        component,
        "phase_start",
        request_id=request_id,
        ts_us=start_ts_us,
        logger=logger,
        force=True,
        phase=phase,
        **fields,
    )
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        log_timeline_event(
            component,
            "phase_end",
            request_id=request_id,
            logger=logger,
            force=True,
            phase=phase,
            status=status,
            duration_us=(time.perf_counter_ns() - start_ns) // 1_000,
            **fields,
        )
