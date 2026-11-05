"""DashScope TPM usage-callback recheck after the early first packet."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor

logger = logging.getLogger(__name__)

DEFAULT_TPM_RECHECK_TIMEOUT_MS = 1000
_MAX_BODY_CHARS = 512


@dataclass(frozen=True)
class TpmRecheckResult:
    """Outcome of one usage-callback probe."""

    ok: bool
    callback_status_code: int
    status_name: str
    message: str
    callback_error_code: str = ""


def _default_status_name(status_code: int) -> str:
    if status_code == 429:
        return "Throttling.ResourceExhausted"
    if status_code == 503:
        return "ServiceUnavailable"
    if status_code == 504:
        return "GatewayTimeout"
    return "InternalError"


def _parse_callback_error(status_code: int, body: str) -> tuple[str, str, str]:
    """Return ``(status_name, message, callback_error_code)``.

    The callback is an internal gateway API; when it returns a DashScope-style
    JSON error, preserve its status name/message/code instead of inventing one.
    """
    payload: Any = None
    if body:
        try:
            payload = json.loads(body)
        except (TypeError, ValueError):
            payload = None
    if not isinstance(payload, dict):
        return _default_status_name(status_code), body[:_MAX_BODY_CHARS], ""

    raw_code = payload.get(
        "status_code", payload.get("code", payload.get("error_code"))
    )
    raw_name = payload.get(
        "status_name", payload.get("error_name", payload.get("type"))
    )
    raw_message = payload.get(
        "status_message",
        payload.get("message", payload.get("error_message", payload.get("error"))),
    )
    callback_error_code = "" if raw_code is None else str(raw_code)
    status_name = (
        raw_name
        if isinstance(raw_name, str) and raw_name
        else _default_status_name(status_code)
    )
    if isinstance(raw_message, str) and raw_message:
        message = raw_message
    elif raw_message is not None:
        message = str(raw_message)
    else:
        message = body[:_MAX_BODY_CHARS]
    return status_name, message, callback_error_code


async def recheck_tpm(
    callback_url: str,
    *,
    timeout_ms: int = DEFAULT_TPM_RECHECK_TIMEOUT_MS,
    request_id: str = "",
) -> TpmRecheckResult:
    """GET the gateway usage callback; 200 means the request may continue."""
    import aiohttp

    timeout_s = max(0.001, timeout_ms / 1000.0)
    started = time.monotonic()
    kmonitor.report(AccMetrics.DASH_SC_TPM_RECHECK_QPS_METRIC, 1)
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(callback_url) as response:
                body = await response.text()
                status_code = response.status
                elapsed_ms = (time.monotonic() - started) * 1000.0
                kmonitor.report(GaugeMetrics.DASH_SC_TPM_RECHECK_RT_METRIC, elapsed_ms)
                if status_code == 200:
                    return TpmRecheckResult(
                        ok=True,
                        callback_status_code=status_code,
                        status_name="",
                        message="",
                    )
                status_name, message, callback_error_code = _parse_callback_error(
                    status_code, body
                )
                metric = (
                    AccMetrics.DASH_SC_TPM_RECHECK_LIMIT_QPS_METRIC
                    if status_code == 429
                    else AccMetrics.DASH_SC_TPM_RECHECK_ERROR_QPS_METRIC
                )
                kmonitor.report(metric, 1, {"status_code": str(status_code)})
                logger.warning(
                    "TPM callback rejected request_id=%s status=%s name=%s message=%s",
                    request_id,
                    status_code,
                    status_name,
                    message,
                )
                return TpmRecheckResult(
                    ok=False,
                    callback_status_code=status_code,
                    status_name=status_name,
                    message=message,
                    callback_error_code=callback_error_code,
                )
    except asyncio.TimeoutError:
        elapsed_ms = (time.monotonic() - started) * 1000.0
        kmonitor.report(GaugeMetrics.DASH_SC_TPM_RECHECK_RT_METRIC, elapsed_ms)
        kmonitor.report(
            AccMetrics.DASH_SC_TPM_RECHECK_ERROR_QPS_METRIC,
            1,
            {"status_code": "timeout"},
        )
        logger.warning(
            "TPM callback timed out request_id=%s timeout_ms=%s",
            request_id,
            timeout_ms,
        )
        return TpmRecheckResult(
            ok=False,
            callback_status_code=504,
            status_name="GatewayTimeout",
            message=f"TPM usage callback timed out after {timeout_ms}ms",
        )
    except Exception as error:
        elapsed_ms = (time.monotonic() - started) * 1000.0
        kmonitor.report(GaugeMetrics.DASH_SC_TPM_RECHECK_RT_METRIC, elapsed_ms)
        kmonitor.report(
            AccMetrics.DASH_SC_TPM_RECHECK_ERROR_QPS_METRIC,
            1,
            {"status_code": "error"},
        )
        logger.warning(
            "TPM callback failed request_id=%s: %s: %s",
            request_id,
            type(error).__name__,
            error,
            exc_info=True,
        )
        return TpmRecheckResult(
            ok=False,
            callback_status_code=500,
            status_name="InternalError",
            message=f"TPM usage callback failed: {type(error).__name__}: {error}",
        )
