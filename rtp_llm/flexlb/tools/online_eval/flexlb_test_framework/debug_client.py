"""Read-only Master debug adapter. Missing/partial evidence never becomes empty success."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

MAX_BYTES = 8 * 1024 * 1024


class DebugUnavailable(RuntimeError):
    """Required observation failed or is incomplete (stage adapters map to ERROR)."""


@dataclass(frozen=True)
class Capture:
    payload: dict
    started_mono: float
    finished_mono: float

    def component(self, name: str) -> dict:
        page = self.payload["components"].get(name)
        if page is None or page["status"] != "ok" or page["truncated"]:
            raise DebugUnavailable(f"component {name!r} missing or incomplete")
        return page


def validate_snapshot(payload: object) -> dict:
    """Fail closed on version/schema drift; retain source-specific status and clocks."""
    try:
        if not isinstance(payload, dict) or payload["schemaVersion"] != 1:
            raise ValueError("unsupported schema")
        if payload["status"] not in ("ok", "partial"):
            raise ValueError("invalid snapshot status")
        for field in ("instanceId", "snapshotId", "endpointScope"):
            if not isinstance(payload[field], str) or not payload[field]:
                raise ValueError(f"missing {field}")
        if type(payload["endpointDirectoryTruncated"]) is not bool:
            raise ValueError("missing directory coverage")
        if not isinstance(payload["components"], dict) or not payload["components"]:
            raise ValueError("missing components")
        for field in ("captureStartedAtMs", "captureFinishedAtMs"):
            if type(payload[field]) is not int:
                raise ValueError(f"invalid {field}")
        for page in payload["components"].values():
            if page["status"] not in (
                "ok",
                "partial",
                "busy",
                "unavailable",
                "budget_exhausted",
                "generation_changed",
                "not_applicable",
            ):
                raise ValueError("invalid component status")
            if (
                type(page["truncated"]) is not bool
                or type(page["scannedCount"]) is not int
            ):
                raise ValueError("missing sampling coverage")
            if not isinstance(page["consistency"], str) or not isinstance(
                page["metadata"], dict
            ):
                raise ValueError("missing source metadata")
            for field in ("captureStartedAtMs", "captureFinishedAtMs"):
                if type(page[field]) is not int:
                    raise ValueError("missing source clock")
            if not isinstance(page["rows"], list):
                raise ValueError("missing rows")
            for row in page["rows"]:
                if not isinstance(row, dict) or not isinstance(row["request_id"], str):
                    raise ValueError("request identity must be a string")
    except (KeyError, TypeError, ValueError) as error:
        raise DebugUnavailable(f"invalid debug response: {error}") from error
    return payload


class DebugClient:
    def __init__(self, base_url: str, timeout_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s

    def snapshot(
        self,
        *,
        request_id: int | None = None,
        include: str = "scheduler",
        limit: int = 500,
        scan_limit: int = 2000,
        endpoint_limit: int = 64,
    ) -> Capture:
        path = (
            "/rtp_llm/debug/snapshot"
            if request_id is None
            else f"/rtp_llm/debug/requests/{int(request_id)}"
        )
        params = urllib.parse.urlencode(
            dict(
                include=include,
                limit=limit,
                scan_limit=scan_limit,
                endpoint_limit=endpoint_limit,
            )
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(
                f"{self.base_url}{path}?{params}", timeout=self.timeout_s
            ) as response:
                body = response.read(MAX_BYTES + 1)
                if len(body) > MAX_BYTES:
                    raise DebugUnavailable("response exceeds byte budget")
                payload = validate_snapshot(json.loads(body))
        except (OSError, ValueError) as error:
            raise DebugUnavailable(f"debug capture failed: {error}") from error
        return Capture(payload, started, time.monotonic())


def check_scheduler_tombstone(row: dict) -> tuple[bool, str]:
    """Only the scheduler's local storage invariant; does not prove Engine release."""
    required = (
        "storage_phase",
        "lifecycle_phase",
        "admission_open",
        "has_item",
        "has_engine_fence",
        "has_preemption",
        "has_admission_mutation",
        "has_request_deadline",
        "has_decision_deadline",
        "has_inactivity_deadline",
        "has_cancel_reason",
        "has_pending_admission_cancel",
    )
    if any(key not in row for key in required):
        raise DebugUnavailable("missing tombstone ownership fields")
    if row["storage_phase"] != "TOMBSTONE":
        raise DebugUnavailable("request is not a scheduler tombstone")
    if any(type(row[key]) is not bool for key in required[2:]):
        raise DebugUnavailable("invalid tombstone ownership fields")
    retained = [key for key in required[2:] if row[key]]
    terminal = row["lifecycle_phase"] in (
        "CANCELLED",
        "TIMED_OUT",
        "FAILED",
        "COMPLETED",
    )
    return (
        terminal and not retained,
        f"scheduler tombstone terminal={terminal}, retained={retained}",
    )
