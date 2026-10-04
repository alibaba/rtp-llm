import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

UNSUPPORTED_LIFECYCLE_CONTROL_FIELDS = (
    "phase",
    "prepare_only",
    "commit_only",
    "drain_only",
    "quiesce_token",
    "expected_incarnation",
    "expected_sleep_epoch",
    "cancel_quiesce_token",
    "resume_metrics_only",
    "freeze_only",
    "target_round",
)


def unsupported_lifecycle_control_field(req: Dict[Any, Any]) -> Optional[str]:
    for field in UNSUPPORTED_LIFECYCLE_CONTROL_FIELDS:
        if field in req:
            return field
    return None


def dedupe_addresses(addresses: List[str]) -> List[str]:
    deduped: List[str] = []
    seen: set = set()
    for address in addresses:
        if address in seen:
            continue
        seen.add(address)
        deduped.append(address)
    return deduped


@dataclass(frozen=True)
class SleepOptions:
    level: int
    mode: str
    timeout_ms: int


def normalize_lifecycle_request(req: Any) -> Dict[str, Any]:
    if isinstance(req, str):
        req = json.loads(req)
    if req is None:
        return {}
    if not isinstance(req, dict):
        raise ValueError("request body must be a JSON object")
    return req


def validate_sleep_request(req: Dict[Any, Any]) -> SleepOptions:
    """Shared HTTP/direct-client validation; backend decides capabilities."""
    unsupported = unsupported_lifecycle_control_field(req)
    if unsupported:
        raise ValueError(f"sleep {unsupported} is unsupported")
    try:
        level = int(req.get("level", 1))
        # Preserve the existing one-hour drain budget when omitted.
        timeout_ms = int(req.get("timeout_ms", 60 * 60 * 1000))
    except (TypeError, ValueError) as error:
        raise ValueError("sleep level and timeout_ms must be integers") from error
    if level not in (0, 1, 2):
        raise ValueError("sleep level must be 0, 1 or 2")
    mode = req.get("mode", "wait")
    if mode not in ("wait", "abort"):
        raise ValueError('sleep mode must be "wait" or "abort"')
    tags = req.get("tags")
    if tags is None:
        tags = []
    if not isinstance(tags, list):
        raise ValueError("sleep tags must be a list")
    if any(not isinstance(tag, str) or not tag for tag in tags):
        raise ValueError("sleep tags must be non-empty strings")
    if tags:
        raise ValueError(
            "non-empty sleep tags are unsupported; partial sleep is not implemented"
        )
    return SleepOptions(level=level, mode=mode, timeout_ms=timeout_ms)
