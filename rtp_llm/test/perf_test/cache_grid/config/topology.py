"""GPU topology shared by benchmark and production throughput charts."""

import json
import math
from pathlib import Path


def _count(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float("nan")
    if (
        isinstance(value, bool)
        or not math.isfinite(parsed)
        or parsed <= 0
        or not parsed.is_integer()
    ):
        raise ValueError("profile GPU counts must be positive integers")
    return int(parsed)


def profile_cards(profile: dict | None) -> int | None:
    """Use world size or TP x DP x PP; EP and CP do not multiply GPU counts."""
    engine = (profile or {}).get("engine") or {}
    if "world_size" in engine:
        return _count(engine["world_size"])
    if any(key in engine for key in ("tp_size", "dp_size", "pp_size")):
        return math.prod(
            _count(engine.get(key, 1)) for key in ("tp_size", "dp_size", "pp_size")
        )
    return None


def detect_cards(input_path: Path, profile: dict | None = None) -> int | None:
    """Read card topology from an explicit or embedded profile."""
    cards = profile_cards(profile)
    if cards is not None:
        return cards
    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    embedded = data.get("profile")
    return profile_cards(embedded if isinstance(embedded, dict) else None)
