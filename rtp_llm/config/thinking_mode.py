from typing import Any

THINK_MODE_DISABLED = "disabled"
THINK_MODE_ADAPTIVE = "adaptive"
THINK_MODE_ENABLED = "enabled"
THINK_MODE_VALUES = (
    THINK_MODE_DISABLED,
    THINK_MODE_ADAPTIVE,
    THINK_MODE_ENABLED,
)


def normalize_think_mode(value: Any) -> str:
    """Return the canonical THINK_MODE string, including legacy 0/1 aliases."""

    if isinstance(value, bool):
        return THINK_MODE_ENABLED if value else THINK_MODE_DISABLED

    normalized = str(value).strip().lower()
    aliases = {
        "0": THINK_MODE_DISABLED,
        "1": THINK_MODE_ENABLED,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in THINK_MODE_VALUES:
        expected = ", ".join((*THINK_MODE_VALUES, "0", "1"))
        raise ValueError(f"invalid THINK_MODE {value!r}; expected one of: {expected}")
    return normalized
