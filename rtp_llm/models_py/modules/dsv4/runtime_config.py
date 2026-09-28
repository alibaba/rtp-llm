"""Strict boolean parsing shared by the DSV4 construction adapters."""

_TRUTHY = ("1", "true", "on", "yes")
_FALSY = ("0", "false", "off", "no")


def parse_bool(raw: str, default=None) -> bool:
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(f"expected one of {sorted(_TRUTHY + _FALSY)}")
