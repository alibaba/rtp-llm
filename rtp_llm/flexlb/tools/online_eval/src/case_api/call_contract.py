"""Pure callable admission, without loading execution or compiler contracts."""

from inspect import signature


def validate_call(function, argument_count, path):
    """Reject incompatible positional calls at discovery, never run user code."""
    try:
        signature(function).bind(*(object() for _ in range(argument_count)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: callable must accept {argument_count} positional arguments: {exc}") from exc
