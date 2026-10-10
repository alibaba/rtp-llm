"""Attempt independent resource cleanup and retain every failure."""

from subprocess import TimeoutExpired


def cleanup_all(operations):
    errors = []
    for name, operation in operations:
        try:
            operation()
        except Exception as exc:
            errors.append((name, exc))
    if errors:
        message = "; ".join(f"{name}: {type(exc).__name__}: {exc}" for name, exc in errors)
        timed_out = any(isinstance(exc, (TimeoutError, TimeoutExpired)) for _, exc in errors)
        error_type = TimeoutError if timed_out else RuntimeError
        raise error_type(message) from errors[0][1]
