"""Describe executable implementations independently of measurement domains."""

def implementation_identity(function):
    """Execution identity is independent of human-readable measurement semantics."""
    if not callable(function):
        raise ValueError("implementation requires a callable")
    return dict(module=function.__module__, function=function.__name__)

