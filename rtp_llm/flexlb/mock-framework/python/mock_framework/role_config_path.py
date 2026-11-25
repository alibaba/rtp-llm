"""Resolve role configurations within the directory of the Case that owns them."""

from pathlib import Path
from typing import Optional


def resolve_role_config_path(value: str, directory: Optional[Path], field: str) -> Path:
    path = Path(value)
    if not value.strip() or directory is None or path.is_absolute() or "://" in value:
        raise ValueError(f"{field}: expected a file relative to the Case directory")

    case_dir = Path(directory).resolve()
    try:
        source = (case_dir / path).resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{field}: cannot resolve {value!r}: {error}") from error
    if not source.is_relative_to(case_dir):
        raise ValueError(f"{field}: {value!r} points outside the Case directory")
    return source
