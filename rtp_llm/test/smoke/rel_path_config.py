"""stdlib-only smoke golden-path resolution (``REL_PATH``).

Imported by ``common_def.py`` and by ``scripts/verify_smoke_paths.py`` so the
latter does not need pydantic or pytest.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _monorepo_internal_smoke_dir(abs_pkg: str) -> Optional[str]:
    """If ``abs_pkg`` is .../github-opensource/rtp_llm/test/smoke, map to sibling internal tree."""
    p = Path(abs_pkg).resolve()
    if len(p.parents) < 4:
        return None
    internal = p.parents[3] / "internal_source" / "rtp_llm" / "test" / "smoke"
    if internal.is_dir():
        return str(internal)
    return None


def compute_smoke_rel_path(package_smoke_dir: str) -> str:
    """Return REL_PATH for golden JSON / task_info under ``package_smoke_dir``.

    ``package_smoke_dir`` must be the realpath of the ``smoke`` package directory
    (the parent of ``common_def.py``).

    Honors ``SMOKE_REL_PATH_PREFER`` (``internal`` / ``oss`` / unset) set **before**
    this module is first imported.
    """
    abs_pkg = os.path.realpath(package_smoke_dir)
    prefer = os.environ.get("SMOKE_REL_PATH_PREFER", "")

    if prefer == "internal":
        resolved = _monorepo_internal_smoke_dir(abs_pkg)
        if resolved is not None:
            return resolved
        candidates = [
            "internal_source/rtp_llm/test/smoke",
            "rtp_llm/test/smoke",
            abs_pkg,
        ]
        return next((p for p in candidates if os.path.isdir(p)), abs_pkg)
    if prefer == "oss":
        candidates = [
            abs_pkg,
            "rtp_llm/test/smoke",
            "internal_source/rtp_llm/test/smoke",
        ]
        return next((p for p in candidates if os.path.isdir(p)), abs_pkg)

    candidates = [
        abs_pkg,
        "internal_source/rtp_llm/test/smoke",
        "rtp_llm/test/smoke",
    ]
    return next((p for p in candidates if os.path.isdir(p)), abs_pkg)
