"""Optional platform extension loader for monorepo-only registrations."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import ModuleType
from typing import Optional

log = logging.getLogger(__name__)


def load_platform_extension() -> Optional[ModuleType]:
    """Load a local platform extension module when the monorepo overlay exists."""
    project_root = Path(__file__).resolve().parents[4]
    repo_root = project_root.parent
    rel = Path("internal_source/rtp_llm/models_py/modules/factory/platform_ext.py")
    for base in (repo_root, project_root):
        path = base / rel
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("_rtp_llm_platform_ext", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            log.warning("Failed to load platform extension %s: %s", path, exc)
            return None
        return module
    return None
