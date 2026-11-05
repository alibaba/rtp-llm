"""Pin one pure-Python training dependency without changing any interpreter.

Only the current Python 3.11 training process receives this sys.path entry.
os.environ/PYTHONPATH is never modified: RTP's Python 3.10 exec children retain
their established software stack. Safe before Ray assigns final GPU masks.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

WHEEL_SHA256 = "54058201ac7087911181bfec4af6091bb59380360f069276601256a76af08193"
MANIFEST_SHA256 = "8941c4a78f1b3bced364b869cfeda6e20aabaa8465e11df200a9095d7cf68169"


def ensure_training_dependencies(project_root=None):
    if sys.version_info[:2] != (3, 11):
        return {"installed": False, "reason": "training_python311_only"}
    root = Path(project_root or os.environ.get("RTP_PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))
    dependency = root / "logs" / "dependencies" / "einops-0.8.2"
    manifest_path = dependency / "manifest.json"
    raw = manifest_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA256, "Project einops manifest changed"
    manifest = json.loads(raw)
    assert (manifest["name"], manifest["version"]) == ("einops", "0.8.2")
    assert manifest["wheel"]["sha256"] == WHEEL_SHA256
    wheel = dependency / manifest["wheel"]["path"]
    wheel_bytes = wheel.read_bytes()
    assert len(wheel_bytes) == manifest["wheel"]["bytes"]
    assert hashlib.sha256(wheel_bytes).hexdigest() == WHEEL_SHA256
    extracted = (dependency / "extracted").resolve()
    for item in manifest["files"]:
        relative = Path(item["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        target = (extracted / relative).resolve()
        assert target.is_relative_to(extracted)
        data = target.read_bytes()
        assert len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"], item["path"]
    loaded = sys.modules.get("einops")
    if loaded is not None:
        assert Path(loaded.__file__).resolve().is_relative_to(extracted), "A different einops is already imported"
    path = str(extracted)
    if path not in sys.path:
        sys.path.insert(0, path)
    return {"installed": True, "name": "einops", "version": "0.8.2", "path": path,
            "wheel_sha256": WHEEL_SHA256, "manifest_sha256": MANIFEST_SHA256,
            "scope": "current_python311_sys_path_only"}
