"""Stable identity fingerprints shared by prefill initialization and resume."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from rtp_llm.test.perf_test.config_resolver import canonical_json
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import ExecutionPlan
from rtp_llm.test.perf_test.prefill.measurement import PREFILL_MEASUREMENT_CONTRACT

RUNNER_CONTRACT = {
    "name": "unified-prefill-runner",
    "version": 3,
    "warmup_runs": 1,
    "prompt_contract": "exact-token-prefix-v1",
    "measurement_contract": PREFILL_MEASUREMENT_CONTRACT,
}


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


_ENGINE_LIBRARIES = (
    "libth_transformer.so",
    "librtp_compute_ops.so",
    "libth_transformer_config.so",
)
_SOURCE_SUFFIXES = {
    ".py",
    ".cc",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cu",
    ".cuh",
    ".hip",
    ".bzl",
}
_FILE_DIGESTS: Dict[Path, tuple] = {}


def _file_digest(path: Path) -> str:
    path = path.resolve(strict=True)
    before = path.stat()
    signature = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    cached = _FILE_DIGESTS.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if signature != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise PlanningError(
            "engine identity changed while reading it; retry with a stable build"
        )
    value = digest.hexdigest()
    _FILE_DIGESTS[path] = signature, value
    return value


def _git_revision(root: Path):
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _source_identity(root: Path) -> Dict[str, Any]:
    def fail_walk(error):
        raise error

    files = {}
    for current, directories, names in os.walk(root, onerror=fail_walk):
        directories[:] = sorted(
            name
            for name in directories
            if name not in {"test", "tests", "__pycache__", ".git", "libs"}
        )
        for name in sorted(names):
            path = Path(current) / name
            if path.suffix in _SOURCE_SUFFIXES or name in {"BUILD", "WORKSPACE"}:
                files[str(path.relative_to(root))] = _file_digest(path)
    return {"path": str(root), "revision": _git_revision(root), "sources": files}


def _engine_fingerprint() -> str:
    # Inspect the actual imported package without importing torch or native ops.
    package = sys.modules.get("rtp_llm")
    package_file = vars(package).get("__file__") if package is not None else None
    root = (
        Path(package_file).resolve().parent
        if package_file
        else Path(__file__).resolve().parents[3]
    )
    try:
        sources = [_source_identity(root)]
        internal = root.parent / "internal_source" / "rtp_llm"
        if internal.is_dir():
            sources.append(_source_identity(internal.resolve()))
        ops = sys.modules.get("rtp_llm.ops")
        loaded_path = vars(ops).get("so_path") if ops is not None else None
        if loaded_path is not None:
            library_root = Path(loaded_path).resolve(strict=True)
            if not (library_root / _ENGINE_LIBRARIES[2]).is_file():
                raise PlanningError(
                    "loaded engine library directory is unavailable; start a new run"
                )
        else:
            # Match the loader's package-local preference; never search sibling
            # worktrees or assemble one build from several artifact directories.
            candidates = (
                root / "libs",
                root.parent / "bazel-bin",
                root.parent / "bazel-bin" / "rtp_llm" / "libs",
            )
            library_root = next(
                (
                    path.resolve()
                    for path in candidates
                    if (path / _ENGINE_LIBRARIES[2]).is_file()
                ),
                None,
            )
        libraries = {
            name: (
                _file_digest(library_root / name)
                if library_root is not None and (library_root / name).is_file()
                else None
            )
            for name in _ENGINE_LIBRARIES
        }
        if not any(source["sources"] for source in sources) and not any(
            libraries.values()
        ):
            raise PlanningError(
                "cannot establish engine source or build identity; start a new run"
            )
        return _sha(
            {
                "sources": sources,
                "library_path": str(library_root),
                "libraries": libraries,
            }
        )
    except OSError as error:
        raise PlanningError(
            f"cannot establish engine identity: {error}; start a new run"
        ) from error


def resume_fingerprints(plan: ExecutionPlan) -> Dict[str, str]:
    return {
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
        "model_fingerprint": _sha(plan.to_dict()["resolved_config"]["model"]),
        "runner_fingerprint": _sha(RUNNER_CONTRACT),
        "engine_fingerprint": _engine_fingerprint(),
    }
