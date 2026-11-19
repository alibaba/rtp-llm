"""Common profile-session manifests and run-relative trace artifact identities."""

import re
from pathlib import Path, PurePosixPath

PROFILE_SCHEMA_VERSION = 2
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def new_profile_manifest(mode, session_id, case_ids, *, backend, capture_scope):
    manifest = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "session_id": session_id,
        "mode": mode,
        "backend": backend,
        "capture_scope": capture_scope,
        "diagnostic_only": True,
        "status": "running",
        "error": None,
        "cases": [
            {
                "case_id": case_id,
                "trace_name": case_id,
                "status": "pending",
                "error": None,
                "artifacts": [],
            }
            for case_id in case_ids
        ],
    }
    return parse_profile_manifest(manifest)


def parse_profile_manifest(value):
    """Validate only the current contract; no old layout or schema fallback."""
    if not isinstance(value, dict) or (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != PROFILE_SCHEMA_VERSION
    ):
        raise ValueError("expected profile manifest schema 2")
    if value.get("mode") not in ("prefill", "decode") or value.get("backend") not in (
        "kineto",
        "nsys",
    ):
        raise ValueError("invalid profile mode/backend")
    if not isinstance(value.get("session_id"), str) or not _NAME.fullmatch(
        value["session_id"]
    ):
        raise ValueError("invalid profile session_id")
    if (
        value.get("diagnostic_only") is not True
        or not isinstance(value.get("capture_scope"), str)
        or not value["capture_scope"]
    ):
        raise ValueError("profile requires diagnostic capture scope")
    if value.get("status") not in ("running", "completed", "failed", "interrupted"):
        raise ValueError("invalid profile status")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("profile requires nonempty cases")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("invalid profile case")
        missing = {"case_id", "trace_name", "status", "error", "artifacts"} - set(case)
        if missing:
            raise ValueError(f"profile case missing fields: {sorted(missing)}")
        if (
            not isinstance(case["trace_name"], str)
            or not _NAME.fullmatch(case["trace_name"])
            or len(case["trace_name"]) > 64
        ):
            raise ValueError(
                "profile trace_name must be a safe prefix of at most 64 characters"
            )
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("case_id"), str)
            or not _NAME.fullmatch(case["case_id"])
        ):
            raise ValueError("invalid profile case_id")
        if case["case_id"] in seen:
            raise ValueError("duplicate profile case_id")
        seen.add(case["case_id"])
        if case.get("status") not in (
            "pending",
            "completed",
            "failed",
        ) or not isinstance(case.get("artifacts"), list):
            raise ValueError("invalid profile case status/artifacts")
        if case["status"] == "completed" and not case["artifacts"]:
            raise ValueError("completed profile case requires trace artifacts")
        if case.get("error") is not None and (
            not isinstance(case["error"], str) or not case["error"]
        ):
            raise ValueError("invalid profile case error")
        for artifact in case["artifacts"]:
            if not isinstance(artifact, dict):
                raise ValueError("invalid profile trace artifact")
            missing = {"path", "format", "rank", "phase"} - set(artifact)
            if missing:
                raise ValueError(
                    f"profile trace artifact missing fields: {sorted(missing)}"
                )
            if not isinstance(artifact, dict) or not isinstance(
                artifact.get("path"), str
            ):
                raise ValueError("invalid profile trace path")
            if artifact.get("format") not in ("chrome_trace", "nsys"):
                raise ValueError("unsupported profile trace format")
            rank = artifact.get("rank")
            if rank is not None and (type(rank) is not int or rank < 0):
                raise ValueError("invalid profile trace rank")
            if artifact.get("phase") not in (None, "prefill", "decode"):
                raise ValueError("invalid profile trace phase")
    return value


def profile_trace_path(root, session_id, case_id, relative_path):
    """Reject unlisted locations and symlink escapes, also for copied run trees."""
    root = Path(root).resolve()
    path = PurePosixPath(relative_path)
    expected = PurePosixPath("traces", session_id, case_id)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != relative_path
        or not path.is_relative_to(expected)
    ):
        raise ValueError("trace path must belong to its run/session/case")
    target = root.joinpath(*path.parts)
    if any(
        parent.is_symlink()
        for parent in (target, *target.parents)
        if parent != root and root in parent.parents
    ):
        raise ValueError("profile trace symlink is forbidden")
    if not target.resolve().is_relative_to(root / "traces" / session_id / case_id):
        raise ValueError("trace path escapes its run/session/case")
    if not target.is_file() or target.stat().st_size == 0:
        raise ValueError(f"missing or empty profile trace: {relative_path}")
    return target


def trace_artifact(root, session_id, case_id, path, *, backend):
    relative = Path(path).absolute().relative_to(Path(root).absolute()).as_posix()
    target = profile_trace_path(root, session_id, case_id, relative)
    suffix = ".json" if backend == "kineto" else ".nsys-rep"
    if target.suffix != suffix:
        raise ValueError(f"expected {backend} trace suffix {suffix}")
    rank = re.search(r"(?:^|_)wr(\d+)(?:_|\.)", target.name)
    phase = re.search(r"(?:^|_)prefill([01])(?:_|\.)", target.name)
    return {
        "path": relative,
        "format": "chrome_trace" if backend == "kineto" else "nsys",
        "rank": int(rank.group(1)) if rank else None,
        "phase": ("prefill" if phase.group(1) == "1" else "decode") if phase else None,
    }
