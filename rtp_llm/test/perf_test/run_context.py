"""File-based run metadata shared by CI, SSH launchers and workers.

Only the descriptor path crosses process boundaries through an explicit argument.
File references are relative to the descriptor, so a bundle can be copied to
another host without rewriting its contents. Credentials never belong here.
The reader also runs as a standalone script before the engine is available.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional, Union

_PATH_FIELDS = ("config_path", "resolved_workload_path")
_FIELDS = frozenset(
    {"schema_version", "mode", "run_id", "repository_revisions", *_PATH_FIELDS}
)


def read_run_context(
    path: Optional[Union[str, Path]] = None, *, required: bool = False
) -> dict[str, Any]:
    source = path
    if not source:
        if required:
            raise ValueError("--run-context must name a run descriptor")
        return {}
    source = Path(source).resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read run descriptor {source}: {error}") from error
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("run descriptor must be a schema_version=1 object")
    if set(value) - _FIELDS:
        raise ValueError(
            f"unknown run descriptor fields: {sorted(set(value) - _FIELDS)}"
        )
    if "mode" in value and value["mode"] not in ("prefill", "decode"):
        raise ValueError("run descriptor mode must be prefill or decode")
    if "run_id" in value and (
        not isinstance(value["run_id"], str)
        or re.fullmatch(r"[A-Za-z0-9_-]+", value["run_id"]) is None
    ):
        raise ValueError(
            "run descriptor run_id must contain only letters, digits, underscores and hyphens"
        )
    for field in _PATH_FIELDS:
        if field in value:
            if not isinstance(value[field], str) or not value[field].strip():
                raise ValueError(f"run descriptor {field} must be a non-empty path")
            value[field] = str((source.parent / value[field]).resolve())
    revisions = value.get("repository_revisions", {})
    if not isinstance(revisions, dict) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is None
        for name, revision in revisions.items()
    ):
        raise ValueError(
            "repository_revisions must map repository names to full Git commit IDs"
        )
    return value


def write_run_context(path: Union[str, Path], **fields: Any) -> Path:
    from rtp_llm.test.perf_test.storage import atomic_write_json

    destination = Path(path).resolve()
    atomic_write_json(destination, {"schema_version": 1, **fields})
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("field", choices=("run_id", "mode", *_PATH_FIELDS))
    args = parser.parse_args()
    context = read_run_context(args.path, required=True)
    if args.field not in context:
        parser.error(f"run descriptor is missing {args.field}")
    print(context[args.field])


if __name__ == "__main__":
    main()
