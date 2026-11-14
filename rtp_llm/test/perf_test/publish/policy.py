"""Required/optional publisher policy and persisted publication outcomes."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Union

from rtp_llm.test.perf_test.errors import PublisherError
from rtp_llm.test.perf_test.run_directory import RunDirectoryLock
from rtp_llm.test.perf_test.storage import atomic_write_json


@dataclass(frozen=True)
class ConfiguredPublisher:
    name: str
    required: bool
    publish: Callable[[], Mapping[str, Any]]


def _read_object(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublisherError(
            f"cannot read publication artifact {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise PublisherError(f"publication artifact must be an object: {path}")
    return value


def run_publishers(
    root: Union[str, Path], publishers: Sequence[ConfiguredPublisher]
) -> Dict[str, Any]:
    """Run publishers, persist outcomes, then enforce required policy."""
    root_path = Path(root)
    names = [publisher.name for publisher in publishers]
    if any(not isinstance(name, str) or not name for name in names):
        raise PublisherError("publisher names must be non-empty strings")
    if len(names) != len(set(names)):
        raise PublisherError("publisher names must be unique")

    with RunDirectoryLock(root_path, action="publish"):
        outcomes = []
        for publisher in publishers:
            try:
                details = publisher.publish()
                if not isinstance(details, Mapping):
                    raise TypeError("publisher outcome must be an object")
                outcomes.append(
                    {
                        "name": publisher.name,
                        "required": publisher.required,
                        "status": "succeeded",
                        "details": copy.deepcopy(dict(details)),
                        "error": None,
                    }
                )
            except Exception as error:
                outcomes.append(
                    {
                        "name": publisher.name,
                        "required": publisher.required,
                        "status": "failed",
                        "details": None,
                        "error": str(error),
                    }
                )

        required_failure = any(
            item["required"] and item["status"] == "failed" for item in outcomes
        )
        optional_failure = any(
            not item["required"] and item["status"] == "failed" for item in outcomes
        )
        status = (
            "failed_required"
            if required_failure
            else "failed_optional" if optional_failure else "succeeded"
        )
        publication = {
            "status": status,
            "required_failure": required_failure,
            "outcomes": outcomes,
        }

        manifest_path = root_path / "manifest.json"
        summary_path = root_path / "results" / "summary.json"
        manifest = _read_object(manifest_path)
        summary = _read_object(summary_path)
        manifest["publication"] = copy.deepcopy(publication)
        summary["publication"] = copy.deepcopy(publication)
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(summary_path, summary)

    if required_failure:
        failures = ", ".join(
            f"{item['name']}: {item['error']}"
            for item in outcomes
            if item["required"] and item["status"] == "failed"
        )
        raise PublisherError(f"required publisher failed: {failures}")
    return publication
