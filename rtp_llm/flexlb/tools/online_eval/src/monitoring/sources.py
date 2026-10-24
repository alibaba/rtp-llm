"""Registered evidence capabilities; definitions select them, not case names."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


@dataclass(frozen=True)
class EvidenceSource:
    factory: Callable[[object, set[str]], dict]
    fields: frozenset[str]
    label: str
    required_fields: frozenset[str] = frozenset()


def selected_sources(plan: dict, sources: Mapping[str, EvidenceSource] | None = None) -> dict:
    sources = {} if sources is None else sources
    selected: dict = {}
    for identity, spec in plan["produced"].items():
        dependency = spec.get("collection")
        if dependency is None:
            continue
        if (
            not isinstance(dependency, dict)
            or set(dependency) != {"source", "field"}
            or dependency["source"] not in sources
            or type(dependency["field"]) is not str
            or not dependency["field"]
        ):
            raise ValueError("invalid collection dependency: " + identity)
        capability = sources[dependency["source"]]
        allowed = capability.fields
        if not capability.required_fields <= allowed or dependency["field"] not in allowed:
            raise ValueError("unknown collection field: " + identity)
        source = selected.setdefault(dependency["source"], dict(fields=sorted(capability.required_fields), metric_ids=[]))
        if dependency["field"] not in source["fields"]:
            source["fields"].append(dependency["field"])
        source["metric_ids"].append(identity)
    for name, spec in selected.items():
        spec["fields"] = sorted(spec["fields"])
        spec["metric_ids"] = sorted(spec["metric_ids"])
        capability = sources[name]
        path = Path(inspect.getfile(capability.factory))
        spec["implementation"] = dict(
            module=capability.factory.__module__,
            factory=capability.factory.__name__,
            label=capability.label,
            transport="prometheus",
            collector_sha256=hashlib.sha256(Path(__file__).with_name("probe.py").read_bytes()).hexdigest(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return selected
