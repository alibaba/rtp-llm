"""Registered evidence capabilities; definitions select them, not case names."""

import hashlib
from pathlib import Path
import inspect

from case_api.registry import registry


def selected_sources(plan):
    selected = {}
    for identity, spec in plan["produced"].items():
        dependency = spec.get("collection")
        if dependency is None:
            continue
        if (not isinstance(dependency, dict) or set(dependency) != {"source", "field"}
                or dependency["source"] not in registry().sources
                or type(dependency["field"]) is not str or not dependency["field"]):
            raise ValueError("invalid collection dependency: " + identity)
        capability = registry().sources[dependency["source"]]
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
        capability = registry().sources[name]
        path = Path(inspect.getfile(capability.factory))
        spec["implementation"] = dict(module=capability.factory.__module__, factory=capability.factory.__name__,
                                      label=capability.label, transport="prometheus",
                                      collector_sha256=hashlib.sha256(Path(__file__).with_name("probe.py").read_bytes()).hexdigest(),
                                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return selected
