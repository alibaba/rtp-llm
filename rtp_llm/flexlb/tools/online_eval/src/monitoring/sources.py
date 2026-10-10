"""Registered evidence capabilities; definitions select them, not case names."""

from importlib import import_module
import hashlib
from pathlib import Path

from monitoring.collectors import EvidenceCollector


SOURCES = {"master_inflight": ("cases.master_ha_failover.observation", "master_adapters", "STATE_FIELDS")}


def selected_sources(plan):
    selected = {}
    for identity, spec in plan["produced"].items():
        dependency = spec.get("collection")
        if dependency is None:
            continue
        if (not isinstance(dependency, dict) or set(dependency) != {"source", "field"}
                or dependency["source"] not in SOURCES
                or type(dependency["field"]) is not str or not dependency["field"]):
            raise ValueError("invalid collection dependency: " + identity)
        module, _, fields = SOURCES[dependency["source"]]
        if dependency["field"] not in getattr(import_module(module), fields):
            raise ValueError("unknown collection field: " + identity)
        source = selected.setdefault(dependency["source"], dict(fields=[], metric_ids=[]))
        if dependency["field"] not in source["fields"]:
            source["fields"].append(dependency["field"])
        source["metric_ids"].append(identity)
    for name, spec in selected.items():
        module, factory, _ = SOURCES[name]
        path = Path(import_module(module).__file__)
        spec["implementation"] = dict(module=module, factory=factory,
                                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return selected


def evidence_collector(plan, source, environment, path, **options):
    selected = selected_sources(plan)
    if source not in selected:
        raise ValueError("evidence source is not selected: " + source)
    module, factory, _ = SOURCES[source]
    adapters = getattr(import_module(module), factory)(environment, set(selected[source]["fields"]))
    return EvidenceCollector(path, adapters, **options)
