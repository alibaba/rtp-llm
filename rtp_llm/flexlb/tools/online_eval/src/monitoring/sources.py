"""Registered evidence capabilities; definitions select them, not case names."""

from dataclasses import dataclass
from importlib import import_module
import hashlib
from pathlib import Path

from monitoring.probe import PrometheusEvidence


@dataclass(frozen=True)
class EvidenceSource:
    module: str
    factory: str
    fields: str
    label: str
    required_fields: frozenset = frozenset()


SOURCES = {"master_inflight": EvidenceSource("cases.master_ha_failover.observation",
    "master_adapters", "STATE_FIELDS", "master", frozenset({"http_up"}))}


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
        capability = SOURCES[dependency["source"]]
        allowed = getattr(import_module(capability.module), capability.fields)
        if not capability.required_fields <= allowed or dependency["field"] not in allowed:
            raise ValueError("unknown collection field: " + identity)
        source = selected.setdefault(dependency["source"], dict(fields=sorted(capability.required_fields), metric_ids=[]))
        if dependency["field"] not in source["fields"]:
            source["fields"].append(dependency["field"])
        source["metric_ids"].append(identity)
    for name, spec in selected.items():
        capability = SOURCES[name]
        path = Path(import_module(capability.module).__file__)
        spec["implementation"] = dict(module=capability.module, factory=capability.factory,
                                      label=capability.label, transport="prometheus",
                                      collector_sha256=hashlib.sha256(Path(import_module("monitoring.probe").__file__).read_bytes()).hexdigest(),
                                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return selected


def evidence_collector(plan, source, environment, path, **options):
    selected = selected_sources(plan)
    if source not in selected:
        raise ValueError("evidence source is not selected: " + source)
    capability = SOURCES[source]
    adapters = getattr(import_module(capability.module), capability.factory)(environment, set(selected[source]["fields"]))
    return PrometheusEvidence(path, adapters, fields=selected[source]["fields"],
                              label=capability.label, source=source, **options)
