"""Validated, declarative Prometheus queries for a case's monitoring plan."""

import re
from pathlib import Path

from schema_contract import matches_schema

from scenario.loader import ScenarioError, load_document


CATALOG = Path(__file__).resolve().parents[2] / "config/monitoring"
DEFAULT_PLAN = "default.yaml"
SOURCE_KINDS = ("mock", "client", "master")
_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_PLAN = re.compile(r"[a-z][a-z0-9_]*\.yaml\Z")
_TOKENS = re.compile(r"\$\{([^}]+)\}")
PLAN_VERSION = 3


def load_plan(name, _stack=()):
    """Compose data-only metric sets; duplicate IDs and include cycles are errors."""
    if not isinstance(name, str) or not _PLAN.fullmatch(name):
        raise ScenarioError("monitoring.query_plan must name a file in config/monitoring")
    if name in _stack:
        raise ScenarioError("metric set include cycle: " + " -> ".join((*_stack, name)))
    path = CATALOG / name
    data = load_document(path)
    if (set(data) - {"metric_plan_schema_version", "sources", "include", "exclude", "produced"}
            or not matches_schema(data, "metric_plan_schema_version", PLAN_VERSION)):
        raise ScenarioError(f"{path}: invalid query plan header")
    sources = {kind: {} for kind in SOURCE_KINDS}
    produced = {}
    includes = _unique_strings(data.get("include", []), path,
                               "include must contain unique metric set filenames")
    for included in includes:
        plan = load_plan(included, (*_stack, name))
        for kind in SOURCE_KINDS:
            _merge(sources[kind], plan["sources"][kind], path, f"metric id {kind}/")
        _merge(produced, plan["produced"], path, "produced metric id ")
    _add_sources(sources, data.get("sources", {}), path)
    _add_produced(produced, data.get("produced", {}), path)
    _exclude(sources, produced, data.get("exclude", []), path)
    if not any(sources.values()) and not produced:
        raise ScenarioError(f"{path}: metric plan is empty")
    plan = dict(metric_plan_schema_version=PLAN_VERSION, sources=sources, produced=produced)
    definitions(plan)
    return plan


def _unique_strings(value, path, message):
    if (not isinstance(value, list) or any(type(x) is not str for x in value)
            or len(set(value)) != len(value)):
        raise ScenarioError(f"{path}: {message}")
    return value


def _merge(target, additions, path, identity_prefix):
    for metric, definition in additions.items():
        if metric in target:
            raise ScenarioError(f"{path}: duplicate {identity_prefix}{metric}")
        target[metric] = definition


def _add_sources(sources, own, path):
    if not isinstance(own, dict) or set(own) - set(SOURCE_KINDS):
        raise ScenarioError(f"{path}: invalid metric source kinds")
    for kind, queries in own.items():
        if not isinstance(queries, dict):
            raise ScenarioError(f"{path}: {kind} queries must be a mapping")
        for metric, spec in queries.items():
            if type(metric) is not str or not _NAME.fullmatch(metric) or metric == "up":
                raise ScenarioError(f"{path}: invalid query id {kind}/{metric}")
            if metric in sources[kind]:
                raise ScenarioError(f"{path}: duplicate metric id {kind}/{metric}")
            _validate_query(spec, kind, metric, path)
            sources[kind][metric] = spec


def _validate_query(spec, kind, metric, path):
    if not isinstance(spec, dict) or set(spec) - {"promql", "required", "unit", "value_kind", "labels", "mode", "measurement", "exported_metrics"}:
        raise ScenarioError(f"{path}: invalid query definition {kind}/{metric}")
    expression = spec.get("promql")
    if not isinstance(expression, str) or not expression.strip():
        raise ScenarioError(f"{path}: missing PromQL for {kind}/{metric}")
    if "${selector}" not in expression or set(_TOKENS.findall(expression)) - {"selector", "window_ms"}:
        raise ScenarioError(f"{path}: invalid PromQL placeholders for {kind}/{metric}")
    if "required" in spec and type(spec["required"]) is not bool:
        raise ScenarioError(f"{path}: required must be boolean for {kind}/{metric}")
    _metadata(spec, path)
    if kind == "master" and "exported_metrics" not in spec:
        raise ScenarioError(f"{path}: master query requires exported_metrics: {metric}")
    if "exported_metrics" in spec:
        names = _unique_strings(spec["exported_metrics"], path, "exported_metrics must be unique names")
        if not names or any(not re.fullmatch(r"[a-zA-Z_:][a-zA-Z0-9_:]*", name) for name in names):
            raise ScenarioError(f"{path}: invalid exported_metrics")
    if spec.get("mode", "evaluated") not in ("evaluated", "scrape"):
        raise ScenarioError(f"{path}: invalid timestamp mode")
    if spec.get("mode") == "scrape" and not re.fullmatch(r"[a-zA-Z_:][a-zA-Z0-9_:]*\$\{selector\}", expression):
        raise ScenarioError(f"{path}: scrape mode requires a raw metric selector")


def _add_produced(produced, own, path):
    if not isinstance(own, dict):
        raise ScenarioError(f"{path}: produced must be a mapping")
    for metric, spec in own.items():
        if metric in produced:
            raise ScenarioError(f"{path}: invalid produced metric {metric}")
        _validate_produced(spec, metric, path)
        produced[metric] = spec


def _validate_produced(spec, metric, path):
    if (type(metric) is not str or not re.fullmatch(r"[a-z][a-z0-9_]*/[a-z][a-z0-9_]*", metric)
            or not isinstance(spec, dict)
            or set(spec) != {"producer", "source_type", "unit", "value_kind", "labels", "measurement"}
            or spec["source_type"] not in ("prometheus", "debug_api", "client_journal")
            or type(spec["producer"]) is not str or not _NAME.fullmatch(spec["producer"])):
        raise ScenarioError(f"{path}: invalid produced metric {metric}")
    _metadata(spec, path)
    from monitoring.producers import PRODUCERS
    if spec["producer"] not in PRODUCERS:
        raise ScenarioError(f"{path}: unknown metric producer {spec['producer']}")


def _exclude(sources, produced, value, path):
    excluded = _unique_strings(value, path, "invalid excluded metric ids")
    for identity in excluded:
        kind, _, metric = identity.partition("/")
        if kind in sources and metric in sources[kind]:
            del sources[kind][metric]
        elif identity in produced:
            del produced[identity]
        else:
            raise ScenarioError(f"{path}: excluded metric does not exist: {identity}")


def _metadata(spec, path):
    if (type(spec.get("unit")) is not str or not spec["unit"]
            or spec.get("value_kind") not in ("gauge", "counter", "scalar")
            or not isinstance(spec.get("labels"), list)
            or any(type(x) is not str or not _NAME.fullmatch(x) for x in spec["labels"])
            or len(set(spec["labels"])) != len(spec["labels"])):
        raise ScenarioError(f"{path}: metric requires unit, value_kind and unique labels")
    if "measurement" in spec:
        validate_measurement(spec["measurement"], path)


def validate_measurement(value, path):
    """Describe evidence semantics; this is metadata, never an expression language."""
    if (not isinstance(value, dict)
            or set(value) != {"method", "population", "accuracy", "requires_request_identity"}
            or any(type(value[key]) is not str or not value[key].strip()
                   for key in ("method", "population"))
            or value["accuracy"] not in ("request_ledger", "sampled", "histogram_estimate", "counter_delta")
            or type(value["requires_request_identity"]) is not bool):
        raise ScenarioError(f"{path}: invalid measurement semantics")


def definitions(plan):
    result = {kind + "/" + metric: dict(spec, source_type="prometheus", source_kind=kind)
              for kind, queries in plan["sources"].items() for metric, spec in queries.items()}
    for kind in SOURCE_KINDS:
        result[kind + "/up"] = dict(source_type="prometheus", source_kind=kind,
            promql="up${selector}", required=True, unit="boolean", value_kind="gauge", labels=[],
            measurement=dict(method="promql_evaluation", population=kind + "_selector",
                             accuracy="sampled", requires_request_identity=False))
    if set(result) & set(plan["produced"]):
        raise ScenarioError("produced metric conflicts with a Prometheus metric")
    return {**result, **plan["produced"]}


def plan_hash(plan):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_export_filter(plan, environment):
    """A declared Master export filter cannot exclude a selected query's inputs.

    Physical dependencies are declared data; this does not parse/rewrite PromQL.
    An absent override keeps the Java export policy, with runtime required-query
    validation still responsible for proving actual metric availability.
    """
    filters = [environment.get("metric_whitelist")]
    filters.extend(patch.get("metric_whitelist") for patch in
                   environment.get("profile_overrides", {}).values() if "metric_whitelist" in patch)
    for value in filters:
        if value is None:
            continue
        prefixes = value.split(",")
        for metric, spec in plan["sources"]["master"].items():
            for name in spec.get("exported_metrics", []):
                if name.startswith("flexlb_") and not any(name.startswith(prefix) for prefix in prefixes):
                    raise ScenarioError(f"master/{metric}: exported metric {name} excluded by metric_whitelist; exclude the query or allow its inputs")


def queries_for_targets(plan, targets, selector, interval, target_kinds=None):
    """Expand one validated plan; target names only identify source instances."""
    window_ms = round(max(4 * interval, 10) * 1000)
    queries, required = {}, set()
    if target_kinds is None:
        target_kinds = {
            source: "mock" if source == "mock" else "client" if source.startswith("client-") else "master"
            for source in targets
        }
    if set(target_kinds) != set(targets) or set(target_kinds.values()) - set(SOURCE_KINDS):
        raise ValueError("target kinds must identify every Prometheus target")
    for source, kind in target_kinds.items():
        queries[source + "/up"] = "up" + selector(source)
        required.add(source + "/up")
        for metric, definition in plan["sources"][kind].items():
            key = source + "/" + metric
            queries[key] = (definition["promql"]
                            .replace("${selector}", selector(source))
                            .replace("${window_ms}", str(window_ms)))
            if definition.get("required", False):
                required.add(key)
    return queries, required
