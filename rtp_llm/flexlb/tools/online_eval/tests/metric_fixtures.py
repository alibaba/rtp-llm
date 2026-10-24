"""Synthetic archives explicitly freeze the metric definitions they exercise."""

import json
from pathlib import Path

from monitoring.archive import export_metrics
from scenario.registry import load_metric_plan as load_plan


def freeze_metrics(directory, case):
    plan = load_plan(case + ".yaml")
    if case == "master_ha_failover":
        from monitoring.collection_plan import select_plan
        from scenario.registry import view as view

        gates = [identity for identity, spec in plan["produced"].items() if spec["value_kind"] == "scalar"]
        plan = select_plan(plan, gates, [view(case + ".yaml")])
    for path in Path(directory).glob("telemetry/*/queries.json"):
        data = json.loads(path.read_text())
        if "metric_plan" not in data:
            data.update(metric_plan=plan, query_plan=case + ".yaml")
            path.write_text(json.dumps(data))
    return export_metrics(directory, plan)


def selected_sources(plan):
    from monitoring.sources import selected_sources as select
    from scenario.registry import registry

    return select(plan, registry().sources)


def collection_plan(plan):
    from monitoring.collection_plan import collection_plan as collect
    from scenario.registry import registry

    return collect(plan, registry().sources)


def instance_plan(instance):
    from monitoring.collection_plan import instance_plan as read
    from scenario.registry import registry

    return read(instance, registry().sources)


def frozen_plan(name, plan):
    from monitoring.collection_plan import frozen_plan as freeze
    from scenario.registry import registry

    return freeze(name, plan, registry().sources)


def publish(*args, producer, **kwargs):
    from monitoring.publication import publish as write
    from scenario.registry import registry

    kwargs.setdefault("describe", registry().metric_contracts.get(producer))
    return write(*args, producer=producer, **kwargs)


def PrometheusSession(*args, **kwargs):
    from monitoring.session import PrometheusSession as Session
    from scenario.registry import load_metric_plan, registry

    kwargs.setdefault("sources", registry().sources)
    if kwargs.get("query_plan") and "metric_plan" not in kwargs:
        kwargs["metric_plan"] = load_metric_plan(kwargs["query_plan"])
    return Session(*args, **kwargs)
