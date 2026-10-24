"""Resolved case capabilities cross the boundary; generic readers never discover cases."""

from functools import partial

import pytest

from monitoring.query_plan import load_plan
from monitoring.sources import selected_sources
from reporting.metric_binding import metric_classification
from reporting.view_config import view
from scenario import registry


def test_bound_monitoring_and_reports_do_not_reenter_registry(monkeypatch):
    current = registry.registry()
    loader = partial(load_plan, contracts=current.metric_contracts, sources=current.sources)

    def forbidden():
        pytest.fail("generic readers must consume supplied capabilities")

    monkeypatch.setattr(registry, "registry", forbidden)
    plan = loader("master_ha_failover.yaml")
    assert selected_sources(plan, current.sources)["master_inflight"]["fields"]
    presentation = view(
        "master_ha_failover.yaml", capability=current.views["master_ha_failover.yaml"], plan_loader=loader
    )
    plotted = {style.metric_id for style in presentation.charts.curves.values()}
    identity, definition = next(
        (identity, spec)
        for identity, spec in plan["produced"].items()
        if spec["value_kind"] == "scalar" and identity not in plotted
    )
    assert metric_classification(identity, definition, presentation) == "GATE_EVIDENCE"


def test_query_plan_requires_explicit_producer_bindings():
    with pytest.raises(ValueError, match="unknown metric producer"):
        load_plan("master_performance.yaml")


@pytest.mark.parametrize("kind", ["functional", "workload"])
def test_explicit_null_monitoring_plan_remains_invalid(kind):
    from scenario.execution_policy import normalize_execution

    with pytest.raises(ValueError, match="query_plan"):
        normalize_execution({"collection": "request", "monitoring": {"query_plan": None}}, kind=kind)
