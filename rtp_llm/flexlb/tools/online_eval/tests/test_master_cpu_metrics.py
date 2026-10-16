"""CPU diagnostics reuse actual JVM meters and remain outside case gates."""
from monitoring.collection_plan import physical_metrics
from monitoring.query_plan import load_plan, queries_for_targets
from reporting.view_config import view
from reporting.curves import project_panels


def test_cpu_export_inputs_and_per_master_queries():
    plan = load_plan("cache_scale_in.yaml")
    assert {"process_cpu_usage", "system_cpu_usage", "system_cpu_count"} <= set(
        physical_metrics(plan, "master"))
    queries, required = queries_for_targets(
        plan, ["master-A", "master-B"], lambda name: '{job="' + name + '"}',
        1, {"master-A": "master", "master-B": "master"})
    for name in ("master-A", "master-B"):
        key = name + "/cpu_usage_percent"
        assert queries[key] == '100 * (process_cpu_usage{job="' + name + '"} >= 0)'
        assert key in required
    assert not any("cpu" in key for key in plan["produced"])


def test_cpu_panel_preserves_percent_capacity_and_missing_samples():
    presentation = view("cache_scale_in.yaml")
    curves = [
        dict(curve_id=identity, metric_id=identity, name=style.name, axis=style.axis,
             points=[dict(x=0, y=None), dict(x=1, y=value)])
        for identity, value in [
            ("master/cpu_usage_percent", 12.5),
            ("master/system_cpu_usage_percent", 40),
            ("master/cpu_available_processors", 256)]
        for style in [presentation.charts.curves[identity]]
    ]
    panel = next(p for p in project_panels(curves, presentation) if p["id"] == "master-cpu")
    assert [s["points"][1]["y"] for s in panel["series"]] == [12.5, 40, 256]
    assert all(s["points"][0]["y"] is None for s in panel["series"])
