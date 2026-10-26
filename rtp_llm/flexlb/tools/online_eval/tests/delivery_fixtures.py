"""Explicit case delivery helpers for frozen-evidence fixtures."""

import json
import time
from pathlib import Path

from artifacts.gate import publish_gate
from metric_fixtures import freeze_metrics
from monitoring.metric_store import MetricStore
from reporting import run_meta, write_bundle
from reporting.curves import build_panels, materialize
from reporting.gate_lines import Threshold, apply_gate_lines
from runtime.deadline import Deadline
from scenario.registry import view as view


def publish_cache(directory, evidence, result, **kwargs):
    from cases.cache_scale_in import metrics, report

    return publish_gate(directory, "cache", evidence, result, metrics.produce, report.write_report, **kwargs)


def publish_performance(directory, evidence, result, **kwargs):
    from cases.master_performance import metrics, report

    return publish_gate(directory, "performance", evidence, result, metrics.produce, report.write_report, **kwargs)


def finalize_metrics(directory, analysis):
    from scenario.registry import registry

    selected = {spec["producer"] for spec in MetricStore.read(directory).document["definitions"].values() if "producer" in spec}
    analysis["metric_directory"] = str(directory)
    deadline = Deadline(time.monotonic() + 30)
    for case in registry().cases.values():
        metrics = case.definition.metrics
        if metrics is not None and selected & metrics.CONTRACTS.keys() and hasattr(metrics, "finalize"):
            metadata = metrics.finalize(directory, analysis, selected, deadline)
            analysis.setdefault("producer_results", {}).update(metadata)


def performance_report(directory, evidence, result):
    freeze_metrics(directory, "master_performance")
    return publish_performance(directory, evidence, result)


def cache_report(directory, evidence, result, prepared=None):
    freeze_metrics(directory, "cache_scale_in")
    bundle = publish_cache(directory, evidence, result, report_options={"prepared": prepared})
    return json.loads((bundle / "report-spec.json").read_text())


def experiment_definition():
    return dict(
        experiment_definition_schema_version=1, plan={"id": "case::default::profile"}, dependencies={"renderer.py": "a" * 64}, views={}
    )


def experiment_execution():
    return dict(
        id="case::default::profile",
        test_kind="functional",
        status="PASS",
        outcome=dict(gate="PASS", validity="VALID", delivery="PASS"),
        stages=[],
        cleanup=[],
    )


CURVE = "mock/rtp_llm_context_tps_engine_mean/P"


REFERENCE = "mock/rtp_llm_context_tps"


def demo_runs(root):
    """Three explicit fixture runs: different floors/windows, one optional floor off."""
    presentation = view("master_performance.yaml")
    paths = []
    for label, floor, start, event, actual, verdict in (
        ("A", 50, 2, 4, 60, "PASS"),
        ("B", 80, 5, 8, 75, "FAIL"),
        ("C", None, 1, 6, 90, "PASS"),
    ):
        curve = materialize(
            CURVE,
            presentation.charts.curves[CURVE],
            [[t, value] for t, value in ((0, actual), (5, None), (10, actual + 5), (20, actual))],
            unit="tokens/s",
        )
        thresholds = {
            REFERENCE: Threshold(floor, start, start + 10) if floor is not None else None,
            "mock/rtp_llm_context_tps_with_cache": None,
            "mock/mock_decode_wall_tps": None,
        }
        panels = apply_gate_lines(build_panels([curve], presentation), presentation, thresholds)
        for panel in panels:
            panel["caption"] = "兼容性演示数据，不是实际压测。" + panel["caption"]
            panel["events"] = [dict(id="checkpoint", name="演示对齐事件", t=event)]
        spec = dict(
            title="门禁线 compare 演示 " + label,
            subtitle="演示数据，不是实际压测",
            timeOriginLabel="演示 run 开始",
            timeAxis=dict(min=0, max=25),
            events=[dict(id="checkpoint", name="演示对齐事件", t=event)],
            panels=panels,
        )
        paths.append(
            write_bundle(
                Path(root) / label,
                "run",
                "fixture",
                dict(verdict=verdict, criteria=dict(floor=floor), purpose="COMPATIBILITY_DEMO"),
                spec,
                meta=run_meta(dict(id="demo-" + label), clock=dict(origin_kind="demo_start")),
                producer="comparison-fixture",
            )
        )
    return paths
