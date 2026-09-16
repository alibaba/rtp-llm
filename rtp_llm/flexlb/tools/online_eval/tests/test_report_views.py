import copy
import unittest

from flexlb_test_framework.workload.view_render import render
from flexlb_test_framework.workload.views import build, digest, metric


class ReportViewsTest(unittest.TestCase):
    def report(self, capacity=10, budget=20):
        config = dict(capacity=capacity, budget=budget, fixed=1)
        return dict(
            id="same",
            configuration=config,
            configuration_sha256=digest(config),
            workload=dict(runtime_validity="VALID"),
            clock_anchor=dict(monotonic_s=100),
            stages=[dict(id="load", status="PASS", started_s=102, finished_s=112)],
            series=dict(
                hit=[[2, 0.8], [3, 0.9]],
                latency=[[2, 10], [3, 20]],
                qps=[[2, 10], [3, 12]],
            ),
        )

    def spec(self):
        return dict(
            kind="sweep",
            title="test",
            vary=["/capacity", "/budget"],
            window=dict(
                start=dict(stage="load", edge="start"),
                end=dict(stage="load", edge="end"),
            ),
            metrics={
                k: dict(series=k, reducer="mean") for k in ["hit", "latency", "qps"]
            },
            x="hit",
            y="latency",
            size="qps",
            color="/budget",
            facet="/capacity",
            constraints=[
                dict(metric="latency", op="le", value=30),
                dict(metric="qps", op="ge", value=5),
            ],
        )

    def test_sweep_masks_only_declared_axes_and_keeps_repeats(self):
        a, b = self.report(), self.report(20, 30)
        result = build([a, b, copy.deepcopy(b)], self.spec())
        self.assertEqual([p["status"] for p in result["points"]], ["FEASIBLE"] * 3)
        self.assertEqual(len(result["points"]), 3)
        self.assertIn("Constraint grid", render(result))
        a["implementation"] = {"sha256": "a"}
        b["implementation"] = {"sha256": "b"}
        with self.assertRaises(ValueError):
            build([a, b], self.spec())
        b["implementation"] = a["implementation"]
        b["configuration"]["fixed"] = 2
        b["configuration_sha256"] = digest(b["configuration"])
        with self.assertRaises(ValueError):
            build([a, b], self.spec())

    def test_missing_invalid_and_counter_reset_never_become_candidates(self):
        a = self.report()
        a["series"]["latency"] = [[2, None]]
        self.assertEqual(
            build([a], self.spec())["points"][0]["status"], "MISSING_OR_INVALID"
        )
        a["stages"][0]["status"] = "BLOCKED"
        self.assertEqual(
            build([a], self.spec())["points"][0]["metrics"]["hit"]["status"],
            "MISSING_WINDOW",
        )
        a = self.report()
        a["series"]["counter"] = [[2, 10], [3, 1]]
        self.assertEqual(
            metric(a, dict(series="counter", reducer="delta"), 2, 5)["status"],
            "COUNTER_RESET_OR_INSUFFICIENT_DATA",
        )
        a["configuration"]["capacity"] = 99
        with self.assertRaises(ValueError):
            build([a], self.spec())

    def test_timeline_translates_without_stretching_and_escapes_labels(self):
        a = self.report()
        b = copy.deepcopy(a)
        b["clock_anchor"]["monotonic_s"] = 200
        b["stages"][0].update(started_s=202, finished_s=222)
        b["series"]["qps"] = [[2, 1], [22, 2]]
        spec = dict(
            kind="timeline",
            title="<script>x</script>",
            align=dict(stage="load", edge="start"),
            events=[dict(stage="load", edge="end", label="done")],
            panels=[dict(title="QPS", metrics=["qps"])],
        )
        result = build([a, b], spec)
        self.assertEqual([r["events"][0]["t"] for r in result["runs"]], [10, 20])
        self.assertEqual(
            result["runs"][1]["panels"][0]["series"]["qps"], [[0, 1], [20, 2]]
        )
        self.assertNotIn("<script>", render(result))
        b["stages"][0]["status"] = "BLOCKED"
        self.assertEqual(
            build([a, b], spec)["runs"][1]["validity"], "MISSING_ALIGNMENT"
        )

    def test_pooled_percentile_and_completion_rate_use_distinct_cohorts(self):
        report = self.report()
        report["clock_anchor"]["epoch_s"] = 1000
        report["request_source_state"] = "VERIFIED"
        report["_request_rows"] = [
            dict(
                send_start_epoch_ms=1002000, wall_clock_ts=1009, status="ok", ttft_ms=3
            ),
            dict(
                send_start_epoch_ms=1003000, wall_clock_ts=1015, status="ok", ttft_ms=99
            ),
        ]
        value = metric(
            report,
            dict(requests="ttft_ms", reducer="p95", cohort="sent", outcomes="success"),
            2,
            12,
        )
        self.assertEqual(value["value"], 99)
        rate = metric(
            report,
            dict(
                requests="count", reducer="rate", cohort="completed", outcomes="success"
            ),
            2,
            12,
        )
        self.assertEqual(rate["value"], 0.1)
        report["request_source_state"] = "INVALID"
        self.assertEqual(
            metric(
                report,
                dict(
                    requests="count",
                    reducer="rate",
                    cohort="completed",
                    outcomes="success",
                ),
                2,
                12,
            )["status"],
            "MISSING_DATA",
        )
