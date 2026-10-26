import unittest
from types import SimpleNamespace

from cases.master_ha_failover.actions import _client_check
from cases.master_performance.inputs import for_profile
from context_fixtures import plan
from metric_fixtures import performance_evidence as evidence
from monitoring.quality import bounded_collection_gaps


def test_collection_quality_projects_mixed_gap_classes_without_losing_order():
    from monitoring.quality import assess_collection

    source = "1/master-A/collection"
    series = {"1/master-A/up/{}": [[0, 1], [2, 1], [10, 1], [12, 1]], "1/master-A/optional/{}": [[1, None]]}
    quality = assess_collection(
        series,
        {source: [1, 4, 7, 11, 30]},
        [],
        dict(
            clock_anchor={"epoch_s": 100},
            expected_telemetry=["1/master-A"],
            expected_outages=[dict(source="1/master-A", started_epoch_s=130, ended_epoch_s=140)],
        ),
        capture_metrics=True,
        max_gap_s=3,
    )
    projected = quality.to_dict()
    assert not quality.valid and quality.completeness == "PARTIAL"
    assert projected["unexpected_telemetry_gaps"] == {source: [1, 4, 7, 11]}
    assert projected["expected_telemetry_gaps"] == {source: [30]}
    assert projected["tolerated_telemetry_gaps"] == {source: [1, 11]}
    assert projected["fatal_telemetry_gaps"] == {source: [4, 7]}
    assert projected["telemetry_gaps"] == {"1/master-A/optional/{}": [1]}


def test_collection_quality_keeps_advisories_separate_from_missing_sources():
    from monitoring.quality import assess_collection

    evidence = dict(clock_anchor={"epoch_s": 0}, expected_telemetry=["1/mock"])
    advisory = dict(severity="diagnostic", error="optional curve absent")
    healthy = {"1/mock/up/{}": [[0, 1]]}
    quality = assess_collection(healthy, {}, [advisory], evidence, capture_metrics=True, max_gap_s=5)
    assert quality.valid and quality.completeness == "PARTIAL"
    assert quality.to_dict()["telemetry_diagnostics"] == [advisory]
    missing = assess_collection({}, {}, [advisory], evidence, capture_metrics=True, max_gap_s=5)
    assert not missing.valid and missing.to_dict()["missing_telemetry"] == ["1/mock"]


def test_workload_projection_cannot_clear_quality_failure_or_change_owned_details():
    from monitoring.quality import CollectionIssue, CollectionQuality
    from scenario.run_result import WorkloadResult

    quality = CollectionQuality(issues=(CollectionIssue("missing", None, "1/mock"),))
    result = WorkloadResult(details={"request": {"count": 1}}, quality=quality)
    projection = result.to_dict()
    projection["runtime_validity"] = "VALID"
    projection["request"]["count"] = 99
    projection["missing_telemetry"].clear()
    assert result.validity == "INVALID"
    assert result.to_dict()["request"] == {"count": 1}
    assert result.to_dict()["missing_telemetry"] == ["1/mock"]


class GatePolicyTest(unittest.TestCase):
    def test_short_gap_requires_success_on_both_sides(self):
        series = {"1/master-B/up/{}": [[0, 1], [1, None], [2, 1]]}
        gap = {"1/master-B/collection": [1]}
        self.assertEqual(bounded_collection_gaps(series, gap, 5), (gap, {}))
        for points in [[[1, None], [2, 1]], [[0, 1], [1, None]], [[0, 1], [1, None], [6, 1]]]:
            self.assertEqual(bounded_collection_gaps({"1/master-B/up/{}": points}, gap, 5), ({}, gap))

    def test_consecutive_failures_use_entire_gap(self):
        gap = {"1/master-B/collection": [1, 2, 3, 4, 5]}
        series = {"1/master-B/up/{}": [[0, 1], [6, 1]]}
        self.assertEqual(bounded_collection_gaps(series, gap, 5), ({}, gap))

    def test_advisory_errors_do_not_waive_sample_coverage_or_single(self):
        rows = [{"status": "ok"}, {"status": "error"}]
        ctx = SimpleNamespace(env_epoch=1, plan=plan({"profile": "batch-window"}), resource=lambda *args: rows)
        deadline = SimpleNamespace(check=lambda: None)
        params = dict(rows="rows", metric="ha_gate/non_ok_count", op="eq", expected=0, min_samples=2, warning_profiles=["batch-window"])
        check = _client_check(ctx, params, deadline).checks[0]
        self.assertEqual((check.status, check.actual, check.expected), ("WARNING", 1, 0))
        self.assertEqual(_client_check(ctx, {**params, "min_samples": 3}, deadline).checks[0].status, "ERROR")
        ctx.plan.profile = "single-nonbatch"
        self.assertEqual(_client_check(ctx, params, deadline).checks[0].status, "FAIL")

    def test_profile_floors_preserve_batch_and_decode(self):
        criteria = evidence()["criteria"]
        criteria["engine_tps"] = {
            "mock/rtp_llm_context_tps": 50000,
            "mock/rtp_llm_context_tps_with_cache": 100000,
            "mock/mock_decode_wall_tps": 2230,
        }
        criteria["engine_tps_by_profile"] = {
            "single-nonbatch": {"mock/rtp_llm_context_tps": 45000, "mock/rtp_llm_context_tps_with_cache": 90000}
        }
        gate_input = dict(
            fields={
                name.split("/")[1]: dict(metric=name, labels=dict(role=role))
                for name, role in {
                    "mock/rtp_llm_context_tps": "prefill",
                    "mock/rtp_llm_context_tps_with_cache": "prefill",
                    "mock/mock_decode_wall_tps": "decode",
                }.items()
            }
        )
        single = for_profile(criteria, "single-nonbatch", gate_input)
        batch = for_profile(criteria, "batch-window", gate_input)
        self.assertEqual(single["engine_tps"]["mock/rtp_llm_context_tps"], 45000)
        self.assertEqual(single["engine_tps"]["mock/mock_decode_wall_tps"], 2230)
        self.assertEqual(batch["engine_tps"]["mock/rtp_llm_context_tps"], 50000)
        self.assertEqual(criteria["engine_tps"]["mock/rtp_llm_context_tps"], 50000)
