"""Metric identity, frozen provenance and invalid-data contracts."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from monitoring.metric_store import (
    MetricStore, MetricContractError, MetricUnavailable, export_metrics, publish,
)
from monitoring.query_plan import load_plan, plan_hash, definitions
from reporting.metric_binding import bindings
from scenario.loader import ScenarioError


def definition(**kwargs):
    return dict(promql="temperature${selector}", unit="K", value_kind="gauge", labels=[], **kwargs)


def archive(root, queries, **kwargs):
    path = Path(root) / "telemetry/1/queries.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(start=1, end=3, step=1, targets={}, queries=queries,
                target_kinds={"worker": "mock"}, query_plan="temperature.yaml", **kwargs)
    path.write_text(json.dumps(data))
    return path


class MetricPlanTest(unittest.TestCase):
    def test_compilation_checks_declared_gate_units_modes_and_labels(self):
        from cases.config import configure_program
        from scenario.loader import load_document
        root = Path(__file__).resolve().parents[1]
        config = load_document(root / "config/scenarios/cache_scale_in.yaml")
        for change in [dict(unit="tokens"), dict(mode="evaluated"), dict(labels=[])]:
            plan = load_plan("cache_scale_in.yaml")
            plan["sources"]["mock"]["running"].update(change)
            with patch("monitoring.query_plan.load_plan", return_value=plan):
                with self.assertRaisesRegex(ScenarioError, "dependency.*mismatch"):
                    configure_program(config, "test")

    def test_ha_state_rejects_missing_fields_without_zero_fallback(self):
        from cases.master_ha_failover.runtime import master_state_fields
        good = dict(scheduler_inflight=0, prefill_endpoints=[], decode_endpoints=[])
        self.assertEqual(master_state_fields(good)["prefill_inflight_requests"], 0)
        for broken in [dict(scheduler_inflight=0), dict(good, prefill_endpoints=[{}]),
                       dict(good, scheduler_inflight=float("nan"))]:
            with self.assertRaises(ValueError):
                master_state_fields(broken)

    def test_sets_compose_exclude_and_pin_expanded_content(self):
        with tempfile.TemporaryDirectory() as d, patch("monitoring.query_plan.CATALOG", Path(d)):
            root = Path(d)
            base = dict(metric_plan_schema_version=2, sources=dict(mock=dict(temperature=definition())))
            (root / "base.yaml").write_text(json.dumps(base))
            (root / "case.yaml").write_text(json.dumps(dict(metric_plan_schema_version=2, include=["base.yaml"],
                sources=dict(mock=dict(second=definition())))))
            original = load_plan("case.yaml")
            self.assertEqual(set(original["sources"]["mock"]), {"temperature", "second"})
            pinned = plan_hash(original)
            base["sources"]["mock"]["temperature"]["unit"] = "C"
            (root / "base.yaml").write_text(json.dumps(base))
            self.assertNotEqual(plan_hash(load_plan("case.yaml")), pinned)
            self.assertEqual(original["sources"]["mock"]["temperature"]["unit"], "K")
            (root / "case.yaml").write_text(json.dumps(dict(metric_plan_schema_version=2, include=["base.yaml"],
                exclude=["mock/temperature"], sources=dict(mock=dict(second=definition())))))
            self.assertEqual(set(load_plan("case.yaml")["sources"]["mock"]), {"second"})

    def test_duplicate_cycle_unknown_exclusion_and_producer_rejected(self):
        for changes, pattern in [
            (dict(include=["base.yaml"], sources=dict(mock=dict(temperature=definition()))), "duplicate"),
            (dict(include=["case.yaml"]), "cycle"),
            (dict(include=["base.yaml"], exclude=["mock/missing"]), "does not exist"),
            (dict(produced={"derived/x": dict(producer="unknown", source_type="derived", unit="K",
                                             value_kind="scalar", labels=[])}), "unknown metric producer"),
        ]:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as d, patch("monitoring.query_plan.CATALOG", Path(d)):
                (Path(d)/"base.yaml").write_text(json.dumps(dict(metric_plan_schema_version=2,
                    sources=dict(mock=dict(temperature=definition())))))
                (Path(d)/"case.yaml").write_text(json.dumps(dict(metric_plan_schema_version=2, **changes)))
                with self.assertRaisesRegex(ScenarioError, pattern):
                    load_plan("case.yaml")

    def test_live_raw_reader_preserves_scrape_clock_and_checks_labels(self):
        from monitoring.session import PrometheusSession
        session = PrometheusSession("unused", {"mock":"http://unused/metrics"},
                                    query_plan="master_performance.yaml")
        row = dict(metric=dict(__name__="rtp_llm_context_tps", role="prefill",
                              engine_name="P0", engine_incarnation="one"), values=[[1.7, "9"]])
        with patch.object(session, "query", return_value=[row]) as query:
            result = session.metric_rows("mock/rtp_llm_context_tps", source="mock", start=1, end=2)
            self.assertEqual(result[0]["values"], [[1.7,"9"]])
            self.assertEqual(result[0]["metric_id"], "mock/rtp_llm_context_tps")
            self.assertEqual(query.call_args.args, ('rtp_llm_context_tps{job="mock"}[1000ms]', 2))
        with self.assertRaisesRegex(ValueError, "undeclared"):
            session.metric_rows("mock/missing", source="mock", start=1, end=2)
        del row["metric"]["engine_incarnation"]
        with patch.object(session,"query",return_value=[row]), self.assertRaisesRegex(ValueError,"labels"):
            session.metric_rows("mock/rtp_llm_context_tps", source="mock", start=1, end=2)


class MetricArtifactTest(unittest.TestCase):
    def test_ha_windows_keep_their_own_source_provenance(self):
        from types import SimpleNamespace
        from cases.master_ha_failover.metrics import publish_gate
        with tempfile.TemporaryDirectory() as d:
            ctx = SimpleNamespace(artifact_dir=Path(d), env_epoch=1,
                                  monitor=SimpleNamespace(query_plan=load_plan("master_ha_failover.yaml")))
            for window, stamp in [("before", 1000), ("after", 2000)]:
                publish_gate(ctx, dict(metric="ha_gate/success_rate", rows=window), 1,
                             [dict(send_start_epoch_ms=stamp)])
            rows = MetricStore.read(d).document["metrics"]["ha_gate/success_rate"]
            self.assertEqual([row["provenance"]["evidence"]["observed_request_bounds"] for row in rows],
                             [[1, 1], [2, 2]])
            self.assertTrue(all(len(row["provenance"]["producer_sha256"]) == 64 for row in rows))

    def plan(self):
        return dict(metric_plan_schema_version=2, sources=dict(mock=dict(temperature=definition()),client={},master={}), produced={})

    def test_query_id_survives_exported_name_and_chunk_boundaries(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [dict(metric={"__name__":"physical_temperature","engine":"a"},values=[[1,"1"],[2,"2"]]),
                    dict(metric={"__name__":"physical_temperature","engine":"a"},values=[[2,"2"],[3,"3"]])]
            archive(d, {"worker/temperature":dict(promql="temperature",result=rows)},metric_plan=self.plan())
            store = export_metrics(d)
            self.assertEqual(len(store.select("mock/temperature")),1)
            self.assertEqual(store.reduce("mock/temperature",op="last",start=1,end=3,min_samples=3,max_gap_s=1),3)
            self.assertEqual(store.document["metrics"]["mock/temperature"][0]["metric_id"], "mock/temperature")
            self.assertIn("1/worker/temperature/", next(iter(store.series(0)[0])))
            for path in Path(d).glob("telemetry/*/queries.json"): path.unlink()
            self.assertEqual(MetricStore.read(d).reduce("mock/temperature",op="max"),3)

    def test_missing_and_invalid_data_are_not_threshold_failures(self):
        with tempfile.TemporaryDirectory() as d:
            path=archive(d, {"worker/temperature":dict(promql="temperature",result=[
                dict(metric={}, values=[[1,"0"],[2,"NaN"],[3,"3"]])])}, metric_plan=self.plan())
            store=export_metrics(d)
            self.assertEqual(store.document["metrics"]["mock/temperature"][0]["points"],[[1,0],[2,None],[3,3]])
            with self.assertRaisesRegex(MetricUnavailable,"invalid samples"):
                store.reduce("mock/temperature",op="max",start=1,end=3)
            self.assertEqual(store.reduce("mock/temperature",op="last",start=2.5,end=3.5),3)
            with self.assertRaises(MetricUnavailable): store.select("mock/temperature",labels={"missing":"x"})
            with self.assertRaises(MetricContractError): store.select("mock/unknown")
            data=json.loads(path.read_text()); data["queries"]["worker/temperature"]["result"]=[]
            data["errors"]=[dict(query="worker/temperature",error="query failed")];path.write_text(json.dumps(data))
            with self.assertRaisesRegex(MetricUnavailable,"query failed"):
                export_metrics(d).select("mock/temperature")

    def test_conflicting_same_timestamp_fails_loud(self):
        with tempfile.TemporaryDirectory() as d:
            archive(d,{"worker/temperature":dict(promql="temperature",result=[
                dict(metric={},values=[[1,"1"],[1,"2"]])])},metric_plan=self.plan())
            with self.assertRaisesRegex(ValueError,"conflicting monitor samples"):
                export_metrics(d)

    def test_windows_and_fleet_aggregation_are_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            archive(d,{"worker/temperature":dict(promql="temperature",result=[
                dict(metric={"engine":name},values=[[1,"1"],[3,"3"]]) for name in ["a","b"]])},metric_plan=self.plan())
            store=export_metrics(d)
            with self.assertRaisesRegex(MetricContractError,"one series"):
                store.reduce("mock/temperature",op="max")
            with self.assertRaisesRegex(MetricUnavailable,"coverage"):
                store.select("mock/temperature",labels={"engine":"a"},start=1,end=3,max_gap_s=1)
            with self.assertRaisesRegex(MetricUnavailable,"samples"):
                store.select("mock/temperature",min_samples=3)

    def test_producer_ownership_labels_and_atomic_publication(self):
        with tempfile.TemporaryDirectory() as d:
            plan=self.plan(); spec=dict(producer="ha_evidence",source_type="debug_api",unit="requests",value_kind="gauge",labels=["master"])
            plan["produced"]["ha/custom"]=spec
            store=export_metrics(d,plan)
            original=(Path(d)/"metrics.json").read_bytes()
            rows=[dict(epoch="1",source="ha",labels={"master":"A"},points=[[1,2],[2,None]])]
            with self.assertRaises(MetricContractError): publish(store,"ha/custom",spec,rows,producer="wrong",evidence={})
            with self.assertRaisesRegex(MetricContractError,"duplicate"):
                publish(store,"ha/custom",spec,rows*2,producer="ha_evidence",evidence={})
            self.assertEqual((Path(d)/"metrics.json").read_bytes(),original)
            publish(store,"ha/custom",spec,rows,producer="ha_evidence",evidence={"endpoint":"/debug"})
            store.save(d)
            frozen=MetricStore.read(d)
            self.assertEqual(frozen.document["metrics"]["ha/custom"][0]["provenance"]["source_type"],"debug_api")
            with self.assertRaises(MetricUnavailable): frozen.select("ha/custom")

    def test_view_has_local_curve_ids_and_independent_label_filters(self):
        presentation=dict(charts=dict(curves={"p_curve":dict(metric_id="mock/temperature",labels={"role":"prefill"},color="red"),
                                  "d_curve":dict(metric_id="mock/temperature",labels={"role":"decode"},color="blue")}))
        self.assertEqual([key for key,_ in bindings(presentation,"mock/temperature",{"role":"prefill"})],["p_curve"])
        self.assertEqual(bindings(presentation,"mock/temperature",{"role":"unknown"}),[])

    def test_reinterpretation_only_writes_new_destination(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/"source"; destination=Path(d)/"new"
            archive(source,{"worker/temperature":dict(promql="temperature",result=[dict(metric={},values=[[1,"1"]])])},metric_plan=self.plan())
            before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
            export_metrics(destination,self.plan(),archive_directory=source)
            self.assertEqual(before,{str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()})
            self.assertEqual(MetricStore.read(destination).reduce("mock/temperature",op="last"),1)


if __name__ == "__main__": unittest.main()
