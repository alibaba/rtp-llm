"""Regression coverage for confirmed PR review findings."""
from pathlib import Path

import pytest

from analysis.statistics import percentile_nr as report_percentile
from flexlb_cfg import ConfigOverride, _retype_ordering, render_env

ROOT = Path(__file__).resolve().parents[1]


def test_redundant_priority_override_preserves_existing_policy():
    ordering = {"type": "PRIORITY", "defaultPriority": 75,
                "preemption": {"allowedVictimStages": ["DECODE_ENGINE_OWNED"],
                               "timeoutMs": 900}}
    doc = {"scheduler": {"ordering": ordering.copy()}}
    _retype_ordering(doc, ConfigOverride(ordering="priority"))
    assert doc["scheduler"]["ordering"] == ordering
    _retype_ordering(doc, ConfigOverride(ordering="priority", default_priority=20,
                                       strip_preemption=True))
    assert doc["scheduler"]["ordering"] == {"type": "PRIORITY", "defaultPriority": 20}


@pytest.mark.parametrize("value", [-1, 0.5, float("nan"), float("inf"), True, "2"])
def test_invalid_decision_lifetime_rejected_before_rendering(value):
    with pytest.raises(ValueError, match="decision_lifetime"):
        render_env("stress-na130", ConfigOverride(decision_lifetime=value))


@pytest.mark.parametrize("values,p,expected", [([10, 20], .5, 10),
    ([10, 20, 30, 40], .2, 10),
    (list(range(1, 101)), .95, 95), ([], .99, None)])
def test_percentile_uses_nearest_rank(values, p, expected):
    assert report_percentile(values, p) == expected


@pytest.mark.parametrize("timestamp", ["bad", None, True, float("nan"),
                                      float("inf"), -float("inf")])
def test_metric_timestamp_is_validated_before_window_filter(timestamp):
    from monitoring.metric_store import MetricStore, MetricUnavailable
    row = dict(metric_id="x", source="P", epoch="1", status="PASS",
               labels={}, provenance={}, points=[[10, 2], [timestamp, 3]])
    store = MetricStore(dict(metrics_schema_version=1,
                            definitions={"x": dict(unit="count")}, metrics={"x": [row]}))
    with pytest.raises(MetricUnavailable, match="timestamp"):
        store.select("x", start=9, end=11)


@pytest.mark.parametrize("field", ["labels", "provenance", "evidence",
                                  "collection_gaps", "errors"])
def test_metric_artifact_rejects_null_metadata(field):
    from monitoring.metric_store import MetricStore, MetricContractError
    row = dict(metric_id="x", source="P", epoch="1", status="PASS",
               labels={}, provenance={}, points=[[10, 2]])
    document = dict(metrics_schema_version=1, definitions={"x": dict(unit="count")},
                    metrics={"x": [row]})
    if field == "evidence":
        row["provenance"][field] = None
    elif field in {"labels", "provenance"}:
        row[field] = None
    else:
        document[field] = None
    with pytest.raises(MetricContractError):
        MetricStore(document)


@pytest.mark.parametrize("data", [{}, {"id": "a", "checks": [{}]},
    {"id": "a", "checks": [{"stage": "gate", "status": "PASS"}]}])
def test_presentation_missing_identity_or_stage_is_a_contract_error(data):
    from reporting.run_model import RunPresentation
    with pytest.raises(ValueError):
        RunPresentation.read(data)


def test_report_rejects_nonfinite_axis_metadata():
    from reporting import render
    with pytest.raises(ValueError):
        render(dict(timeAxis=dict(min=0,max=1),panels=[dict(id="a",timeX=True,
            axes={"y":dict(min=float("nan"))},series=[])]))


def test_long_report_identity_is_bounded_and_collision_resistant(tmp_path):
    from reporting import write_bundle
    a=write_bundle(tmp_path,"run","a"*300,{},dict(panels=[]))
    b=write_bundle(tmp_path,"run","a"*299+"b",{},dict(panels=[]))
    assert len(a.name)<=105 and a!=b


def test_directory_analysis_verifies_bundle_once(tmp_path,monkeypatch):
    import reporting.core as core
    from unittest.mock import Mock
    bundle=core.write_bundle(tmp_path,"run","a",{"value":1},dict(panels=[]))
    read=Mock(wraps=core.read_bundle);monkeypatch.setattr(core,"read_bundle",read)
    assert core.load_analysis(bundle)=={"value":1}
    assert read.call_count==1


def test_uncalibrated_fidelity_profile_retains_unavailable_identity(tmp_path):
    import json
    from test_traffic_fidelity import fixture
    from analysis.traffic_fidelity import run
    raw,profile=fixture();profile["calibration"]={}
    source=tmp_path/"source.xz";source.write_bytes(raw)
    path=tmp_path/"profile.json";path.write_text(json.dumps(profile))
    result=run(path,[source],count=20)
    assert result["identity"]["status"]=="UNAVAILABLE"
    assert result["rows"][0]["fit_sha256"] is None


def test_https_probe_keeps_scheme_host_port_and_path(tmp_path,monkeypatch):
    import json
    from unittest.mock import Mock,MagicMock
    from types import SimpleNamespace
    from monitoring.session import PrometheusSession
    session=object.__new__(PrometheusSession)
    session.url="http://prometheus";session.closed=False;session.probes={};session.directory=tmp_path
    session.query=Mock(return_value=[{"value":[1,"1"]}]);session._write_session=Mock()
    (tmp_path/"prometheus.json").write_text(json.dumps({"scrape_configs":[]}))
    monkeypatch.setattr("urllib.request.urlopen",Mock(return_value=MagicMock()))
    probe=SimpleNamespace(job="probe",names={"up":"probe_up"},error=None)
    session.add_probe(probe,"https://[::1]:8443/custom-metrics")
    config=json.loads((tmp_path/"prometheus.json").read_text())["scrape_configs"][0]
    assert config["scheme"]=="https" and config["metrics_path"]=="/custom-metrics"
    assert config["static_configs"][0]["targets"]==["[::1]:8443"]


def test_invalid_survivor_measurement_cannot_be_selected(tmp_path):
    from metric_fixtures import freeze_metrics
    from cases.cache_scale_in.metrics import produce
    from monitoring.metric_store import MetricStore,MetricUnavailable
    freeze_metrics(tmp_path,"cache_scale_in")
    evidence=dict(provenance=dict(env_epoch=1),clock=dict(origin_epoch_s=100),survivors=["p"])
    result=dict(verdict="INVALID",windows=[dict(end=10,hit=.9,errors=[])],gate_metrics={})
    produce(tmp_path,evidence,result)
    with pytest.raises(MetricUnavailable):
        MetricStore.read(tmp_path).select("derived/survivor_hit_ratio")


@pytest.mark.parametrize("spec", [None, {"origin":None}])
def test_producer_source_without_file_has_clear_error(tmp_path,monkeypatch,spec):
    from types import SimpleNamespace
    from metric_fixtures import freeze_metrics
    from monitoring.metric_store import publish,MetricContractError
    store=freeze_metrics(tmp_path,"cache_scale_in")
    monkeypatch.setattr("importlib.util.find_spec",lambda name: None if spec is None else SimpleNamespace(**spec))
    definition=store.document["definitions"]["derived/survivor_hit_ratio"]
    with pytest.raises(MetricContractError,match="producer module"):
        publish(store,"derived/survivor_hit_ratio",definition,[],producer="cache_windows",evidence={})


@pytest.mark.parametrize("metrics", [{}, {"x": [dict(metric_id="x", epoch="1", source="P",
    labels={}, points=[[10, 2], [11, None]], provenance={})]}])
def test_minimal_metric_store_exports_without_optional_diagnostics(metrics):
    from monitoring.metric_store import MetricStore
    store=MetricStore(dict(metrics_schema_version=1,definitions={"x":dict(unit="count")},metrics=metrics))
    series,sources,gaps,errors=store.series(10)
    assert gaps=={} and errors==[]
    assert list(series.values())==([[ [0,2], [1,None] ]] if metrics else [])
    assert len(sources)==len(series)


def test_metric_store_preserves_and_copies_recorded_diagnostics():
    from monitoring.metric_store import MetricStore
    document=dict(metrics_schema_version=1,definitions={},metrics={},
        collection_gaps={"P":[10,12]},errors=[dict(reason="scrape failed")])
    _,_,gaps,errors=MetricStore(document).series(10)
    assert gaps=={"P":[0,2]} and errors==document["errors"]
    errors[0]["reason"]="changed"
    assert document["errors"][0]["reason"]=="scrape failed"


@pytest.mark.parametrize("kind", ["mismatch", "value_error", "contract_error"])
def test_published_contract_errors_keep_the_original_cause(tmp_path,monkeypatch,kind):
    from metric_fixtures import freeze_metrics
    from monitoring.metric_store import publish,MetricContractError
    store=freeze_metrics(tmp_path,"cache_scale_in")
    definition=store.document["definitions"]["derived/survivor_hit_ratio"]
    failure=ValueError("bad calculation") if kind=="value_error" else MetricContractError("bad contract")
    def contract(*args):
        if kind=="mismatch":return {}
        raise failure
    monkeypatch.setattr("monitoring.producers.output_contract",contract)
    with pytest.raises(MetricContractError) as raised:
        publish(store,"derived/survivor_hit_ratio",definition,[],producer="cache_windows",evidence={})
    if kind=="value_error":
        assert raised.value.__cause__ is failure
    else:
        assert raised.value.__cause__ is None
        if kind=="contract_error":assert raised.value is failure
