import copy
import json

from reporting import write_bundle, read_bundle
from reporting.renderer import _statistics_samples
from reporting.run_model import RunPresentation
from reporting.run_context import canonical_spec


def test_presentation_copies_metadata_without_traversing_time_series():
    class Evidence(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("time-series evidence must not be copied for presentation metadata")
    analysis = dict(id="example", workload={}, configuration={"nested": [1]}, series=Evidence())
    run = RunPresentation.read(analysis)
    analysis["configuration"]["nested"].append(2)
    assert run.configuration == {"nested": [1]}


def test_bundle_keeps_full_statistics_and_does_not_mutate_source(tmp_path):
    points = [dict(x=x, y=None if x == 2 else x / 3) for x in range(10)]
    spec = dict(panels=[dict(id="curves", type="line", timeX=True,
        axes={"y": dict(title="count")}, series=[dict(name="a", points=points[::3],
        statistics_points=points)])], timeAxis=dict(min=0, max=9))
    source = copy.deepcopy(spec)
    meta = dict(run_meta_schema_version=1, identity=dict(id="example", kind="run"))
    original_meta = copy.deepcopy(meta)
    bundle = write_bundle(tmp_path, "run", "example", {}, spec, meta=meta)
    assert read_bundle(bundle) == bundle
    assert spec == source and meta == original_meta
    frozen = json.loads((bundle / "report-spec.json").read_text())
    assert frozen["panels"][0]["series"][0]["statistics_points"] == points
    encoded = _statistics_samples(points)
    assert list(zip(encoded["times"], encoded["values"])) == [(p["x"], p["y"]) for p in points]
    html = (bundle / "report.html").read_text()
    assert '"statistics_samples":' in html and '"statistics_points":' not in html


def test_clock_projection_shifts_full_statistics_and_owns_containers():
    spec = dict(timeOriginEpochS=100, timeAxis=dict(min=0, max=10),
        panels=[dict(id="a", timeX=True, series=[dict(name="a", points=[dict(x=1,y=2)],
        statistics_points=[dict(x=1,y=2),dict(x=2,y=None)])])])
    original = copy.deepcopy(spec)
    analysis = dict(id="a", workload={}, report_timeline=dict(status="AVAILABLE",
        origin=dict(epoch_s=101,selector=dict(event="traffic_started")),
        start=dict(epoch_s=101),end=dict(epoch_s=110)))
    projected = canonical_spec(spec, analysis)
    curve = projected["panels"][0]["series"][0]
    assert curve["points"] == [dict(x=0,y=2)]
    assert curve["statistics_points"] == [dict(x=0,y=2),dict(x=1,y=None)]
    assert spec == original


def test_projection_accepts_absent_series_and_optional_statistics():
    analysis = dict(id="a", workload={}, report_timeline=dict(status="AVAILABLE",
        origin=dict(epoch_s=101,selector=dict(event="traffic_started")),
        start=dict(epoch_s=101),end=dict(epoch_s=110)))
    source = dict(timeOriginEpochS=100, panels=[dict(id="empty",timeX=True),
        dict(id="a",timeX=True,series=[dict(points=[dict(x=1,y=2)],statistics_points=None)])])
    result = canonical_spec(source,analysis)
    assert result["panels"][0]["series"] == []
    assert result["panels"][1]["series"][0]["statistics_points"] is None
