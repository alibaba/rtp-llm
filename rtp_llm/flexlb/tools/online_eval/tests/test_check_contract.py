"""Shared checks distinguish threshold failures, missing evidence and contract bugs."""

import copy

import pytest

from analysis.checks import compare, evaluate, check_metric
from monitoring.metric_store import MetricStore, MetricContractError


def store():
    return MetricStore(dict(metrics_schema_version=1,
        definitions={"example/value": dict(unit="requests", value_kind="gauge", labels=["worker"])},
        metrics={"example/value": [dict(source="test", epoch="1", labels=dict(worker="A"),
            status="OK", points=[[10, 0], [11, 2], [12, 4]],
            provenance=dict(source_type="prometheus", expression="frozen_query"))]}))


@pytest.mark.parametrize("actual,op,expected,passed", [
    (True, "eq", True, True), (False, "eq", True, False),
    ("done", "eq", "done", True), (1, "ge", 1.0, True),
    (3, "le", 2, False), (0, "eq", 0, True),
])
def test_scalar_comparison_preserves_type_and_boundary_semantics(actual, op, expected, passed):
    assert compare(actual, op, expected) is passed
    result = evaluate("criterion", actual, op, expected)
    assert result.status == ("PASS" if passed else "FAIL")
    assert evaluate("criterion", actual, op, expected, advisory=True).status == (
        "PASS" if passed else "WARNING")


@pytest.mark.parametrize("actual,op,expected", [
    (1, "eq", True), (True, "eq", 1), (None, "eq", 0),
    (float("nan"), "le", 1), (1, "ge", float("inf")),
    (True, "ge", False), (1, "unknown", 1), ([], "eq", []),
])
def test_no_missing_as_zero_coercion_or_unbounded_comparison(actual, op, expected):
    with pytest.raises(ValueError):
        compare(actual, op, expected)


def metric_check(data, **kwargs):
    return check_metric(data, "criterion", "example/value", reduction="max", op="le",
                        expected=2, **kwargs)


def test_metric_window_uses_frozen_samples_and_retains_provenance_without_mutation():
    data = store()
    original = copy.deepcopy(data.document)
    result = metric_check(data, start=10, end=11, labels=dict(worker="A"),
                          min_samples=2, max_gap_s=1)
    assert result.status == "PASS"
    assert result.actual == 2
    assert metric_check(data, start=11, end=12).status == "FAIL"
    assert result.evidence["selection"]["start"] == 10
    assert result.evidence["observations"][0]["provenance"]["expression"] == "frozen_query"
    result.evidence["definition"]["unit"] = "changed"
    assert data.document == original


@pytest.mark.parametrize("selection", [
    dict(labels=dict(worker="missing")), dict(source="missing"), dict(epoch=2),
    dict(start=10, end=12, min_samples=4), dict(start=10, end=12, max_gap_s=0.5),
])
def test_unavailable_data_is_error_even_for_advisory_checks(selection):
    result = metric_check(store(), advisory=True, **selection)
    assert result.status == "ERROR"
    assert result.evidence["validity"] == "INVALID"
    assert result.actual is None


def test_missing_point_and_query_failure_cannot_pass_a_zero_threshold():
    for change in (dict(points=[[10, None]]), dict(status="ERROR")):
        data = store()
        data.document["metrics"]["example/value"][0].update(change)
        assert metric_check(data).status == "ERROR"


def test_unknown_definition_or_ambiguous_series_remains_a_contract_error():
    data = store()
    with pytest.raises(MetricContractError):
        check_metric(data, "criterion", "example/missing", reduction="last", op="eq", expected=0)
    data.document["metrics"]["example/value"].append(copy.deepcopy(
        data.document["metrics"]["example/value"][0]))
    with pytest.raises(MetricContractError, match="exactly one"):
        metric_check(data)
