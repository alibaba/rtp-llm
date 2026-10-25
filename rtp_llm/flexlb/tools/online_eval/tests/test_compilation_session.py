"""Compilation inputs have scoped lifetime and isolated mutable projections."""

import pytest

from scenario.registry import compilation_scope, load_metric_plan


def test_metric_cache_is_scoped_nested_and_returns_isolated_projections(monkeypatch):
    calls = []
    current = {"value": 1}

    def read(name, **kwargs):
        calls.append(name)
        return {"values": [current["value"]]}

    monkeypatch.setattr("monitoring.query_plan.load_plan", read)
    with compilation_scope():
        first = load_metric_plan("test.yaml")
        first["values"].append(2)
        with compilation_scope():
            assert load_metric_plan("test.yaml") == {"values": [1]}
        assert calls == ["test.yaml"]
    current["value"] = 3
    with compilation_scope():
        assert load_metric_plan("test.yaml") == {"values": [3]}
    assert calls == ["test.yaml", "test.yaml"]


def test_failed_compilation_does_not_leave_cached_inputs(monkeypatch):
    calls = []
    monkeypatch.setattr("monitoring.query_plan.load_plan", lambda *args, **kwargs: calls.append(1) or {})
    with pytest.raises(RuntimeError), compilation_scope():
        load_metric_plan("test.yaml")
        raise RuntimeError("builder failed")
    load_metric_plan("test.yaml")
    assert len(calls) == 2
