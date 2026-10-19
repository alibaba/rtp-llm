import copy
import io
import json
from unittest.mock import patch

import pytest

from adapters.debug_client import DebugUnavailable
from adapters.master_diagnostics import MasterDiagnosticsClient, DiagnosticCapture, validate_snapshot


def payload():
    return dict(schema_version=2, instance_id="process-1", snapshot_id="sample-1", status="ok",
                capture_started_at_ms=1, capture_finished_at_ms=2,
                capabilities={"workers": dict(version=1, features=["poll_health", "applied_observation", "cache_progress"])},
                components={"workers": dict(status="ok", capture_started_at_ms=1, capture_finished_at_ms=2,
                    coverage=dict(scope="prefill_decode_endpoint_registry", complete=True, truncated=False, returned_count=1),
                    rows=[dict(worker_id="prefill:a", address="a", role="prefill", generation="9007199254740993",
                        lifecycle_state="active", last_reported_alive=True, poll_response_age_ms=10,
                        applied_observation_age_ms=20, applied_status_version="1", received_cache_version="2",
                        indexed_cache_version="2", index_initialized=True, cached_key_count=3)])})


def test_generation_is_preserved_and_workers_are_domain_rows():
    with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload()).encode())):
        capture = MasterDiagnosticsClient("http://localhost").snapshot(include=("workers",))
    assert capture.component("workers")["rows"][0]["generation"] == "9007199254740993"


@pytest.mark.parametrize("change", ["coverage", "truncated", "partial"])
def test_incomplete_component_cannot_prove_absence(change):
    data = payload()
    page = data["components"]["workers"]
    if change == "partial":
        page["status"] = data["status"] = "partial"
    else:
        page["coverage"]["complete" if change == "coverage" else "truncated"] = change == "truncated"
    with pytest.raises(DebugUnavailable):
        DiagnosticCapture(validate_snapshot(data), 1, 2).component("workers")


@pytest.mark.parametrize("field,value", [("generation", 1), ("poll_response_age_ms", True),
    ("applied_observation_age_ms", -1), ("applied_observation_age_ms", float("nan")),
    ("last_reported_alive", 1), ("index_initialized", 1), ("cached_key_count", -1)])
def test_invalid_typed_fields_fail_closed(field, value):
    data = payload()
    data["components"]["workers"]["rows"][0][field] = value
    with pytest.raises(DebugUnavailable):
        validate_snapshot(data)


def test_unknown_health_remains_unknown():
    data = payload()
    data["components"]["workers"]["rows"][0]["applied_observation_age_ms"] = None
    assert validate_snapshot(data)["components"]["workers"]["rows"][0]["applied_observation_age_ms"] is None


def test_duplicate_rows_are_not_a_complete_fleet():
    data = payload()
    page = data["components"]["workers"]
    page["rows"].append(copy.deepcopy(page["rows"][0]))
    page["coverage"]["returned_count"] = 2
    with pytest.raises(DebugUnavailable):
        validate_snapshot(data)


def test_transport_failure_is_explicit():
    with patch("urllib.request.urlopen", side_effect=TimeoutError("unavailable")), pytest.raises(DebugUnavailable):
        MasterDiagnosticsClient("http://localhost").snapshot()
