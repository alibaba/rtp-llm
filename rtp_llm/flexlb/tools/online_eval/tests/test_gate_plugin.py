"""Frozen gate functions separate adjudication from metric and HTML delivery."""

from unittest.mock import Mock

import pytest

from artifacts.errors import ArtifactNotProduced
from artifacts.gate import commit_gate, freeze_gate, load_gate, project_gate, publish_gate, render_gate


def decision():
    return dict(verdict="PASS", checks=[dict(id="business", status="PASS")], errors=[])


def test_live_commit_does_not_publish_or_render(tmp_path):
    analyze = Mock(return_value=decision())
    result, path = commit_gate(tmp_path, "example", {"samples": [1]}, analyze)
    assert path.name == "example-gate-result.json" and result == decision()
    assert load_gate(tmp_path, "example") == ({"samples": [1]}, decision())
    analyze.assert_called_once_with({"samples": [1]})
    assert {p.name for p in tmp_path.iterdir()} == {"example-gate-evidence.json", "example-gate-result.json", "example-gate-manifest.json"}


def test_publication_uses_committed_data_and_never_readjudicates(tmp_path):
    project = Mock(side_effect=lambda directory, evidence, result: result.update(verdict="FAIL"))
    report = Mock(return_value=tmp_path / "report")
    assert publish_gate(tmp_path, "example", {}, decision(), project, report) == tmp_path / "report"
    assert report.call_args.args[2]["verdict"] == "PASS"
    assert load_gate(tmp_path, "example")[1]["verdict"] == "PASS"


def test_delivery_error_preserves_gate_and_never_renders(tmp_path):
    report = Mock()
    with pytest.raises(RuntimeError, match="projection failed"):
        publish_gate(tmp_path, "example", {}, decision(), Mock(side_effect=RuntimeError("projection failed")), report)
    assert load_gate(tmp_path, "example")[1]["verdict"] == "PASS"
    report.assert_not_called()


def test_corrupt_gate_cannot_reach_projection_or_presentation(tmp_path):
    freeze_gate(tmp_path, "example", {}, decision())
    (tmp_path / "example-gate-result.json").write_text("{}")
    project, report = Mock(), Mock()
    for call in (
        lambda: project_gate(tmp_path, "example", project),
        lambda: render_gate(tmp_path, {}, {}, name="example", write_report=report),
    ):
        with pytest.raises(ValueError, match="checksum"):
            call()
    project.assert_not_called()
    report.assert_not_called()


def test_selected_view_passes_run_and_presentation_without_reanalysis(tmp_path):
    freeze_gate(tmp_path, "example", {}, decision())
    report = Mock(return_value=tmp_path / "report")
    run, presentation = {"status": "FAIL"}, {"title": "test"}
    assert render_gate(tmp_path, run, presentation, name="example", write_report=report) == tmp_path / "report/report.html"
    assert report.call_args.kwargs == dict(run=run, presentation=presentation)


def test_missing_commit_skips_projection_but_selected_view_fails(tmp_path):
    project, report = Mock(), Mock()
    project_gate(tmp_path, "example", project)
    with pytest.raises(ArtifactNotProduced):
        render_gate(tmp_path, {}, {}, name="example", write_report=report)
    project.assert_not_called()
    report.assert_not_called()


def test_inconsistent_verdict_cannot_commit(tmp_path):
    with pytest.raises(ValueError, match="verdict"):
        commit_gate(tmp_path, "example", {}, lambda evidence: dict(decision(), verdict="FAIL"))
    assert not list(tmp_path.iterdir())
