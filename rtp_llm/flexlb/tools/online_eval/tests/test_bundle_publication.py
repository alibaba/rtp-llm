"""Publication failures preserve committed reports; checked loads do not repeat I/O."""

import collections
from pathlib import Path
from unittest.mock import patch

import pytest

from reporting import load_analysis, read_bundle, write_bundle
from reporting.comparison_inputs import load_snapshot


@pytest.mark.parametrize("target", ["reporting.core._render_validated", "reporting.core.write_text"])
def test_failed_republish_preserves_the_entire_committed_bundle(tmp_path, target):
    evidence = dict(verdict="PASS", series={"metric": [[0, 1], [1, None]]}, statistic_sources={})
    bundle = write_bundle(tmp_path, "run", "case", evidence, dict(panels=[]), attachments={"old.html": "old attachment"})
    before = {path.name: path.read_bytes() for path in bundle.iterdir()}
    with patch(target, side_effect=OSError("injected publication failure")), pytest.raises(OSError, match="injected"):
        write_bundle(tmp_path, "run", "case", dict(verdict="FAIL"), dict(panels=[]))
    assert {path.name: path.read_bytes() for path in bundle.iterdir()} == before
    assert load_analysis(bundle) == evidence
    assert not list(bundle.parent.glob(".report-*"))


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_replacement_failure_restores_or_retains_the_previous_report(tmp_path, monkeypatch, rollback_fails):
    bundle = write_bundle(tmp_path, "run", "case", dict(verdict="PASS"), dict(panels=[]))
    replace = Path.replace

    def fail_publication(path, target):
        if path.name == "bundle" or rollback_fails and path.name == "previous":
            raise OSError("injected replacement failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_publication)
    with pytest.raises(OSError, match="injected replacement"):
        write_bundle(tmp_path, "run", "case", dict(verdict="FAIL"), dict(panels=[]))
    previous = next(bundle.parent.glob(".report-*/previous")) if rollback_fails else bundle
    assert load_analysis(previous) == dict(verdict="PASS")
    if not rollback_fails:
        assert not list(bundle.parent.glob(".report-*"))


def test_successful_replacement_contains_only_current_members(tmp_path):
    bundle = write_bundle(tmp_path, "run", "case", {}, dict(panels=[]), attachments={"old.html": "old"})
    write_bundle(tmp_path, "run", "case", dict(verdict="FAIL"), dict(panels=[]), attachments={"new.html": "new"})
    assert read_bundle(bundle) == bundle
    assert not (bundle / "old.html").exists()
    assert (bundle / "new.html").read_text() == "new"
    assert not list(bundle.parent.glob(".report-*"))


def test_comparison_load_reads_each_canonical_member_once(tmp_path, monkeypatch):
    bundle = write_bundle(tmp_path, "run", "case", dict(verdict="PASS"), dict(panels=[]))
    counts = collections.Counter()
    open_file = Path.open

    def counted(path, *args, **kwargs):
        if path.parent == bundle:
            counts[path.name] += 1
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    snapshot = load_snapshot(bundle)
    assert snapshot.outcome["verdict"] == "PASS"
    assert counts == {"manifest.json": 1, "analysis.json": 1, "report-spec.json": 1, "report.html": 1}
