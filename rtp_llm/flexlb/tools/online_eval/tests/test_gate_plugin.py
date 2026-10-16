"""Plugin lifecycle: adjudication is explicit and independent of delivery."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from case_api.gate_plugin import GatePlugin
from artifacts.gate import load_gate
from artifacts.errors import ArtifactNotProduced


def decision():
    return dict(verdict='PASS', checks=[dict(id='business', status='PASS')], errors=[])


def plugin(tmp_path):
    return GatePlugin('example', Mock(return_value=decision()), Mock(),
                      Mock(return_value=tmp_path / 'report'))


def test_live_commit_does_not_publish_or_render(tmp_path):
    gate = plugin(tmp_path)
    committed = gate.commit(tmp_path, {'samples': [1]})
    assert committed.path.name == 'example-gate-result.json'
    assert load_gate(tmp_path, 'example') == ({'samples': [1]}, decision())
    gate.analyze.assert_called_once_with({'samples': [1]})
    gate.project_metrics.assert_not_called()
    gate.write_report.assert_not_called()


def test_publication_uses_committed_data_and_never_readjudicates(tmp_path):
    gate = plugin(tmp_path)
    gate.analyze.side_effect = AssertionError('reanalysis')
    # Mutating an in-memory projection cannot rewrite the committed verdict.
    gate.project_metrics.side_effect = lambda directory, evidence, result: result.update(verdict='FAIL')
    assert gate.publish(tmp_path, {'samples': [1]}, decision()) == tmp_path / 'report'
    assert gate.write_report.call_args.args[2]['verdict'] == 'PASS'
    assert load_gate(tmp_path, 'example')[1]['verdict'] == 'PASS'
    gate.analyze.assert_not_called()


def test_delivery_error_preserves_gate_and_never_renders(tmp_path):
    gate = plugin(tmp_path)
    gate.project_metrics.side_effect = RuntimeError('projection failed')
    with pytest.raises(RuntimeError, match='projection failed'):
        gate.publish(tmp_path, {}, decision())
    assert load_gate(tmp_path, 'example')[1]['verdict'] == 'PASS'
    gate.write_report.assert_not_called()


def test_corrupt_gate_cannot_reach_projection_or_presentation(tmp_path):
    gate = plugin(tmp_path)
    gate.commit(tmp_path, {})
    (tmp_path / 'example-gate-result.json').write_text('{}')
    for call in (lambda: gate.project(tmp_path), lambda: gate.render(tmp_path, {}, {})):
        with pytest.raises(ValueError, match='checksum'):
            call()
    gate.project_metrics.assert_not_called()
    gate.write_report.assert_not_called()


def test_render_passes_selected_view_and_does_not_analyze(tmp_path):
    gate = plugin(tmp_path)
    gate.commit(tmp_path, {})
    gate.analyze.reset_mock()
    view = {'report': {'id': 'alternate'}}
    run = {'id': 'example::default::batch-window'}
    assert gate.render(tmp_path, run, view) == tmp_path / 'report/report.html'
    assert gate.write_report.call_args.kwargs == dict(run=run, presentation=view)
    gate.analyze.assert_not_called()


def test_missing_gate_is_not_fabricated(tmp_path):
    gate = plugin(tmp_path)
    gate.project(tmp_path)  # Aborted runs need not have a committed gate.
    with pytest.raises(ArtifactNotProduced):
        gate.render(tmp_path, {}, {})
    gate.analyze.assert_not_called()
    gate.project_metrics.assert_not_called()


def test_inconsistent_analysis_cannot_commit(tmp_path):
    gate = plugin(tmp_path)
    gate.analyze.return_value = dict(decision(), verdict='FAIL')
    with pytest.raises(ValueError, match='verdict disagrees'):
        gate.commit(tmp_path, {})
    assert not list(tmp_path.iterdir())


def test_report_callback_contract_is_checked_before_execution(tmp_path):
    with pytest.raises(ValueError, match='presentation keywords'):
        GatePlugin('example', lambda evidence: decision(), lambda a, b, c: None,
                   lambda directory, evidence, result: Path(directory))
