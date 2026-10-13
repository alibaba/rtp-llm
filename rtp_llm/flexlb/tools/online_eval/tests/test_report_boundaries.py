"""Report value ownership and extension boundaries remain independent of case names."""
import ast
from pathlib import Path
from unittest.mock import patch
import pytest
from reporting.view_config import view
from reporting.view_model import ViewDefinition
from reporting.report_model import ExperimentConditions, MeasurementIdentity


def test_subtitle_fields_are_registered_not_inferred_from_producer():
    from case_api.registry import ReportView
    raw = view('master_ha_failover.yaml').to_dict()
    raw['report'].update(producer='arbitrary', subtitle='{verdict} / {trial}')
    capability = ReportView(lambda *args: None, lambda *args: None, {'trial'})
    with patch('reporting.view_config.load_document', return_value=raw), \
         patch('reporting.view_config.view_capabilities', return_value={'master_ha_failover.yaml':capability}):
        assert view('master_ha_failover.yaml').subtitle.render(verdict='PASS', trial='A') == 'PASS / A'
    raw['report']['producer'] = 'cache-gate'
    with pytest.raises(ValueError, match='unknown subtitle'):
        ViewDefinition.read(raw)


def test_view_values_reject_nested_mutation_and_exports_are_detached():
    config = view('cache_scale_in.yaml')
    raw = config.to_dict()
    key = next(iter(config.charts.events))
    with pytest.raises(TypeError):
        config.charts.events[key]['label'] = 'changed'
    with pytest.raises(TypeError):
        config.sections['added'] = next(iter(config.sections.values()))
    with pytest.raises(TypeError):
        config.charts.curves['added'] = next(iter(config.charts.curves.values()))
    raw['charts']['events'][key]['label'] = 'new'
    assert config.charts.events[key]['label'] != 'new'
    assert view('cache_scale_in.yaml').to_dict() == config.to_dict()


def test_measurement_and_conditions_do_not_alias_mutable_inputs():
    raw = dict(metric_id='metric', labels={'role':'p'}, measurement={'sources':['a']})
    identity = MeasurementIdentity.read(raw)
    signature = identity.signature()
    raw['labels']['role'] = 'd'
    raw['measurement']['sources'].append('b')
    assert identity.signature() == signature
    with pytest.raises(TypeError):
        identity.labels['role'] = 'changed'
    conditions = ExperimentConditions.read({'configuration':{'settings':[{'cap':2}]}})
    exported = conditions.to_dict()
    exported['configuration']['settings'][0]['cap'] = 99
    assert conditions.to_dict()['configuration']['settings'][0]['cap'] == 2
    with pytest.raises(TypeError):
        conditions.configuration['settings'][0]['cap'] = 99


def test_report_modules_do_not_import_scenario_or_monitoring_in_schema():
    root = Path(__file__).parents[1] / 'src'
    for name in ('reporting/view_config.py','reporting/events.py','reporting/view_schema.py'):
        imports = [node.module for node in ast.walk(ast.parse((root/name).read_text()))
                   if isinstance(node, ast.ImportFrom) and node.module]
        assert not any(module.startswith('scenario') for module in imports)
        if name.endswith('view_schema.py'):
            assert not any(module.startswith('monitoring') for module in imports)
    for name in ('reporting/components.py','reporting/report_model.py'):
        imports = [node.module for node in ast.walk(ast.parse((root/name).read_text()))
                   if isinstance(node, ast.ImportFrom)]
        assert 'reporting.core' not in imports and 'reporting.renderer' not in imports


def test_default_publication_decodes_run_once(tmp_path):
    from test_workload_report_views import payload
    from reporting.workload import write_report
    from reporting.run_model import RunPresentation
    with patch.object(RunPresentation, 'read', wraps=RunPresentation.read) as read:
        write_report(tmp_path, payload())
    assert read.call_count == 1


def test_subtitle_extension_names_require_a_collection():
    from case_api.registry import ReportView
    from reporting.view_model import SubtitleTemplate
    with pytest.raises(ValueError, match='collection'):
        ReportView(lambda *args: None, lambda *args: None, 'trial')
    with pytest.raises(ValueError, match='collection'):
        SubtitleTemplate('{verdict}', fields='trial')
    with pytest.raises(ValueError, match='collection'):
        ViewDefinition.read({'kind':'default','report':{'subtitle':'{verdict}'},'charts':{}},
                            subtitle_fields='trial')
