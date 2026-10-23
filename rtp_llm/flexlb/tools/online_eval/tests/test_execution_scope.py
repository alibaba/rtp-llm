"""Frozen children depend on selected business code and shared runtime code."""
import json

import pytest

from case_api.registry import discover, registry, using_cases
from scenario.compiler import ScenarioCompiler
from scenario.execution_plan import freeze_plan, load_plan
from test_scenario_runtime import source


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    import scenario.execution_plan as plans
    instance = ScenarioCompiler().compile([('fixture', source())])[0].to_dict()
    instance['implementation'] = {'configuration': {'case': 'request_completion'}}
    for relative in ('src/cases/request_completion/program.py', 'src/cases/other_case/program.py',
                     'src/runtime/shared.py', 'src/cases/__init__.py',
                     'config/report_views/unselected.yaml', 'src/reporting/assets/chart.js'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# initial source\n')
    monkeypatch.setattr(plans, 'ROOT', tmp_path)
    plan = tmp_path / 'plan.json'
    sha = freeze_plan(plan, [instance])
    return tmp_path, plan, sha


@pytest.mark.parametrize('changed', ['src/cases/other_case/program.py',
                                   'config/report_views/unselected.yaml', 'src/reporting/assets/chart.js'])
def test_unrelated_case_or_unused_presentation_does_not_invalidate_frozen_plan(frozen, changed):
    root, plan, sha = frozen
    (root / changed).write_text('changed unselected input')
    assert load_plan(plan, sha)[0]['id'].startswith('lifecycle::')
    dependencies = json.loads(plan.read_text())['dependencies']
    assert 'src/cases/other_case/program.py' not in dependencies
    assert 'src/cases/request_completion/program.py' in dependencies
    assert 'config/report_views/unselected.yaml' not in dependencies
    assert 'src/reporting/assets/chart.js' not in dependencies
    assert 'src/runtime/shared.py' in dependencies
    assert 'src/cases/__init__.py' in dependencies


@pytest.mark.parametrize('changed', ['src/cases/request_completion/program.py', 'src/runtime/shared.py'])
def test_selected_or_shared_source_change_invalidates_frozen_plan(frozen, changed):
    root, plan, sha = frozen
    (root / changed).write_text('# changed source\n')
    with pytest.raises(ValueError, match='code/config changed'):
        load_plan(plan, sha)


def test_registry_scope_is_owned_and_restored():
    original = registry()
    with using_cases({'request_completion'}):
        assert set(registry().cases) == {'request_completion'}
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as workers:
            assert workers.submit(lambda: set(registry().cases)).result() == {'request_completion'}
    assert registry() is original


def test_selected_discovery_does_not_import_a_broken_unrelated_case(tmp_path, monkeypatch):
    package = tmp_path / 'scope_fixture'
    package.mkdir()
    (package / '__init__.py').write_text('')
    for name in ('selected', 'unselected'):
        path = package / name
        path.mkdir()
        (path / '__init__.py').write_text('')
        (path / 'program.py').write_text(
            'from case_api.registry import CaseDefinition\nCASE = CaseDefinition({"default": lambda case: None})\n'
            if name == 'selected' else 'raise RuntimeError("unrelated import must not execute")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    result = discover(package, package='scope_fixture', names={'selected'})
    assert set(result.cases) == {'selected'}
    with pytest.raises(ValueError, match='unknown selected cases'):
        discover(package, package='scope_fixture', names={'missing'})


def test_workload_keeps_selected_views_and_shared_rendering_assets():
    from scenario.execution_plan import scope_dependencies
    instance = {'implementation': {'configuration': {'case': 'master_performance'}},
                'metadata': {'kind': 'workload'}, 'reports': ['master_performance.yaml']}
    dependencies = dict.fromkeys(('config/report_views/default.yaml',
                                  'config/report_views/master_performance.yaml',
                                  'config/report_views/cache_scale_in.yaml',
                                  'src/reporting/assets/chart.js'), 'digest')
    selected = scope_dependencies(dependencies, [instance])
    assert set(selected) == set(dependencies) - {'config/report_views/cache_scale_in.yaml'}
