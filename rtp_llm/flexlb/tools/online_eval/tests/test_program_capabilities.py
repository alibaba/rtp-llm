"""Extension contracts: new programs and report identities need no generic branches."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import yaml

from cases.config import configure_program
from cases.programs import PROGRAMS
from reporting import discover_reports, write_bundle
from scenario import ScenarioError, compile_scenarios
from scenario.catalog import handlers
from workload.cache_gate_ab import compare, load_comparison_policy
from workload.evidence_analysis import analyze_report
from workload.report import build_spec
import mode_profiles
from flexlb_profile_data import REGISTERED_PROFILE_SPECS

ROOT = Path(__file__).resolve().parents[1]


class ProgramCapabilitiesTest(unittest.TestCase):
    def test_second_program_loads_and_compiles_with_declared_analysis(self):
        original = yaml.safe_load((ROOT / 'config/scenarios/cache_scale_in.yaml').read_text())
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            module_path = root / 'second_cache.py'
            module_path.write_text((ROOT / 'src/cases/programs/cache_scale_in.py').read_text())
            module = types.ModuleType('second_cache')
            module.__file__ = str(module_path)
            exec(compile(module_path.read_text(), str(module_path), 'exec'), module.__dict__)
            config = copy.deepcopy(original)
            config['case'] = 'second_cache'
            path = root / 'scenario.yaml'
            path.write_text(yaml.safe_dump(config))
            with mock.patch.dict(PROGRAMS, second_cache='second_cache'), mock.patch.dict(sys.modules, second_cache=module):
                self.assertEqual(load_comparison_policy(path), original['analysis'])
                with mock.patch('scenario.compiler.VICTIM_OFFSETS', (700, 701, 702)):
                    first = compile_scenarios([('first', configure_program(original, 'first'))], handlers=handlers())
                    second = compile_scenarios([('second', configure_program(config, 'second'))], handlers=handlers())
                self.assertEqual([p['stages'] for p in first], [p['stages'] for p in second])
                del module.ANALYSIS_POLICY_VALIDATOR
                with self.assertRaisesRegex(ScenarioError, 'does not support analysis'):
                    configure_program(config, 'second')
                with self.assertRaisesRegex(ScenarioError, 'does not support analysis'):
                    load_comparison_policy(path)
                config['analysis']['action'] = 'grant_permission'
                with self.assertRaisesRegex(ScenarioError, 'cannot orchestrate'):
                    configure_program(config, 'second')

    def test_gate_discovery_and_workload_links_use_manifest_role(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            spec = dict(run_id='fixture', title='fixture', panels=[], sections=[], kpis=[])
            expected = []
            for name in ('different-cache', 'second-gate'):
                bundle = write_bundle(root, 'run', name, {}, spec, role='gate')
                expected.append(str((bundle / 'report.html').resolve()))
            write_bundle(root, 'run', 'ordinary', {}, spec)
            result = dict(id='fixture', status='PASS', stages=[], workload=dict(runtime_validity='VALID'))
            evidence = dict(clock_anchor=dict(epoch_s=0), phases=[])
            with mock.patch('monitoring.session.archived_series', return_value=({}, {}, {}, [])):
                payload = analyze_report(root, result, evidence)
            self.assertEqual(payload['gate_reports'], expected)
            self.assertEqual(payload['gate_report'], expected[0])
            report = build_spec(payload, root)
            self.assertIn('different-cache', json.dumps(report))
            self.assertIn('second-gate', json.dumps(report))
            Path(expected[0]).write_text('corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                discover_reports(root, role='gate')

    def test_new_registered_profile_and_runtime_are_table_driven(self):
        tables = mode_profiles.load_mode_tables()
        tables['runtime_modes']['second'] = dict(tables['runtime_modes']['functional'],
                                                default_profile='second-profile', default_master_mode='sn')
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(
            REGISTERED_PROFILE_SPECS, {'second-profile': dict(decision='single', dispatcher='non_batch')}
        ):
            path = Path(d) / 'modes.yaml'
            path.write_text(yaml.safe_dump(tables))
            loaded = mode_profiles.load_mode_tables(path)
            self.assertEqual(mode_profiles.resolve_mode('second', 'sn', tables=loaded)['master_profile'], 'second-profile')
            self.assertEqual(mode_profiles.master_mode_for_profile('second-profile'), 'sn')
            for patch, message in (({'default_profile': 'missing'}, 'unknown default'),
                                   ({'default_master_mode': 'missing'}, 'unknown default'),
                                   ({'default_master_mode': 'wb'}, 'axes disagree')):
                bad = copy.deepcopy(tables)
                bad['runtime_modes']['second'].update(patch)
                path.write_text(yaml.safe_dump(bad))
                with self.assertRaisesRegex(ValueError, message):
                    mode_profiles.load_mode_tables(path)

    def test_compare_validates_actual_event_before_reading_evidence(self):
        for event in ('', '  ', 12, False):
            with self.assertRaisesRegex(ValueError, 'alignment_event'):
                compare('missing-a', 'missing-b', 'unused', alignment_event=event)
