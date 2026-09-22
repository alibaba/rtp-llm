"""CI membership, instance kind and file location remain independent."""
import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from scenario import ScenarioError, load_scenarios
from scenario.suites import classify, default_suite, normalize_test, preselect_documents, suite_names

ROOT = Path(__file__).resolve().parents[1]


class SuiteOwnershipTest(unittest.TestCase):
    def test_mixed_file_filters_variants_before_compilation(self):
        docs = load_scenarios(ROOT / 'config/scenarios/master_lifecycle.yaml')
        original = copy.deepcopy(docs)
        functional = preselect_documents(docs, 'functional')
        workload = preselect_documents(docs, 'workload')
        self.assertEqual(['kill_single'], [v['id'] for v in functional[0][1]['variants']])
        self.assertEqual({'freeze_short_long', 'kill_dual_b_to_a'},
                         {v['id'] for v in workload[0][1]['variants']})
        self.assertEqual(docs, original)
        self.assertEqual(functional[0][1].implementation, docs[0][1].implementation)

    def test_ci_can_select_one_workload_independent_of_directory_or_kind(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'suites.yaml'
            path.write_text(yaml.safe_dump(dict(schema_version=2, default_suite='smoke',
                                               ci_suites={'smoke': ['custom::load']})))
            test = dict(kind='workload', description='Load under restart', collection='request')
            plan = dict(scenario_id='custom', variant_id='load', source_path='/any/core/place.yaml', test=test)
            self.assertEqual('smoke', default_suite(path))
            self.assertEqual(('smoke', 'functional', 'workload', 'all'), suite_names(path))
            self.assertEqual('workload', classify([plan], 'smoke', path)[0]['test_kind'])
            self.assertEqual([], classify([plan], 'functional', path))
            plan['source_path'] = '/different/workload/place.yaml'
            self.assertEqual('workload', classify([plan], 'smoke', path)[0]['test_kind'])
            with self.assertRaisesRegex(ScenarioError, 'missing: custom::load'):
                classify([], 'smoke', path)
            path.write_text(yaml.safe_dump(dict(schema_version=2, default_suite='smoke',
                                               ci_suites={'smoke': ['custom::load', 'custom::load']})))
            with self.assertRaisesRegex(ScenarioError, 'unique'):
                suite_names(path)

    def test_missing_metadata_and_invalid_monitoring_fail_loud(self):
        base = dict(kind='workload', description='Scale in', collection='request')
        for key in base:
            invalid = {k: v for k, v in base.items() if k != key}
            with self.subTest(key=key), self.assertRaises(ScenarioError):
                normalize_test(invalid)
        with self.assertRaisesRegex(ScenarioError, 'shorter'):
            normalize_test(dict(base, monitoring={'sample_interval_s': 10, 'max_sample_gap_s': 5}))
