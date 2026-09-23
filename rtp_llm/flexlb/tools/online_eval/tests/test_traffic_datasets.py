"""Capture integrity and selection without an admission registry."""
import hashlib
import json
import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from traffic import datasets
from traffic.prefix_lineage import encode

ROOT = Path(__file__).resolve().parents[1]


class TrafficDatasetsTest(unittest.TestCase):
    def test_bundled_captures_have_reproducible_statistics(self):
        for name, path in datasets.trace_models().items():
            with self.subTest(name=name):
                manifest = datasets.read_manifest(path)
                self.assertEqual(datasets.build_manifest(path, manifest['source']), manifest)
                self.assertEqual('real', manifest['data_kind'])
                self.assertIn('spectrum', manifest['source'])
                self.assertIn('model', manifest['source'])
        path = datasets.model_path()
        digest = datasets.read_manifest(path)['sha256']
        fixture = json.loads(path.with_suffix('.templates.json').read_text())
        profile = json.loads(datasets.profile_path().read_text())
        self.assertEqual(digest, fixture['source_sha256'])
        self.assertEqual(digest, profile['calibration']['model_sha256'])
        self.assertEqual(path, (datasets.profile_path().parent / profile['source_capture']).resolve())
        self.assertEqual('synthetic', profile['data_kind'])

    def test_observed_statistics_and_manual_source_preservation(self):
        from scripts.pipeline.describe_traffic import main
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'example.xz'
            model.write_bytes(encode([[0, 1024, -1, 0], [1000, 1536, 0, 1],
                                      [3000, 512, -1, 0]]))
            manifest = datasets.build_manifest(model)
            stats = manifest['statistics']
            self.assertEqual(1, stats['arrival']['mean_qps'])
            self.assertEqual(0, stats['arrival']['requests_per_second']['min'])
            self.assertEqual(4, stats['arrival']['requests_per_second']['count'])
            self.assertEqual(1024, stats['input_tokens']['p50'])
            self.assertEqual(3072, stats['input_tokens']['total'])
            self.assertAlmostEqual(1 / 6, stats['prefix_structure']['token_weighted_shared_fraction'])
            self.assertEqual(2, stats['prefix_structure']['parentless_requests'])
            self.assertIsNone(stats['output_tokens'])
            self.assertEqual('unconfirmed', manifest['source']['model']['status'])
            manifest['source']['model'] = dict(name='test-model', status='confirmed', evidence='test fixture')
            sidecar = model.with_suffix('.manifest.json')
            sidecar.write_text(json.dumps(manifest))
            main([str(model)])
            self.assertEqual(manifest['source'], datasets.read_manifest(model)['source'])
            sidecar.unlink()
            with self.assertRaisesRegex(ValueError, 'missing manifest'):
                datasets.read_manifest(model)
            sidecar.write_text(json.dumps(manifest))
            model.write_bytes(model.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256 disagree'):
                datasets.read_manifest(model)

    def test_new_file_is_selectable_without_registration(self):
        from runtime import stress
        from traffic.traffic_source import validate_plan
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            model = data / 'traffic_models/custom_capture.xz'
            model.parent.mkdir()
            model.write_bytes(encode([[0, 512, -1, 0], [1000, 512, 0, 1]]))
            model.with_suffix('.manifest.json').write_text(json.dumps(datasets.build_manifest(model)))
            with mock.patch.object(datasets, 'DATA', data):
                args = stress.parse_args(['--dry-run', '--traffic-model', 'custom_capture', '--limit', '1'])
                out = data / 'plan.jsonl'
                stress._traffic(args, out)
                self.assertEqual(1, validate_plan(out))
                profile = data / 'calibration/custom_shape.profile.json'
                profile.parent.mkdir()
                profile.write_text(json.dumps(dict(parameters=dict(families=2, prefix_blocks=2,
                    zipf_alpha=1, cold_fraction=0), calibration=dict(origin='unit-test'))))
                from traffic.realistic import resolve
                params, provenance = resolve(dict(profile='custom_shape', seed=1, count=2, output_tokens=1))
                self.assertEqual(2, params['families'])
                self.assertEqual('unit-test', provenance['origin'])

    def test_existing_scenarios_pin_matching_files(self):
        from scenario.loader import load_document

        def sources(node):
            if isinstance(node, dict):
                if node.get('kind') == 'trace' and isinstance(node.get('parameters'), dict):
                    yield node['parameters']
                for value in node.values():
                    yield from sources(value)
            elif isinstance(node, list):
                for value in node:
                    yield from sources(value)

        # 入库身份来自 Git，目录位置不意味着所有外部依赖也已入库。
        tracked = {(ROOT / name).resolve() for name in subprocess.check_output(
            ['git', 'ls-files', '--', 'data/traffic_models'], cwd=ROOT, text=True).splitlines()}
        for case in (ROOT / 'config/scenarios').glob('*.yaml'):
            for params in sources(load_document(case)):
                path = (case.parent / params['path']).resolve()
                with self.subTest(case=case.name, source=params['path']):
                    self.assertRegex(params['sha256'], r'^[0-9a-f]{64}$')
                    self.assertIs(type(params['count']), int)
                    self.assertGreater(params['count'], 0)
                    if path not in tracked and not path.is_file():
                        self.skipTest(f'external capture unavailable: {path}; runtime SHA check remains required')
                    manifest = datasets.read_manifest(path)
                    self.assertEqual(manifest['sha256'], params['sha256'], str(case))
                    self.assertEqual(manifest['count'], params['count'], str(case))

    def test_default_profile_preserves_fixed_seed_trace(self):
        from traffic.realistic import write_trace
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixed.jsonl'
            semantics = write_trace(path, {'seed': 12345, 'count': 10000,
                                           'output_tokens': 420}, 'fixed')
            self.assertEqual('3ab3bb294b96acdcf188ff08897a1b8e41537903c4a56d6570a594c42c415d69',
                             hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(datasets.read_manifest(datasets.model_path())['provenance'],
                             semantics['calibration']['provenance'])

    def test_bundled_models_are_selectable_and_materialize(self):
        from runtime import stress
        from traffic.traffic_source import validate_plan
        with tempfile.TemporaryDirectory() as directory:
            for name in datasets.trace_models():
                with self.subTest(name=name):
                    args = stress.parse_args(['--dry-run', '--traffic-model', name, '--limit', '8'])
                    output = Path(directory) / f'{name}.jsonl'
                    stress._traffic(args, output)
                    self.assertEqual(8, validate_plan(output))


if __name__ == '__main__':
    unittest.main()
