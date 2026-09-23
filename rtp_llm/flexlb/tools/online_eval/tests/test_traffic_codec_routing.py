"""版本路由和精确长度在各入口的一致性。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from traffic import datasets, prefix_lineage, prefix_lineage_v3
from traffic.derive_synthetic_parameters import derive_parameters
from traffic.codecs import decode
from traffic.playback_config import comparison_notice
from scripts.pipeline.derive_master_templates import derive
from scripts.pipeline.materialize_traffic import main as materialize_main


class CodecRoutingTest(unittest.TestCase):
    def test_both_versions_use_file_admission_calibration_templates_and_stress(self):
        for version, codec in ((2, prefix_lineage), (3, prefix_lineage_v3)):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = root/'traffic_trace/fixture.xz'
                model.parent.mkdir()
                raw = codec.encode([[0, 513, -1, 0], [1000, 1025, 0, 1], [2000, 1, -1, 0]])
                model.write_bytes(raw)
                manifest = datasets.build_manifest(model)
                sidecar = model.with_suffix('.manifest.json')
                sidecar.write_text(json.dumps(manifest))
                self.assertEqual(manifest, datasets.read_manifest(model))
                profile = derive_parameters(raw, {}, None, manifest)
                self.assertEqual((1539 if version == 3 else 2048) / 3,
                                 profile['calibration']['targets']['mean_input_tokens'])
                templates = derive(model, count=3)['templates']
                self.assertEqual([513, 1025, 1] if version == 3 else [512, 1024, 512],
                                 [row['il'] for row in templates])
                if version == 3:
                    self.assertEqual(templates[0]['labels'][0], templates[1]['labels'][0])
                    self.assertNotEqual(templates[0]['labels'][1], templates[1]['labels'][1])
                materialize_main(['--lineage-model', str(model), '--out', str(root/'cli.jsonl'), '--namespace', 'test'])
                from runtime import stress
                with mock.patch.object(datasets, 'DATA', root):
                    args = stress.parse_args(['--dry-run', '--traffic-model', 'fixture', '--limit', '3'])
                    stress._traffic(args, root/'stress.jsonl')
                    self.assertEqual([row['il'] for row in templates],
                        [json.loads(line)['il'] for line in (root/'stress.jsonl').read_text().splitlines()])
                for wrong in (None, 99, 2 if version == 3 else 3):
                    bad = dict(manifest, codec=dict(name='prefix_lineage', version=wrong))
                    sidecar.write_text(json.dumps(bad))
                    with self.assertRaises((ValueError, UnicodeDecodeError)):
                        datasets.read_manifest(model)
                sidecar.write_text(json.dumps(dict(manifest, block_size=1024)))
                with self.assertRaisesRegex(ValueError, 'block_size'):
                    datasets.read_manifest(model)

    def test_codec_difference_is_comparison_notice_even_with_same_tail(self):
        a = dict(source=dict(kind='trace', model='prefix_lineage', version='2'), tail='same')
        b = dict(source=dict(kind='trace', model='prefix_lineage', version='3'), tail='same')
        self.assertIn('trace codec version', comparison_notice(a, b))
        self.assertIn('TPS', comparison_notice(a, b))
        self.assertIsNone(comparison_notice(a, a))
