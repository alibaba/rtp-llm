"""版本路由和精确长度在各入口的一致性。"""
import json
import tempfile
import unittest
from pathlib import Path

from traffic import datasets, prefix_lineage, prefix_lineage_v3
from traffic.derive_synthetic_parameters import derive_parameters
from traffic.codecs import decode
from scripts.pipeline.derive_master_templates import derive
from scripts.pipeline.materialize_traffic import main as materialize_main


class CodecRoutingTest(unittest.TestCase):
    def test_both_versions_use_file_admission_calibration_templates_and_materialization(self):
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
                from traffic.traffic_source import materialize
                specification = dict(kind='trace', model='prefix_lineage', version=str(version),
                    parameters=dict(path=model.name, sha256=manifest['sha256'],
                                    count=manifest['count'], output_tokens=1, priority=50))
                plan = materialize(root/'plan.jsonl', specification, 'test', model.parent,
                                   max_requests=3)
                self.assertEqual([row['il'] for row in templates],
                    [json.loads(line)['il'] for line in plan.read_text().splitlines()])
                for wrong in (None, 99, 2 if version == 3 else 3):
                    bad = dict(manifest, codec=dict(name='prefix_lineage', version=wrong))
                    sidecar.write_text(json.dumps(bad))
                    with self.assertRaises((ValueError, UnicodeDecodeError)):
                        datasets.read_manifest(model)
                sidecar.write_text(json.dumps(dict(manifest, block_size=1024)))
                with self.assertRaisesRegex(ValueError, 'block_size'):
                    datasets.read_manifest(model)
