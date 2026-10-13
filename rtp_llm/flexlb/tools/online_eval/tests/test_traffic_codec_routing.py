"""版本路由和精确长度在各入口的一致性。"""
import hashlib
import json
import lzma
import tempfile
import unittest
from pathlib import Path

from traffic import datasets, prefix_lineage, prefix_lineage_v3
from traffic.synthetic.derivation import derive_parameters
from traffic.codecs import decode, decode_model
from scripts.pipeline.derive_master_templates import derive
from scripts.pipeline.materialize_traffic import main as materialize_main


class CodecRoutingTest(unittest.TestCase):
    def test_detection_uses_header_not_length_or_file_name(self):
        events = [[0, 512, -1, 0], [1000, 1024, 0, 1]]
        for version, codec in ((2, prefix_lineage), (3, prefix_lineage_v3)):
            model = decode_model(codec.encode(events))
            self.assertEqual(version, model.metadata.version)
            self.assertEqual([512, 1024], [event.input_tokens for event in model.events])

    def test_unknown_and_missing_headers_are_rejected_without_fallback(self):
        for metadata in ({}, {'version': 4}, {'version': '3'}, {'version': True}):
            raw = lzma.compress(json.dumps(dict(metadata=metadata, events=[])).encode())
            with self.assertRaises(ValueError):
                decode_model(raw)

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
                profile = derive_parameters(decode_model(raw, manifest), {}, None, model_sha256=hashlib.sha256(raw).hexdigest())
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
                specification = dict(kind='trace', model='prefix_lineage', parameters=dict(path=model.name,
                                    count=manifest['count'], output_tokens=1, priority=50))
                plan = materialize(root/'plan.jsonl', specification, 'test', model.parent,
                                   max_requests=3)
                resolved = json.loads(plan.with_suffix('.manifest.json').read_text())
                self.assertEqual(str(version), resolved['source']['version'])
                self.assertNotIn('version', resolved['requested_source'])
                self.assertEqual(version, resolved['source_codec']['version'])
                pinned = dict(specification, version=str(3 if version == 2 else 2))
                with self.assertRaisesRegex(ValueError, 'version disagrees'):
                    materialize(root/'conflict.jsonl', pinned, 'test', model.parent)
                self.assertFalse((root/'conflict.jsonl').exists())
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
