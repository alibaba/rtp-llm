"""Decoded lineage projection has the same content rules for both codec versions."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from traffic import datasets
from traffic.codecs import decode
from traffic.prefix_lineage import encode as encode_v2
from traffic.prefix_lineage_v3 import encode as encode_v3
from traffic.traffic_source import materialize


class LineageTransformsTest(unittest.TestCase):
    def _project(self, root, raw, version, *, cap=512, limit=None):
        model = root / f'v{version}.xz'
        model.write_bytes(raw)
        spec = dict(kind='trace', model='prefix_lineage', version=str(version),
            parameters=dict(path=model.name, sha256=hashlib.sha256(raw).hexdigest(),
                count=2, output_tokens=9, priority=50, max_input_tokens=cap))
        output = root / f'v{version}.jsonl'
        materialize(output, spec, 'sample', root, max_requests=limit)
        return ([json.loads(line) for line in output.read_text().splitlines()],
            json.loads(output.with_suffix('.manifest.json').read_text()))

    def test_excluded_parent_keeps_child_prefix_identity_in_both_codecs(self):
        events = [[0, 2048, -1, 0], [1, 512, 0, 1]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for version, encode in ((2, encode_v2), (3, encode_v3)):
                with self.subTest(version=version):
                    rows, manifest = self._project(root, encode(events), version)
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]['input_token_blocks'], [1])
                    self.assertEqual(rows[0]['il'], 512)
                    self.assertEqual(manifest['length_filter'], dict(
                        max_input_tokens=512, selected=1, excluded_before_limit=1))
                    self.assertEqual(manifest['transformations']['applied'][0]['kind'],
                        'input_length_filter')

    def test_bundled_v2_accepts_32k_filter_through_public_materialize(self):
        model = datasets.model_path()
        manifest = datasets.read_manifest(model)
        self.assertEqual(manifest['codec']['version'], 2)
        _, events = decode(model.read_bytes(), manifest)
        expected_excluded = 0
        selected = 0
        for event in events:
            if event[1] * 512 > 32768:
                expected_excluded += 1
            else:
                selected += 1
                if selected == 5:
                    break
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = dict(kind='trace', model='prefix_lineage', version='2',
                parameters=dict(path=str(model), sha256=manifest['sha256'],
                    count=manifest['count'], output_tokens=8, priority=50,
                    max_input_tokens=32768))
            output = root / 'filtered.jsonl'
            materialize(output, spec, 'bundled', root, max_requests=5)
            result = json.loads(output.with_suffix('.manifest.json').read_text())
            self.assertEqual(result['length_filter']['selected'], 5)
            self.assertEqual(result['length_filter']['excluded_before_limit'], expected_excluded)
            self.assertTrue(all(json.loads(line)['il'] <= 32768
                for line in output.read_text().splitlines()))

    def test_invalid_filter_fails_loud(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for version, encode in ((2, encode_v2), (3, encode_v3)):
                with self.subTest(version=version):
                    with self.assertRaisesRegex(ValueError, 'invalid input length filter'):
                        self._project(root, encode([[0, 512, -1, 0], [1, 512, 0, 1]]),
                            version, cap=0)
