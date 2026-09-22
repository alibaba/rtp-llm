import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from traffic.prefix_lineage_v3 import encode, decode
from traffic.traffic_source import materialize


class ExactLineageTest(unittest.TestCase):
    def test_filter_preserves_exact_boundary_and_excluded_parent_prefix(self):
        events = [[0, 32769, -1, 0], [1, 32768, 0, 64], [2, 513, 1, 1], [3, 1, -1, 0]]
        raw = encode(events)
        self.assertEqual(decode(raw)[1], events)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'model.xz').write_bytes(raw)
            parameters = dict(path='model.xz', sha256=hashlib.sha256(raw).hexdigest(),
                              count=4, output_tokens=350, priority=50)
            def project(name, params, limit=None):
                path = materialize(root/name, dict(kind='trace', model='prefix_lineage',
                    version='3', parameters=params), 'test', root, max_requests=limit)
                return [json.loads(line) for line in path.read_text().splitlines()]
            full = project('all.jsonl', parameters)
            filtered = project('filtered.jsonl', dict(parameters, max_input_tokens=32768))
            self.assertEqual(filtered, full[1:])
            self.assertEqual([r['il'] for r in filtered], [32768, 513, 1])
            self.assertEqual(filtered[0]['input_token_blocks'], full[0]['input_token_blocks'][:64])
            self.assertEqual(project('limited.jsonl', dict(parameters, max_input_tokens=32768), 1), full[1:2])

    def test_partial_parent_block_cannot_be_reused(self):
        with self.assertRaisesRegex(ValueError, 'shared prefix'):
            encode([[0, 513, -1, 0], [1, 1024, 0, 2]])
