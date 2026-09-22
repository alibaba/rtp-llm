import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from traffic.prefix_lineage_v3 import encode, decode, output_sampler, write_trace
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

    def test_sampled_outputs_are_paired_across_namespaces_and_filtering(self):
        events = [[i, 32769 if i % 3 == 0 else 513, -1, 0] for i in range(2000)]
        raw = encode(events)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'model.xz').write_bytes(raw)
            params = dict(path='model.xz', sha256=hashlib.sha256(raw).hexdigest(),
                count=len(events), output_tokens=8192, priority=50,
                output_distribution=dict(kind='geometric', mean_tokens=400, seed=20260922))
            def project(name, parameters, limit=None):
                file = root/name
                provenance = write_trace(file, parameters, name, root, max_requests=limit)
                self.assertEqual(provenance['output_semantics'], 'EXPLICIT_GEOMETRIC_SPLITMIX64_EVENT_INDEX_V1')
                return {int(r['rid'].split(':')[-1]): r['ol']
                        for r in map(json.loads, file.read_text().splitlines())}
            full = project('a', params)
            filtered = project('b', dict(params, max_input_tokens=32768), 100)
            self.assertEqual(filtered, {i: full[i] for i in filtered})
            self.assertEqual(len(filtered), 100)
            self.assertGreater(len(set(full.values())), 100)
            self.assertTrue(all(1 <= v <= 8192 for v in full.values()))
            self.assertTrue(360 < sum(full.values()) / len(full) < 440)

    def test_output_distribution_contract(self):
        policy = dict(kind='geometric', mean_tokens=400, seed=20260922)
        for invalid in [dict(policy, mean_tokens=0), dict(policy, mean_tokens=float('nan')),
                        dict(policy, seed=True), dict(policy, kind='unknown'), dict(policy, extra=1)]:
            with self.assertRaises(ValueError):
                output_sampler(dict(output_tokens=20, output_distribution=invalid))
        sample = output_sampler(dict(output_tokens=20, output_distribution=policy))
        self.assertTrue(all(1 <= sample(i) <= 20 for i in range(100)))
        self.assertEqual(output_sampler(dict(output_tokens=20))(3), 20)
        self.assertEqual(output_sampler(dict(output_tokens=20,
            output_distribution=dict(policy, mean_tokens=1)))(3), 1)
