"""Adversarial tests over independent fixed Q36/Bridge evidence, not a live EP8 model."""
import copy
import os
from pathlib import Path
import tempfile
import unittest

import q36_ep8_independent_coverage_candidate as audit

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
FIXTURES = Path(os.environ.get('Q36_COVERAGE_FIXTURE_DIR', str(HERE)))
REGISTRY = Path(os.environ.get('Q36_EP8_BRIDGE_REGISTRY', str(PROJECT / 'work/qwen36-megatron-4gpu-integration-20261109/REAL-BRIDGE-REGISTRY-MAP.json')))


class CoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(path.is_file() for path in (FIXTURES / 'q36-mounted-config.json',
                FIXTURES / 'q36-mounted-model.safetensors.index.json',
                FIXTURES / 'q36-provider-ddp-sr-manifest.json', REGISTRY)):
            raise unittest.SkipTest('set Q36_COVERAGE_FIXTURE_DIR and Q36_EP8_BRIDGE_REGISTRY to the verified independent evidence')
        cls.expected = audit.derive_expected(
            FIXTURES / 'q36-mounted-config.json', FIXTURES / 'q36-mounted-model.safetensors.index.json',
            FIXTURES / 'q36-provider-ddp-sr-manifest.json', REGISTRY)

    def ledger(self):
        return {'schema': 'q36-ep8-live-provider-sr-v1',
                'binding_scope': 'independent-source-and-live-provider-reviewed',
                'tp': 4, 'ep': 8, 'etp': 1, 'pp': 1,
                'rank_parameters': {rank: [copy.deepcopy(row) for row in rows.values()]
                                    for rank, rows in self.expected.items()}}

    def test_complete_coverage_and_wrong_global_expert(self):
        ledger = self.ledger()
        self.assertEqual(audit.audit_ledger(ledger, self.expected)['global_expert_parameters'], 20480)
        row = ledger['rank_parameters']['7'][0]
        row['source_global_expert'] = 0
        with self.assertRaisesRegex(ValueError, 'source_global_expert'):
            audit.audit_ledger(ledger, self.expected)

    def test_common_missing_expert_does_not_self_validate(self):
        ledger = self.ledger()
        for rank in ledger['rank_parameters']:
            ledger['rank_parameters'][rank].pop()
        with self.assertRaisesRegex(ValueError, 'independent provider coverage'):
            audit.audit_ledger(ledger, self.expected)

    def test_mislabeled_or_wrong_bridge_mapping_is_rejected(self):
        ledger = self.ledger()
        row = ledger['rank_parameters']['3'][0]
        row['kind'] = 'dense_replica'
        with self.assertRaisesRegex(ValueError, 'kind'):
            audit.audit_ledger(ledger, self.expected)
        row['kind'] = 'expert'
        row['bridge_mapping']['source_names'] = ['wrong-source']
        with self.assertRaisesRegex(ValueError, 'bridge_mapping'):
            audit.audit_ledger(ledger, self.expected)

    def test_fixed_inputs_cannot_be_swapped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'evidence SHA mismatch'):
                audit.read_locked(path, 'config')


if __name__ == '__main__': unittest.main()
