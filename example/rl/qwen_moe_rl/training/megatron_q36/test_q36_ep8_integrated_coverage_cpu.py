"""Adversarial integration test: only fixed external evidence supplies expectations."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import q36_ep8_independent_coverage_candidate as independent
import q36_ep8_launch_contract_v2_candidate as contract

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
FIXTURES = Path(os.environ.get('Q36_COVERAGE_FIXTURE_DIR', str(HERE)))
REGISTRY = Path(os.environ.get('Q36_EP8_BRIDGE_REGISTRY', str(PROJECT / 'work/qwen36-megatron-4gpu-integration-20261109/REAL-BRIDGE-REGISTRY-MAP.json')))


class Attr(dict):
    def __getattr__(self, name):
        try: return self[name]
        except KeyError as error: raise AttributeError(name) from error
    __setattr__ = dict.__setitem__


def attr(value):
    return Attr({k: attr(v) if isinstance(v, dict) else v for k, v in value.items()})


class IntegratedCoverage(unittest.TestCase):
    def test_valid_synthetic_and_rehashed_missing_expert(self):
        if not all(path.is_file() for path in (FIXTURES / 'q36-mounted-config.json',
                FIXTURES / 'q36-mounted-model.safetensors.index.json',
                FIXTURES / 'q36-provider-ddp-sr-manifest.json', REGISTRY)):
            self.skipTest('set Q36_COVERAGE_FIXTURE_DIR and Q36_EP8_BRIDGE_REGISTRY to the verified independent evidence')
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder).resolve()
            shutil.copyfile(FIXTURES / 'q36-mounted-config.json', model / 'config.json')
            shutil.copyfile(FIXTURES / 'q36-mounted-model.safetensors.index.json', model / 'model.safetensors.index.json')
            provider = FIXTURES / 'q36-provider-ddp-sr-manifest.json'
            bridge = REGISTRY
            expected = independent.derive_expected(model / 'config.json', model / 'model.safetensors.index.json', provider, bridge)
            ledger = {'schema': 'q36-ep8-live-provider-sr-v1',
                      'binding_scope': 'independent-source-and-live-provider-reviewed',
                      'tp': 4, 'ep': 8, 'etp': 1, 'pp': 1,
                      'rank_parameters': {rank: [copy.deepcopy(row) for row in rows.values()]
                                          for rank, rows in expected.items()}}
            path = model / 'reviewed.json'
            def save():
                path.write_text(json.dumps(ledger))
                return hashlib.sha256(path.read_bytes()).hexdigest()
            env = {'Q36_EP8_SR_MANIFEST': str(path), 'Q36_EP8_SR_MANIFEST_SHA256': save(),
                   'Q36_DISAGG_TRAIN_GPUS': '8', 'Q36_DISAGG_ROLLOUT_GPUS': '4',
                   'RTP_EP_SIZE': '4', 'RTP_LLM_RL_Q36_EP4': '1', 'RTP_RL_SHARED_BACKING': '0',
                   'Q36_CODE_DIR': str(HERE), 'Q36_MODEL_PATH': str(model), 'Q36_SR_MANIFEST': str(provider),
                   'Q36_EP8_BRIDGE_REGISTRY': str(bridge)}
            config = attr({'trainer': {'nnodes': 2, 'n_gpus_per_node': 4}, 'actor_rollout_ref': {
                'actor': {'megatron': {'tensor_model_parallel_size': 4, 'expert_model_parallel_size': 8,
                                        'expert_tensor_parallel_size': 1}},
                'rollout': {'tensor_model_parallel_size': 4, 'data_parallel_size': 1}}})
            baseline = types.ModuleType('q36_launch_contract')
            def get(config, *keys):
                for key in keys: config = config[key]
                return config
            baseline.get = get
            def validate_base(copied):
                self.assertEqual(copied.trainer.nnodes, 1)
                self.assertEqual(copied.actor_rollout_ref.actor.megatron.expert_model_parallel_size, 1)
                self.assertEqual(copied.actor_rollout_ref.actor.megatron.expert_tensor_parallel_size, 4)
            baseline.validate = validate_base
            with patch.dict(sys.modules, {'q36_launch_contract': baseline}), patch.dict(os.environ, env):
                result = contract.validate(config)
                self.assertEqual(result['independent_expert_coverage']['global_expert_parameters'], 20480)
                self.assertEqual(config.trainer.nnodes, 2)
                self.assertEqual(config.actor_rollout_ref.actor.megatron.expert_model_parallel_size, 8)
                ledger['rank_parameters']['0'].pop()
                os.environ['Q36_EP8_SR_MANIFEST_SHA256'] = save()
                with self.assertRaisesRegex(ValueError, 'independent provider coverage'):
                    contract.validate(config)


if __name__ == '__main__': unittest.main()
