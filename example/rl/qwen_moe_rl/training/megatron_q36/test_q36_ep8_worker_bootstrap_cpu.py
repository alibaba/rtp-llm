"""CPU contract tests; no distributed, native or RL qualification."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import q36_ep8_worker_bootstrap_candidate as bootstrap


class Attr(dict):
    def __getattr__(self, key):
        try: return self[key]
        except KeyError as error: raise AttributeError(key) from error
    __setattr__ = dict.__setitem__


def attr(value):
    return Attr({k: attr(v) if isinstance(v, dict) else v for k, v in value.items()})


class EP8Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.manifest = self.root / 'reviewed.json'
        self.manifest.write_text(json.dumps({'schema': 'q36-ep8-live-provider-sr-v1',
            'binding_scope': 'independent-source-and-live-provider-reviewed',
            'tp': 4, 'ep': 8, 'etp': 1, 'pp': 1,
            'rank_parameters': {str(i): [] for i in range(8)}}))
        self.digest = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.env = {'Q36_EP8_SR_MANIFEST': str(self.manifest), 'Q36_EP8_SR_MANIFEST_SHA256': self.digest,
                    'Q36_DISAGG_TRAIN_GPUS': '8', 'Q36_DISAGG_ROLLOUT_GPUS': '4',
                    'RTP_EP_SIZE': '4', 'RTP_LLM_RL_Q36_EP4': '1', 'RTP_RL_SHARED_BACKING': '0'}
        self.config = attr({'trainer': {'nnodes': 2, 'n_gpus_per_node': 4}, 'actor_rollout_ref': {
            'actor': {'megatron': {'tensor_model_parallel_size': 4, 'expert_model_parallel_size': 8,
                                    'expert_tensor_parallel_size': 1}},
            'rollout': {'tensor_model_parallel_size': 4, 'data_parallel_size': 1}}})

    def test_mapper_world_vote_precedes_binder(self):
        class Flag:
            def __init__(self, value): self.value = value
            def item(self): return self.value
        torch = types.ModuleType('torch')
        torch.int32 = 'int32'
        torch.tensor = lambda value, **kw: Flag(value)
        torch.cuda = types.SimpleNamespace(current_device=lambda: 0)
        dist = types.ModuleType('torch.distributed')
        dist.ReduceOp = types.SimpleNamespace(MIN='min')
        dist.all_reduce = lambda flag, op: None
        torch.distributed = dist
        core = types.ModuleType('megatron.core')
        core.optimizer = types.ModuleType('megatron.core.optimizer')
        megatron = types.ModuleType('megatron')
        megatron.core = core
        injection = types.ModuleType('q36_nonshared_bf16_worker_candidate')
        injection.verify_candidate_sources = lambda root: None
        captured = {}
        def install_hooks(_verl, _core, _factory, _torch, _dist, mapper): captured['mapper'] = mapper
        injection.install_hooks = install_hooks
        binder = types.SimpleNamespace(bind_all_ranks=lambda *args: {'binding': args})
        modules = {'torch': torch, 'torch.distributed': dist, 'megatron': megatron,
                   'megatron.core': core, 'megatron.core.optimizer': core.optimizer,
                   'q36_nonshared_bf16_worker_candidate': injection}
        def load(name, root): return binder if name == 'q36_ep8_sr_binder_v2' else object()
        env = dict(self.env, Q36_OPT_SOURCE_DIR=str(self.root))
        with patch.dict(sys.modules, modules), patch.dict(os.environ, env), patch.object(bootstrap, '_load_verified', load), patch.object(sys, 'path', [str(self.root)]):
            bootstrap._install_ep8_optimizer_hooks(object())
            self.assertIn('binding', captured['mapper']([]))
            dist.all_reduce = lambda flag, op: setattr(flag, 'value', 0)
            binder.bind_all_ranks = lambda *args: self.fail('binder entered after peer rejected ledger')
            with self.assertRaisesRegex(RuntimeError, 'all-rank EP8'):
                captured['mapper']([])

    def test_bootstrap_rejects_late_replacement(self):
        baseline = types.SimpleNamespace(_INSTALLED=True, _install_optimizer_hooks=lambda module: None)
        with patch.dict(sys.modules, {'q36_nonshared_worker_bootstrap': baseline}):
            with self.assertRaisesRegex(RuntimeError, 'before'):
                bootstrap._select_ep8()


if __name__ == '__main__': unittest.main()
