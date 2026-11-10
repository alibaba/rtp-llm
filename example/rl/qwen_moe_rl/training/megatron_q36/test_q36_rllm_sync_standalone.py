import os
import sys
import types
import unittest
from unittest.mock import patch

import q36_rllm_sync_compat as compat


def config(backend='rtp_nixl_guarded', module='verl_nixl_guarded', async_enabled=False):
    ckpt = types.SimpleNamespace(backend=backend, custom_backend_module=module)
    return types.SimpleNamespace(
        rllm={'async_training': {'enable': async_enabled}},
        actor_rollout_ref=types.SimpleNamespace(rollout=types.SimpleNamespace(checkpoint_engine=ckpt),
                                                hybrid_engine=False))


class SyncTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.utils = types.ModuleType('rllm.trainer.verl.utils')
        self.utils._SHARED_KEYS = [compat.OLD]
        def original(cfg, *args, **kwargs):
            self.calls.append((args, kwargs))
            cfg.actor_rollout_ref.hybrid_engine = True
            cfg.actor_rollout_ref.rollout.checkpoint_engine.backend = 'naive'
            return 'original result'
        self.utils.sync_config = original
        modules = {name: types.ModuleType(name) for name in ('rllm', 'rllm.trainer', 'rllm.trainer.verl')}
        modules['rllm.trainer.verl.utils'] = self.utils
        modules['rllm'].trainer = modules['rllm.trainer']
        modules['rllm.trainer'].verl = modules['rllm.trainer.verl']
        modules['rllm.trainer.verl'].utils = self.utils
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        compat.install()
        compat.install()

    def test_opt_in_preserves_backend_and_all_original_side_effects(self):
        c = config()
        with patch.dict(os.environ, {'Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT':'1',
                                    'Q36_STRICT_4PLUS4_CLAIM':'1', 'Q36_DISAGG_RESIDENT_KV':'1'}, clear=True):
            self.assertEqual(self.utils.sync_config(c, hydra_overrides=['a=b']), 'original result')
        self.assertEqual(self.calls, [((), {'hydra_overrides':['a=b']})])
        self.assertEqual(c.actor_rollout_ref.rollout.checkpoint_engine.backend, 'rtp_nixl_guarded')
        self.assertIs(c.actor_rollout_ref.hybrid_engine, True)
        self.assertEqual(self.utils._SHARED_KEYS, [compat.NEW])

    def test_default_remains_original(self):
        c = config()
        with patch.dict(os.environ, {}, clear=True): self.utils.sync_config(c)
        self.assertEqual(c.actor_rollout_ref.rollout.checkpoint_engine.backend, 'naive')

    def test_unknown_router_mapping_fails_closed(self):
        self.utils._SHARED_KEYS = [('unexpected', 'mapping')]
        with self.assertRaisesRegex(RuntimeError, 'router-replay'):
            compat.install()

    def test_bad_profile_fails_before_original(self):
        env = {'Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT':'1','Q36_STRICT_4PLUS4_CLAIM':'1',
               'Q36_DISAGG_RESIDENT_KV':'1'}
        with patch.dict(os.environ, env, clear=True):
            for c in (config(backend='naive'),config(module='unexpected'),config(async_enabled=True)):
                with self.assertRaises(ValueError): self.utils.sync_config(c)
        with patch.dict(os.environ, {'Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT':'1'}, clear=True):
            with self.assertRaises(ValueError): self.utils.sync_config(config())
        self.assertFalse(self.calls)

    def test_unexpected_original_behavior_fails_closed(self):
        def bad(c): c.actor_rollout_ref.hybrid_engine = False
        self.utils.sync_config = bad
        compat.install()
        with patch.dict(os.environ, {'Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT':'1',
                                    'Q36_STRICT_4PLUS4_CLAIM':'1', 'Q36_DISAGG_RESIDENT_KV':'1'}, clear=True):
            with self.assertRaises(RuntimeError): self.utils.sync_config(config())

    def test_original_exception_is_preserved(self):
        err = RuntimeError('pinned conversion failed')
        def bad(c): raise err
        self.utils.sync_config = bad
        compat.install()
        with patch.dict(os.environ, {'Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT':'1',
                                    'Q36_STRICT_4PLUS4_CLAIM':'1', 'Q36_DISAGG_RESIDENT_KV':'1'}, clear=True):
            with self.assertRaises(RuntimeError) as result: self.utils.sync_config(config())
        self.assertIs(result.exception, err)


if __name__ == '__main__': unittest.main()
