"""Verify v4 registration and Ray setup-hook chain without Ray startup."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


class BootstrapTest(unittest.TestCase):
    def test_v4_adapter_replica_and_worker_hook_are_selected(self):
        bridge = types.ModuleType('install_rtp_bridge')
        bridge._after_import = lambda name, module: None
        bridge.install = lambda: None
        base = types.ModuleType('q36_nonshared_worker_bootstrap')
        base.worker_setup = lambda: None
        runtime = types.ModuleType('rllm.trainer.verl.ray_runtime_env')
        def install_driver_hook(target):
            target.get_ppo_ray_runtime_env = lambda: {
                'worker_process_setup_hook': 'q36_nonshared_worker_bootstrap.worker_setup',
                'env_vars': {'CUDA_VISIBLE_DEVICES': ''}}
        base.install_driver_hook = install_driver_hook
        pkg = types.ModuleType('rllm.trainer.verl')
        pkg.ray_runtime_env = runtime
        replica = types.ModuleType('verl.workers.rollout.replica')
        choices = {}
        replica.RolloutReplicaRegistry = types.SimpleNamespace(register=lambda name, loader: choices.update({name: loader}))
        base_rollout = types.ModuleType('verl.workers.rollout.base')
        base_rollout._ROLLOUT_REGISTRY = {}
        target = types.ModuleType('standalone_rtp_adapter_v4')
        target.StandaloneRTPReplica = type('RealReplica', (), {})
        modules = {'install_rtp_bridge': bridge, 'q36_nonshared_worker_bootstrap': base,
                   'rllm.trainer.verl': pkg, 'rllm.trainer.verl.ray_runtime_env': runtime,
                   'standalone_rtp_adapter_v4': target}
        with patch.dict(sys.modules, modules):
            path = HERE / 'standalone_bootstrap_v4.py'
            spec = importlib.util.spec_from_file_location('q36_bootstrap_test', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.install_driver()
            bridge._after_import('verl.workers.rollout.base', base_rollout)
            bridge._after_import('verl.workers.rollout.replica', replica)
            self.assertEqual(base_rollout._ROLLOUT_REGISTRY[('rtp', 'async')],
                             'standalone_rtp_adapter_v4.StandaloneRTPAdapter')
            self.assertIs(choices['rtp'](), target.StandaloneRTPReplica)
            result = runtime.get_ppo_ray_runtime_env()
            self.assertEqual(result['worker_process_setup_hook'], 'standalone_bootstrap_v4.worker_setup')
            self.assertNotIn('CUDA_VISIBLE_DEVICES', result['env_vars'])


if __name__ == '__main__': unittest.main()
