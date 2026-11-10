"""CPU-only selector and hook-chain qualification; no Ray or GPU."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO_TRAIN = Path(os.environ.get('Q36_TRAIN_SOURCE_DIR', str(HERE))).resolve()


def load(name):
    spec = importlib.util.spec_from_file_location('test_' + name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SelectorTest(unittest.TestCase):
    def test_profile_choice_and_locked_v4_source(self):
        selector = load('q36_standalone_bootstrap_selector')
        v3 = types.ModuleType('standalone_bootstrap_v3')
        v4 = types.ModuleType('standalone_bootstrap_v4')
        v4.__file__ = str(REPO_TRAIN / 'standalone_bootstrap_v4.py')
        with patch.dict(sys.modules, {'standalone_bootstrap_v3': v3, 'standalone_bootstrap_v4': v4}), \
             patch.object(sys, 'path', [str(REPO_TRAIN)] + sys.path), \
             patch.dict(os.environ, {'Q36_CODE_DIR': str(REPO_TRAIN), 'Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT': '0'}):
            self.assertIs(selector.select_bootstrap(), v3)
            os.environ['Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT'] = '1'
            self.assertIs(selector.select_bootstrap(), v4)
            with patch.object(selector, 'V4_SHA256', '0' * 64), self.assertRaisesRegex(RuntimeError, 'source identity'):
                selector.select_bootstrap()
            os.environ['Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT'] = 'yes'
            with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                selector.select_bootstrap()

    def test_4plus4_calls_selector_not_an_implicit_v4(self):
        source = (HERE / 'run_standalone_q36_candidate_v3.py').read_text()
        main = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        text = ast.get_source_segment(source, main)
        self.assertIn('select_bootstrap().install_driver()', text)
        self.assertNotIn('standalone_bootstrap_v4.install_driver()', text)

    def test_ep8_worker_and_driver_use_same_selected_hook(self):
        module = load('q36_ep8_worker_bootstrap_candidate')
        for chosen in ('standalone_bootstrap_v3', 'standalone_bootstrap_v4'):
            with self.subTest(chosen=chosen):
                events = []
                selected = types.ModuleType(chosen)
                selected.worker_setup = lambda: events.append('worker')
                runtime = types.ModuleType('rllm.trainer.verl.ray_runtime_env')
                selected.install_driver = lambda: setattr(runtime, 'get_ppo_ray_runtime_env', lambda: {
                    'worker_process_setup_hook': chosen + '.worker_setup', 'env_vars': {'CUDA_VISIBLE_DEVICES': ''}})
                selector = types.ModuleType('q36_standalone_bootstrap_selector')
                selector.select_bootstrap = lambda: selected
                pkg = types.ModuleType('rllm.trainer.verl')
                pkg.ray_runtime_env = runtime
                with patch.object(module, '_select_ep8', lambda: None), patch.dict(sys.modules, {
                    'q36_standalone_bootstrap_selector': selector, 'rllm.trainer.verl': pkg,
                    'rllm.trainer.verl.ray_runtime_env': runtime}):
                    module.worker_setup()
                    module.install_driver()
                    result = runtime.get_ppo_ray_runtime_env()
                    self.assertEqual(events, ['worker'])
                    self.assertEqual(result['worker_process_setup_hook'],
                                     'q36_ep8_worker_bootstrap_candidate.worker_setup')
                    self.assertNotIn('CUDA_VISIBLE_DEVICES', result['env_vars'])


if __name__ == '__main__': unittest.main()
