"""Opt-in, lazy registration for the Q36 on-policy standalone candidate."""
import functools
import hashlib
import importlib
import importlib.machinery
import os
from pathlib import Path
import sys

V5_SHA256 = 'b1adfbd8a300197568af4c3b6706e35bc46f34f07b4e368571b0f697354e10e1'


def _verify_v5():
    name = 'standalone_rtp_adapter_v5'
    source = Path(os.environ['Q36_ADAPTER_DIR']).resolve(strict=True) / (name + '.py')
    spec = importlib.machinery.PathFinder.find_spec(name, sys.path)
    loaded = sys.modules.get(name)
    if (source.is_symlink() or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != V5_SHA256
        or spec is None or not spec.origin or Path(spec.origin).resolve() != source
        or loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != source):
        raise RuntimeError('selected v5 standalone adapter source identity mismatch')


def _patch_after_import():
    import install_rtp_bridge
    original = install_rtp_bridge._after_import
    if getattr(original, '_q36_standalone', False):
        return

    @functools.wraps(original)
    def after_import(name, module):
        original(name, module)
        if name == 'verl.workers.rollout.base':
            module._ROLLOUT_REGISTRY[('rtp', 'async')] = 'standalone_rtp_adapter_v5.StandaloneRTPAdapter'
        elif name == 'verl.workers.rollout.replica':
            module.RolloutReplicaRegistry.register(
                'rtp', lambda: importlib.import_module('standalone_rtp_adapter_v5').StandaloneRTPReplica)
        elif name == 'rllm.trainer.verl.verl_backend':
            import onpolicy_standalone_candidate
            onpolicy_standalone_candidate.install(module)

    after_import._q36_standalone = True
    install_rtp_bridge._after_import = after_import
    for name in ('verl.workers.rollout.base', 'verl.workers.rollout.replica',
                 'rllm.trainer.verl.verl_backend'):
        if name in sys.modules:
            after_import(name, sys.modules[name])


def worker_setup():
    # Original hooks remain passive: no Torch import before Ray assigns a GPU.
    _verify_v5()
    import q36_nonshared_worker_bootstrap
    q36_nonshared_worker_bootstrap.worker_setup()
    _patch_after_import()


def install_driver():
    worker_setup()
    import install_rtp_bridge
    install_rtp_bridge.install()
    from rllm.trainer.verl import ray_runtime_env
    import q36_nonshared_worker_bootstrap
    q36_nonshared_worker_bootstrap.install_driver_hook(ray_runtime_env)
    original = ray_runtime_env.get_ppo_ray_runtime_env
    if getattr(original, '_q36_standalone', False): return

    @functools.wraps(original)
    def runtime():
        value = original()
        if value.get('worker_process_setup_hook') != 'q36_nonshared_worker_bootstrap.worker_setup':
            raise RuntimeError('unexpected worker bootstrap chain')
        value['worker_process_setup_hook'] = 'standalone_bootstrap_v5.worker_setup'
        value.setdefault('env_vars', {}).pop('CUDA_VISIBLE_DEVICES', None)
        return value

    runtime._q36_standalone = True
    ray_runtime_env.get_ppo_ray_runtime_env = runtime
