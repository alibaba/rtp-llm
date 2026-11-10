"""Opt-in, lazy registration for the Q36 on-policy standalone candidate."""
import functools
import importlib
import sys


def _patch_after_import():
    import install_rtp_bridge
    original = install_rtp_bridge._after_import
    if getattr(original, '_q36_standalone', False):
        return

    @functools.wraps(original)
    def after_import(name, module):
        original(name, module)
        if name == 'verl.workers.rollout.base':
            module._ROLLOUT_REGISTRY[('rtp', 'async')] = 'standalone_rtp_adapter_v4.StandaloneRTPAdapter'
        elif name == 'verl.workers.rollout.replica':
            module.RolloutReplicaRegistry.register(
                'rtp', lambda: importlib.import_module('standalone_rtp_adapter_v4').StandaloneRTPReplica)
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
        value['worker_process_setup_hook'] = 'standalone_bootstrap_v4.worker_setup'
        value.setdefault('env_vars', {}).pop('CUDA_VISIBLE_DEVICES', None)
        return value

    runtime._q36_standalone = True
    ray_runtime_env.get_ppo_ray_runtime_env = runtime
