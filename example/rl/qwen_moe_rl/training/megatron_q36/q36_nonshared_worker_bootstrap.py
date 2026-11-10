"""Ray worker bootstrap candidate for normal copying RTP and BF16 Megatron.

The live fixed-TP4 SR mapper is bound at optimizer construction; importing
this module does not start a native service or training.
"""
import functools
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import sys

_INSTALLED = False


def _locked_import(name, digest):
    root = Path(os.environ['Q36_CODE_DIR']).resolve(strict=True)
    path = root / (name + '.py')
    spec = importlib.machinery.PathFinder.find_spec(name, sys.path)
    loaded = sys.modules.get(name)
    if (path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest or
            spec is None or not spec.origin or Path(spec.origin).resolve() != path or
            loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != path):
        raise RuntimeError('qualified worker guard source mismatch: ' + name)
    return importlib.import_module(name)


def _install_optimizer_hooks(module):
    import megatron.core.optimizer as core_optim
    import torch
    import torch.distributed as dist
    import q36_nonshared_bf16_worker_candidate as injection
    candidate_dir = os.environ.get('Q36_OPT_SOURCE_DIR')
    if not candidate_dir:
        raise RuntimeError('Q36_OPT_SOURCE_DIR must point to verified private optimizer candidates')
    injection.verify_candidate_sources(candidate_dir)
    root = Path(candidate_dir).resolve()
    names = {Path(name).stem for name in injection.SOURCE_HASHES}
    def check_loaded():
        for name in names:
            loaded = sys.modules.get(name)
            if loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != root / (name + '.py'):
                raise RuntimeError(f'optimizer candidate was imported from an unverified location: {name}')
    check_loaded()
    if str(root) not in sys.path: sys.path.insert(0, str(root))
    factory = importlib.import_module('bf16_megatron_factory_fixed_tp4_candidate')
    def map_live_model(chunks):
        # The real mapper is deliberately late-bound at optimizer creation,
        # because only then do worker-local model parameter IDs exist.
        mapper = importlib.import_module('q36_fixed_tp4_sr_binder_ddp_candidate')
        # Let the binder perform its all-rank rejection, including for missing
        # values; a local raise here could strand peers inside its collective.
        manifest = os.environ.get('Q36_SR_MANIFEST', '')
        digest = os.environ.get('Q36_SR_MANIFEST_SHA256', '')
        return mapper.bind_all_ranks(chunks, manifest, digest)
    injection.install_hooks(module, core_optim, factory, torch, dist, map_live_model)
    check_loaded()


def worker_setup():
    # CPU Ray workers can create more workers. Do not forward their empty or
    # local CUDA mask into those later GPU actors; Ray owns each actor mask.
    excluded = [x for x in os.environ.get('RLLM_EXCLUDE', '').split(',') if x]
    if 'CUDA_VISIBLE_DEVICES' not in excluded:
        excluded.append('CUDA_VISIBLE_DEVICES')
    os.environ['RLLM_EXCLUDE'] = ','.join(excluded)
    import full_tensor_refit
    full_tensor_refit.worker_setup()
    global _INSTALLED
    if _INSTALLED: return
    _INSTALLED = True
    targets = {'verl.utils.megatron.optimizer', 'verl.models.mcore.bridge',
               'rllm.trainer.verl.utils', 'verl.checkpoint_engine',
               'verl.workers.engine.megatron.transformer_impl', 'rtp_native_server'}

    def after_import(fullname, module):
        if fullname == 'verl.utils.megatron.optimizer':
            _install_optimizer_hooks(module)
        elif fullname == 'verl.models.mcore.bridge':
            import q36_text_bridge_candidate
            q36_text_bridge_candidate.install_for_verl()
        elif fullname == 'rllm.trainer.verl.utils':
            import q36_rllm_sync_compat
            q36_rllm_sync_compat.install()
        elif fullname == 'verl.checkpoint_engine':
            import q36_checkpoint_manager_compat
            q36_checkpoint_manager_compat.install(module)
        elif fullname == 'verl.workers.engine.megatron.transformer_impl':
            _locked_import('q36_training_library_guard',
                '1fcfb073744e15deb6a363cb5e63ddeaa0ae48d68e1c85975a1a2a6600a4a5a1').install(module)
            import q36_resident_export_compat
            q36_resident_export_compat.install(module)
            import q36_text_forward_candidate
            q36_text_forward_candidate.install()
        elif fullname == 'rtp_native_server':
            _locked_import('q36_source_binding',
                '486c709867a8039eaefb3b087bfaaa093a606cd012fbf9502a98881ab632808f').verify_source(fullname, module=module)
            import q36_rollout_budget_guard
            q36_rollout_budget_guard.install(module)

    class Loader(importlib.abc.Loader):
        def __init__(self, wrapped, fullname): self.wrapped = wrapped; self.fullname = fullname
        def create_module(self, spec):
            callback = getattr(self.wrapped, 'create_module', None)
            return callback(spec) if callback is not None else None
        def exec_module(self, module):
            self.wrapped.exec_module(module)
            after_import(self.fullname, module)
        def __getattr__(self, name): return getattr(self.wrapped, name)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in targets: return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if fullname == 'rtp_native_server':
                _locked_import('q36_source_binding',
                    '486c709867a8039eaefb3b087bfaaa093a606cd012fbf9502a98881ab632808f').verify_source(fullname, spec=spec)
            if spec is not None and spec.loader is not None: spec.loader = Loader(spec.loader, fullname)
            return spec

    sys.meta_path.insert(0, Finder())
    for target in targets:
        if target in sys.modules:
            after_import(target, sys.modules[target])


def install_driver_hook(ray_runtime_env):
    import full_tensor_refit
    full_tensor_refit.install_driver_hook(ray_runtime_env)
    original = ray_runtime_env.get_ppo_ray_runtime_env
    if getattr(original, '_q36_nonshared_bootstrap', False): return

    @functools.wraps(original)
    def get_runtime():
        runtime = original()
        if runtime.get('worker_process_setup_hook') != 'full_tensor_refit.worker_setup':
            raise RuntimeError('unexpected Ray worker setup hook')
        runtime['worker_process_setup_hook'] = 'q36_nonshared_worker_bootstrap.worker_setup'
        runtime.setdefault('env_vars', {}).pop('CUDA_VISIBLE_DEVICES', None)
        return runtime

    get_runtime._q36_nonshared_bootstrap = True
    ray_runtime_env.get_ppo_ray_runtime_env = get_runtime
