"""Private, opt-in EP8 worker hook candidate; not GPU-qualified.

Preserves the existing nonshared BF16 injection and all EP8 manifest checks.
Call in a fresh process before installing the standalone bootstrap. The
independently sourced live-provider ledger is not created by this module.
"""
import functools
import hashlib
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import sys

EP8_SOURCES = {
    'q36_ep8_sr_binder_v2': 'b914bd678d7b2a6d3c94844b360e1d6c2d993c445ef7694ed061ad42e1d2ecd8',
    'bf16_megatron_factory_q36_ep8_v2': '07a8a7e4772c09bb5a4351f6c4c69bcdfccdc1e2e1e18c7774695b32b08f9b1c',
    'q36_ep_profile': '5ae5ca94fd861113db775b1b3427637429cebc5426b63cd3fbe9e7de0a8b4a16',
}


def _load_verified(name, root):
    path = root / (name + '.py')
    spec = importlib.machinery.PathFinder.find_spec(name, sys.path)
    loaded = sys.modules.get(name)
    if (path.is_symlink() or not path.is_file() or
        hashlib.sha256(path.read_bytes()).hexdigest() != EP8_SOURCES[name] or
        spec is None or not spec.origin or Path(spec.origin).resolve() != path or
        loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != path):
        raise RuntimeError('EP8 candidate source binding failed: ' + name)
    return importlib.import_module(name)


def _install_ep8_optimizer_hooks(verl_optim):
    import megatron.core.optimizer as core_optim
    import torch
    import torch.distributed as dist
    import q36_nonshared_bf16_worker_candidate as injection

    root = Path(os.environ['Q36_OPT_SOURCE_DIR']).resolve(strict=True)
    if str(root) not in sys.path:
        raise RuntimeError('verified optimizer root must already precede imports on PYTHONPATH')
    injection.verify_candidate_sources(root)
    factory = _load_verified('bf16_megatron_factory_q36_ep8_v2', root)
    # Resolve dependencies now, before any collective-bearing mapper is called.
    _load_verified('q36_ep_profile', root)
    binder = _load_verified('q36_ep8_sr_binder_v2', root)

    def map_live_model(chunks):
        manifest = os.environ.get('Q36_EP8_SR_MANIFEST', '')
        digest = os.environ.get('Q36_EP8_SR_MANIFEST_SHA256', '')
        local_error = None
        try:
            path = Path(manifest)
            if (path.is_symlink() or not path.is_file() or len(digest) != 64 or
                    hashlib.sha256(path.read_bytes()).hexdigest() != digest):
                raise ValueError('independently reviewed EP8 source/live manifest required')
            value = json.loads(path.read_bytes())
            if (value.get('schema') != 'q36-ep8-live-provider-sr-v1' or
                value.get('binding_scope') != 'independent-source-and-live-provider-reviewed'):
                raise ValueError('incorrect reviewed EP8 manifest schema')
        except Exception as error:
            local_error = error
        # Every rank must enter the binder or none do. Its own collective is
        # nested inside the outer injection's world-vote error barrier.
        flag = torch.tensor(int(local_error is None), dtype=torch.int32,
                            device=torch.cuda.current_device())
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if not bool(flag.item()):
            raise RuntimeError('all-rank EP8 reviewed source ledger preflight failed') from local_error
        return binder.bind_all_ranks(chunks, manifest, digest)

    injection.install_hooks(verl_optim, core_optim, factory, torch, dist, map_live_model)


def _select_ep8():
    import q36_nonshared_worker_bootstrap as base
    current = base._install_optimizer_hooks
    if current is _install_ep8_optimizer_hooks:
        return
    if getattr(base, '_INSTALLED', False):
        raise RuntimeError('EP8 hook must be selected before the existing worker bootstrap')
    base._install_optimizer_hooks = _install_ep8_optimizer_hooks


def worker_setup():
    _select_ep8()
    from q36_standalone_bootstrap_selector import select_bootstrap
    select_bootstrap().worker_setup()


def install_driver():
    _select_ep8()
    from q36_standalone_bootstrap_selector import select_bootstrap
    selected = select_bootstrap()
    selected.install_driver()
    from rllm.trainer.verl import ray_runtime_env
    original = ray_runtime_env.get_ppo_ray_runtime_env
    if getattr(original, '_q36_ep8_worker', False):
        return

    @functools.wraps(original)
    def runtime():
        result = original()
        if result.get('worker_process_setup_hook') != selected.__name__ + '.worker_setup':
            raise RuntimeError('unexpected standalone worker setup chain')
        result['worker_process_setup_hook'] = 'q36_ep8_worker_bootstrap_candidate.worker_setup'
        result.setdefault('env_vars', {}).pop('CUDA_VISIBLE_DEVICES', None)
        return result

    runtime._q36_ep8_worker = True
    ray_runtime_env.get_ppo_ray_runtime_env = runtime
