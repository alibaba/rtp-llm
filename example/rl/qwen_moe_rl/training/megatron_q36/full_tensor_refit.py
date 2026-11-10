"""Private baseline adapter for Bridge full-tensor export into existing RTP.

Install explicitly in every rollout worker *after* importing the existing RTP
adapter. This preserves the worker's naive-sync lifecycle and delegates its
native transaction, per-rank identity checks, complete-source gate and ACKs to
the original implementation. No sharded transfer is implemented here.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time

_WORKER_INSTALLED = False


def _source_guard():
    path = Path(os.environ['Q36_CODE_DIR']).resolve(strict=True) / 'q36_source_binding.py'
    spec = importlib.machinery.PathFinder.find_spec('q36_source_binding', sys.path)
    loaded = sys.modules.get('q36_source_binding')
    if (path.is_symlink() or not path.is_file() or
            hashlib.sha256(path.read_bytes()).hexdigest() != '486c709867a8039eaefb3b087bfaaa093a606cd012fbf9502a98881ab632808f' or
            spec is None or not spec.origin or Path(spec.origin).resolve() != path or
            loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != path):
        raise RuntimeError('qualified Q36 source guard is unavailable')
    return importlib.import_module('q36_source_binding')


def _patch_loaded(module):
    _source_guard().verify_source('rtp_verl_backend', module=module)
    adapter_class = getattr(module, "RTPServerAdapter", None)
    if adapter_class is None:
        raise RuntimeError("RTPServerAdapter is unavailable in the RTP module")
    install(adapter_class)


def worker_setup():
    """Chain RTP's existing passive worker hook and patch its lazy import."""
    import install_rtp_bridge
    install_rtp_bridge.worker_setup()
    global _WORKER_INSTALLED
    if _WORKER_INSTALLED:
        return
    _WORKER_INSTALLED = True
    loaded = sys.modules.get("rtp_verl_backend")
    if loaded is not None:
        _patch_loaded(loaded)
        return

    class Loader(importlib.abc.Loader):
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def create_module(self, spec):
            create = getattr(self.wrapped, "create_module", None)
            return create(spec) if create is not None else None

        def exec_module(self, module):
            self.wrapped.exec_module(module)
            _patch_loaded(module)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "rtp_verl_backend":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            _source_guard().verify_source(fullname, spec=spec)
            if spec is not None and spec.loader is not None:
                spec.loader = Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, Finder())


def install_driver_hook(ray_runtime_env):
    """Chain the existing driver's setup-hook configuration without starting Ray."""
    original = ray_runtime_env.get_ppo_ray_runtime_env
    if getattr(original, "_q36_megatron_full_tensor_hook", False):
        return

    @functools.wraps(original)
    def runtime_env():
        result = original()
        previous = result.get("worker_process_setup_hook")
        if previous not in (None, "install_rtp_bridge.worker_setup", "full_tensor_refit.worker_setup"):
            raise RuntimeError("a different Ray worker setup hook is configured")
        result["worker_process_setup_hook"] = "full_tensor_refit.worker_setup"
        return result

    runtime_env._q36_megatron_full_tensor_hook = True
    ray_runtime_env.get_ppo_ray_runtime_env = runtime_env


class _DeferredFP32:
    """Move the tiny GDN cast into the original adapter's guarded block."""

    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def dtype(self):
        import torch
        return torch.float32

    @property
    def device(self):
        return self._tensor.device

    def detach(self):
        return self._tensor.detach().float()


def _baseline_stream(weights, stats):
    for name, tensor in weights:
        stats["exported_tensors"] += 1
        # RTP's existing HF source protocol requires these GDN values as FP32.
        # This is an upcast of the current training value; it cannot recover
        # precision lost if the persistent Megatron parameter is BF16.
        if name.endswith((".linear_attn.A_log", ".linear_attn.dt_bias")):
            if str(tensor.dtype) == "torch.bfloat16":
                tensor = _DeferredFP32(tensor)
                stats["gdn_bf16_to_fp32"] += 1
        yield name, tensor


def install(adapter_class):
    """Install the baseline wrapper once; returns the effective method.

    This is not a drop-in recipe: caller must install it inside each Ray worker,
    qualify the pinned exporter/runtime and ensure exact checkpoint key names.
    The original adapter retains all source/identity/version/ACK checks.
    """
    original = adapter_class.update_weights
    if getattr(original, "_q36_megatron_full_tensor_baseline", False):
        return original

    @functools.wraps(original)
    async def update_weights(self, weights, global_steps=None, **kwargs):
        stats = {"exported_tensors": 0, "gdn_bf16_to_fp32": 0}
        start = time.monotonic()
        result = await original(self, _baseline_stream(weights, stats), global_steps=global_steps, **kwargs)
        # This is total exporter + transfer + transaction time, not a component
        # breakdown, and is reported only after the original transaction returns.
        print("Q36_MEGATRON_FULL_REFIT " + json.dumps({
            "global_step": global_steps,
            "elapsed_seconds": time.monotonic() - start,
            "exported_tensors": stats["exported_tensors"],
            "gdn_bf16_to_fp32": stats["gdn_bf16_to_fp32"],
            "transport": "existing_rtp_full_hf_tensor",
        }, sort_keys=True), flush=True)
        return result

    update_weights._q36_megatron_full_tensor_baseline = True
    adapter_class.update_weights = update_weights
    return update_weights
