"""Limit lazy RTP package exports in the explicitly selected Py312 RPC worker."""
import hashlib
import importlib
import os
from pathlib import Path
import sys

RTP_INIT_SHA256 = 'f28a8d35ed1e1eba31652e2ecfe2e131916437d329a4a036f030aa948779fe94'
ALLOWED_LAZY = frozenset(('_ft_pickler', 'enable_compile_monitor'))


def install(repo):
    if os.environ.get('Q36_USE_RPC_OVERLAY') != '1' or sys.version_info[:2] != (3, 12):
        raise RuntimeError('RPC-only package guard requires explicitly selected Py312 RPC profile')
    root = Path(repo).resolve(strict=True)
    init = root / 'rtp_llm' / '__init__.py'
    if init.is_symlink() or not init.is_file() or hashlib.sha256(init.read_bytes()).hexdigest() != RTP_INIT_SHA256:
        raise RuntimeError('RPC-only RTP package source identity mismatch')
    if 'rtp_llm.ops' in sys.modules:
        raise RuntimeError('native RTP ops already loaded in RPC-only process')
    module = importlib.import_module('rtp_llm')
    if Path(module.__file__).resolve() != init.resolve():
        raise RuntimeError('RPC-only RTP package loaded from an unqualified source')
    original = module.__dict__.get('__getattr__')
    if getattr(original, '_q36_rpc_only_guard', False):
        return
    if not callable(original):
        raise RuntimeError('qualified RTP lazy export hook is missing')

    def guarded(name):
        if name in ALLOWED_LAZY:
            return original(name)
        raise AttributeError(f"module {module.__name__!r} has no attribute {name!r}")

    guarded._q36_rpc_only_guard = True
    module.__getattr__ = guarded
