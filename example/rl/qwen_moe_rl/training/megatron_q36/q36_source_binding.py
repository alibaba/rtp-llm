"""Bind lazy RTP imports to the reviewed adapter directory and exact bytes."""
import hashlib
import importlib.machinery
import os
from pathlib import Path
import sys

SOURCE_SHA = {
    'rtp_verl_backend': '77ba91ba14584b4a996d77d2d6c80a8b7185de413307cdc70532ef469727c599',
    'rtp_native_server': '91a9fe875d2f757b9f3c820514311446114bddcaa71d4a83ce9b11c406ec0635',
    'q36_refit_timing': '11f4ea8c99d19d4651e797b9c61a5f59e826897fce2224320d46375a041bf554',
    'q36_refit_bounded_digest_candidate': '118cd27ccb5ab5a6f22e28dba6cc7916584788195f39dd571655ae40844caf43',
}


def verify_source(name, *, spec=None, module=None):
    if name not in SOURCE_SHA:
        raise ValueError('unrecognized Q36 adapter module: ' + name)
    root = Path(os.environ['Q36_ADAPTER_DIR']).resolve(strict=True)
    expected = root / (name + '.py')
    if expected.is_symlink() or not expected.is_file():
        raise RuntimeError('adapter source missing: ' + name)
    chosen = spec or importlib.machinery.PathFinder.find_spec(name, sys.path)
    if chosen is None or not chosen.origin or Path(chosen.origin).resolve() != expected:
        raise RuntimeError('adapter module resolved outside qualified source: ' + name)
    loaded = module or sys.modules.get(name)
    if loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != expected:
        raise RuntimeError('loaded adapter module differs from qualified source: ' + name)
    if hashlib.sha256(expected.read_bytes()).hexdigest() != SOURCE_SHA[name]:
        raise RuntimeError('adapter source digest mismatch: ' + name)
