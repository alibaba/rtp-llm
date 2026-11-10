"""Bind the native ARM RoPE hook to the explicitly qualified source."""
import hashlib
import importlib.machinery
import os
from pathlib import Path
import sys


def verify_rope_source():
    root = Path(os.environ['Q36_ARM_ROPE_DIR']).resolve(strict=True)
    expected = root / 'rtp_arm_bf16_import_hook.py'
    digest = os.environ['Q36_ARM_ROPE_SHA256']
    if len(digest) != 64 or expected.is_symlink() or not expected.is_file():
        raise RuntimeError('qualified ARM RoPE source and SHA are required')
    spec = importlib.machinery.PathFinder.find_spec('rtp_arm_bf16_import_hook', sys.path)
    if spec is None or not spec.origin or Path(spec.origin).resolve() != expected:
        raise RuntimeError('ARM RoPE resolves outside the qualified directory')
    loaded = sys.modules.get('rtp_arm_bf16_import_hook')
    if loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != expected:
        raise RuntimeError('loaded ARM RoPE differs from the qualified source')
    if hashlib.sha256(expected.read_bytes()).hexdigest() != digest:
        raise RuntimeError('ARM RoPE source digest mismatch')
