"""Explicitly select the v4 no-concurrent on-policy adapter when requested."""
import hashlib
import importlib
import importlib.machinery
import os
from pathlib import Path
import sys

V4_SHA256 = 'd3c993deb80505b578b97fd5222ff396dec4f0b5362dc6937aa503ff211e66a8'


def select_bootstrap():
    flag = os.environ.get('Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT', '0')
    if flag not in ('0', '1'):
        raise ValueError('Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT must be 0 or 1')
    name = 'standalone_bootstrap_v4' if flag == '1' else 'standalone_bootstrap_v3'
    if flag == '1':
        root = Path(os.environ['Q36_CODE_DIR']).resolve(strict=True)
        source = root / (name + '.py')
        spec = importlib.machinery.PathFinder.find_spec(name, sys.path)
        loaded = sys.modules.get(name)
        if (source.is_symlink() or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != V4_SHA256 or
            spec is None or not spec.origin or Path(spec.origin).resolve() != source or
            loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != source):
            raise RuntimeError('selected v4 standalone bootstrap source identity mismatch')
    return importlib.import_module(name)
