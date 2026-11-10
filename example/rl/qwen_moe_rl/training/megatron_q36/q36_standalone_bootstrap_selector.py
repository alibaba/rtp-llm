"""Explicitly select the v4 no-concurrent on-policy adapter when requested."""
import hashlib
import importlib
import importlib.machinery
import os
from pathlib import Path
import sys

V4_SHA256 = 'd3c993deb80505b578b97fd5222ff396dec4f0b5362dc6937aa503ff211e66a8'
V5_SHA256 = 'b934d6812d788e21918bd2a18e00c1147129da177ab558c6d0decaffd7a9d6b4'
RPC_SHA256 = '6c92bdb89c6e83659e4036debea191f77e88e01cb716d18acc02c082fb019e0d'


def select_bootstrap():
    flag = os.environ.get('Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT', '0')
    if flag not in ('0', '1'):
        raise ValueError('Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT must be 0 or 1')
    resident = os.environ.get('Q36_DISAGG_RESIDENT_KV', '0')
    if resident not in ('0', '1') or (resident == '1' and flag != '1'):
        raise ValueError('resident KV requires explicit no-concurrent on-policy selection')
    rpc = os.environ.get('Q36_USE_RPC_OVERLAY', '0')
    if rpc not in ('0', '1') or (rpc == '1' and resident != '1'):
        raise ValueError('RPC overlay requires the explicit resident KV profile')
    name = ('standalone_bootstrap_rpc_v6' if rpc == '1' else
            'standalone_bootstrap_v5' if resident == '1' else
            'standalone_bootstrap_v4' if flag == '1' else 'standalone_bootstrap_v3')
    if flag == '1':
        root = Path(os.environ['Q36_CODE_DIR']).resolve(strict=True)
        source = root / (name + '.py')
        spec = importlib.machinery.PathFinder.find_spec(name, sys.path)
        loaded = sys.modules.get(name)
        expected = RPC_SHA256 if rpc == '1' else V5_SHA256 if resident == '1' else V4_SHA256
        if (source.is_symlink() or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected or
            spec is None or not spec.origin or Path(spec.origin).resolve() != source or
            loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != source):
            raise RuntimeError('selected standalone bootstrap source identity mismatch')
    return importlib.import_module(name)
