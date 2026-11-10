"""Explicit RPC stubs plus the existing standalone worker bootstrap."""
import functools
import hashlib
import importlib.machinery
import os
from pathlib import Path
import socket
import sys


def _load_v5():
    name = 'standalone_bootstrap_v5'
    source = Path(os.environ['Q36_CODE_DIR']).resolve(strict=True) / (name + '.py')
    spec = importlib.machinery.PathFinder.find_spec(name, sys.path)
    loaded = sys.modules.get(name)
    if (source.is_symlink() or not source.is_file() or
        hashlib.sha256(source.read_bytes()).hexdigest() != '329b976038d82df9183f62731524a0336a7603a2bfc10d17f5a1aae3f3dece45' or
        spec is None or not spec.origin or Path(spec.origin).resolve() != source or
        loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != source):
        raise RuntimeError('RPC bootstrap v5 source identity mismatch')
    import standalone_bootstrap_v5
    return standalone_bootstrap_v5


def _install_proto():
    node = socket.gethostbyname(socket.gethostname())
    choices = {
        '10.44.49.51': ('/home/admin/q36-disagg-20261110/manager-rtp-d4b851c0',
                         '05440944d117c3405b9af04211e780a0866476217eb0534c1a68f156760eb957'),
        '10.44.33.31': ('/home/admin/q36-disagg-20261110/rtp-source',
                         '34ffa19d9c3a452a44258f062383993210dfdbdbe3588c74f2b1da21cd61093e'),
    }
    source, expected = choices[node]
    client = Path(source) / 'rtp_llm/RL/replica_rpc_client.py'
    if hashlib.sha256(client.read_bytes()).hexdigest() != expected:
        raise RuntimeError('node-specific RPC client identity mismatch')
    os.environ['Q36_PROTO_REPO'] = source
    if source not in sys.path:
        sys.path.insert(0, source)
    name = 'q36_proto_overlay_candidate'
    overlay = Path(os.environ['Q36_PROTO_OVERLAY_DIR']).resolve(strict=True) / (name + '.py')
    spec = importlib.machinery.PathFinder.find_spec(name, sys.path)
    loaded = sys.modules.get(name)
    if (overlay.is_symlink() or not overlay.is_file() or
        hashlib.sha256(overlay.read_bytes()).hexdigest() != 'c12420e515e8744ce0ac6394b629963e5f630341132a6dd5aa2343d71a3d0b77' or
        spec is None or not spec.origin or Path(spec.origin).resolve() != overlay or
        loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != overlay):
        raise RuntimeError('RPC proto overlay source identity mismatch')
    import q36_proto_overlay_candidate
    return q36_proto_overlay_candidate.install()


def worker_setup():
    _install_proto()
    _load_v5().worker_setup()


def install_driver():
    _install_proto()
    _load_v5().install_driver()
    from rllm.trainer.verl import ray_runtime_env
    original = ray_runtime_env.get_ppo_ray_runtime_env

    @functools.wraps(original)
    def runtime():
        value = original()
        if value.get('worker_process_setup_hook') != 'standalone_bootstrap_v5.worker_setup':
            raise RuntimeError('unexpected standalone worker hook')
        value['worker_process_setup_hook'] = 'standalone_bootstrap_rpc_v6.worker_setup'
        return value

    ray_runtime_env.get_ppo_ray_runtime_env = runtime
