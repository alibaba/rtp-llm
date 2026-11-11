"""Opt-in custom_backend_module entry point; exact private bootstrap only."""
import hashlib
from pathlib import Path
import production_nixl_bootstrap_candidate as bootstrap

path=Path(bootstrap.__file__)
if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!='9dce1870a334d8c867eebac8ef7f430c60737c47597febcff5d14810be4df976':
    raise RuntimeError('unqualified managed NIXL bootstrap')
bootstrap.install()
