"""Cache-directory configuration shared by test launchers and the runtime."""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping
from urllib.parse import urlparse

_AUTOMATIC_CACHE_ENVS = "_RTP_LLM_AUTOMATIC_JIT_CACHE_ENVS"


def _automatic_cache_envs(environ: Mapping[str, str]) -> dict[str, str]:
    try:
        values = json.loads(environ.get(_AUTOMATIC_CACHE_ENVS, "{}"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(values, dict):
        return {}
    return {k: v for k, v in values.items() if isinstance(v, str) and v.strip()}


@dataclass(frozen=True)
class CacheEnvConfig:
    """Capture opt-outs before automatic paths are exported to libraries."""

    explicit_envs: frozenset[str]


def read_cache_env(
    env_names: Iterable[str], environ: Mapping[str, str] | None = None
) -> CacheEnvConfig:
    environ = os.environ if environ is None else environ
    automatic = _automatic_cache_envs(environ)
    return CacheEnvConfig(
        frozenset(
            name
            for name in env_names
            if (value := environ.get(name, "").strip()) and automatic.get(name) != value
        )
    )


def configure_cache_env(
    env_name: str,
    directory: str | Path,
    *,
    automatic: bool | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Write a library path and its provenance together, preserving other entries.

    By default, retain an inherited automatic path's provenance. Legacy
    launcher defaults otherwise remain opt-outs, just as an ordinary preset is.
    """
    environ = os.environ if environ is None else environ
    marked = _automatic_cache_envs(environ)
    if automatic is None:
        current = environ.get(env_name, "").strip()
        automatic = bool(current and marked.get(env_name) == current)
    value = str(directory)
    environ[env_name] = value
    if automatic:
        marked[env_name] = value
    else:
        marked.pop(env_name, None)
    if marked:
        environ[_AUTOMATIC_CACHE_ENVS] = json.dumps(marked, sort_keys=True)
    else:
        environ.pop(_AUTOMATIC_CACHE_ENVS, None)


def safe_local_path(path: str | Path) -> Path | None:
    path = Path(os.path.abspath(Path(path).expanduser()))
    try:
        if any(candidate.is_symlink() for candidate in (path, *path.parents)):
            return None
        path = path.resolve()
    except OSError:
        return None
    remote = os.environ.get("REMOTE_JIT_DIR", "").strip()
    if remote and not urlparse(remote).scheme:
        remote_path = Path(remote).expanduser().resolve()
        if path == remote_path or remote_path in path.parents:
            return None
    return path


def ensure_writable_directory(path: str | Path) -> Path | None:
    path = safe_local_path(path)
    if path is None:
        return None
    try:
        try:
            path.mkdir(mode=0o700, parents=True)
        except FileExistsError:
            pass
        else:
            path.chmod(0o700)
        if path.stat().st_uid != os.getuid():
            return None
        with tempfile.NamedTemporaryFile(prefix=".write_probe_", dir=path):
            pass
    except OSError:
        return None
    return path


def _warn_on_explicit_cache_path(path: str | Path) -> None:
    # Explicit caches can deliberately be symlinked or shared; leave their modes alone.
    try:
        resolved = Path(os.path.abspath(Path(path).expanduser()))
        if resolved.is_symlink() or any(c.is_symlink() for c in resolved.parents):
            logging.warning("[JIT] explicit cache path is a symlink: %s", resolved)
        elif resolved.is_dir():
            st = resolved.stat()
            if st.st_uid != os.getuid():
                logging.warning(
                    "[JIT] explicit cache path is owned by uid %s: %s",
                    st.st_uid,
                    resolved,
                )
            elif st.st_mode & 0o022:
                logging.warning(
                    "[JIT] explicit cache path is group/world writable: %s", resolved
                )
    except OSError:
        pass


def local_jit_fallback(name: str) -> Path:
    return Path(tempfile.gettempdir()).resolve() / f"rtp-llm-{os.getuid()}" / name


def configure_writable_cache_env(
    env_name: str,
    default_path: Path,
    fallback_name: str,
    cache_subpath: tuple[str, ...] = (),
) -> None:
    def writable_base(path):
        base = safe_local_path(path)
        if base is None:
            return None
        cache_dir = base.joinpath(*cache_subpath) if cache_subpath else base
        return base if ensure_writable_directory(cache_dir) is not None else None

    requested = os.environ.get(env_name, "").strip()
    if requested:
        _warn_on_explicit_cache_path(requested)
        directory = Path(os.path.abspath(Path(requested).expanduser()))
        configure_cache_env(env_name, directory)
    else:
        directory = writable_base(default_path)
        if directory is None:
            directory = writable_base(local_jit_fallback(fallback_name))
        if directory is None:
            raise OSError(f"no writable {fallback_name} cache directory")
        configure_cache_env(env_name, directory, automatic=True)
