"""AWI adaptation-evaluator startup facts (measured, never declared).

Two responsibilities:

1. ``install_startup_network_probe`` — a ``sys.addaudithook`` counter for
   outbound network fetches during startup. Same-host traffic (unix
   sockets, loopback, and this host's own interface IPs used by intra-host
   gRPC/NCCL bootstrap) is excluded; anything else (package/model
   downloads, external services) increments the counter. The count is
   reported through ``GET /awi/observations`` as
   ``startup_network_fetches``.

2. ``publish_startup_facts`` — pushes build identity, effective runtime
   configuration and the measured fetch count into the backend C++
   ``AwiCollector`` (via the ``libth_transformer.awi`` pybind submodule)
   so the observation RPC serves them per rank.

Build identity note: a commit cannot embed its own SHA, so the SHAs ride
in launcher env (``AWI_BUILD_INTERNAL_SHA`` / ``AWI_BUILD_PUBLIC_SHA``),
recorded in the delivered candidate launch spec; the evaluator
independently verifies ``source_dirs`` HEADs against the candidate pair.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading

_CONNECT_EVENTS = frozenset({"socket.connect", "socket.connect_ex"})

_lock = threading.Lock()
_fetch_count = 0
_installed = False
_local_hosts: set = set()


def _collect_local_hosts() -> set:
    hosts = {"localhost", "127.0.0.1", "::1", "[::1]", ""}
    try:
        hostname = socket.gethostname()
        hosts.add(hostname)
        for info in socket.getaddrinfo(hostname, None):
            addr = info[4][0]
            if isinstance(addr, str):
                hosts.add(addr)
    except OSError:
        pass
    return hosts


def _is_same_host(addr) -> bool:
    # AF_UNIX addresses arrive as str/bytes paths: same-host IPC.
    if isinstance(addr, (str, bytes)):
        return True
    if not isinstance(addr, tuple) or not addr:
        return True
    host = addr[0]
    if not isinstance(host, str):
        return True
    if host.startswith("127.") or host in ("::1", "[::1]", "", "localhost"):
        return True
    return host in _local_hosts


def _audit_hook(event: str, args) -> None:
    global _fetch_count
    if event in _CONNECT_EVENTS:
        address = args[1] if len(args) > 1 else None
        if not _is_same_host(address):
            with _lock:
                _fetch_count += 1


def install_startup_network_probe() -> None:
    """Install the audit-hook counter once per process, as early as possible."""
    global _installed, _local_hosts
    with _lock:
        if _installed:
            return
        _installed = True
        _local_hosts = _collect_local_hosts()
    sys.addaudithook(_audit_hook)


def startup_network_fetches() -> int:
    with _lock:
        return _fetch_count


def probe_installed() -> bool:
    with _lock:
        return _installed


def build_configuration_json(py_env_configs, engine=None) -> str:
    """Effective configuration derived from the live engine objects."""
    context_tokens = -1
    try:
        if engine is not None and getattr(engine, "config", None) is not None:
            context_tokens = int(engine.config.max_seq_len)
    except Exception:  # noqa: BLE001 - configuration must never break startup
        logging.warning("AWI: could not read max_seq_len from engine config")
    memory_kv_reuse = bool(
        getattr(getattr(py_env_configs, "kv_cache_config", None), "reuse_cache", False)
    )
    graph_backend = "hip" if getattr(__import__("torch").version, "hip", None) else "cuda"
    return json.dumps(
        {
            "context_tokens": context_tokens,
            "memory_kv_reuse": memory_kv_reuse,
            "graph_backend": graph_backend,
        },
        sort_keys=True,
    )


def publish_startup_facts(py_env_configs=None, engine=None) -> None:
    """Push startup facts into the C++ collector (backend ranks only)."""
    try:
        import importlib

        awi = getattr(importlib.import_module("libth_transformer"), "awi", None)
        if awi is None:
            logging.warning("AWI: libth_transformer.awi unavailable; facts not published")
            return
        awi.set_build_info(
            os.environ.get("AWI_BUILD_INTERNAL_SHA", ""),
            os.environ.get("AWI_BUILD_PUBLIC_SHA", ""),
        )
        if py_env_configs is not None:
            awi.set_configuration_json(build_configuration_json(py_env_configs, engine))
        awi.set_startup_network_fetches(startup_network_fetches())
        logging.info(
            "AWI: startup facts published (network_fetches=%d)",
            startup_network_fetches(),
        )
    except Exception as e:  # noqa: BLE001 - never fail startup for telemetry
        logging.warning("AWI: failed to publish startup facts: %s", e)
