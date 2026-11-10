"""Shared helpers for the AWI evaluator instrumentation contract.

The evaluator protocol (evaluator-protocol.md) requires per-rank telemetry
derived from real execution: weight placement censuses written by each
backend rank at load time, a startup network-fetch counter, and the build
source pair. All ranks and the frontend share one container filesystem, so
rank documents are exchanged through ``AWI_REPORT_DIR``; engine events flow
through the AwiCommand gRPC channel (see AwiRankStore).
"""

import json
import logging
import os
import socket
import threading
from typing import Any, Dict, List, Optional

DEFAULT_REPORT_DIR = "/tmp/rtp_llm_awi"
_STAMP_ENV_INTERNAL = "AWI_BUILD_INTERNAL_SHA"
_STAMP_ENV_PUBLIC = "AWI_BUILD_PUBLIC_SHA"


def report_dir() -> str:
    return os.environ.get("AWI_REPORT_DIR", DEFAULT_REPORT_DIR)


def _rank_tag() -> str:
    # WORLD_RANK is set per backend process by the gang launcher; the
    # frontend falls back to its frontend_server_id so documents never
    # collide.
    rank = os.environ.get("WORLD_RANK")
    if rank is not None:
        return f"rank{rank}"
    return f"frontend{os.environ.get('FRONTEND_SERVER_ID', os.getpid())}"


def write_rank_document(name: str, document: Any) -> str:
    """Atomically publish one JSON document for this rank/process."""
    directory = report_dir()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}_{_rank_tag()}.json")
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as handle:
        json.dump(document, handle)
    os.replace(tmp, path)
    return path


def read_rank_documents(name: str) -> List[Dict[str, Any]]:
    """Read every rank document with the given base name."""
    directory = report_dir()
    documents = []
    try:
        entries = sorted(os.listdir(directory))
    except FileNotFoundError:
        return documents
    prefix = f"{name}_rank"
    for entry in entries:
        if not entry.startswith(prefix) or not entry.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, entry)) as handle:
                documents.append(json.load(handle))
        except (OSError, ValueError) as error:
            logging.warning("awi document %s unreadable: %s", entry, error)
    return documents


def build_source() -> Dict[str, str]:
    """The paired source SHAs this image was built from.

    Resolution order: launch environment overrides (the candidate launch
    spec declares the exact built pair, reviewable against the CI receipt)
    then the committed packaging stamp. Both are declarations; the binding
    proof is the pipeline-18410 receipt plus the evaluator's own git HEAD
    check of launch.source_dirs.
    """
    from rtp_llm.utils.awi_source_stamp import BUILD_SOURCE

    internal = os.environ.get(_STAMP_ENV_INTERNAL) or BUILD_SOURCE.get(
        "internal", ""
    )
    public = os.environ.get(_STAMP_ENV_PUBLIC) or BUILD_SOURCE.get("public", "")
    return {"internal": internal, "public": public}


class StartupNetworkGuard:
    """Count outbound network connections attempted during startup.

    The contract requires ``startup_network_fetches`` to be a real
    measurement: the guard wraps socket connect entry points from the very
    beginning of every server process and records each attempt to a
    non-loopback, non-local address. Loopback connects are the internal
    frontend<->backend gRPC/HTTP plumbing of the same container and are not
    fetches. The count is published per process for aggregation.
    """

    _installed = False
    _lock = threading.Lock()
    fetches: List[Dict[str, Any]] = []

    @classmethod
    def install(cls) -> None:
        with cls._lock:
            if cls._installed:
                return
            cls._installed = True

        original_connect = socket.socket.connect
        original_connect_ex = socket.socket.connect_ex
        original_create_connection = socket.create_connection

        def is_external(address: Any) -> bool:
            host = None
            if isinstance(address, tuple) and address:
                host = str(address[0])
            elif isinstance(address, (str, bytes)):
                host = address.decode() if isinstance(address, bytes) else address
            if not host:
                return False
            if host.startswith("/"):  # unix socket
                return False
            if host in ("localhost", "::1"):
                return False
            try:
                packed = socket.inet_aton(host)
                return packed[0:1] not in (b"\x7f", b"\x00")
            except OSError:
                return True  # hostname => DNS/external intent

        def record(address: Any) -> None:
            try:
                cls.fetches.append({"address": repr(address)})
            except Exception:  # noqa: BLE001
                pass

        def connect(self: socket.socket, address: Any) -> Any:
            if is_external(address):
                record(address)
            return original_connect(self, address)

        def connect_ex(self: socket.socket, address: Any) -> Any:
            if is_external(address):
                record(address)
            return original_connect_ex(self, address)

        def create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
            if is_external(address):
                record(address)
            return original_create_connection(address, *args, **kwargs)

        socket.socket.connect = connect  # type: ignore[method-assign]
        socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
        socket.create_connection = create_connection  # type: ignore[assignment]

    @classmethod
    def publish(cls) -> Optional[str]:
        """Snapshot the process-local fetch count as a rank document."""
        try:
            return write_rank_document(
                "net_fetches",
                {
                    "count": len(cls.fetches),
                    "attempts": cls.fetches[:64],
                    "pid": os.getpid(),
                },
            )
        except OSError as error:
            logging.warning("awi net guard publish failed: %s", error)
            return None


def startup_network_fetch_count() -> int:
    """Sum published per-process fetch counts (frontend + every rank)."""
    total = 0
    for document in read_rank_documents("net_fetches"):
        try:
            total += int(document.get("count", 0))
        except (TypeError, ValueError):
            continue
    return total
