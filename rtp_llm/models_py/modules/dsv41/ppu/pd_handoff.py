"""Internal CP8 -> DP8 TCP bring-up seam, separate from native CacheStore.

The wire carries the existing canonical fixture, token history and P's first
sampled token. Each D rank receives only its own requests. A stored ACK is
followed by a decoded result; P can distinguish byte transfer from continuation.
RDMA is deliberately unsupported here: no implicit transport fallback.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import socket
import struct
import tempfile
import time
from pathlib import Path

from rtp_llm.models_py.modules.dsv41.ppu.native_protocol import native_cache_protocol

PROTOCOL = "v41-ppu-fixture-tcp-v1"
MAX_HEADER = 16 * 2**20
MAX_PAYLOAD = 8 * 2**30
CHUNK = 2**20


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def config_signature(config, checkpoint=None):
    # Location is not model identity; P and D may mount the same checkpoint
    # at different paths. Token-map contents and all model fields are covered.
    text = {k: v for k, v in config.items() if k != "checkpoint_path"}
    text["_native_cache_protocol"] = native_cache_protocol()
    if checkpoint is not None:
        root = Path(checkpoint)
        text["_checkpoint_metadata"] = {
            name: file_sha256(root / name)
            for name in (
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
            )
            if (root / name).is_file()
        }
    return hashlib.sha256(json.dumps(text, sort_keys=True).encode()).hexdigest()


def runtime_identity(root=None):
    """Verify the published source manifest and lock relevant wheel versions."""
    root = Path(root) if root else Path(__file__).resolve().parents[5]
    manifest = json.loads((root / ".v41_snapshot.json").read_text())
    actual_manifest = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True).encode()
    ).hexdigest()
    if (
        manifest.get("mode") != "snapshot"
        or actual_manifest != manifest["source_sha256"]
    ):
        raise ValueError("invalid committed source snapshot manifest")
    for entry in manifest["files"]:
        path = root / entry["path"]
        actual = os.readlink(path) if entry["kind"] == "symlink" else file_sha256(path)
        if actual != entry["value"]:
            raise ValueError(f"published source changed: {entry['path']}")
    versions = {}
    for name in (
        "torch",
        "triton",
        "deep-gemm",
        "deep-ep",
        "flash-mla",
        "safetensors",
        "transformers",
        "fast-hadamard-transform",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return {
        "snapshot_id": manifest["snapshot_id"],
        "source_sha256": manifest["source_sha256"],
        "dependencies": versions,
        "native_cache_protocol": native_cache_protocol(),
    }


def validate_handoff(manifest, *, identity, model_signature, run_id, owner, world):
    """Reject incompatible or misrouted state before accepting its payload."""
    pd = manifest.get("meta", {}).get("pd", {})
    if not manifest.get("pool_layouts"):
        raise ValueError("PD requires explicit pool dtype/stride/block layouts")
    expected = {
        "protocol": PROTOCOL,
        "identity": identity,
        "model_signature": model_signature,
        "run_id": run_id,
        "owner": owner,
        "target_dp": world,
        "target_tp": 1,
        "source_layout": "replicated-logical-cp-cache",
    }
    for key, value in expected.items():
        if pd.get(key) != value:
            raise ValueError(f"PD contract mismatch: {key}")
    if manifest.get("source_cp") != world:
        raise ValueError("PD source CP must equal the declared CP world")
    if manifest.get("producer_snapshot", {}).get("snapshot") != identity["snapshot_id"]:
        raise ValueError("PD producer snapshot mismatch")
    lengths = manifest.get("lengths", {})
    histories = pd.get("histories", {})
    seeds = pd.get("seed_tokens", {})
    owners = pd.get("owner_map", {})
    if not (lengths.keys() == histories.keys() == seeds.keys() == owners.keys()):
        raise ValueError("PD request/history/first-token/owner sets differ")
    for rid, length in lengths.items():
        if len(histories[rid]) != length or not 0 < length <= 32768:
            raise ValueError(f"PD history length mismatch: {rid}")
        if owners[rid] != owner:
            raise ValueError(f"PD request {rid} belongs to a different D owner")
        if not isinstance(seeds[rid], int) or seeds[rid] < 0:
            raise ValueError(f"invalid first token for request {rid}")
    return pd


def export_handoffs(
    cache,
    root,
    *,
    histories,
    seed_tokens,
    owner_map,
    identity,
    model_signature,
    run_id,
    world=8,
):
    """Export an independent, compact canonical cache for EVERY D owner."""
    ids = set(cache._request_tokens)
    if not (ids == histories.keys() == seed_tokens.keys() == owner_map.keys()):
        raise ValueError("PD export requires exact request/history/seed/owner coverage")
    if any(not 0 <= int(owner) < world for owner in owner_map.values()):
        raise ValueError("D owner is outside its DP world")
    paths = []
    for owner in range(world):
        local_ids = [rid for rid in sorted(ids) if owner_map[rid] == owner]
        local = cache.select_requests(local_ids)
        pd = {
            "protocol": PROTOCOL,
            "identity": identity,
            "model_signature": model_signature,
            "run_id": run_id,
            "source_layout": "replicated-logical-cp-cache",
            "target_dp": world,
            "target_tp": 1,
            "owner": owner,
            "histories": {str(rid): list(histories[rid]) for rid in local_ids},
            "seed_tokens": {str(rid): int(seed_tokens[rid]) for rid in local_ids},
            "owner_map": {str(rid): owner for rid in local_ids},
            "state_checksum": local.state_checksum(),
        }
        path = Path(root) / f"owner{owner}"
        manifest = local.export_fixture(
            path,
            meta={"source_cp": world, "snapshot": identity["snapshot_id"], "pd": pd},
        )
        validate_handoff(
            manifest,
            identity=identity,
            model_signature=model_signature,
            run_id=run_id,
            owner=owner,
            world=world,
        )
        paths.append(path)
    return paths


def _recv_exact(sock, length):
    result = bytearray()
    while len(result) < length:
        chunk = sock.recv(min(CHUNK, length - len(result)))
        if not chunk:
            raise ConnectionError("partial PD transfer: peer closed the connection")
        result.extend(chunk)
    return bytes(result)


def _send_json(sock, value):
    payload = json.dumps(value, sort_keys=True).encode()
    if len(payload) > MAX_HEADER:
        raise ValueError("PD control frame exceeds limit")
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


def _recv_json(sock):
    (length,) = struct.unpack("!Q", _recv_exact(sock, 8))
    if not 0 < length <= MAX_HEADER:
        raise ValueError("invalid PD control frame length")
    return json.loads(_recv_exact(sock, length))


def _expect(sock, state):
    reply = _recv_json(sock)
    if reply.get("state") != state:
        raise RuntimeError(f"PD expected {state}: {reply}")
    return reply


def send_handoff(path, host, port, *, timeout=1800):
    """Upload a fixture and wait for D's continuation result (TCP only)."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    payload = path / "pools.pt"
    size = payload.stat().st_size
    if not 0 < size <= MAX_PAYLOAD or file_sha256(payload) != manifest["checksum"]:
        raise ValueError("invalid outgoing PD payload")
    started = time.monotonic()
    deadline = started + timeout
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=min(5, timeout))
            break
        except (OSError, TimeoutError):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"D owner endpoint not ready: {host}:{port}")
            time.sleep(min(1, max(0, deadline - time.monotonic())))
    with sock:
        sock.settimeout(max(0.01, deadline - time.monotonic()))
        _send_json(sock, {"manifest": manifest, "payload_bytes": size})
        _expect(sock, "ready")
        transfer_started = time.monotonic()
        with payload.open("rb") as handle:
            for block in iter(lambda: handle.read(CHUNK), b""):
                sock.sendall(block)
        stored = _expect(sock, "stored")
        if stored.get("checksum") != manifest["checksum"]:
            raise ValueError("D acknowledged different cache bytes")
        transfer_seconds = time.monotonic() - transfer_started
        result = _expect(sock, "decoded")
        if result.get("checksum") != manifest["checksum"] or not result.get("ok"):
            raise ValueError("D continuation failed or acknowledged different state")
    return {
        "owner": manifest["meta"]["pd"]["owner"],
        "transport": "tcp",
        "payload_bytes": size,
        "checksum": manifest["checksum"],
        "transfer_seconds": round(transfer_seconds, 4),
        "ready_wait_seconds": round(transfer_started - started, 4),
        "roundtrip_seconds": round(time.monotonic() - started, 4),
        "decode_result": result,
    }


class HandoffSession:
    """An accepted transfer awaiting explicit loaded-and-decoded completion."""

    def __init__(self, sock, path, manifest):
        self.sock, self.path, self.manifest = sock, Path(path), manifest

    def finish(self, result):
        try:
            _send_json(
                self.sock,
                dict(result, state="decoded", checksum=self.manifest["checksum"]),
            )
        finally:
            self.sock.close()

    def fail(self, error):
        try:
            _send_json(self.sock, {"state": "failed", "error": str(error)[:500]})
        except OSError:
            pass
        finally:
            self.sock.close()


def receive_handoff(
    path,
    bind_host,
    port,
    *,
    identity,
    model_signature,
    run_id,
    owner,
    world=8,
    timeout=1800,
):
    """Accept once; validate, stream and atomically publish a complete fixture.

    Timeout, partial writes and checksum errors remove staging data. A previous
    completed run is never overwritten. Native CacheStore owns its own protocol;
    this diagnostic endpoint does not use its ports or claim its validation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"PD destination already exists: {path}")
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}.partial-", dir=path.parent))
    sock = None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.settimeout(timeout)
            listener.bind((bind_host, port))
            listener.listen(1)
            print(f"[PD owner {owner}] listening on {bind_host}:{port}", flush=True)
            sock, _ = listener.accept()
        sock.settimeout(timeout)
        header = _recv_json(sock)
        manifest = header["manifest"]
        validate_handoff(
            manifest,
            identity=identity,
            model_signature=model_signature,
            run_id=run_id,
            owner=owner,
            world=world,
        )
        remaining = header["payload_bytes"]
        if not isinstance(remaining, int) or not 0 < remaining <= MAX_PAYLOAD:
            raise ValueError("invalid PD payload length")
        _send_json(sock, {"state": "ready"})
        digest = hashlib.sha256()
        with (staging / "pools.pt").open("wb") as handle:
            while remaining:
                block = _recv_exact(sock, min(CHUNK, remaining))
                handle.write(block)
                digest.update(block)
                remaining -= len(block)
        if digest.hexdigest() != manifest["checksum"]:
            raise ValueError("PD payload checksum mismatch")
        (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
        os.rename(staging, path)
        _send_json(sock, {"state": "stored", "checksum": manifest["checksum"]})
        return HandoffSession(sock, path, manifest)
    except Exception as error:
        if sock is not None:
            HandoffSession(sock, path, {}).fail(error)
        if isinstance(error, socket.timeout):
            raise TimeoutError("PD receive timed out") from error
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
