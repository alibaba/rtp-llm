"""Two-tier remote test result cache: local files + REAPI ActionCache.

Caches test results keyed by (test_nodeid, input_root_hash, gpu_type, gpu_count).
On cache hit, reconstructs the full ExecutionResult (including CAS digest
references to stdout, stderr, and output archives) so that the plugin can
replay the result without dispatching to a remote worker.

Tier 1 (local): JSON manifest file under .pytest_cache/remote_test_cache/.
Tier 2 (ActionCache): REAPI ActionCache service for cross-machine CI sharing.

The manifest is keyed per input_root_hash (one manifest per code state).
Individual test results are sub-keyed within the manifest by a hash of
(test_nodeid, gpu_type, gpu_count).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import remote_execution_pb2 as re_pb2
from .merkle import sha256_digest

log = logging.getLogger(__name__)

MANIFEST_VERSION = 3
CACHE_SENTINEL_CMD = "rtp-test-cache-manifest-v1"


@dataclass
class DigestRef:
    """Lightweight reference to a CAS blob."""

    hash: str
    size_bytes: int

    def to_proto(self) -> re_pb2.Digest:
        return re_pb2.Digest(hash=self.hash, size_bytes=self.size_bytes)

    @classmethod
    def from_proto(cls, d: re_pb2.Digest) -> Optional[DigestRef]:
        if not d or not d.hash:
            return None
        return cls(hash=d.hash, size_bytes=d.size_bytes)

    def to_dict(self) -> dict:
        return {"hash": self.hash, "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional[DigestRef]:
        if not d or not d.get("hash"):
            return None
        return cls(hash=d["hash"], size_bytes=d.get("size_bytes", 0))


@dataclass
class CacheEntry:
    """Cached test result with full artifact digest references."""

    test_nodeid: str
    gpu_type: str
    gpu_count: int
    result: str  # "passed"
    exit_code: int
    timestamp: float
    duration_s: float
    worker_ip: Optional[str] = None
    stdout_digest: Optional[DigestRef] = None
    stderr_digest: Optional[DigestRef] = None
    output_files: Dict[str, DigestRef] = field(default_factory=dict)
    junit_testcase_xml: Optional[str] = None
    session_profile: Optional[str] = None
    session_scope: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict = {
            "test_nodeid": self.test_nodeid,
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "result": self.result,
            "exit_code": self.exit_code,
            "timestamp": self.timestamp,
            "duration_s": self.duration_s,
        }
        if self.worker_ip:
            d["worker_ip"] = self.worker_ip
        if self.stdout_digest:
            d["stdout_digest"] = self.stdout_digest.to_dict()
        if self.stderr_digest:
            d["stderr_digest"] = self.stderr_digest.to_dict()
        if self.output_files:
            d["output_files"] = {k: v.to_dict() for k, v in self.output_files.items()}
        if self.junit_testcase_xml:
            d["junit_testcase_xml"] = self.junit_testcase_xml
        if self.session_profile is not None:
            d["session_profile"] = self.session_profile
        if self.session_scope is not None:
            d["session_scope"] = self.session_scope
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CacheEntry:
        output_files = {}
        for k, v in d.get("output_files", {}).items():
            ref = DigestRef.from_dict(v)
            if ref:
                output_files[k] = ref
        return cls(
            test_nodeid=d["test_nodeid"],
            gpu_type=d.get("gpu_type", ""),
            gpu_count=d.get("gpu_count", 0),
            result=d["result"],
            exit_code=d.get("exit_code", 0),
            timestamp=d.get("timestamp", 0.0),
            duration_s=d.get("duration_s", 0.0),
            worker_ip=d.get("worker_ip"),
            stdout_digest=DigestRef.from_dict(d.get("stdout_digest")),
            stderr_digest=DigestRef.from_dict(d.get("stderr_digest")),
            output_files=output_files,
            junit_testcase_xml=d.get("junit_testcase_xml"),
            session_profile=d.get("session_profile"),
            session_scope=d.get("session_scope"),
        )


def _per_test_key(test_nodeid: str, gpu_type: str, gpu_count: int) -> str:
    raw = f"{test_nodeid}|{gpu_type}|{gpu_count}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _manifest_action_digest(input_root_digest: re_pb2.Digest) -> re_pb2.Digest:
    """Build a deterministic synthetic Action digest for the cache manifest.

    The action encodes a fixed sentinel command + the actual input_root_digest,
    so the resulting digest changes only when code changes.
    """
    cmd = re_pb2.Command(arguments=[CACHE_SENTINEL_CMD])
    cmd_data = cmd.SerializeToString()
    cmd_digest = sha256_digest(cmd_data)

    action = re_pb2.Action(
        command_digest=cmd_digest,
        input_root_digest=input_root_digest,
    )
    action_data = action.SerializeToString()
    return sha256_digest(action_data)


class RemoteTestCache:
    """Two-tier test result cache: local files + REAPI ActionCache."""

    def __init__(
        self,
        cache_dir: Path,
        action_cache_client: Optional[Any] = None,
        cas_client: Optional[Any] = None,
        ttl_days: int = 7,
    ):
        self._cache_dir = cache_dir
        self._ac = action_cache_client
        self._cas = cas_client
        self._ttl_seconds = ttl_days * 86400
        self._lock = threading.Lock()
        self._dirty = False

    # ------------------------------------------------------------------
    # Synthetic Action digest
    # ------------------------------------------------------------------

    def manifest_action_digest(self, input_root_digest: re_pb2.Digest) -> re_pb2.Digest:
        return _manifest_action_digest(input_root_digest)

    # ------------------------------------------------------------------
    # Manifest load: Tier 1 (local) then Tier 2 (ActionCache)
    # ------------------------------------------------------------------

    def load_manifest(self, input_root_digest: re_pb2.Digest) -> dict:
        """Load the test result manifest for this code state.

        Priority: local file -> ActionCache -> empty manifest.
        Returns the manifest dict (mutable, updated in-place during the session).
        """
        root_hash = input_root_digest.hash

        # Tier 1: local file
        manifest = self._load_local(root_hash)
        if manifest and self._is_compatible_manifest(manifest) and manifest.get("tests"):
            log.info(
                "Test cache: loaded %d entries from local cache",
                len(manifest["tests"]),
            )
            return manifest

        # Tier 2: ActionCache
        if self._ac is not None:
            manifest = self._load_from_action_cache(input_root_digest)
            if (
                manifest
                and self._is_compatible_manifest(manifest)
                and manifest.get("tests")
            ):
                log.info(
                    "Test cache: loaded %d entries from ActionCache",
                    len(manifest["tests"]),
                )
                self._save_local(root_hash, manifest)
                return manifest

        # Empty manifest
        return self._empty_manifest(root_hash)

    def _empty_manifest(self, input_root_hash: str) -> dict:
        return {
            "version": MANIFEST_VERSION,
            "input_root_hash": input_root_hash,
            "tests": {},
        }

    def _is_compatible_manifest(self, manifest: dict) -> bool:
        version = manifest.get("version")
        if version != MANIFEST_VERSION:
            log.info(
                "Ignoring remote test cache manifest version %r (expected %d)",
                version,
                MANIFEST_VERSION,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Tier 1: Local file cache
    # ------------------------------------------------------------------

    def _local_path(self, input_root_hash: str) -> Path:
        return self._cache_dir / f"{input_root_hash[:16]}.json"

    def _load_local(self, input_root_hash: str) -> Optional[dict]:
        path = self._local_path(input_root_hash)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version") != MANIFEST_VERSION:
                log.debug("Local cache version mismatch, ignoring")
                return None
            return data
        except Exception as exc:
            log.debug("Failed to load local cache %s: %s", path, exc)
            return None

    def _save_local(self, input_root_hash: str, manifest: dict) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._local_path(input_root_hash)
            path.write_text(
                json.dumps(manifest, separators=(",", ":"), sort_keys=True),
                encoding="utf-8",
            )
        except Exception as exc:
            log.debug("Failed to save local cache: %s", exc)

    # ------------------------------------------------------------------
    # Tier 2: REAPI ActionCache
    # ------------------------------------------------------------------

    def _load_from_action_cache(
        self, input_root_digest: re_pb2.Digest
    ) -> Optional[dict]:
        if self._ac is None:
            return None
        action_digest = self.manifest_action_digest(input_root_digest)
        try:
            result = self._ac.get(action_digest)
        except Exception as exc:
            log.warning("ActionCache.get failed: %s", exc)
            return None
        if result is None:
            return None
        raw = result.stdout_raw
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            log.warning("Failed to parse ActionCache manifest: %s", exc)
            return None

    def _save_to_action_cache(
        self, input_root_digest: re_pb2.Digest, manifest: dict
    ) -> bool:
        if self._ac is None:
            return False
        action_digest = self.manifest_action_digest(input_root_digest)
        raw = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        action_result = re_pb2.ActionResult(
            exit_code=0,
            stdout_raw=raw,
        )
        try:
            return self._ac.update(action_digest, action_result)
        except Exception as exc:
            log.warning("ActionCache.update failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Per-test lookup and store
    # ------------------------------------------------------------------

    def lookup(
        self,
        manifest: dict,
        test_nodeid: str,
        gpu_type: str,
        gpu_count: int,
    ) -> Optional[CacheEntry]:
        """Look up a test in the manifest. Returns CacheEntry or None."""
        key = _per_test_key(test_nodeid, gpu_type, gpu_count)
        entry_dict = manifest.get("tests", {}).get(key)
        if entry_dict is None:
            return None

        if entry_dict.get("result") != "passed":
            return None

        ts = entry_dict.get("timestamp", 0.0)
        if time.time() - ts > self._ttl_seconds:
            log.debug("Cache entry expired for %s", test_nodeid)
            return None

        try:
            return CacheEntry.from_dict(entry_dict)
        except Exception as exc:
            log.debug("Failed to parse cache entry for %s: %s", test_nodeid, exc)
            return None

    def store(
        self,
        manifest: dict,
        entry: CacheEntry,
    ) -> None:
        """Store a test result in the in-memory manifest (thread-safe)."""
        key = _per_test_key(entry.test_nodeid, entry.gpu_type, entry.gpu_count)
        with self._lock:
            manifest.setdefault("tests", {})[key] = entry.to_dict()
            self._dirty = True

    def ensure_stdout_digest(
        self,
        stdout_raw: bytes,
        stdout_digest: Optional[re_pb2.Digest],
    ) -> Optional[DigestRef]:
        """Get or create a CAS digest for stdout.

        If stdout_digest is available, use it. Otherwise upload stdout_raw to CAS.
        """
        if stdout_digest and stdout_digest.hash:
            return DigestRef.from_proto(stdout_digest)
        if stdout_raw and self._cas is not None:
            try:
                d = self._cas.upload_blob(stdout_raw)
                return DigestRef.from_proto(d)
            except Exception as exc:
                log.debug("Failed to upload stdout to CAS for caching: %s", exc)
        return None

    def ensure_stderr_digest(
        self,
        stderr_raw: bytes,
        stderr_digest: Optional[re_pb2.Digest],
    ) -> Optional[DigestRef]:
        """Same as ensure_stdout_digest, for stderr."""
        if stderr_digest and stderr_digest.hash:
            return DigestRef.from_proto(stderr_digest)
        if stderr_raw and self._cas is not None:
            try:
                d = self._cas.upload_blob(stderr_raw)
                return DigestRef.from_proto(d)
            except Exception as exc:
                log.debug("Failed to upload stderr to CAS for caching: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Flush: persist to local + ActionCache
    # ------------------------------------------------------------------

    def flush(self, input_root_digest: re_pb2.Digest, manifest: dict) -> None:
        """Persist the manifest to both tiers. Called once at session end."""
        with self._lock:
            dirty = self._dirty
            self._dirty = False

        if not dirty:
            return
        if not manifest.get("tests"):
            return

        root_hash = input_root_digest.hash
        self._save_local(root_hash, manifest)
        ok = self._save_to_action_cache(input_root_digest, manifest)
        n = len(manifest.get("tests", {}))
        if ok:
            log.info("Test cache: flushed %d entries to ActionCache + local", n)
        else:
            log.info("Test cache: flushed %d entries to local only", n)

    def close(self) -> None:
        """Close underlying gRPC clients."""
        if self._ac is not None:
            self._ac.close()

    # ------------------------------------------------------------------
    # Session mode helpers
    # ------------------------------------------------------------------

    def get_cached_nodeids(self, manifest: dict, items_with_gpu: list) -> List[str]:
        """Return node IDs that are cached (for session --deselect).

        items_with_gpu: list of (nodeid, gpu_type, gpu_count) tuples.
        """
        cached = []
        for nodeid, gpu_type, gpu_count in items_with_gpu:
            entry = self.lookup(manifest, nodeid, gpu_type, gpu_count)
            if entry is not None:
                cached.append(nodeid)
        return cached
