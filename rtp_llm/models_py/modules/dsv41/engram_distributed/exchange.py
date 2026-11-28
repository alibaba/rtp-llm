# SPDX-License-Identifier: Apache-2.0
"""One-shot TCPStore metadata exchange and bounded startup failure propagation."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import traceback
from datetime import timedelta

from .checkpoint import CheckpointError, checkpoint_samples
from .config import build_config
from .manifest import prepare_manifest

MAX_MESSAGE = 4 << 20
RESIDENT_DIR_ENV = "RTP_ENGRAM_RESIDENT_DIR"
RESIDENT_GENERATION_ENV = "RTP_ENGRAM_RESIDENT_GENERATION"

# Only these protocol codes may cross the wire. An arbitrary exception message,
# including one wrapped in ExchangeError by an extension, is never public data.
_WIRE_ERRORS = frozenset(
    """
    MESSAGE_TOO_LARGE INVALID_MESSAGE INVALID_GROUP INVALID_GENERATION
    INVALID_REVISION INVALID_MEMBERS INVALID_MEMBER INVALID_INSTANCE_ID
    INVALID_WORLD_SIZE EVERY_INSTANCE_MUST_HOLD_SHARDS INVALID_SHARD
    INVALID_SHARD_INDEX DUPLICATE_OR_INVALID_HOST INVALID_PUBLISHER_RANK
    INCOMPLETE_SHARD_SET UNKNOWN_INSTANCE INCOMPLETE_INSTANCE_SHARDS
    INVALID_DESCRIPTOR DESCRIPTOR_IDENTITY_MISMATCH COHORT_TIMEOUT
    GROUP_MISMATCH CONFLICTING_REPUBLICATION ASSEMBLY_FAILED ACK_MISMATCH
    COHORT_STORE_UNAVAILABLE INSTANCE_STARTUP_TIMEOUT
    INSTANCE_STORE_WAIT_FAILED_OR_TIMEOUT UNSAFE_CHECKPOINT_PATH
""".split()
)

_MANIFEST_ERRORS = frozenset(
    """
    INVALID_LIMIT INVALID_LAYOUT DUPLICATE_LAYER UNSUPPORTED_LAYOUT
    UNSUPPORTED_BACKEND INVALID_INTEGER INVALID_TEXT INVALID_DIGEST
    INVALID_FIELDS UNSUPPORTED_SCHEMA TENSOR_LIMIT UNSUPPORTED_SHARDING
    INVALID_ALLOCATION_SIZE TOTAL_ALLOCATION_LIMIT GENERATION_MISMATCH
    BACKEND_MISMATCH MODEL_CONTRACT_MISMATCH INCOMPLETE_GENERATION
""".split()
)

_ERROR_CLASSES = frozenset(
    (
        "exchange",
        "manifest",
        "file_not_found",
        "permission",
        "io",
        "import",
        "validation",
        "missing_field",
        "type",
        "internal",
    )
)
_CAUSE_CODES = (
    _WIRE_ERRORS
    | _MANIFEST_ERRORS
    | frozenset(
        (
            "FILE_NOT_FOUND",
            "PERMISSION_DENIED",
            "IO_ERROR",
            "IMPORT_FAILED",
            "VALIDATION_FAILED",
            "MISSING_FIELD",
            "INVALID_TYPE",
            "INTERNAL_ERROR",
            "MANIFEST_VALIDATION_FAILED",
        )
    )
)


def _allowed(value, known, fallback=None):
    return value if type(value) is str and value in known else fallback


def _log(level, message, *args):
    # Startup progress and failure propagation must not depend on log handlers.
    try:
        logging.getLogger(__name__).log(level, message, *args)
    except Exception:
        pass


class ExchangeError(RuntimeError):
    """Public protocol code plus optional, bounded diagnostic categories."""

    def __init__(self, code, *, error_class=None, cause_code=None):
        self.code = code
        self.error_class = _allowed(error_class, _ERROR_CLASSES)
        self.cause_code = _allowed(cause_code, _CAUSE_CODES)
        super().__init__(code)


def classify_failure(error):
    """Classify a startup exception without exporting arbitrary exception text."""
    from .manifest import ManifestError

    if isinstance(error, ExchangeError):
        category, code = "exchange", _allowed(
            error.code, _WIRE_ERRORS, "INTERNAL_ERROR"
        )
    elif isinstance(error, ManifestError):
        category, code = "manifest", _allowed(
            error.code, _MANIFEST_ERRORS, "MANIFEST_VALIDATION_FAILED"
        )
    elif isinstance(error, CheckpointError):
        category, code = "validation", _allowed(
            error.code, _WIRE_ERRORS, "VALIDATION_FAILED"
        )
    else:
        # Match subclasses before their base classes; never inspect error text,
        # filenames, custom class names or descriptor values.
        category, code = "internal", "INTERNAL_ERROR"
        for kind, category_name, cause in (
            (FileNotFoundError, "file_not_found", "FILE_NOT_FOUND"),
            (PermissionError, "permission", "PERMISSION_DENIED"),
            (OSError, "io", "IO_ERROR"),
            (ImportError, "import", "IMPORT_FAILED"),
            (ValueError, "validation", "VALIDATION_FAILED"),
            (KeyError, "missing_field", "MISSING_FIELD"),
            (TypeError, "type", "INVALID_TYPE"),
        ):
            if isinstance(error, kind):
                category, code = category_name, cause
                break
    return ExchangeError("ASSEMBLY_FAILED", error_class=category, cause_code=code)


def encode(value):
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if len(data) > MAX_MESSAGE:
        raise ExchangeError("MESSAGE_TOO_LARGE")
    return data


def decode(data):
    if len(data) > MAX_MESSAGE:
        raise ExchangeError("MESSAGE_TOO_LARGE")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ExchangeError("INVALID_MESSAGE")
            result[key] = value
        return result

    def invalid(_):
        raise ExchangeError("INVALID_MESSAGE")

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
        if type(value) is not dict:
            raise ExchangeError("INVALID_MESSAGE")
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise ExchangeError("INVALID_MESSAGE") from None


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_submission(group, instance_id, documents):
    members = {m["instance_id"]: m for m in group["members"]}
    if instance_id not in members or type(documents) is not list:
        raise ExchangeError("UNKNOWN_INSTANCE")
    expected = {s["index"]: s["host_id"] for s in members[instance_id]["shards"]}
    if len(documents) != len(expected):
        raise ExchangeError("INCOMPLETE_INSTANCE_SHARDS")
    seen = set()
    for doc in documents:
        if type(doc) is not dict or type(doc.get("shard")) is not int:
            raise ExchangeError("INVALID_DESCRIPTOR")
        index = doc["shard"]
        if (
            index in seen
            or index not in expected
            or doc.get("host_id") != expected[index]
            or doc.get("generation") != group["generation"]
            or doc.get("model_revision") != group["model_revision"]
        ):
            raise ExchangeError("DESCRIPTOR_IDENTITY_MISMATCH")
        seen.add(index)


_FAILURE_BYTES = 512


_FAILURE_POLL_SECONDS = 0.05


_STARTUP_PHASES = frozenset(
    (
        "config",
        "publish",
        "descriptors",
        "assembly",
        "verify",
        "package",
        "stage",
        "ack",
        "cleanup",
    )
)


class _StartupPeerFailure(ExchangeError):
    """A bounded diagnostic from another rank in this exact startup attempt."""

    def __init__(self, rank, phase, *, error_class, cause_code):
        super().__init__(
            "INSTANCE_STARTUP_FAILED", error_class=error_class, cause_code=cause_code
        )
        self.rank, self.phase = rank, phase

    def __str__(self):
        return (
            f"{self.code} rank={self.rank} phase={self.phase} "
            f"error_class={self.error_class} cause_code={self.cause_code}"
        )


def _publish_failure(store, key, rank, phase, error):
    """First failure wins; reporting must never replace the original exception."""
    try:
        failure = classify_failure(error)
        # Preserve an already classified remote failure instead of reducing it
        # to ASSEMBLY_FAILED and losing the coordinator's safe cause.
        if isinstance(error, ExchangeError) and error.error_class is not None:
            classified = ExchangeError(
                failure.code,
                error_class=error.error_class,
                cause_code=error.cause_code,
            )
            if classified.error_class is not None:
                failure = classified
        payload = encode(
            dict(
                rank=rank,
                phase=phase,
                error_class=failure.error_class,
                cause_code=failure.cause_code,
            )
        )
        if len(payload) > _FAILURE_BYTES:
            return
        current = store.compare_set(key, b"", payload)
        logging.getLogger(__name__).error(
            "engram event=startup_failed rank=%d phase=%s error_class=%s "
            "cause_code=%s first_failure=%s",
            rank,
            phase,
            failure.error_class,
            failure.cause_code,
            current == payload,
        )
    except Exception:
        # A disconnected Store cannot carry the notification. Keep the
        # publisher's original traceback; peers still have their deadline.
        pass


def _raise_peer_failure(store, key):
    if not store.check([key]):
        return
    payload = store.get(key)  # The terminal key is retained until Store teardown.
    if len(payload) > _FAILURE_BYTES:
        raise ExchangeError("INVALID_INSTANCE_FAILURE")
    value = decode(payload)
    if (
        type(value) is not dict
        or set(value) != {"rank", "phase", "error_class", "cause_code"}
        or type(value["rank"]) is not int
        or not 0 <= value["rank"] < 2**31
        or type(value["phase"]) is not str
        or value["phase"] not in _STARTUP_PHASES
    ):
        raise ExchangeError("INVALID_INSTANCE_FAILURE")
    raise _StartupPeerFailure(**value)


def _wait(store, keys, deadline, *, failure_key=None):
    # TCPStore.wait is a key-availability operation, not a rank collective.
    # Use its per-call timeout: get() otherwise inherits the Store's (possibly
    # very long) timeout. Never mutate the shared Store's global timeout.
    if failure_key is not None:
        # Store.check is a non-waiting key-availability RPC. Poll only during
        # startup: short TCPStore.wait timeouts would emit repeated C++ warning
        # logs while a publisher legitimately loads a large checkpoint. As with
        # set/add, a failed Store connection is governed by its transport timeout;
        # this loop does not mutate that shared timeout or interrupt an RPC.
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ExchangeError("INSTANCE_STARTUP_TIMEOUT")
            try:
                _raise_peer_failure(store, failure_key)
                if store.check(keys):
                    _raise_peer_failure(store, failure_key)
                    if time.monotonic() >= deadline:
                        raise ExchangeError("INSTANCE_STARTUP_TIMEOUT")
                    return
            except ExchangeError:
                raise
            except RuntimeError:
                raise ExchangeError("INSTANCE_STORE_WAIT_FAILED_OR_TIMEOUT") from None
            time.sleep(min(_FAILURE_POLL_SECONDS, max(0, deadline - time.monotonic())))
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ExchangeError("INSTANCE_STARTUP_TIMEOUT")
    try:
        store.wait(keys, timedelta(seconds=remaining))
    except RuntimeError:
        raise ExchangeError("INSTANCE_STORE_WAIT_FAILED_OR_TIMEOUT") from None


def _open_store(group, instance_id, address, deadline, *, group_digest):
    """Only cohort representatives join this temporary Store; no process group."""
    from torch.distributed import PrefixStore, TCPStore

    coordinator = group["members"][0]["instance_id"]
    if instance_id not in {m["instance_id"] for m in group["members"]}:
        raise ExchangeError("UNKNOWN_INSTANCE")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ExchangeError("COHORT_TIMEOUT")
    try:
        store = TCPStore(
            address[0],
            address[1],
            world_size=None,
            is_master=instance_id == coordinator,
            wait_for_workers=False,
            timeout=timedelta(seconds=remaining),
        )
    except RuntimeError:
        raise ExchangeError("COHORT_STORE_UNAVAILABLE") from None
    if instance_id == coordinator:
        store.set("engram/group", group_digest)
    _wait(store, ["engram/group"], deadline)
    if store.get("engram/group") != group_digest.encode():
        raise ExchangeError("GROUP_MISMATCH")
    # An extra/restarted representative cannot consume the previous attempt.
    # The Store itself is new for each complete cohort startup.
    attempt = store.add(f"engram/start_count/{instance_id}", 1)
    return PrefixStore(f"engram/{group['generation']}/attempt/{attempt}/", store)


def _get_exchange_store(config, world_store, attempt, deadline):
    """Give host representatives one metadata channel, independent of topology."""
    group = config["group"]
    if config["reuse_world_store"]:
        from torch.distributed import PrefixStore

        if world_store is None:
            raise ExchangeError("INSTANCE_STORE_REQUIRED")
        _log(logging.INFO, "engram event=exchange_store source=world")
        return PrefixStore(
            f"engram/{group['generation']}/attempt/{attempt}", world_store
        )
    address = config["coordinator"]
    if (
        type(address) is not dict
        or set(address) != {"host", "port"}
        or type(address["port"]) is not int
        or not 1 <= address["port"] <= 65535
    ):
        raise ExchangeError("INVALID_COORDINATOR")
    _log(
        logging.INFO,
        "engram event=exchange_store source=cohort instance=%s coordinator=%s:%d members=%d",
        config["instance_id"],
        address["host"],
        address["port"],
        len(group["members"]),
    )
    return _open_store(
        group,
        config["instance_id"],
        (address["host"], address["port"]),
        deadline,
        group_digest=config["group_digest"],
    )


def _exchange_metadata(
    store, group, instance_id, documents, deadline, build, prepare, *, group_digest
):
    """Publish descriptors, verify the assembled package, then release readers."""
    members = [member["instance_id"] for member in group["members"]]
    coordinator = instance_id == members[0]
    if instance_id not in members:
        raise ExchangeError("UNKNOWN_INSTANCE")
    index = members.index(instance_id)
    phase = "config"
    failed = False
    try:
        validate_submission(group, instance_id, documents)
        submission = encode(dict(group_digest=group_digest, documents=documents))
        key = "member/" + instance_id
        if store.compare_set(key, b"", submission) != submission:
            raise ExchangeError("CONFLICTING_REPUBLICATION")
        phase = "descriptors"
        if coordinator:
            keys = ["member/" + member for member in members]
            _wait(store, keys, deadline, failure_key="failure")
            all_documents = []
            for member, key in zip(members, keys):
                value = decode(store.get(key))
                if type(value) is not dict or set(value) != {
                    "group_digest",
                    "documents",
                }:
                    raise ExchangeError("INVALID_MESSAGE")
                if value["group_digest"] != group_digest:
                    raise ExchangeError("GROUP_MISMATCH")
                validate_submission(group, member, value["documents"])
                all_documents.extend(value["documents"])
            phase = "assembly"
            package, prepared = build(all_documents)
            payload = encode(dict(group_digest=group_digest, package=package))
            store.set("package", payload)
            _log(
                logging.INFO,
                "engram event=package_assembled instances=%d bytes=%d",
                len(members),
                len(payload),
            )
        else:
            phase = "package"
            _wait(store, ["package"], deadline, failure_key="failure")
            payload = store.get("package")
            value = decode(payload)
            if (
                set(value) != {"group_digest", "package"}
                or value["group_digest"] != group_digest
            ):
                raise ExchangeError("GROUP_MISMATCH")
            phase = "verify"
            package = value["package"]
            prepared = prepare(package)
        receipt = digest(payload).encode()
        store.set("verified/" + instance_id, receipt)
        phase = "ack"
        if coordinator:
            keys = ["verified/" + member for member in members]
            _wait(store, keys, deadline, failure_key="failure")
            if any(store.get(key) != receipt for key in keys):
                raise ExchangeError("ACK_MISMATCH")
            store.set("ready", b"1")
        _wait(store, ["ready"], deadline, failure_key="failure")
        return package, prepared
    except BaseException as error:
        failed = True
        _publish_failure(store, "failure", index, phase, error)
        raise
    finally:
        # Keep the coordinator alive until every reader has observed READY or
        # the terminal failure. On failure, absent peers cannot delay teardown.
        try:
            store.set("departed/" + instance_id, b"1")
            if coordinator:
                keys = ["departed/" + member for member in members]
                if failed:
                    until = min(deadline, time.monotonic() + 1)
                    while time.monotonic() < until and not store.check(keys):
                        time.sleep(0.01)
                else:
                    _wait(store, keys, deadline, failure_key="failure")
                    # Every representative has read READY. Reused Stores must
                    # not retain capabilities or per-member startup metadata.
                    for key in (
                        ["package", "ready"]
                        + keys
                        + ["member/" + member for member in members]
                        + ["verified/" + member for member in members]
                    ):
                        store.delete_key(key)
        except Exception:
            if not failed:
                raise


def _exchange_group(config, documents, deadline, *, world_store, attempt):
    """The same exchange and lifetime for reused and temporary TCPStores."""
    store = None
    try:
        store = _get_exchange_store(config, world_store, attempt, deadline)
        group = config["group"]
        return _exchange_metadata(
            store,
            group,
            config["instance_id"],
            documents,
            deadline,
            lambda docs: _assemble_package(group, docs, config["checkpoint"]),
            lambda package: _prepare_package(group, package),
            group_digest=config["group_digest"],
        )
    except BaseException as error:
        # Finished helper frames otherwise retain the Store through the error
        # traceback and keep the listener open. Preserve stack locations, release
        # frame locals, and let the original error reach the caller.
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        # Release only our reference. A reused Store is still owned by the
        # DistributedServer; a temporary Store releases its listener here.
        del store


def _parse_publications(group, documents):
    if not isinstance(documents, list) or any(
        not isinstance(d, dict) for d in documents
    ):
        raise ExchangeError("INVALID_PACKAGE")
    expected = {s["index"]: s["host_id"] for m in group["members"] for s in m["shards"]}
    if any(
        type(doc.get("shard")) is not int
        or doc["shard"] not in expected
        or doc.get("host_id") != expected[doc["shard"]]
        for doc in documents
    ):
        raise ExchangeError("PACKAGE_HOST_MISMATCH")
    manifest, handles = prepare_manifest(
        documents, len(expected), group["generation"], group["model_revision"], None
    )
    return dict(
        manifest=manifest,
        generation=group["generation"],
        model_revision=group["model_revision"],
        capabilities=handles,
    )


def _assemble_package(group, documents, checkpoint, *, sample_rows=checkpoint_samples):
    prepared = _parse_publications(group, documents)
    samples = sample_rows(checkpoint, prepared["manifest"].manifest)
    prepared["samples"] = samples
    return dict(documents=documents, samples=samples), prepared


def _prepare_package(group, package):
    """Validate and decode the host publications in a received package."""
    if type(package) is not dict or set(package) != {"documents", "samples"}:
        raise ExchangeError("INVALID_PACKAGE")
    prepared = _parse_publications(group, package["documents"])
    samples = package["samples"]
    if not isinstance(samples, dict) or set(samples) != {
        tensor.name for tensor in prepared["manifest"].manifest.tensors
    }:
        raise ExchangeError("PACKAGE_SAMPLES_MISMATCH")
    prepared["samples"] = samples
    return prepared


class EngramStartup:
    """Stage the Engram package for one RTP rank.

    Worker-owned Fabric backing uses a world Store borrowed from DistributedServer.
    Resident Fabric backing is discovered independently of the worker cohort.
    """

    def __init__(
        self,
        store,
        *,
        rank,
        world_size,
        publish,
        install,
        environ=None,
        gang_hosts=None,
        checkpoint=None,
        local_device=0,
    ):
        self._started = time.monotonic()
        self._env = os.environ if environ is None else environ
        from .config import StorageMode, StoragePolicy

        self._resident = None
        self._local_startup = None
        if any(
            name in self._env
            for name in (
                RESIDENT_DIR_ENV,
                RESIDENT_GENERATION_ENV,
                "RTP_ENGRAM_RESIDENT_TRANSPORT",
            )
        ):
            from .resident import ResidentGeneration

            self._resident = ResidentGeneration(self._env, checkpoint=checkpoint)
            self._install = install
            return
        if StoragePolicy.from_environ(self._env).mode == StorageMode.CUDA_LOCAL:
            from .local import LocalCudaStartup

            self._local_startup = LocalCudaStartup(
                store,
                rank=rank,
                world_size=world_size,
                device=local_device,
                checkpoint=checkpoint,
                gang_hosts=gang_hosts,
                environ=self._env,
                install=install,
            )
            self._config = None
            return
        self._config = build_config(
            self._env,
            world_size,
            rank=rank,
            gang_hosts=gang_hosts,
            checkpoint=checkpoint,
            local_device=local_device,
        )
        self._store = store
        self._rank = rank
        self._world_size = world_size
        self._publish = publish
        self._install = install
        self._phase = "config"

    def run(self):
        """Publish, exchange and install once, before engine construction."""
        if self._resident is not None:
            self._install(self._resident.load(), self._resident.options)
            return None
        if self._local_startup is not None:
            return self._local_startup.run()
        if self._config is None:
            return None
        config = self._config
        self._group = config["group"]
        group_digest = config["group_digest"]
        members = {m["instance_id"]: m for m in self._group["members"]}
        self._member = members[config["instance_id"]]
        self._local_world_size = self._member["world_size"]
        self._local_rank = self._rank - config["rank_base"]
        if not 0 <= self._local_rank < self._local_world_size:
            raise ExchangeError("INSTANCE_WORLD_SIZE_MISMATCH")
        self._options = config["options"]
        self._shard_count = self._options.shards.shard_count
        self._deadline = time.monotonic() + config["timeout_seconds"]
        if self._store is None:
            if self._local_world_size != 1:
                raise ExchangeError("INSTANCE_STORE_REQUIRED")
            from torch.distributed import HashStore

            self._store = HashStore()
        self._attempt = self._store.add(f"engram/start_count/{self._rank}", 1)
        self._leader = self._local_rank == 0
        attempt_prefix = f'engram/{self._group["generation"]}/attempt/{self._attempt}/'
        self._prefix = attempt_prefix + config["instance_id"] + "/"
        self._failure_key = attempt_prefix + "failure"
        self._config_keys = [
            self._prefix + f"config/{i}" for i in range(self._local_world_size)
        ]
        logging.getLogger(__name__).info(
            "engram startup: rank=%d world=%d instance=%s local_rank=%d p=%d "
            "leader=%s attempt=%d timeout=%.0fs",
            self._rank,
            self._world_size,
            config["instance_id"],
            self._local_rank,
            self._shard_count,
            self._leader,
            self._attempt,
            config["timeout_seconds"],
        )
        try:
            self._store.set(
                self._prefix + f"config/{self._local_rank}", group_digest.encode()
            )
            if self._leader:
                logging.getLogger(__name__).info(
                    "engram startup: waiting for %d group digests",
                    self._local_world_size,
                )
                self._wait(self._config_keys)
                if any(
                    self._store.get(k) != group_digest.encode()
                    for k in self._config_keys
                ):
                    raise ExchangeError("INSTANCE_GROUP_CONFIG_MISMATCH")
            self._phase = "publish"
            documents = self._publish_shards() if self._leader else None
            self._exchange_and_install(documents)
            self._acknowledge()
        except BaseException as error:
            # A failure on a separate cohort Store must reach local waiters.
            if not isinstance(error, _StartupPeerFailure) or (
                self._phase == "assembly" and not config["reuse_world_store"]
            ):
                _publish_failure(
                    self._store, self._failure_key, self._rank, self._phase, error
                )
            raise
        logging.getLogger(__name__).info(
            "engram event=control_plane_ready phase=exchange duration_s=%.3f",
            time.monotonic() - self._started,
        )
        return None

    def _wait(self, keys):
        _wait(self._store, keys, self._deadline, failure_key=self._failure_key)

    def _publish_shards(self):
        config = self._config
        memory_kind = self._options.policy.memory_kind
        documents = []
        for shard in self._member["shards"]:
            if memory_kind == "host" and self._env.get("RTP_ENGRAM_LOCAL_NUMA") is None:
                logging.getLogger(__name__).warning(
                    "RTP_ENGRAM_LOCAL_NUMA unset; host backing defaults to NUMA %s",
                    config["local_numa"],
                )
            logging.getLogger(__name__).info(
                "engram startup: publishing shard %d of %d on %s (device %d, NUMA %s) memory=%s gpu_devices=%s",
                shard["index"] + 1,
                self._shard_count,
                shard["host_id"],
                config["local_device"],
                config["local_numa"],
                memory_kind,
                config["gpu_devices"] if memory_kind == "gpu" else (),
            )
            started = time.monotonic()
            document = self._publish(
                config["checkpoint"],
                self._group["model_revision"],
                shard["host_id"],
                self._group["generation"],
                shard["index"],
                self._shard_count,
                device=config["local_device"],
                numa=config["local_numa"],
                memory_kind=memory_kind,
                gpu_devices=config["gpu_devices"],
                gpu_reserve_bytes=config["gpu_reserve_bytes"],
                release_input_page_cache=config["release_page_cache"],
                deadline=self._deadline,
                max_allocation_bytes=config["max_allocation_bytes"],
                budget_bytes=config["budget_bytes"],
            )
            logging.getLogger(__name__).info(
                "engram startup: published shard %d in %.1fs",
                shard["index"],
                time.monotonic() - started,
            )
            documents.append(document)
        return documents

    def _exchange_and_install(self, documents):
        if self._leader:
            self._phase = "assembly"
            package, prepared = _exchange_group(
                self._config,
                documents,
                self._deadline,
                world_store=self._store,
                attempt=self._attempt,
            )
            payload = encode(package)
            self._store.set(self._prefix + "package", payload)
            logging.getLogger(__name__).info(
                "engram startup: package assembled (%d bytes, %d shards)",
                len(payload),
                len(documents),
            )
        else:
            self._phase = "package"
            self._wait([self._prefix + "package"])
            package = decode(self._store.get(self._prefix + "package"))
            self._phase = "verify"
            prepared = _prepare_package(self._group, package)
        self._phase = "stage"
        self._install(prepared, self._options)
        logging.getLogger(__name__).info(
            "engram startup: package staged in %.1fs", time.monotonic() - self._started
        )

    def _acknowledge(self):
        self._phase = "ack"
        received = [
            self._prefix + f"received/{i}" for i in range(self._local_world_size)
        ]
        self._store.set(self._prefix + f"received/{self._local_rank}", b"1")
        if self._leader:
            self._wait(received)
            self._phase = "cleanup"
            # Every worker holds its package; serving performs no Store reads.
            for key in [self._prefix + "package"] + self._config_keys + received:
                self._store.delete_key(key)
