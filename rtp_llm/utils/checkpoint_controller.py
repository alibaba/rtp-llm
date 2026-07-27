"""External CUDA process-checkpoint transaction controller.

Level-3 sleep must be driven by a process other than the CUDA target.  This
module keeps a durable, node-local transaction manifest while driving the CUDA
``cuCheckpointProcess*`` state machine::

    RUNNING -> LOCKED -> CHECKPOINTED -> LOCKED -> RUNNING

The manifest is written before the first mutating driver call and after every
successful transition.  A coordinator that dies between a driver call and the
following write can therefore reconcile the manifest with the driver and retry
the same operation safely.
"""

import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

CUDA_SUCCESS = 0

DRIVER_STATE_RUNNING = 0
DRIVER_STATE_LOCKED = 1
DRIVER_STATE_CHECKPOINTED = 2
DRIVER_STATE_FAILED = 3

_DRIVER_STATE_NAMES = {
    DRIVER_STATE_RUNNING: "RUNNING",
    DRIVER_STATE_LOCKED: "LOCKED",
    DRIVER_STATE_CHECKPOINTED: "CHECKPOINTED",
    DRIVER_STATE_FAILED: "FAILED",
}

_MANIFEST_VERSION = 1


class ProcessState(str, Enum):
    RUNNING = "RUNNING"
    LOCKED = "LOCKED"
    CHECKPOINTED = "CHECKPOINTED"
    RESTORED = "RESTORED"
    UNLOCKED = "UNLOCKED"


@dataclass(frozen=True)
class CheckpointTarget:
    pid: int
    rank: int
    address: str
    expected_starttime: Optional[int] = None


@dataclass
class ProcessRecord:
    pid: int
    starttime: int
    rank: int
    address: str
    state: ProcessState


@dataclass(frozen=True)
class ProcessRecoveryStatus:
    pid: int
    starttime: int
    rank: int
    address: str
    state: ProcessState
    identity_valid: bool
    driver_state: Optional[str]
    error: Optional[str]


@dataclass(frozen=True)
class RecoveryStatus:
    epoch: Optional[str]
    phase: str
    manifest_exists: bool
    recovery_required: bool
    checkpoint_complete: bool
    restore_complete: bool
    processes: Tuple[ProcessRecoveryStatus, ...]
    last_error: Optional[str]


@dataclass
class _Manifest:
    epoch: str
    processes: List[ProcessRecord]
    recovery_required: bool = False
    last_error: Optional[str] = None
    # Level-3 multicast keeper durability. The keeper holder owns every multicast
    # FD kept open across checkpoint; if it exits/changes, wake is unrecoverable,
    # so the holder identity (and its rank team) is persisted and verified.
    holder_instance: Optional[str] = None
    team: Optional[str] = None


class CheckpointError(RuntimeError):
    """Base error for checkpoint driver and transaction failures."""


class ManifestError(CheckpointError):
    """The durable transaction manifest is absent or malformed unexpectedly."""


class StaleProcessError(CheckpointError):
    """A PID no longer identifies the process recorded in the manifest."""


class CheckpointTransactionError(CheckpointError):
    """A transaction failed, with its durable status attached for the caller."""

    def __init__(self, message: str, status: Optional[RecoveryStatus] = None):
        super().__init__(message)
        self.status = status


class RecoveryRequiredError(CheckpointTransactionError):
    """The transaction cannot proceed until restore recovery is completed."""


class CudaCheckpointDriver(Protocol):
    def get_state(self, pid: int) -> int: ...

    def lock(self, pid: int, timeout_ms: int) -> int: ...

    def checkpoint(self, pid: int) -> int: ...

    def restore(self, pid: int) -> int: ...

    def unlock(self, pid: int) -> int: ...

    def error_string(self, result: int) -> str: ...


class _CUcheckpointLockArgs(ctypes.Structure):
    _fields_ = [
        ("timeoutMs", ctypes.c_uint),
        ("reserved0", ctypes.c_uint),
        ("reserved1", ctypes.c_uint64 * 7),
    ]


class _CUcheckpointCheckpointArgs(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_uint64 * 8)]


class _CUcheckpointRestoreArgs(ctypes.Structure):
    # CUDA 13 CUcheckpointRestoreArgs. gpuPairs is optional and remains NULL.
    _fields_ = [
        ("gpuPairs", ctypes.c_void_p),
        ("gpuPairsCount", ctypes.c_uint),
        ("reserved", ctypes.c_char * (52 - ctypes.sizeof(ctypes.c_void_p))),
        ("reserved1", ctypes.c_uint64),
    ]


class _CUcheckpointUnlockArgs(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_uint64 * 8)]


class LibCudaCheckpointDriver:
    """ctypes binding for the CUDA process-checkpoint driver API."""

    def __init__(self, lib_name: str = "libcuda.so.1"):
        try:
            self._lib = ctypes.CDLL(lib_name)
            self._configure_signatures()
        except (AttributeError, OSError) as error:
            raise CheckpointError(
                f"CUDA process checkpoint API is unavailable: {error}"
            ) from error

        result = self._lib.cuInit(0)
        if result != CUDA_SUCCESS:
            raise CheckpointError(f"cuInit failed: {self.error_string(result)}")

    def _configure_signatures(self) -> None:
        self._lib.cuInit.argtypes = [ctypes.c_uint]
        self._lib.cuInit.restype = ctypes.c_int

        self._lib.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self._lib.cuGetErrorString.restype = ctypes.c_int

        self._lib.cuCheckpointProcessGetState.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.cuCheckpointProcessGetState.restype = ctypes.c_int

        self._lib.cuCheckpointProcessLock.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(_CUcheckpointLockArgs),
        ]
        self._lib.cuCheckpointProcessLock.restype = ctypes.c_int

        self._lib.cuCheckpointProcessCheckpoint.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(_CUcheckpointCheckpointArgs),
        ]
        self._lib.cuCheckpointProcessCheckpoint.restype = ctypes.c_int

        self._lib.cuCheckpointProcessRestore.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(_CUcheckpointRestoreArgs),
        ]
        self._lib.cuCheckpointProcessRestore.restype = ctypes.c_int

        self._lib.cuCheckpointProcessUnlock.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(_CUcheckpointUnlockArgs),
        ]
        self._lib.cuCheckpointProcessUnlock.restype = ctypes.c_int

    def error_string(self, result: int) -> str:
        message = ctypes.c_char_p()
        error_result = self._lib.cuGetErrorString(result, ctypes.byref(message))
        if error_result == CUDA_SUCCESS and message.value:
            return f"{result} ({message.value.decode('utf-8', 'replace')})"
        return str(result)

    def get_state(self, pid: int) -> int:
        state = ctypes.c_int(-1)
        result = self._lib.cuCheckpointProcessGetState(pid, ctypes.byref(state))
        if result != CUDA_SUCCESS:
            raise CheckpointError(
                f"GetState pid {pid} failed: {self.error_string(result)}"
            )
        return state.value

    def lock(self, pid: int, timeout_ms: int) -> int:
        args = _CUcheckpointLockArgs()
        args.timeoutMs = timeout_ms
        return self._lib.cuCheckpointProcessLock(pid, ctypes.byref(args))

    def checkpoint(self, pid: int) -> int:
        args = _CUcheckpointCheckpointArgs()
        return self._lib.cuCheckpointProcessCheckpoint(pid, ctypes.byref(args))

    def restore(self, pid: int) -> int:
        args = _CUcheckpointRestoreArgs()
        return self._lib.cuCheckpointProcessRestore(pid, ctypes.byref(args))

    def unlock(self, pid: int) -> int:
        args = _CUcheckpointUnlockArgs()
        return self._lib.cuCheckpointProcessUnlock(pid, ctypes.byref(args))


def read_process_starttime(pid: int, proc_root: str = "/proc") -> int:
    """Read field 22 from /proc/<pid>/stat without misparsing spaces in comm."""

    if pid <= 0:
        raise StaleProcessError(f"invalid pid {pid}")
    stat_path = os.path.join(proc_root, str(pid), "stat")
    try:
        with open(stat_path, "r", encoding="utf-8") as stat_file:
            stat = stat_file.read()
    except (FileNotFoundError, ProcessLookupError) as error:
        raise StaleProcessError(f"pid {pid} no longer exists") from error
    except OSError as error:
        raise StaleProcessError(f"cannot inspect pid {pid}: {error}") from error

    comm_end = stat.rfind(")")
    if comm_end < 0:
        raise StaleProcessError(f"malformed {stat_path}")
    fields_after_comm = stat[comm_end + 1 :].split()
    if len(fields_after_comm) <= 19:
        raise StaleProcessError(f"malformed {stat_path}")
    try:
        return int(fields_after_comm[19])
    except ValueError as error:
        raise StaleProcessError(f"malformed starttime in {stat_path}") from error


def checkpoint_manifest_path(
    control_addresses: Sequence[str],
    directory: Optional[str] = None,
    namespace: Optional[str] = None,
) -> str:
    """Return a stable node-local manifest path for a backend instance.

    ``namespace`` scopes the manifest to a specific (instance_generation, role)
    so a shared node cannot reuse a stale manifest across instance generations or
    PD roles that happen to reuse control addresses. It is additive: omitting it
    reproduces the legacy address-only path (backward compatible with L1/L2).
    """

    addresses = sorted(str(address) for address in control_addresses)
    if not addresses or any(not address for address in addresses):
        raise ValueError("control_addresses must contain non-empty addresses")
    key_material = "|".join(addresses)
    if namespace:
        key_material = f"{namespace}|{key_material}"
    key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:20]
    return os.path.join(
        directory or tempfile.gettempdir(), f"rtp_llm_cuda_checkpoint_{key}.json"
    )


class _ManifestStore:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.lock_path = f"{self.path}.lock"
        self.directory = os.path.dirname(self.path)

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        os.makedirs(self.directory, mode=0o700, exist_ok=True)
        lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def read(self) -> Optional[_Manifest]:
        try:
            with open(self.path, "r", encoding="utf-8") as manifest_file:
                payload = json.load(manifest_file)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as error:
            raise ManifestError(f"cannot read checkpoint manifest {self.path}: {error}")
        return self._decode(payload)

    def write(self, manifest: _Manifest) -> None:
        payload = self._encode(manifest)
        temp_fd, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(self.path)}.", dir=self.directory
        )
        try:
            os.fchmod(temp_fd, 0o600)
            with os.fdopen(temp_fd, "w", encoding="utf-8") as temp_file:
                temp_fd = -1
                json.dump(payload, temp_file, sort_keys=True, separators=(",", ":"))
                temp_file.write("\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, self.path)
            self._fsync_directory()
        finally:
            if temp_fd >= 0:
                os.close(temp_fd)
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

    def clear(self) -> None:
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            return
        self._fsync_directory()

    def _fsync_directory(self) -> None:
        directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @staticmethod
    def _encode(manifest: _Manifest) -> Dict[str, object]:
        return {
            "version": _MANIFEST_VERSION,
            "epoch": manifest.epoch,
            "recovery_required": manifest.recovery_required,
            "last_error": manifest.last_error,
            "holder_instance": manifest.holder_instance,
            "team": manifest.team,
            "processes": [
                {
                    "pid": record.pid,
                    "starttime": record.starttime,
                    "rank": record.rank,
                    "address": record.address,
                    "state": record.state.value,
                }
                for record in manifest.processes
            ],
        }

    @staticmethod
    def _decode(payload: object) -> _Manifest:
        if not isinstance(payload, dict) or payload.get("version") != _MANIFEST_VERSION:
            raise ManifestError("invalid checkpoint manifest version")
        epoch = payload.get("epoch")
        recovery_required = payload.get("recovery_required")
        last_error = payload.get("last_error")
        holder_instance = payload.get("holder_instance")
        team = payload.get("team")
        raw_processes = payload.get("processes")
        if not isinstance(epoch, str) or not epoch:
            raise ManifestError("checkpoint manifest has an invalid epoch")
        if not isinstance(recovery_required, bool):
            raise ManifestError("checkpoint manifest has an invalid recovery flag")
        if last_error is not None and not isinstance(last_error, str):
            raise ManifestError("checkpoint manifest has an invalid error")
        if holder_instance is not None and not isinstance(holder_instance, str):
            raise ManifestError("checkpoint manifest has an invalid holder instance")
        if team is not None and not isinstance(team, str):
            raise ManifestError("checkpoint manifest has an invalid keeper team")
        if not isinstance(raw_processes, list) or not raw_processes:
            raise ManifestError("checkpoint manifest has no process records")

        records: List[ProcessRecord] = []
        seen_pids = set()
        seen_ranks = set()
        for raw_record in raw_processes:
            if not isinstance(raw_record, dict):
                raise ManifestError("checkpoint manifest has an invalid process record")
            pid = raw_record.get("pid")
            starttime = raw_record.get("starttime")
            rank = raw_record.get("rank")
            address = raw_record.get("address")
            if (
                not isinstance(pid, int)
                or isinstance(pid, bool)
                or pid <= 0
                or not isinstance(starttime, int)
                or isinstance(starttime, bool)
                or starttime < 0
                or not isinstance(rank, int)
                or isinstance(rank, bool)
                or rank < 0
                or not isinstance(address, str)
                or not address
            ):
                raise ManifestError("checkpoint manifest has invalid process identity")
            try:
                state = ProcessState(raw_record.get("state"))
            except (TypeError, ValueError) as error:
                raise ManifestError(
                    "checkpoint manifest has an invalid process state"
                ) from error
            if pid in seen_pids or rank in seen_ranks:
                raise ManifestError("checkpoint manifest has duplicate pid or rank")
            seen_pids.add(pid)
            seen_ranks.add(rank)
            records.append(ProcessRecord(pid, starttime, rank, address, state))
        return _Manifest(
            epoch,
            records,
            recovery_required,
            last_error,
            holder_instance,
            team,
        )


class CheckpointController:
    """Durable, idempotent orchestration for a set of local CUDA processes."""

    def __init__(
        self,
        manifest_path: str,
        driver: Optional[CudaCheckpointDriver] = None,
        lock_timeout_ms: int = 60000,
        starttime_reader: Callable[[int], int] = read_process_starttime,
    ):
        if lock_timeout_ms < 0 or lock_timeout_ms > 0xFFFFFFFF:
            raise ValueError("lock_timeout_ms must fit in an unsigned 32-bit integer")
        self._store = _ManifestStore(manifest_path)
        self._driver = driver if driver is not None else LibCudaCheckpointDriver()
        self._lock_timeout_ms = lock_timeout_ms
        self._starttime_reader = starttime_reader

    def checkpoint_all(
        self,
        targets: Sequence[CheckpointTarget],
        epoch: Union[str, int],
        holder_instance: Optional[str] = None,
        team: Optional[str] = None,
    ) -> RecoveryStatus:
        if epoch is None or isinstance(epoch, bool):
            raise ValueError("epoch must be a non-empty string or integer")
        normalized_epoch = str(epoch)
        if not normalized_epoch:
            raise ValueError("epoch must not be empty")
        normalized_targets = self._validate_targets(targets)

        with self._store.locked():
            manifest = self._store.read()
            if manifest is None:
                manifest = self._new_manifest(normalized_targets, normalized_epoch)
                manifest.holder_instance = holder_instance
                manifest.team = team
                self._store.write(manifest)
            else:
                self._validate_transaction(
                    manifest, normalized_targets, normalized_epoch
                )
                # Re-verify the durable multicast keeper holder before resuming an
                # in-flight checkpoint transaction: if the holder exited or was
                # replaced, the multicast FDs are gone and wake is unrecoverable.
                self._verify_holder_instance(
                    manifest, holder_instance, team, "checkpoint"
                )
                if all(
                    record.state == ProcessState.UNLOCKED
                    for record in manifest.processes
                ):
                    self._require_all_driver_states(
                        manifest, DRIVER_STATE_RUNNING, "restart checkpoint"
                    )
                    for record in manifest.processes:
                        record.state = ProcessState.RUNNING
                    manifest.recovery_required = False
                    manifest.last_error = None
                    self._store.write(manifest)
                elif manifest.recovery_required or any(
                    record.state in (ProcessState.RESTORED, ProcessState.UNLOCKED)
                    for record in manifest.processes
                ):
                    status = self._status_locked(manifest, inspect_driver=True)
                    raise RecoveryRequiredError(
                        "an incomplete checkpoint transaction must be restored first",
                        status,
                    )

            try:
                self._lock_all(manifest)
                self._checkpoint_locked_all(manifest)
                self._require_all_driver_states(
                    manifest, DRIVER_STATE_CHECKPOINTED, "finish checkpoint"
                )
            except Exception as error:
                self._record_failure(manifest, str(error))
                rollback_errors = self._rollback_locked(manifest)
                message = f"checkpoint transaction failed: {error}"
                if rollback_errors:
                    message += "; recovery incomplete: " + "; ".join(rollback_errors)
                manifest.last_error = message
                self._store.write(manifest)
                status = self._status_locked(manifest, inspect_driver=False)
                exception_type = (
                    RecoveryRequiredError
                    if rollback_errors
                    else CheckpointTransactionError
                )
                raise exception_type(message, status) from error

            manifest.recovery_required = False
            manifest.last_error = None
            self._store.write(manifest)
            return self._status_locked(manifest, inspect_driver=True)

    def restore_all(
        self,
        expected_holder_instance: Optional[str] = None,
        expected_team: Optional[str] = None,
    ) -> RecoveryStatus:
        with self._store.locked():
            manifest = self._store.read()
            if manifest is None:
                return self._empty_status()

            # Fail closed before touching the driver: a checkpoint whose multicast
            # keeper holder is gone/changed cannot be restored (its multicast FDs
            # were closed when the holder exited).
            self._verify_holder_instance(
                manifest, expected_holder_instance, expected_team, "restore"
            )

            identity_errors = self._validate_all_identities(manifest)
            if identity_errors:
                self._record_failure(manifest, "; ".join(identity_errors))
                status = self._status_locked(manifest, inspect_driver=False)
                raise RecoveryRequiredError(
                    "refusing restore because a target PID identity is stale", status
                )

            errors = self._restore_then_unlock_all(manifest)
            if errors:
                self._record_failure(manifest, "; ".join(errors))
                status = self._status_locked(manifest, inspect_driver=False)
                raise RecoveryRequiredError(
                    "restore transaction requires retry: " + "; ".join(errors), status
                )

            manifest.recovery_required = False
            manifest.last_error = None
            self._store.write(manifest)
            completed = self._status_locked(manifest, inspect_driver=True)
            self._store.clear()
            return RecoveryStatus(
                epoch=completed.epoch,
                phase="UNLOCKED",
                manifest_exists=False,
                recovery_required=False,
                checkpoint_complete=False,
                restore_complete=True,
                processes=completed.processes,
                last_error=None,
            )

    def recovery_status(self) -> RecoveryStatus:
        with self._store.locked():
            manifest = self._store.read()
            if manifest is None:
                return self._empty_status()
            return self._status_locked(manifest, inspect_driver=True)

    def _new_manifest(
        self, targets: Sequence[CheckpointTarget], epoch: str
    ) -> _Manifest:
        records = []
        for target in targets:
            actual_starttime = self._starttime_reader(target.pid)
            if (
                target.expected_starttime is not None
                and actual_starttime != target.expected_starttime
            ):
                raise StaleProcessError(
                    f"pid {target.pid} starttime {actual_starttime} differs from "
                    f"backend-reported {target.expected_starttime}"
                )
            records.append(
                ProcessRecord(
                    target.pid,
                    actual_starttime,
                    target.rank,
                    target.address,
                    ProcessState.RUNNING,
                )
            )
        manifest = _Manifest(epoch, records)
        self._require_all_driver_states(
            manifest, DRIVER_STATE_RUNNING, "begin checkpoint"
        )
        return manifest

    @staticmethod
    def _validate_targets(
        targets: Sequence[CheckpointTarget],
    ) -> List[CheckpointTarget]:
        normalized = list(targets)
        if not normalized:
            raise ValueError("at least one checkpoint target is required")
        if any(not isinstance(target, CheckpointTarget) for target in normalized):
            raise TypeError("targets must be CheckpointTarget instances")
        pids = [target.pid for target in normalized]
        ranks = [target.rank for target in normalized]
        if (
            any(
                not isinstance(target.pid, int)
                or isinstance(target.pid, bool)
                or target.pid <= 0
                or not isinstance(target.rank, int)
                or isinstance(target.rank, bool)
                or target.rank < 0
                or not isinstance(target.address, str)
                or not target.address
                or (
                    target.expected_starttime is not None
                    and (
                        not isinstance(target.expected_starttime, int)
                        or isinstance(target.expected_starttime, bool)
                        or target.expected_starttime <= 0
                    )
                )
                for target in normalized
            )
            or len(set(pids)) != len(pids)
            or len(set(ranks)) != len(ranks)
        ):
            raise ValueError("targets require unique positive pids/ranks and addresses")
        return sorted(normalized, key=lambda target: target.rank)

    def _verify_holder_instance(
        self,
        manifest: _Manifest,
        holder_instance: Optional[str],
        team: Optional[str],
        context: str,
    ) -> None:
        """Fail closed if the persisted multicast keeper holder cannot be honored.

        Only enforced when a holder was durably recorded (keeper-enabled Level 3);
        manifests without a holder (L1/L2 or keeper disabled) are unaffected. A
        missing/changed holder means the multicast FDs are gone, so wake is
        unrecoverable and the transaction must not proceed.
        """
        if manifest.holder_instance is None:
            return
        if not holder_instance:
            self._record_failure(
                manifest,
                f"{context}: multicast keeper holder is gone "
                f"(expected {manifest.holder_instance})",
            )
            raise RecoveryRequiredError(
                "multicast keeper holder is no longer present; wake is unrecoverable",
                self._status_locked(manifest, inspect_driver=False),
            )
        if holder_instance != manifest.holder_instance:
            self._record_failure(
                manifest,
                f"{context}: multicast keeper holder changed "
                f"({manifest.holder_instance} -> {holder_instance})",
            )
            raise RecoveryRequiredError(
                "multicast keeper holder changed since checkpoint; wake is "
                "unrecoverable",
                self._status_locked(manifest, inspect_driver=False),
            )
        if manifest.team is not None and team is not None and team != manifest.team:
            self._record_failure(
                manifest,
                f"{context}: multicast keeper team changed "
                f"({manifest.team} -> {team})",
            )
            raise RecoveryRequiredError(
                "multicast keeper team changed since checkpoint; wake is "
                "unrecoverable",
                self._status_locked(manifest, inspect_driver=False),
            )

    def _validate_transaction(
        self,
        manifest: _Manifest,
        targets: Sequence[CheckpointTarget],
        epoch: str,
    ) -> None:
        expected = [(target.pid, target.rank, target.address) for target in targets]
        actual = [
            (record.pid, record.rank, record.address) for record in manifest.processes
        ]
        if manifest.epoch != epoch or actual != expected:
            raise RecoveryRequiredError(
                "checkpoint manifest belongs to a different epoch or target set",
                self._status_locked(manifest, inspect_driver=True),
            )
        expected_starttimes = {
            target.pid: target.expected_starttime
            for target in targets
            if target.expected_starttime is not None
        }
        for record in manifest.processes:
            expected_starttime = expected_starttimes.get(record.pid)
            if (
                expected_starttime is not None
                and record.starttime != expected_starttime
            ):
                raise RecoveryRequiredError(
                    "checkpoint manifest identity differs from backend status",
                    self._status_locked(manifest, inspect_driver=True),
                )
        identity_errors = self._validate_all_identities(manifest)
        if identity_errors:
            self._record_failure(manifest, "; ".join(identity_errors))
            raise RecoveryRequiredError(
                "checkpoint target identity changed",
                self._status_locked(manifest, inspect_driver=False),
            )

    def _lock_all(self, manifest: _Manifest) -> None:
        for record in manifest.processes:
            if record.state == ProcessState.CHECKPOINTED:
                continue
            driver_state = self._get_driver_state(record)
            if record.state == ProcessState.RUNNING:
                if driver_state == DRIVER_STATE_RUNNING:
                    self._call_driver(record, "lock", self._lock_timeout_ms)
                    record.state = ProcessState.LOCKED
                    self._store.write(manifest)
                elif driver_state == DRIVER_STATE_LOCKED:
                    record.state = ProcessState.LOCKED
                    self._store.write(manifest)
                elif driver_state == DRIVER_STATE_CHECKPOINTED:
                    record.state = ProcessState.CHECKPOINTED
                    self._store.write(manifest)
                else:
                    raise CheckpointError(
                        f"pid {record.pid} is {_driver_state_name(driver_state)}"
                    )
            elif record.state == ProcessState.LOCKED:
                if driver_state == DRIVER_STATE_CHECKPOINTED:
                    record.state = ProcessState.CHECKPOINTED
                    self._store.write(manifest)
                elif driver_state != DRIVER_STATE_LOCKED:
                    raise CheckpointError(
                        f"pid {record.pid} manifest is LOCKED but driver is "
                        f"{_driver_state_name(driver_state)}"
                    )

    def _checkpoint_locked_all(self, manifest: _Manifest) -> None:
        for record in manifest.processes:
            driver_state = self._get_driver_state(record)
            if driver_state == DRIVER_STATE_CHECKPOINTED:
                if record.state != ProcessState.CHECKPOINTED:
                    record.state = ProcessState.CHECKPOINTED
                    self._store.write(manifest)
                continue
            if (
                record.state != ProcessState.LOCKED
                or driver_state != DRIVER_STATE_LOCKED
            ):
                raise CheckpointError(
                    f"pid {record.pid} is not locked before checkpoint "
                    f"(manifest={record.state.value}, "
                    f"driver={_driver_state_name(driver_state)})"
                )
            self._call_driver(record, "checkpoint")
            record.state = ProcessState.CHECKPOINTED
            self._store.write(manifest)

    def _rollback_locked(self, manifest: _Manifest) -> List[str]:
        errors = self._restore_then_unlock_all(manifest)
        manifest.recovery_required = bool(errors)
        self._store.write(manifest)
        return errors

    def _restore_then_unlock_all(self, manifest: _Manifest) -> List[str]:
        """Restore every rank to LOCKED before allowing any rank to run."""

        restore_errors, already_running = self._restore_all_to_locked(manifest)
        if restore_errors:
            return restore_errors
        if already_running:
            return []

        unlock_errors = self._unlock_all(manifest)
        unlock_errors.extend(self._verify_all_running(manifest))
        return unlock_errors

    def _restore_all_to_locked(self, manifest: _Manifest) -> Tuple[List[str], bool]:
        """Phase one of restore: establish a durable all-ranks LOCKED barrier."""

        errors: List[str] = []
        driver_states: Dict[int, int] = {}
        for record in manifest.processes:
            try:
                driver_states[record.pid] = self._get_driver_state(record)
            except Exception as error:
                errors.append(f"pid {record.pid} restore state: {error}")

        if not errors and all(
            driver_states[record.pid] == DRIVER_STATE_RUNNING
            for record in manifest.processes
        ):
            for record in manifest.processes:
                record.state = ProcessState.UNLOCKED
            self._store.write(manifest)
            return [], True

        for record in manifest.processes:
            driver_state = driver_states.get(record.pid)
            if driver_state is None:
                continue
            try:
                if driver_state == DRIVER_STATE_CHECKPOINTED:
                    self._call_driver(record, "restore")
                elif driver_state == DRIVER_STATE_RUNNING:
                    # A retry may observe ranks already unlocked by an earlier
                    # interrupted phase two. Re-lock them to re-establish the
                    # all-ranks barrier before attempting unlock again.
                    self._call_driver(record, "lock", self._lock_timeout_ms)
                elif driver_state != DRIVER_STATE_LOCKED:
                    raise CheckpointError(
                        f"driver state is {_driver_state_name(driver_state)}"
                    )
                record.state = ProcessState.RESTORED
                self._store.write(manifest)
            except Exception as error:
                errors.append(f"pid {record.pid} restore: {error}")

        for record in manifest.processes:
            try:
                driver_state = self._get_driver_state(record)
                if driver_state != DRIVER_STATE_LOCKED:
                    errors.append(
                        f"pid {record.pid} is {_driver_state_name(driver_state)}, "
                        "not LOCKED after restore"
                    )
                elif record.state != ProcessState.RESTORED:
                    record.state = ProcessState.RESTORED
                    self._store.write(manifest)
            except Exception as error:
                errors.append(f"pid {record.pid} restore verification: {error}")

        return errors, False

    def _unlock_all(self, manifest: _Manifest) -> List[str]:
        errors: List[str] = []
        for record in manifest.processes:
            try:
                driver_state = self._get_driver_state(record)
                if driver_state == DRIVER_STATE_LOCKED:
                    self._call_driver(record, "unlock")
                    record.state = ProcessState.UNLOCKED
                    self._store.write(manifest)
                elif driver_state == DRIVER_STATE_RUNNING:
                    if record.state != ProcessState.UNLOCKED:
                        record.state = ProcessState.UNLOCKED
                        self._store.write(manifest)
                else:
                    raise CheckpointError(
                        f"driver state is {_driver_state_name(driver_state)}"
                    )
            except Exception as error:
                errors.append(f"pid {record.pid} unlock: {error}")
        return errors

    def _verify_all_running(self, manifest: _Manifest) -> List[str]:
        errors: List[str] = []
        for record in manifest.processes:
            try:
                driver_state = self._get_driver_state(record)
                if driver_state != DRIVER_STATE_RUNNING:
                    errors.append(
                        f"pid {record.pid} is {_driver_state_name(driver_state)}, "
                        "not RUNNING"
                    )
                elif record.state != ProcessState.UNLOCKED:
                    record.state = ProcessState.UNLOCKED
                    self._store.write(manifest)
            except Exception as error:
                errors.append(f"pid {record.pid} verification: {error}")
        return errors

    def _require_all_driver_states(
        self, manifest: _Manifest, expected_state: int, action: str
    ) -> None:
        for record in manifest.processes:
            driver_state = self._get_driver_state(record)
            if driver_state != expected_state:
                raise CheckpointError(
                    f"cannot {action}: pid {record.pid} is "
                    f"{_driver_state_name(driver_state)}, expected "
                    f"{_driver_state_name(expected_state)}"
                )

    def _validate_all_identities(self, manifest: _Manifest) -> List[str]:
        errors = []
        for record in manifest.processes:
            try:
                self._validate_identity(record)
            except Exception as error:
                errors.append(str(error))
        return errors

    def _validate_identity(self, record: ProcessRecord) -> None:
        actual_starttime = self._starttime_reader(record.pid)
        if actual_starttime != record.starttime:
            raise StaleProcessError(
                f"stale pid {record.pid}: starttime changed from "
                f"{record.starttime} to {actual_starttime}"
            )

    def _get_driver_state(self, record: ProcessRecord) -> int:
        self._validate_identity(record)
        return self._driver.get_state(record.pid)

    def _call_driver(self, record: ProcessRecord, operation: str, *args: int) -> None:
        self._validate_identity(record)
        method = getattr(self._driver, operation)
        result = method(record.pid, *args)
        if result != CUDA_SUCCESS:
            raise CheckpointError(
                f"{operation} pid {record.pid} failed: "
                f"{self._driver.error_string(result)}"
            )

    def _record_failure(self, manifest: _Manifest, message: str) -> None:
        manifest.recovery_required = True
        manifest.last_error = message
        self._store.write(manifest)

    def _status_locked(
        self, manifest: _Manifest, inspect_driver: bool
    ) -> RecoveryStatus:
        process_statuses: List[ProcessRecoveryStatus] = []
        inspection_failed = False
        for record in manifest.processes:
            identity_valid = True
            driver_state_name: Optional[str] = None
            error_message: Optional[str] = None
            if inspect_driver:
                try:
                    driver_state_name = _driver_state_name(
                        self._get_driver_state(record)
                    )
                except Exception as error:
                    identity_valid = not isinstance(error, StaleProcessError)
                    error_message = str(error)
                    inspection_failed = True
            process_statuses.append(
                ProcessRecoveryStatus(
                    record.pid,
                    record.starttime,
                    record.rank,
                    record.address,
                    record.state,
                    identity_valid,
                    driver_state_name,
                    error_message,
                )
            )

        all_checkpointed = all(
            record.state == ProcessState.CHECKPOINTED for record in manifest.processes
        ) and (
            not inspect_driver
            or all(status.driver_state == "CHECKPOINTED" for status in process_statuses)
        )
        all_unlocked = all(
            record.state == ProcessState.UNLOCKED for record in manifest.processes
        ) and (
            not inspect_driver
            or all(status.driver_state == "RUNNING" for status in process_statuses)
        )
        recovery_required = (
            manifest.recovery_required
            or inspection_failed
            or not (all_checkpointed or all_unlocked)
        )
        if all_checkpointed and not recovery_required:
            phase = "CHECKPOINTED"
        elif all_unlocked and not recovery_required:
            phase = "UNLOCKED"
        else:
            phase = "RECOVERY_REQUIRED"
        return RecoveryStatus(
            epoch=manifest.epoch,
            phase=phase,
            manifest_exists=True,
            recovery_required=recovery_required,
            checkpoint_complete=all_checkpointed,
            restore_complete=all_unlocked,
            processes=tuple(process_statuses),
            last_error=manifest.last_error,
        )

    @staticmethod
    def _empty_status() -> RecoveryStatus:
        return RecoveryStatus(
            epoch=None,
            phase="NONE",
            manifest_exists=False,
            recovery_required=False,
            checkpoint_complete=False,
            restore_complete=True,
            processes=(),
            last_error=None,
        )


def _driver_state_name(state: int) -> str:
    return _DRIVER_STATE_NAMES.get(state, f"UNKNOWN({state})")


def checkpoint_all(
    manifest_path: str,
    targets: Sequence[CheckpointTarget],
    epoch: Union[str, int],
    driver: Optional[CudaCheckpointDriver] = None,
    lock_timeout_ms: int = 60000,
    starttime_reader: Callable[[int], int] = read_process_starttime,
    holder_instance: Optional[str] = None,
    team: Optional[str] = None,
) -> RecoveryStatus:
    """Checkpoint all targets. Safe to retry with the same epoch and targets."""

    return CheckpointController(
        manifest_path, driver, lock_timeout_ms, starttime_reader
    ).checkpoint_all(targets, epoch, holder_instance=holder_instance, team=team)


def restore_all(
    manifest_path: str,
    driver: Optional[CudaCheckpointDriver] = None,
    starttime_reader: Callable[[int], int] = read_process_starttime,
    expected_holder_instance: Optional[str] = None,
    expected_team: Optional[str] = None,
) -> RecoveryStatus:
    """Restore all manifest targets. Missing manifests are successful no-ops."""

    return CheckpointController(
        manifest_path, driver, starttime_reader=starttime_reader
    ).restore_all(
        expected_holder_instance=expected_holder_instance,
        expected_team=expected_team,
    )


def recovery_status(
    manifest_path: str,
    driver: Optional[CudaCheckpointDriver] = None,
    starttime_reader: Callable[[int], int] = read_process_starttime,
) -> RecoveryStatus:
    """Inspect durable and driver state without mutating the transaction."""

    return CheckpointController(
        manifest_path, driver, starttime_reader=starttime_reader
    ).recovery_status()
