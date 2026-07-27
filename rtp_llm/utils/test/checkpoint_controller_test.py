import ctypes
import fcntl
import json
import multiprocessing
import os
import queue
import tempfile
import unittest
from unittest import mock

from rtp_llm.utils.checkpoint_controller import (
    CUDA_SUCCESS,
    DRIVER_STATE_CHECKPOINTED,
    DRIVER_STATE_LOCKED,
    DRIVER_STATE_RUNNING,
    CheckpointController,
    CheckpointTarget,
    CheckpointTransactionError,
    LibCudaCheckpointDriver,
    ManifestError,
    ProcessRecord,
    ProcessState,
    RecoveryRequiredError,
    StaleProcessError,
    _CUcheckpointCheckpointArgs,
    _CUcheckpointLockArgs,
    _CUcheckpointRestoreArgs,
    _CUcheckpointUnlockArgs,
    _Manifest,
    _ManifestStore,
    checkpoint_all,
    checkpoint_manifest_path,
    read_process_starttime,
    recovery_status,
    restore_all,
)

CUDA_ERROR = 999


class FakeDriver:
    def __init__(self, pids):
        self.states = {pid: DRIVER_STATE_RUNNING for pid in pids}
        self.failures = {}
        self.calls = []

    def fail_once(self, operation, pid):
        self.failures[(operation, pid)] = 1

    def _fails(self, operation, pid):
        key = (operation, pid)
        remaining = self.failures.get(key, 0)
        if remaining:
            self.failures[key] = remaining - 1
            return True
        return False

    def get_state(self, pid):
        self.calls.append(("get_state", pid))
        if self._fails("get_state", pid):
            raise RuntimeError("injected get_state failure")
        return self.states[pid]

    def lock(self, pid, timeout_ms):
        self.calls.append(("lock", pid, timeout_ms))
        if self._fails("lock", pid):
            return CUDA_ERROR
        if self.states[pid] != DRIVER_STATE_RUNNING:
            return CUDA_ERROR
        self.states[pid] = DRIVER_STATE_LOCKED
        return CUDA_SUCCESS

    def checkpoint(self, pid):
        self.calls.append(("checkpoint", pid))
        if self._fails("checkpoint", pid):
            return CUDA_ERROR
        if self.states[pid] != DRIVER_STATE_LOCKED:
            return CUDA_ERROR
        self.states[pid] = DRIVER_STATE_CHECKPOINTED
        return CUDA_SUCCESS

    def restore(self, pid):
        self.calls.append(("restore", pid))
        if self._fails("restore", pid):
            return CUDA_ERROR
        if self.states[pid] != DRIVER_STATE_CHECKPOINTED:
            return CUDA_ERROR
        self.states[pid] = DRIVER_STATE_LOCKED
        return CUDA_SUCCESS

    def unlock(self, pid):
        self.calls.append(("unlock", pid))
        if self._fails("unlock", pid):
            return CUDA_ERROR
        if self.states[pid] != DRIVER_STATE_LOCKED:
            return CUDA_ERROR
        self.states[pid] = DRIVER_STATE_RUNNING
        return CUDA_SUCCESS

    @staticmethod
    def error_string(result):
        return f"fake CUDA error {result}"

    def mutation_calls(self):
        return [call for call in self.calls if call[0] != "get_state"]


class FakeCFunction:
    def __init__(self, result=CUDA_SUCCESS):
        self.result = result
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.result


class FakeCudaLibrary:
    def __init__(self):
        self.cuInit = FakeCFunction()
        self.cuGetErrorString = FakeCFunction()
        self.cuCheckpointProcessGetState = FakeCFunction()
        self.cuCheckpointProcessLock = FakeCFunction()
        self.cuCheckpointProcessCheckpoint = FakeCFunction()
        self.cuCheckpointProcessRestore = FakeCFunction()
        self.cuCheckpointProcessUnlock = FakeCFunction()


def _locked_status_worker(path, ready, done):
    ready.put(True)
    controller = CheckpointController(path, driver=FakeDriver([]))
    done.put(controller.recovery_status().phase)


class LibCudaCheckpointDriverTest(unittest.TestCase):
    def test_configures_explicit_ctypes_signatures(self):
        library = FakeCudaLibrary()
        with mock.patch(
            "rtp_llm.utils.checkpoint_controller.ctypes.CDLL", return_value=library
        ):
            LibCudaCheckpointDriver()

        self.assertEqual(library.cuInit.argtypes, [ctypes.c_uint])
        self.assertEqual(library.cuInit.restype, ctypes.c_int)
        self.assertEqual(
            library.cuCheckpointProcessGetState.argtypes,
            [ctypes.c_int, ctypes.POINTER(ctypes.c_int)],
        )
        self.assertEqual(
            library.cuCheckpointProcessLock.argtypes,
            [ctypes.c_int, ctypes.POINTER(_CUcheckpointLockArgs)],
        )
        self.assertEqual(
            library.cuCheckpointProcessCheckpoint.argtypes,
            [ctypes.c_int, ctypes.POINTER(_CUcheckpointCheckpointArgs)],
        )
        self.assertEqual(
            library.cuCheckpointProcessRestore.argtypes,
            [ctypes.c_int, ctypes.POINTER(_CUcheckpointRestoreArgs)],
        )
        self.assertEqual(
            library.cuCheckpointProcessUnlock.argtypes,
            [ctypes.c_int, ctypes.POINTER(_CUcheckpointUnlockArgs)],
        )
        for function in (
            library.cuGetErrorString,
            library.cuCheckpointProcessGetState,
            library.cuCheckpointProcessLock,
            library.cuCheckpointProcessCheckpoint,
            library.cuCheckpointProcessRestore,
            library.cuCheckpointProcessUnlock,
        ):
            self.assertEqual(function.restype, ctypes.c_int)

    def test_restore_args_match_cuda_13_abi_size(self):
        self.assertEqual(ctypes.sizeof(_CUcheckpointRestoreArgs), 64)


class CheckpointControllerTest(unittest.TestCase):
    PIDS = (111, 222)
    EPOCH = "sleep-epoch-7"

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.path = os.path.join(self.tempdir.name, "checkpoint.json")
        self.targets = [
            CheckpointTarget(111, 0, "127.0.0.1:19000"),
            CheckpointTarget(222, 1, "127.0.0.1:19001"),
        ]
        self.starttimes = {111: 1001, 222: 1002}
        self.driver = FakeDriver(self.PIDS)
        self.controller = self._controller()

    def _controller(self):
        return CheckpointController(
            self.path,
            driver=self.driver,
            lock_timeout_ms=1234,
            starttime_reader=self.starttimes.__getitem__,
        )

    def _manifest(self):
        return _ManifestStore(self.path).read()

    def _write_manifest(self, states, recovery_required=False):
        records = [
            ProcessRecord(
                target.pid,
                self.starttimes[target.pid],
                target.rank,
                target.address,
                state,
            )
            for target, state in zip(self.targets, states)
        ]
        _ManifestStore(self.path).write(
            _Manifest(
                self.EPOCH,
                records,
                recovery_required=recovery_required,
                last_error="injected crash" if recovery_required else None,
            )
        )

    def test_success_persists_identity_and_lockstep_order_then_restores(self):
        status = self.controller.checkpoint_all(self.targets, self.EPOCH)

        self.assertEqual(status.phase, "CHECKPOINTED")
        self.assertTrue(status.checkpoint_complete)
        mutations = self.driver.mutation_calls()
        self.assertEqual(
            [call[0] for call in mutations],
            ["lock", "lock", "checkpoint", "checkpoint"],
        )
        self.assertEqual(mutations[0], ("lock", 111, 1234))
        with open(self.path, "r", encoding="utf-8") as manifest_file:
            payload = json.load(manifest_file)
        self.assertEqual(payload["epoch"], self.EPOCH)
        self.assertEqual(
            payload["processes"],
            [
                {
                    "address": "127.0.0.1:19000",
                    "pid": 111,
                    "rank": 0,
                    "starttime": 1001,
                    "state": "CHECKPOINTED",
                },
                {
                    "address": "127.0.0.1:19001",
                    "pid": 222,
                    "rank": 1,
                    "starttime": 1002,
                    "state": "CHECKPOINTED",
                },
            ],
        )

        restored = self.controller.restore_all()
        self.assertEqual(restored.phase, "UNLOCKED")
        self.assertTrue(restored.restore_complete)
        self.assertFalse(restored.manifest_exists)
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(
            self.driver.states,
            {111: DRIVER_STATE_RUNNING, 222: DRIVER_STATE_RUNNING},
        )

    def test_backend_reported_starttime_mismatch_precedes_driver_mutation(self):
        targets = [
            CheckpointTarget(
                111,
                0,
                "127.0.0.1:19000",
                expected_starttime=self.starttimes[111] + 1,
            )
        ]

        with self.assertRaisesRegex(StaleProcessError, "backend-reported"):
            self.controller.checkpoint_all(targets, self.EPOCH)

        self.assertEqual(self.driver.calls, [])
        self.assertFalse(os.path.exists(self.path))

    def test_backend_reported_starttime_is_persisted_when_valid(self):
        targets = [
            CheckpointTarget(
                target.pid,
                target.rank,
                target.address,
                expected_starttime=self.starttimes[target.pid],
            )
            for target in self.targets
        ]

        status = self.controller.checkpoint_all(targets, self.EPOCH)

        self.assertTrue(status.checkpoint_complete)
        self.assertEqual(
            [record.starttime for record in self._manifest().processes],
            [1001, 1002],
        )

    def test_checkpoint_and_restore_are_idempotent(self):
        first = self.controller.checkpoint_all(self.targets, self.EPOCH)
        mutation_count = len(self.driver.mutation_calls())
        second = self.controller.checkpoint_all(self.targets, self.EPOCH)
        self.assertEqual(first.phase, second.phase)
        self.assertEqual(len(self.driver.mutation_calls()), mutation_count)

        self.controller.restore_all()
        mutation_count = len(self.driver.mutation_calls())
        status = self.controller.restore_all()
        self.assertEqual(status.phase, "NONE")
        self.assertEqual(len(self.driver.mutation_calls()), mutation_count)

    def test_partial_lock_failure_rolls_back_and_retry_succeeds(self):
        self.driver.fail_once("lock", 222)
        with self.assertRaises(CheckpointTransactionError) as raised:
            self.controller.checkpoint_all(self.targets, self.EPOCH)

        self.assertFalse(raised.exception.status.recovery_required)
        self.assertEqual(
            [record.state for record in self._manifest().processes],
            [ProcessState.UNLOCKED, ProcessState.UNLOCKED],
        )
        self.assertEqual(set(self.driver.states.values()), {DRIVER_STATE_RUNNING})

        status = self.controller.checkpoint_all(self.targets, self.EPOCH)
        self.assertEqual(status.phase, "CHECKPOINTED")

    def test_partial_checkpoint_failure_rolls_back_and_retry_succeeds(self):
        self.driver.fail_once("checkpoint", 222)
        with self.assertRaises(CheckpointTransactionError):
            self.controller.checkpoint_all(self.targets, self.EPOCH)

        self.assertEqual(set(self.driver.states.values()), {DRIVER_STATE_RUNNING})
        self.assertTrue(os.path.exists(self.path))
        status = self.controller.checkpoint_all(self.targets, self.EPOCH)
        self.assertTrue(status.checkpoint_complete)

    def test_incomplete_checkpoint_rollback_is_never_cleared(self):
        self.driver.fail_once("checkpoint", 222)
        self.driver.fail_once("restore", 111)
        with self.assertRaises(RecoveryRequiredError) as raised:
            self.controller.checkpoint_all(self.targets, self.EPOCH)

        self.assertTrue(raised.exception.status.recovery_required)
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(self.driver.states[111], DRIVER_STATE_CHECKPOINTED)
        with self.assertRaises(RecoveryRequiredError):
            self.controller.checkpoint_all(self.targets, self.EPOCH)

        recovered = self.controller.restore_all()
        self.assertTrue(recovered.restore_complete)
        self.assertFalse(os.path.exists(self.path))

    def test_partial_restore_failure_retains_manifest_and_retry_succeeds(self):
        self.controller.checkpoint_all(self.targets, self.EPOCH)
        self.driver.calls.clear()
        self.driver.fail_once("restore", 222)

        with self.assertRaises(RecoveryRequiredError) as raised:
            self.controller.restore_all()
        self.assertTrue(raised.exception.status.recovery_required)
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(self.driver.states[111], DRIVER_STATE_LOCKED)
        self.assertEqual(self.driver.states[222], DRIVER_STATE_CHECKPOINTED)
        self.assertNotIn("unlock", [call[0] for call in self.driver.mutation_calls()])
        self.assertEqual(
            [record.state for record in self._manifest().processes],
            [ProcessState.RESTORED, ProcessState.CHECKPOINTED],
        )
        self.assertTrue(self._manifest().recovery_required)

        self.driver.calls.clear()
        status = self.controller.restore_all()
        self.assertTrue(status.restore_complete)
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(
            self.driver.mutation_calls(),
            [("restore", 222), ("unlock", 111), ("unlock", 222)],
        )

    def test_restore_state_verification_failure_does_not_unlock_any_rank(self):
        self.controller.checkpoint_all(self.targets, self.EPOCH)
        self.driver.calls.clear()
        original_restore = self.driver.restore

        def restore_without_state_transition(pid):
            if pid == 222:
                self.driver.calls.append(("restore", pid))
                return CUDA_SUCCESS
            return original_restore(pid)

        with mock.patch.object(
            self.driver, "restore", side_effect=restore_without_state_transition
        ):
            with self.assertRaises(RecoveryRequiredError) as raised:
                self.controller.restore_all()

        self.assertTrue(raised.exception.status.recovery_required)
        self.assertEqual(self.driver.states[111], DRIVER_STATE_LOCKED)
        self.assertEqual(self.driver.states[222], DRIVER_STATE_CHECKPOINTED)
        self.assertNotIn("unlock", [call[0] for call in self.driver.mutation_calls()])
        self.assertTrue(self._manifest().recovery_required)

    def test_partial_unlock_failure_retains_manifest_and_retry_succeeds(self):
        self.controller.checkpoint_all(self.targets, self.EPOCH)
        self.driver.fail_once("unlock", 222)

        with self.assertRaises(RecoveryRequiredError):
            self.controller.restore_all()
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(self.driver.states[111], DRIVER_STATE_RUNNING)
        self.assertEqual(self.driver.states[222], DRIVER_STATE_LOCKED)

        self.driver.calls.clear()
        status = self.controller.restore_all()
        self.assertTrue(status.restore_complete)
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(
            self.driver.mutation_calls(),
            [("lock", 111, 1234), ("unlock", 111), ("unlock", 222)],
        )

    def test_checkpoint_rollback_restore_failure_does_not_unlock_any_rank(self):
        self.driver.fail_once("checkpoint", 222)
        self.driver.fail_once("restore", 111)

        with self.assertRaises(RecoveryRequiredError) as raised:
            self.controller.checkpoint_all(self.targets, self.EPOCH)

        self.assertTrue(raised.exception.status.recovery_required)
        self.assertEqual(self.driver.states[111], DRIVER_STATE_CHECKPOINTED)
        self.assertEqual(self.driver.states[222], DRIVER_STATE_LOCKED)
        self.assertNotIn("unlock", [call[0] for call in self.driver.mutation_calls()])
        self.assertTrue(self._manifest().recovery_required)

        self.driver.calls.clear()
        status = self.controller.restore_all()
        self.assertTrue(status.restore_complete)
        self.assertEqual(
            self.driver.mutation_calls(),
            [("restore", 111), ("unlock", 111), ("unlock", 222)],
        )

    def test_stale_pid_blocks_operations_and_preserves_manifest(self):
        self.controller.checkpoint_all(self.targets, self.EPOCH)
        self.driver.calls.clear()
        self.starttimes[222] += 1

        with self.assertRaises(RecoveryRequiredError):
            self.controller.restore_all()
        self.assertEqual(self.driver.mutation_calls(), [])
        self.assertTrue(os.path.exists(self.path))
        status = self.controller.recovery_status()
        self.assertTrue(status.recovery_required)
        stale = next(process for process in status.processes if process.pid == 222)
        self.assertFalse(stale.identity_valid)
        self.assertIn("starttime changed", stale.error)

    def test_crash_after_lock_before_manifest_write_is_reconciled(self):
        self._write_manifest([ProcessState.RUNNING, ProcessState.RUNNING])
        self.driver.states[111] = DRIVER_STATE_LOCKED

        status = self.controller.checkpoint_all(self.targets, self.EPOCH)

        self.assertTrue(status.checkpoint_complete)
        self.assertNotIn(("lock", 111, 1234), self.driver.mutation_calls())
        self.assertIn(("checkpoint", 111), self.driver.mutation_calls())

    def test_crash_after_checkpoint_before_manifest_write_is_reconciled(self):
        self._write_manifest([ProcessState.LOCKED, ProcessState.LOCKED])
        self.driver.states = {
            111: DRIVER_STATE_CHECKPOINTED,
            222: DRIVER_STATE_LOCKED,
        }

        status = self.controller.checkpoint_all(self.targets, self.EPOCH)

        self.assertTrue(status.checkpoint_complete)
        self.assertNotIn(("checkpoint", 111), self.driver.mutation_calls())
        self.assertIn(("checkpoint", 222), self.driver.mutation_calls())

    def test_crash_after_restore_before_manifest_write_is_reconciled(self):
        self._write_manifest([ProcessState.CHECKPOINTED, ProcessState.CHECKPOINTED])
        self.driver.states = {111: DRIVER_STATE_LOCKED, 222: DRIVER_STATE_CHECKPOINTED}

        status = self.controller.restore_all()

        self.assertTrue(status.restore_complete)
        self.assertNotIn(("restore", 111), self.driver.mutation_calls())
        self.assertIn(("unlock", 111), self.driver.mutation_calls())
        self.assertIn(("restore", 222), self.driver.mutation_calls())

    def test_different_epoch_does_not_overwrite_active_manifest(self):
        self.controller.checkpoint_all(self.targets, self.EPOCH)
        with open(self.path, "rb") as manifest_file:
            before = manifest_file.read()

        with self.assertRaises(RecoveryRequiredError):
            self.controller.checkpoint_all(self.targets, "different-epoch")

        with open(self.path, "rb") as manifest_file:
            self.assertEqual(manifest_file.read(), before)

    def test_module_level_frontend_apis(self):
        status = checkpoint_all(
            self.path,
            self.targets,
            self.EPOCH,
            driver=self.driver,
            starttime_reader=self.starttimes.__getitem__,
        )
        self.assertEqual(status.phase, "CHECKPOINTED")
        self.assertEqual(
            recovery_status(
                self.path,
                driver=self.driver,
                starttime_reader=self.starttimes.__getitem__,
            ).phase,
            "CHECKPOINTED",
        )
        self.assertTrue(
            restore_all(
                self.path,
                driver=self.driver,
                starttime_reader=self.starttimes.__getitem__,
            ).restore_complete
        )


class ManifestPersistenceTest(unittest.TestCase):
    def _manifest(self, state=ProcessState.RUNNING):
        return _Manifest(
            "epoch",
            [ProcessRecord(123, 456, 0, "127.0.0.1:19000", state)],
        )

    def test_atomic_replace_failure_preserves_previous_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "manifest.json")
            store = _ManifestStore(path)
            store.write(self._manifest())
            with open(path, "rb") as manifest_file:
                original = manifest_file.read()

            with mock.patch(
                "rtp_llm.utils.checkpoint_controller.os.replace",
                side_effect=OSError("injected replace failure"),
            ):
                with self.assertRaises(OSError):
                    store.write(self._manifest(ProcessState.LOCKED))

            with open(path, "rb") as manifest_file:
                self.assertEqual(manifest_file.read(), original)
            self.assertEqual(
                [
                    name
                    for name in os.listdir(directory)
                    if name.startswith(".manifest")
                ],
                [],
            )

    def test_atomic_write_fsyncs_file_and_parent_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _ManifestStore(os.path.join(directory, "manifest.json"))
            with mock.patch(
                "rtp_llm.utils.checkpoint_controller.os.fsync", wraps=os.fsync
            ) as fsync:
                store.write(self._manifest())
            self.assertGreaterEqual(fsync.call_count, 2)

    def test_manifest_process_lock_serializes_other_processes(self):
        if "fork" not in multiprocessing.get_all_start_methods():
            self.skipTest("requires multiprocessing fork")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "manifest.json")
            lock_path = f"{path}.lock"
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            context = multiprocessing.get_context("fork")
            ready = context.Queue()
            done = context.Queue()
            process = context.Process(
                target=_locked_status_worker, args=(path, ready, done)
            )
            process.start()
            try:
                self.assertTrue(ready.get(timeout=5))
                with self.assertRaises(queue.Empty):
                    done.get(timeout=0.2)
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                self.assertEqual(done.get(timeout=5), "NONE")
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            self.assertEqual(process.exitcode, 0)

    def test_corrupt_manifest_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "manifest.json")
            with open(path, "w", encoding="utf-8") as manifest_file:
                manifest_file.write('{"version":1,"processes":[]}')
            controller = CheckpointController(path, driver=FakeDriver([]))
            with self.assertRaises(ManifestError):
                controller.recovery_status()
            self.assertTrue(os.path.exists(path))


class ProcessIdentityTest(unittest.TestCase):
    def test_reads_starttime_when_process_name_contains_spaces_and_parentheses(self):
        with tempfile.TemporaryDirectory() as proc_root:
            process_dir = os.path.join(proc_root, "123")
            os.mkdir(process_dir)
            fields = ["S"] + [str(field) for field in range(4, 23)]
            with open(os.path.join(process_dir, "stat"), "w", encoding="utf-8") as stat:
                stat.write("123 (worker ) name) " + " ".join(fields))
            self.assertEqual(read_process_starttime(123, proc_root), 22)

    def test_missing_pid_is_stale(self):
        with tempfile.TemporaryDirectory() as proc_root:
            with self.assertRaises(StaleProcessError):
                read_process_starttime(999, proc_root)

    def test_manifest_path_is_stable_and_instance_specific(self):
        addresses = ["127.0.0.1:19001", "127.0.0.1:19000"]
        self.assertEqual(
            checkpoint_manifest_path(addresses, "/tmp"),
            checkpoint_manifest_path(list(reversed(addresses)), "/tmp"),
        )
        self.assertNotEqual(
            checkpoint_manifest_path(addresses, "/tmp"),
            checkpoint_manifest_path(["127.0.0.1:19002"], "/tmp"),
        )

    def test_manifest_path_namespace_prevents_cross_role_instance_collision(self):
        # FIX #6: scoping the manifest by (instance_generation, role) must yield a
        # distinct path even when the control addresses are identical, so a shared
        # node cannot reuse a stale manifest across roles/generations.
        addresses = ["127.0.0.1:19000", "127.0.0.1:19001"]
        base = checkpoint_manifest_path(addresses, "/tmp")
        prefill_g1 = checkpoint_manifest_path(
            addresses, "/tmp", namespace="RoleType.PREFILL/gen-1"
        )
        decode_g1 = checkpoint_manifest_path(
            addresses, "/tmp", namespace="RoleType.DECODE/gen-1"
        )
        prefill_g2 = checkpoint_manifest_path(
            addresses, "/tmp", namespace="RoleType.PREFILL/gen-2"
        )
        self.assertNotEqual(base, prefill_g1)
        self.assertNotEqual(prefill_g1, decode_g1)
        self.assertNotEqual(prefill_g1, prefill_g2)
        # Same namespace + addresses is stable.
        self.assertEqual(
            prefill_g1,
            checkpoint_manifest_path(
                list(reversed(addresses)), "/tmp", namespace="RoleType.PREFILL/gen-1"
            ),
        )


class KeeperHolderDurabilityTest(CheckpointControllerTest):
    """FIX #5: durable multicast keeper holder persistence and fail-closed verify."""

    HOLDER = "keeper-holder-A"
    TEAM = "RoleType.PREFILL/gen-1"

    def _manifest_payload(self):
        with open(self.path, "r", encoding="utf-8") as manifest_file:
            return json.load(manifest_file)

    def test_checkpoint_persists_holder_instance_and_team(self):
        self.controller.checkpoint_all(
            self.targets, self.EPOCH, holder_instance=self.HOLDER, team=self.TEAM
        )
        payload = self._manifest_payload()
        self.assertEqual(payload["holder_instance"], self.HOLDER)
        self.assertEqual(payload["team"], self.TEAM)
        # Round-trips through decode without loss.
        manifest = self._manifest()
        self.assertEqual(manifest.holder_instance, self.HOLDER)
        self.assertEqual(manifest.team, self.TEAM)

    def test_restore_requires_the_same_holder_instance(self):
        self.controller.checkpoint_all(
            self.targets, self.EPOCH, holder_instance=self.HOLDER, team=self.TEAM
        )
        # A changed holder must fail closed (multicast FDs are gone).
        with self.assertRaises(RecoveryRequiredError):
            self._controller().restore_all(
                expected_holder_instance="keeper-holder-B", expected_team=self.TEAM
            )
        # A vanished holder must also fail closed.
        with self.assertRaises(RecoveryRequiredError):
            self._controller().restore_all(expected_holder_instance=None)
        # The manifest is retained and marked for recovery, never silently cleared.
        self.assertTrue(os.path.exists(self.path))
        self.assertTrue(self._manifest().recovery_required)

    def test_restore_succeeds_when_holder_matches(self):
        self.controller.checkpoint_all(
            self.targets, self.EPOCH, holder_instance=self.HOLDER, team=self.TEAM
        )
        status = self._controller().restore_all(
            expected_holder_instance=self.HOLDER, expected_team=self.TEAM
        )
        self.assertTrue(status.restore_complete)
        self.assertFalse(os.path.exists(self.path))

    def test_checkpoint_retry_fails_closed_when_holder_changed(self):
        # Simulate a mid-transaction manifest (LOCKED) that pinned holder A.
        self._write_manifest([ProcessState.LOCKED, ProcessState.LOCKED])
        manifest = self._manifest()
        manifest.holder_instance = self.HOLDER
        manifest.team = self.TEAM
        _ManifestStore(self.path).write(manifest)
        for pid in self.PIDS:
            self.driver.states[pid] = DRIVER_STATE_LOCKED
        # Resuming checkpoint with a different holder must fail closed.
        with self.assertRaises(RecoveryRequiredError):
            self.controller.checkpoint_all(
                self.targets,
                self.EPOCH,
                holder_instance="keeper-holder-B",
                team=self.TEAM,
            )

    def test_manifest_without_holder_stays_backward_compatible(self):
        # L1/L2 (or keeper disabled): no holder recorded, no enforcement.
        self.controller.checkpoint_all(self.targets, self.EPOCH)
        payload = self._manifest_payload()
        self.assertIsNone(payload["holder_instance"])
        self.assertIsNone(payload["team"])
        status = self._controller().restore_all()
        self.assertTrue(status.restore_complete)


class CheckpointAdapterRankTest(unittest.TestCase):
    """FIX #6: the checkpoint adapter must trust backend world_rank, not order."""

    def _adapter(self):
        from rtp_llm.utils.grpc_client_wrapper import _CheckpointControllerAdapter

        return _CheckpointControllerAdapter()

    def test_target_rank_prefers_backend_world_rank_over_address_order(self):
        adapter = self._adapter()
        # default_rank (address order) is 0, but the backend reports world_rank 5.
        self.assertEqual(adapter._target_rank({"world_rank": 5}, default_rank=0), 5)

    def test_target_rank_falls_back_to_address_order_only_when_absent(self):
        adapter = self._adapter()
        self.assertEqual(adapter._target_rank({}, default_rank=3), 3)

    def test_targets_use_reported_world_rank_not_enumeration_order(self):
        import rtp_llm.utils.checkpoint_controller as module

        adapter = self._adapter()
        # Control addresses in one order, but the backend reports the OPPOSITE
        # world_rank assignment. The manifest targets must follow world_rank.
        control_addresses = ["127.0.0.1:19000", "127.0.0.1:19001"]
        terminal_statuses = [
            {
                "address": "127.0.0.1:19000",
                "world_rank": 1,
                "process_id": 111,
                "sleep_epoch": 7,
                "process_starttime": 5001,
                "process_pid_namespace": 4026531836,
                "process_boot_id": "boot",
            },
            {
                "address": "127.0.0.1:19001",
                "world_rank": 0,
                "process_id": 222,
                "sleep_epoch": 7,
                "process_starttime": 5002,
                "process_pid_namespace": 4026531836,
                "process_boot_id": "boot",
            },
        ]
        with mock.patch.object(
            module,
            "read_process_starttime",
            side_effect=lambda pid: {111: 5001, 222: 5002}[pid],
        ), mock.patch(
            "rtp_llm.utils.grpc_client_wrapper._local_process_identity",
            return_value={"pid_namespace": 4026531836, "boot_id": "boot"},
        ):
            targets, epoch = adapter._targets_and_epoch(
                module, control_addresses, terminal_statuses
            )
        self.assertEqual(epoch, 7)
        rank_by_pid = {target.pid: target.rank for target in targets}
        self.assertEqual(rank_by_pid[111], 1)
        self.assertEqual(rank_by_pid[222], 0)


class GrpcClientWrapperKeyNamespaceTest(unittest.TestCase):
    """FIX #6: lease/recovery/manifest keys namespaced by (role, generation)."""

    def _wrapper(self):
        from rtp_llm.utils.grpc_client_wrapper import GrpcClientWrapper

        return GrpcClientWrapper(server_port=12345)

    def _statuses(self, role, generation):
        return [
            {
                "world_rank": 0,
                "role": role,
                "instance_generation_uuid": generation,
            },
            {
                "world_rank": 1,
                "role": role,
                "instance_generation_uuid": generation,
            },
        ]

    def test_keys_are_base_before_identity_is_resolved(self):
        wrapper = self._wrapper()
        # Legacy / unresolved backend: no namespace suffix (backward compatible).
        self.assertEqual(wrapper._lifecycle_lease_key(), wrapper.LIFECYCLE_LEASE_KEY)
        self.assertEqual(
            wrapper._lifecycle_recovery_key(), wrapper.LIFECYCLE_RECOVERY_KEY
        )
        self.assertIsNone(wrapper._manifest_namespace())

    def test_identity_namespaces_all_lifecycle_keys(self):
        wrapper = self._wrapper()
        error = wrapper._apply_instance_identity(
            self._statuses("RoleType.PREFILL", "gen-1")
        )
        self.assertEqual(error, "")
        self.assertTrue(
            wrapper._lifecycle_lease_key().startswith(wrapper.LIFECYCLE_LEASE_KEY)
        )
        self.assertIn("RoleType.PREFILL", wrapper._lifecycle_lease_key())
        self.assertIn("gen-1", wrapper._lifecycle_lease_key())
        self.assertEqual(wrapper._manifest_namespace(), "RoleType.PREFILL/gen-1")

    def test_prefill_and_decode_keys_do_not_collide(self):
        prefill = self._wrapper()
        prefill._apply_instance_identity(self._statuses("RoleType.PREFILL", "gen-1"))
        decode = self._wrapper()
        decode._apply_instance_identity(self._statuses("RoleType.DECODE", "gen-1"))
        self.assertNotEqual(
            prefill._lifecycle_lease_key(), decode._lifecycle_lease_key()
        )
        self.assertNotEqual(
            prefill._lifecycle_recovery_key(), decode._lifecycle_recovery_key()
        )
        self.assertNotEqual(prefill._manifest_namespace(), decode._manifest_namespace())

    def test_distinct_generations_do_not_collide(self):
        gen1 = self._wrapper()
        gen1._apply_instance_identity(self._statuses("RoleType.PREFILL", "gen-1"))
        gen2 = self._wrapper()
        gen2._apply_instance_identity(self._statuses("RoleType.PREFILL", "gen-2"))
        self.assertNotEqual(gen1._manifest_namespace(), gen2._manifest_namespace())
        self.assertNotEqual(gen1._lifecycle_lease_key(), gen2._lifecycle_lease_key())

    def test_inconsistent_roles_are_rejected(self):
        wrapper = self._wrapper()
        error = wrapper._apply_instance_identity(
            [
                {"world_rank": 0, "role": "RoleType.PREFILL"},
                {"world_rank": 1, "role": "RoleType.DECODE"},
            ]
        )
        self.assertIn("inconsistent roles", error)

    def test_generation_taken_from_world_rank_zero(self):
        wrapper = self._wrapper()
        wrapper._apply_instance_identity(
            [
                {
                    "world_rank": 1,
                    "role": "RoleType.PREFILL",
                    "instance_generation_uuid": "gen-other",
                },
                {
                    "world_rank": 0,
                    "role": "RoleType.PREFILL",
                    "instance_generation_uuid": "gen-canonical",
                },
            ]
        )
        self.assertEqual(
            wrapper._manifest_namespace(), "RoleType.PREFILL/gen-canonical"
        )

    def test_keeper_holder_is_sourced_from_rank_zero_status(self):
        wrapper = self._wrapper()
        statuses = [
            {"world_rank": 0, "holder_instance": "keeper-A"},
            {"world_rank": 1, "holder_instance": "keeper-A"},
        ]
        self.assertEqual(wrapper._keeper_holder_instance(statuses), "keeper-A")
        # No holder reported -> None (backward compatible, keeper disabled/L1/L2).
        self.assertIsNone(
            wrapper._keeper_holder_instance([{"world_rank": 0}, {"world_rank": 1}])
        )


if __name__ == "__main__":
    unittest.main()
