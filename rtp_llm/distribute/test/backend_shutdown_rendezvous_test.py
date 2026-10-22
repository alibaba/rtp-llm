import threading
import unittest
from unittest.mock import patch

from torch.distributed import HashStore

import rtp_llm.distribute.distributed_server as ds


class _FakeStore:
    def __init__(self):
        self.data = {}
        self.checked_keys = []

    def set(self, key, value):
        self.data[key] = value.encode("utf-8")

    def get(self, key):
        return self.data[key]

    def check(self, keys):
        self.checked_keys.append(keys)
        return all(key in self.data for key in keys)

    def compare_set(self, key, expected, desired):
        expected_bytes = expected.encode("utf-8")
        if self.data.get(key, b"") == expected_bytes:
            self.data[key] = desired.encode("utf-8")
        return self.data.get(key, b"")


class _TransientFailureStore(_FakeStore):
    def __init__(self, operation, failures=1):
        super().__init__()
        self.operation = operation
        self.failures = failures

    def _maybe_fail(self, operation):
        if self.operation == operation and self.failures > 0:
            self.failures -= 1
            raise RuntimeError(f"transient {operation} failure")

    def set(self, *args):
        self._maybe_fail("set")
        return super().set(*args)

    def check(self, *args):
        self._maybe_fail("check")
        return super().check(*args)

    def compare_set(self, *args):
        self._maybe_fail("compare_set")
        return super().compare_set(*args)


class _UnavailableStore(_FakeStore):
    def check(self, keys):
        raise RuntimeError("store unavailable")

    def compare_set(self, key, expected, desired):
        raise RuntimeError("store unavailable")


class _AckFailureStore(_FakeStore):
    def __init__(self, ack_key):
        super().__init__()
        self.ack_key = ack_key
        self.failed = False

    def set(self, key, value):
        if key == self.ack_key and not self.failed:
            self.failed = True
            raise RuntimeError("transient ACK failure")
        return super().set(key, value)


class BackendShutdownRendezvousTest(unittest.TestCase):
    def _server(self, rank=1, world_size=2):
        server = ds.DistributedServer.__new__(ds.DistributedServer)
        server.rank = rank
        server.world_size = world_size
        server.store = _FakeStore()
        return server

    def _choose(self, server, timeout, local_step, armed=None, cancelled=None):
        if armed is None:
            armed = []
        if cancelled is None:
            cancelled = []
        return server.choose_backend_stop_step(
            timeout,
            local_step,
            armed.append,
            lambda: cancelled.append(True),
        )

    def test_rendezvous_waits_for_all_world_ranks(self):
        server = self._server(rank=2, world_size=4)
        for rank in (0, 1, 3):
            server.store.set(f"backend_shutdown_drained_rank_{rank}", "ready")

        server.wait_for_backend_shutdown(7.5, "drained")

        self.assertEqual(server.store.get("backend_shutdown_drained_rank_2"), b"ready")
        self.assertEqual(
            server.store.checked_keys[1],
            [f"backend_shutdown_drained_rank_{rank}" for rank in range(4)],
        )

    def test_rendezvous_observes_peer_abort(self):
        server = self._server()
        server.store.set(server.BACKEND_SHUTDOWN_DECISION_KEY + "drained", "abort:0")

        with self.assertRaisesRegex(TimeoutError, "aborted by a peer"):
            server.wait_for_backend_shutdown(7.5, "drained")

    def test_rendezvous_publishes_abort_on_timeout(self):
        server = self._server()

        with patch.object(ds.time, "monotonic", side_effect=[10.0, 11.0]):
            with self.assertRaisesRegex(TimeoutError, "timed out on rank 1"):
                server.wait_for_backend_shutdown(0.5, "drained")

        self.assertEqual(
            server.store.get(server.BACKEND_SHUTDOWN_DECISION_KEY + "drained"),
            b"abort:1",
        )

    def test_late_timeout_observes_atomic_success_decision(self):
        server = self._server()
        server.store.set("backend_shutdown_drained_rank_0", "ready")
        server.store.set(server.BACKEND_SHUTDOWN_DECISION_KEY + "drained", "success")

        with patch.object(ds.time, "monotonic", side_effect=[10.0, 11.0]):
            server.wait_for_backend_shutdown(0.5, "drained")

    def test_shutdown_phases_use_independent_keys(self):
        server = self._server()
        server.store.set("backend_shutdown_engine_stopped_rank_0", "ready")

        server.wait_for_backend_shutdown(7.5, "engine_stopped")

        self.assertEqual(
            server.store.get("backend_shutdown_engine_stopped_rank_1"), b"ready"
        )

    def test_shutdown_request_is_job_wide(self):
        server = self._server()

        self.assertFalse(server.is_backend_shutdown_requested())
        server.request_backend_shutdown()
        self.assertTrue(server.is_backend_shutdown_requested())
        self.assertEqual(server.store.get(server.BACKEND_SHUTDOWN_REQUEST_KEY), b"1")

    def test_choose_stop_step_uses_global_max_with_margin(self):
        server = self._server(rank=1, world_size=3)
        server.store.set(server.BACKEND_SHUTDOWN_STEP_KEY + "0", "64")
        server.store.set(server.BACKEND_SHUTDOWN_STEP_KEY + "2", "128")
        server.store.set(server.BACKEND_SHUTDOWN_STEP_ACK_KEY + "0", "384")
        server.store.set(server.BACKEND_SHUTDOWN_STEP_ACK_KEY + "2", "384")

        armed = []
        target = self._choose(server, 7.5, 96, armed=armed)

        self.assertEqual(target, 384)
        self.assertEqual(armed, [384])
        self.assertEqual(server.store.get(server.BACKEND_SHUTDOWN_TARGET_KEY), b"384")
        self.assertEqual(
            server.store.get(server.BACKEND_SHUTDOWN_STEP_DECISION_KEY), b"success"
        )

    def test_terminal_engine_cannot_fall_back_to_single_rank_force_stop(self):
        server = self._server(rank=0, world_size=1)
        with self.assertRaisesRegex(
            ds.BackendStopConsensusError, "matching multi-rank"
        ):
            self._choose(server, 1, -2)
        self.assertEqual(self._choose(server, 1, -1), -1)

    def test_terminal_capability_selects_boundary_not_future_steps(self):
        server = self._server(rank=1, world_size=8)
        for rank in range(8):
            if rank != server.rank:
                server.store.set(server.BACKEND_SHUTDOWN_STEP_KEY + str(rank), "-2")
                server.store.set(server.BACKEND_SHUTDOWN_STEP_ACK_KEY + str(rank), "0")
        armed = []
        self.assertEqual(self._choose(server, 1, -2, armed=armed), 0)
        self.assertEqual(armed, [0])
        self.assertEqual(server.store.get(server.BACKEND_SHUTDOWN_TARGET_KEY), b"0")

    def test_terminal_capability_mismatch_aborts_without_force_stop_target(self):
        for peer_step in (-1, 64):
            with self.subTest(peer_step=peer_step):
                server = self._server()
                server.store.set(server.BACKEND_SHUTDOWN_STEP_KEY + "0", str(peer_step))
                server.store.set(server.BACKEND_SHUTDOWN_STEP_CANCEL_KEY + "0", "ready")
                armed = []
                with self.assertRaisesRegex(
                    ds.BackendStopConsensusError, "incompatible.*terminal"
                ):
                    self._choose(server, 1, -2, armed=armed)
                self.assertEqual(armed, [])
                self.assertNotIn(server.BACKEND_SHUTDOWN_TARGET_KEY, server.store.data)

    def test_terminal_topology_rejection_never_acknowledges_cancellation(self):
        server = self._server()
        server.store.set(server.BACKEND_SHUTDOWN_STEP_KEY + "0", "-2")
        server.store.set(server.BACKEND_SHUTDOWN_STEP_CANCEL_KEY + "0", "ready")
        cancelled = []

        def reject_topology(target):
            self.assertEqual(target, 0)
            raise RuntimeError("terminal stop requires one TP world")

        with self.assertRaisesRegex(
            ds.BackendStopConsensusError, "boundary not established"
        ):
            server.choose_backend_stop_step(
                1, -2, reject_topology, lambda: cancelled.append(True)
            )
        self.assertEqual(cancelled, [])
        self.assertNotIn(
            server.BACKEND_SHUTDOWN_STEP_CANCEL_KEY + "1", server.store.data
        )
        self.assertEqual(
            server.store.get(server.BACKEND_SHUTDOWN_STEP_DECISION_KEY), b"abort:1"
        )

    def test_late_root_drives_terminal_after_follower_arm_timeout(self):
        servers = [self._server(rank=rank) for rank in range(2)]
        store = HashStore()
        for server in servers:
            server.store = store
            store.set(server.BACKEND_SHUTDOWN_STEP_KEY + str(server.rank), "-2")
        recovery_waiting = threading.Event()
        boundary = threading.Event()
        calls = [0, 0]
        cancelled = [False, False]
        errors = [None, None]

        def arm(rank):
            calls[rank] += 1
            if rank == 0:
                boundary.set()
            elif calls[rank] == 1:
                raise TimeoutError("follower arm timed out before root arrived")
            else:
                recovery_waiting.set()
                if not boundary.wait(2):
                    raise TimeoutError("root never drove the terminal boundary")

        def cancel(rank):
            if not boundary.is_set():
                raise RuntimeError("cancel before terminal boundary")
            cancelled[rank] = True

        def choose(rank):
            try:
                servers[rank].choose_backend_stop_step(
                    2, -2, lambda target: arm(rank), lambda: cancel(rank)
                )
            except Exception as error:
                errors[rank] = error

        follower = threading.Thread(target=choose, args=(1,))
        follower.start()
        try:
            self.assertTrue(recovery_waiting.wait(2))
            self.assertEqual(
                store.get(servers[0].BACKEND_SHUTDOWN_STEP_DECISION_KEY), b"abort:1"
            )
            choose(0)  # root sees abort before ever entering its normal arm path
        finally:
            follower.join(5)
        self.assertFalse(follower.is_alive())
        self.assertEqual(calls, [1, 2])
        self.assertEqual(cancelled, [True, True])
        for rank, error in enumerate(errors):
            self.assertIs(type(error), TimeoutError)  # safe, not the park-only subclass
            self.assertEqual(
                store.get(servers[0].BACKEND_SHUTDOWN_STEP_CANCEL_KEY + str(rank)),
                b"terminal_ready",
            )

    def test_terminal_abort_rejects_legacy_unarmed_cancel_ack(self):
        server = self._server()
        server.store.set(server.BACKEND_SHUTDOWN_STEP_KEY + "0", "-2")
        server.store.set(server.BACKEND_SHUTDOWN_STEP_DECISION_KEY, "abort:0")
        server.store.set(server.BACKEND_SHUTDOWN_STEP_CANCEL_KEY + "0", "ready")
        with self.assertRaisesRegex(
            ds.BackendStopConsensusError, "peer did not acknowledge"
        ):
            self._choose(server, 1, -2)

    def test_terminal_cancel_failure_cannot_authorize_local_force_stop(self):
        server = self._server()
        server.store.set(server.BACKEND_SHUTDOWN_STEP_KEY + "0", "-2")
        server.store.set(server.BACKEND_SHUTDOWN_STEP_CANCEL_KEY + "0", "ready")

        arm_calls = []

        def fail_arm(target):
            arm_calls.append(target)
            if len(arm_calls) == 1:
                raise RuntimeError("arm timed out")

        def fail_cancel():
            raise RuntimeError("terminal boundary not reached")

        with patch.object(ds.time, "monotonic", side_effect=[0, 2]), self.assertRaises(
            ds.BackendStopConsensusError
        ):
            server.choose_backend_stop_step(1, -2, fail_arm, fail_cancel)
        self.assertNotIn(
            server.BACKEND_SHUTDOWN_STEP_CANCEL_KEY + "1", server.store.data
        )

    def test_choose_stop_step_disables_coordination_job_wide(self):
        server = self._server(rank=1, world_size=2)
        server.store.set(server.BACKEND_SHUTDOWN_STEP_KEY + "0", "-1")
        server.store.set(server.BACKEND_SHUTDOWN_STEP_ACK_KEY + "0", "-1")

        armed = []
        target = self._choose(server, 7.5, 64, armed=armed)

        self.assertEqual(target, -1)
        self.assertEqual(armed, [])
        self.assertEqual(server.store.get(server.BACKEND_SHUTDOWN_TARGET_KEY), b"-1")

    def test_choose_stop_step_publishes_shared_abort_on_snapshot_timeout(self):
        server = self._server(rank=1, world_size=2)
        server.store.set(server.BACKEND_SHUTDOWN_STEP_CANCEL_KEY + "0", "ready")

        with patch.object(ds.time, "monotonic", side_effect=[10.0, 11.0]):
            with self.assertRaisesRegex(
                ds.BackendStopConsensusError, "incomplete backend stop capabilities"
            ):
                self._choose(server, 0.5, 64)

        self.assertEqual(
            server.store.get(server.BACKEND_SHUTDOWN_STEP_DECISION_KEY), b"abort:1"
        )

    def test_choose_stop_step_observes_peer_abort(self):
        server = self._server(rank=1, world_size=2)
        server.store.set(server.BACKEND_SHUTDOWN_STEP_KEY + "0", "64")
        server.store.set(server.BACKEND_SHUTDOWN_STEP_DECISION_KEY, "abort:0")
        server.store.set(server.BACKEND_SHUTDOWN_STEP_CANCEL_KEY + "0", "ready")

        with self.assertRaisesRegex(TimeoutError, "aborted by a peer"):
            self._choose(server, 7.5, 64)

    def test_choose_stop_step_timeout_loses_to_atomic_success(self):
        server = self._server(rank=1, world_size=2)
        server.store.set(server.BACKEND_SHUTDOWN_STEP_KEY + "0", "64")
        server.store.set(server.BACKEND_SHUTDOWN_TARGET_KEY, "320")
        server.store.set(server.BACKEND_SHUTDOWN_STEP_DECISION_KEY, "success")

        with patch.object(ds.time, "monotonic", side_effect=[10.0, 11.0]):
            self.assertEqual(self._choose(server, 0.5, 64), 320)

    def test_choose_stop_step_retries_transient_store_failure(self):
        server = self._server(rank=1, world_size=2)
        server.store = _TransientFailureStore("check")
        server.store.set(server.BACKEND_SHUTDOWN_STEP_KEY + "0", "64")
        server.store.set(server.BACKEND_SHUTDOWN_STEP_ACK_KEY + "0", "320")

        with patch.object(ds.time, "sleep"):
            target = self._choose(server, 7.5, 64)

        self.assertEqual(target, 320)
        self.assertEqual(
            server.store.get(server.BACKEND_SHUTDOWN_STEP_DECISION_KEY), b"success"
        )

    def test_choose_stop_step_retries_armed_ack_publication(self):
        server = self._server(rank=1, world_size=2)
        ack_key = server.BACKEND_SHUTDOWN_STEP_ACK_KEY + "1"
        server.store = _AckFailureStore(ack_key)
        server.store.set(server.BACKEND_SHUTDOWN_STEP_KEY + "0", "64")
        server.store.set(server.BACKEND_SHUTDOWN_STEP_ACK_KEY + "0", "320")
        armed = []

        with patch.object(ds.time, "sleep"):
            target = self._choose(server, 7.5, 64, armed=armed)

        self.assertEqual(target, 320)
        self.assertEqual(armed, [320])
        self.assertEqual(server.store.get(ack_key), b"320")

    def test_choose_stop_step_arm_failure_publishes_abort(self):
        server = self._server(rank=1, world_size=2)
        server.store.set(server.BACKEND_SHUTDOWN_STEP_KEY + "0", "64")
        server.store.set(server.BACKEND_SHUTDOWN_STEP_CANCEL_KEY + "0", "ready")

        def fail_arm(_target):
            raise RuntimeError("target already passed")

        with self.assertRaisesRegex(TimeoutError, "arming failed"):
            server.choose_backend_stop_step(0.5, 64, fail_arm, lambda: None)

        self.assertEqual(
            server.store.get(server.BACKEND_SHUTDOWN_STEP_DECISION_KEY), b"abort:1"
        )

    def test_choose_stop_step_peer_abort_cancels_local_arm(self):
        server = self._server(rank=1, world_size=2)
        server.store.set(server.BACKEND_SHUTDOWN_STEP_KEY + "0", "64")
        server.store.set(server.BACKEND_SHUTDOWN_STEP_DECISION_KEY, "abort:0")
        cancelled = []

        with self.assertRaisesRegex(TimeoutError, "aborted by a peer"):
            self._choose(server, 7.5, 64, cancelled=cancelled)

        self.assertEqual(cancelled, [])

    def test_choose_stop_step_abort_waits_for_peer_cancellation(self):
        server = self._server(rank=1, world_size=2)
        server.store.set(server.BACKEND_SHUTDOWN_STEP_KEY + "0", "64")
        server.store.set(server.BACKEND_SHUTDOWN_STEP_DECISION_KEY, "abort:0")

        with patch.object(ds.time, "monotonic", side_effect=[10.0, 10.0, 11.0]):
            with self.assertRaisesRegex(
                ds.BackendStopConsensusError, "cancellation rendezvous timed out"
            ):
                self._choose(server, 0.5, 64)

        self.assertEqual(
            server.store.get(server.BACKEND_SHUTDOWN_STEP_CANCEL_KEY + "1"),
            b"ready",
        )

    def test_choose_stop_step_persistent_store_failure_is_bounded_and_safe(self):
        server = self._server(rank=1, world_size=2)
        server.store = _UnavailableStore()

        with patch.object(ds.time, "monotonic", side_effect=[10.0, 11.0]):
            with self.assertRaisesRegex(
                ds.BackendStopConsensusError, "store unavailable"
            ):
                self._choose(server, 0.5, 64)


if __name__ == "__main__":
    unittest.main()
