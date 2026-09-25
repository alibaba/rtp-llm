import json
import threading
import time
import unittest

from rtp_llm.utils.backend_shutdown import (
    ShutdownError,
    graceful_backend_shutdown,
    register_shutdown_member,
)
from rtp_llm.utils.lifecycle_lease import LifecycleLease


class FakeStore:
    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()

    def set_timeout(self, timeout):
        self.timeout = timeout

    def set(self, key, value):
        with self.lock:
            self.data[key] = value.encode() if isinstance(value, str) else value

    def check(self, keys):
        with self.lock:
            return all(key in self.data for key in keys)

    def get(self, key):
        with self.lock:
            return self.data[key]

    def compare_set(self, key, expected, desired):
        with self.lock:
            current = self.data.get(key, b"")
            if current == expected.encode():
                self.data[key] = desired.encode()
            return self.data.get(key, b"")


class FakeControl:
    def __init__(self, rank, state="RUNNING", round_id=0):
        self.rank = rank
        self.incarnation = f"worker-{rank}"
        self.state = state
        self.round_id = round_id
        self.calls = []
        self.hooks = {}

    def call(self, name, *args):
        self.calls.append((name, *args))
        if name in self.hooks:
            self.hooks[name](*args)

    def shutdown_status(self):
        return {"worker_incarnation": self.incarnation, "state": self.state}

    def begin_shutdown(self):
        self.call("begin")

    def drain_shutdown(self, timeout_ms, seal):
        self.call("seal" if seal else "drain", timeout_ms)

    def freeze_shutdown(self):
        self.call("freeze")
        return self.round_id

    def quiesce_shutdown(self, target, timeout_ms):
        self.call("quiesce", target, timeout_ms)

    def terminate_shutdown(self):
        self.call("terminate")


class FakeClock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class BackendShutdownTest(unittest.TestCase):
    def setUp(self):
        self.store = FakeStore()

    def register(self, controls):
        for control in controls:
            register_shutdown_member(self.store, control.rank, control)

    def run_group(self, controls, timeout=2):
        self.register(controls)
        errors = [None] * len(controls)

        def run(index, control):
            try:
                graceful_backend_shutdown(
                    control,
                    self.store,
                    index,
                    len(controls),
                    control.incarnation,
                    timeout,
                )
            except Exception as error:
                errors[index] = error

        threads = [
            threading.Thread(target=run, args=(i, control), daemon=True)
            for i, control in enumerate(controls)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout + 1)
            self.assertFalse(thread.is_alive(), "shutdown test worker hung")
        return errors

    def run_single(self, control, timeout=1, **kwargs):
        return graceful_backend_shutdown(
            control, self.store, 0, 1, control.incarnation, timeout, **kwargs
        )

    def names(self, control):
        return [call[0] for call in control.calls]

    def test_single_rank_order_and_terminal_lease(self):
        control = FakeControl(0, round_id=7)
        self.register([control])
        self.run_single(control)
        self.assertEqual(
            self.names(control),
            ["begin", "drain", "seal", "freeze", "quiesce", "seal", "terminate"],
        )
        self.assertEqual(control.calls[-3][1], 7)
        self.assertEqual(
            json.loads(self.store.get(LifecycleLease.KEY))["operation"], "shutdown"
        )
        self.assertTrue(LifecycleLease(self.store, None, True).acquire("wake")[1])

    def test_all_ranks_quiesce_before_any_termination(self):
        controls = [FakeControl(0, round_id=3), FakeControl(1, round_id=9)]
        quiesced = set()
        lock = threading.Lock()

        def quiesce(rank, target, timeout_ms):
            self.assertEqual(target, 9)
            self.assertGreater(timeout_ms, 0)
            with lock:
                quiesced.add(rank)

        def terminate():
            with lock:
                self.assertEqual(quiesced, {0, 1})

        for control in controls:
            control.hooks["quiesce"] = lambda target, timeout, r=control.rank: quiesce(
                r, target, timeout
            )
            control.hooks["terminate"] = terminate
        self.assertEqual(self.run_group(controls), [None, None])
        self.assertEqual(sum("observed/1" in key for key in self.store.data), 1)

    def test_real_tcp_store_missing_error_keys_do_not_block_shutdown(self):
        from datetime import timedelta

        from torch.distributed import TCPStore

        self.store = TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=3))
        # Unlike wait/get, check returns False immediately for missing keys.
        # Exercise every barrier against a real store, with no error keys set.
        started = time.monotonic()
        self.assertFalse(self.store.check(["missing/error/0"]))
        controls = [FakeControl(0, round_id=3), FakeControl(1, round_id=9)]
        self.assertEqual(self.run_group(controls, timeout=5), [None, None])
        self.assertLess(time.monotonic() - started, 5)
        self.assertTrue(all("terminate" in self.names(c) for c in controls))

    def test_sleeping_exit_does_not_resume_or_freeze(self):
        controls = [FakeControl(0, "SLEEPING"), FakeControl(1, "SLEEPING")]
        self.assertEqual(self.run_group(controls), [None, None])
        for control in controls:
            self.assertEqual(
                self.names(control), ["begin", "drain", "seal", "seal", "terminate"]
            )

    def test_unsignalled_peer_is_not_stopped(self):
        controls = [FakeControl(0), FakeControl(1)]
        self.register(controls)
        clock = FakeClock()
        with self.assertRaisesRegex(ShutdownError, "deadline"):
            graceful_backend_shutdown(
                controls[0],
                self.store,
                0,
                2,
                controls[0].incarnation,
                1,
                clock=clock,
                sleep=clock.sleep,
            )
        self.assertEqual(self.names(controls[0]), ["begin"])
        self.assertEqual(controls[1].calls, [])
        self.assertNotIn(LifecycleLease.KEY, self.store.data)

    def test_inflight_sleep_lease_is_not_stolen(self):
        control = FakeControl(0)
        self.register([control])
        record, error = LifecycleLease(self.store, None, True).acquire("sleep")
        self.assertFalse(error)
        clock = FakeClock()
        with self.assertRaisesRegex(ShutdownError, "deadline"):
            self.run_single(control, clock=clock, sleep=clock.sleep)
        self.assertEqual(self.store.get(LifecycleLease.KEY).decode(), record)
        self.assertEqual(self.names(control), ["begin"])

    def test_released_lifecycle_lease_can_be_acquired(self):
        control = FakeControl(0)
        self.register([control])
        lease = LifecycleLease(self.store, None, True)
        record, _ = lease.acquire("wake")
        clock = FakeClock()

        def sleep(seconds):
            lease.release(record)
            clock.sleep(seconds)

        self.run_single(control, clock=clock, sleep=sleep)
        self.assertIn("terminate", self.names(control))

    def test_mixed_states_fail_before_drain(self):
        controls = [FakeControl(0), FakeControl(1, "SLEEPING")]
        errors = self.run_group(controls)
        self.assertTrue(all(isinstance(error, ShutdownError) for error in errors))
        for control in controls:
            self.assertEqual(self.names(control), ["begin"])

    def test_shutdown_does_not_claim_success_after_a_competing_wake_fails(self):
        control = FakeControl(0, "WAKING_UP")
        self.register([control])
        lease = LifecycleLease(self.store, None, True)
        record, _ = lease.acquire("wake")
        clock = FakeClock()

        def finish_wake(seconds):
            # A wake that lost to terminal intent could not resume this rank.
            control.state = "ERROR"
            lease.release(record)
            clock.sleep(seconds)

        with self.assertRaisesRegex(ShutdownError, "partially transitioned"):
            self.run_single(control, clock=clock, sleep=finish_wake)
        self.assertEqual(self.names(control), ["begin"])
        self.assertEqual(
            json.loads(self.store.get(LifecycleLease.KEY))["operation"], "shutdown"
        )

    def test_transitional_state_fails_closed(self):
        control = FakeControl(0, "WAKING_UP")
        self.register([control])
        with self.assertRaisesRegex(ShutdownError, "partially transitioned"):
            self.run_single(control)
        self.assertEqual(self.names(control), ["begin"])

    def test_quiesce_failure_prevents_every_termination(self):
        controls = [FakeControl(0), FakeControl(1)]

        def fail(*args):
            raise RuntimeError("GPU synchronization failed")

        controls[1].hooks["quiesce"] = fail
        errors = self.run_group(controls)
        self.assertTrue(all(isinstance(error, ShutdownError) for error in errors))
        for control in controls:
            self.assertNotIn("terminate", self.names(control))

    def test_late_transfers_drain_after_all_ranks_quiesce_before_termination(self):
        controls = [FakeControl(0), FakeControl(1)]
        quiesced = set()
        final_drained = set()
        pending = set()
        lock = threading.Lock()

        def quiesce(rank, *args):
            # Finishing an async runner may perform deferred stream cleanup
            # and start cache writeback after the earlier drain barriers.
            with lock:
                quiesced.add(rank)
                pending.add(rank)

        def drain(rank, *args):
            with lock:
                if rank in quiesced:
                    self.assertEqual(quiesced, {0, 1})
                    pending.remove(rank)
                    final_drained.add(rank)

        def terminate():
            with lock:
                self.assertEqual(final_drained, {0, 1})
                self.assertFalse(pending)

        for control in controls:
            control.hooks["quiesce"] = lambda *args, r=control.rank: quiesce(r, *args)
            control.hooks["seal"] = lambda *args, r=control.rank: drain(r, *args)
            control.hooks["terminate"] = terminate
        self.assertEqual(self.run_group(controls), [None, None])

    def test_late_transfer_failure_prevents_every_termination(self):
        controls = [FakeControl(0), FakeControl(1)]

        def drain(*args):
            if "quiesce" in self.names(controls[1]):
                raise RuntimeError("late cache write still owns backing")

        controls[1].hooks["seal"] = drain
        errors = self.run_group(controls)
        self.assertTrue(all(isinstance(error, ShutdownError) for error in errors))
        self.assertTrue(any("late cache write" in str(error) for error in errors))
        for control in controls:
            self.assertNotIn("terminate", self.names(control))

    def test_post_quiesce_drain_uses_remaining_deadline(self):
        control = FakeControl(0)
        self.register([control])
        clock = FakeClock()
        control.hooks["quiesce"] = lambda *args: clock.sleep(0.75)
        final_budgets = []

        def drain(timeout):
            if "quiesce" in self.names(control):
                final_budgets.append(timeout)
                clock.sleep(0.30)

        control.hooks["seal"] = drain
        with self.assertRaisesRegex(ShutdownError, "deadline"):
            self.run_single(control, clock=clock, sleep=clock.sleep)
        self.assertEqual(len(final_budgets), 1)
        self.assertLessEqual(final_budgets[0], 250)
        self.assertNotIn("terminate", self.names(control))

    def test_cleanup_must_finish_before_freeze(self):
        control = FakeControl(0)
        self.register([control])

        def fail(*args):
            raise RuntimeError("active continuation remains")

        control.hooks["seal"] = fail
        with self.assertRaisesRegex(ShutdownError, "active continuation"):
            self.run_single(control)
        self.assertNotIn("freeze", self.names(control))

    def test_member_replacement_invalidates_old_acks(self):
        control = FakeControl(0)
        self.register([control])

        def replace(*args):
            self.store.set("rtp_llm_backend_shutdown/member/0", "replacement")

        control.hooks["drain"] = replace
        with self.assertRaisesRegex(ShutdownError, "incarnation changed"):
            self.run_single(control)
        self.assertNotIn("freeze", self.names(control))

    def test_duplicate_members_fail_closed(self):
        controls = [FakeControl(0), FakeControl(1)]
        controls[1].incarnation = controls[0].incarnation
        errors = self.run_group(controls)
        self.assertTrue(all(isinstance(error, ShutdownError) for error in errors))
        self.assertNotIn(LifecycleLease.KEY, self.store.data)

    def test_total_deadline_not_reset_between_phases(self):
        control = FakeControl(0)
        self.register([control])
        clock = FakeClock()
        control.hooks["drain"] = lambda timeout: clock.sleep(0.75)
        control.hooks["seal"] = lambda timeout: clock.sleep(0.3)
        with self.assertRaisesRegex(ShutdownError, "deadline"):
            self.run_single(control, clock=clock, sleep=clock.sleep)
        self.assertLessEqual(control.calls[-1][1], 250)
        self.assertNotIn("freeze", self.names(control))

    def test_empty_registration_rejected(self):
        control = FakeControl(0)
        control.incarnation = ""
        with self.assertRaisesRegex(ShutdownError, "no incarnation"):
            self.register([control])

    def test_invalid_arguments_rejected_before_control(self):
        control = FakeControl(0)
        for rank, size, timeout in [(0, 0, 1), (2, 2, 1), (0, 1, 0)]:
            with self.assertRaises(ValueError):
                graceful_backend_shutdown(
                    control, self.store, rank, size, control.incarnation, timeout
                )
        self.assertFalse(control.calls)


if __name__ == "__main__":
    unittest.main()
