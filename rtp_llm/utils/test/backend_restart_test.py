"""Unit tests for in-place backend restart: budget, barrier, stop precedence, heartbeats."""

import os
import subprocess
import sys
import threading
import time
import unittest
import unittest.mock

from rtp_llm.test.utils.port_util import PortManager
from rtp_llm.utils.backend_restart import (
    CONNECT_ATTEMPT_S,
    BackendRestarter,
    GiveUp,
    RestartOutcome,
    RowCoordinator,
    RowEvent,
    RowStoreError,
    ShutdownRequested,
    _BoundedStore,
    kill_process_group,
    process_group_members,
)

_PORT_LOCKS = []


def free_port() -> int:
    # The lock stays held for the whole test process, so a port that an earlier test's
    # store may still hold is never handed out again, nor to a concurrent test.
    ports, locks = PortManager().get_consecutive_ports(1)
    _PORT_LOCKS.extend(locks)
    return ports[0]


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


class FakeProc:
    _next_pid = 1000

    def __init__(self):
        FakeProc._next_pid += 1
        self.pid = FakeProc._next_pid
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass

    def kill(self):
        self.alive = False


def _late_group_leader(delay_s):
    """A backend still starting: it leads its own group only after a while, then forks ranks."""
    time.sleep(delay_s)
    os.setpgid(0, 0)
    for _ in range(2):
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    time.sleep(60)


class Harness:
    """A single-machine restarter whose backends, clock and process kills are fakes."""

    def __init__(self, limit=3, ready=lambda: True, kill_ok=True, gate_file=None):
        self.clock = FakeClock()
        self.spawned = []
        self.killed = []
        self.shutdown = False
        self.kill_ok = kill_ok
        self.ready = ready
        self.restarter = BackendRestarter(
            restart_limit=limit,
            spawn_backend=self._spawn,
            coordinator=RowCoordinator(1, 0, clock=self.clock),
            is_shutdown_requested=lambda: self.shutdown,
            gate_file=gate_file,
            kill_group=self._kill,
            sleep=self.clock.sleep,
            clock=self.clock,
        )

    def _spawn(self):
        proc = FakeProc()
        self.spawned.append(proc)
        return proc, self.ready

    def _kill(self, pgid, timeout):
        self.killed.append(pgid)
        return self.kill_ok

    def fail(self, proc):
        proc.alive = False
        return self.restarter.handle_backend_failure([proc], local=True)


class SingleMachineRestartTest(unittest.TestCase):
    def test_not_restarted_before_first_start_completes(self):
        h = Harness()
        outcome, _ = h.fail(FakeProc())
        self.assertEqual(outcome, RestartOutcome.GIVE_UP)
        self.assertEqual(h.spawned, [])

    def test_restart_kills_old_group_toggles_gate_and_backs_off(self):
        gate = os.path.join(self.tmp, "gate")
        open(gate, "w").close()
        seen_gate = []
        h = Harness(
            gate_file=gate, ready=lambda: seen_gate.append(os.path.exists(gate)) or True
        )
        h.restarter.arm()
        old = FakeProc()
        outcome, procs = h.fail(old)
        self.assertEqual(outcome, RestartOutcome.RESTARTED)
        self.assertEqual(procs, h.spawned)
        self.assertEqual(h.killed, [old.pid])
        self.assertEqual(
            seen_gate, [False]
        )  # /health stays 503 while the new one starts
        self.assertTrue(os.path.exists(gate))
        self.assertAlmostEqual(sum(h.clock.slept), 10.0)  # first backoff
        self.assertEqual(h.restarter.coordinator.generation, 1)

    def test_budget_exhaustion_gives_up_and_backoff_doubles(self):
        h = Harness(limit=2)
        h.restarter.arm()
        proc = FakeProc()
        waited = []
        for _ in range(2):
            before = h.clock.now
            outcome, (proc,) = h.fail(proc)
            self.assertEqual(outcome, RestartOutcome.RESTARTED)
            waited.append(round(h.clock.now - before, 6))
        outcome, _ = h.fail(proc)
        self.assertEqual(outcome, RestartOutcome.GIVE_UP)
        self.assertEqual(waited, [10.0, 20.0])

    def test_budget_resets_after_healthy_period(self):
        h = Harness(limit=1)
        h.restarter.arm()
        outcome, (proc,) = h.fail(FakeProc())
        self.assertEqual(outcome, RestartOutcome.RESTARTED)
        h.clock.now += 600
        outcome, _ = h.fail(proc)
        self.assertEqual(outcome, RestartOutcome.RESTARTED)

    def test_failed_start_counts_against_budget(self):
        attempts = []

        def ready():
            attempts.append(1)
            raise RuntimeError("rank failed to load")

        h = Harness(limit=3, ready=ready)
        h.restarter.arm()
        outcome, _ = h.fail(FakeProc())
        self.assertEqual(outcome, RestartOutcome.GIVE_UP)
        self.assertEqual(len(h.spawned), 3)
        self.assertEqual(len(h.killed), 4)  # the original and every failed start

    def test_uncleanable_group_gives_up(self):
        h = Harness(kill_ok=False)
        h.restarter.arm()
        outcome, _ = h.fail(FakeProc())
        self.assertEqual(outcome, RestartOutcome.GIVE_UP)
        self.assertEqual(h.spawned, [])

    def test_shutdown_during_backoff_takes_normal_stop(self):
        h = Harness()
        h.restarter.arm()
        sleep = h.clock.sleep

        def sleep_then_sigterm(seconds):
            sleep(seconds)
            h.shutdown = True

        h.restarter._sleep = sleep_then_sigterm
        outcome, procs = h.fail(FakeProc())
        self.assertEqual(outcome, RestartOutcome.SHUTDOWN)
        self.assertEqual(procs, [])
        self.assertEqual(h.spawned, [])
        self.assertTrue(h.restarter.coordinator.stop_requested())

    def test_shutdown_while_starting_hands_back_the_new_backend(self):
        h = Harness()
        h.restarter.arm()

        def ready():
            h.shutdown = True
            return False

        h.ready = ready
        outcome, procs = h.fail(FakeProc())
        self.assertEqual(outcome, RestartOutcome.SHUTDOWN)
        self.assertEqual(procs, h.spawned)

    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp()


def wait_until(predicate, timeout=15.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class RowCoordinatorTest(unittest.TestCase):
    """Real TCPStore on localhost; each coordinator stands for one machine's start_server."""

    def setUp(self):
        self.port = free_port()
        self.rows = []

    def tearDown(self):
        for row in reversed(self.rows):
            row.close()
            row._store = None  # shut the store server down with its last reference
        self.rows.clear()

    def make_row(self, n=2, hb_timeout=2.0, barrier_timeout=10.0):
        for node in range(n):
            row = RowCoordinator(
                n,
                node,
                "127.0.0.1",
                self.port,
                connect_timeout_s=20,
                heartbeat_interval_s=0.1,
                heartbeat_timeout_s=hb_timeout,
                barrier_timeout_s=barrier_timeout,
            )
            row.start()
            self.rows.append(row)
        return self.rows

    def run_round(self, row, decide, results, abort=lambda: False):
        try:
            results[row.node_index] = row.await_next_generation(decide, abort)
        except (GiveUp, ShutdownRequested) as e:
            results[row.node_index] = type(e).__name__

    def test_failure_broadcast_and_clean_barrier(self):
        leader, follower = self.make_row()
        self.assertFalse(leader.row_failure_pending())
        follower.report_failure("rank died")
        self.assertTrue(wait_until(leader.row_failure_pending))
        results = {}
        threads = [
            threading.Thread(target=self.run_round, args=(r, lambda: 0.0, results))
            for r in (leader, follower)
        ]
        leader.report_clean()
        for t in threads:
            t.start()
        time.sleep(1.0)
        self.assertEqual(results, {})  # follower has not cleaned yet: nobody moves on
        follower.report_clean()
        for t in threads:
            t.join(10)
        self.assertEqual(results, {0: 1, 1: 1})
        time.sleep(0.5)
        self.assertFalse(leader.row_failure_pending())  # the new generation is clean

    def test_budget_exhausted_at_leader_gives_up_the_whole_row(self):
        leader, follower = self.make_row()
        results = {}
        threads = [
            threading.Thread(target=self.run_round, args=(r, lambda: None, results))
            for r in (leader, follower)
        ]
        leader.report_clean()
        follower.report_clean()
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        self.assertEqual(results, {0: "GiveUp", 1: "GiveUp"})

    def test_missing_clean_times_out_into_give_up(self):
        leader, follower = self.make_row(barrier_timeout=1.0)
        results = {}
        leader.report_clean()
        t = threading.Thread(target=self.run_round, args=(follower, None, results))
        t.start()
        self.run_round(leader, lambda: 0.0, results)
        t.join(10)
        self.assertEqual(results, {0: "GiveUp", 1: "GiveUp"})

    def test_stop_outranks_failure_and_give_up(self):
        leader, follower = self.make_row()
        follower.report_failure("rank died")
        leader.mark_stop()
        leader._store("set", "giveup", "0")
        self.assertTrue(wait_until(follower.stop_requested))
        self.assertFalse(follower.row_failure_pending())
        results = {}
        self.run_round(follower, None, results)
        self.assertEqual(results, {1: "ShutdownRequested"})

    def test_heartbeat_survives_a_blocked_main_thread_and_detects_a_dead_peer(self):
        leader, follower = self.make_row(hb_timeout=1.0)
        time.sleep(3.0)  # main thread busy, as during a long model load
        self.assertFalse(leader.peer_lost())
        self.assertFalse(follower.peer_lost())
        follower.close()  # the follower's start_server is gone
        self.assertTrue(wait_until(leader.peer_lost, timeout=10))
        self.assertTrue(leader.row_failure_pending())
        self.assertIn("node 1 stopped beating", leader.lost_reason)

    def test_a_late_machine_is_not_reported_lost(self):
        leader = RowCoordinator(
            2,
            0,
            "127.0.0.1",
            self.port,
            heartbeat_interval_s=0.1,
            heartbeat_timeout_s=0.5,
        )
        leader.start()
        self.rows.append(leader)
        time.sleep(2.0)
        self.assertFalse(leader.peer_lost())

    def test_row_ready_requires_every_machine(self):
        leader, follower = self.make_row()
        leader.report_ready()
        time.sleep(0.5)
        self.assertFalse(leader.row_ready())
        self.assertIsNone(leader.row_ready_at)
        follower.report_ready()
        self.assertTrue(wait_until(leader.row_ready))
        self.assertIsNotNone(leader.row_ready_at)

    def test_gate_opens_only_when_every_machine_runs_the_new_generation(self):
        import tempfile

        leader, follower = self.make_row()
        gate = os.path.join(tempfile.mkdtemp(), "gate")
        restarter = BackendRestarter(
            3,
            lambda: (FakeProc(), lambda: True),
            leader,
            is_shutdown_requested=lambda: False,
            gate_file=gate,
            backoff_base_s=0.0,
            kill_group=lambda pgid, timeout: True,
        )
        restarter.arm()
        follower.report_ready()
        dead = FakeProc()
        dead.alive = False
        results = {}
        t = threading.Thread(
            target=lambda: results.update(
                outcome=restarter.handle_backend_failure([dead], local=True)[0]
            )
        )
        t.start()
        self.assertTrue(wait_until(follower.row_failure_pending))
        follower.report_clean()
        follower.await_next_generation(None, lambda: False)
        time.sleep(1.0)  # the leader's backend is ready; the follower's is not yet
        self.assertFalse(os.path.exists(gate))
        self.assertNotIn("outcome", results)
        follower.report_ready()
        t.join(10)
        self.assertEqual(results["outcome"], RestartOutcome.RESTARTED)
        self.assertTrue(os.path.exists(gate))

    def test_restarter_reports_row_stop_from_another_machine(self):
        leader, follower = self.make_row()
        restarter = BackendRestarter(
            1, lambda: None, leader, is_shutdown_requested=lambda: False
        )
        self.assertEqual(restarter.poll(), RowEvent.NONE)
        follower.mark_stop()
        self.assertTrue(wait_until(lambda: restarter.poll() == RowEvent.STOP))
        dead = FakeProc()
        dead.alive = False
        restarter.arm()
        outcome, procs = restarter.handle_backend_failure([dead], local=True)
        self.assertEqual((outcome, procs), (RestartOutcome.SHUTDOWN, []))


class RestartFailurePathsTest(unittest.TestCase):
    """Expected failures inside a restart end in GIVE_UP with every started backend owned."""

    def test_spawn_error_counts_as_a_failed_round(self):
        h = Harness(limit=2)
        h.restarter.arm()

        def failing_spawn():
            raise OSError("cannot start process")

        h.restarter._spawn_backend = failing_spawn
        outcome, procs = h.fail(FakeProc())
        self.assertEqual(outcome, RestartOutcome.GIVE_UP)
        self.assertEqual(procs, [])

    def test_store_failure_after_spawn_hands_the_new_backend_back(self):
        h = Harness()
        h.restarter.arm()

        def broken_report_ready():
            raise RowStoreError("store stalled")

        h.restarter.coordinator.report_ready = broken_report_ready
        outcome, procs = h.fail(FakeProc())
        self.assertEqual(outcome, RestartOutcome.GIVE_UP)
        self.assertEqual(procs, h.spawned)
        self.assertTrue(procs[0].is_alive())

    def test_stop_while_reporting_ready_keeps_the_gate_closed(self):
        gate = os.path.join(self.tmp, "gate")
        h = Harness(gate_file=gate)
        h.restarter.arm()
        report_ready = h.restarter.coordinator.report_ready

        def slow_report_ready():
            report_ready()
            h.shutdown = True  # SIGTERM lands while the ready write is in flight

        h.restarter.coordinator.report_ready = slow_report_ready
        outcome, procs = h.fail(FakeProc())
        self.assertEqual(outcome, RestartOutcome.SHUTDOWN)
        self.assertFalse(os.path.exists(gate))

    def test_stop_during_post_start_step_keeps_the_gate_closed(self):
        gate = os.path.join(self.tmp, "gate")
        h = Harness(gate_file=gate)
        h.restarter.arm()

        def warmup():
            h.shutdown = True  # SIGTERM arrives while the warmup runs

        h.restarter._after_ready = warmup
        outcome, procs = h.fail(FakeProc())
        self.assertEqual(outcome, RestartOutcome.SHUTDOWN)
        self.assertEqual(procs, h.spawned)
        self.assertFalse(os.path.exists(gate))

    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp()


class RowConnectTest(unittest.TestCase):
    """The first connection to part0's store: interruptible, under one overall deadline."""

    def setUp(self):
        self.port = free_port()
        self.rows = []

    def tearDown(self):
        for row in reversed(self.rows):
            row.close()
            row._store = None
        self.rows.clear()

    def row(self, node, connect_timeout_s=60.0):
        row = RowCoordinator(
            2,
            node,
            "127.0.0.1",
            self.port,
            connect_timeout_s=connect_timeout_s,
            heartbeat_interval_s=0.1,
        )
        self.rows.append(row)
        return row

    def test_stop_while_waiting_for_part0_ends_the_wait(self):
        follower = self.row(1)
        stop = threading.Event()
        threading.Timer(0.5, stop.set).start()
        start = time.monotonic()
        with self.assertRaises(ShutdownRequested):
            follower.start(should_abort=stop.is_set)
        # One attempt in flight at most (plus torch's retry delay), not the 60 s startup bound.
        self.assertLess(time.monotonic() - start, 20.0)
        self.assertIsNone(follower._store)
        self.assertIsNone(follower._hb_thread)

    def test_attempts_share_one_deadline(self):
        clock = FakeClock()
        requested = []

        def factory(host, port, is_master, timeout_s):
            # Each attempt uses up exactly the timeout it was given, then fails.
            requested.append(timeout_s)
            clock.now += timeout_s
            raise RuntimeError("connect timed out")

        follower = RowCoordinator(
            2,
            1,
            "127.0.0.1",
            self.port,
            connect_timeout_s=5.0,
            store_factory=factory,
            clock=clock,
        )
        self.rows.append(follower)
        with self.assertRaises(RowStoreError):
            follower.start()
        # The timeouts handed to torch add up to the one deadline; no attempt restarts it.
        self.assertEqual(requested, [CONNECT_ATTEMPT_S, CONNECT_ATTEMPT_S, 1.0])

    def test_follower_connects_once_part0_comes_up(self):
        follower, leader = self.row(1), self.row(0)
        threading.Timer(1.5, leader.start).start()
        follower.start()
        self.assertTrue(wait_until(lambda: leader._store is not None))
        # The short attempt length does not stay as the store's operation timeout.
        self.assertEqual(follower._store._store.timeout.total_seconds(), 60.0)

    def test_stop_during_the_attempt_that_connects_still_starts_nothing(self):
        stop = threading.Event()
        attempts = []

        def factory(host, port, is_master, timeout_s):
            attempts.append(timeout_s)
            stop.set()  # SIGTERM lands while the connect is blocked, then part0 answers
            return unittest.mock.MagicMock()

        follower = RowCoordinator(2, 1, "127.0.0.1", self.port, store_factory=factory)
        self.rows.append(follower)
        with self.assertRaises(ShutdownRequested):
            follower.start(should_abort=stop.is_set)
        self.assertEqual(len(attempts), 1)
        self.assertIsNone(follower._store)
        self.assertIsNone(follower._hb_thread)

    def test_leader_retries_local_client_timeout_before_spending_restart_budget(self):
        from torch.distributed import DistNetworkError

        raw = unittest.mock.MagicMock()
        factory = unittest.mock.Mock(side_effect=[
            DistNetworkError("The client socket has timed out after 2000ms while trying to connect"),
            raw,
        ])
        leader = RowCoordinator(2, 0, "127.0.0.1", self.port, store_factory=factory)
        self.assertIs(leader._connect(lambda: False), raw)
        self.assertEqual(factory.call_count, 2)
        self.assertTrue(all(call.args[2] for call in factory.call_args_list))

    def test_part0_that_cannot_host_fails_at_once(self):
        holder = self.row(0)
        holder.start()  # takes the port
        second = self.row(0)
        start = time.monotonic()
        with self.assertRaises(RowStoreError):
            second.start()
        self.assertLess(
            time.monotonic() - start, 20.0
        )  # no retries until the 60 s deadline


class StalledStoreTest(unittest.TestCase):
    """A store call that never returns must not hold up the restart logic."""

    class Stalling:
        def __init__(self, stall_after):
            self.calls = 0
            self.stall_after = stall_after
            self.release = threading.Event()

        def _maybe_stall(self):
            self.calls += 1
            if self.calls > self.stall_after:
                self.release.wait()

        def add(self, key, value):
            self._maybe_stall()
            return 1

        def check(self, keys):
            self._maybe_stall()
            return False

        def set(self, key, value):
            self._maybe_stall()

        def set_timeout(self, timeout):
            pass

    def test_bounded_call_raises_instead_of_blocking(self):
        store = self.Stalling(stall_after=0)
        bounded = _BoundedStore(store, timeout_s=0.5)
        start = time.monotonic()
        with self.assertRaises(RowStoreError):
            bounded("add", "k", 1)
        self.assertLess(time.monotonic() - start, 3)
        with self.assertRaises(RowStoreError):  # later calls fail fast
            bounded("set", "k", "v")
        store.release.set()

    def test_stalled_heartbeat_reads_as_a_lost_row(self):
        stores = []

        def factory(host, port, is_master, timeout_s):
            # main connection never stalls; the heartbeat's stalls after a few calls
            store = self.Stalling(stall_after=10**9 if not stores else 6)
            stores.append(store)
            return store

        row = RowCoordinator(
            2,
            1,
            heartbeat_interval_s=0.05,
            heartbeat_timeout_s=0.5,
            store_factory=factory,
        )
        row.start()
        try:
            self.assertFalse(row.peer_lost())
            self.assertTrue(wait_until(row.peer_lost, timeout=5))
            start = time.monotonic()
            self.assertTrue(row.row_failure_pending())  # answered from cached state
            self.assertLess(time.monotonic() - start, 0.1)
            # A stalled coordinator is reported as such, not as a machine gone.
            self.assertIn("have not completed", row.lost_reason)
        finally:
            for store in stores:
                store.release.set()
            row.close()


class ProcessManagerRestartHookTest(unittest.TestCase):
    """How ProcessManager hands stops and failures to the restarter."""

    class Restarter:
        def __init__(self, raise_in_handle=False):
            self.stops = 0
            self.live = []
            self.raise_in_handle = raise_in_handle

        def on_shutdown(self):
            self.stops += 1

        def poll(self):
            return RowEvent.NONE

        def handle_backend_failure(self, backend, local):
            started = FakeProc()
            self.live = backend + [started]
            if self.raise_in_handle:
                raise RuntimeError("unexpected")
            return RestartOutcome.RESTARTED, [started]

    def manager(self, restarter):
        from rtp_llm.utils.process_manager import ProcessManager

        with unittest.mock.patch("signal.signal"):
            pm = ProcessManager(shutdown_timeout=1)
        pm.backend_restarter = restarter
        return pm

    def run_monitor(self, pm):
        with unittest.mock.patch.object(
            pm, "_terminate_processes"
        ), unittest.mock.patch.object(pm, "_force_kill_processes"), unittest.mock.patch(
            "time.sleep"
        ):
            pm._monitor_processes_health()

    def test_failure_cleanup_is_not_published_as_a_row_stop(self):
        restarter = self.Restarter()
        pm = self.manager(restarter)
        pm.add_process(FakeProc(), shutdown_group="frontend")
        pm.request_failure_shutdown()
        self.run_monitor(pm)
        self.assertEqual(restarter.stops, 0)

    def test_normal_stop_is_published_to_the_row(self):
        restarter = self.Restarter()
        pm = self.manager(restarter)
        pm.add_process(FakeProc(), shutdown_group="frontend")
        pm.shutdown_requested = True  # what SIGTERM does
        self.run_monitor(pm)
        self.assertEqual(restarter.stops, 1)

    def test_unexpected_restart_error_fails_with_the_started_backend_owned(self):
        restarter = self.Restarter(raise_in_handle=True)
        pm = self.manager(restarter)
        dead = FakeProc()
        dead.alive = False
        pm.add_process(dead, shutdown_group="backend")
        self.assertFalse(pm._restart_backend(local=True))
        self.assertEqual(pm.process_groups["backend"], restarter.live)


class ProcessGroupTest(unittest.TestCase):
    def test_kill_process_group_empties_it(self):
        leader = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os,subprocess,sys,time\n"
                "os.setpgid(0,0)\n"
                "[subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']) for _ in range(2)]\n"
                "time.sleep(60)\n",
            ]
        )
        try:
            self.assertTrue(
                wait_until(lambda: len(process_group_members(leader.pid)) == 3)
            )
            self.assertTrue(kill_process_group(leader.pid, timeout=10))
            leader.wait(5)
            self.assertEqual(process_group_members(leader.pid), [])
        finally:
            if leader.poll() is None:
                leader.kill()

    def test_clean_waits_for_a_backend_that_does_not_lead_its_group_yet(self):
        import multiprocessing

        proc = multiprocessing.get_context("spawn").Process(
            target=_late_group_leader, args=(2.0,)
        )
        proc.start()
        try:
            restarter = BackendRestarter(
                1,
                lambda: None,
                RowCoordinator(1, 0),
                is_shutdown_requested=lambda: False,
                cleanup_timeout_s=10,
            )
            restarter.live = [proc]
            self.assertTrue(restarter._clean())
            self.assertFalse(proc.is_alive())
            self.assertEqual(restarter.live, [])
            time.sleep(3)  # had it survived, it would now lead a group with ranks
            self.assertEqual(process_group_members(proc.pid), [])
        finally:
            if proc.is_alive():
                proc.kill()

    def test_refuses_own_group(self):
        self.assertFalse(kill_process_group(os.getpgrp(), timeout=1))


if __name__ == "__main__":
    unittest.main()
