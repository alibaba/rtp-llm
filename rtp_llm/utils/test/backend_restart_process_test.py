"""Process-level contract of in-place backend restart, with real processes and signals.

Each supervisor below is what start_server does around the model: a ProcessManager with a
BackendRestarter, a frontend that must survive, and a backend that leads its own process
group with two "rank" children. A backend exits 1 when one of its ranks dies, as the real
one does. Several supervisors on one TCPStore stand for the machines of one row.
"""

import glob
import json
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import time
import unittest

from rtp_llm.test.utils.port_util import PortManager
from rtp_llm.utils.backend_restart import (
    BackendRestarter,
    RowCoordinator,
    ShutdownRequested,
    process_group_members,
)
from rtp_llm.utils.process_lifecycle import arm_parent_death_sigkill
from rtp_llm.utils.process_manager import ProcessManager

CTX = multiprocessing.get_context("spawn")


def _fake_backend(workdir, node, parent_pid, pipe_writer):
    arm_parent_death_sigkill(parent_pid)
    os.setpgid(0, 0)
    ranks = [
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
        for _ in range(2)
    ]
    with open(os.path.join(workdir, f"backend_{node}_{os.getpid()}.json"), "w") as f:
        json.dump({"pgid": os.getpgrp(), "ranks": [r.pid for r in ranks]}, f)
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # like the real backend's deferral
    pipe_writer.send({"status": "success"})
    pipe_writer.close()
    while True:
        if stop:
            for r in ranks:
                r.kill()
            os._exit(0)
        if any(r.poll() is not None for r in ranks):
            for r in ranks:
                r.kill()
            os._exit(1)
        time.sleep(0.05)


def _fake_frontend(workdir, node):
    with open(os.path.join(workdir, f"frontend_{node}.pid"), "w") as f:
        f.write(str(os.getpid()))
    while True:
        time.sleep(1)


def _supervisor(workdir, node, num_nodes, port, limit, backoff_s):
    logging.basicConfig(level=logging.INFO)
    pm = ProcessManager(shutdown_timeout=10, backend_post_frontend_drain_seconds=0)
    gate = os.path.join(workdir, f"gate_{node}")
    coordinator = RowCoordinator(
        num_nodes,
        node,
        "127.0.0.1",
        port,
        connect_timeout_s=30,
        heartbeat_interval_s=0.2,
        heartbeat_timeout_s=3.0,
        barrier_timeout_s=20.0,
        store_op_timeout_s=3.0,
    )
    open(os.path.join(workdir, f"connecting_{node}"), "w").close()
    try:
        coordinator.start(should_abort=lambda: pm.shutdown_requested)
    except ShutdownRequested:  # as start_server: stopped while waiting, no backend
        pm.monitor_and_release_processes()
        open(os.path.join(workdir, f"clean_exit_{node}"), "w").close()
        return

    def spawn():
        reader, writer = CTX.Pipe(duplex=False)
        proc = CTX.Process(
            target=_fake_backend, args=(workdir, node, os.getpid(), writer)
        )
        proc.start()
        writer.close()
        return proc, lambda: reader.poll(0) and reader.recv()["status"] == "success"

    restarter = BackendRestarter(
        limit,
        spawn,
        coordinator,
        is_shutdown_requested=lambda: pm.shutdown_requested,
        gate_file=gate,
        backoff_base_s=backoff_s,
        cleanup_timeout_s=10,
    )
    pm.backend_restarter = restarter
    frontend = CTX.Process(target=_fake_frontend, args=(workdir, node))
    frontend.start()
    pm.add_process(frontend, shutdown_group="frontend")
    backend, ready = spawn()
    pm.add_process(backend, shutdown_group="backend")
    while not ready():
        time.sleep(0.05)
    open(gate, "w").close()
    restarter.arm()
    pm.monitor_and_release_processes()  # exits 1 itself on failure
    coordinator.close()
    open(os.path.join(workdir, f"clean_exit_{node}"), "w").close()


_PORT_LOCKS = []


def _free_port() -> int:
    # The lock stays held for the whole test process, so concurrent tests never share a port.
    ports, locks = PortManager().get_consecutive_ports(1)
    _PORT_LOCKS.extend(locks)
    return ports[0]


def _wait(predicate, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


class BackendRestartProcessTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.workdir = tempfile.mkdtemp()
        self.supervisors = []

    def tearDown(self):
        for sup in self.supervisors:
            if sup.is_alive():
                sup.kill()
                sup.join(5)
        for path in glob.glob(os.path.join(self.workdir, "backend_*.json")):
            with open(path) as f:
                pgid = json.load(f)["pgid"]
            for pid in process_group_members(pgid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        # Frontends have no parent-death signal: a SIGKILLed supervisor leaves them behind.
        for path in glob.glob(os.path.join(self.workdir, "frontend_*.pid")):
            with open(path) as f:
                pid = int(f.read())
            if _alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def start_row(self, num_nodes=1, limit=2, backoff_s=0.1):
        port = _free_port()
        for node in range(num_nodes):
            sup = CTX.Process(
                target=_supervisor,
                args=(self.workdir, node, num_nodes, port, limit, backoff_s),
            )
            sup.start()
            self.supervisors.append(sup)
        for node in range(num_nodes):
            self.assertTrue(_wait(lambda: len(self.backends(node)) == 1))
            self.assertTrue(_wait(lambda: os.path.exists(self.path(f"gate_{node}"))))
        return self.supervisors

    def path(self, name):
        return os.path.join(self.workdir, name)

    def backends(self, node):
        result = []
        for path in sorted(
            glob.glob(self.path(f"backend_{node}_*.json")), key=os.path.getmtime
        ):
            with open(path) as f:
                result.append(json.load(f))
        return result

    def frontend_pid(self, node):
        with open(self.path(f"frontend_{node}.pid")) as f:
            return int(f.read())

    def kill_rank(self, node, generation):
        os.kill(self.backends(node)[generation]["ranks"][0], signal.SIGKILL)

    def assert_group_empty(self, backend):
        self.assertTrue(_wait(lambda: not process_group_members(backend["pgid"]), 15))

    def test_rank_death_restarts_only_the_backend_then_sigterm_exits_zero(self):
        (sup,) = self.start_row()
        frontend = self.frontend_pid(0)
        self.kill_rank(0, 0)
        self.assertTrue(_wait(lambda: len(self.backends(0)) == 2))
        self.assert_group_empty(self.backends(0)[0])
        self.assertTrue(_wait(lambda: os.path.exists(self.path("gate_0"))))
        self.assertTrue(_alive(frontend))
        os.kill(sup.pid, signal.SIGTERM)
        sup.join(60)
        self.assertEqual(sup.exitcode, 0)
        self.assertTrue(os.path.exists(self.path("clean_exit_0")))
        self.assert_group_empty(self.backends(0)[1])
        self.assertFalse(_alive(frontend))

    def test_exhausted_budget_exits_one_and_leaves_nothing(self):
        (sup,) = self.start_row(limit=1)
        self.kill_rank(0, 0)
        self.assertTrue(_wait(lambda: len(self.backends(0)) == 2))
        self.assertTrue(_wait(lambda: os.path.exists(self.path("gate_0"))))
        self.kill_rank(0, 1)
        sup.join(60)
        self.assertEqual(sup.exitcode, 1)
        self.assertFalse(os.path.exists(self.path("clean_exit_0")))
        for backend in self.backends(0):
            self.assert_group_empty(backend)

    def test_frontend_death_is_not_restarted(self):
        (sup,) = self.start_row()
        os.kill(self.frontend_pid(0), signal.SIGKILL)
        sup.join(60)
        self.assertEqual(sup.exitcode, 1)
        self.assertEqual(len(self.backends(0)), 1)

    def test_sigterm_during_backoff_takes_the_normal_stop(self):
        (sup,) = self.start_row(backoff_s=30)
        self.kill_rank(0, 0)
        self.assertTrue(_wait(lambda: not os.path.exists(self.path("gate_0")), 15))
        time.sleep(1)
        os.kill(sup.pid, signal.SIGTERM)
        sup.join(60)
        self.assertEqual(sup.exitcode, 0)
        self.assertTrue(os.path.exists(self.path("clean_exit_0")))
        self.assertEqual(len(self.backends(0)), 1)

    def test_row_restarts_together_and_one_stop_stops_the_row(self):
        sup0, sup1 = self.start_row(num_nodes=2)
        self.kill_rank(1, 0)  # a rank of the non-part0 machine
        for node in (0, 1):
            self.assertTrue(_wait(lambda: len(self.backends(node)) == 2))
            self.assert_group_empty(self.backends(node)[0])
        self.kill_rank(0, 1)  # now a rank on part0
        for node in (0, 1):
            self.assertTrue(_wait(lambda: len(self.backends(node)) == 3))
        os.kill(sup1.pid, signal.SIGTERM)
        for sup in (sup0, sup1):
            sup.join(60)
            self.assertEqual(sup.exitcode, 0)
        for node in (0, 1):
            self.assertTrue(os.path.exists(self.path(f"clean_exit_{node}")))
            self.assert_group_empty(self.backends(node)[2])

    def test_losing_a_machine_fails_the_row(self):
        sup0, sup1 = self.start_row(num_nodes=2)
        os.kill(sup1.pid, signal.SIGKILL)  # its backend follows by parent-death signal
        sup0.join(60)
        self.assertEqual(sup0.exitcode, 1)
        self.assertEqual(len(self.backends(0)), 1)

    def test_losing_part0_fails_the_row(self):
        sup0, sup1 = self.start_row(num_nodes=2)
        os.kill(sup0.pid, signal.SIGKILL)  # takes the row's coordination store with it
        sup1.join(60)
        self.assertEqual(sup1.exitcode, 1)

    def test_a_stalled_part0_fails_the_row_in_bounded_time(self):
        sup0, sup1 = self.start_row(num_nodes=2)
        os.kill(
            sup0.pid, signal.SIGSTOP
        )  # the store stops answering, connections stay up
        try:
            start = time.monotonic()
            sup1.join(60)
            self.assertEqual(sup1.exitcode, 1)
            self.assertLess(time.monotonic() - start, 30)
        finally:
            os.kill(sup0.pid, signal.SIGKILL)

    def test_sigterm_while_waiting_for_part0_exits_zero_without_a_backend(self):
        # part1 starts first and waits for a part0 that never comes; the operator stops it.
        sup1 = CTX.Process(
            target=_supervisor,
            args=(self.workdir, 1, 2, _free_port(), 2, 0.1),
        )
        sup1.start()
        self.supervisors.append(sup1)
        self.assertTrue(_wait(lambda: os.path.exists(self.path("connecting_1"))))
        time.sleep(1.0)  # inside a connect attempt
        start = time.monotonic()
        os.kill(sup1.pid, signal.SIGTERM)
        sup1.join(20)
        self.assertEqual(sup1.exitcode, 0)
        # exit 0, not the 1 of the 30 s connect bound; time is only sane-checked (torch retry delay)
        self.assertLess(time.monotonic() - start, 25)
        self.assertTrue(os.path.exists(self.path("clean_exit_1")))
        self.assertEqual(self.backends(1), [])


if __name__ == "__main__":
    unittest.main()
