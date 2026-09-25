"""Real Flume transport with spawned reporters; no GPU or external collector needed."""

import importlib
import multiprocessing
import socketserver
import struct
import threading
import unittest
from unittest.mock import patch

from thrift.protocol import TCompactProtocol
from thrift.transport import TTransport

from rtp_llm.aios.kmonitor.python_client.flume import ThriftSourceProtocol
from rtp_llm.aios.kmonitor.python_client.kmonitor import reporting
from rtp_llm.aios.kmonitor.python_client.kmonitor.utils.hippo_helper import HippoHelper


def _reporter(state, port, control):
    reporting.configure(state)
    # Importing this module creates its default reporter. Keep that reporter
    # offline even when the test inherits production HIPPO environment values.
    with patch.object(HippoHelper, "is_hippo_env", return_value=False):
        module = importlib.import_module(
            "rtp_llm.aios.kmonitor.python_client.kmonitor.report_worker"
        )
    module.report_worker.stop()
    from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.acc_metric import (
        AccMetric,
    )
    from rtp_llm.aios.kmonitor.python_client.kmonitor.metrics.gauge_metric import (
        GaugeMetric,
    )

    # Drive real sends explicitly so assertions do not depend on the one-second
    # background tick. Only endpoint selection and scheduling are substituted.
    with patch.object(module.ReportWorker, "start"), patch.object(
        module.HippoHelper, "is_hippo_env", return_value=True
    ), patch.object(module, "_ReportWorker__REPORT_HOST", "127.0.0.1"), patch.object(
        module, "_ReportWorker__REPORT_PORT", port
    ):
        worker = module.ReportWorker()
        gauge = GaugeMetric("transport_load", {})
        counter = AccMetric("transport_qps", {})
        worker.register_metric(gauge)
        worker.register_metric(counter)
        control.send("ready")
        try:
            while True:
                command, value = control.recv()
                if command == "stop":
                    return
                if command in ("queue", "flush"):
                    gauge.report(value)
                    counter.report(value)
                if command == "flush":
                    worker.do_report()
                control.send("done")
        finally:
            if worker.flume is not None:
                worker.flume.close()
            control.close()


class _Flume(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self):
        self.condition = threading.Condition()
        self.opened = 0
        self.closed = 0
        self.batches = 0
        self.values = []
        self.errors = []
        super().__init__(("127.0.0.1", 0), _Handler)

    def wait_for(self, predicate):
        with self.condition:
            if not self.condition.wait_for(predicate, timeout=10):
                raise AssertionError(
                    f"Flume timeout: opened={self.opened}, closed={self.closed}, "
                    f"batches={self.batches}, values={self.values}, errors={self.errors}"
                )
            if self.errors:
                raise AssertionError(self.errors)


class _Handler(socketserver.BaseRequestHandler):
    def _read(self, size):
        data = b""
        while len(data) < size:
            chunk = self.request.recv(size - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def handle(self):
        server = self.server
        with server.condition:
            server.opened += 1
            server.condition.notify_all()
        try:
            while True:
                header = self._read(4)
                if header is None:
                    return
                size = struct.unpack("!I", header)[0]
                if not 0 < size < 1024 * 1024:
                    raise ValueError(f"invalid frame size {size}")
                frame = self._read(size)
                if frame is None:
                    raise ValueError("truncated Flume frame")
                protocol = TCompactProtocol.TCompactProtocol(
                    TTransport.TMemoryBuffer(frame)
                )
                method, _, _ = protocol.readMessageBegin()
                if method != "appendBatch":
                    raise ValueError(method)
                args = ThriftSourceProtocol.appendBatch_args()
                args.read(protocol)
                protocol.readMessageEnd()
                with server.condition:
                    server.batches += 1
                    for event in args.events:
                        name, _, value, *_ = event.body.decode().split()
                        server.values.append((name, float(value)))
                    server.condition.notify_all()
                # The Python client intentionally sends without reading a reply.
        except Exception as error:
            with server.condition:
                server.errors.append(repr(error))
        finally:
            with server.condition:
                server.closed += 1
                server.condition.notify_all()


class SleepReportingTransportTest(unittest.TestCase):
    def setUp(self):
        self.context = multiprocessing.get_context("spawn")
        self.server = _Flume()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.children = []

    def tearDown(self):
        for child, control in self.children:
            if child.is_alive():
                try:
                    control.send(("stop", None))
                except (BrokenPipeError, OSError):
                    pass
            child.join(5)
            if child.is_alive():
                child.terminate()
                child.join(5)
            control.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)

    def start_reporter(self, state):
        parent, child_control = self.context.Pipe()
        child = self.context.Process(
            target=_reporter,
            args=(state, self.server.server_address[1], child_control),
        )
        child.start()
        child_control.close()
        self.children.append((child, parent))
        self.assertTrue(parent.poll(15), "reporter failed to start")
        self.assertEqual(parent.recv(), "ready")
        return parent

    def command(self, controls, command, value):
        for control in controls:
            control.send((command, value))
        for control in controls:
            self.assertTrue(control.poll(10), "reporter blocked")
            self.assertEqual(control.recv(), "done")

    def test_sleep_disabled_preserves_connections_and_publication(self):
        controls = [self.start_reporter(None) for _ in range(2)]
        for round_number in range(3):
            self.command(controls, "flush", 37)
            self.server.wait_for(lambda: self.server.batches == 2 * (round_number + 1))
        self.assertEqual(self.server.opened, 2)
        self.assertEqual(self.server.closed, 0)
        self.assertEqual(
            [value for name, value in self.server.values if name == "transport_load"],
            [37] * 6,
        )

    def test_all_rank_sleep_closes_connections_and_wake_sends_fresh_samples(self):
        # Rank count changes the all-rank completion gate; actual GPU execution
        # and the C++ SDK remain covered by their separate service E2E/tests.
        for rank_count in (1, 2, 8):
            with self.subTest(rank_count=rank_count):
                state = reporting.ReportingState(rank_count, self.context)
                opened = self.server.opened
                controls = [self.start_reporter(state) for _ in range(2)]
                batches = self.server.batches
                self.command(controls, "flush", 37)
                self.server.wait_for(lambda: self.server.batches == batches + 2)
                self.command(controls, "queue", 9001)
                for rank in range(rank_count):
                    state.set_rank_enabled(rank, False)
                self.command(controls, "flush", 0)
                self.server.wait_for(lambda: self.server.closed == self.server.opened)
                sleeping_batches = self.server.batches
                # A reporter started while asleep must not connect at all.
                controls.append(self.start_reporter(state))
                for rank in range(rank_count - 1):
                    state.set_rank_enabled(rank, True)
                self.command(controls, "flush", 0)
                self.assertEqual(self.server.opened, opened + 2)
                self.assertEqual(self.server.batches, sleeping_batches)
                state.set_rank_enabled(rank_count - 1, True)
                before_values = len(self.server.values)
                self.command(controls, "flush", 51)
                self.server.wait_for(
                    lambda: self.server.batches == sleeping_batches + 3
                )
                self.assertEqual(self.server.opened, opened + 5)
                self.assertEqual(
                    [
                        value
                        for name, value in self.server.values[before_values:]
                        if name == "transport_load"
                    ],
                    [51] * 3,
                )
                for control in controls:
                    control.send(("stop", None))
                self.server.wait_for(lambda: self.server.closed == self.server.opened)


if __name__ == "__main__":
    unittest.main()
