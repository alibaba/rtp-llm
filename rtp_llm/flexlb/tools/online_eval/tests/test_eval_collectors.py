"""eval_collectors G6 (master server_latency counter poller) unit tests.

G6 is one lane of the eval_collectors process run_online_eval.sh starts
per run. The wire contract that MUST NOT drift:

  * output file master_counters_timeseries.txt, one kv line per round:
      ts_epoch_ms=<epoch_ms int> arrival_count=<int> completion_count=<int>
    — parsed by consolidate_run_outputs.parse_counter_timeseries into
    master.json["counters_timeseries"] rows, whose keys are consumed by
    aggregate_canvas_run (master_arrivals_ts differential).
  * a failed round (connection error / non-200 / bad JSON) writes NO line
    (no zero-fabricated sample);
  * a JSON body missing the counter keys writes explicit zeros;
  * urlopen happens before the timestamp is taken (started-of-round
    semantics — the line's ts is the round start, not the response time).

The tests run a scripted local HTTP server and drive either the thread
function directly or the full CLI subprocess (SIGTERM graceful stop).
"""

import contextlib
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import consolidate_run_outputs  # noqa: E402
import eval_collectors  # noqa: E402

COLLECTORS = TOOLS_DIR / "eval_collectors.py"


class _ScriptedHandler(BaseHTTPRequestHandler):
    """Serves /rtp_llm/server_latency from a scripted (status, body) list.

    One entry is consumed per request; the last entry repeats forever.
    """

    script: list = []
    requests = 0

    def do_GET(self):
        if self.path != "/rtp_llm/server_latency":
            self.send_error(404)
            return
        cls = type(self)
        cls.requests += 1
        status, body = cls.script[min(cls.requests - 1, len(cls.script) - 1)]
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass  # keep test output clean


@contextlib.contextmanager
def _serving(script):
    handler = type(
        "Handler", (_ScriptedHandler,), {"script": list(script), "requests": 0}
    )
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_address[1]}", handler
    finally:
        server.shutdown()
        server.server_close()


def _wait_for_lines(path, count, timeout_s=5.0):
    deadline = time.time() + timeout_s
    lines = []
    while time.time() < deadline:
        if path.is_file():
            lines = [
                line for line in path.read_text(encoding="utf-8").splitlines() if line
            ]
            if len(lines) >= count:
                return lines
        time.sleep(0.02)
    raise AssertionError(
        f"expected >= {count} lines in {path.name} within {timeout_s}s, got {len(lines)}"
    )


def _run_thread_until(addr, out_path, min_lines, interval=0.05, timeout_s=5.0):
    eval_collectors._STOP.clear()
    thread = threading.Thread(
        target=eval_collectors.run_master_counter_poller,
        args=(addr, str(out_path), interval),
        daemon=True,
    )
    thread.start()
    try:
        return _wait_for_lines(out_path, min_lines, timeout_s)
    finally:
        eval_collectors._STOP.set()
        thread.join(timeout=3.0)


class G6CounterPollerTest(unittest.TestCase):
    def test_line_format_matches_counters_timeseries_schema(self):
        body = '{"arrival_count": 7, "completion_count": 5, "other": "ignored"}'
        with tempfile.TemporaryDirectory() as tmp, _serving([(200, body)]) as (addr, _):
            out = Path(tmp) / "master_counters_timeseries.txt"
            lines = _run_thread_until(addr, out, min_lines=3)
            # The consolidation-side parser must reproduce the exact
            # counters_timeseries row schema (key set + int typing) the
            # aggregator consumes from master.json.
            rows = consolidate_run_outputs.parse_counter_timeseries(out)
        self.assertGreaterEqual(len(lines), 3)
        for line in lines:
            self.assertRegex(
                line, r"^ts_epoch_ms=\d+ arrival_count=7 completion_count=5$"
            )
        self.assertGreaterEqual(len(rows), 3)
        now_ms = int(time.time() * 1000)
        for row in rows:
            self.assertEqual(
                {"ts_epoch_ms", "arrival_count", "completion_count"}, set(row)
            )
            self.assertEqual(7, row["arrival_count"])
            self.assertEqual(5, row["completion_count"])
            self.assertIsInstance(row["ts_epoch_ms"], int)
            self.assertLess(abs(row["ts_epoch_ms"] - now_ms), 60_000)

    def test_missing_counter_keys_default_to_zero(self):
        with tempfile.TemporaryDirectory() as tmp, _serving([(200, "{}")]) as (addr, _):
            out = Path(tmp) / "master_counters_timeseries.txt"
            lines = _run_thread_until(addr, out, min_lines=2)
        for line in lines:
            self.assertRegex(
                line, r"^ts_epoch_ms=\d+ arrival_count=0 completion_count=0$"
            )

    def test_failed_rounds_write_no_line(self):
        # First two requests fail (HTTP 500), later ones succeed: the failed
        # rounds must be skipped entirely — no zero-fabricated sample, no
        # empty kv line.
        body = '{"arrival_count": 3, "completion_count": 2}'
        with tempfile.TemporaryDirectory() as tmp, _serving(
            [(500, "boom"), (500, "boom"), (200, body)]
        ) as (addr, handler):
            out = Path(tmp) / "master_counters_timeseries.txt"
            lines = _run_thread_until(addr, out, min_lines=3)
            requests = handler.requests
            rows = consolidate_run_outputs.parse_counter_timeseries(out)
        self.assertGreaterEqual(requests, 5)  # two failures already consumed
        for line in lines:
            self.assertRegex(
                line, r"^ts_epoch_ms=\d+ arrival_count=3 completion_count=2$"
            )
        self.assertEqual(len(rows), len(lines))

    def test_g6_only_cli_end_to_end(self):
        # G6-only invocation: no mock/prometheus/inflight/pid argv — exactly
        # what run_online_eval.sh passes when FLEXLB_SECONDARY_POLLERS_ENABLED=0
        # && START_FLEXLB=1.
        body = '{"arrival_count": 11, "completion_count": 9}'
        with tempfile.TemporaryDirectory() as tmp, _serving([(200, body)]) as (addr, _):
            out = Path(tmp) / "master_counters_timeseries.txt"
            with subprocess.Popen(
                [
                    sys.executable,
                    str(COLLECTORS),
                    "--counter-http-addr",
                    addr,
                    "--counter-out",
                    str(out),
                    "--counter-interval",
                    "0.05",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ) as proc:
                try:
                    lines = _wait_for_lines(out, 2)
                finally:
                    proc.terminate()
                    proc.wait(timeout=5.0)
                stderr = proc.stderr.read()
                self.assertEqual(0, proc.returncode, stderr)
        for line in lines:
            self.assertRegex(
                line, r"^ts_epoch_ms=\d+ arrival_count=11 completion_count=9$"
            )

    def test_cli_rejects_half_passed_counter_argv(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(COLLECTORS),
                "--counter-http-addr",
                "127.0.0.1:1",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(2, proc.returncode)
        self.assertIn(
            "--counter-http-addr and --counter-out must be passed together",
            proc.stderr,
        )

    def test_cli_rejects_removed_group_flag(self):
        # --group is gone from the CLI (single collector process, lanes are
        # argv pairs). This also guards the shell call site: if a future edit
        # reintroduced `--group secondary` into run_online_eval.sh, every
        # run would die here at startup — rc 2 instead of a silent drift.
        proc = subprocess.run(
            [sys.executable, str(COLLECTORS), "--group", "secondary"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(2, proc.returncode)
        self.assertIn("unrecognized arguments: --group", proc.stderr)

    def test_sigterm_burst_exits_cleanly(self):
        # Regression (SIGTERM re-entry deadlock): run_online_eval.sh's stop
        # loop used to send one SIGTERM per legacy PID variable — up to 5
        # rapid SIGTERMs to the SAME unified collector process. A second
        # signal landing inside the handler's Event.set() window re-entered
        # the handler on the Event's non-reentrant Condition lock and froze
        # the whole process (one leaked frozen collector per run; py-spy
        # showed 3-level recursive handler frames). The handler now masks
        # both stop signals on entry, so a burst must still terminate
        # gracefully: in-flight round finishes, output flushed, exit 0.
        # proc.wait(timeout=...) failing (TimeoutExpired) is the deadlock
        # detector here.
        body = '{"arrival_count": 4, "completion_count": 3}'
        with tempfile.TemporaryDirectory() as tmp, _serving([(200, body)]) as (addr, _):
            out = Path(tmp) / "master_counters_timeseries.txt"
            with subprocess.Popen(
                [
                    sys.executable,
                    str(COLLECTORS),
                    "--counter-http-addr",
                    addr,
                    "--counter-out",
                    str(out),
                    "--counter-interval",
                    "0.05",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ) as proc:
                try:
                    _wait_for_lines(out, 2)
                finally:
                    # 5 back-to-back SIGTERMs, no spacing — the worst the old
                    # shell kill loop could do (same semantics as `kill pid` x5).
                    for _ in range(5):
                        try:
                            proc.send_signal(signal.SIGTERM)
                        except ProcessLookupError:
                            pass  # already gone — also a graceful outcome
                    proc.wait(timeout=10.0)
                stderr = proc.stderr.read()
                self.assertEqual(0, proc.returncode, stderr)

    def test_request_stop_masks_both_signals_and_is_idempotent(self):
        # Handler semantics, driven directly: on entry it must mask BOTH stop
        # signals (a same-signal burst AND a SIGTERM/SIGINT cross must not be
        # able to nest a second Event.set()), it must be safe to invoke
        # repeatedly, and it must leave _STOP set. The test process's own
        # signal dispositions are restored afterwards.
        old_term = signal.getsignal(signal.SIGTERM)
        old_int = signal.getsignal(signal.SIGINT)
        eval_collectors._STOP.clear()
        try:
            eval_collectors._request_stop(signal.SIGTERM, None)
            self.assertIs(signal.SIG_IGN, signal.getsignal(signal.SIGTERM))
            self.assertIs(signal.SIG_IGN, signal.getsignal(signal.SIGINT))
            self.assertTrue(eval_collectors._STOP.is_set())
            # Second invocation (the re-entry the burst would cause): must
            # return without raising and keep the flag set.
            eval_collectors._request_stop(signal.SIGINT, None)
            self.assertTrue(eval_collectors._STOP.is_set())
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
            eval_collectors._STOP.clear()


if __name__ == "__main__":
    unittest.main()
