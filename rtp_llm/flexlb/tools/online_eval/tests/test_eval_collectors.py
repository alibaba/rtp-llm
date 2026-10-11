"""eval_collectors G3 (master prometheus lane) unit tests.

G3 is one lane of the eval_collectors process run_online_eval.sh starts per
run — and, since the G6/G4 collapse, the sole master-plane collector. The
wire contract that MUST NOT drift:

  * output file master_prometheus_timeseries.prom, one "# ts=<epoch_ms>"
    grouped block per round, block body = the whitelisted sample lines
    verbatim — parsed by consolidate_run_outputs
    .parse_grouped_prometheus_timeseries into master.json
    ["prometheus_timeseries"], whose rows feed aggregate_canvas_run
    (master_arrivals_ts counter differencing, inflight_ts gauge sums, and
    the pre-existing queue/KV/age series);
  * the whitelist (MASTER_PROMETHEUS_PREFIXES) carries exactly the series
    the aggregate consumes — the G6/G4 collapse added the
    auto_tpm_request_count / all_qps counters and the five inflight gauges
    — so an unconsumed flexlb series or a jvm.* line must be filtered out;
  * a failed round (connection error / non-200) writes NO block;
  * urlopen happens before the timestamp is taken (started-of-round
    semantics — the block's ts is the round start, not the response time).

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

import eval_collectors  # noqa: E402

COLLECTORS = TOOLS_DIR / "eval_collectors.py"

# A management-port /actuator/prometheus sample: whitelisted series from
# every family (incl. the G6/G4-collapse additions) plus two lines that the
# whitelist must filter out (an unconsumed flexlb series, a jvm.* line).
PROM_BODY = "\n".join(
    [
        'flexlb_app_cache_used_kv_cache_tokens{role="PREFILL",engineIp="10.0.0.1"} 1000.0',
        "flexlb_app_cache_hit_ratio 0.5",
        'flexlb_app_engine_balancing_master_dispatch_reason_total{reason="batch_full"} 2.0',
        'flexlb_auto_tpm_request_count_total{priority="50"} 42.0',
        'flexlb_app_engine_balancing_master_all_qps_total{code="ok"} 40.0',
        "flexlb_app_flexlb_scheduler_inflight_size 7.0",
        'flexlb_app_flexlb_inflight_batch_count{engineIp="10.0.0.1"} 3.0',
        'flexlb_app_flexlb_inflight_request_count{engineIp="10.0.0.1"} 30.0',
        'flexlb_auto_tpm_decode_reserved_count{endpoint="10.0.0.2:1"} 5.0',
        'flexlb_auto_tpm_decode_running_count{endpoint="10.0.0.2:1"} 4.0',
        "flexlb_app_unrelated_metric 999.0",
        "jvm_memory_used_bytes 12345.0",
    ]
)


class _ScriptedHandler(BaseHTTPRequestHandler):
    """Serves /actuator/prometheus from a scripted (status, body) list.

    One entry is consumed per request; the last entry repeats forever.
    """

    script: list = []
    requests = 0

    def do_GET(self):
        if self.path != "/actuator/prometheus":
            self.send_error(404)
            return
        cls = type(self)
        cls.requests += 1
        status, body = cls.script[min(cls.requests - 1, len(cls.script) - 1)]
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
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
        yield server.server_address[1], handler
    finally:
        server.shutdown()
        server.server_close()


def _wait_for_blocks(path, count, timeout_s=5.0):
    deadline = time.time() + timeout_s
    blocks = []
    while time.time() < deadline:
        if path.is_file():
            blocks = [
                block
                for block in path.read_text(encoding="utf-8").split("# ts=")
                if block.strip()
            ]
            if len(blocks) >= count:
                return blocks
        time.sleep(0.02)
    raise AssertionError(
        f"expected >= {count} '# ts=' blocks in {path.name} within {timeout_s}s, "
        f"got {len(blocks)}"
    )


def _run_prom_thread_until(port, out_path, min_blocks, interval=0.05, timeout_s=5.0):
    eval_collectors._STOP.clear()
    thread = threading.Thread(
        target=eval_collectors.run_master_prometheus_poller,
        args=(port, str(out_path), interval),
        daemon=True,
    )
    thread.start()
    try:
        return _wait_for_blocks(out_path, min_blocks, timeout_s)
    finally:
        eval_collectors._STOP.set()
        thread.join(timeout=3.0)


class G3PrometheusLaneTest(unittest.TestCase):
    def test_whitelist_keeps_consumed_series_and_filters_the_rest(self):
        # The G3 whitelist is the consumed-set contract: every analyzer-fed
        # family (incl. the G6/G4-collapse counter/gauge additions) rides
        # through verbatim; an unconsumed flexlb series and jvm.* lines are
        # dropped before anything is appended.
        with tempfile.TemporaryDirectory() as tmp, _serving([(200, PROM_BODY)]) as (
            port,
            _,
        ):
            out = Path(tmp) / "master_prometheus_timeseries.prom"
            blocks = _run_prom_thread_until(port, out, min_blocks=2)
        self.assertGreaterEqual(len(blocks), 2)
        for block in blocks:
            # split("# ts=") leaves the ts value as the block's first line.
            lines = block.splitlines()[1:]
            for line in lines:
                self.assertNotIn("flexlb_app_unrelated_metric", line)
                self.assertNotIn("jvm_memory_used_bytes", line)
                self.assertTrue(
                    line.startswith(eval_collectors.MASTER_PROMETHEUS_PREFIXES)
                )
            joined = "\n".join(lines)
            for kept in (
                "flexlb_app_cache_used_kv_cache_tokens",
                "flexlb_auto_tpm_request_count_total",
                "flexlb_app_engine_balancing_master_all_qps_total",
                "flexlb_app_flexlb_scheduler_inflight_size",
                "flexlb_app_flexlb_inflight_batch_count",
                "flexlb_app_flexlb_inflight_request_count",
                "flexlb_auto_tpm_decode_reserved_count",
                "flexlb_auto_tpm_decode_running_count",
                "flexlb_app_engine_balancing_master_dispatch_reason_total",
            ):
                self.assertIn(kept, joined)

    def test_failed_rounds_write_no_block(self):
        # First two requests fail (HTTP 500), later ones succeed: the failed
        # rounds must be skipped entirely — no empty block, no fabricated
        # sample.
        with tempfile.TemporaryDirectory() as tmp, _serving(
            [(500, "boom"), (500, "boom"), (200, PROM_BODY)]
        ) as (port, handler):
            out = Path(tmp) / "master_prometheus_timeseries.prom"
            blocks = _run_prom_thread_until(port, out, min_blocks=3)
            requests = handler.requests
        self.assertGreaterEqual(requests, 5)  # two failures already consumed
        for block in blocks:
            self.assertIn("flexlb_auto_tpm_request_count_total", block)

    def test_g3_only_cli_end_to_end(self):
        # G3-only invocation: no mock/pid argv — exactly what
        # run_online_eval.sh passes when FLEXLB_SECONDARY_POLLERS_ENABLED=0
        # && START_FLEXLB=1 (the M7 exemption keeps the master-plane data
        # source alive on A/B-off baselines).
        with tempfile.TemporaryDirectory() as tmp, _serving([(200, PROM_BODY)]) as (
            port,
            _,
        ):
            out = Path(tmp) / "master_prometheus_timeseries.prom"
            with subprocess.Popen(
                [
                    sys.executable,
                    str(COLLECTORS),
                    "--prometheus-port",
                    str(port),
                    "--prometheus-out",
                    str(out),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ) as proc:
                try:
                    blocks = _wait_for_blocks(out, 2)
                finally:
                    proc.terminate()
                    proc.wait(timeout=5.0)
                stderr = proc.stderr.read()
                self.assertEqual(0, proc.returncode, stderr)
        self.assertGreaterEqual(len(blocks), 2)
        for block in blocks:
            self.assertIn("flexlb_auto_tpm_request_count_total", block)
            self.assertNotIn("flexlb_app_unrelated_metric", block)

    def test_cli_rejects_half_passed_prometheus_argv(self):
        proc = subprocess.run(
            [sys.executable, str(COLLECTORS), "--prometheus-port", "1234"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(2, proc.returncode)
        self.assertIn(
            "--prometheus-port and --prometheus-out must be passed together",
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

    def test_cli_rejects_removed_counter_and_inflight_flags(self):
        # The G6 counter and G4 inflight lanes were collapsed into G3, so
        # their argv pairs are gone from the CLI. This locks the removal AND
        # guards the shell call site: a stale --counter-*/--inflight-* pair
        # in run_online_eval.sh would kill the collector at startup (rc 2)
        # instead of silently collecting nothing.
        for stale_flag in ("--counter-http-addr", "--inflight-http-addr"):
            proc = subprocess.run(
                [sys.executable, str(COLLECTORS), stale_flag, "127.0.0.1:1"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(2, proc.returncode, stale_flag)
            self.assertIn(f"unrecognized arguments: {stale_flag}", proc.stderr)

    def test_sigterm_burst_exits_cleanly(self):
        # Regression (SIGTERM re-entry deadlock): run_online_eval.sh's stop
        # loop used to send one SIGTERM per PID variable — several rapid
        # SIGTERMs to the SAME unified collector process. A second signal
        # landing inside the handler's Event.set() window re-entered the
        # handler on the Event's non-reentrant Condition lock and froze the
        # whole process (one leaked frozen collector per run; py-spy showed
        # 3-level recursive handler frames). The handler now masks both stop
        # signals on entry, so a burst must still terminate gracefully:
        # in-flight round finishes, output flushed, exit 0.
        # proc.wait(timeout=...) failing (TimeoutExpired) is the deadlock
        # detector here.
        with tempfile.TemporaryDirectory() as tmp, _serving([(200, PROM_BODY)]) as (
            port,
            _,
        ):
            out = Path(tmp) / "master_prometheus_timeseries.prom"
            with subprocess.Popen(
                [
                    sys.executable,
                    str(COLLECTORS),
                    "--prometheus-port",
                    str(port),
                    "--prometheus-out",
                    str(out),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ) as proc:
                try:
                    _wait_for_blocks(out, 2)
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
