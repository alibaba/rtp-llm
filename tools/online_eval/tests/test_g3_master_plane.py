"""G6/G4 collapse — G3 master-plane sole-source tests (aggregate side).

Since the G6 counter poller and G4 inflight poller were collapsed into the
G3 prometheus lane, aggregate_canvas_run derives master_arrivals_ts and
inflight_ts from the G3 prometheus timeline — the sole master-plane source
(the legacy counters_timeseries / inflight_timeseries master.json keys are
no longer read; old runs cannot be re-aggregated):

  * master_arrivals_ts — flexlb_auto_tpm_request_count_total{priority} and
    flexlb_app_engine_balancing_master_all_qps_total{code} label variants
    are summed per sample into cumulative series, then the pre-existing
    differencing runs unchanged: positive delta / interval seconds, a
    negative delta (counter reset) drops the interval, completions clamped
    at 0;
  * inflight_ts — scheduler_inflight_size direct, per-engine prefill
    batch/request counts and per-endpoint decode reserved/running gauges
    summed per sample (2s SchedulerRuntime cadence rides along);
  * legacy keys ignored — a run dir that still carries counters_timeseries /
    inflight_timeseries (old runs) rebuilds from the G3 timeline anyway.

Output key names/structures are unchanged (downstream canvas/compare_twin
consume them as-is).
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
AGGREGATE = TOOLS_DIR / "aggregate_canvas_run.py"
T0 = 1_788_283_848_000  # epoch ms anchor (== first client send)

ARR_BASE = "flexlb_auto_tpm_request_count_total"
QPS_BASE = "flexlb_app_engine_balancing_master_all_qps_total"
SCHED_BASE = "flexlb_app_flexlb_scheduler_inflight_size"
PB_BASE = "flexlb_app_flexlb_inflight_batch_count"
PR_BASE = "flexlb_app_flexlb_inflight_request_count"
DRV_BASE = "flexlb_auto_tpm_decode_reserved_count"
DRN_BASE = "flexlb_auto_tpm_decode_running_count"

REQ_P50 = ARR_BASE + '{priority="50"}'
REQ_P10 = ARR_BASE + '{priority="10"}'
QPS_OK = QPS_BASE + '{code="ok"}'

# 1s counter timeline (priority/code label variants summed per sample):
#   cumulative arrivals 0 -> 15 -> 40 -> 25(reset) -> 50
#   cumulative completions 0 -> 8 -> 36 -> 20 -> 44
# plus 2s inflight gauge frames (per-engine/per-endpoint variants summed).
PROM_TIMELINE = [
    {REQ_P50: 0.0, REQ_P10: 0.0, QPS_OK: 0.0},
    {
        REQ_P50: 10.0,
        REQ_P10: 5.0,
        QPS_OK: 8.0,
        SCHED_BASE: 2.0,
        PB_BASE + '{engineIp="10.0.0.1"}': 3.0,
        PB_BASE + '{engineIp="10.0.0.2"}': 4.0,
        PR_BASE + '{engineIp="10.0.0.1"}': 30.0,
        PR_BASE + '{engineIp="10.0.0.2"}': 40.0,
        DRV_BASE + '{endpoint="ep-a"}': 5.0,
        DRV_BASE + '{endpoint="ep-b"}': 6.0,
        DRN_BASE + '{endpoint="ep-a"}': 4.0,
        DRN_BASE + '{endpoint="ep-b"}': 5.0,
    },
    {REQ_P50: 30.0, REQ_P10: 10.0, QPS_OK: 36.0},
    {
        REQ_P50: 25.0,
        REQ_P10: 0.0,
        QPS_OK: 20.0,
        SCHED_BASE: 4.0,
        PB_BASE + '{engineIp="10.0.0.1"}': 2.0,
        PB_BASE + '{engineIp="10.0.0.2"}': 2.0,
        PR_BASE + '{engineIp="10.0.0.1"}': 10.0,
        PR_BASE + '{engineIp="10.0.0.2"}': 20.0,
        DRV_BASE + '{endpoint="ep-a"}': 1.0,
        DRV_BASE + '{endpoint="ep-b"}': 1.0,
        DRN_BASE + '{endpoint="ep-a"}': 0.0,
        DRN_BASE + '{endpoint="ep-b"}': 1.0,
    },
    {REQ_P50: 45.0, REQ_P10: 5.0, QPS_OK: 44.0},
]
PROM_TS = [T0 - 1000, T0, T0 + 1000, T0 + 2000, T0 + 3000]

MASTER_LOG = (
    # window row (10s snapshot; fail-closed zero rows is a hard error)
    "2026-09-03 01:30:31,123 flexlb_server_schedule_latency count=3 "
    "arrival_qps=1.0 completion_qps=1.0 server_p50_ms=1.0 "
    "server_p95_ms=2.0 server_p99_ms=3.0 grpc_queue_p95_ms=0.5 "
    "route_submit_p95_ms=0.5 batch_wait_p95_ms=0.5 "
    "dispatch_ack_p95_ms=0.5 ack_response_p95_ms=0.5\n"
)


def _run_aggregate(cwd):
    proc = subprocess.run(
        [sys.executable, str(AGGREGATE)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "aggregate failed:\nstdout="
        + proc.stdout[-2000:]
        + "\nstderr="
        + proc.stderr[-2000:]
    )
    return json.loads(proc.stdout)


def _client_row(i, send):
    return {
        "rid": "r%d" % i,
        "request_id": i,
        "status": "ok",
        "error": "",
        "send_start_epoch_ms": send,
        "input_len": 80,
        "output_len": 8,
        "wall_clock_ts": 1.0 + i,
        "ttft_ms": 0,
        "total_ms": 200,
        "schedule_ms": 5,
    }


def _ev_prefill(rid, send):
    return {
        "event": "prefill_done",
        "rid": rid,
        "engine_name": "p1",
        "batch_id": rid,
        "engine_arrival_ms": send + 10,
        "prefill_start_ms": send + 20,
        "prefill_done_ms": send + 30,
        "ttft_ms": 20,
        "exec_ms": 10,
        "batch_size": 1,
        "input_len": 80,
        "cache_hit_tokens": 0,
        "kv_used_tokens": 80,
        "cancelled": False,
    }


def _ev_decode(rid, send):
    return {
        "event": "decode_done",
        "rid": rid,
        "engine_name": "d1",
        "batch_id": 100 + rid,
        "engine_arrival_ms": send + 40,
        "decode_start_ms": send + 50,
        "decode_done_ms": send + 80,
        "exec_ms": 30,
        "batch_size": 1,
        "output_len": 8,
        "kv_used_tokens": 88,
        "cancelled": False,
    }


def _write_scaffold(run_dir, master_json):
    """Minimal consolidated run dir; master.json is caller-supplied."""
    run_dir = Path(run_dir)
    (run_dir / "client.json").write_text(
        json.dumps({"server_latency": {}}), encoding="utf-8"
    )
    (run_dir / "mock.json").write_text(
        json.dumps(
            {
                "final_snapshot": {"engines": []},
                "stats": [
                    {
                        "ts_epoch_ms": 1000,
                        "avg_batch_size": 10.0,
                        "avg_batch_ms": 300.0,
                        "prefill_waiting": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_meta.json").write_text(
        json.dumps({"params": {"n_prefill": 1, "n_decode": 1}}),
        encoding="utf-8",
    )
    (run_dir / "master.json").write_text(json.dumps(master_json), encoding="utf-8")
    (run_dir / "master.log").write_text(MASTER_LOG, encoding="utf-8")
    client_rows = [_client_row(i, T0 + i * 1000) for i in range(3)]
    ev_rows = []
    for i in range(3):
        send = T0 + i * 1000
        ev_rows.append(_ev_prefill(i, send))
        ev_rows.append(_ev_decode(i, send))
    (run_dir / "client_events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in client_rows) + "\n", encoding="utf-8"
    )
    (run_dir / "engine_events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in ev_rows) + "\n", encoding="utf-8"
    )


def _prom_ts(master_json):
    master_json["prometheus_timeseries"] = [
        {"ts": ts, "metrics": samples} for ts, samples in zip(PROM_TS, PROM_TIMELINE)
    ]
    return master_json


class PromSourceArrivalsTest(unittest.TestCase):
    """master_arrivals_ts / inflight_ts from the G3 timeline."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls._tmp.name) / "run"
        cls.run_dir.mkdir()
        _write_scaffold(cls.run_dir, _prom_ts({}))
        cls.agg = _run_aggregate(cls.run_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_label_variants_summed_differenced_reset_dropped(self):
        rows = self.agg["master_arrivals_ts"]
        # 4 sample gaps, one negative (counter reset 40 -> 25) dropped; each
        # rate row is anchored at the interval's right-end sample ts.
        self.assertEqual(
            [
                {"t": 0.0, "arrivals": 15.0, "completions": 8.0, "cum_arrivals": 15},
                {"t": 1.0, "arrivals": 25.0, "completions": 28.0, "cum_arrivals": 40},
                {"t": 3.0, "arrivals": 25.0, "completions": 24.0, "cum_arrivals": 50},
            ],
            rows,
        )

    def test_inflight_gauges_summed_per_sample(self):
        rows = self.agg["inflight_ts"]
        # 2s gauge cadence rides along with the samples' own ts.
        self.assertEqual(
            [
                {
                    "t": 0.0,
                    "scheduler": 2,
                    "prefill_batches": 7,
                    "prefill_requests": 70,
                    "decode_reserved": 11,
                    "decode_confirmed_running": 9,
                },
                {
                    "t": 2.0,
                    "scheduler": 4,
                    "prefill_batches": 4,
                    "prefill_requests": 30,
                    "decode_reserved": 2,
                    "decode_confirmed_running": 1,
                },
            ],
            rows,
        )


class LegacyKeysIgnoredTest(unittest.TestCase):
    """Legacy counters_timeseries / inflight_timeseries keys are ignored."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls._tmp.name) / "run"
        cls.run_dir.mkdir()
        master_json = _prom_ts(
            {
                "counters_timeseries": [
                    {"ts_epoch_ms": T0, "arrival_count": 100, "completion_count": 90},
                    {
                        "ts_epoch_ms": T0 + 1000,
                        "arrival_count": 250,
                        "completion_count": 200,
                    },
                ],
                "inflight_timeseries": [
                    {
                        "ts_epoch_ms": T0,
                        "inflight": {
                            "scheduler_inflight": 9,
                            "prefill_endpoints": [
                                {"inflight_batches": 2, "inflight_requests": 20}
                            ],
                            "decode_endpoints": [
                                {"reserved_total": 3, "confirmed_running": 2}
                            ],
                        },
                    }
                ],
            }
        )
        _write_scaffold(cls.run_dir, master_json)
        cls.agg = _run_aggregate(cls.run_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_arrivals_rebuilt_from_prometheus_not_legacy_counters(self):
        # The legacy rows (100 -> 250) must NOT win: the G3 prometheus
        # timeline (0 -> 15 -> 40 ...) is the sole source, so the output
        # matches the G3-derived expectations exactly.
        rows = self.agg["master_arrivals_ts"]
        self.assertEqual(
            [
                {"t": 0.0, "arrivals": 15.0, "completions": 8.0, "cum_arrivals": 15},
                {"t": 1.0, "arrivals": 25.0, "completions": 28.0, "cum_arrivals": 40},
                {"t": 3.0, "arrivals": 25.0, "completions": 24.0, "cum_arrivals": 50},
            ],
            rows,
        )

    def test_inflight_rebuilt_from_gauges_not_legacy_snapshots(self):
        # The legacy snapshot (scheduler=9, prefill_batches=2, ...) must
        # NOT win: the G3 gauge sums (identical to the sole-source test
        # above) are the only series produced.
        rows = self.agg["inflight_ts"]
        self.assertEqual(
            [
                {
                    "t": 0.0,
                    "scheduler": 2,
                    "prefill_batches": 7,
                    "prefill_requests": 70,
                    "decode_reserved": 11,
                    "decode_confirmed_running": 9,
                },
                {
                    "t": 2.0,
                    "scheduler": 4,
                    "prefill_batches": 4,
                    "prefill_requests": 30,
                    "decode_reserved": 2,
                    "decode_confirmed_running": 1,
                },
            ],
            rows,
        )


if __name__ == "__main__":
    unittest.main()
