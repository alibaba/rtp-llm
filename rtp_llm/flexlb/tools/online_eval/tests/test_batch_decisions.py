"""batch_decisions 段（原 analyze_slo_batch.py 并入 aggregate，20260903）。

迁移语义锚点（与历史 test_analyze_slo_batch.py 数值对齐）：
  * dispatch/complete 结构化行解析自 master.log（时间戳前缀 + 合并视图
    形态），不进 master_events.jsonl（window/generation 事件流语义不同）；
  * Prometheus counter（master.json prometheus_after）作 reason 总量权威
    计数（decisions.source=prometheus_counter），结构化日志行只承担分布
    样本，log_coverage_ratio = log_count / count；
  * 不变量三规则：cap 多成员须低于阈值（batch_size>1 且 predicted>=
    threshold）、fixed_window_timeout 须等满 fixed_wait-2ms、batch_full
    须达到 batch_size_max；
  * config 双源：run_meta.params.flexlb_config 优先，其次
    process_config_json 的 zone_process_setting→envs→FLEXLB_CONFIG；
  * 分位口径 ceil-rank（math.ceil(nq)−1）——与 aggregate 主链
    nearest-rank 刻意不同，保持与历史 slo_batch_analysis.json 可比；
  * batch.mock_last 改为 aggregate 从 mock stats 末帧直出（数值归一），
    报告层 mock_last 行数据源切换但面板不变。

fixture 数值锚点：
  dispatch1(cap, size=8, wait=40, pred=510) + dispatch2(batch_full,
  size=31, wait=20, pred=480)；threshold=500 / fixed_wait=160 / max=32。
  cap 组 predicted>=threshold 且 size>1 → 违规 1；batch_full 组
  31<32 → 违规 1（共 2）。complete(batch_id=1, gap=10) → matched=1。
  prom counter：cap 70+30=100、fixed_window_timeout 25 → count=125，
  log_coverage_ratio = round(2/125, 6) = 0.016。
  batch_size sorted [8,31]：p50=ceil(1)-1=0→8 / p90..p99=1→31 / max=31。
  gap [10]：mean=10.0 / p50=10 / max=10。
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
AGGREGATE = TOOLS_DIR / "aggregate_canvas_run.py"
CANVAS = TOOLS_DIR / "canvas_report_gen.py"
T0 = 1_788_283_848_000  # epoch ms 锚点（与 client_events 首发送同拍）

DISPATCH_REASON_BASE = "flexlb_app_engine_balancing_master_dispatch_reason_total"

MASTER_LOG = (
    # window 行（10s 快照，live master 恒有；fail-closed 零行即硬错）
    "2026-09-03 01:30:31,123 flexlb_server_schedule_latency count=3 "
    "arrival_qps=1.0 completion_qps=1.0 server_p50_ms=1.0 "
    "server_p95_ms=2.0 server_p99_ms=3.0 grpc_queue_p95_ms=0.5 "
    "route_submit_p95_ms=0.5 batch_wait_p95_ms=0.5 "
    "dispatch_ack_p95_ms=0.5 ack_response_p95_ms=0.5\n"
    # dispatch 行 1：cap 多成员且 predicted>=threshold → 不变量违规
    "2026-09-03 01:30:32,456 flexlb_batch_dispatch batch_id=1 "
    "reason=predicted_execution_cap batch_size=8 wait_ms=40 "
    "predicted_ms=510 threshold_ms=500 fixed_wait_ms=160 "
    "batch_size_max=32 queue_after=2 worker=127.0.0.1:61000\n"
    # dispatch 行 2：batch_full 但 31 < 32 → 不变量违规
    "2026-09-03 01:30:33,789 flexlb_batch_dispatch batch_id=2 "
    "reason=batch_full batch_size=31 wait_ms=20 "
    "predicted_ms=480 threshold_ms=500 fixed_wait_ms=160 "
    "batch_size_max=32 queue_after=0 worker=127.0.0.1:61000\n"
    # complete 行：batch_id=1 matched
    "2026-09-03 01:30:34,001 flexlb_batch_complete batch_id=1 "
    "predicted_ms=510 actual_ms=520 gap_ms=10 batch_size=8 "
    "engine=127.0.0.1\n"
)

PROM_AFTER = {
    DISPATCH_REASON_BASE
    + '{engineIp="127.0.0.1",reason="predicted_execution_cap",role="PREFILL"}': 70.0,
    DISPATCH_REASON_BASE
    + '{engineIp="127.0.0.2",reason="predicted_execution_cap",role="PREFILL"}': 30.0,
    DISPATCH_REASON_BASE
    + '{engineIp="127.0.0.1",reason="fixed_window_timeout",role="PREFILL"}': 25.0,
}

FLEXLB_CONFIG_DOC = {
    "schemaVersion": 2,
    "scheduler": {"type": "QUEUE", "ordering": {"type": "PRIORITY"}},
    "dispatcher": {"type": "BATCH"},
}


def _run(cmd, cwd):
    proc = subprocess.run(
        [sys.executable] + [str(c) for c in cmd],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        str(cmd[-1])
        + " failed:\nstdout="
        + proc.stdout[-2000:]
        + "\nstderr="
        + proc.stderr[-2000:]
    )
    return proc


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


def _write_scaffold(
    run_dir,
    master_log=MASTER_LOG,
    prom_after=PROM_AFTER,
    flexlb_config=json.dumps(FLEXLB_CONFIG_DOC),
    process_config_json=None,
):
    """run 目录最小采集产物（consolidated 布局：master.log 优先链）。"""
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
                    },
                    {
                        "ts_epoch_ms": 2000,
                        "avg_batch_size": 19.5,
                        "avg_batch_ms": 500.0,
                        "prefill_waiting": 2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    params = {"n_prefill": 1, "n_decode": 1}
    if flexlb_config is not None:
        params["flexlb_config"] = flexlb_config
    run_meta = {"params": params}
    if process_config_json is not None:
        run_meta["process_config_json"] = process_config_json
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta), encoding="utf-8")
    master_json = {}
    if prom_after is not None:
        master_json["prometheus_after"] = prom_after
    (run_dir / "master.json").write_text(json.dumps(master_json), encoding="utf-8")
    if master_log is not None:
        (run_dir / "master.log").write_text(master_log, encoding="utf-8")
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


class BatchDecisionsAggregateTest(unittest.TestCase):
    """aggregate：batch_decisions 段全链出数（原 analyze() 语义迁移）。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls._tmp.name) / "run"
        cls.run_dir.mkdir()
        _write_scaffold(cls.run_dir)
        proc = _run([AGGREGATE], cwd=cls.run_dir)
        cls.agg = json.loads(proc.stdout)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_config_from_run_meta_flexlb_config(self):
        cfg = self.agg["batch_decisions"]["config"]
        self.assertEqual(500, cfg["predict_threshold_ms"])
        self.assertEqual(160, cfg["fixed_wait_ms"])
        self.assertEqual(32, cfg["batch_size_max"])
        self.assertEqual("QUEUE", cfg["scheduler_type"])
        self.assertEqual("PRIORITY", cfg["ordering_type"])
        self.assertEqual("BATCH", cfg["dispatcher_type"])

    def test_prometheus_counter_is_authoritative_count(self):
        dec = self.agg["batch_decisions"]["decisions"]
        self.assertEqual(125, dec["count"])
        self.assertEqual("prometheus_counter", dec["source"])
        self.assertEqual(
            {"fixed_window_timeout": 25, "predicted_execution_cap": 100},
            dec["reasons"],
        )
        self.assertEqual(2, dec["log_count"])
        self.assertEqual(0.016, dec["log_coverage_ratio"])
        self.assertEqual(
            {"batch_full": 1, "predicted_execution_cap": 1}, dec["log_reasons"]
        )

    def test_distributions_ceil_rank_caliber(self):
        dec = self.agg["batch_decisions"]["decisions"]
        # batch_size sorted [8,31]：ceil-rank 口径 p50=8 / p90..99=31
        bs = dec["batch_size"]
        self.assertEqual(2, bs["count"])
        self.assertEqual(19.5, bs["mean"])
        self.assertEqual(8, bs["p50"])
        self.assertEqual(31, bs["p90"])
        self.assertEqual(31, bs["p95"])
        self.assertEqual(31, bs["p99"])
        self.assertEqual(31, bs["max"])
        # wait_ms sorted [20,40]
        self.assertEqual(20, dec["wait_ms"]["p50"])
        self.assertEqual(40, dec["wait_ms"]["max"])
        # estimated = wait + predicted sorted [500,550]
        est = dec["estimated_wait_plus_prefill_ms"]
        self.assertEqual(525.0, est["mean"])
        self.assertEqual(500, est["p50"])
        self.assertEqual(550, est["max"])
        self.assertEqual("structured_log", dec["distribution_source"])

    def test_invariant_violations(self):
        dec = self.agg["batch_decisions"]["decisions"]
        self.assertEqual(2, dec["invariant_violation_count"])
        self.assertEqual(
            {1, 2}, {s["batch_id"] for s in dec["invariant_violation_samples"]}
        )

    def test_completions_match_and_gap(self):
        comp = self.agg["batch_decisions"]["completions"]
        self.assertEqual(1, comp["count"])
        self.assertEqual(1, comp["matched_decision_count"])
        gap = comp["prediction_gap_ms"]
        self.assertEqual(1, gap["count"])
        self.assertEqual(10.0, gap["mean"])
        self.assertEqual(10, gap["p50"])
        self.assertEqual(10, gap["max"])
        self.assertEqual(520, comp["actual_ms"]["max"])

    def test_batch_mock_last_direct_from_mock_stats(self):
        # C 块：mock_last 不再经 slo 转发——aggregate 直取 stats 末帧
        mock_last = self.agg["batch"]["mock_last"]
        self.assertEqual(19.5, mock_last["avg_batch_size"])
        self.assertEqual(500.0, mock_last["avg_batch_ms"])
        self.assertEqual(2, mock_last["prefill_waiting"])

    def test_dispatch_reason_ts_untouched(self):
        # dispatch_reason_ts（prometheus 时序视角）保留不动：本 fixture 无
        # prometheus_timeseries → 空 list，不因 batch_decisions 并入而报错
        self.assertEqual([], self.agg["dispatch_reason_ts"])


class BatchDecisionsLogOnlyTest(unittest.TestCase):
    """无 prometheus counter：log 计数即总量（source=structured_log）。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls._tmp.name) / "run"
        cls.run_dir.mkdir()
        _write_scaffold(cls.run_dir, prom_after={})
        proc = _run([AGGREGATE], cwd=cls.run_dir)
        cls.agg = json.loads(proc.stdout)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_log_reasons_are_the_count(self):
        dec = self.agg["batch_decisions"]["decisions"]
        self.assertEqual(2, dec["count"])
        self.assertEqual("structured_log", dec["source"])
        self.assertEqual(
            {"batch_full": 1, "predicted_execution_cap": 1}, dec["reasons"]
        )
        self.assertEqual(1.0, dec["log_coverage_ratio"])


class BatchDecisionsEmptyTest(unittest.TestCase):
    """负路径：master.log 无 dispatch/complete 行 → 空结构，不硬错。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls._tmp.name) / "run"
        cls.run_dir.mkdir()
        window_only = MASTER_LOG.splitlines()[0] + "\n"
        _write_scaffold(cls.run_dir, master_log=window_only, prom_after={})
        proc = _run([AGGREGATE], cwd=cls.run_dir)
        cls.agg = json.loads(proc.stdout)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_empty_structures_not_hard_error(self):
        # window 行存在（master log 链健康）但无批决策行：段为空结构，
        # 不抛错（与 window 行 fail-closed 语义互补）
        bd = self.agg["batch_decisions"]
        self.assertEqual(0, bd["decisions"]["count"])
        self.assertEqual({}, bd["decisions"]["reasons"])
        self.assertEqual(0, bd["decisions"]["log_count"])
        self.assertEqual(0.0, bd["decisions"]["log_coverage_ratio"])
        self.assertEqual(0, bd["decisions"]["batch_size"]["count"])
        self.assertEqual(0, bd["completions"]["count"])
        self.assertEqual(0, bd["completions"]["matched_decision_count"])
        self.assertEqual(0, bd["config"]["predict_threshold_ms"])

    def test_mock_last_still_present(self):
        # dispatch 行缺席不影响 mock_last（mock stats 独立数据源）
        self.assertEqual(19.5, self.agg["batch"]["mock_last"]["avg_batch_size"])


class BatchDecisionsProcessConfigSourceTest(unittest.TestCase):
    """config 回退源：run_meta 无 flexlb_config 时走 process_config_json。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls._tmp.name) / "run"
        cls.run_dir.mkdir()
        _write_scaffold(
            cls.run_dir,
            flexlb_config=None,
            process_config_json={
                "zone_name": "z",
                "zone_process_setting": {
                    "process_info": {
                        "envs": [["FLEXLB_CONFIG", json.dumps(FLEXLB_CONFIG_DOC)]]
                    }
                },
            },
        )
        proc = _run([AGGREGATE], cwd=cls.run_dir)
        cls.agg = json.loads(proc.stdout)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_config_falls_back_to_process_config_json(self):
        cfg = self.agg["batch_decisions"]["config"]
        self.assertEqual("QUEUE", cfg["scheduler_type"])
        self.assertEqual("PRIORITY", cfg["ordering_type"])
        self.assertEqual("BATCH", cfg["dispatcher_type"])


class BatchDecisionsCanvasTest(unittest.TestCase):
    """canvas：mock_last 数据源切到 aggregate batch 段后面板不变。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls._tmp.name) / "run"
        cls.run_dir.mkdir()
        _write_scaffold(cls.run_dir)
        proc = _run([AGGREGATE], cwd=cls.run_dir)
        agg_path = cls.run_dir / "aggregate.json"
        agg_path.write_text(proc.stdout, encoding="utf-8")
        out_path = Path(cls._tmp.name) / "report.html"
        _run([CANVAS, "--aggregate", agg_path, "--out", out_path], cwd=cls.run_dir)
        cls.html = out_path.read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_mock_last_row_present(self):
        # 汇总表 batch 行（avg size 19.5 / avg ms 500）是 JSX 中间产物，
        # Table 不进最终渲染 HTML（与 test_ttft_engine_caliber.py 的汇总
        # 表注释同一难处）：黑盒可验证部分是「报告正常生成 + mock stats
        # 数据链出数」——p10 平均 batch size 面板的「累计均值（参照）」
        # 线读 mock_stats 的 avg_batch_size 列（与 batch.mock_last 同源
        # 同链，fixture 末帧 19.5）。
        flat = self.html.replace(" ", "")
        self.assertIn("平均batchsize", flat)
        self.assertIn("累计均值（参照）", flat)
        self.assertIn("19.5", flat)

    def test_canvas_runs_without_slo_input(self):
        # --slo 已删：报告生成不再依赖任何 slo 文件（数据源全部内嵌
        # aggregate；fixture 目录本来就不存在 slo_batch_analysis.json，
        # 生成成功即证明无回归）
        self.assertFalse((self.run_dir / "slo_batch_analysis.json").exists())


if __name__ == "__main__":
    unittest.main()
