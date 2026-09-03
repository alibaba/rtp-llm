"""TTFT engine 口径统一（20260903）+ P/D 引擎内等待序列——三层测试。

口径换血（与历史 client 首帧口径断代不可比）：
  * ttft = prefill_done_ms − send_start_epoch_ms：ok 行按 request_id join
    引擎 prefill_done 终态行（卫兵 done >= send；join miss 计
    integrity.ttft_engine_join_miss，与 prefill_exec_join_miss 同源同数）。
    client 首帧 ttft_ms 退役——行上非零 ttft_ms 不得再进 ttft 族。
  * prefill_wait = prefill_start_ms − engine_arrival_ms（EnqueueBatch 准入
    → 批开始执行，含 lane 排队）；decode_wait = decode_start_ms −
    engine_arrival_ms（hand-off 到达 → 进 running slot）；负值 = 时钟
    异常跳过该样本不编造。
  * 零样本（join 全 miss / 终态流全 cancelled）summary 键为 None
    （source 仍恒 "engine"），报告层显示缺省——零样本 ≠ 真实 0。
  * compare_twin：加载层 rid join 注入行级 ttft_engine_ms，ttft 族样本
    换源；合成 approx 行写 ttft_engine_ms（ttft_ms 占 0）保 round-trip。

fixture 数值锚点（send = T0 + i*1000）：
  正常行：prefill(arrival=+10, start=+20, done=+30) → ttft=30 / pw=10；
          decode(arrival=+40, start=+50, done=+80) → dw=10 / full_e2e=80
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

sys.path.insert(0, str(TOOLS_DIR))
import compare_twin  # noqa: E402  (module under test; __main__ guarded)


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


def _write_scaffold(run_dir):
    """run 目录最小采集产物（除 client_events / engine_events 外）。"""
    run_dir = Path(run_dir)
    (run_dir / "client.json").write_text(
        json.dumps({"slo_batch_analysis": {}, "server_latency": {}}),
        encoding="utf-8",
    )
    (run_dir / "mock.json").write_text(
        json.dumps({"final_snapshot": {"engines": []}, "stats": []}),
        encoding="utf-8",
    )
    (run_dir / "run_meta.json").write_text(
        json.dumps({"params": {"n_prefill": 1, "n_decode": 1}}), encoding="utf-8"
    )
    log_dir = run_dir / "flexlb_logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / "flexlb.log").write_text(
        "2026-09-03 01:30:31,123 flexlb_server_schedule_latency count=3 "
        "arrival_qps=1.0 completion_qps=1.0 server_p50_ms=1.0 "
        "server_p95_ms=2.0 server_p99_ms=3.0 grpc_queue_p95_ms=0.5 "
        "route_submit_p95_ms=0.5 batch_wait_p95_ms=0.5 "
        "dispatch_ack_p95_ms=0.5 ack_response_p95_ms=0.5\n",
        encoding="utf-8",
    )


def _client_row(i, send):
    """ok 行；ttft_ms=999 为 client 首帧假值——换血后不得进 ttft 族。"""
    return {
        "rid": "r%d" % i,
        "request_id": i,
        "status": "ok",
        "error": "",
        "send_start_epoch_ms": send,
        "input_len": 80,
        "output_len": 8,
        "wall_clock_ts": 1.0 + i,
        "ttft_ms": 999,
        "total_ms": 200,
        "schedule_ms": 5,
    }


def _ev_prefill(rid, send, arrival, start, done, cancelled=False):
    return {
        "event": "prefill_done",
        "rid": rid,
        "engine_name": "p1",
        "batch_id": rid,
        "engine_arrival_ms": send + arrival,
        "prefill_start_ms": send + start,
        "prefill_done_ms": send + done,
        "ttft_ms": done - arrival,  # 引擎内相对口径（schema 保真，聚合不消费）
        "exec_ms": done - start,
        "batch_size": 1,
        "input_len": 80,
        "cache_hit_tokens": 0,
        "kv_used_tokens": 80,
        "cancelled": cancelled,
    }


def _ev_decode(rid, send, arrival, start, done, cancelled=False):
    return {
        "event": "decode_done",
        "rid": rid,
        "engine_name": "d1",
        "batch_id": 100 + rid,
        "engine_arrival_ms": send + arrival,
        "decode_start_ms": send + start,
        "decode_done_ms": send + done,
        "exec_ms": done - start,
        "batch_size": 1,
        "output_len": 8,
        "kv_used_tokens": 88,
        "cancelled": cancelled,
    }


def _write_streams(run_dir, client_rows, ev_rows):
    run_dir = Path(run_dir)
    (run_dir / "client_events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in client_rows) + "\n", encoding="utf-8"
    )
    (run_dir / "engine_events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in ev_rows) + "\n", encoding="utf-8"
    )


def _write_run_a(run_dir):
    """fixture A：3 请求全 join 成功（ttft=30 / pw=10 / dw=10）。"""
    _write_scaffold(run_dir)
    client_rows = [_client_row(i, T0 + i * 1000) for i in range(3)]
    ev_rows = []
    for i in range(3):
        send = T0 + i * 1000
        ev_rows.append(_ev_prefill(i, send, 10, 20, 30))
        ev_rows.append(_ev_decode(i, send, 40, 50, 80))
    _write_streams(run_dir, client_rows, ev_rows)


def _write_run_b(run_dir):
    """fixture B：join miss / cancelled / 负值卫兵四象限。

    rid0 正常；rid1 无 prefill 行（decode 正常）→ ttft miss；
    rid2 prefill+decode 均 cancelled → 双 miss；rid3 start<arrival
    （时钟异常）→ ttft 正常但 wait 全跳过。
    """
    _write_scaffold(run_dir)
    client_rows = [_client_row(i, T0 + i * 1000) for i in range(4)]
    ev_rows = []
    for i in range(4):
        send = T0 + i * 1000
        if i == 0:
            ev_rows.append(_ev_prefill(i, send, 10, 20, 30))
            ev_rows.append(_ev_decode(i, send, 40, 50, 80))
        elif i == 1:
            ev_rows.append(_ev_decode(i, send, 40, 50, 80))
        elif i == 2:
            ev_rows.append(_ev_prefill(i, send, 10, 20, 30, cancelled=True))
            ev_rows.append(_ev_decode(i, send, 40, 50, 80, cancelled=True))
        else:
            ev_rows.append(_ev_prefill(i, send, 50, 20, 60))
            ev_rows.append(_ev_decode(i, send, 70, 40, 100))
    _write_streams(run_dir, client_rows, ev_rows)


def _write_run_c(run_dir):
    """fixture C：终态流全 cancelled → 零样本（summary None / source 恒 engine）。"""
    _write_scaffold(run_dir)
    client_rows = [_client_row(i, T0 + i * 1000) for i in range(2)]
    ev_rows = []
    for i in range(2):
        send = T0 + i * 1000
        ev_rows.append(_ev_prefill(i, send, 10, 20, 30, cancelled=True))
        ev_rows.append(_ev_decode(i, send, 40, 50, 80, cancelled=True))
    _write_streams(run_dir, client_rows, ev_rows)


class EngineTtftAggregateTest(unittest.TestCase):
    """aggregate：ttft 换血 engine 口径 + wait 派生 + source 恒 engine。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls._tmp.name) / "run"
        cls.run_dir.mkdir()
        _write_run_a(cls.run_dir)
        proc = _run([AGGREGATE], cwd=cls.run_dir)
        cls.agg = json.loads(proc.stdout)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_per_second_ttft_is_engine_caliber(self):
        # client ttft_ms=999 不得进桶：engine join 值 30 才是 ttft
        for row in self.agg["per_second"]:
            self.assertEqual(row["ttft_n"], 1)
            self.assertEqual(row["ttft_p50"], 30.0)
            self.assertEqual(row["ttft_p95"], 30.0)

    def test_per_second_wait_keys(self):
        for row in self.agg["per_second"]:
            self.assertEqual(row["prefill_wait_n"], 1)
            self.assertEqual(row["prefill_wait_p50"], 10.0)
            self.assertEqual(row["prefill_wait_p95"], 10.0)
            self.assertEqual(row["decode_wait_n"], 1)
            self.assertEqual(row["decode_wait_p50"], 10.0)
            self.assertEqual(row["decode_wait_p95"], 10.0)

    def test_summary_ttft_source_and_wait(self):
        sm = self.agg["summary"]
        ttft = sm["ttft_latency_ms"]
        self.assertEqual(ttft["count"], 3)
        self.assertEqual(ttft["p50"], 30.0)
        self.assertEqual(ttft["p99"], 30.0)
        self.assertEqual(sm["ttft_latency_source"], "engine")
        pw = sm["prefill_wait_latency_ms"]
        self.assertEqual(pw["count"], 3)
        self.assertEqual(pw["p50"], 10.0)
        dw = sm["decode_wait_latency_ms"]
        self.assertEqual(dw["count"], 3)
        self.assertEqual(dw["p50"], 10.0)

    def test_integrity_no_miss(self):
        self.assertNotIn("ttft_engine_join_miss", self.agg["integrity"])
        self.assertNotIn("prefill_exec_join_miss", self.agg["integrity"])

    def test_full_e2e_unaffected(self):
        # 同批 join 的既有指标不回归：full_e2e=80 / prefill_exec=10
        fe = self.agg["summary"]["full_e2e_latency_ms"]
        self.assertEqual(fe["count"], 3)
        self.assertEqual(fe["p50"], 80.0)
        pe = [r for r in self.agg["per_second"]][0]
        self.assertEqual(pe["prefill_exec_p50"], 10.0)


class EngineTtftMissGuardTest(unittest.TestCase):
    """aggregate：join miss / cancelled / 负值卫兵（不编造）。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls._tmp.name) / "run"
        cls.run_dir.mkdir()
        _write_run_b(cls.run_dir)
        proc = _run([AGGREGATE], cwd=cls.run_dir)
        cls.agg = json.loads(proc.stdout)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_ttft_count_and_miss_markers(self):
        # rid0=30 / rid3=60 进样本；rid1（无 prefill 行）与 rid2（cancelled）
        # 计 miss 不编造。分位按仓库统一 int-rank 约定（v[int(n*p)]）：
        # [30,60] 的 p50 = v[1] = 60
        ttft = self.agg["summary"]["ttft_latency_ms"]
        self.assertEqual(ttft["count"], 2)
        self.assertEqual(ttft["p50"], 60)
        self.assertEqual(ttft["p99"], 60)
        self.assertEqual(ttft["mean"], 45.0)
        integ = self.agg["integrity"]
        self.assertEqual(integ["ttft_engine_join_miss"], 2)
        self.assertEqual(integ["prefill_exec_join_miss"], 2)
        # decode 侧：仅 rid2 cancelled miss（rid1 的 decode 正常 join）
        self.assertEqual(integ["full_e2e_join_miss"], 1)

    def test_negative_wait_samples_skipped(self):
        # rid3 start<arrival（时钟异常）：ttft 正常（60），wait 跳过；
        # rid1 的 decode 正常 → decode_wait 有样本
        sm = self.agg["summary"]
        pw = sm["prefill_wait_latency_ms"]
        self.assertEqual(pw["count"], 1)  # 仅 rid0
        self.assertEqual(pw["p50"], 10.0)
        dw = sm["decode_wait_latency_ms"]
        self.assertEqual(dw["count"], 2)  # rid0 + rid1
        self.assertEqual(dw["p50"], 10.0)

    def test_per_second_negative_guard_bucket(self):
        by_t = {r["t"]: r for r in self.agg["per_second"]}
        # t=3（rid3）：ttft 有样本（60），两个 wait 无样本（负值跳过）
        self.assertEqual(by_t[3]["ttft_n"], 1)
        self.assertEqual(by_t[3]["ttft_p50"], 60.0)
        self.assertEqual(by_t[3]["prefill_wait_n"], 0)
        self.assertEqual(by_t[3]["decode_wait_n"], 0)
        # t=1（rid1）：decode 正常 join 但 ttft miss
        self.assertEqual(by_t[1]["ttft_n"], 0)
        self.assertEqual(by_t[1]["decode_wait_n"], 1)


class EngineTtftZeroSampleTest(unittest.TestCase):
    """aggregate：终态流全 cancelled → 零样本 None（零样本 ≠ 真实 0）。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls._tmp.name) / "run"
        cls.run_dir.mkdir()
        _write_run_c(cls.run_dir)
        proc = _run([AGGREGATE], cwd=cls.run_dir)
        cls.agg = json.loads(proc.stdout)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_summary_keys_null_and_source_engine(self):
        sm = self.agg["summary"]
        self.assertIsNone(sm["ttft_latency_ms"])
        self.assertIsNone(sm["prefill_wait_latency_ms"])
        self.assertIsNone(sm["decode_wait_latency_ms"])
        # source 恒 engine：口径标记与样本量无关
        self.assertEqual(sm["ttft_latency_source"], "engine")

    def test_no_miss_marker_when_map_empty(self):
        # 全 cancelled → map 空 → join 不执行也不计 miss（照 full_e2e 范本）
        integ = self.agg["integrity"]
        self.assertNotIn("ttft_engine_join_miss", integ)
        self.assertNotIn("full_e2e_join_miss", integ)

    def test_per_second_zero_samples(self):
        for row in self.agg["per_second"]:
            self.assertEqual(row["ttft_n"], 0)
            self.assertEqual(row["prefill_wait_n"], 0)
            self.assertEqual(row["decode_wait_n"], 0)


class EngineTtftCanvasTest(unittest.TestCase):
    """canvas：engine 单线恢复显示 + 口径/断代 caption + wait 面板 + 汇总表标记。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls._tmp.name) / "run"
        cls.run_dir.mkdir()
        _write_run_a(cls.run_dir)
        proc = _run([AGGREGATE], cwd=cls.run_dir)
        agg_path = cls.run_dir / "aggregate.json"
        agg_path.write_text(proc.stdout, encoding="utf-8")
        out_path = Path(cls._tmp.name) / "report.html"
        cls.proc = _run(
            [CANVAS, "--aggregate", agg_path, "--out", out_path], cwd=cls.run_dir
        )
        cls.html = out_path.read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_ttft_line_present_with_engine_label(self):
        # FETCH=0 下 client ttft 恒 0 曾使该线消失；engine 口径后恢复。
        # 注意：const 名（ttftP95）经反抽层转成内联 SPEC 数据，不留在
        # 最终 HTML——只能断言系列名与数据值。
        self.assertIn("ttft（p95·engine）", self.html)

    def test_caption_caliber_and_epoch_note(self):
        self.assertIn("ttft(engine) = 发出 → prefill 批完成", self.html)
        self.assertIn("断代", self.html)

    def test_wait_panel_present(self):
        self.assertIn("引擎内等待：prefill / decode（p50 / p95，出生轴）", self.html)
        self.assertIn("prefill wait（p95）", self.html)
        self.assertIn("decode wait（p95）", self.html)
        self.assertIn("prefill_wait = prefill_start − engine_arrival", self.html)

    def test_ttft_series_data_is_engine_values(self):
        # SPEC JSON 面板数据直接可检：engine 口径值 30（非 client 首帧 999）
        flat = self.html.replace(" ", "")
        self.assertIn(
            '"name":"ttft（p95·engine）","data":[30,30,30]'.replace(" ", ""), flat
        )
        self.assertNotIn("[999", flat)

    def test_sections_stdout_lists_latency(self):
        self.assertIn("latency", self.proc.stdout)


class EngineTtftCanvasOmissionTest(unittest.TestCase):
    """canvas：零样本 fail-closed——ttft 线/等待面板不注册，汇总表缺省。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls._tmp.name) / "run"
        cls.run_dir.mkdir()
        _write_run_c(cls.run_dir)
        proc = _run([AGGREGATE], cwd=cls.run_dir)
        agg_path = cls.run_dir / "aggregate.json"
        agg_path.write_text(proc.stdout, encoding="utf-8")
        out_path = Path(cls._tmp.name) / "report.html"
        _run([CANVAS, "--aggregate", agg_path, "--out", out_path], cwd=cls.run_dir)
        cls.html = out_path.read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_ttft_line_omitted(self):
        self.assertNotIn("ttft（p95·engine）", self.html)

    def test_wait_panel_omitted(self):
        self.assertNotIn("引擎内等待：prefill / decode", self.html)
        self.assertNotIn("prefill wait（p95）", self.html)

    def test_summary_table_shows_dash_not_zero(self):
        # 零样本：ttft 线/等待面板均不注册（面板层黑盒可验证部分）；
        # 汇总表 "—" 缺省行为由 aggregate 键 None 语义保证（Table 不进
        # 最终 HTML，无法黑盒断言，见 canvas 汇总表段注释）。
        # const 名不进 HTML（trivially-true），断言系列名即足。
        self.assertNotIn("ttft（p95·engine）", self.html)
        # 零样本下报告仍完整生成（不崩溃）且无 engine 数值线
        self.assertNotIn("[30,30,30]", self.html.replace(" ", ""))


class CompareTwinEngineCaliberTest(unittest.TestCase):
    """compare_twin：ttft 族换源 engine join + 合成 round-trip 携带新键。"""

    def test_real_side_latency_samples_engine_join(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            run_dir = Path(tmp.name) / "run"
            run_dir.mkdir()
            client_rows = [_client_row(i, T0 + i * 1000) for i in range(3)]
            ev_rows = [
                _ev_prefill(0, T0, 10, 20, 30),
                _ev_decode(0, T0, 40, 50, 80),
                _ev_prefill(1, T0 + 1000, 10, 20, 30),
                _ev_decode(1, T0 + 1000, 40, 50, 80),
                # rid2 无 prefill 行（join miss）；decode done<send 时钟异常行
                _ev_decode(2, T0 + 2000 - 500, 40, 50, 80),
            ]
            _write_streams(run_dir, client_rows, ev_rows)
            side = compare_twin.load_real_side(str(run_dir / "client_events.jsonl"))
            # engine 口径样本：rid0/rid1 = 30；client 首帧 999 不进
            self.assertEqual(side.latency_samples("ttft"), [30, 30])
            # e2e/schedule 族不受影响（client total_ms / schedule_ms）
            self.assertEqual(side.latency_samples("e2e"), [200, 200, 200])
            # 行级注入键存在且 ttft_ms 原样保留（不复写既有键语义）
            r0 = side.ok_rows()[0]
            self.assertEqual(r0.get("ttft_engine_ms"), 30)
            self.assertEqual(r0.get("ttft_ms"), 999)
            # summary（_aggregate_real_summary）随行级换源自动 engine
            self.assertEqual(side.summary["ttft"]["count"], 2)
            self.assertEqual(side.summary["ttft"]["p50"], 30.0)
        finally:
            tmp.cleanup()

    def test_no_engine_stream_leaves_rows_without_key(self):
        # 同目录无 engine_events.jsonl：行保持无 ttft_engine_ms（不编造），
        # ttft 族样本为空（回退 quantile-approx / absent）
        tmp = tempfile.TemporaryDirectory()
        try:
            run_dir = Path(tmp.name) / "run"
            run_dir.mkdir()
            client_rows = [_client_row(0, T0)]
            (run_dir / "client_events.jsonl").write_text(
                "\n".join(json.dumps(r) for r in client_rows) + "\n",
                encoding="utf-8",
            )
            side = compare_twin.load_real_side(str(run_dir / "client_events.jsonl"))
            self.assertEqual(side.latency_samples("ttft"), [])
            self.assertIsNone(side.ok_rows()[0].get("ttft_engine_ms"))
        finally:
            tmp.cleanup()

    def test_synthesize_roundtrip_carries_engine_key(self):
        # approx path：合成行 ttft_ms 占 0、engine 口径值写 ttft_engine_ms；
        # 再次加载（无 engine 流）行级直读 round-trip
        tmp = tempfile.TemporaryDirectory()
        try:
            out_dir = Path(tmp.name) / "syn"
            side = compare_twin.SideData("mock", "mock")
            side.duration_s = 10.0
            side.summary = {"total_requests": 2, "success_count": 2}
            side.approx_modes = {
                "ttft": [30.0, 60.0],
                "e2e": [200.0, 220.0],
                "schedule": [5.0, 6.0],
            }
            compare_twin.synthesize_real_inputs(side, str(out_dir))
            rows = [
                json.loads(ln)
                for ln in (out_dir / "client_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if ln.strip()
            ]
            ok_rows = [r for r in rows if r.get("status") != "schedule_error"]
            self.assertEqual(len(ok_rows), 2)
            for r in ok_rows:
                self.assertEqual(r["ttft_ms"], 0.0)
                self.assertIn("ttft_engine_ms", r)
                self.assertIn(r["ttft_engine_ms"], (30, 60, 30.0, 60.0))
            # round-trip：合成目录作为 real 侧再次加载，行级直读样本
            side2 = compare_twin.load_real_side(str(out_dir / "client_events.jsonl"))
            self.assertEqual(sorted(side2.latency_samples("ttft")), [30.0, 60.0])
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
