"""cache 命中率三口径端到端 fixture 测试（aggregate → canvas 全链）。

已知命中结构（block=16 tok/key 的语义背景；fixture 直接构造采集产物，
引擎行为由 flexlb-mock-engine 的 CacheKeyHitMetricsTest 覆盖）：
  * engine key counter 差分三窗（= 3 个 prefill 请求）：
      0/3（冷启动，keys=[k1,k2,k3] 无命中）→ 3/3（暖，全命中）→
      2/3（部分前缀，keys=[k1,k2,k90] 命中 k1,k2）
    → key 级 run = 5/9 ≈ 55.6%（对齐生产 recent_cache_key_hit）
  * token 时序窗口 (with_cache−context)/with_cache = 600/1000 = 60%；
    run 级 = Σhit_tokens_total(120) ÷ Σok il(3×80=240) = 50%
    （对齐生产 reuse/input）
  * master 路由 counter 对差分两窗 100/150 → 150/150；run = 250/300
    ≈ 83.3%（对齐生产 whale-lb app.cache routing_selected_match 对）
三口径互异（83.3 / 55.6 / 50.0），双图差值语义可独立断言。
旧 run（无新 counter / master 无 routing 系列）→ 口径独立缺省 +
面板/KPI 省略路径另测（test_old_run_*）。
"""

import gzip
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

_P1 = 'role="prefill",engine_ip="10.1.1.1",engine_name="p1"'
_ROUTE_HIT = 'flexlb_app_cache_routing_selected_match_hit_tokens_total{role="PREFILL"}'
_ROUTE_TOTAL = (
    'flexlb_app_cache_routing_selected_match_total_tokens_total{role="PREFILL"}'
)
_KEY_HITS = "mock_engine_cache_key_hits_total{" + _P1 + "}"
_KEY_REQ = "mock_engine_cache_keys_requested_total{" + _P1 + "}"
_CTX_TPS = "rtp_llm_context_tps{" + _P1 + "}"
_CTX_WC = "rtp_llm_context_tps_with_cache{" + _P1 + "}"


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


def _write_log(run_dir):
    """flexlb.log：server_schedule_latency 行（真实 run 标准形态；
    stage 面板与 schedule 样本量 caption 的数据源）。"""
    log_dir = Path(run_dir) / "flexlb_logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / "flexlb.log").write_text(
        "2026-09-02 01:30:31,123 flexlb_server_schedule_latency count=3 "
        "arrival_qps=1.0 completion_qps=1.0 server_p50_ms=1.0 "
        "server_p95_ms=2.0 server_p99_ms=3.0 grpc_queue_p95_ms=0.5 "
        "route_submit_p95_ms=0.5 batch_wait_p95_ms=0.5 "
        "dispatch_ack_p95_ms=0.5 ack_response_p95_ms=0.5\n",
        encoding="utf-8",
    )


def _write_engine_events(run_dir, n):
    """engine_events.jsonl：每请求一对 prefill_done/decode_done 终态行
    （多组件 jsonl 数据层的引擎侧流；aggregate 以 rid join 客户端行，
    done 时刻均在各自 send 之后以满足 join 时序校验）。"""
    rows = []
    for i in range(n):
        send = T0 + i * 1000
        rows.append(
            {
                "event": "prefill_done",
                "rid": i,
                "engine_name": "p1",
                "batch_id": i,
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
        )
        rows.append(
            {
                "event": "decode_done",
                "rid": i,
                "engine_name": "d1",
                "batch_id": 100 + i,
                "engine_arrival_ms": send + 40,
                "decode_start_ms": send + 50,
                "decode_done_ms": send + 80,
                "exec_ms": 30,
                "batch_size": 1,
                "output_len": 8,
                "kv_used_tokens": 88,
                "cancelled": False,
            }
        )
    (Path(run_dir) / "engine_events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def _write_full_run(run_dir):
    """构造三口径数据齐备的 run 目录（数值见模块 docstring）。"""
    run_dir = Path(run_dir)
    # client.json：空 slo / server_latency（validity 缺数据路径，不误报）
    (run_dir / "client.json").write_text(
        json.dumps({"slo_batch_analysis": {}, "server_latency": {}}), encoding="utf-8"
    )
    # client_events：3 行 ok，input_len 80 ×3 = 240（token 级 run 分母）
    rows = [
        {
            "rid": "r%d" % i,
            "request_id": i,
            "status": "ok",
            "error": "",
            "send_start_epoch_ms": T0 + i * 1000,
            "input_len": 80,
            "output_len": 8,
            "wall_clock_ts": 1.0 + i,
        }
        for i in range(3)
    ]
    (run_dir / "client_events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    # mock.json：final_snapshot 单 prefill 引擎，hit_tokens_total = 120
    # （token 级 run 分子；与 tps 时序窗口 60% 是两个独立已知量）
    (run_dir / "mock.json").write_text(
        json.dumps(
            {
                "final_snapshot": {
                    "engines": [
                        {"role": "prefill", "hit_tokens_total": 120, "running": 0}
                    ]
                },
                "stats": [],
            }
        ),
        encoding="utf-8",
    )
    # run_meta.json：引擎数回退链第 1 级（1P/1D，避免「引擎数未知」回退）
    (run_dir / "run_meta.json").write_text(
        json.dumps({"params": {"n_prefill": 1, "n_decode": 1}}), encoding="utf-8"
    )
    # master.json：G3 prometheus routing counter 对（master 侧零改动，
    # 系列原样来自 whale-lb；只有 role="PREFILL" 变体）
    (run_dir / "master.json").write_text(
        json.dumps(
            {
                "prometheus_timeseries": [
                    {"ts": T0, "metrics": {_ROUTE_HIT: 0, _ROUTE_TOTAL: 0}},
                    {"ts": T0 + 1000, "metrics": {_ROUTE_HIT: 100, _ROUTE_TOTAL: 150}},
                    {"ts": T0 + 2000, "metrics": {_ROUTE_HIT: 250, _ROUTE_TOTAL: 300}},
                ]
            }
        ),
        encoding="utf-8",
    )
    # mock_per_engine_timeseries.json.gz：G1 白名单系列（新 key counter 对
    # + 既有 rtp_llm_context_tps 对）。counter 累计 0/0 → 0/3 → 3/6 → 5/9；
    # tps 窗口 with_cache=1000 / context=400（token 级窗口 60%）。
    per_engine = []
    key_hits = [0, 0, 3, 5]
    key_req = [0, 3, 6, 9]
    for i in range(4):
        per_engine.append(
            {
                "ts": T0 + i * 1000,
                "metrics": {
                    _KEY_HITS: key_hits[i],
                    _KEY_REQ: key_req[i],
                    _CTX_TPS: 400 if i else 0,
                    _CTX_WC: 1000 if i else 0,
                },
            }
        )
    with gzip.open(run_dir / "mock_per_engine_timeseries.json.gz", "wt") as f:
        json.dump(per_engine, f)
    _write_engine_events(run_dir, 3)
    _write_log(run_dir)


def _write_old_run(run_dir):
    """旧 run（G1 白名单无新 counter、master 无 routing 系列）：只有
    final_snapshot hit_tokens_total + ok 行（token 级 run 级可算，时序与
    另两口径整体缺省）。"""
    run_dir = Path(run_dir)
    (run_dir / "client.json").write_text(
        json.dumps({"slo_batch_analysis": {}, "server_latency": {}}), encoding="utf-8"
    )
    # 2 行 ok（input_len 80 ×2 = 160）：token 级 run = 120/160 =
    # 75.0%；第 2 行同时保证 T_END > 0（canvas 时间轴断言 min < max）
    old_rows = [
        {
            "rid": "r%d" % i,
            "request_id": i,
            "status": "ok",
            "error": "",
            "send_start_epoch_ms": T0 + i * 1000,
            "input_len": 80,
            "output_len": 8,
            "wall_clock_ts": 1.0 + i,
        }
        for i in range(2)
    ]
    (run_dir / "client_events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in old_rows) + "\n",
        encoding="utf-8",
    )
    (run_dir / "mock.json").write_text(
        json.dumps(
            {
                "final_snapshot": {
                    "engines": [
                        {"role": "prefill", "hit_tokens_total": 120, "running": 0}
                    ]
                },
                "stats": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_meta.json").write_text(
        json.dumps({"params": {"n_prefill": 1, "n_decode": 1}}), encoding="utf-8"
    )
    _write_engine_events(run_dir, 2)
    _write_log(run_dir)


class CacheHitAggregateTest(unittest.TestCase):
    """aggregate 三口径解析：summary run 级值 + cache_hit_ts 窗口值。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls._tmp.name) / "run"
        cls.run_dir.mkdir()
        _write_full_run(cls.run_dir)
        proc = _run([AGGREGATE], cwd=cls.run_dir)
        cls.agg = json.loads(proc.stdout)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_summary_three_calibers_run_level(self):
        sm = self.agg["summary"]["cache_hit_summary"]
        # 口径 1：master 路由 counter 对末拍比 250/300
        self.assertEqual(sm["master_routing_hit_pct"], 83.3)
        self.assertEqual(sm["master_routing_hit_tokens"], 250)
        self.assertEqual(sm["master_routing_total_tokens"], 300)
        # 口径 2：engine key counter 对 5/9
        self.assertEqual(sm["engine_key_hit_pct"], 55.6)
        self.assertEqual(sm["engine_key_hits"], 5)
        self.assertEqual(sm["engine_keys_requested"], 9)
        # 口径 3：Σhit_tokens_total(120) ÷ Σok il(240)
        self.assertEqual(sm["engine_token_hit_pct"], 50.0)
        self.assertEqual(sm["engine_hit_tokens"], 120)
        self.assertEqual(sm["engine_input_tokens"], 240)

    def test_cache_hit_ts_window_values(self):
        rows = self.agg["cache_hit_ts"]
        by_t = {r["t"]: r for r in rows}
        # master 路由差分窗：100/150 → 150/150（T0 首拍无差分窗）
        self.assertAlmostEqual(by_t[1.0]["master_routing"], 0.6667, places=3)
        self.assertAlmostEqual(by_t[2.0]["master_routing"], 1.0, places=3)
        # engine key 差分窗：0/3 → 3/3 → 2/3（任务书示例：命中 2/3）
        self.assertAlmostEqual(by_t[1.0]["engine_key"], 0.0, places=3)
        self.assertAlmostEqual(by_t[2.0]["engine_key"], 1.0, places=3)
        self.assertAlmostEqual(by_t[3.0]["engine_key"], 0.6667, places=3)
        # engine token 窗口：(1000−400)/1000
        for t in (1.0, 2.0, 3.0):
            self.assertAlmostEqual(by_t[t]["engine_token"], 0.6, places=3)
        # 首拍（t=0）无任何口径窗口（差分需要前一拍）
        self.assertNotIn(0.0, by_t)


class CacheHitCanvasTest(unittest.TestCase):
    """canvas 5c 面板：双图 + 差值语义标注 + KPI 读数行 + fail-closed 断言。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls._tmp.name) / "run"
        cls.run_dir.mkdir()
        _write_full_run(cls.run_dir)
        agg_path = cls.run_dir / "aggregate.json"
        proc = _run([AGGREGATE], cwd=cls.run_dir)
        agg_path.write_text(proc.stdout, encoding="utf-8")
        cls.agg_stdout = proc.stdout
        out_path = Path(cls._tmp.name) / "report.html"
        cls.proc = _run(
            [CANVAS, "--aggregate", agg_path, "--out", out_path], cwd=cls.run_dir
        )
        cls.html = out_path.read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_both_panels_and_annotations_present(self):
        # 图 1：master 路由 vs engine 执行（差值 = 调度损耗）
        self.assertIn("cache 命中率：master 路由 vs engine 执行", self.html)
        self.assertIn("master 路由口径", self.html)
        self.assertIn("engine 执行口径", self.html)
        self.assertIn("调度损耗", self.html)
        # 图 2：key 级（理论）vs token 级（实际）（差值 = 命中深度覆盖）
        self.assertIn("engine 命中率：key 级（理论）vs token 级（实际）", self.html)
        self.assertIn("key 级理论口径", self.html)
        self.assertIn("token 级实际口径", self.html)
        self.assertIn("命中深度覆盖", self.html)

    def test_kpi_chart_run_level_values(self):
        # run 级汇总柱状图：三柱（口径名即 categories）+ 生产对齐 caption
        self.assertIn("cache 命中率三口径：run 级汇总", self.html)
        self.assertIn("对齐生产", self.html)
        self.assertIn("[83.3,55.6,50]", self.html.replace(" ", ""))
        self.assertIn(
            '["master路由口径","key级理论口径","token级实际口径"]',
            self.html.replace(" ", ""),
        )

    def test_window_series_values_rendered(self):
        # 前向填充后的窗口值：master_routing 66.7/100/100（末拍跨拍窗
        # 前向填充）、engine_key 0/100/66.7、engine_token 60×3（两图各
        # 一份，SPEC JSON 面板数据直接可检）
        flat = self.html.replace(" ", "")
        self.assertIn("[66.7,100,100]", flat)
        self.assertIn("[0,100,66.7]", flat)
        self.assertEqual(flat.count("[60,60,60]"), 2)

    def test_sections_stdout_lists_cache_hit(self):
        self.assertIn("cache-hit", self.proc.stdout)

    def test_line_charts_omitted_when_token_caliber_missing(self):
        """单口径孤悬省略路径：抽掉 engine_token 后双时序图均缺参照系
        （图 1 需 master_routing + engine_token、图 2 需 engine_key +
        engine_token）→ 双图整体省略（标题不出现）；run 级柱状图不受
        影响保留（其 caption 的柱差读法句仍含差值词，属合法呈现，故
        差值词不作 absent 断言，改断言图标题 absent）。fail-closed
        断言组的触发由「标注串在则断言在」的源码顺序保证。"""
        tmp = tempfile.TemporaryDirectory()
        try:
            run_dir = Path(tmp.name) / "run"
            run_dir.mkdir()
            _write_full_run(run_dir)
            agg = json.loads(self.agg_stdout)
            # 抽掉 engine_token 列：两图都缺参照系 → 双图均省略
            for r in agg["cache_hit_ts"]:
                r.pop("engine_token", None)
            agg["summary"]["cache_hit_summary"].pop("engine_token_hit_pct", None)
            agg_path = run_dir / "aggregate.json"
            agg_path.write_text(json.dumps(agg), encoding="utf-8")
            out_path = Path(tmp.name) / "report.html"
            proc = _run(
                [CANVAS, "--aggregate", agg_path, "--out", out_path], cwd=run_dir
            )
            html = out_path.read_text(encoding="utf-8")
            self.assertNotIn("cache 命中率：master 路由 vs engine 执行", html)
            self.assertNotIn("engine 命中率：key 级（理论）vs token 级（实际）", html)
            # run 级柱状图保留（两柱：master 路由 83.3 + key 级 55.6）
            self.assertIn("cache 命中率三口径：run 级汇总", html)
            self.assertIn("[83.3,55.6]", html.replace(" ", ""))
        finally:
            tmp.cleanup()


class CacheHitOldRunOmissionTest(unittest.TestCase):
    """旧 run（无新 counter/系列）：口径独立缺省 + 面板省略。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls._tmp.name) / "run"
        cls.run_dir.mkdir()
        _write_old_run(cls.run_dir)
        proc = _run([AGGREGATE], cwd=cls.run_dir)
        cls.agg = json.loads(proc.stdout)
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

    def test_summary_only_token_caliber(self):
        sm = self.agg["summary"]["cache_hit_summary"]
        self.assertNotIn("master_routing_hit_pct", sm)
        self.assertNotIn("engine_key_hit_pct", sm)
        # token 级 run 仍可算（120/160）
        self.assertEqual(sm["engine_token_hit_pct"], 75.0)
        self.assertEqual(sm["engine_hit_tokens"], 120)
        self.assertEqual(sm["engine_input_tokens"], 160)

    def test_cache_hit_ts_empty(self):
        self.assertEqual(self.agg["cache_hit_ts"], [])

    def test_canvas_omits_panels_keeps_kpi(self):
        # 双时序图省略（差值语义标注不出现），run 级柱状图仅 token 级
        # 一柱（75.0%）
        self.assertNotIn("cache 命中率：master 路由 vs engine 执行", self.html)
        self.assertNotIn("engine 命中率：key 级（理论）vs token 级（实际）", self.html)
        self.assertIn("cache 命中率三口径：run 级汇总", self.html)
        self.assertIn("对齐生产", self.html)
        self.assertIn("[75]", self.html.replace(" ", ""))
        self.assertIn("cache-hit", self.proc.stdout)


if __name__ == "__main__":
    unittest.main()
