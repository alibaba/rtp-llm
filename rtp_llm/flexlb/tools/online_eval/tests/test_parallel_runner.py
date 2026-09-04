"""parallel_runner unit tests (lane packing / case sharding / port matrix / aggregation).

The orchestrator is pure planning + subprocess glue; these tests pin the
contracts that keep parallel lanes from colliding and the P1 case-level
flattening honest:

  * plan_lanes — every category lands in exactly ONE lane (a dropped or
    duplicated category would silently change the case set), and the LPT
    greedy stays balanced;
  * plan_case_lanes — the P1 per-case LPT: every case lands exactly
    once, deterministic, lanes balanced, runner-registration order
    restored within a lane;
  * lane_env — the per-lane port partition is disjoint across lanes and
    matches the documented formulas (master group / mock base strides,
    including the compressed --mock-stride), and touches NOTHING beyond
    the six port keys (operator env like FLEXLB_FT_HA_DUAL_MASTER must
    pass through);
  * aggregate — per-lane runner JSONs merge into the single-runner schema
    (summary counts summed from the runner's own summary blocks, verdict
    recomputed from normal graded cases, per-case lane field added), and
    a non-zero lane rc forces exit_code 1 even with zero FAIL rows;
    shard=case runs record summary.shard + lanes[].case_names;
  * runner --cases — exact-name selection wins over --category/--filter,
    profile filtering still applies, unknown names exit 2.

Plus --dry-run CLI smokes (the only jar-free execution paths).
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import parallel_runner  # noqa: E402

PARALLEL_RUNNER = TOOLS_DIR / "parallel_runner.py"
RUNNER = TOOLS_DIR / "flexlb_functional_tests.py"

# Default-stride lane cap (mock band 55151 + 2000*(N-1) + 151 <= 65535).
DEFAULT_LANES = parallel_runner.max_lanes(
    parallel_runner.MOCK_PORT_STRIDE, parallel_runner.MOCK_BASE_GRPC_PORT
)

# Family weights = per-case seconds x live case count (batch-window
# profile, 2026-09 snapshot) — mirrors family_weights() without needing
# the runner --list call inside unit tests.
_FAMILY_WEIGHTS = {
    "master": 60 * 8,
    "engine_fault": 30 * 13,
    "status": 15 * 24,
    "admission": 30 * 11,
    "elastic": 40 * 8,
    "kv": 20 * 15,
    "cancel": 12 * 13,
    "balance": 15 * 6,
    "direct": 8 * 1,
}


class PlanLanesTest(unittest.TestCase):
    def test_every_category_lands_exactly_once(self):
        lanes = parallel_runner.plan_lanes(dict(_FAMILY_WEIGHTS), parallel=4)
        packed = [c for lane in lanes for c in lane]
        self.assertCountEqual(list(_FAMILY_WEIGHTS), packed)

    def test_lpt_keeps_lanes_balanced_and_nonempty(self):
        lanes = parallel_runner.plan_lanes(dict(_FAMILY_WEIGHTS), parallel=4)
        loads = [sum(_FAMILY_WEIGHTS[c] for c in lane) for lane in lanes]
        # 9 categories into 4 lanes: every lane gets work...
        self.assertTrue(all(lane for lane in lanes))
        # ...and the packing stays within ~1.5x of the lightest lane
        # (total ≈ 2434 units / 4 ≈ 609 ideal).
        self.assertLessEqual(max(loads) / min(loads), 1.5)

    def test_single_lane_plan_uses_legacy_all_path(self):
        args = argparse.Namespace(parallel=1, profile="batch-window", categories=None)
        lanes, _ = parallel_runner._plan(args)
        self.assertEqual([["all"]], lanes)

    def test_partial_subset_stays_per_category_at_parallel_one(self):
        # --categories direct at parallel=1 must NOT become `--category all`
        # (that would run the unrequested categories too).
        args = argparse.Namespace(
            parallel=1, profile="batch-window", categories="direct"
        )
        lanes, weights = parallel_runner._plan(args)
        self.assertEqual([["direct"]], lanes)
        self.assertEqual({"direct": 8}, weights)

    def test_kebab_case_category_normalizes(self):
        args = argparse.Namespace(
            parallel=2, profile="batch-window", categories="engine-fault,direct"
        )
        lanes, weights = parallel_runner._plan(args)
        packed = sorted(c for lane in lanes for c in lane)
        self.assertEqual(["direct", "engine_fault"], packed)
        # spawn argv uses the runner's kebab-case choices
        self.assertEqual(
            "engine-fault", parallel_runner._runner_cli_name("engine_fault")
        )
        self.assertEqual("kv", parallel_runner._runner_cli_name("kv"))

    def test_parallel_two_packs_heavy_families_apart(self):
        lanes = parallel_runner.plan_lanes(dict(_FAMILY_WEIGHTS), parallel=2)
        # The two heaviest families (master ≈ 480, engine_fault ≈ 390)
        # must not share a lane — that pairing would dominate the wall.
        for lane in lanes:
            self.assertFalse({"master", "engine_fault"} <= set(lane))


class LaneEnvTest(unittest.TestCase):
    def test_lane_port_footprints_are_disjoint(self):
        # A lane owns a master group [m..m+5] (single-master/Tier-1 A:
        # http/mgmt/grpc = +0/+1/+2; Tier-1 B: +3..+5) and a mock scan
        # window [base-1 .. base+151] (mock http, engines, victim zone).
        # Footprints across lanes must be disjoint sets.
        footprints = []
        for i in range(DEFAULT_LANES):
            m = parallel_runner.MASTER_HTTP_BASE + 10 * i
            base = parallel_runner.MOCK_BASE_GRPC_PORT + 2000 * i
            ports = set(range(m, m + 6)) | set(range(base - 1, base + 152))
            footprints.append(ports)
        for i in range(len(footprints)):
            for j in range(i + 1, len(footprints)):
                self.assertEqual(
                    set(),
                    footprints[i] & footprints[j],
                    f"lane {i} and lane {j} share ports",
                )

    def test_single_master_and_ha_tier_ports_share_the_group_head(self):
        # By design (harness port plan): the single-master path, HA Tier-1 A
        # and HA Tier-3 all bind the group head — they are three MUTUALLY
        # EXCLUSIVE env shapes inside one runner process, never concurrent.
        env = parallel_runner.lane_env(2)
        self.assertEqual(
            env["FLEXLB_FT_MASTER_HTTP_PORT"],
            env["FLEXLB_FT_HA_MASTER_A_HTTP_PORT"],
        )
        self.assertEqual(
            env["FLEXLB_FT_MASTER_HTTP_PORT"],
            env["FLEXLB_FT_HA_TIER3_MASTER_HTTP_PORT"],
        )
        self.assertEqual(
            int(env["FLEXLB_FT_HA_MASTER_B_HTTP_PORT"])
            - int(env["FLEXLB_FT_HA_MASTER_A_HTTP_PORT"]),
            3,
        )

    def test_formulas_match_documented_strides(self):
        for i in range(DEFAULT_LANES):
            env = parallel_runner.lane_env(i)
            m = parallel_runner.MASTER_HTTP_BASE + 10 * i
            self.assertEqual(str(m), env["FLEXLB_FT_MASTER_HTTP_PORT"])
            self.assertEqual(str(m + 1), env["FLEXLB_FT_MASTER_MANAGEMENT_PORT"])
            self.assertEqual(str(m), env["FLEXLB_FT_HA_MASTER_A_HTTP_PORT"])
            self.assertEqual(str(m + 3), env["FLEXLB_FT_HA_MASTER_B_HTTP_PORT"])
            self.assertEqual(str(m), env["FLEXLB_FT_HA_TIER3_MASTER_HTTP_PORT"])
            self.assertEqual(
                str(parallel_runner.MOCK_BASE_GRPC_PORT + 2000 * i),
                env["FLEXLB_FT_MOCK_BASE_GRPC_PORT"],
            )

    def test_overlay_touches_only_the_six_port_keys(self):
        # Operator env (e.g. FLEXLB_FT_HA_DUAL_MASTER=1) must pass through
        # untouched — the overlay is exactly the port partition.
        self.assertEqual(
            {
                "FLEXLB_FT_MASTER_HTTP_PORT",
                "FLEXLB_FT_MASTER_MANAGEMENT_PORT",
                "FLEXLB_FT_HA_MASTER_A_HTTP_PORT",
                "FLEXLB_FT_HA_MASTER_B_HTTP_PORT",
                "FLEXLB_FT_HA_TIER3_MASTER_HTTP_PORT",
                "FLEXLB_FT_MOCK_BASE_GRPC_PORT",
            },
            set(parallel_runner.lane_env(3)),
        )

    def test_base_offset_env_shifts_whole_matrix(self):
        # FLEXLB_FT_PARALLEL_MASTER_BASE / FLEXLB_FT_PARALLEL_MOCK_BASE
        # shift the whole partition (e.g. to dodge a concurrent default-
        # band user on a shared host) while keeping the lane strides.
        with mock.patch.dict(
            os.environ,
            {
                "FLEXLB_FT_PARALLEL_MASTER_BASE": "20080",
                "FLEXLB_FT_PARALLEL_MOCK_BASE": "50000",
            },
        ):
            e0 = parallel_runner.lane_env(0)
            e1 = parallel_runner.lane_env(1)
        self.assertEqual("20080", e0["FLEXLB_FT_MASTER_HTTP_PORT"])
        self.assertEqual("20081", e0["FLEXLB_FT_MASTER_MANAGEMENT_PORT"])
        self.assertEqual("20090", e1["FLEXLB_FT_MASTER_HTTP_PORT"])
        self.assertEqual("50000", e0["FLEXLB_FT_MOCK_BASE_GRPC_PORT"])
        self.assertEqual("52000", e1["FLEXLB_FT_MOCK_BASE_GRPC_PORT"])

    def test_base_offset_footprints_stay_disjoint(self):
        with mock.patch.dict(
            os.environ,
            {
                "FLEXLB_FT_PARALLEL_MASTER_BASE": "20080",
                "FLEXLB_FT_PARALLEL_MOCK_BASE": "50000",
            },
        ):
            footprints = []
            for i in range(parallel_runner.max_lanes(2000, 50000)):
                m = parallel_runner._master_base() + 10 * i
                base = parallel_runner._mock_base() + 2000 * i
                ports = set(range(m, m + 6)) | set(range(base - 1, base + 152))
                footprints.append(ports)
        for i in range(len(footprints)):
            for j in range(i + 1, len(footprints)):
                self.assertEqual(
                    set(),
                    footprints[i] & footprints[j],
                    f"lane {i} and lane {j} share ports",
                )

    def test_invalid_base_env_is_rejected(self):
        with mock.patch.dict(
            os.environ, {"FLEXLB_FT_PARALLEL_MASTER_BASE": "not-a-port"}
        ):
            with self.assertRaises(SystemExit):
                parallel_runner._master_base()
        with mock.patch.dict(os.environ, {"FLEXLB_FT_PARALLEL_MOCK_BASE": "0"}):
            with self.assertRaises(SystemExit):
                parallel_runner._mock_base()


def _write_runner_json(path: Path, cases: list[dict], summary: dict) -> None:
    path.write_text(json.dumps({"summary": summary, "cases": cases}), encoding="utf-8")


def _lane_result(lane_idx: int, runs, wall_s: float = 1.0):
    lr = parallel_runner.LaneResult(lane_idx, [cat for cat, _, _ in runs])
    lr.runs = runs
    lr.wall_s = wall_s
    return lr


class AggregateTest(unittest.TestCase):
    def test_merges_lanes_into_single_runner_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            j0, j1 = d / "a.json", d / "b.json"
            _write_runner_json(
                j0,
                [
                    {
                        "category": "master",
                        "name": "master_kill",
                        "status": "PASS",
                        "expected_fail": False,
                        "duration_ms": 1500,
                        "grade": {"achieved": "strict"},
                    },
                    {
                        "category": "master",
                        "name": "probe_x",
                        "status": "FINDING-CONFIRMED",
                        "expected_fail": True,
                        "duration_ms": 500,
                        "grade": {"achieved": "loose"},
                    },
                ],
                {
                    "total": 2,
                    "passed": 1,
                    "failed": 0,
                    "finding_confirmed": 1,
                    "finding_resolved": 0,
                },
            )
            _write_runner_json(
                j1,
                [
                    {
                        "category": "kv",
                        "name": "kv_match_mixed",
                        "status": "PASS",
                        "expected_fail": False,
                        "duration_ms": 2500,
                        "grade": {"achieved": "normal"},
                    },
                ],
                {
                    "total": 1,
                    "passed": 1,
                    "failed": 0,
                    "finding_confirmed": 0,
                    "finding_resolved": 0,
                },
            )
            args = argparse.Namespace(
                parallel=2, profile="batch-window", grade="normal"
            )
            payload = parallel_runner.aggregate(
                [
                    _lane_result(0, [("master", 0, j0)]),
                    _lane_result(1, [("kv", 0, j1)]),
                ],
                args,
                wall_s=12.0,
            )
        s = payload["summary"]
        self.assertEqual(3, s["total"])
        self.assertEqual(2, s["passed"])
        self.assertEqual(1, s["finding_confirmed"])
        # verdict recomputed from NORMAL graded cases only: the
        # expected_fail probe's "loose" must not drag it below good.
        self.assertEqual("good", s["verdict"])
        self.assertEqual(0, s["exit_code"])
        # serial case-time sum + per-row lane field
        self.assertAlmostEqual(4.5, s["serial_case_time_s"])
        self.assertEqual([0, 0, 1], [row["lane"] for row in payload["cases"]])
        self.assertEqual(2, len(payload["lanes"]))

    def test_lane_rc_failure_forces_exit_code_even_without_fail_rows(self):
        # A runner that crashed before writing JSON has no FAIL rows, but
        # its rc must still gate the aggregate exit code.
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            args = argparse.Namespace(
                parallel=1, profile="batch-window", grade="normal"
            )
            payload = parallel_runner.aggregate(
                [_lane_result(0, [("status", 2, missing)])], args, wall_s=1.0
            )
        self.assertEqual(0, payload["summary"]["total"])
        self.assertEqual(1, payload["summary"]["exit_code"])
        self.assertEqual({"status": 2}, payload["lanes"][0]["exit_codes"])


class DryRunCLITest(unittest.TestCase):
    def test_dry_run_prints_plan_and_exits_zero(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(PARALLEL_RUNNER),
                "--parallel",
                "4",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("lane plan", proc.stdout)
        self.assertIn("port partition", proc.stdout)
        self.assertIn("mock base", proc.stdout)

    def test_parallel_bounds_are_enforced(self):
        proc = subprocess.run(
            [sys.executable, str(PARALLEL_RUNNER), "--parallel", "9", "--dry-run"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(2, proc.returncode)
        self.assertIn("--parallel must be 1..6", proc.stderr)


class RunnerCasesFlagTest(unittest.TestCase):
    """--cases exact-name selection semantics (the runner side)."""

    def _run(self, *extra: str):
        return subprocess.run(
            [sys.executable, str(RUNNER)] + list(extra),
            capture_output=True,
            text=True,
        )

    def test_exact_name_list_runs_only_those_cases(self):
        proc = self._run("--list", "--cases", "cancel_basic,cancel_idempotent")
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("Total: 2 cases", proc.stdout)
        self.assertIn("cancel_basic", proc.stdout)
        self.assertNotIn("cancel_sibling_isolation", proc.stdout)

    def test_cases_wins_over_category_and_filter(self):
        # --cases takes priority: --category/--filter must not narrow it.
        proc = self._run(
            "--list",
            "--cases",
            "cancel_basic",
            "--category",
            "kv",
            "--filter",
            "zzz",
        )
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("Total: 1 cases", proc.stdout)

    def test_unknown_case_name_exits_two(self):
        proc = self._run("--list", "--cases", "no_such_case")
        self.assertEqual(2, proc.returncode)
        self.assertIn("unknown --cases entries", proc.stderr)

    def test_profile_filter_still_applies_to_cases(self):
        # prio_order_basic is single-nonbatch-only: the profile filter
        # must still drop it from an explicit --cases list.
        proc = self._run(
            "--list", "--cases", "prio_order_basic", "--profile", "batch-window"
        )
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("Total: 0 cases", proc.stdout)


class PlanCaseLanesTest(unittest.TestCase):
    def test_every_case_lands_exactly_once(self):
        costs = [(f"case_{i}", float(i * 10 % 37)) for i in range(20)]
        lanes = parallel_runner.plan_case_lanes(costs, parallel=4)
        packed = [n for lane in lanes for n in lane]
        self.assertCountEqual([n for n, _ in costs], packed)

    def test_deterministic_and_registration_order_within_lane(self):
        costs = [("b", 5.0), ("a", 5.0), ("c", 1.0), ("d", 9.0), ("e", 5.0)]
        lanes1 = parallel_runner.plan_case_lanes(costs, parallel=2)
        lanes2 = parallel_runner.plan_case_lanes(costs, parallel=2)
        self.assertEqual(lanes1, lanes2)
        # Within a lane the runner-REGISTRATION order is restored (each
        # lane executes its slice in serial-baseline order).
        rank = {n: i for i, (n, _) in enumerate(costs)}
        for lane in lanes1:
            self.assertEqual(lane, sorted(lane, key=rank.__getitem__))

    def test_lpt_spreads_a_heavy_family_across_lanes(self):
        # The P0 pain: one heavy family (24 status-like cases ~80s) must
        # NOT pile onto one lane — the flattening is the whole point.
        heavy = [(f"status_{i}", 80.0) for i in range(24)]
        light = [(f"cancel_{i}", 10.0) for i in range(10)]
        lanes = parallel_runner.plan_case_lanes(heavy + light, parallel=6)
        total = 24 * 80.0 + 10 * 10.0
        ideal = total / 6
        loads = [sum(w for n, w in heavy + light if n in set(lane)) for lane in lanes]
        # LPT bound: max load <= ideal + heaviest item.
        self.assertLessEqual(max(loads), ideal + 80.0)
        self.assertGreaterEqual(min(loads), ideal - 80.0)
        per_lane_heavy = [
            sum(1 for n in lane if n.startswith("status_")) for lane in lanes
        ]
        self.assertLessEqual(max(per_lane_heavy), 5)  # ceil(24/6)+1

    def test_uniform_costs_round_robin(self):
        # No timing baseline → every case weighs 1 → even counts.
        costs = [(f"c{i:02d}", 1.0) for i in range(12)]
        lanes = parallel_runner.plan_case_lanes(costs, parallel=3)
        self.assertEqual([4, 4, 4], [len(lane) for lane in lanes])


class LoadTimingBaselineTest(unittest.TestCase):
    def test_reads_case_seconds_from_aggregate_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "agg.json"
            p.write_text(
                json.dumps(
                    {
                        "summary": {"total": 2},
                        "cases": [
                            {"name": "a", "duration_ms": 1500},
                            {"name": "b", "duration_ms": 2500},
                            {"garbage": True},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            timing = parallel_runner.load_timing_baseline(str(p))
        self.assertAlmostEqual(1.5, timing["a"])
        self.assertAlmostEqual(2.5, timing["b"])
        self.assertEqual(2, len(timing))

    def test_unreadable_or_corrupt_file_returns_none(self):
        self.assertIsNone(parallel_runner.load_timing_baseline("/nonexistent/x.json"))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.json"
            p.write_text("{not json", encoding="utf-8")
            self.assertIsNone(parallel_runner.load_timing_baseline(str(p)))


class PlanCaseShardTest(unittest.TestCase):
    """_plan_case_shard: baseline wiring, fallbacks, --categories bounds."""

    _PAIRS = [
        ("cancel_a", "cancel"),
        ("cancel_b", "cancel"),
        ("status_a", "status"),
        ("status_b", "status"),
        ("kv_a", "kv"),
        ("direct_a", "direct"),
    ]

    def _args(self, **kw):
        base = {
            "shard": "case",
            "profile": "batch-window",
            "categories": None,
            "timing_json": None,
            "parallel": 2,
        }
        base.update(kw)
        return argparse.Namespace(**base)

    def _plan(self, args):
        buf = io.StringIO()
        with mock.patch.object(
            parallel_runner, "list_case_pairs", return_value=list(self._PAIRS)
        ):
            with contextlib.redirect_stderr(buf):
                lanes, weights = parallel_runner._plan_case_shard(args)
        return lanes, weights, buf.getvalue()

    def test_no_baseline_degrades_to_uniform_with_warning(self):
        lanes, weights, err = self._plan(self._args())
        self.assertIn("falling back to uniform case split", err)
        self.assertTrue(all(w == 1.0 for w in weights.values()))
        self.assertEqual([3, 3], [len(lane) for lane in lanes])

    def test_unreadable_baseline_warns_and_goes_uniform(self):
        args = self._args(timing_json="/nonexistent/t.json")
        lanes, weights, err = self._plan(args)
        self.assertIn("unreadable", err)
        self.assertTrue(all(w == 1.0 for w in weights.values()))

    def test_partial_baseline_uses_measurements_plus_family_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.json"
            p.write_text(
                json.dumps({"cases": [{"name": "cancel_a", "duration_ms": 40000}]}),
                encoding="utf-8",
            )
            args = self._args(timing_json=str(p))
            lanes, weights, err = self._plan(args)
        self.assertIn("missing from the timing baseline", err)
        self.assertAlmostEqual(40.0, weights["cancel_a"])
        self.assertEqual(
            parallel_runner.CATEGORY_WEIGHTS["status"], weights["status_a"]
        )
        self.assertEqual(
            parallel_runner.CATEGORY_WEIGHTS["direct"], weights["direct_a"]
        )

    def test_categories_subset_bounds_the_case_pool(self):
        args = self._args(categories="status,direct")
        lanes, weights, err = self._plan(args)
        packed = sorted(n for lane in lanes for n in lane)
        self.assertEqual(["direct_a", "status_a", "status_b"], packed)


class AggregateShardSchemaTest(unittest.TestCase):
    def test_case_shard_records_shard_and_lane_case_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            j0 = Path(tmp) / "cases.json"
            _write_runner_json(
                j0,
                [
                    {
                        "category": "status",
                        "name": "status_a",
                        "status": "PASS",
                        "expected_fail": False,
                        "duration_ms": 100,
                        "grade": {"achieved": "normal"},
                    }
                ],
                {
                    "total": 1,
                    "passed": 1,
                    "failed": 0,
                    "finding_confirmed": 0,
                    "finding_resolved": 0,
                },
            )
            lr = parallel_runner.LaneResult(0, ["status"], case_names=["status_a"])
            lr.runs = [("cases", 0, j0)]
            lr.wall_s = 2.0
            args = argparse.Namespace(
                parallel=1, profile="batch-window", grade="normal", shard="case"
            )
            payload = parallel_runner.aggregate([lr], args, wall_s=2.0)
        self.assertEqual("case", payload["summary"]["shard"])
        self.assertEqual(["status_a"], payload["lanes"][0]["case_names"])
        self.assertEqual(1, payload["lanes"][0]["cases"])

    def test_category_mode_namespace_stays_compatible(self):
        # Old-style Namespace (P0 call sites) without shard/mock_stride
        # attributes must still aggregate: shard defaults to "category"
        # and lanes[] carries no case_names key.
        with tempfile.TemporaryDirectory() as tmp:
            j0 = Path(tmp) / "a.json"
            _write_runner_json(
                j0,
                [
                    {
                        "category": "kv",
                        "name": "kv_a",
                        "status": "PASS",
                        "expected_fail": False,
                        "duration_ms": 100,
                        "grade": {"achieved": "normal"},
                    }
                ],
                {
                    "total": 1,
                    "passed": 1,
                    "failed": 0,
                    "finding_confirmed": 0,
                    "finding_resolved": 0,
                },
            )
            args = argparse.Namespace(
                parallel=1, profile="batch-window", grade="normal"
            )
            payload = parallel_runner.aggregate(
                [_lane_result(0, [("kv", 0, j0)])], args, wall_s=1.0
            )
        self.assertEqual("category", payload["summary"]["shard"])
        self.assertNotIn("case_names", payload["lanes"][0])


class MockStrideTest(unittest.TestCase):
    def test_max_lanes_derivation(self):
        # default band: 55151 + 2000*5 + 151 = 65302 <= 65535 → 6 lanes
        self.assertEqual(6, parallel_runner.max_lanes(2000, 55151))
        # compressed: 55151 + 500*20 + 151 = 65302 → 21 lanes
        self.assertEqual(21, parallel_runner.max_lanes(500, 55151))

    def test_lane_env_honors_mock_stride(self):
        e = parallel_runner.lane_env(3, mock_stride=500)
        self.assertEqual(
            str(parallel_runner.MOCK_BASE_GRPC_PORT + 500 * 3),
            e["FLEXLB_FT_MOCK_BASE_GRPC_PORT"],
        )
        # master group stride is untouched by --mock-stride
        self.assertEqual(
            str(parallel_runner.MASTER_HTTP_BASE + 10 * 3),
            e["FLEXLB_FT_MASTER_HTTP_PORT"],
        )

    def test_compressed_stride_footprints_stay_disjoint(self):
        stride = 500
        cap = parallel_runner.max_lanes(stride, parallel_runner.MOCK_BASE_GRPC_PORT)
        footprints = []
        for i in range(cap):
            base = parallel_runner.MOCK_BASE_GRPC_PORT + stride * i
            footprints.append(set(range(base - 1, base + 152)))
        for i in range(len(footprints)):
            for j in range(i + 1, len(footprints)):
                self.assertEqual(
                    set(),
                    footprints[i] & footprints[j],
                    f"lane {i} and lane {j} share ports",
                )


class DryRunCaseShardTest(unittest.TestCase):
    def test_dry_run_prints_case_matrix_and_warns_without_baseline(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(PARALLEL_RUNNER),
                "--shard",
                "case",
                "--parallel",
                "4",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("shard=case", proc.stdout)
        self.assertIn("case plan", proc.stdout)
        self.assertIn("uniform (no baseline)", proc.stdout)
        self.assertIn("falling back to uniform case split", proc.stderr)
        self.assertIn("port partition", proc.stdout)

    def test_dry_run_with_timing_json_prints_baseline_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.json"
            p.write_text(
                json.dumps({"cases": [{"name": "cancel_basic", "duration_ms": 1000}]}),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    str(PARALLEL_RUNNER),
                    "--shard",
                    "case",
                    "--timing-json",
                    str(p),
                    "--parallel",
                    "2",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
            )
            stderr = proc.stderr
        self.assertEqual(0, proc.returncode, stderr)
        self.assertIn("case plan", proc.stdout)
        self.assertIn("missing from the timing baseline", stderr)

    def test_eight_lanes_allowed_with_compressed_stride(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(PARALLEL_RUNNER),
                "--shard",
                "case",
                "--mock-stride",
                "500",
                "--parallel",
                "8",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("stride 500", proc.stdout)

    def test_stride_below_mock_window_is_rejected(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(PARALLEL_RUNNER),
                "--mock-stride",
                "100",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(2, proc.returncode)
        self.assertIn("--mock-stride must be >= 153", proc.stderr)

    def test_parallel_above_derived_cap_is_rejected(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(PARALLEL_RUNNER),
                "--mock-stride",
                "500",
                "--parallel",
                "22",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(2, proc.returncode)
        self.assertIn("--parallel must be 1..21", proc.stderr)


if __name__ == "__main__":
    unittest.main()
