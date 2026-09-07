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
    profile filtering still applies, unknown names exit 2;
  * --shard defaults to CASE (single entry point); category stays
    selectable;
  * timing-baseline self-maintenance — every completed run merges its
    per-case durations into the shared baseline (write_timing_baseline),
    case mode auto-reads it when --timing-json is absent, and the
    FLEXLB_FT_TIMING_BASELINE env pins the shared location (tests isolate
    it so a developer's real /tmp baseline cannot leak in);
  * _lane_ports — the fixed 159-port per-lane window (master group +
    full mock window) matches the lane_env offsets and never overlaps
    across lanes;
  * port-window preflight (_resolve_port_bases) — the default-mode
    auto-shift ladder (busy / locked / stress-band gates) vs the
    explicit-base contract (fail-fast, never shifted), plus the
    machine-level flock window lock (mutual exclusion, dies with the
    holder process, dry-run probe-and-release).

Plus --dry-run CLI smokes (the only jar-free execution paths).
"""

import argparse
import contextlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import parallel_runner  # noqa: E402

PARALLEL_RUNNER = TOOLS_DIR / "parallel_runner.py"
RUNNER = TOOLS_DIR / "flexlb_functional_tests.py"

# Default-stride lane cap (mock band 55151 + 500*(N-1) + 151 <= 65535).
DEFAULT_LANES = parallel_runner.max_lanes(
    parallel_runner.MOCK_PORT_STRIDE, parallel_runner.MOCK_BASE_GRPC_PORT
)

# Representative family weights for deterministic lane-packing tests.
_FAMILY_WEIGHTS = {
    "master": 60 * 8,
    "engine_fault": 30 * 13,
    "status": 15 * 24,
    "admission": 30 * 11,
    "elastic": 40 * 8,
    "kv": 20 * 15,
    "cancel": 12 * 13,
    "balance": 15 * 6,
    "priority": 30 * 1,
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
        # ...and the packing stays within ~1.5x of the lightest lane.
        self.assertLessEqual(max(loads) / min(loads), 1.5)

    def test_single_lane_plan_uses_legacy_all_path(self):
        args = argparse.Namespace(parallel=1, profile="batch-window", categories=None)
        lanes, _ = parallel_runner._plan(args)
        self.assertEqual([["all"]], lanes)

    def test_partial_subset_stays_per_category_at_parallel_one(self):
        # --categories master at parallel=1 must NOT become `--category all`
        # (that would run the unrequested categories too).
        args = argparse.Namespace(
            parallel=1, profile="batch-window", categories="master"
        )
        with mock.patch.object(
            parallel_runner, "family_weights", return_value=dict(_FAMILY_WEIGHTS)
        ):
            lanes, weights = parallel_runner._plan(args)
        self.assertEqual([["master"]], lanes)
        self.assertEqual({"master": _FAMILY_WEIGHTS["master"]}, weights)

    def test_kebab_case_category_normalizes(self):
        args = argparse.Namespace(
            parallel=2, profile="batch-window", categories="engine-fault,master"
        )
        with mock.patch.object(
            parallel_runner, "family_weights", return_value=dict(_FAMILY_WEIGHTS)
        ):
            lanes, weights = parallel_runner._plan(args)
        packed = sorted(c for lane in lanes for c in lane)
        self.assertEqual(["engine_fault", "master"], packed)
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
            base = parallel_runner.MOCK_BASE_GRPC_PORT + (
                parallel_runner.MOCK_PORT_STRIDE * i
            )
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
        # By design (harness port plan): the single-master path and HA
        # Tier-1 A both bind the group head — they are two MUTUALLY
        # EXCLUSIVE env shapes inside one runner process, never concurrent.
        env = parallel_runner.lane_env(2)
        self.assertEqual(
            env["FLEXLB_FT_MASTER_HTTP_PORT"],
            env["FLEXLB_FT_HA_MASTER_A_HTTP_PORT"],
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
            self.assertEqual(
                str(
                    parallel_runner.MOCK_BASE_GRPC_PORT
                    + parallel_runner.MOCK_PORT_STRIDE * i
                ),
                env["FLEXLB_FT_MOCK_BASE_GRPC_PORT"],
            )

    def test_overlay_touches_only_the_five_port_keys(self):
        # Operator env (e.g. FLEXLB_FT_HA_DUAL_MASTER=1) must pass through
        # untouched — the overlay is exactly the port partition.
        self.assertEqual(
            {
                "FLEXLB_FT_MASTER_HTTP_PORT",
                "FLEXLB_FT_MASTER_MANAGEMENT_PORT",
                "FLEXLB_FT_HA_MASTER_A_HTTP_PORT",
                "FLEXLB_FT_HA_MASTER_B_HTTP_PORT",
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
        self.assertEqual(
            str(50000 + parallel_runner.MOCK_PORT_STRIDE),
            e1["FLEXLB_FT_MOCK_BASE_GRPC_PORT"],
        )

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


# Point the shared-baseline lookup at a path that never exists: CLI
# subprocess tests must not read (or write) the developer's real
# /tmp/flexlb_ft_timing_baseline.json — otherwise a locally established
# baseline would flip "uniform (no baseline)" assertions.
def _env_without_baseline() -> dict[str, str]:
    return {
        **os.environ,
        "FLEXLB_FT_TIMING_BASELINE": "/nonexistent/flexlb_ft_test_baseline.json",
    }


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
        # Explicit category shard: the family-level branch keeps its own
        # dry-run smoke (the default branch is covered by ShardDefaultTest).
        proc = subprocess.run(
            [
                sys.executable,
                str(PARALLEL_RUNNER),
                "--shard",
                "category",
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
            [sys.executable, str(PARALLEL_RUNNER), "--parallel", "22", "--dry-run"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(2, proc.returncode)
        self.assertIn("--parallel must be 1..21", proc.stderr)

    @staticmethod
    def _drop_stashed_window_lock():
        # A non-dry main() earlier in THIS process may have stashed the
        # window locks in the module-level holder (held until process
        # exit); drop them so the retake below is not blocked by our
        # own flocks.
        holders = parallel_runner._WINDOW_LOCK_FILES
        parallel_runner._WINDOW_LOCK_FILES = None
        parallel_runner._close_window_locks(holders)

    def test_dry_run_shows_port_provenance_and_status(self):
        # The preflight runs inside the CLI dry-run: bases provenance
        # line + per-lane FREE/BUSY tags must be visible, and a dry-run
        # subprocess must not HOLD the window lock past its exit
        # (probe-and-release; only inert lock files stay in /tmp).
        self._drop_stashed_window_lock()
        proc = subprocess.run(
            [sys.executable, str(PARALLEL_RUNNER), "--dry-run"],
            capture_output=True,
            text=True,
            env=_env_without_baseline(),
        )
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("bases:", proc.stdout)
        # A selected candidate's lanes are FREE by definition (a busy
        # candidate is never selected), so FREE shows up even after an
        # auto-shift on a busy host.
        self.assertIn("FREE", proc.stdout)
        retake = parallel_runner._try_window_lock(18080, 55151, 500, 4)
        if retake is None:
            # A REAL run on the same default window may legitimately
            # hold the machine locks right now — not this smoke's
            # business.  (CLI default --parallel 4 / stride 500.)
            self.skipTest("default window lock currently held by a real run")
        parallel_runner._close_window_locks(retake)


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
        ("master_a", "master"),
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
            with mock.patch.dict(
                os.environ, {"FLEXLB_FT_TIMING_BASELINE": "/nonexistent/b.json"}
            ):
                with contextlib.redirect_stderr(buf):
                    lanes, weights = parallel_runner._plan_case_shard(args)
        return lanes, weights, buf.getvalue()

    def test_no_baseline_degrades_to_uniform_quietly(self):
        # timing_json=None + no shared baseline on disk = the FIRST run
        # on a host: uniform split is the expected state, not a warning.
        lanes, weights, err = self._plan(self._args())
        self.assertEqual("", err)
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
            parallel_runner.CATEGORY_WEIGHTS["master"], weights["master_a"]
        )

    def test_categories_subset_bounds_the_case_pool(self):
        args = self._args(categories="status,master")
        lanes, weights, err = self._plan(args)
        packed = sorted(n for lane in lanes for n in lane)
        self.assertEqual(["master_a", "status_a", "status_b"], packed)


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
        # wide stride: 55151 + 2000*5 + 151 = 65302 <= 65535 → 6 lanes
        self.assertEqual(6, parallel_runner.max_lanes(2000, 55151))
        # default band: 55151 + 500*20 + 151 = 65302 → 21 lanes
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
    def test_dry_run_case_shard_without_baseline_is_quiet_uniform(self):
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
            env=_env_without_baseline(),
        )
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("shard=case", proc.stdout)
        self.assertIn("case plan", proc.stdout)
        self.assertIn("uniform (no baseline", proc.stdout)
        # First run (no shared baseline yet) must stay QUIET on stderr —
        # a warning would suggest something needs fixing when it doesn't.
        self.assertNotIn("falling back", proc.stderr)
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
            env=_env_without_baseline(),
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


class ShardDefaultTest(unittest.TestCase):
    """--shard defaults to case (the single-entry-point decision)."""

    def test_default_dry_run_is_case_shard(self):
        proc = subprocess.run(
            [sys.executable, str(PARALLEL_RUNNER), "--parallel", "2", "--dry-run"],
            capture_output=True,
            text=True,
            env=_env_without_baseline(),
        )
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("shard=case", proc.stdout)
        self.assertIn("case plan", proc.stdout)

    def test_category_shard_still_selectable(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(PARALLEL_RUNNER),
                "--shard",
                "category",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("shard=category", proc.stdout)
        self.assertIn("lane plan", proc.stdout)


class TimingBaselineMaintenanceTest(unittest.TestCase):
    """Shared-baseline self-maintenance: write-merge, auto-read, env pin."""

    def test_write_merges_and_keeps_untouched_cases(self):
        # A partial run (e.g. --categories cancel) must refresh only the
        # cases it ran — a prior full-run baseline keeps the rest.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "b.json"
            p.write_text(
                json.dumps(
                    {
                        "cases": [
                            {"name": "a", "duration_ms": 1000},
                            {"name": "b", "duration_ms": 2000},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            payload = {
                "cases": [
                    {"name": "a", "duration_ms": 1500},  # refreshed
                    {"name": "c", "duration_ms": 3000},  # added
                ]
            }
            path, n = parallel_runner.write_timing_baseline(payload, path=p)
            stored = parallel_runner.load_timing_baseline(str(path))
        self.assertEqual(3, n)
        self.assertAlmostEqual(1.5, stored["a"])
        self.assertAlmostEqual(2.0, stored["b"])  # kept from the prior run
        self.assertAlmostEqual(3.0, stored["c"])

    def test_write_rebuilds_from_corrupt_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "b.json"
            p.write_text("{corrupt", encoding="utf-8")
            payload = {"cases": [{"name": "x", "duration_ms": 700}]}
            path, n = parallel_runner.write_timing_baseline(payload, path=p)
            stored = parallel_runner.load_timing_baseline(str(path))
        self.assertEqual(1, n)
        self.assertAlmostEqual(0.7, stored["x"])

    def test_default_path_env_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "custom.json"
            with mock.patch.dict(os.environ, {"FLEXLB_FT_TIMING_BASELINE": str(p)}):
                self.assertEqual(p, parallel_runner._default_timing_baseline())
                path, _ = parallel_runner.write_timing_baseline(
                    {"cases": [{"name": "x", "duration_ms": 100}]}
                )
            self.assertEqual(p, path)
            self.assertTrue(p.exists())

    def test_plan_reads_default_baseline_automatically(self):
        # No --timing-json given: the shared baseline is picked up on its
        # own ("auto baseline"), cases absent from it still fall back to
        # the family per-case weight.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "b.json"
            p.write_text(
                json.dumps({"cases": [{"name": "cancel_a", "duration_ms": 40000}]}),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                shard="case",
                profile="batch-window",
                categories=None,
                timing_json=None,
                parallel=2,
            )
            with mock.patch.object(
                parallel_runner,
                "list_case_pairs",
                return_value=list(PlanCaseShardTest._PAIRS),
            ):
                with mock.patch.dict(os.environ, {"FLEXLB_FT_TIMING_BASELINE": str(p)}):
                    with contextlib.redirect_stderr(io.StringIO()):
                        lanes, weights = parallel_runner._plan_case_shard(args)
        self.assertAlmostEqual(40.0, weights["cancel_a"])
        self.assertIn("auto baseline", args.cost_source)
        self.assertEqual(
            parallel_runner.CATEGORY_WEIGHTS["status"], weights["status_a"]
        )


class BaselineWriteWiringTest(unittest.TestCase):
    """main() updates the shared baseline after a completed run (any mode)."""

    def setUp(self):
        # A non-dry main() stashes the selected window locks in the
        # module-level holder for the rest of the PROCESS; release them
        # after each test so later tests in this same process are not
        # blocked by our own flocks (flock is per-open-file-description:
        # a second open of the same lock file inside THIS process
        # would EWOULDBLOCK).
        def _drop():
            holders = parallel_runner._WINDOW_LOCK_FILES
            parallel_runner._WINDOW_LOCK_FILES = None
            parallel_runner._close_window_locks(holders)

        self.addCleanup(_drop)
        # main() runs IN-PROCESS here, so it must not depend on the
        # machine's real port/lock state (110常态: default band often
        # busy → real auto-shift + FLEXLB_FT_PARALLEL_*_BASE pollution
        # that cascades into later alphabetical test classes).  Isolate
        # the probe and the lock dir, and restore the two base env keys.
        probe = mock.patch.object(parallel_runner, "port_in_use", return_value=False)
        probe.start()
        self.addCleanup(probe.stop)
        locktmp = tempfile.TemporaryDirectory(prefix="ft_wiring_lock_")
        self.addCleanup(locktmp.cleanup)
        lockdir = mock.patch.object(
            parallel_runner, "PORT_WINDOW_LOCK_DIR", Path(locktmp.name)
        )
        lockdir.start()
        self.addCleanup(lockdir.stop)
        base_keys = (
            "FLEXLB_FT_PARALLEL_MASTER_BASE",
            "FLEXLB_FT_PARALLEL_MOCK_BASE",
        )
        saved_env = {k: os.environ.get(k) for k in base_keys}

        def _restore_env():
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.addCleanup(_restore_env)

    def _fake_lane(self, lane_idx, items, args, out_dir, run_stamp):
        json_path = Path(out_dir) / f"lane{lane_idx}" / "cases.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        _write_runner_json(
            json_path,
            [
                {
                    "category": "cancel",
                    "name": "cancel_a",
                    "status": "PASS",
                    "expected_fail": False,
                    "duration_ms": 4321,
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
        lr = parallel_runner.LaneResult(lane_idx, ["cancel"], case_names=["cancel_a"])
        lr.runs = [("cases", 0, json_path)]
        lr.wall_s = 5.0
        return lr

    def test_completed_run_updates_shared_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            out_json = d / "agg.json"
            baseline = d / "baseline.json"
            argv = [
                "parallel_runner.py",
                "--parallel",
                "1",
                "--json",
                str(out_json),
                "--out-dir",
                str(d / "out"),
            ]
            with mock.patch.object(sys, "argv", argv):
                plan = ([["cancel_a"]], {"cancel_a": 1.0})
                with mock.patch.object(
                    parallel_runner,
                    "_plan",
                    return_value=plan,
                ):
                    with mock.patch.object(
                        parallel_runner, "run_lane", side_effect=self._fake_lane
                    ):
                        with mock.patch.dict(
                            os.environ, {"FLEXLB_FT_TIMING_BASELINE": str(baseline)}
                        ):
                            buf = io.StringIO()
                            with contextlib.redirect_stdout(buf):
                                rc = parallel_runner.main()
            stored = parallel_runner.load_timing_baseline(str(baseline))
        self.assertEqual(0, rc)
        self.assertAlmostEqual(4.321, stored["cancel_a"])
        self.assertIn("timing baseline updated", buf.getvalue())

    def test_dry_run_never_writes_the_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline.json"
            argv = ["parallel_runner.py", "--parallel", "2", "--dry-run"]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(
                    parallel_runner,
                    "_plan",
                    return_value=([["a"], ["b"]], {"a": 1.0, "b": 1.0}),
                ):
                    with mock.patch.dict(
                        os.environ, {"FLEXLB_FT_TIMING_BASELINE": str(baseline)}
                    ):
                        with contextlib.redirect_stdout(io.StringIO()):
                            rc = parallel_runner.main()
        self.assertEqual(0, rc)
        self.assertFalse(baseline.exists())


class LanePortsTest(unittest.TestCase):
    """_lane_ports: the fixed 159-port per-lane window."""

    def test_every_lane_covers_exactly_159_ports(self):
        # 6 master-group ports + the full 153-port mock window
        # (http control base-1, engines, victim zone) — fixed width
        # regardless of engine count or lane index.
        cases = [
            (0, 500, 18080, 55151),
            (3, 500, 18080, 55151),
            (2, 2000, 20080, 50000),
        ]
        for lane, stride, m, b in cases:
            ports = parallel_runner._lane_ports(lane, stride, m, b)
            self.assertEqual(159, len(ports), (lane, stride, m, b))
            self.assertEqual(len(ports), len(set(ports)), "duplicates")

    def test_lanes_never_overlap(self):
        for stride, m, b in ((500, 18080, 55151), (2000, 20080, 50000)):
            windows = [
                set(parallel_runner._lane_ports(i, stride, m, b)) for i in range(6)
            ]
            for i in range(len(windows)):
                for j in range(i + 1, len(windows)):
                    self.assertEqual(set(), windows[i] & windows[j], (stride, i, j))

    def test_matches_lane_env_offsets(self):
        # _lane_ports is the port-set twin of lane_env: master head
        # m..m+5 and mock window base-1..base+151 at the SAME offsets
        # the env overlay pins per lane (a mismatch would make the
        # preflight probe different ports than the lanes actually bind).
        with mock.patch.dict(os.environ, {}, clear=True):
            for lane, stride in ((0, 500), (4, 500), (2, 2000)):
                env = parallel_runner.lane_env(lane, mock_stride=stride)
                m = int(env["FLEXLB_FT_MASTER_HTTP_PORT"])
                b = int(env["FLEXLB_FT_MOCK_BASE_GRPC_PORT"])
                self.assertEqual(
                    set(range(m, m + 6)) | set(range(b - 1, b + 152)),
                    set(parallel_runner._lane_ports(lane, stride, 18080, 55151)),
                )


class PortPreflightResolveTest(unittest.TestCase):
    """_resolve_port_bases: default auto-shift vs explicit contract.

    port_in_use is ALWAYS mocked (never bind the real 18080/55151 band)
    and the lock dir is always an isolated temp dir — the only real
    syscalls under test are the flock ones.
    """

    def setUp(self):
        lockdir = Path(tempfile.mkdtemp(prefix="ft_portlock_"))
        patcher = mock.patch.object(parallel_runner, "PORT_WINDOW_LOCK_DIR", lockdir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._release_stashed_lock)

    @staticmethod
    def _release_stashed_lock():
        holders = parallel_runner._WINDOW_LOCK_FILES
        parallel_runner._WINDOW_LOCK_FILES = None
        parallel_runner._close_window_locks(holders)

    def _args(self, parallel=6, mock_stride=500, dry_run=False):
        return argparse.Namespace(
            parallel=parallel, mock_stride=mock_stride, dry_run=dry_run
        )

    def test_all_free_default_keeps_bases_and_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(
                parallel_runner, "port_in_use", return_value=False
            ) as probe:
                args = self._args()
                parallel_runner._resolve_port_bases(args)
            self.assertNotIn("FLEXLB_FT_PARALLEL_MASTER_BASE", os.environ)
            self.assertNotIn("FLEXLB_FT_PARALLEL_MOCK_BASE", os.environ)
        self.assertEqual("default 18080/55151", args.port_provenance)
        self.assertEqual({i: "FREE" for i in range(6)}, args.lane_port_status)
        # Full-window probing: 6 lanes x 159 ports.
        self.assertEqual(6 * 159, probe.call_count)

    def test_lane0_master_busy_auto_shifts_whole_matrix(self):
        def busy(port, host):
            return 18080 <= port <= 18085  # lane 0's master group

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(parallel_runner, "port_in_use", side_effect=busy):
                buf = io.StringIO()
                with contextlib.redirect_stderr(buf):
                    args = self._args()
                    parallel_runner._resolve_port_bases(args)
                # k=1: master 18080-10*6, mock 55151-500*6.
                self.assertEqual("18020", os.environ["FLEXLB_FT_PARALLEL_MASTER_BASE"])
                self.assertEqual("52151", os.environ["FLEXLB_FT_PARALLEL_MOCK_BASE"])
        self.assertIn("auto", args.port_provenance)
        self.assertIn("18080/55151 -> 18020/52151", args.port_provenance)
        warning = buf.getvalue()
        self.assertIn("lane 0: master 18080", warning)
        self.assertIn("auto-shifting bases 18080/55151 -> 18020/52151", warning)
        self.assertIn("to make it a contract", warning)
        # The selected matrix stays under the stress band.
        self.assertLess(
            parallel_runner._matrix_tail(52151, 500, 6),
            parallel_runner.STRESS_BAND_FLOOR,
        )

    def test_explicit_busy_fails_fast_with_lane_diagnosis(self):
        def busy(port, host):
            return 18300 <= port <= 18305

        with mock.patch.dict(
            os.environ,
            {"FLEXLB_FT_PARALLEL_MASTER_BASE": "18300"},
            clear=True,
        ):
            with mock.patch.object(parallel_runner, "port_in_use", side_effect=busy):
                args = self._args()
                with self.assertRaises(SystemExit) as caught:
                    parallel_runner._resolve_port_bases(args)
                # The contract env survives verbatim — never shifted,
                # never rewritten.
                self.assertEqual(
                    "18300",
                    os.environ["FLEXLB_FT_PARALLEL_MASTER_BASE"],
                )
        message = str(caught.exception)
        self.assertIn("lane 0", message)
        self.assertIn("18300", message)
        self.assertIn("BUSY", message)
        self.assertIn("contract", message)

    def test_exhausted_candidates_exit_listing_each(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(parallel_runner, "port_in_use", return_value=True):
                with self.assertRaises(SystemExit) as caught:
                    parallel_runner._resolve_port_bases(self._args())
        message = str(caught.exception)
        self.assertIn("no usable port window", message)
        self.assertIn("k=0 bases 18080/55151", message)
        self.assertIn("k=1 bases 18020/52151", message)

    def test_stress_band_hard_bound_shifts_even_when_free(self):
        # stride 2000: k=0 tail 65302 >= 61000 is rejected on the HARD
        # stress-band bound alone (all ports FREE), selecting k=1
        # (mock base 55151 - 2000*6 = 43151, tail 53302 < 61000).
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(parallel_runner, "port_in_use", return_value=False):
                buf = io.StringIO()
                with contextlib.redirect_stderr(buf):
                    args = self._args(mock_stride=2000)
                    parallel_runner._resolve_port_bases(args)
                self.assertEqual("43151", os.environ["FLEXLB_FT_PARALLEL_MOCK_BASE"])
                self.assertEqual("18020", os.environ["FLEXLB_FT_PARALLEL_MASTER_BASE"])
        self.assertIn("auto", args.port_provenance)
        selected_tail = parallel_runner._matrix_tail(43151, 2000, 6)
        self.assertEqual(53302, selected_tail)
        self.assertLess(selected_tail, parallel_runner.STRESS_BAND_FLOOR)
        self.assertIn("stress band", buf.getvalue())

    def test_explicit_stress_band_crossing_only_warns(self):
        # An explicit base landing in the 61000+ stress/lease band must
        # NOT exit (leased-but-unlistened ports are invisible to bind
        # probing — refusing would false-positive); it warns.
        with mock.patch.dict(
            os.environ,
            {"FLEXLB_FT_PARALLEL_MOCK_BASE": "62000"},
            clear=True,
        ):
            with mock.patch.object(parallel_runner, "port_in_use", return_value=False):
                buf = io.StringIO()
                with contextlib.redirect_stderr(buf):
                    args = self._args()
                    parallel_runner._resolve_port_bases(args)  # no exit
                self.assertEqual("62000", os.environ["FLEXLB_FT_PARALLEL_MOCK_BASE"])
        self.assertEqual("explicit 18080/62000", args.port_provenance)
        self.assertIn("stress band", buf.getvalue())
        self.assertIn("contract", buf.getvalue())

    def test_dry_run_explicit_busy_shows_status_without_exiting(self):
        def busy(port, host):
            return 18300 <= port <= 18305

        with mock.patch.dict(
            os.environ,
            {"FLEXLB_FT_PARALLEL_MASTER_BASE": "18300"},
            clear=True,
        ):
            with mock.patch.object(parallel_runner, "port_in_use", side_effect=busy):
                buf = io.StringIO()
                with contextlib.redirect_stderr(buf):
                    args = self._args(dry_run=True)
                    parallel_runner._resolve_port_bases(args)
        self.assertIn("BUSY(:18300", args.lane_port_status[0])
        self.assertEqual("FREE", args.lane_port_status[1])
        warning = buf.getvalue()
        self.assertIn("fail-fast", warning)
        self.assertIn("contract", warning)


class WindowLockTest(unittest.TestCase):
    """Machine-level window locks: exclusion, crash immunity, dry-run skip."""

    def setUp(self):
        self.lockdir = Path(tempfile.mkdtemp(prefix="ft_portlock_"))
        patcher = mock.patch.object(
            parallel_runner, "PORT_WINDOW_LOCK_DIR", self.lockdir
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _spawn_mock_lock_holder(self, master, mock, stride=500, n_lanes=6):
        """A child that flocks the mock-window lock files for 60s.

        Holding just ONE side's files (the mock windows) must be
        enough to block the whole matrix: locks are per conflict
        domain, so partial overlap is still mutual exclusion.
        """
        lockfiles = [
            p
            for p in parallel_runner._window_lock_paths(master, mock, stride, n_lanes)
            if p.name.startswith("g")
        ]
        return subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import fcntl, time\n"
                "fs = [" + ", ".join(repr(str(f)) for f in lockfiles) + "]\n"
                "hs = [open(f, 'a') for f in fs]\n"
                "for h in hs:\n"
                "    fcntl.flock(h, fcntl.LOCK_EX)\n"
                "time.sleep(60)\n",
            ]
        )

    def _wait_until_child_holds(self, master, mock):
        """Poll until the child's flock is observable (or 10s timeout)."""
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            probe = parallel_runner._try_window_lock(master, mock, 500, 6)
            if probe is None:
                return True
            parallel_runner._close_window_locks(probe)
            time.sleep(0.05)
        return False

    def test_second_taker_on_same_key_is_refused(self):
        first = parallel_runner._try_window_lock(18080, 55151, 500, 6)
        self.assertIsNotNone(first)
        try:
            second = parallel_runner._try_window_lock(18080, 55151, 500, 6)
            self.assertIsNone(second)
        finally:
            parallel_runner._close_window_locks(first)
        # Released → immediately re-takeable (no stale state).
        again = parallel_runner._try_window_lock(18080, 55151, 500, 6)
        self.assertIsNotNone(again)
        parallel_runner._close_window_locks(again)

    def test_lock_dies_with_the_holding_process(self):
        child = self._spawn_mock_lock_holder(18080, 55151)
        try:
            self.assertTrue(
                self._wait_until_child_holds(18080, 55151),
                "child never acquired the lock",
            )
        finally:
            child.kill()
            child.wait()
        # kill -9 releases the flock instantly — no stale-lock cleanup
        # is ever needed (that is the whole reason flock was chosen).
        after = parallel_runner._try_window_lock(18080, 55151, 500, 6)
        self.assertIsNotNone(after)
        parallel_runner._close_window_locks(after)

    def test_dry_run_skips_locked_candidate_and_shifts(self):
        child = self._spawn_mock_lock_holder(18080, 55151)
        try:
            self.assertTrue(
                self._wait_until_child_holds(18080, 55151),
                "child never acquired the lock",
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(
                    parallel_runner, "port_in_use", return_value=False
                ):
                    buf = io.StringIO()
                    with contextlib.redirect_stderr(buf):
                        args = argparse.Namespace(
                            parallel=6, mock_stride=500, dry_run=True
                        )
                        parallel_runner._resolve_port_bases(args)
                    # k=0 is LOCKED → k=1 selected and pinned into env.
                    self.assertEqual(
                        "18020",
                        os.environ["FLEXLB_FT_PARALLEL_MASTER_BASE"],
                    )
                    self.assertEqual(
                        "52151",
                        os.environ["FLEXLB_FT_PARALLEL_MOCK_BASE"],
                    )
                    self.assertIn("auto", args.port_provenance)
                    # Probe-and-release: the dry-run holds nothing.
                    retake = parallel_runner._try_window_lock(18020, 52151, 500, 6)
                    self.assertIsNotNone(retake)
                    parallel_runner._close_window_locks(retake)
        finally:
            child.kill()
            child.wait()

    def test_one_sided_base_shift_still_excludes_on_shared_side(self):
        # MAJOR-2 regression: run B pins ONLY the master base while
        # run A holds the default window — the old (m, b)-pair lock
        # key let both proceed into the SAME mock ports.  Per-lane-
        # interval lock files must refuse B on the shared mock side,
        # and the mirror case (pinning only the mock base) on the
        # shared master side.
        first = parallel_runner._try_window_lock(18080, 55151, 500, 6)
        self.assertIsNotNone(first)
        try:
            self.assertIsNone(parallel_runner._try_window_lock(18300, 55151, 500, 6))
            self.assertIsNone(parallel_runner._try_window_lock(18080, 56151, 500, 6))
            # Fully disjoint windows still coexist.
            other = parallel_runner._try_window_lock(19300, 60151, 500, 6)
            self.assertIsNotNone(other)
            parallel_runner._close_window_locks(other)
        finally:
            parallel_runner._close_window_locks(first)

    def test_smaller_lane_count_still_excludes_shared_units(self):
        # A 6-lane holder and a 4-lane contender share the first four
        # lane units on BOTH sides — interval OVERLAP (not just exact
        # key equality) must exclude.
        first = parallel_runner._try_window_lock(18080, 55151, 500, 6)
        self.assertIsNotNone(first)
        try:
            self.assertIsNone(parallel_runner._try_window_lock(18080, 55151, 500, 4))
        finally:
            parallel_runner._close_window_locks(first)

    def test_lock_files_created_world_writable_despite_umask(self):
        # MAJOR-1: os.open's 0o666 is AND-ed with the umask (022 →
        # 0644), which would stop a different uid from opening the
        # files to contend; the fchmod best-effort must restore 0o666.
        holders = parallel_runner._try_window_lock(18080, 55151, 500, 6)
        self.assertIsNotNone(holders)
        try:
            for path in parallel_runner._window_lock_paths(18080, 55151, 500, 6):
                self.assertEqual(0o666, path.stat().st_mode & 0o777, path)
        finally:
            parallel_runner._close_window_locks(holders)

    def test_lock_dir_created_sticky_world_writable_despite_umask(self):
        # mkdir's mode arg is umask-filtered (022 → 0755) and some
        # platforms (macOS) ignore it outright — the 0o1777 lock-dir
        # promise (sticky bit + others-writable: cross-uid can contend,
        # own stale files cleanable) must come from the explicit chmod.
        with tempfile.TemporaryDirectory() as tmp:
            lockdir = Path(tmp) / "portlocks"  # does not exist yet
            with mock.patch.object(parallel_runner, "PORT_WINDOW_LOCK_DIR", lockdir):
                holders = parallel_runner._try_window_lock(18080, 55151, 500, 4)
                self.assertIsNotNone(holders)
                try:
                    mode = stat.S_IMODE(os.stat(lockdir).st_mode)
                    self.assertEqual(0o1777, mode)
                finally:
                    parallel_runner._close_window_locks(holders)


if __name__ == "__main__":
    unittest.main()
