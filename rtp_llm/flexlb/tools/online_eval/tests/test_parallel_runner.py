"""Port window allocation, mutual exclusion and lane environment safety."""

import argparse
import contextlib
import io
import os
import select
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parents[1]

import parallel_runner

PARALLEL_RUNNER = TOOLS_DIR / "parallel_runner.py"


DEFAULT_LANES = parallel_runner.max_lanes(
    parallel_runner.MOCK_PORT_STRIDE, parallel_runner.MOCK_BASE_GRPC_PORT
)

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

    @mock.patch.dict(
        os.environ,
        {"FLEXLB_FT_PARALLEL_MASTER_BASE": "", "FLEXLB_FT_PARALLEL_MOCK_BASE": ""},
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


class MockStrideTest(unittest.TestCase):
    def test_max_lanes_derivation(self):
        # wide stride: 55151 + 2000*5 + 151 = 65302 <= 65535 → 6 lanes
        self.assertEqual(6, parallel_runner.max_lanes(2000, 55151))
        # default band: 55151 + 500*20 + 151 = 65302 → 21 lanes
        self.assertEqual(21, parallel_runner.max_lanes(500, 55151))

    @mock.patch.dict(
        os.environ,
        {"FLEXLB_FT_PARALLEL_MASTER_BASE": "", "FLEXLB_FT_PARALLEL_MOCK_BASE": ""},
    )
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
        tmp = tempfile.TemporaryDirectory(prefix="ft_portlock_")
        self.addCleanup(tmp.cleanup)
        self.lockdir = Path(tmp.name)
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

    def test_partial_and_cross_side_intersections_are_refused(self):
        first = parallel_runner._try_window_lock(18080, 55151, 500, 1)
        self.assertIsNotNone(first)
        try:
            for master, base in [(18083, 56000), (19000, 55200), (55152, 57000)]:
                with self.subTest(master=master, mock=base):
                    other = parallel_runner._try_window_lock(master, base, 500, 1)
                    try:
                        self.assertIsNone(other)
                    finally:
                        parallel_runner._close_window_locks(other)
            adjacent = parallel_runner._try_window_lock(18086, 55304, 500, 1)
            self.assertIsNotNone(adjacent)
            parallel_runner._close_window_locks(adjacent)
        finally:
            parallel_runner._close_window_locks(first)
        # Overlapping inert filenames do not block a new reservation.
        later = parallel_runner._try_window_lock(18083, 55200, 500, 1)
        self.assertIsNotNone(later)
        parallel_runner._close_window_locks(later)

    def test_matrix_cannot_overlap_its_own_master_and_mock_ports(self):
        self.assertIsNone(parallel_runner._try_window_lock(18080, 18081, 500, 1))
        self.assertIn("matrix overlaps itself", parallel_runner._WINDOW_LOCK_DIAGNOSTIC)

    def test_explicit_locked_window_fails_without_shifting(self):
        first = parallel_runner._try_window_lock(18080, 55151, 500, 1)
        self.assertIsNotNone(first)
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "FLEXLB_FT_PARALLEL_MASTER_BASE": "18083",
                    "FLEXLB_FT_PARALLEL_MOCK_BASE": "56000",
                },
                clear=True,
            ), mock.patch.object(parallel_runner, "port_in_use", return_value=False):
                args = argparse.Namespace(parallel=1, mock_stride=500, dry_run=False)
                with self.assertRaisesRegex(
                    SystemExit, "explicit port window unusable"
                ) as ctx:
                    parallel_runner._resolve_port_bases(args)
                self.assertIn("m18080_18085.lock", str(ctx.exception))
                self.assertEqual("18083", os.environ["FLEXLB_FT_PARALLEL_MASTER_BASE"])
        finally:
            parallel_runner._close_window_locks(first)

    def test_simultaneous_cold_start_with_different_lane_counts(self):
        code = (
            "import sys\nfrom pathlib import Path\n"
            f"sys.path.insert(0, {str(TOOLS_DIR)!r})\n"
            "import parallel_runner as p\n"
            f"p.PORT_WINDOW_LOCK_DIR = Path({str(self.lockdir)!r})\n"
            "print('READY', flush=True)\nsys.stdin.readline()\n"
            "h = p._try_window_lock(int(sys.argv[1]), int(sys.argv[2]), 500, int(sys.argv[3]))\n"
            "print('HELD' if h is not None else 'BLOCKED', flush=True)\n"
            "sys.stdin.readline()\np._close_window_locks(h)\n"
        )
        children = [
            subprocess.Popen(
                [sys.executable, "-u", "-c", code, str(m), str(b), str(n)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for m, b, n in [(18080, 55151, 6), (18083, 55200, 4)]
        ]

        def line(child):
            self.assertTrue(
                select.select([child.stdout], [], [], 10)[0], "child timeout"
            )
            return child.stdout.readline().strip()

        try:
            for child in children:
                self.assertEqual("READY", line(child))
            for child in children:
                child.stdin.write("go\n")
                child.stdin.flush()
            self.assertCountEqual(["HELD", "BLOCKED"], [line(c) for c in children])
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.communicate(timeout=10)

    def test_failed_acquisition_releases_gate_and_partial_locks(self):
        original = parallel_runner._open_window_lock

        def fail_mock(path, **kwargs):
            if path.name.startswith("g"):
                raise PermissionError("unwritable window")
            return original(path, **kwargs)

        with mock.patch.object(
            parallel_runner, "_open_window_lock", side_effect=fail_mock
        ):
            self.assertIsNone(parallel_runner._try_window_lock(18080, 55151, 500, 1))
        again = parallel_runner._try_window_lock(18080, 55151, 500, 1)
        self.assertIsNotNone(again)
        parallel_runner._close_window_locks(again)

    def test_existing_lock_is_opened_without_create_flag(self):
        path = self.lockdir / ".admission.lock"
        path.touch()
        original = os.open

        def protected_open(name, flags, *args):
            if flags & os.O_CREAT and not flags & os.O_EXCL:
                raise PermissionError("Linux sticky-directory protected_regular")
            return original(name, flags, *args)

        with mock.patch.object(os, "open", side_effect=protected_open):
            with parallel_runner._open_window_lock(path) as holder:
                self.assertFalse(holder.closed)

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
