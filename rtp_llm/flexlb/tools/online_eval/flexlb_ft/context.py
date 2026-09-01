"""Case registry + execution context shared by all case categories.

Terminology (unified 2026-09, suite reorg task #85):

  * mock engine CASE test (场景测试) — this framework: the
    flexlb_functional_tests.py runner and the flexlb_ft/cases/ category
    modules (cancel / status / kv / balance / elastic / engine_fault /
    master / admission / direct).
  * mock engine STRESS test (压测) — the online_eval load pipeline
    (run_online_eval.sh + eval scripts), a separate lineage.

The legacy "e2e test" / "chaos test" suite wording is retired; fault
injection is a MECHANISM inside case tests, not a suite name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .engine_ops import EngineOps
from .harness import EnvManager, EnvSpec, default_perf, profile_dispatches_batch

SMOKE_LABEL_PERF = default_perf()


@dataclass
class CaseDef:
    """One test case: name, category, callable and optional profile
    restriction.

    ``category`` is one of the nine scenario categories (see
    flexlb_ft/cases/): cancel, status, kv, balance, elastic,
    engine_fault, master, admission, direct.

    ``profiles`` restricts a case to specific scheduling profiles (None = all
    profiles apply).  ``requires`` declares semantic capabilities the case
    needs (vocabulary: see harness.PROFILE_CAPS, e.g. ``enqueue_batch``,
    ``generate_stream``); a case runs only under profiles whose capability
    set is a superset.
    """

    name: str  # e.g. cancel_t1
    category: str  # cancel | status | kv | balance | elastic | engine_fault | master | admission | direct
    fn: Callable  # ctx -> (passed, detail)
    profiles: Optional[List[str]] = None  # None = all profiles apply
    requires: Optional[List[str]] = None  # semantic capability requirements
    source: str = ""  # legacy script this was ported from


class CaseContext:
    """Per-run context handed to every case function."""

    # Monotonic counter across the whole process: every case invocation gets a
    # fresh offset so request ids never collide with the master's dedup table
    # when the environment (and its in-memory dedup state) is reused.
    _case_seq = 0

    def __init__(
        self,
        env_manager: EnvManager,
        profile: str,
        run_root: Path,
        log_fn: Optional[Callable[[str], None]] = None,
        grade: str = "normal",
    ):
        CaseContext._case_seq += 1
        self.case_seq = CaseContext._case_seq
        self.env_manager = env_manager
        self.profile = profile  # scheduling profile (harness.PROFILES)
        # Assertion grade for this run (grade.GRADES): strict/normal/loose —
        # graded property cases build their GradeReport from it.
        self.grade = grade
        self.run_root = run_root
        self._log_fn = log_fn or (lambda msg: None)
        self._ops_cache: dict = {}
        self._current_smoke_ops: Optional[EngineOps] = None

    def log(self, msg: str) -> None:
        self._log_fn(msg)

    # -- profile helpers ----------------------------------------------------

    def batch_dispatch(self) -> bool:
        """True when the current profile dispatches via BATCH (master sends
        EnqueueBatch, clients consume FetchResponse); False for NON_BATCH
        (frontend sends GenerateStreamCall).  The client path itself is
        decided per response by ``enqueued_by_master``."""
        return profile_dispatches_batch(self.profile)

    # -- environments ------------------------------------------------------

    def smoke_spec(self) -> EnvSpec:
        """Shared default environment: 2P + 4D, standard perf, master in
        ctx.profile (legacy name kept: the runner API predates the
        category reorg)."""
        return EnvSpec(
            label=f"smoke_{self.profile}",
            n_prefill=2,
            n_decode=4,
            perf=default_perf(),
            master_profile=self.profile,
        )

    def engine_ops(self, env) -> EngineOps:
        """EngineOps bound to a live env (cached per env instance)."""
        key = id(env)
        if key not in self._ops_cache:
            self._ops_cache[key] = EngineOps(
                "127.0.0.1",
                env.master_http_port,
                env.mock_http_port,
            )
        return self._ops_cache[key]

    def ops(self) -> EngineOps:
        """Ensure the shared smoke env and return its EngineOps."""
        env = self.env_manager.ensure(self.smoke_spec())
        return self.engine_ops(env)

    # -- helpers ------------------------------------------------------------

    def case_dir(self, name: str) -> Path:
        d = self.run_root / "cases" / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def close(self) -> None:
        for ops in self._ops_cache.values():
            ops.close()
        self._ops_cache.clear()


# request-id bases per case category × scheduling profile.  All bases live
# in a single < 1M window (RID_BASES + case_seq * 1M can never collide
# across (category, profile) pairs because every pairwise base distance is
# < 1M), so a reused master's dedup table stays collision-free across
# profiles and reruns.  The nine categories map 1:1 onto the nine
# flexlb_ft/cases/ modules (task #85 reorg); the pre-reorg families
# (scheduling / anomaly / chaos) were dissolved into status, balance,
# kv, engine_fault, master and direct.
RID_BASES = {
    "cancel": {
        "batch-window": 100_000,
        "single-nonbatch": 125_000,
        "single-batch": 150_000,
        "window-nonbatch": 175_000,
    },
    "status": {
        "batch-window": 200_000,
        "single-nonbatch": 225_000,
        "single-batch": 250_000,
        "window-nonbatch": 275_000,
    },
    "balance": {
        "batch-window": 300_000,
        "single-nonbatch": 325_000,
        "single-batch": 350_000,
        "window-nonbatch": 375_000,
    },
    "direct": {
        "batch-window": 400_000,
        "single-nonbatch": 425_000,
        "single-batch": 450_000,
        "window-nonbatch": 475_000,
    },
    "elastic": {
        "batch-window": 500_000,
        "single-nonbatch": 525_000,
        "single-batch": 550_000,
        "window-nonbatch": 575_000,
    },
    "kv": {
        "batch-window": 600_000,
        "single-nonbatch": 625_000,
        "single-batch": 650_000,
        "window-nonbatch": 675_000,
    },
    "engine_fault": {
        "batch-window": 700_000,
        "single-nonbatch": 725_000,
        "single-batch": 750_000,
        "window-nonbatch": 775_000,
    },
    "master": {
        "batch-window": 800_000,
        "single-nonbatch": 825_000,
        "single-batch": 850_000,
        "window-nonbatch": 875_000,
    },
    "admission": {
        "batch-window": 900_000,
        "single-nonbatch": 925_000,
        "single-batch": 950_000,
        "window-nonbatch": 975_000,
    },
}


def rid_base(ctx: CaseContext, family: str) -> int:
    # ctx.case_seq makes each invocation of a case use a distinct id range
    # (RID_BASES cover the profile axis, case_seq lifts per re-run).
    # pid offset: sibling framework processes share the same case_seq
    # sequence, so two agents hammering the same (reused) master would
    # generate colliding ids — the pid term keeps id spaces disjoint.
    pid_offset = (os.getpid() % 100) * 100_000_000
    return RID_BASES[family][ctx.profile] + ctx.case_seq * 1_000_000 + pid_offset
