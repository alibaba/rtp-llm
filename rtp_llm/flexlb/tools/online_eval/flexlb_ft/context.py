"""Case registry + execution context shared by all suites."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .engine_ops import EngineOps
from .harness import EnvManager, EnvSpec, default_perf

SMOKE_LABEL_PERF = default_perf()


@dataclass
class CaseDef:
    """One test case: name, suite, callable and optional mode restriction."""

    name: str  # e.g. smoke_cancel_t1
    suite: str  # smoke | chaos
    fn: Callable  # ctx -> (passed, detail)
    modes: Optional[List[str]] = None  # None = all modes apply
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
        mode: str,
        run_root: Path,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        CaseContext._case_seq += 1
        self.case_seq = CaseContext._case_seq
        self.env_manager = env_manager
        self.mode = mode  # batch | direct | queue
        self.run_root = run_root
        self._log_fn = log_fn or (lambda msg: None)
        self._ops_cache: dict = {}
        self._current_smoke_ops: Optional[EngineOps] = None

    def log(self, msg: str) -> None:
        self._log_fn(msg)

    # -- environments ------------------------------------------------------

    def smoke_spec(self) -> EnvSpec:
        """Shared smoke environment: 2P + 4D, standard perf, master in ctx.mode."""
        return EnvSpec(
            label=f"smoke_{self.mode}",
            n_prefill=2,
            n_decode=4,
            perf=default_perf(),
            master_mode=self.mode,
        )

    def engine_ops(self, env) -> EngineOps:
        """EngineOps bound to a live env (cached per env instance)."""
        key = id(env)
        if key not in self._ops_cache:
            self._ops_cache[key] = EngineOps(
                "127.0.0.1",
                env.master_http_port,
                env.mock_http_port,
                deploy_mode=self.mode,
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


# request-id bases per legacy run_matrix_smoke.sh group config
RID_BASES = {
    "cancel": {"batch": 10000, "direct": 40000, "queue": 70000},
    "scheduling": {"batch": 20000, "direct": 50000, "queue": 80000},
    "anomaly": {"batch": 30000, "direct": 60000, "queue": 90000},
    "chaos": {"batch": 110000, "direct": 120000, "queue": 130000},
}


def rid_base(ctx: CaseContext, family: str) -> int:
    # ctx.case_seq makes each invocation of a case use a distinct id range
    # (RID_BASES cover batch/direct/queue, case_seq lifts per re-run).
    # pid offset: sibling framework processes share the same case_seq
    # sequence, so two agents hammering the same (reused) master would
    # generate colliding ids — the pid term keeps id spaces disjoint.
    pid_offset = (os.getpid() % 100) * 100_000_000
    return RID_BASES[family][ctx.mode] + ctx.case_seq * 1_000_000 + pid_offset
