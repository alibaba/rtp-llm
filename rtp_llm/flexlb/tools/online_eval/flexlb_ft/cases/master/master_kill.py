from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...harness import AssertUtils, _cleanup_dynamic, _elastic_env, wait_for
from ...registry import case
from ...support.ha import ha_dual_enabled
from ...support.master import _master_http, _master_kill_dual


@case(
    "master_kill",
    category="master",
    profiles=["batch-window"],  # _elastic_env pins the legacy fault axes
    source="master HA: kill -9 master → restart → clean state + recovery",
)
def master_kill(ctx: CaseContext):
    # HA generalized branch (brief p3: "master_kill 用例泛化双 master 定向",
    # p4 left lane: start sticky-on-B, kill -9 B) — gated on
    # FLEXLB_FT_HA_DUAL_MASTER=1 so the default run keeps the historical
    # single-master flow byte-identical (compat hard constraint).
    if ha_dual_enabled():
        return _master_kill_dual(ctx)
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "master")
    try:
        _cleanup_dynamic(ops, env)

        # Baseline: one request succeeds before the kill.
        addr, err0 = ops.run_one_request(
            ops.next_request_id(base),
            output_len=2,
            block_keys=[base + 11],
            stream_timeout_s=10.0,
        )
        del addr
        if err0:
            return False, f"baseline request failed: {err0}"

        # kill -9 the master, restart it from the same argv/env.
        ctx.env_manager.kill_master9(env)
        time.sleep(2.0)  # settle; port release
        ctx.env_manager.start_master(env)

        # Wait for the full topology to re-converge (ready + alive workers).
        alive_ok = wait_for(
            lambda: ops.master_alive_count("PREFILL") >= 2
            and ops.master_alive_count("DECODE") >= 4,
            60.0,
            1.0,
        )
        # Fresh master must start from clean inflight state.
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 10.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = alive_ok and inflight_ok and recovery_ok
        return passed, (
            f"master_restarted, topology_reconverged={alive_ok}"
            f"(alive P:{ops.master_alive_count('PREFILL')}/"
            f"D:{ops.master_alive_count('DECODE')}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        # If the case died mid-way with the master down, bring it back so the
        # shared env stays usable (and teardown finds a ManagedProcess).
        try:
            if env.master is None:
                ctx.env_manager.start_master(env)
        except Exception:
            pass
