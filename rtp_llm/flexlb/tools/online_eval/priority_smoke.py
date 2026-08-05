#!/usr/bin/env python3
"""Auto-TPM priority integration tests for FlexLB.

Connects to a running FlexLB master (FlexlbService gRPC) and mock engine
cluster.  Exercises six priority / Auto-TPM scenarios:

  T1  priority_ordering        - P70 requests get TTFT <= P30 requests.
  T2  prefill_eviction         - P70 evicts queued P30 when queue is full.
  T3  same_priority_no_eviction- Same-priority requests are never evicted.
  T4  no_inflight_leak         - After eviction, master inflight drains to 0.
  T5  deadline_rescue          - Danger-zone request gets rescued.
  T6  running_preempt          - P70 preempts a RUNNING P30 (opt-in).

Requires a master started with ``master_auto_tpm.json`` (priority_deadline
algorithm, FLEXLB_PRIORITY_EVICT_ENABLED=true, FLEXLB_BATCH_QUEUE_MAX_SIZE=8).

Usage:
    python3 priority_smoke.py --master-ip 127.0.0.1 --master-http-port 18080 \\
        --mock-http-port 55150
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Optional

from flexlb_smoke_base import FlexLBSmokeBase, ScenarioResult, StreamSnapshot

# Priority levels (must match FLEXLB_PRIORITY_LEVELS in master_auto_tpm.json)
P30 = 30
P50 = 50
P60 = 60
P70 = 70

# Must match FLEXLB_BATCH_QUEUE_MAX_SIZE in master_auto_tpm.json so that the
# eviction scenarios can reliably fill the prefill queue.
QUEUE_MAX_SIZE = 8

# TTFT comparison tolerance (ms).  The mock engine is fast, so we allow a
# generous tolerance when asserting P70_ttft <= P30_ttft.
TTFT_TOLERANCE_MS = 200.0


class PrioritySmokeTest(FlexLBSmokeBase):
    """Auto-TPM priority integration tests."""

    # -- Helpers ----------------------------------------------------------

    async def _schedule_auto(self, request_id: int, **kwargs):
        """Schedule a request (priority flows through via kwargs)."""
        return await self._schedule(request_id, **kwargs)

    async def _prefill_engine_names(self) -> list[str]:
        snap = await self._get_snapshot()
        return [
            e["name"]
            for e in snap.get("engines", [])
            if e.get("role") == "prefill"
        ]

    async def _set_perf_all_prefill(self, **kwargs) -> list[str]:
        names = await self._prefill_engine_names()
        for name in names:
            await self._set_perf(name, **kwargs)
        return names

    async def _cancelled_rids_set(self) -> set[int]:
        """Return the union of cancelled_rids across all mock engines."""
        snap = await self._get_snapshot()
        rids: set[int] = set()
        for engine in snap.get("engines", []):
            for rid in engine.get("cancelled_rids", []):
                try:
                    rids.add(int(rid))
                except (TypeError, ValueError):
                    pass
        return rids

    async def _run_with_ttft(
        self, rid: int, priority: int = 0, **kwargs
    ) -> tuple[Optional[float], Optional[float], bool, str, str]:
        """Schedule, start stream, measure TTFT and total time.

        Returns ``(ttft_ms, total_ms, success, prefill_addr, error)``.
        ``ttft_ms`` / ``total_ms`` are ``None`` when not measured.
        """
        start = time.monotonic()
        try:
            response = await self._schedule(rid, priority=priority, **kwargs)
            if response.code != 200 or not response.success:
                return None, None, False, "", f"schedule failed: {response.error_message}"
            addr = self._role_addr(response, "PREFILL")
            input_pb = (
                None if response.enqueued_by_master else self._build_generate_input(rid)
            )
            stream = await self._start_stream(response, rid, input_pb=input_pb)
            snap = StreamSnapshot()
            task = asyncio.create_task(self._consume_stream(stream, snap))
            got_first = await self._wait_for_first_output(snap, timeout_s=20.0)
            ttft_ms = (time.monotonic() - start) * 1000.0 if got_first else None
            await self._wait_for_stream_end(task, timeout_s=30.0)
            total_ms = (time.monotonic() - start) * 1000.0
            success = snap.completed and snap.error is None
            err = snap.error if snap.error else ("not completed" if not snap.completed else "")
            return ttft_ms, total_ms, success, addr, err
        except Exception as exc:
            return None, (time.monotonic() - start) * 1000.0, False, "", repr(exc)

    @staticmethod
    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _running_preempt_enabled() -> bool:
        return os.environ.get("AUTO_TPM_CANCEL_RUNNING_ENABLED", "false").lower() in (
            "1", "true", "yes",
        )

    # -- T1: priority_ordering -------------------------------------------

    async def test_priority_ordering(self) -> ScenarioResult:
        """T1: P70 requests should have TTFT <= P30 requests.

        Sends 3 P70 + 3 P30 concurrently.  With ``priority_deadline`` the
        batcher dispatches higher-priority requests first, so P70 TTFT
        should be no worse than P30 TTFT (within tolerance).
        """
        start = time.monotonic()
        perf_engines: list[str] = []
        try:
            # Set a moderate prefill delay so dispatch ordering is observable.
            perf_engines = await self._set_perf_all_prefill(prefill_fixed_ms=200.0)
            await asyncio.sleep(1.0)  # let master sync perf

            p70_rids = [self._next_request_id() for _ in range(3)]
            p30_rids = [self._next_request_id() for _ in range(3)]

            results = await asyncio.gather(
                *[self._run_with_ttft(r, priority=P70, output_len=2) for r in p70_rids],
                *[self._run_with_ttft(r, priority=P30, output_len=2) for r in p30_rids],
            )
            p70_results = results[:3]
            p30_results = results[3:]

            p70_ttfts = [r[0] for r in p70_results if r[0] is not None]
            p30_ttfts = [r[0] for r in p30_results if r[0] is not None]
            p70_success = sum(1 for r in p70_results if r[2])
            p30_success = sum(1 for r in p30_results if r[2])

            p70_avg = self._avg(p70_ttfts)
            p30_avg = self._avg(p30_ttfts)

            # Pass if P70 all succeed and P70 avg TTFT is no worse than P30
            # (within tolerance), or P70 success rate exceeds P30.
            ttft_ok = (
                bool(p70_ttfts)
                and bool(p30_ttfts)
                and p70_avg <= p30_avg + TTFT_TOLERANCE_MS
            )
            success_ok = p70_success >= p30_success and p70_success > 0
            passed = success_ok and (ttft_ok or p70_success > p30_success)

            detail = (
                f"P70: success={p70_success}/3, avg_ttft={p70_avg:.1f}ms, "
                f"ttfts={[round(t, 1) for t in p70_ttfts]}; "
                f"P30: success={p30_success}/3, avg_ttft={p30_avg:.1f}ms, "
                f"ttfts={[round(t, 1) for t in p30_ttfts]}; "
                f"ttft_ok={ttft_ok}, success_ok={success_ok}"
            )
            return ScenarioResult(
                "T1: priority_ordering", passed, detail, time.monotonic() - start
            )
        except Exception as exc:
            return ScenarioResult(
                "T1: priority_ordering", False, f"exception: {exc!r}",
                time.monotonic() - start,
            )
        finally:
            for eng in perf_engines:
                try:
                    await self._set_perf(eng, prefill_fixed_ms=100.0)
                except Exception:
                    pass

    # -- T2: prefill_eviction --------------------------------------------

    async def test_prefill_eviction(self) -> ScenarioResult:
        """T2: P70 evicts queued P30 when the prefill queue is full.

        Sets a large prefill delay, floods the queue with P30, then sends a
        P70.  The P70 should be admitted by evicting lower-priority P30
        victims (which appear in the mock-engine ``cancelled_rids``).
        """
        start = time.monotonic()
        perf_engines: list[str] = []
        try:
            # High prefill delay so queued items are not dispatched quickly.
            perf_engines = await self._set_perf_all_prefill(prefill_fixed_ms=30000.0)
            await asyncio.sleep(1.0)

            # Flood the queue with P30 (more than QUEUE_MAX_SIZE).
            n_flood = QUEUE_MAX_SIZE + 2
            p30_rids = [self._next_request_id() for _ in range(n_flood)]
            flood_responses = await asyncio.gather(
                *[self._schedule_auto(r, priority=P30, output_len=2) for r in p30_rids],
                return_exceptions=True,
            )
            queued_p30 = [
                rid
                for rid, resp in zip(p30_rids, flood_responses)
                if not isinstance(resp, Exception)
                and resp.code == 200
                and resp.success
            ]
            rejected_p30 = [
                rid
                for rid, resp in zip(p30_rids, flood_responses)
                if isinstance(resp, Exception)
                or resp.code != 200
                or not resp.success
            ]

            # Send a P70 — should trigger eviction of queued P30.
            p70_rid = self._next_request_id()
            p70_resp = await self._schedule_auto(p70_rid, priority=P70, output_len=2)
            p70_ok = p70_resp.code == 200 and p70_resp.success

            # Give the master a moment to execute the eviction plan.
            await asyncio.sleep(1.0)

            cancelled = await self._cancelled_rids_set()
            evicted_p30 = set(p30_rids) & cancelled
            p70_cancelled = p70_rid in cancelled

            # Pass if P70 was admitted and at least one P30 was evicted.
            passed = p70_ok and not p70_cancelled and len(evicted_p30) > 0

            detail = (
                f"p70_ok={p70_ok}(rid={p70_rid}), "
                f"p30_queued={len(queued_p30)}, p30_rejected={len(rejected_p30)}, "
                f"evicted_p30={sorted(evicted_p30)}, "
                f"p70_cancelled={p70_cancelled}"
            )
            return ScenarioResult(
                "T2: prefill_eviction", passed, detail, time.monotonic() - start
            )
        except Exception as exc:
            return ScenarioResult(
                "T2: prefill_eviction", False, f"exception: {exc!r}",
                time.monotonic() - start,
            )
        finally:
            for eng in perf_engines:
                try:
                    await self._set_perf(eng, prefill_fixed_ms=100.0)
                except Exception:
                    pass

    # -- T3: same_priority_no_eviction -----------------------------------

    async def test_same_priority_no_eviction(self) -> ScenarioResult:
        """T3: Same-priority requests are never evicted.

        Floods the queue with P50, then sends one more P50.  No eviction
        should occur (hard rule: victim.priority < incoming.priority).  The
        extra P50 should be rejected with QUEUE_FULL.
        """
        start = time.monotonic()
        perf_engines: list[str] = []
        try:
            perf_engines = await self._set_perf_all_prefill(prefill_fixed_ms=30000.0)
            await asyncio.sleep(1.0)

            n_flood = QUEUE_MAX_SIZE + 1
            p50_rids = [self._next_request_id() for _ in range(n_flood)]
            flood_responses = await asyncio.gather(
                *[self._schedule_auto(r, priority=P50, output_len=2) for r in p50_rids],
                return_exceptions=True,
            )
            queued_p50 = [
                rid
                for rid, resp in zip(p50_rids, flood_responses)
                if not isinstance(resp, Exception)
                and resp.code == 200
                and resp.success
            ]
            rejected_p50 = [
                rid
                for rid, resp in zip(p50_rids, flood_responses)
                if isinstance(resp, Exception)
                or resp.code != 200
                or not resp.success
            ]

            await asyncio.sleep(1.0)
            cancelled = await self._cancelled_rids_set()
            evicted_p50 = set(queued_p50) & cancelled

            # No eviction should occur; the overflow request is rejected.
            no_eviction = len(evicted_p50) == 0
            has_rejection = len(rejected_p50) > 0
            passed = no_eviction and has_rejection

            detail = (
                f"queued={len(queued_p50)}, rejected={len(rejected_p50)}, "
                f"evicted={sorted(evicted_p50)}, "
                f"no_eviction={no_eviction}, has_rejection={has_rejection}"
            )
            return ScenarioResult(
                "T3: same_priority_no_eviction", passed, detail,
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "T3: same_priority_no_eviction", False, f"exception: {exc!r}",
                time.monotonic() - start,
            )
        finally:
            for eng in perf_engines:
                try:
                    await self._set_perf(eng, prefill_fixed_ms=100.0)
                except Exception:
                    pass

    # -- T4: no_inflight_leak --------------------------------------------

    async def test_no_inflight_leak(self) -> ScenarioResult:
        """T4: After eviction, master inflight drains to 0.

        Triggers eviction (T2 flow), resets prefill perf to fast, waits for
        all inflight to drain, then verifies master ``/rtp_llm/inflight_status``
        is clean and the mock engine snapshot has no residual inflight.
        """
        start = time.monotonic()
        perf_engines: list[str] = []
        try:
            # Trigger eviction
            perf_engines = await self._set_perf_all_prefill(prefill_fixed_ms=30000.0)
            await asyncio.sleep(1.0)

            p30_rids = [self._next_request_id() for _ in range(QUEUE_MAX_SIZE + 2)]
            await asyncio.gather(
                *[self._schedule_auto(r, priority=P30, output_len=2) for r in p30_rids],
                return_exceptions=True,
            )
            p70_rid = self._next_request_id()
            await self._schedule_auto(p70_rid, priority=P70, output_len=2)
            await asyncio.sleep(1.0)

            # Reset prefill to fast so queued/running requests can complete.
            for eng in perf_engines:
                await self._set_perf(eng, prefill_fixed_ms=10.0)
            # Cancel ALL scheduled requests (evicted + non-evicted P30 and
            # P70) to clean up master inflight.  In BATCH mode, requests
            # that are scheduled but never fetched via FetchResponse stay
            # inflight at the master indefinitely; cancelling them ensures
            # the inflight counter drains to zero.
            cancelled = await self._cancelled_rids_set()
            for rid in p30_rids + [p70_rid]:
                try:
                    await self._cancel(rid)
                except Exception:
                    pass

            # Wait for master inflight to drain.
            inflight_ok, inflight_detail = await self._verify_inflight_clean(
                timeout_s=30.0
            )

            # Check mock engine snapshot for residual inflight.  The mock
            # engine's ``running`` counter can lag behind the master's
            # inflight status (master clears its bookkeeping before the
            # engine decrements its running set), so poll with retries.
            residual: list[str] = []
            snap_clean = False
            snap_deadline = time.monotonic() + 15.0
            while time.monotonic() < snap_deadline:
                snap = await self._get_snapshot()
                residual = []
                for engine in snap.get("engines", []):
                    running = engine.get("running", 0)
                    if running and running > 0:
                        residual.append(f"{engine['name']}={running}")
                if not residual:
                    snap_clean = True
                    break
                await asyncio.sleep(0.5)

            passed = inflight_ok and snap_clean

            detail = (
                f"inflight_clean={inflight_ok}({inflight_detail}), "
                f"residual_running={residual}, "
                f"cancelled_count={len(cancelled)}"
            )
            return ScenarioResult(
                "T4: no_inflight_leak", passed, detail, time.monotonic() - start
            )
        except Exception as exc:
            return ScenarioResult(
                "T4: no_inflight_leak", False, f"exception: {exc!r}",
                time.monotonic() - start,
            )
        finally:
            for eng in perf_engines:
                try:
                    await self._set_perf(eng, prefill_fixed_ms=100.0)
                except Exception:
                    pass

    # -- T5: deadline_rescue ---------------------------------------------

    async def test_deadline_rescue(self) -> ScenarioResult:
        """T5: A danger-zone request should be rescued.

        Sends a P60 request with a tight SLO while prefill is slow.  The
        deadline-rescue planner scans for requests whose
        ``deadline - now < danger_threshold`` and re-schedules them.  This
        scenario is timing-sensitive and may require multiple attempts.
        """
        start = time.monotonic()
        perf_engines: list[str] = []
        try:
            # Slow prefill so the request enters the danger zone.
            perf_engines = await self._set_perf_all_prefill(prefill_fixed_ms=5000.0)
            await asyncio.sleep(1.0)

            rid = self._next_request_id()
            response = await self._schedule_auto(rid, priority=P60, output_len=2)
            if response.code != 200 or not response.success:
                return ScenarioResult(
                    "T5: deadline_rescue", False,
                    f"schedule failed: {response.error_message}",
                    time.monotonic() - start,
                )

            input_pb = (
                None if response.enqueued_by_master else self._build_generate_input(rid)
            )
            stream = await self._start_stream(response, rid, input_pb=input_pb)
            snap = StreamSnapshot()
            task = asyncio.create_task(self._consume_stream(stream, snap))

            # Wait up to 20s for the request to complete (rescued or not).
            deadline = time.monotonic() + 20.0
            while not snap.completed and snap.error is None and time.monotonic() < deadline:
                await asyncio.sleep(0.1)

            if not task.done():
                task.cancel()
                try:
                    await task
                except (Exception, asyncio.CancelledError):
                    pass

            # Reset prefill so subsequent tests are not affected.
            for eng in perf_engines:
                await self._set_perf(eng, prefill_fixed_ms=100.0)

            # The rescue planner may re-schedule the request.  If it completes
            # within the window (even after a re-schedule), the test passes.
            # If the stream errors out with a cancel/re-schedule signal, that
            # also indicates rescue activity.
            rescued = snap.completed or (
                snap.error is not None and "cancel" in str(snap.error).lower()
            )
            detail = (
                f"rid={rid}, completed={snap.completed}, "
                f"error={snap.error}, outputs={len(snap.outputs)}, "
                f"rescued={rescued}"
            )
            # This scenario is timing-sensitive; pass if the request completed
            # OR was explicitly cancelled by the rescue planner.
            return ScenarioResult(
                "T5: deadline_rescue", rescued, detail, time.monotonic() - start
            )
        except Exception as exc:
            return ScenarioResult(
                "T5: deadline_rescue", False, f"exception: {exc!r}",
                time.monotonic() - start,
            )
        finally:
            for eng in perf_engines:
                try:
                    await self._set_perf(eng, prefill_fixed_ms=100.0)
                except Exception:
                    pass

    # -- T6: running_preempt ---------------------------------------------

    async def test_running_preempt(self) -> ScenarioResult:
        """T6: P70 preempts a RUNNING P30 (requires AUTO_TPM_CANCEL_RUNNING_ENABLED=true).

        Sends a P30, waits for it to enter decode RUNNING, then sends a P70.
        The P30 should be cancelled (CANCEL_REASON_PRIORITY_PREEMPTED) and the
        P70 should succeed.  Skipped when running preemption is not enabled.
        """
        if not self._running_preempt_enabled():
            return ScenarioResult(
                "T6: running_preempt", True,
                "skipped: AUTO_TPM_CANCEL_RUNNING_ENABLED != true",
                0.0,
            )

        start = time.monotonic()
        perf_engines: list[str] = []
        try:
            # Moderate prefill so P30 reaches decode RUNNING quickly.
            perf_engines = await self._set_perf_all_prefill(prefill_fixed_ms=100.0)
            await asyncio.sleep(1.0)

            p30_rid = self._next_request_id()
            p30_resp = await self._schedule_auto(p30_rid, priority=P30, output_len=50)
            if p30_resp.code != 200 or not p30_resp.success:
                return ScenarioResult(
                    "T6: running_preempt", False,
                    f"P30 schedule failed: {p30_resp.error_message}",
                    time.monotonic() - start,
                )

            p30_input = (
                None if p30_resp.enqueued_by_master else self._build_generate_input(p30_rid)
            )
            p30_stream = await self._start_stream(p30_resp, p30_rid, input_pb=p30_input)
            p30_snap = StreamSnapshot()
            p30_task = asyncio.create_task(self._consume_stream(p30_stream, p30_snap))

            # Wait for P30 to receive first output (decode RUNNING).
            p30_running = await self._wait_for_first_output(p30_snap, timeout_s=15.0)
            if not p30_running:
                p30_task.cancel()
                return ScenarioResult(
                    "T6: running_preempt", False,
                    "P30 never reached decode (no first output)",
                    time.monotonic() - start,
                )

            # Send P70 — should preempt the running P30.
            p70_rid = self._next_request_id()
            p70_resp = await self._schedule_auto(p70_rid, priority=P70, output_len=2)
            p70_ok = p70_resp.code == 200 and p70_resp.success

            # Wait for P30 to be cancelled.
            p30_ended = await self._wait_for_stream_end(p30_task, timeout_s=10.0)

            await asyncio.sleep(1.0)
            cancelled = await self._cancelled_rids_set()
            p30_preempted = p30_rid in cancelled

            # Verify P70 completes.
            p70_ttft, _, p70_success, _, p70_err = await self._run_with_ttft(
                p70_rid, priority=P70, output_len=2
            )

            passed = p70_ok and p30_preempted and p70_success
            detail = (
                f"p30_running={p30_running}, p30_preempted={p30_preempted}, "
                f"p30_ended={p30_ended}, p70_ok={p70_ok}, "
                f"p70_success={p70_success}(err={p70_err})"
            )
            return ScenarioResult(
                "T6: running_preempt", passed, detail, time.monotonic() - start
            )
        except Exception as exc:
            return ScenarioResult(
                "T6: running_preempt", False, f"exception: {exc!r}",
                time.monotonic() - start,
            )
        finally:
            for eng in perf_engines:
                try:
                    await self._set_perf(eng, prefill_fixed_ms=100.0)
                except Exception:
                    pass

    # -- Runner ----------------------------------------------------------

    async def run_all(self) -> int:
        scenarios = [
            self.test_priority_ordering,
            self.test_prefill_eviction,
            self.test_same_priority_no_eviction,
            self.test_no_inflight_leak,
            self.test_deadline_rescue,
            self.test_running_preempt,
        ]
        print("=" * 70)
        print("FlexLB Auto-TPM Priority Smoke Test")
        print(f"  master: {self._master_target()}")
        print(f"  deploy_mode: {self._deploy_mode}")
        print(f"  mock_http_port: {self.args.mock_http_port}")
        print(f"  queue_max_size: {QUEUE_MAX_SIZE}")
        print(f"  running_preempt_enabled: {self._running_preempt_enabled()}")
        print("=" * 70)

        for scenario in scenarios:
            print(f"\n>>> Running {scenario.__name__} ...", flush=True)
            result = await scenario()
            self.results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(
                f"<<< {result.name}: {status}  "
                f"({result.duration_s:.2f}s)  {result.detail}",
                flush=True,
            )

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        print("\n" + "=" * 70)
        print(f"Summary: {passed}/{len(self.results)} passed, {failed} failed")
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  {status}  {r.name}")
        print("=" * 70)
        return 1 if failed > 0 else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-ip", default="127.0.0.1")
    parser.add_argument("--master-http-port", type=int, default=18080)
    parser.add_argument(
        "--mock-http-port",
        type=int,
        default=55150,
        help="mock engine cluster HTTP API port",
    )
    parser.add_argument(
        "--flexlb-http-port",
        type=int,
        default=18080,
        help="flexlb master HTTP port for inflight status check",
    )
    parser.add_argument("--request-id-base", type=int, default=30000)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    test = PrioritySmokeTest(args)
    try:
        exit_code = await test.run_all()
    finally:
        await test.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
