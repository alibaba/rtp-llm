#!/usr/bin/env python3
"""Extreme-scenario smoke tests for FlexLB.

Connects to a running FlexLB master and mock engine cluster. Exercises five
extreme fault/pressure scenarios that the existing anomaly suite (E1-E3) does not
cover:

  SM-C1  all_prefill_dead_recovery
         Stop every prefill engine gRPC server -> new requests are rejected
         -> restart all prefill -> requests succeed again and inflight is
         clean. Verifies full-EP-death fail-closed behaviour and recovery.

  SM-C2  selective_omit_stale_evict
         Schedule a long-decode request, then inject ``omit_request_ids``
         on the decode engine holding it. The engine stays alive (gRPC
         reachable, status_version still bumps) but never reports that
         requestId. After ``STALE_EVICT_ROUNDS`` (3) consecutive calibrate
         rounds FlexLB evicts the bound engineTask as stale and — via the
         A3 fix — drives the bound InflightItem to a terminal state, so
         inflight drops to zero without waiting for the 60s TTL backstop.

  SM-C3  kv_exhaustion_rejection
         Set ``active_kv_tokens = total_kv`` on every decode engine so
         ``available_kv = 0 < seqLen``. CostBasedDecodeStrategy filters
         all decode EPs with KV_CAPACITY -> schedule returns
         NO_AVAILABLE_WORKER (8400). Restore KV -> verify recovery and
         no inflight leak.

  SM-C4  dispatch_pool_saturation
         Set ``prefill_fixed_ms=10000`` + ``max_prefill_concurrency=1`` on
         all prefill engines. The gRPC EnqueueBatch deadline (5 s) fires
         before the engine finishes -> items ``failTimeout``. With a small
         dispatch pool (4 threads / 2 queue, set via env vars in
         run_extreme_smoke.sh) concurrent dispatches may also trigger
         RejectedExecutionException -> ``failDispatch``. Verify all items
         settle (no leak), then restore perf and recover.

  SM-C5  cold_start_burst
         Send 20+ concurrent requests immediately after the master starts
         (first scenario, no warm-up). Some requests may be rejected while
         EPs are not yet synced; verify all settle and inflight converges
         to zero.

Usage:
    python3 extreme_smoke.py --master-ip 127.0.0.1 \\
        --master-http-port 18080 --mock-http-port 55150 \\
        --flexlb-http-port 18080 --flexlb-log /path/to/flexlb.log
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from flexlb_smoke_base import FlexLBSmokeBase, ScenarioResult, StreamSnapshot


class ExtremeSmokeTest(FlexLBSmokeBase):
    """Extreme fault-injection smoke tests."""

    # FlexLB defaults (see FlexlbConfig.java / PrefillEndpoint / DecodeEndpoint)
    #   syncStatusInterval = 20ms  -> one calibrate round every ~20ms
    #   STALE_EVICT_ROUNDS = 3     -> eviction after 3 unseen rounds
    # 8s covers hundreds of sync rounds — ample margin for STALE eviction.
    STALE_WAIT_S = 8.0
    # Time to let FlexLB detect engines going down / coming back up.
    ENGINE_DOWN_DETECT_S = 5.0
    ENGINE_UP_DETECT_S = 5.0
    # Long decode so the request stays in decode-running while we inject omit.
    # At 20ms/step (batch=1) this is ~100s of decode — plenty of headroom.
    LONG_OUTPUT_LEN = 5000
    # How long to wait for a request to show up in decode engine_tasks.
    DECODE_ENTRY_TIMEOUT_S = 15.0
    # Stream timeout when verifying requests are rejected (SM-C1).
    REJECT_STREAM_TIMEOUT_S = 8.0
    # Time to wait for FlexLB sync to pick up KV pressure changes (SM-C3).
    KV_PRESSURE_SYNC_S = 3.0
    # Per-request timeout for dispatch saturation and cold-start burst tests.
    DISPATCH_REQUEST_TIMEOUT_S = 30.0
    # Number of concurrent requests for burst tests (SM-C4, SM-C5).
    BURST_CONCURRENT = 20

    # -- HTTP mock-engine helpers (stop / start engine) ------------------

    async def _stop_engine(self, name: str) -> dict:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{self.args.mock_http_port}/stop_engine",
                json={"engine": name},
            ) as resp:
                return await resp.json()

    async def _start_engine(self, name: str) -> dict:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{self.args.mock_http_port}/start_engine",
                json={"engine": name},
            ) as resp:
                return await resp.json()

    async def _engine_names(self, role: str) -> list[str]:
        snap = await self._get_snapshot()
        return [e["name"] for e in snap["engines"] if e["role"] == role]

    async def _get_inflight_status(self) -> dict:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://127.0.0.1:{self.args.flexlb_http_port}/rtp_llm/inflight_status"
            ) as resp:
                return await resp.json()

    async def _find_decode_engine_for_rid(
        self, rid: int
    ) -> tuple[str | None, str | None]:
        """Return (engine_name, grpc_addr) of the decode engine holding rid."""
        snap = await self._get_snapshot()
        for e in snap["engines"]:
            if e["role"] != "decode":
                continue
            lc = e.get("request_lifecycle", {})
            if str(rid) in lc:
                return e["name"], e.get("grpc_addr")
        return None, None

    async def _decode_engine_tasks_total(self) -> int:
        st = await self._get_inflight_status()
        return sum(ep.get("engine_tasks", 0) for ep in st.get("decode_endpoints", []))

    async def _cancel_on_engine(self, grpc_addr: str, rid: int) -> None:
        """Call Worker RpcService.Cancel directly on a specific engine."""
        if not grpc_addr:
            return
        stub = self.pb2_grpc.RpcServiceStub(await self._channel(grpc_addr))
        await stub.Cancel(self.pb2.CancelRequestPB(request_id=rid), timeout=10.0)

    def _grep_log(self, *needles: str) -> bool:
        """Return True if any log line contains all needle substrings.

        If ``--flexlb-log`` is a directory, search every ``*.log`` file in
        it (logback writes syncLogger → sync.log, flexlbLogger → flexlb.log,
        root → application.log, plus stdout → master_stdout.log).  If it's
        a single file, search that file and also check ``sync.log`` /
        ``application.log`` in the same directory.
        """
        path = getattr(self.args, "flexlb_log", None)
        if not path:
            return False
        candidates: list[str] = []
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith(".log"):
                    candidates.append(os.path.join(path, f))
        elif os.path.isfile(path):
            candidates.append(path)
            d = os.path.dirname(path)
            for extra in ("sync.log", "application.log"):
                p = os.path.join(d, extra)
                if p not in candidates and os.path.exists(p):
                    candidates.append(p)
        for c in candidates:
            try:
                with open(c, "r", errors="replace") as fh:
                    for line in fh:
                        if all(n in line for n in needles):
                            return True
            except Exception:
                continue
        return False

    async def _send_and_complete(
        self, rid: int, output_len: int = 2
    ) -> tuple[bool, str]:
        """Schedule a request and consume its stream to completion."""
        response = await self._schedule(rid, output_len=output_len)
        if response.code != 200 or not response.success:
            return False, f"schedule failed: {response.error_message}"
        input_pb = (
            None if response.enqueued_by_master else self._build_generate_input(rid)
        )
        stream = await self._start_stream(response, rid, input_pb=input_pb)
        snap = StreamSnapshot()
        await self._consume_stream(stream, snap)
        if snap.error:
            return False, f"stream error: {snap.error}"
        if not snap.completed:
            return False, "stream did not complete"
        return True, f"ok (outputs={len(snap.outputs)})"

    # -- SM-C1: all prefill dead -> recovery -----------------------------

    async def test_all_prefill_dead_recovery(self) -> ScenarioResult:
        """Stop all prefill engines -> requests rejected -> restart -> recover."""
        start = time.monotonic()
        name = "SM-C1: all_prefill_dead_recovery"
        prefill_names: list[str] = []
        try:
            # 1. Baseline: confirm the cluster serves a request normally.
            rid = self._next_request_id()
            ok, detail = await self._send_and_complete(rid, output_len=2)
            if not ok:
                return ScenarioResult(
                    name, False, f"baseline failed: {detail}", time.monotonic() - start
                )

            # 2. Stop every prefill engine (simulates full EP death).
            prefill_names = await self._engine_names("prefill")
            if not prefill_names:
                return ScenarioResult(
                    name, False, "no prefill engines found", time.monotonic() - start
                )
            for n in prefill_names:
                await self._stop_engine(n)
            # Let FlexLB sync detect the dead prefill servers.
            await asyncio.sleep(self.ENGINE_DOWN_DETECT_S)

            # 3. Send a request; it must be rejected (schedule error or stream
            #    failure / timeout — all prefill gRPC servers are down).
            rid2 = self._next_request_id()
            rejected = False
            reject_detail = "no error observed"
            response2 = None
            try:
                response2 = await self._schedule(rid2, output_len=2)
                if response2.code != 200 or not response2.success:
                    rejected = True
                    reject_detail = (
                        f"schedule rejected: code={response2.code} "
                        f"msg={response2.error_message}"
                    )
                else:
                    input_pb2 = (
                        None
                        if response2.enqueued_by_master
                        else self._build_generate_input(rid2)
                    )
                    stream2 = await self._start_stream(
                        response2, rid2, input_pb=input_pb2
                    )
                    snap2 = StreamSnapshot()
                    task2 = asyncio.create_task(self._consume_stream(stream2, snap2))
                    try:
                        await asyncio.wait_for(
                            task2, timeout=self.REJECT_STREAM_TIMEOUT_S
                        )
                    except asyncio.TimeoutError:
                        task2.cancel()
                        try:
                            await task2
                        except (Exception, asyncio.CancelledError):
                            pass
                    if snap2.error:
                        rejected = True
                        reject_detail = f"stream error: {snap2.error}"
                    elif not snap2.completed:
                        rejected = True
                        reject_detail = (
                            f"stream did not complete within "
                            f"{self.REJECT_STREAM_TIMEOUT_S}s"
                        )
            except Exception as exc:
                rejected = True
                reject_detail = f"exception: {exc!r}"

            # Best-effort cleanup of the rejected request.
            if response2 is not None and response2.success:
                try:
                    await self._cancel(rid2, response2)
                except Exception:
                    pass

            if not rejected:
                return ScenarioResult(
                    name,
                    False,
                    "request was NOT rejected after all prefill stopped",
                    time.monotonic() - start,
                )

            # 4. Restart every prefill engine.
            for n in prefill_names:
                await self._start_engine(n)
            await asyncio.sleep(self.ENGINE_UP_DETECT_S)

            # 5. Verify recovery + inflight clean.
            recovery_ok, recovery_msg = await self._verify_recovery()
            inflight_ok, inflight_detail = await self._verify_inflight_clean(
                timeout_s=10.0
            )
            passed = recovery_ok and inflight_ok
            return ScenarioResult(
                name,
                passed,
                f"prefill_stopped={prefill_names}, "
                f"rejected=({reject_detail}), "
                f"recovery={recovery_msg}, inflight={inflight_detail}",
                time.monotonic() - start,
            )
        except Exception as exc:
            # Best-effort: restart any prefill we stopped so later suites run.
            for n in prefill_names:
                try:
                    await self._start_engine(n)
                except Exception:
                    pass
            return ScenarioResult(
                name, False, f"exception: {exc!r}", time.monotonic() - start
            )

    # -- SM-C2: selective_omit STALE eviction ----------------------------

    async def test_selective_omit_stale_evict(self) -> ScenarioResult:
        """Inject omit_request_ids -> STALE eviction -> inflight clean."""
        start = time.monotonic()
        name = "SM-C2: selective_omit_stale_evict"
        target_engine: str | None = None
        target_addr: str | None = None
        rid = self._next_request_id()
        try:
            # 1. Schedule a long-decode request so it stays in decode-running.
            response = await self._schedule(rid, output_len=self.LONG_OUTPUT_LEN)
            if response.code != 200 or not response.success:
                return ScenarioResult(
                    name,
                    False,
                    f"schedule failed: {response.error_message}",
                    time.monotonic() - start,
                )

            # 2. Wait until the request shows up in decode engine_tasks.
            deadline = time.monotonic() + self.DECODE_ENTRY_TIMEOUT_S
            entered = False
            while time.monotonic() < deadline:
                total = await self._decode_engine_tasks_total()
                if total >= 1:
                    entered = True
                    break
                await asyncio.sleep(0.1)
            if not entered:
                return ScenarioResult(
                    name,
                    False,
                    "request did not enter decode running in time",
                    time.monotonic() - start,
                )

            # 3. Locate the decode engine holding this requestId.
            target_engine, target_addr = await self._find_decode_engine_for_rid(rid)
            if target_engine is None:
                return ScenarioResult(
                    name,
                    False,
                    f"could not locate decode engine for rid={rid}",
                    time.monotonic() - start,
                )

            # 4. Inject omit_request_ids on that engine. The engine stays
            #    alive and keeps bumping status_version, but worker_status
            #    hides this requestId from running_task_info and
            #    finished_task_list — FlexLB never sees it.
            await self._inject(target_engine, {"omit_request_ids": [rid]})

            # 5. Wait for STALE eviction (STALE_EVICT_ROUNDS consecutive
            #    unseen calibrate rounds).
            await asyncio.sleep(self.STALE_WAIT_S)

            # 6. Verify STALE eviction occurred — BEFORE clearing the inject,
            #    otherwise the engine would resume reporting the requestId and
            #    FlexLB could re-create the engineTask.
            # 6a. Master log must contain the stale-eviction warning for rid.
            log_stale = self._grep_log("evicting as stale", str(rid))
            # 6b. inflight_status: decode engine_tasks, inflight_requests and
            #    scheduler_inflight must all be zero (A3 drives the bound
            #    InflightItem terminal on STALE eviction).
            st = await self._get_inflight_status()
            sched_inflight = st.get("scheduler_inflight", -1)
            decode_et = sum(
                ep.get("engine_tasks", 0) for ep in st.get("decode_endpoints", [])
            )
            decode_ir = sum(
                ep.get("inflight_requests", 0) for ep in st.get("decode_endpoints", [])
            )
            inflight_zero = sched_inflight == 0 and decode_et == 0 and decode_ir == 0

            # 7. Clean up: cancel the orphaned task on the mock engine so it
            #    stops decoding, then clear the inject so the engine reports
            #    normally again.
            try:
                if target_addr:
                    await self._cancel_on_engine(target_addr, rid)
            except Exception:
                pass
            try:
                await self._clear_inject(target_engine)
            except Exception:
                pass

            passed = log_stale and inflight_zero
            return ScenarioResult(
                name,
                passed,
                f"target={target_engine}, log_stale={log_stale}, "
                f"scheduler_inflight={sched_inflight}, "
                f"decode_engine_tasks={decode_et}, "
                f"decode_inflight_requests={decode_ir}",
                time.monotonic() - start,
            )
        except Exception as exc:
            # Best-effort cleanup.
            try:
                if target_engine is not None:
                    await self._clear_inject(target_engine)
            except Exception:
                pass
            try:
                if target_addr:
                    await self._cancel_on_engine(target_addr, rid)
            except Exception:
                pass
            return ScenarioResult(
                name, False, f"exception: {exc!r}", time.monotonic() - start
            )

    # -- SM-C3: KV exhaustion rejection ----------------------------------

    async def test_kv_exhaustion_rejection(self) -> ScenarioResult:
        """Exhaust all decode KV -> schedule rejected -> restore -> recover.

        Sets ``active_kv_tokens = total_kv`` on every decode engine so
        ``available_kv = 0 < seqLen`` (2048). CostBasedDecodeStrategy's
        hard filter rejects all decode EPs with KV_CAPACITY, and the
        router returns NO_AVAILABLE_WORKER (8400).  The request is never
        enqueued, so no inflight leak is possible.  After restoring KV,
        a fresh request must succeed.
        """
        start = time.monotonic()
        name = "SM-C3: kv_exhaustion_rejection"
        injected_engines: list[str] = []
        try:
            # 1. Baseline: confirm normal operation.
            rid = self._next_request_id()
            ok, detail = await self._send_and_complete(rid, output_len=2)
            if not ok:
                return ScenarioResult(
                    name,
                    False,
                    f"baseline failed: {detail}",
                    time.monotonic() - start,
                )

            # 2. Exhaust KV on every decode engine.
            snap = await self._get_snapshot()
            for e in snap["engines"]:
                if e.get("role") != "decode":
                    continue
                avail = e.get("available_kv_tokens", 0)
                active = e.get("active_kv_tokens", 0)
                total_kv = avail + active
                if total_kv <= 0:
                    continue
                await self._set_kv_pressure(e["name"], total_kv)
                injected_engines.append(e["name"])

            if not injected_engines:
                return ScenarioResult(
                    name,
                    False,
                    "no decode engines with KV capacity found",
                    time.monotonic() - start,
                )

            # 3. Wait for FlexLB sync to pick up the KV pressure.
            await asyncio.sleep(self.KV_PRESSURE_SYNC_S)

            # 4. Send a request -> should be rejected.
            rid2 = self._next_request_id()
            rejected = False
            reject_detail = "no error observed"
            response2 = None
            try:
                response2 = await self._schedule(rid2, output_len=2)
                if response2.code != 200 or not response2.success:
                    rejected = True
                    reject_detail = (
                        f"schedule rejected: code={response2.code} "
                        f"msg={response2.error_message}"
                    )
                else:
                    # Schedule succeeded unexpectedly — try the stream.
                    input_pb2 = (
                        None
                        if response2.enqueued_by_master
                        else self._build_generate_input(rid2)
                    )
                    stream2 = await self._start_stream(
                        response2, rid2, input_pb=input_pb2
                    )
                    snap2 = StreamSnapshot()
                    task2 = asyncio.create_task(self._consume_stream(stream2, snap2))
                    try:
                        await asyncio.wait_for(
                            task2, timeout=self.REJECT_STREAM_TIMEOUT_S
                        )
                    except asyncio.TimeoutError:
                        task2.cancel()
                        try:
                            await task2
                        except (Exception, asyncio.CancelledError):
                            pass
                    if snap2.error:
                        rejected = True
                        reject_detail = f"stream error: {snap2.error}"
                    elif not snap2.completed:
                        rejected = True
                        reject_detail = (
                            f"stream did not complete within "
                            f"{self.REJECT_STREAM_TIMEOUT_S}s"
                        )
            except Exception as exc:
                rejected = True
                reject_detail = f"exception: {exc!r}"

            # Best-effort cleanup of the rejected request.
            if response2 is not None and response2.success:
                try:
                    await self._cancel(rid2, response2)
                except Exception:
                    pass

            if not rejected:
                return ScenarioResult(
                    name,
                    False,
                    "request was NOT rejected after KV exhaustion",
                    time.monotonic() - start,
                )

            # 5. Verify no inflight leak.
            inflight_ok, inflight_detail = await self._verify_inflight_clean(
                timeout_s=10.0
            )

            # 6. Restore KV pressure.
            for n in injected_engines:
                try:
                    await self._set_kv_pressure(n, 0)
                except Exception:
                    pass
            await asyncio.sleep(self.KV_PRESSURE_SYNC_S)

            # 7. Verify recovery.
            recovery_ok, recovery_msg = await self._verify_recovery()

            passed = inflight_ok and recovery_ok
            return ScenarioResult(
                name,
                passed,
                f"kv_exhausted={injected_engines}, "
                f"rejected=({reject_detail}), "
                f"inflight={inflight_detail}, recovery={recovery_msg}",
                time.monotonic() - start,
            )
        except Exception as exc:
            for n in injected_engines:
                try:
                    await self._set_kv_pressure(n, 0)
                except Exception:
                    pass
            return ScenarioResult(
                name, False, f"exception: {exc!r}", time.monotonic() - start
            )

    # -- SM-C4: dispatch pool saturation ---------------------------------

    async def test_dispatch_pool_saturation(self) -> ScenarioResult:
        """Long prefill + max_concurrency=1 -> 20 concurrent -> no leak.

        Sets ``prefill_fixed_ms=10000`` + ``max_prefill_concurrency=1`` on
        all prefill engines.  The gRPC EnqueueBatch deadline (5 s) fires
        before the mock engine finishes, so every item ``failTimeout``s.
        With a small dispatch pool (env-configured in run_extreme_smoke.sh)
        concurrent dispatches may also trigger RejectedExecutionException
        -> ``failDispatch``.  Either way, all items must settle and inflight
        must return to zero.  After restoring perf, a fresh request must
        succeed.
        """
        start = time.monotonic()
        name = "SM-C4: dispatch_pool_saturation"
        prefill_names: list[str] = []
        try:
            prefill_names = await self._engine_names("prefill")
            if not prefill_names:
                return ScenarioResult(
                    name,
                    False,
                    "no prefill engines found",
                    time.monotonic() - start,
                )

            # 1. Set long prefill_ms + max_concurrency=1 on all prefill engines.
            for n in prefill_names:
                await self._set_perf(
                    n, prefill_fixed_ms=10000.0, max_prefill_concurrency=1
                )

            # 2. Send a burst of concurrent requests.
            rids = [self._next_request_id() for _ in range(self.BURST_CONCURRENT)]

            async def _send_one(rid: int) -> tuple[bool, str]:
                try:
                    return await asyncio.wait_for(
                        self._send_and_complete(rid, output_len=2),
                        timeout=self.DISPATCH_REQUEST_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    return False, f"timeout after {self.DISPATCH_REQUEST_TIMEOUT_S}s"
                except Exception as exc:
                    return False, f"exception: {exc!r}"

            results = await asyncio.gather(*[_send_one(rid) for rid in rids])
            failed_count = sum(1 for ok, _ in results if not ok)
            succeeded_count = self.BURST_CONCURRENT - failed_count

            # 3. Check log for dispatch pool rejection evidence (best-effort).
            log_rejected = self._grep_log("dispatch rejected")

            # 4. Verify no inflight leak.
            inflight_ok, inflight_detail = await self._verify_inflight_clean(
                timeout_s=30.0
            )

            # 5. Restore perf.
            for n in prefill_names:
                try:
                    await self._set_perf(
                        n, prefill_fixed_ms=100.0, max_prefill_concurrency=1
                    )
                except Exception:
                    pass
            await asyncio.sleep(2.0)

            # 6. Verify recovery.
            recovery_ok, recovery_msg = await self._verify_recovery()

            # Pass if: at least some requests failed (expected under
            # saturation), no inflight leak, and recovery works.
            passed = (failed_count > 0) and inflight_ok and recovery_ok
            sample_failures = [msg for ok, msg in results if not ok][:3]
            return ScenarioResult(
                name,
                passed,
                f"concurrent={self.BURST_CONCURRENT}, "
                f"failed={failed_count}, succeeded={succeeded_count}, "
                f"dispatch_rejected_log={log_rejected}, "
                f"inflight={inflight_detail}, recovery={recovery_msg}"
                + (f", sample_failures={sample_failures}" if sample_failures else ""),
                time.monotonic() - start,
            )
        except Exception as exc:
            for n in prefill_names:
                try:
                    await self._set_perf(
                        n, prefill_fixed_ms=100.0, max_prefill_concurrency=1
                    )
                except Exception:
                    pass
            return ScenarioResult(
                name, False, f"exception: {exc!r}", time.monotonic() - start
            )

    # -- SM-C5: cold start burst -----------------------------------------

    async def test_cold_start_burst(self) -> ScenarioResult:
        """20+ concurrent requests with no warm-up -> verify convergence.

        Runs as the **first** scenario (closest to cold start).  Some
        requests may be rejected while EPs are not yet synced; the test
        verifies that all requests settle (no leak) and inflight converges
        to zero.  A fresh recovery request must also succeed.
        """
        start = time.monotonic()
        name = "SM-C5: cold_start_burst"
        try:
            rids = [self._next_request_id() for _ in range(self.BURST_CONCURRENT)]

            async def _send_one(rid: int) -> tuple[bool, str]:
                try:
                    return await asyncio.wait_for(
                        self._send_and_complete(rid, output_len=2),
                        timeout=self.DISPATCH_REQUEST_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    return False, f"timeout after {self.DISPATCH_REQUEST_TIMEOUT_S}s"
                except Exception as exc:
                    return False, f"exception: {exc!r}"

            results = await asyncio.gather(*[_send_one(rid) for rid in rids])
            failed_count = sum(1 for ok, _ in results if not ok)
            succeeded_count = self.BURST_CONCURRENT - failed_count

            # Verify inflight converges to zero (all items settled).
            inflight_ok, inflight_detail = await self._verify_inflight_clean(
                timeout_s=30.0
            )

            # Verify recovery (a fresh request works).
            recovery_ok, recovery_msg = await self._verify_recovery()

            # Pass if: inflight is clean and recovery works.
            # Some failures are expected during cold start.
            passed = inflight_ok and recovery_ok
            return ScenarioResult(
                name,
                passed,
                f"concurrent={self.BURST_CONCURRENT}, "
                f"failed={failed_count}, succeeded={succeeded_count}, "
                f"inflight={inflight_detail}, recovery={recovery_msg}",
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                name, False, f"exception: {exc!r}", time.monotonic() - start
            )

    # -- Runner ----------------------------------------------------------

    async def run_all(self) -> int:
        scenarios = [
            self.test_cold_start_burst,  # SM-C5: first (cold start)
            self.test_all_prefill_dead_recovery,  # SM-C1
            self.test_selective_omit_stale_evict,  # SM-C2
            self.test_kv_exhaustion_rejection,  # SM-C3
            self.test_dispatch_pool_saturation,  # SM-C4: last (high load)
        ]
        print("=" * 70)
        print("FlexLB Extreme Scenario Smoke Test")
        print(f"  master: {self._master_target()}")
        print(f"  deploy_mode: {self._deploy_mode}")
        print(f"  mock_http_port: {self.args.mock_http_port}")
        print(f"  flexlb_log: {getattr(self.args, 'flexlb_log', 'N/A')}")
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
    parser.add_argument("--request-id-base", type=int, default=50000)
    parser.add_argument(
        "--flexlb-log",
        default=None,
        help="path to flexlb master log file or log directory (for STALE eviction grep)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    test = ExtremeSmokeTest(args)
    try:
        exit_code = await test.run_all()
    finally:
        await test.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
