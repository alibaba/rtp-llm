#!/usr/bin/env python3
"""Resilience smoke tests for FlexLB: engine process-level stop/restart.

Connects to a running FlexLB master and mock engine cluster.  Exercises
six resilience scenarios that simulate **hard failures** (gRPC connection
refused) via the mock engine's stop_engine/start_engine HTTP API:

  R1  prefill_engine_stop   - Stop one prefill engine → verify graceful
                               degradation → restart → recovery ok +
                               traffic redistribution.
  R2  decode_engine_stop    - Stop one decode engine → verify graceful
                               degradation → restart → recovery ok +
                               traffic redistribution.
  R3  engine_stop_restart    - Stop engine mid-stream → verify stream error
                               / connection refused → restart → gRPC channel
                               recovery + stuck inflight cleanup.
  R4  all_prefill_stop       - Stop ALL prefill engines → verify all requests
                               fail → restart all → recovery ok.
  R5  all_decode_stop        - Stop ALL decode engines → verify all requests
                               fail (batch/direct) or no decode traffic
                               (queue) → restart all → recovery ok.
  R6  partial_decode_stop    - Stop K decode engines (1 < K < N) → verify
                               surviving engines serve + traffic redistribution
                               → restart → recovery ok.

Usage:
    python3 resilience_smoke.py --master-ip 127.0.0.1 \\
        --master-http-port 18080 --mock-http-port 55150 \\
        --schedule-mode batch
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from flexlb_smoke_base import FlexLBSmokeBase, ScenarioResult, StreamSnapshot


class ResilienceSmokeTest(FlexLBSmokeBase):
    """Resilience path smoke tests: engine stop/restart hard failures."""

    # Test-specific timeouts (seconds)
    ENGINE_STOP_WAIT_S = 2.0  # Wait for master to detect gRPC failure
    ENGINE_RESTART_WAIT_S = 3.0  # Wait for master to reconnect after restart
    STREAM_TIMEOUT_S = 10.0  # Max wait for stream to complete/fail
    BASELINE_VERIFY_TIMEOUT_S = 15.0

    # -- Helpers ----------------------------------------------------------

    async def _schedule_auto(self, request_id: int, **kwargs):
        """Schedule with the configured schedule_mode from CLI args."""
        kwargs.setdefault("schedule_mode", self.args.schedule_mode)
        return await self._schedule(request_id, **kwargs)

    async def _get_engine_names(self, role: str) -> list[str]:
        """Return engine names filtered by role from the mock cluster snapshot."""
        snapshot = await self._get_snapshot()
        return [e["name"] for e in snapshot["engines"] if e["role"] == role]

    async def _stop_all(self, names: list[str]) -> None:
        """Stop all engines in the given name list."""
        for name in names:
            await self._stop_engine(name)

    async def _start_all(self, names: list[str]) -> None:
        """Restart all engines in the given name list."""
        for name in names:
            try:
                await self._start_engine(name)
            except Exception:
                pass

    async def _send_and_verify_request(
        self, rid: int, *, timeout_s: float = STREAM_TIMEOUT_S
    ) -> tuple[bool, str, object]:
        """Schedule + stream a single request.

        Returns (success, detail_str, response).
        """
        try:
            response = await self._schedule_auto(rid)
            if response.code != 200 or not response.success:
                return False, f"schedule error: {response.error_message}", response
            input_pb = (
                None if response.enqueued_by_master else self._build_generate_input(rid)
            )
            stream = await self._start_stream(response, rid, input_pb=input_pb)
            snap = StreamSnapshot()
            task = asyncio.create_task(self._consume_stream(stream, snap))
            try:
                await asyncio.wait_for(task, timeout=timeout_s)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (Exception, asyncio.CancelledError):
                    pass
            if snap.error:
                return False, f"stream error: {snap.error}", response
            if not snap.completed:
                return False, "stream did not complete", response
            return True, f"ok (outputs={len(snap.outputs)})", response
        except Exception as exc:
            return False, f"exception: {exc!r}", None

    async def _snapshot_by_name(self) -> dict[str, dict]:
        """Return a dict mapping engine name → full snapshot dict."""
        snapshot = await self._get_snapshot()
        return {e["name"]: e for e in snapshot["engines"]}

    async def _verify_traffic_redistribution(
        self, target_name: str
    ) -> tuple[bool, str]:
        """Send 5 concurrent requests and verify *target_name* received traffic.

        Compares the engine's ``accepted`` counter before and after sending.
        Returns (success, detail_str).
        """
        snap_before = await self._snapshot_by_name()
        before_accepted = snap_before.get(target_name, {}).get("accepted", 0)

        rids = [self._next_request_id() for _ in range(5)]
        results = await asyncio.gather(
            *[self._send_and_verify_request(rid) for rid in rids]
        )

        for i, result in enumerate(results):
            ok, _detail, response = result
            if response is not None and getattr(response, "success", False):
                try:
                    await self._cancel(rids[i], response)
                except Exception:
                    pass

        snap_after = await self._snapshot_by_name()
        after_accepted = snap_after.get(target_name, {}).get("accepted", 0)

        redistributed = after_accepted > before_accepted
        detail = (
            f"target={target_name}, "
            f"accepted_before={before_accepted}, "
            f"accepted_after={after_accepted}"
        )
        return redistributed, detail

    # -- R1: prefill_engine_stop -----------------------------------------

    async def test_prefill_engine_stop(self) -> ScenarioResult:
        """Stop one prefill engine → verify traffic routing → restart → recovery."""
        start = time.monotonic()
        stopped_names: list[str] = []
        response = None
        try:
            # Baseline: verify normal operation
            baseline_ok, baseline_msg = await self._verify_recovery()
            if not baseline_ok:
                return ScenarioResult(
                    "R1: prefill_engine_stop",
                    False,
                    f"baseline failed: {baseline_msg}",
                    time.monotonic() - start,
                )

            # Stop one prefill engine
            prefill_names = await self._get_engine_names("prefill")
            if len(prefill_names) < 2:
                return ScenarioResult(
                    "R1: prefill_engine_stop",
                    False,
                    f"need >=2 prefill engines for single-stop test, "
                    f"got {len(prefill_names)}",
                    time.monotonic() - start,
                )

            target = prefill_names[0]
            await self._stop_engine(target)
            stopped_names.append(target)

            # Wait for master to detect the gRPC failure
            await asyncio.sleep(self.ENGINE_STOP_WAIT_S)

            # Send request — should route to surviving engine or fail gracefully
            rid = self._next_request_id()
            req_ok, req_detail, response = await self._send_and_verify_request(rid)

            # Cancel any failed request to clean up server-side inflight
            if response is not None and response.success:
                try:
                    await self._cancel(rid, response)
                except Exception:
                    pass

            # Restart the stopped engine
            await self._start_engine(target)
            stopped_names.clear()

            await asyncio.sleep(self.ENGINE_RESTART_WAIT_S)

            recovery_ok, recovery_msg = await self._verify_recovery()

            # inflight cleanup verification (batch path only)
            inflight_ok = True
            inflight_detail = "N/A (non-batch path)"
            if self.args.schedule_mode == "batch":
                inflight_ok, inflight_detail = await self._verify_inflight_clean(
                    timeout_s=10.0
                )

            # Traffic redistribution: verify restarted engine receives new traffic
            redistributed, redistribute_detail = (
                await self._verify_traffic_redistribution(target)
            )

            passed = recovery_ok and inflight_ok and redistributed
            return ScenarioResult(
                "R1: prefill_engine_stop",
                passed,
                f"stopped={target}, during_stop_request_ok={req_ok}"
                f"({req_detail}), "
                f"recovery={recovery_msg}, "
                f"inflight_clean={inflight_ok}({inflight_detail}), "
                f"traffic_redistributed={redistributed}({redistribute_detail})",
                time.monotonic() - start,
            )
        except Exception as exc:
            # Cleanup: restart any stopped engines
            for name in stopped_names:
                try:
                    await self._start_engine(name)
                except Exception:
                    pass
            return ScenarioResult(
                "R1: prefill_engine_stop",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- R2: decode_engine_stop ------------------------------------------

    async def test_decode_engine_stop(self) -> ScenarioResult:
        """Stop one decode engine → verify degradation → restart → recovery."""
        start = time.monotonic()
        stopped_names: list[str] = []
        response = None
        try:
            # Baseline: verify normal operation
            baseline_ok, baseline_msg = await self._verify_recovery()
            if not baseline_ok:
                return ScenarioResult(
                    "R2: decode_engine_stop",
                    False,
                    f"baseline failed: {baseline_msg}",
                    time.monotonic() - start,
                )

            # Stop one decode engine
            decode_names = await self._get_engine_names("decode")
            if len(decode_names) < 2:
                return ScenarioResult(
                    "R2: decode_engine_stop",
                    False,
                    f"need >=2 decode engines for single-stop test, "
                    f"got {len(decode_names)}",
                    time.monotonic() - start,
                )

            target = decode_names[0]
            await self._stop_engine(target)
            stopped_names.append(target)

            # Wait for master to detect the gRPC failure
            await asyncio.sleep(self.ENGINE_STOP_WAIT_S)

            # Send request — may route to surviving decode engines or fail
            rid = self._next_request_id()
            req_ok, req_detail, response = await self._send_and_verify_request(rid)

            # Cancel any failed request to clean up server-side inflight
            if response is not None and response.success:
                try:
                    await self._cancel(rid, response)
                except Exception:
                    pass

            # Restart the stopped engine
            await self._start_engine(target)
            stopped_names.clear()

            await asyncio.sleep(self.ENGINE_RESTART_WAIT_S)

            recovery_ok, recovery_msg = await self._verify_recovery()

            inflight_ok = True
            inflight_detail = "N/A (non-batch path)"
            if self.args.schedule_mode == "batch":
                inflight_ok, inflight_detail = await self._verify_inflight_clean(
                    timeout_s=10.0
                )

            # Traffic redistribution: verify restarted engine receives new traffic
            redistributed, redistribute_detail = (
                await self._verify_traffic_redistribution(target)
            )

            passed = recovery_ok and inflight_ok and redistributed
            return ScenarioResult(
                "R2: decode_engine_stop",
                passed,
                f"stopped={target}, during_stop_request_ok={req_ok}"
                f"({req_detail}), "
                f"recovery={recovery_msg}, "
                f"inflight_clean={inflight_ok}({inflight_detail}), "
                f"traffic_redistributed={redistributed}({redistribute_detail})",
                time.monotonic() - start,
            )
        except Exception as exc:
            for name in stopped_names:
                try:
                    await self._start_engine(name)
                except Exception:
                    pass
            return ScenarioResult(
                "R2: decode_engine_stop",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- R3: engine_stop_restart -----------------------------------------

    async def test_engine_stop_restart(self) -> ScenarioResult:
        """Stop engine mid-stream → verify error → restart → channel recovery.

        This test focuses on gRPC channel recovery after a hard engine stop
        and verifies that stuck inflight entries are cleaned up.
        """
        start = time.monotonic()
        stopped_names: list[str] = []
        response = None
        try:
            # Baseline: verify normal operation
            baseline_ok, baseline_msg = await self._verify_recovery()
            if not baseline_ok:
                return ScenarioResult(
                    "R3: engine_stop_restart",
                    False,
                    f"baseline failed: {baseline_msg}",
                    time.monotonic() - start,
                )

            # Start a request stream (baseline flow)
            rid = self._next_request_id()
            response = await self._schedule_auto(rid)
            if response.code != 200 or not response.success:
                return ScenarioResult(
                    "R3: engine_stop_restart",
                    False,
                    f"schedule failed: {response.error_message}",
                    time.monotonic() - start,
                )
            input_pb = (
                None if response.enqueued_by_master else self._build_generate_input(rid)
            )
            stream = await self._start_stream(response, rid, input_pb=input_pb)
            snap = StreamSnapshot()
            task = asyncio.create_task(self._consume_stream(stream, snap))

            # Wait for first output to ensure stream is active
            got_first = await self._wait_for_first_output(snap)

            # Stop one prefill engine to cause connection error
            prefill_names = await self._get_engine_names("prefill")
            if not prefill_names:
                return ScenarioResult(
                    "R3: engine_stop_restart",
                    False,
                    "no prefill engines found",
                    time.monotonic() - start,
                )

            target = prefill_names[0]
            await self._stop_engine(target)
            stopped_names.append(target)

            # Wait for the stream to detect the failure
            stream_errored = False
            stream_detail = "no error"
            try:
                await asyncio.wait_for(task, timeout=self.STREAM_TIMEOUT_S)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (Exception, asyncio.CancelledError):
                    pass
                stream_detail = "stream timed out (no error observed)"
            except Exception as exc:
                stream_detail = f"stream task exception: {exc!r}"

            if snap.error:
                stream_errored = True
                stream_detail = f"stream error: {snap.error}"
            elif not snap.completed and not got_first:
                stream_errored = True
                stream_detail = "no output received (connection refused)"

            # Cancel the failed request to clean up server-side inflight
            try:
                await self._cancel(rid, response)
            except Exception:
                pass

            # Send a new request while engine is still stopped
            rid2 = self._next_request_id()
            req2_ok, req2_detail, resp2 = await self._send_and_verify_request(
                rid2, timeout_s=self.STREAM_TIMEOUT_S
            )
            if resp2 is not None and resp2.success:
                try:
                    await self._cancel(rid2, resp2)
                except Exception:
                    pass

            # Restart the stopped engine
            await self._start_engine(target)
            stopped_names.clear()

            await asyncio.sleep(self.ENGINE_RESTART_WAIT_S)

            # Verify gRPC channel recovery
            recovery_ok, recovery_msg = await self._verify_recovery()

            # Verify stuck inflight cleanup
            inflight_ok = True
            inflight_detail = "N/A (non-batch path)"
            if self.args.schedule_mode == "batch":
                inflight_ok, inflight_detail = await self._verify_inflight_clean(
                    timeout_s=15.0
                )

            passed = recovery_ok and inflight_ok
            return ScenarioResult(
                "R3: engine_stop_restart",
                passed,
                f"stopped={target}, stream_first_output={got_first}, "
                f"stream_errored={stream_errored}({stream_detail}), "
                f"during_stop_request_ok={req2_ok}({req2_detail}), "
                f"recovery={recovery_msg}, "
                f"inflight_clean={inflight_ok}({inflight_detail})",
                time.monotonic() - start,
            )
        except Exception as exc:
            for name in stopped_names:
                try:
                    await self._start_engine(name)
                except Exception:
                    pass
            return ScenarioResult(
                "R3: engine_stop_restart",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- R4: all_prefill_stop --------------------------------------------

    async def test_all_prefill_stop(self) -> ScenarioResult:
        """Stop ALL prefill engines → verify all requests fail → restart → recovery."""
        start = time.monotonic()
        stopped_names: list[str] = []
        response = None
        try:
            # Baseline: verify normal operation
            baseline_ok, baseline_msg = await self._verify_recovery()
            if not baseline_ok:
                return ScenarioResult(
                    "R4: all_prefill_stop",
                    False,
                    f"baseline failed: {baseline_msg}",
                    time.monotonic() - start,
                )

            # Stop ALL prefill engines
            prefill_names = await self._get_engine_names("prefill")
            if not prefill_names:
                return ScenarioResult(
                    "R4: all_prefill_stop",
                    False,
                    "no prefill engines found",
                    time.monotonic() - start,
                )

            await self._stop_all(prefill_names)
            stopped_names = list(prefill_names)

            # Wait for master to detect all gRPC failures
            await asyncio.sleep(self.ENGINE_STOP_WAIT_S)

            # Send request — should fail (no prefill available)
            rid = self._next_request_id()
            error_observed = False
            error_detail = "no error observed"
            try:
                response = await self._schedule_auto(rid)
                if response.code != 200 or not response.success:
                    error_observed = True
                    error_detail = f"schedule error: {response.error_message}"
                else:
                    # Even if schedule succeeds, the stream should fail
                    input_pb = (
                        None
                        if response.enqueued_by_master
                        else self._build_generate_input(rid)
                    )
                    stream = await self._start_stream(response, rid, input_pb=input_pb)
                    snap = StreamSnapshot()
                    task = asyncio.create_task(self._consume_stream(stream, snap))
                    try:
                        await asyncio.wait_for(task, timeout=self.STREAM_TIMEOUT_S)
                    except asyncio.TimeoutError:
                        task.cancel()
                        try:
                            await task
                        except (Exception, asyncio.CancelledError):
                            pass
                    if snap.error:
                        error_observed = True
                        error_detail = f"stream error: {snap.error}"
                    elif not snap.completed:
                        error_observed = True
                        error_detail = "stream did not complete"
            except Exception as exc:
                error_observed = True
                error_detail = f"exception: {exc!r}"

            # Cancel any failed request to clean up server-side inflight
            if response is not None and response.success:
                try:
                    await self._cancel(rid, response)
                except Exception:
                    pass

            # Restart ALL prefill engines
            await self._start_all(stopped_names)
            stopped_names.clear()

            await asyncio.sleep(self.ENGINE_RESTART_WAIT_S)

            recovery_ok, recovery_msg = await self._verify_recovery()

            inflight_ok = True
            inflight_detail = "N/A (non-batch path)"
            if self.args.schedule_mode == "batch":
                inflight_ok, inflight_detail = await self._verify_inflight_clean(
                    timeout_s=15.0
                )

            passed = error_observed and recovery_ok and inflight_ok
            return ScenarioResult(
                "R4: all_prefill_stop",
                passed,
                f"stopped_all={len(prefill_names)} prefill engines, "
                f"error_observed={error_observed}({error_detail}), "
                f"recovery={recovery_msg}, "
                f"inflight_clean={inflight_ok}({inflight_detail})",
                time.monotonic() - start,
            )
        except Exception as exc:
            # Cleanup: restart all stopped engines
            for name in stopped_names:
                try:
                    await self._start_engine(name)
                except Exception:
                    pass
            return ScenarioResult(
                "R4: all_prefill_stop",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- R5: all_decode_stop ---------------------------------------------

    async def test_all_decode_stop(self) -> ScenarioResult:
        """Stop ALL decode engines -> verify requests fail (batch/direct) or
        no decode traffic (queue) -> restart -> recovery.

        In queue mode the Schedule RPC does not check decode worker
        availability (queueing semantics): the request is routed to a prefill
        engine which completes the full flow independently, so it may succeed.
        Instead of asserting an error we verify that no decode engine
        received traffic (accepted counter unchanged), confirming the decode
        stop took effect.
        """
        start = time.monotonic()
        stopped_names: list[str] = []
        response = None
        try:
            # Baseline: verify normal operation
            baseline_ok, baseline_msg = await self._verify_recovery()
            if not baseline_ok:
                return ScenarioResult(
                    "R5: all_decode_stop",
                    False,
                    f"baseline failed: {baseline_msg}",
                    time.monotonic() - start,
                )

            # Stop ALL decode engines
            decode_names = await self._get_engine_names("decode")
            if len(decode_names) < 2:
                return ScenarioResult(
                    "R5: all_decode_stop",
                    False,
                    f"need >=2 decode engines for all-stop test, "
                    f"got {len(decode_names)}",
                    time.monotonic() - start,
                )

            await self._stop_all(decode_names)
            stopped_names = list(decode_names)

            # Wait for master to detect all gRPC failures
            await asyncio.sleep(self.ENGINE_STOP_WAIT_S)

            is_queue_mode = self.args.schedule_mode == "queue"

            # In queue mode, record decode accepted counts before sending the
            # request so we can verify no decode engine received traffic.
            decode_before: dict[str, int] = {}
            if is_queue_mode:
                snap_before = await self._snapshot_by_name()
                decode_before = {
                    name: snap_before.get(name, {}).get("accepted", 0)
                    for name in decode_names
                }

            # Send request -- should fail in batch/direct mode (no decode
            # available).  In queue mode it may succeed.
            rid = self._next_request_id()
            error_observed = False
            error_detail = "no error observed"
            try:
                response = await self._schedule_auto(rid)
                if response.code != 200 or not response.success:
                    error_observed = True
                    error_detail = f"schedule error: {response.error_message}"
                else:
                    # Even if schedule succeeds, the stream should fail
                    input_pb = (
                        None
                        if response.enqueued_by_master
                        else self._build_generate_input(rid)
                    )
                    stream = await self._start_stream(response, rid, input_pb=input_pb)
                    snap = StreamSnapshot()
                    task = asyncio.create_task(self._consume_stream(stream, snap))
                    try:
                        await asyncio.wait_for(task, timeout=self.STREAM_TIMEOUT_S)
                    except asyncio.TimeoutError:
                        task.cancel()
                        try:
                            await task
                        except (Exception, asyncio.CancelledError):
                            pass
                    if snap.error:
                        error_observed = True
                        error_detail = f"stream error: {snap.error}"
                    elif not snap.completed:
                        error_observed = True
                        error_detail = "stream did not complete"
            except Exception as exc:
                error_observed = True
                error_detail = f"exception: {exc!r}"

            # Cancel any failed request to clean up server-side inflight
            if response is not None and response.success:
                try:
                    await self._cancel(rid, response)
                except Exception:
                    pass

            # In queue mode, verify no decode engine received traffic.
            # All decode engines are stopped, so their accepted counters
            # must not have increased during the request.
            no_decode_traffic = True
            no_decode_detail = "N/A (non-queue mode)"
            if is_queue_mode:
                snap_after = await self._snapshot_by_name()
                changed = []
                for name in decode_names:
                    before = decode_before.get(name, 0)
                    after = snap_after.get(name, {}).get("accepted", 0)
                    if after > before:
                        no_decode_traffic = False
                        changed.append(f"{name}: {before}->{after}")
                no_decode_detail = (
                    "no decode traffic observed"
                    if no_decode_traffic
                    else f"decode traffic changed: {', '.join(changed)}"
                )

            # Restart ALL decode engines
            await self._start_all(stopped_names)
            stopped_names.clear()

            await asyncio.sleep(self.ENGINE_RESTART_WAIT_S)

            recovery_ok, recovery_msg = await self._verify_recovery()

            inflight_ok = True
            inflight_detail = "N/A (non-batch path)"
            if self.args.schedule_mode == "batch":
                inflight_ok, inflight_detail = await self._verify_inflight_clean(
                    timeout_s=15.0
                )

            if is_queue_mode:
                passed = no_decode_traffic and recovery_ok and inflight_ok
            else:
                passed = error_observed and recovery_ok and inflight_ok
            return ScenarioResult(
                "R5: all_decode_stop",
                passed,
                f"stopped_all={len(decode_names)} decode engines, "
                f"mode={self.args.schedule_mode}, "
                f"error_observed={error_observed}({error_detail}), "
                f"no_decode_traffic={no_decode_traffic}({no_decode_detail}), "
                f"recovery={recovery_msg}, "
                f"inflight_clean={inflight_ok}({inflight_detail})",
                time.monotonic() - start,
            )
        except Exception as exc:
            # Cleanup: restart all stopped engines
            for name in stopped_names:
                try:
                    await self._start_engine(name)
                except Exception:
                    pass
            return ScenarioResult(
                "R5: all_decode_stop",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- R6: partial_decode_stop -----------------------------------------

    async def test_partial_decode_stop(self) -> ScenarioResult:
        """Stop K decode engines → surviving engines serve → restart → recovery."""
        start = time.monotonic()
        stopped_names: list[str] = []
        try:
            # Baseline: verify normal operation
            baseline_ok, baseline_msg = await self._verify_recovery()
            if not baseline_ok:
                return ScenarioResult(
                    "R6: partial_decode_stop",
                    False,
                    f"baseline failed: {baseline_msg}",
                    time.monotonic() - start,
                )

            # Get all decode engine names
            decode_names = await self._get_engine_names("decode")
            if len(decode_names) < 3:
                return ScenarioResult(
                    "R6: partial_decode_stop",
                    False,
                    f"need >=3 decode engines for partial-stop test, "
                    f"got {len(decode_names)}",
                    time.monotonic() - start,
                )

            # Stop first 2 decode engines, keep the rest alive
            stopped_names = decode_names[0:2]
            surviving_names = decode_names[2:]
            await self._stop_all(stopped_names)

            # Wait for master to detect gRPC failures
            await asyncio.sleep(self.ENGINE_STOP_WAIT_S)

            # Snapshot before: record accepted counts on surviving engines
            snap_before = await self._snapshot_by_name()
            surviving_before = {
                name: snap_before.get(name, {}).get("accepted", 0)
                for name in surviving_names
            }

            # Send 5 concurrent requests — surviving engines should serve
            rids = [self._next_request_id() for _ in range(5)]
            results = await asyncio.gather(
                *[self._send_and_verify_request(rid) for rid in rids]
            )

            # Cancel any successful requests to clean up inflight
            for i, result in enumerate(results):
                ok, _detail, response = result
                if response is not None and getattr(response, "success", False):
                    try:
                        await self._cancel(rids[i], response)
                    except Exception:
                        pass

            all_failed = all(not r[0] for r in results)
            success_count = sum(1 for r in results if r[0])

            # Snapshot after: check surviving engines' accepted counts increased
            snap_after = await self._snapshot_by_name()
            surviving_accepted = any(
                snap_after.get(name, {}).get("accepted", 0)
                > surviving_before.get(name, 0)
                for name in surviving_names
            )
            surviving_detail = ", ".join(
                f"{name}: {surviving_before.get(name, 0)}→"
                f"{snap_after.get(name, {}).get('accepted', 0)}"
                for name in surviving_names
            )

            # Restart stopped engines
            await self._start_all(stopped_names)
            stopped_names.clear()

            await asyncio.sleep(self.ENGINE_RESTART_WAIT_S)

            recovery_ok, recovery_msg = await self._verify_recovery()

            inflight_ok = True
            inflight_detail = "N/A (non-batch path)"
            if self.args.schedule_mode == "batch":
                inflight_ok, inflight_detail = await self._verify_inflight_clean(
                    timeout_s=15.0
                )

            passed = (
                not all_failed and surviving_accepted and recovery_ok and inflight_ok
            )
            return ScenarioResult(
                "R6: partial_decode_stop",
                passed,
                f"stopped={decode_names[0:2]}, surviving={surviving_names}, "
                f"all_failed={all_failed}, success={success_count}/5, "
                f"surviving_accepted={surviving_accepted}({surviving_detail}), "
                f"recovery={recovery_msg}, "
                f"inflight_clean={inflight_ok}({inflight_detail})",
                time.monotonic() - start,
            )
        except Exception as exc:
            # Cleanup: restart any stopped engines
            for name in stopped_names:
                try:
                    await self._start_engine(name)
                except Exception:
                    pass
            return ScenarioResult(
                "R6: partial_decode_stop",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- Runner ----------------------------------------------------------

    async def run_all(self) -> int:
        scenarios = [
            self.test_prefill_engine_stop,
            self.test_decode_engine_stop,
            self.test_engine_stop_restart,
            self.test_all_prefill_stop,
            self.test_all_decode_stop,
            self.test_partial_decode_stop,
        ]
        print("=" * 70)
        print("FlexLB Resilience Smoke Test")
        print(f"  master: {self._master_target()}")
        print(f"  schedule_mode: {self.args.schedule_mode}")
        print(f"  mock_http_port: {self.args.mock_http_port}")
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
    parser.add_argument(
        "--schedule-mode",
        choices=["auto", "batch", "direct", "queue"],
        default="batch",
    )
    parser.add_argument("--request-id-base", type=int, default=40000)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    test = ResilienceSmokeTest(args)
    try:
        exit_code = await test.run_all()
    finally:
        await test.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
