#!/usr/bin/env python3
"""Cancel smoke tests for FlexLB master + mock engine cluster.

Connects to a running FlexLB master (FlexlbService gRPC) and mock prefill
worker (RpcService gRPC).  Exercises six cancel scenarios and verifies
correctness of stream termination, idempotency, isolation, and recovery.

Usage:
    python3 cancel_smoke.py --master-ip 127.0.0.1 --master-http-port 18080
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from flexlb_smoke_base import FlexLBSmokeBase, ScenarioResult, StreamSnapshot


class CancelSmokeTest(FlexLBSmokeBase):
    """Drive six cancel scenarios against a FlexLB master + mock cluster."""

    # -- Helpers ----------------------------------------------------------

    async def _schedule_auto(self, request_id: int, **kwargs):
        """Schedule with the configured schedule_mode from CLI args."""
        kwargs.setdefault("schedule_mode", self.args.schedule_mode)
        return await self._schedule(request_id, **kwargs)

    # -- Scenario T1: basic_cancel ---------------------------------------

    async def test_basic_cancel(self) -> ScenarioResult:
        rid = self._next_request_id()
        start = time.monotonic()
        try:
            response = await self._schedule_auto(rid)
            if response.code != 200 or not response.success:
                return ScenarioResult(
                    "T1: basic_cancel",
                    False,
                    f"schedule failed: {response.error_message}",
                )
            input_pb = (
                None if response.enqueued_by_master else self._build_generate_input(rid)
            )
            stream = await self._start_stream(response, rid, input_pb=input_pb)
            snap = StreamSnapshot()
            task = asyncio.create_task(self._consume_stream(stream, snap))

            got_first = await self._wait_for_first_output(snap)
            if not got_first:
                task.cancel()
                return ScenarioResult(
                    "T1: basic_cancel",
                    False,
                    "no output received before cancel window",
                )

            cancel_at = time.monotonic()
            await self._cancel(rid, response)
            ended = await self._wait_for_stream_end(task)
            cancel_latency = time.monotonic() - cancel_at

            recovery_ok, recovery_msg = await self._verify_recovery()

            # engine 侧验证
            method = (
                "enqueue_batch" if response.enqueued_by_master else "generate_stream"
            )
            engine_recv, recv_detail = await self._verify_engine_received(rid, method)
            engine_cancelled, cancel_detail = await self._verify_engine_cancelled(rid)

            # batch 路径验证 inflight 清理
            inflight_ok = True
            inflight_detail = "N/A"
            if response.enqueued_by_master:
                inflight_ok, inflight_detail = await self._verify_inflight_clean(
                    timeout_s=10.0
                )

            passed = ended and recovery_ok
            return ScenarioResult(
                "T1: basic_cancel",
                passed,
                f"cancel_latency={cancel_latency:.3f}s, "
                f"stream_terminated={ended}, outputs={len(snap.outputs)}, "
                f"engine_recv={engine_recv}({recv_detail}), "
                f"engine_cancelled={engine_cancelled}({cancel_detail}), "
                f"inflight_clean={inflight_ok}({inflight_detail}), "
                f"recovery={recovery_msg}",
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "T1: basic_cancel",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- Scenario T2: cancel_idempotency ---------------------------------

    async def test_cancel_idempotency(self) -> ScenarioResult:
        rid = self._next_request_id()
        start = time.monotonic()
        try:
            response = await self._schedule_auto(rid)
            if response.code != 200 or not response.success:
                return ScenarioResult(
                    "T2: cancel_idempotency",
                    False,
                    f"schedule failed: {response.error_message}",
                )
            input_pb = (
                None if response.enqueued_by_master else self._build_generate_input(rid)
            )
            stream = await self._start_stream(response, rid, input_pb=input_pb)
            snap = StreamSnapshot()
            task = asyncio.create_task(self._consume_stream(stream, snap))

            got_first = await self._wait_for_first_output(snap)
            if not got_first:
                task.cancel()
                return ScenarioResult(
                    "T2: cancel_idempotency",
                    False,
                    "no output received before cancel window",
                )

            await self._cancel(rid, response)
            second_cancel_ok = True
            second_cancel_err = ""
            try:
                await self._cancel(rid, response)
            except Exception as exc:
                second_cancel_ok = False
                second_cancel_err = repr(exc)

            ended = await self._wait_for_stream_end(task)
            recovery_ok, recovery_msg = await self._verify_recovery()

            # engine 侧验证
            method = (
                "enqueue_batch" if response.enqueued_by_master else "generate_stream"
            )
            engine_recv, recv_detail = await self._verify_engine_received(rid, method)
            engine_cancelled, cancel_detail = await self._verify_engine_cancelled(rid)

            # batch 路径验证 inflight 清理
            inflight_ok = True
            inflight_detail = "N/A"
            if response.enqueued_by_master:
                inflight_ok, inflight_detail = await self._verify_inflight_clean(
                    timeout_s=10.0
                )

            passed = second_cancel_ok and ended and recovery_ok
            detail = (
                f"second_cancel_ok={second_cancel_ok} {second_cancel_err}, "
                f"stream_terminated={ended}, "
                f"engine_recv={engine_recv}({recv_detail}), "
                f"engine_cancelled={engine_cancelled}({cancel_detail}), "
                f"inflight_clean={inflight_ok}({inflight_detail}), "
                f"recovery={recovery_msg}"
            )
            return ScenarioResult(
                "T2: cancel_idempotency",
                passed,
                detail,
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "T2: cancel_idempotency",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- Scenario T3: multi_request_isolation -----------------------------

    async def test_multi_request_isolation(self) -> ScenarioResult:
        start = time.monotonic()
        rids = [self._next_request_id() for _ in range(3)]
        cancel_rid = rids[1]  # B
        try:
            # Schedule all three concurrently
            responses = await asyncio.gather(*[self._schedule_auto(r) for r in rids])
            for i, resp in enumerate(responses):
                if resp.code != 200 or not resp.success:
                    return ScenarioResult(
                        "T3: multi_request_isolation",
                        False,
                        f"schedule failed for rid={rids[i]}: " f"{resp.error_message}",
                    )

            # Start streams for all three
            streams_snaps: list[tuple[int, asyncio.Task, StreamSnapshot]] = []
            for rid, resp in zip(rids, responses):
                input_pb = (
                    None if resp.enqueued_by_master else self._build_generate_input(rid)
                )
                stream = await self._start_stream(resp, rid, input_pb=input_pb)
                snap = StreamSnapshot()
                task = asyncio.create_task(self._consume_stream(stream, snap))
                streams_snaps.append((rid, task, snap))

            # Wait for all to receive first output
            first_results = await asyncio.gather(
                *[
                    self._wait_for_first_output(snap, timeout_s=15.0)
                    for _, _, snap in streams_snaps
                ]
            )
            if not all(first_results):
                for _, task, _ in streams_snaps:
                    task.cancel()
                return ScenarioResult(
                    "T3: multi_request_isolation",
                    False,
                    "not all requests received first output",
                )

            # Cancel B
            await self._cancel(cancel_rid, responses[1])

            # Wait for B's stream to end
            b_idx = 1
            b_ended = await self._wait_for_stream_end(streams_snaps[b_idx][1])

            # Wait for A and C to complete
            a_complete = await self._wait_for_stream_end(
                streams_snaps[0][1], timeout_s=30.0
            )
            c_complete = await self._wait_for_stream_end(
                streams_snaps[2][1], timeout_s=30.0
            )

            a_snap = streams_snaps[0][2]
            c_snap = streams_snaps[2][2]
            b_snap = streams_snaps[b_idx][2]

            recovery_ok, recovery_msg = await self._verify_recovery()

            # engine 侧验证（对被 cancel 的请求 B）
            method = (
                "enqueue_batch"
                if responses[1].enqueued_by_master
                else "generate_stream"
            )
            engine_recv, recv_detail = await self._verify_engine_received(
                cancel_rid, method
            )
            engine_cancelled, cancel_detail = await self._verify_engine_cancelled(
                cancel_rid
            )

            # batch 路径验证 inflight 清理
            inflight_ok = True
            inflight_detail = "N/A"
            if responses[1].enqueued_by_master:
                inflight_ok, inflight_detail = await self._verify_inflight_clean(
                    timeout_s=10.0
                )

            passed = (
                b_ended
                and a_complete
                and a_snap.completed
                and c_complete
                and c_snap.completed
                and not b_snap.completed
                and recovery_ok
            )
            detail = (
                f"A_completed={a_snap.completed}(outputs={len(a_snap.outputs)}), "
                f"B_cancelled={b_ended}(completed={b_snap.completed}), "
                f"C_completed={c_snap.completed}(outputs={len(c_snap.outputs)}), "
                f"engine_recv={engine_recv}({recv_detail}), "
                f"engine_cancelled={engine_cancelled}({cancel_detail}), "
                f"inflight_clean={inflight_ok}({inflight_detail}), "
                f"recovery={recovery_msg}"
            )
            return ScenarioResult(
                "T3: multi_request_isolation",
                passed,
                detail,
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "T3: multi_request_isolation",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- Scenario T4: cancel_after_completion -----------------------------

    async def test_cancel_after_completion(self) -> ScenarioResult:
        rid = self._next_request_id()
        start = time.monotonic()
        try:
            # Use very short output so the request completes quickly
            response = await self._schedule_auto(rid, output_len=1)
            if response.code != 200 or not response.success:
                return ScenarioResult(
                    "T4: cancel_after_completion",
                    False,
                    f"schedule failed: {response.error_message}",
                )
            input_pb = (
                None if response.enqueued_by_master else self._build_generate_input(rid)
            )
            stream = await self._start_stream(response, rid, input_pb=input_pb)
            snap = StreamSnapshot()
            task = asyncio.create_task(self._consume_stream(stream, snap))

            # Wait for completion
            deadline = time.monotonic() + 30.0
            while not snap.completed and time.monotonic() < deadline:
                await asyncio.sleep(0.05)

            if not snap.completed:
                task.cancel()
                return ScenarioResult(
                    "T4: cancel_after_completion",
                    False,
                    "request did not complete before timeout",
                )

            # Cancel after completion
            cancel_ok = True
            cancel_err = ""
            try:
                await self._cancel(rid, response)
            except Exception as exc:
                cancel_ok = False
                cancel_err = repr(exc)

            # Stream should already be done
            await self._wait_for_stream_end(task, timeout_s=2.0)
            recovery_ok, recovery_msg = await self._verify_recovery()

            # engine 侧验证
            method = (
                "enqueue_batch" if response.enqueued_by_master else "generate_stream"
            )
            engine_recv, recv_detail = await self._verify_engine_received(rid, method)
            engine_cancelled, cancel_detail = await self._verify_engine_cancelled(rid)

            # batch 路径验证 inflight 清理
            inflight_ok = True
            inflight_detail = "N/A"
            if response.enqueued_by_master:
                inflight_ok, inflight_detail = await self._verify_inflight_clean(
                    timeout_s=10.0
                )

            passed = cancel_ok and recovery_ok
            return ScenarioResult(
                "T4: cancel_after_completion",
                passed,
                f"cancel_ok={cancel_ok} {cancel_err}, "
                f"completed={snap.completed}, "
                f"engine_recv={engine_recv}({recv_detail}), "
                f"engine_cancelled={engine_cancelled}({cancel_detail}), "
                f"inflight_clean={inflight_ok}({inflight_detail}), "
                f"recovery={recovery_msg}",
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "T4: cancel_after_completion",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- Scenario T5: cancel_nonexistent ---------------------------------

    async def test_cancel_nonexistent(self) -> ScenarioResult:
        start = time.monotonic()
        try:
            fake_rid = 99999
            cancel_ok = True
            cancel_err = ""
            try:
                await self._cancel(fake_rid)
            except Exception as exc:
                cancel_ok = False
                cancel_err = repr(exc)
            passed = cancel_ok
            return ScenarioResult(
                "T5: cancel_nonexistent",
                passed,
                f"cancel(rid={fake_rid}) ok={cancel_ok} {cancel_err}, "
                f"engine_verify=N/A (nonexistent request)",
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "T5: cancel_nonexistent",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- Scenario T6: cancel_at_prefill_vs_decode -------------------------

    async def test_cancel_at_prefill_vs_decode(self) -> ScenarioResult:
        start = time.monotonic()
        try:
            # Request A: cancel in prefill phase (before any output)
            rid_a = self._next_request_id()
            resp_a = await self._schedule_auto(rid_a)
            if resp_a.code != 200 or not resp_a.success:
                return ScenarioResult(
                    "T6: cancel_at_prefill_vs_decode",
                    False,
                    f"schedule A failed: {resp_a.error_message}",
                )
            input_pb_a = (
                None if resp_a.enqueued_by_master else self._build_generate_input(rid_a)
            )
            stream_a = await self._start_stream(resp_a, rid_a, input_pb=input_pb_a)
            snap_a = StreamSnapshot()
            task_a = asyncio.create_task(self._consume_stream(stream_a, snap_a))

            # Wait ~100ms — should be in prefill phase (no output yet)
            await asyncio.sleep(0.1)
            a_in_prefill = not snap_a.first_received
            await self._cancel(rid_a, resp_a)
            a_ended = await self._wait_for_stream_end(task_a)

            # Request B: cancel in decode phase (after first output)
            rid_b = self._next_request_id()
            resp_b = await self._schedule_auto(rid_b)
            if resp_b.code != 200 or not resp_b.success:
                return ScenarioResult(
                    "T6: cancel_at_prefill_vs_decode",
                    False,
                    f"schedule B failed: {resp_b.error_message}",
                )
            input_pb_b = (
                None if resp_b.enqueued_by_master else self._build_generate_input(rid_b)
            )
            stream_b = await self._start_stream(resp_b, rid_b, input_pb=input_pb_b)
            snap_b = StreamSnapshot()
            task_b = asyncio.create_task(self._consume_stream(stream_b, snap_b))

            # Wait until first output (decode phase)
            b_got_first = await self._wait_for_first_output(snap_b)
            if not b_got_first:
                task_b.cancel()
                return ScenarioResult(
                    "T6: cancel_at_prefill_vs_decode",
                    False,
                    "B never received first output (decode phase)",
                )
            await self._cancel(rid_b, resp_b)
            b_ended = await self._wait_for_stream_end(task_b)

            recovery_ok, recovery_msg = await self._verify_recovery()

            # engine 侧验证（对请求 A 和 B）
            method_a = (
                "enqueue_batch" if resp_a.enqueued_by_master else "generate_stream"
            )
            method_b = (
                "enqueue_batch" if resp_b.enqueued_by_master else "generate_stream"
            )
            engine_recv_a, recv_detail_a = await self._verify_engine_received(
                rid_a, method_a
            )
            engine_cancelled_a, cancel_detail_a = await self._verify_engine_cancelled(
                rid_a
            )
            engine_recv_b, recv_detail_b = await self._verify_engine_received(
                rid_b, method_b
            )
            engine_cancelled_b, cancel_detail_b = await self._verify_engine_cancelled(
                rid_b
            )

            # batch 路径验证 inflight 清理
            inflight_ok = True
            inflight_detail = "N/A"
            if resp_a.enqueued_by_master or resp_b.enqueued_by_master:
                inflight_ok, inflight_detail = await self._verify_inflight_clean(
                    timeout_s=10.0
                )

            passed = a_ended and b_ended and a_in_prefill and recovery_ok
            detail = (
                f"A_prefill_phase={a_in_prefill}, A_terminated={a_ended}, "
                f"A_outputs={len(snap_a.outputs)}, "
                f"B_decode_phase={b_got_first}, B_terminated={b_ended}, "
                f"B_outputs={len(snap_b.outputs)}, "
                f"engine_recv_A={engine_recv_a}, engine_cancel_A={engine_cancelled_a}, "
                f"engine_recv_B={engine_recv_b}, engine_cancel_B={engine_cancelled_b}, "
                f"inflight_clean={inflight_ok}({inflight_detail}), "
                f"recovery={recovery_msg}"
            )
            return ScenarioResult(
                "T6: cancel_at_prefill_vs_decode",
                passed,
                detail,
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "T6: cancel_at_prefill_vs_decode",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- Runner ----------------------------------------------------------

    async def run_all(self) -> int:
        scenarios = [
            self.test_basic_cancel,
            self.test_cancel_idempotency,
            self.test_multi_request_isolation,
            self.test_cancel_after_completion,
            self.test_cancel_nonexistent,
            self.test_cancel_at_prefill_vs_decode,
        ]
        print("=" * 70)
        print("FlexLB Cancel Smoke Test")
        print(f"  master: {self._master_target()}")
        print(f"  schedule_mode: {self.args.schedule_mode}")
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

        # Summary
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
    parser.add_argument("--request-id-base", type=int, default=10000)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    test = CancelSmokeTest(args)
    try:
        exit_code = await test.run_all()
    finally:
        await test.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
