#!/usr/bin/env python3
"""Anomaly path smoke tests for FlexLB: cancel, timeout, worker fail.

Connects to a running FlexLB master and mock engine cluster.  Exercises
three anomaly-path scenarios:

  E1  cancel_path_test   - Schedule -> first output -> cancel -> stream
                           terminates within 5s -> recovery ok.
  E2  timeout_test        - Inject no_respond on all prefill workers -> request
                           errors -> clear inject -> recovery ok.
  E3  worker_fail_test    - Inject enqueue_error on all prefill workers -> request
                           errors -> clear inject -> recovery ok.

Usage:
    python3 anomaly_smoke.py --master-ip 127.0.0.1 \\
        --master-http-port 18080 --mock-http-port 55150 \\
        --schedule-mode batch
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from flexlb_smoke_base import FlexLBSmokeBase, ScenarioResult, StreamSnapshot


class AnomalySmokeTest(FlexLBSmokeBase):
    """Anomaly path smoke tests: cancel, timeout, worker failure."""

    # Test-specific timeouts (seconds)
    TIMEOUT_WAIT_S = 5.0
    STREAM_TIMEOUT_S = 10.0
    WORKER_RECOVERY_WAIT_S = 3.0

    # -- Helpers ----------------------------------------------------------

    async def _schedule_auto(self, request_id: int, **kwargs):
        """Schedule with the configured schedule_mode from CLI args."""
        kwargs.setdefault("schedule_mode", self.args.schedule_mode)
        return await self._schedule(request_id, **kwargs)

    async def _inject_all_prefill(self, config: dict) -> list[str]:
        """Inject error config on every prefill worker, return their names.

        In direct/queue mode the master may route to any prefill worker, so
        we must inject on all of them to guarantee the error is observed.
        """
        snapshot = await self._get_snapshot()
        prefill_names = [
            e["name"] for e in snapshot["engines"] if e["role"] == "prefill"
        ]
        for name in prefill_names:
            await self._inject(name, config)
        return prefill_names

    async def _clear_all_prefill_inject(self, names: list[str]) -> None:
        """Clear injection on the given prefill workers."""
        for name in names:
            try:
                await self._clear_inject(name)
            except Exception:
                pass

    # -- E1: cancel_path_test --------------------------------------------

    async def test_cancel(self) -> ScenarioResult:
        """Schedule -> first output -> cancel -> stream terminates -> recovery."""
        start = time.monotonic()
        rid = self._next_request_id()
        try:
            response = await self._schedule_auto(rid)
            if response.code != 200 or not response.success:
                return ScenarioResult(
                    "E1: cancel_path_test",
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

            got_first = await self._wait_for_first_output(snap)
            if not got_first:
                task.cancel()
                return ScenarioResult(
                    "E1: cancel_path_test",
                    False,
                    "no output received before cancel window",
                    time.monotonic() - start,
                )

            cancel_at = time.monotonic()
            await self._cancel(rid, response)
            ended = await self._wait_for_stream_end(task)
            cancel_latency = time.monotonic() - cancel_at

            recovery_ok, recovery_msg = await self._verify_recovery()

            # inflight 清理验证（仅 batch 路径有意义）
            inflight_ok = True
            inflight_detail = "N/A (non-batch path)"
            if self.args.schedule_mode == "batch":
                inflight_ok, inflight_detail = await self._verify_inflight_clean(
                    timeout_s=10.0
                )

            passed = ended and recovery_ok
            return ScenarioResult(
                "E1: cancel_path_test",
                passed,
                f"cancel_latency={cancel_latency:.3f}s, "
                f"stream_terminated={ended}, "
                f"outputs={len(snap.outputs)}, "
                f"inflight_clean={inflight_ok}({inflight_detail}), "
                f"recovery={recovery_msg}",
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "E1: cancel_path_test",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- E2: timeout_test -------------------------------------------------

    async def test_timeout(self) -> ScenarioResult:
        """Inject no_respond -> request errors -> clear -> recovery."""
        start = time.monotonic()
        rid = self._next_request_id()
        error_observed = False
        error_detail = "no error observed"
        injected_names: list[str] = []
        response = None
        try:
            injected_names = await self._inject_all_prefill({"no_respond": True})

            try:
                response = await self._schedule_auto(rid)
                if response.code != 200 or not response.success:
                    error_observed = True
                    error_detail = f"schedule error: {response.error_message}"
                else:
                    input_pb = (
                        None
                        if response.enqueued_by_master
                        else self._build_generate_input(rid)
                    )
                    stream = await self._start_stream(response, rid, input_pb=input_pb)
                    snap = StreamSnapshot()
                    task = asyncio.create_task(self._consume_stream(stream, snap))
                    try:
                        await asyncio.wait_for(task, timeout=self.TIMEOUT_WAIT_S)
                    except asyncio.TimeoutError:
                        task.cancel()
                        try:
                            await task
                        except (Exception, asyncio.CancelledError):
                            pass
                        error_observed = True
                        error_detail = (
                            f"stream timed out "
                            f"(no response within {self.TIMEOUT_WAIT_S}s)"
                        )
                    if snap.error:
                        error_observed = True
                        error_detail = f"stream error: {snap.error}"
            except Exception as exc:
                error_observed = True
                error_detail = f"exception: {exc!r}"
            finally:
                await self._clear_all_prefill_inject(injected_names)

            # Explicitly cancel the failed request to clean up server-side inflight.
            # In batch mode, the scheduler holds the request in its inflight map
            # even after the client-side stream times out. Without this cancel,
            # the decode KV reservation leaks until TTL eviction (300s).
            if response is not None and response.success:
                try:
                    await self._cancel(rid, response)
                except Exception:
                    pass

            # Master may need a few seconds to detect worker state change
            await asyncio.sleep(self.WORKER_RECOVERY_WAIT_S)

            recovery_ok, recovery_msg = await self._verify_recovery()

            # inflight 清理验证（仅 batch 路径有意义）
            inflight_ok = True
            inflight_detail = "N/A (non-batch path)"
            if self.args.schedule_mode == "batch":
                inflight_ok, inflight_detail = await self._verify_inflight_clean(
                    timeout_s=10.0
                )

            passed = error_observed and recovery_ok
            return ScenarioResult(
                "E2: timeout_test",
                passed,
                f"error_observed={error_observed} "
                f"({error_detail}), inflight_clean={inflight_ok}({inflight_detail}), "
                f"recovery={recovery_msg}",
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "E2: timeout_test",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- E3: worker_fail_test --------------------------------------------

    async def test_worker_fail(self) -> ScenarioResult:
        """Inject enqueue_error -> request errors -> clear -> recovery."""
        start = time.monotonic()
        rid = self._next_request_id()
        error_observed = False
        error_detail = "no error observed"
        injected_names: list[str] = []
        response = None
        try:
            injected_names = await self._inject_all_prefill({"enqueue_error": True})

            try:
                response = await self._schedule_auto(rid)
                if response.code != 200 or not response.success:
                    error_observed = True
                    error_detail = f"schedule error: {response.error_message}"
                else:
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
            finally:
                await self._clear_all_prefill_inject(injected_names)

            # Explicitly cancel the failed request to clean up server-side inflight.
            # In batch mode, the scheduler holds the request in its inflight map
            # even after the client-side stream errors. Without this cancel,
            # the decode KV reservation leaks until TTL eviction (300s).
            if response is not None and response.success:
                try:
                    await self._cancel(rid, response)
                except Exception:
                    pass

            # Master may need a few seconds to detect worker state change
            await asyncio.sleep(self.WORKER_RECOVERY_WAIT_S)

            recovery_ok, recovery_msg = await self._verify_recovery()

            # inflight 清理验证（仅 batch 路径有意义）
            inflight_ok = True
            inflight_detail = "N/A (non-batch path)"
            if self.args.schedule_mode == "batch":
                inflight_ok, inflight_detail = await self._verify_inflight_clean(
                    timeout_s=10.0
                )

            passed = error_observed and recovery_ok
            return ScenarioResult(
                "E3: worker_fail_test",
                passed,
                f"error_observed={error_observed} "
                f"({error_detail}), inflight_clean={inflight_ok}({inflight_detail}), "
                f"recovery={recovery_msg}",
                time.monotonic() - start,
            )
        except Exception as exc:
            return ScenarioResult(
                "E3: worker_fail_test",
                False,
                f"exception: {exc!r}",
                time.monotonic() - start,
            )

    # -- Runner ----------------------------------------------------------

    async def run_all(self) -> int:
        scenarios = [
            self.test_cancel,
            self.test_timeout,
            self.test_worker_fail,
        ]
        print("=" * 70)
        print("FlexLB Anomaly Smoke Test")
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
    parser.add_argument("--request-id-base", type=int, default=30000)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    test = AnomalySmokeTest(args)
    try:
        exit_code = await test.run_all()
    finally:
        await test.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
