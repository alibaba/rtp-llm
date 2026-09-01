#!/usr/bin/env python3
"""End-to-end smoke for RUNNING Decode priority preemption.

This scenario complements client-cancel coverage (now in the
``flexlb_ft/`` framework): it proves that a P70
incoming request evicts a P30 request already running on Decode through the
Master -> original-Prefill weak-Cancel protocol.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass

from flexlb_smoke_base import FlexLBSmokeBase, StreamSnapshot


@dataclass
class VictimTerminal:
    grpc_code: object | None = None
    raw_error_code: int | None = None
    error: str | None = None


class PriorityPreemptionSmoke(FlexLBSmokeBase):
    POLL_INTERVAL_S = 0.01

    async def _master_inflight(self) -> dict:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://127.0.0.1:{self.args.flexlb_http_port}"
                "/rtp_llm/inflight_status"
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def _wait_master_inflight(
        self, predicate, description: str, timeout_s: float = 5.0
    ) -> dict:
        deadline = time.monotonic() + timeout_s
        last = None
        while time.monotonic() < deadline:
            last = await self._master_inflight()
            if predicate(last):
                return last
            await asyncio.sleep(self.POLL_INTERVAL_S)
        raise AssertionError(f"timeout waiting for {description}; last_inflight={last}")

    @staticmethod
    def _engine(snapshot: dict, grpc_target: str) -> dict | None:
        return next(
            (
                engine
                for engine in snapshot.get("engines", [])
                if engine.get("grpc_addr") == grpc_target
            ),
            None,
        )

    @staticmethod
    def _lifecycle(snapshot: dict, request_id: int) -> list[tuple[dict, dict]]:
        result = []
        for engine in snapshot.get("engines", []):
            lifecycle = engine.get("request_lifecycle", {}).get(str(request_id))
            if lifecycle is not None:
                result.append((engine, lifecycle))
        return result

    async def _wait_snapshot(self, predicate, description: str, timeout_s: float = 5.0):
        deadline = time.monotonic() + timeout_s
        last = None
        while time.monotonic() < deadline:
            last = await self._get_snapshot()
            if predicate(last):
                return last
            await asyncio.sleep(self.POLL_INTERVAL_S)
        raise AssertionError(f"timeout waiting for {description}; last_snapshot={last}")

    async def _worker_status(self, target: str):
        stub = self.pb2_grpc.RpcServiceStub(await self._channel(target))
        return await stub.GetWorkerStatus(
            self.pb2.StatusVersionPB(latest_finished_version=-1), timeout=3.0
        )

    async def _wait_prefill_progress(
        self, target: str, request_id: int, progress: int, timeout_s: float
    ):
        deadline = time.monotonic() + timeout_s
        last = None
        while time.monotonic() < deadline:
            last = await self._worker_status(target)
            tasks = (
                last.running_task_info
                if progress == self.pb2.PRIORITY_PREEMPTION_CANCELING
                else last.finished_task_list
            )
            matches = [
                task
                for task in tasks
                if int(task.request_id) == request_id
                and task.priority_preemption_progress == progress
            ]
            if matches:
                return last, matches
            await asyncio.sleep(self.POLL_INTERVAL_S)
        raise AssertionError(
            f"timeout waiting for Prefill progress={progress}; last_status={last}"
        )

    async def _consume_victim(self, stream, terminal: VictimTerminal) -> None:
        import grpc

        try:
            async for _ in stream:
                pass
        except grpc.aio.AioRpcError as exc:
            terminal.grpc_code = exc.code()
            terminal.error = exc.details()
            metadata = dict(exc.trailing_metadata() or ())
            encoded_details = metadata.get("grpc-status-details-bin")
            if encoded_details is not None:
                details = self.pb2.ErrorDetailsPB.FromString(encoded_details)
                terminal.raw_error_code = int(details.error_code)
        except Exception as exc:  # pragma: no cover - diagnostic branch
            terminal.error = repr(exc)

    @staticmethod
    def _cancel_counts(snapshot: dict) -> dict[str, int]:
        return {
            engine["name"]: int(engine.get("rpc_counts", {}).get("cancel", 0))
            for engine in snapshot.get("engines", [])
        }

    @staticmethod
    def _rpc_total(snapshot: dict, method: str) -> int:
        return sum(
            int(engine.get("rpc_counts", {}).get(method, 0))
            for engine in snapshot.get("engines", [])
        )

    def _assert_queue_request_accounting(
        self, inflight: dict, *, require_active: bool
    ) -> None:
        prefill_endpoints = inflight.get("prefill_endpoints", [])
        request_counts = [
            int(endpoint.get("inflight_requests", 0)) for endpoint in prefill_endpoints
        ]
        route_counts = [
            int(endpoint.get("inflight_route_requests", 0))
            for endpoint in prefill_endpoints
        ]
        assert all(
            int(endpoint.get("inflight_batches", 0)) == 0
            for endpoint in prefill_endpoints
        ), f"QUEUE delivery created a Prefill batch ledger: {inflight}"
        if require_active:
            assert (
                sum(request_counts) >= 1
            ), f"Master did not account the active Prefill request: {inflight}"
            assert (
                sum(route_counts) >= 1
            ), f"Master did not account the active route request: {inflight}"
        if self.args.prefill_request_cap > 0:
            assert all(
                count <= self.args.prefill_request_cap for count in route_counts
            ), (
                "Prefill route-request cap exceeded: "
                f"cap={self.args.prefill_request_cap}, inflight={inflight}"
            )

    @staticmethod
    def _all_master_accounting_clean(inflight: dict) -> bool:
        prefill_endpoints = inflight.get("prefill_endpoints", [])
        decode_endpoints = inflight.get("decode_endpoints", [])
        return (
            bool(prefill_endpoints)
            and bool(decode_endpoints)
            and int(inflight.get("scheduler_inflight", 0)) == 0
            and all(
                int(endpoint.get(field, 0)) == 0
                for endpoint in prefill_endpoints
                for field in (
                    "inflight_batches",
                    "inflight_requests",
                    "inflight_route_requests",
                )
            )
            and all(
                int(endpoint.get("inflight_requests", 0)) == 0
                for endpoint in decode_endpoints
            )
        )

    async def run(self) -> tuple[bool, str]:
        import grpc

        low_rid = self._next_request_id()
        high_rid = self._next_request_id()
        low_stream_task: asyncio.Task | None = None
        high_stream_task: asyncio.Task | None = None
        high_schedule_task: asyncio.Task | None = None
        selected_decode_name = ""
        try:
            delivery_baseline = await self._get_snapshot()
            low_block_keys = [low_rid * 100 + 1]
            low_response = await self._schedule(
                low_rid,
                input_len=512,
                output_len=200,
                priority=30,
                block_keys=low_block_keys,
            )
            assert low_response.success and low_response.code == 200, (
                f"low schedule failed: code={low_response.code} "
                f"message={low_response.error_message}"
            )
            if self.args.schedule_mode == "queue":
                assert (
                    not low_response.enqueued_by_master
                ), "QUEUE low request was enqueued by Master"
                low_route_inflight = await self._master_inflight()
                self._assert_queue_request_accounting(
                    low_route_inflight, require_active=True
                )
            prefill_target = self._role_addr(low_response, "PREFILL")
            decode_target = self._role_addr(low_response, "DECODE")
            assert prefill_target, "low response has no original Prefill route"
            assert decode_target, "low response has no Decode route"

            low_stream = await self._start_stream(
                low_response,
                low_rid,
                input_pb=self._build_generate_input(
                    low_rid,
                    input_len=512,
                    output_len=200,
                    block_keys=low_block_keys,
                ),
            )
            victim_terminal = VictimTerminal()
            low_stream_task = asyncio.create_task(
                self._consume_victim(low_stream, victim_terminal)
            )

            running_snapshot = await self._wait_snapshot(
                lambda snap: (
                    (engine := self._engine(snap, decode_target)) is not None
                    and engine.get("request_lifecycle", {})
                    .get(str(low_rid), {})
                    .get("end_state")
                    == "running"
                    and int(engine.get("active_kv_tokens", 0)) > 0
                ),
                "low-priority victim RUNNING on selected Decode",
                timeout_s=8.0,
            )
            selected_decode = self._engine(running_snapshot, decode_target)
            assert selected_decode is not None
            selected_decode_name = selected_decode["name"]
            low_kv_before = int(selected_decode["active_kv_tokens"])

            # No cleanup-delay injection here: cancel_cleanup_delay_ms was a
            # feature of the deleted Python mock engine; the Java engine only
            # parses the four boolean fault flags and silently ignores other
            # config keys. The weak-ACK observation window relies on the
            # WorkerStatus 20ms polling cadence instead.

            # WorkerStatus sync runs every 20ms.  A short stability interval
            # ensures Master has ingested the Decode RUNNING task, so this is
            # specifically a RUNNING-victim test rather than a reserved race.
            await asyncio.sleep(0.15)
            baseline_snapshot = await self._get_snapshot()
            baseline_cancel = self._cancel_counts(baseline_snapshot)
            inflight_before = await self._master_inflight()
            scheduler_inflight_before = int(
                inflight_before.get("scheduler_inflight", 0)
            )
            decode_inflight_before = sum(
                int(endpoint.get("inflight_requests", 0))
                for endpoint in inflight_before.get("decode_endpoints", [])
            )
            assert (
                scheduler_inflight_before >= 1
            ), f"Master did not account the low victim: {inflight_before}"
            if self.args.schedule_mode == "queue":
                self._assert_queue_request_accounting(
                    inflight_before, require_active=False
                )

            high_block_keys = [high_rid * 100 + 1]
            high_schedule_task = asyncio.create_task(
                self._schedule(
                    high_rid,
                    input_len=512,
                    output_len=2,
                    priority=70,
                    block_keys=high_block_keys,
                )
            )

            canceling_status, canceling_matches = await self._wait_prefill_progress(
                prefill_target,
                low_rid,
                self.pb2.PRIORITY_PREEMPTION_CANCELING,
                timeout_s=5.0,
            )
            assert len(canceling_matches) == 1
            assert (
                not high_schedule_task.done()
            ), "high request completed Schedule during weak ACK before typed CANCELED"
            assert not any(
                int(task.request_id) == low_rid
                and task.priority_preemption_progress
                == self.pb2.PRIORITY_PREEMPTION_CANCELED
                for task in canceling_status.finished_task_list
            ), "Prefill published CANCELED in the weak-ACK window"

            weak_snapshot = await self._get_snapshot()
            weak_decode = self._engine(weak_snapshot, decode_target)
            assert weak_decode is not None
            assert (
                weak_decode.get("request_lifecycle", {})
                .get(str(low_rid), {})
                .get("end_state")
                == "running"
            ), "Decode released the victim before the Prefill completion fence"
            assert (
                int(weak_decode.get("active_kv_tokens", 0)) == low_kv_before
            ), "Decode KV accounting changed during weak ACK"
            assert not self._lifecycle(
                weak_snapshot, high_rid
            ), "high request reached an engine before typed CANCELED"
            weak_cancel = self._cancel_counts(weak_snapshot)
            original_prefill_name = self._engine(weak_snapshot, prefill_target)["name"]
            for name, count in weak_cancel.items():
                delta = count - baseline_cancel.get(name, 0)
                expected = 1 if name == original_prefill_name else 0
                assert delta == expected, (
                    "Master Cancel routing mismatch: "
                    f"engine={name} delta={delta} expected={expected}"
                )
            inflight_during = await self._master_inflight()
            scheduler_inflight_during = int(
                inflight_during.get("scheduler_inflight", 0)
            )
            decode_inflight_during = sum(
                int(endpoint.get("inflight_requests", 0))
                for endpoint in inflight_during.get("decode_endpoints", [])
            )
            assert scheduler_inflight_during >= scheduler_inflight_before, (
                "Master released victim accounting on weak ACK: "
                f"before={inflight_before}, during={inflight_during}; "
                f"decode_shadow_before={decode_inflight_before}, "
                f"decode_shadow_during={decode_inflight_during}"
            )
            if self.args.schedule_mode == "queue":
                self._assert_queue_request_accounting(
                    inflight_during, require_active=False
                )

            _, canceled_matches = await self._wait_prefill_progress(
                prefill_target,
                low_rid,
                self.pb2.PRIORITY_PREEMPTION_CANCELED,
                timeout_s=5.0,
            )
            assert (
                len(canceled_matches) == 1
            ), f"expected one typed CANCELED terminal, got {len(canceled_matches)}"
            terminal_task = canceled_matches[0]
            assert int(terminal_task.error_info.error_code) == 8429
            canceled_end_ms = int(terminal_task.end_time_ms)

            high_response = await asyncio.wait_for(high_schedule_task, timeout=5.0)
            assert high_response.success and high_response.code == 200, (
                f"high schedule failed: code={high_response.code} "
                f"message={high_response.error_message}"
            )
            if self.args.schedule_mode == "queue":
                assert (
                    not high_response.enqueued_by_master
                ), "QUEUE high request was enqueued by Master"
                high_route_inflight = await self._master_inflight()
                self._assert_queue_request_accounting(
                    high_route_inflight, require_active=True
                )
                high_stream = await self._start_stream(
                    high_response,
                    high_rid,
                    input_pb=self._build_generate_input(
                        high_rid,
                        input_len=512,
                        output_len=2,
                        block_keys=high_block_keys,
                    ),
                )
                high_snap = StreamSnapshot()
                high_stream_task = asyncio.create_task(
                    self._consume_stream(high_stream, high_snap)
                )
            dispatched_snapshot = await self._wait_snapshot(
                lambda snap: any(
                    engine.get("role") == "decode"
                    and str(high_rid) in engine.get("request_lifecycle", {})
                    for engine in snap.get("engines", [])
                ),
                "high-priority request dispatch after victim CANCELED",
                timeout_s=5.0,
            )
            high_decode_entries = [
                (engine, lifecycle)
                for engine, lifecycle in self._lifecycle(dispatched_snapshot, high_rid)
                if engine.get("role") == "decode"
            ]
            assert len(high_decode_entries) == 1
            high_running_ms = int(high_decode_entries[0][1].get("running_ms", 0))
            assert high_running_ms >= canceled_end_ms, (
                "high Decode dispatch preceded the typed CANCELED fence: "
                f"high_running_ms={high_running_ms}, canceled_end_ms={canceled_end_ms}"
            )

            await asyncio.wait_for(low_stream_task, timeout=5.0)
            assert (
                victim_terminal.grpc_code == grpc.StatusCode.RESOURCE_EXHAUSTED
            ), f"victim gRPC status mismatch: {victim_terminal}"
            assert (
                victim_terminal.raw_error_code == 8429
            ), f"victim raw engine code mismatch: {victim_terminal}"

            # Retry the same priority Cancel directly against the original
            # Prefill tombstone.  It must ACK ACCEPTED and must not republish.
            prefill_stub = self.pb2_grpc.RpcServiceStub(
                await self._channel(prefill_target)
            )
            retry_ack = await prefill_stub.Cancel(
                self.pb2.CancelRequestPB(request_id=low_rid), timeout=3.0
            )
            assert retry_ack.status == self.pb2.CANCEL_STATUS_ACCEPTED
            retry_status = await self._worker_status(prefill_target)
            retry_terminals = [
                task
                for task in retry_status.finished_task_list
                if int(task.request_id) == low_rid
                and task.priority_preemption_progress
                == self.pb2.PRIORITY_PREEMPTION_CANCELED
                and int(task.error_info.error_code) == 8429
            ]
            assert (
                len(retry_terminals) == 1
            ), "repeated Cancel republished typed CANCELED+8429"

            final_snapshot = await self._get_snapshot()
            final_cancel = self._cancel_counts(final_snapshot)
            for name, count in final_cancel.items():
                delta = count - baseline_cancel.get(name, 0)
                expected = 2 if name == original_prefill_name else 0
                assert delta == expected, (
                    "Cancel was delivered to a non-original-Prefill or Decode: "
                    f"engine={name} delta={delta} expected={expected}"
                )

            if high_stream_task is None:
                high_stream = await self._start_stream(high_response, high_rid)
                high_snap = StreamSnapshot()
                high_stream_task = asyncio.create_task(
                    self._consume_stream(high_stream, high_snap)
                )
            await asyncio.wait_for(high_stream_task, timeout=5.0)
            assert (
                high_snap.completed and high_snap.error is None
            ), f"high request did not complete normally: {high_snap}"

            delivery_detail = ""
            if self.args.schedule_mode == "queue":
                final_inflight = await self._wait_master_inflight(
                    self._all_master_accounting_clean,
                    "QUEUE request ledgers and scheduler accounting to drain",
                    timeout_s=8.0,
                )
                final_snapshot = await self._get_snapshot()
                enqueue_delta = self._rpc_total(
                    final_snapshot, "enqueue_batch"
                ) - self._rpc_total(delivery_baseline, "enqueue_batch")
                generate_delta = self._rpc_total(
                    final_snapshot, "generate_stream"
                ) - self._rpc_total(delivery_baseline, "generate_stream")
                assert enqueue_delta == 0, (
                    "QUEUE delivery called EnqueueBatch: "
                    f"delta={enqueue_delta}, snapshot={final_snapshot}"
                )
                assert generate_delta >= 2, (
                    "QUEUE frontend delivery did not call GenerateStream for both requests: "
                    f"delta={generate_delta}, snapshot={final_snapshot}"
                )
                self._assert_queue_request_accounting(
                    final_inflight, require_active=False
                )
                delivery_detail = (
                    f"; route_delivery=2; enqueue_batch_delta={enqueue_delta}; "
                    f"generate_stream_delta={generate_delta}; ledgers=clean"
                )

            return True, (
                f"victim={low_rid} P30 RUNNING -> CANCELED+8429; "
                f"incoming={high_rid} P70 dispatched after fence; "
                f"cancel_route={original_prefill_name}; repeated_ack=ACCEPTED"
                f"{delivery_detail}"
            )
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        finally:
            if selected_decode_name:
                try:
                    await self._clear_inject(selected_decode_name)
                except Exception:
                    pass
            for task in (low_stream_task, high_stream_task, high_schedule_task):
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-ip", default="127.0.0.1")
    parser.add_argument("--master-http-port", type=int, default=18080)
    parser.add_argument("--mock-http-port", type=int, default=55150)
    parser.add_argument("--flexlb-http-port", type=int, default=18080)
    parser.add_argument("--request-id-base", type=int, default=20000)
    parser.add_argument(
        "--schedule-mode", choices=["auto", "batch", "direct", "queue"], default="batch"
    )
    parser.add_argument(
        "--prefill-request-cap",
        type=int,
        default=0,
        help="Configured per-Prefill route-request cap (0 disables the cap)",
    )
    args = parser.parse_args()
    if args.prefill_request_cap < 0:
        parser.error("--prefill-request-cap must be non-negative")
    return args


async def main() -> None:
    args = parse_args()
    smoke = PriorityPreemptionSmoke(args)
    started = time.monotonic()
    try:
        passed, detail = await smoke.run()
    finally:
        await smoke.close()
    status = "PASS" if passed else "FAIL"
    print(
        f"{status} priority_preemption_running_victim "
        f"({time.monotonic() - started:.2f}s): {detail}",
        flush=True,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
