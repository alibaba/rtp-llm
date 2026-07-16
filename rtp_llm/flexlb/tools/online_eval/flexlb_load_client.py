#!/usr/bin/env python3
"""Replay queries against a running SpringBoot flexlb-api and mock engines."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import time
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
from online_eval.mock_engine import encode_unique_key
from online_eval.proto_utils import ensure_proto_modules, ensure_schedule_proto_modules
from online_eval.report import write_markdown_report
from online_eval.rt_model import to_signed_int64
from online_eval.stats import load_balance_summary, summarize_latencies
from online_eval.trace_loader import (
    ReplayRequest,
    load_replay_requests,
    stable_request_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="query/trace JSONL")
    parser.add_argument("--flexlb-http-addr", default="127.0.0.1:7001")
    parser.add_argument("--flexlb-grpc-target", help="override flexlb gRPC target")
    parser.add_argument(
        "--output-dir", default="rtp_llm/flexlb/tools/online_eval/run/load_client"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="trace-time replay duration; with --loop, wall-clock timeout in seconds",
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        help="1=real inter-arrival, 10=10x faster, 0=as fast as possible",
    )
    parser.add_argument("--max-concurrency", type=int, default=16384)
    parser.add_argument(
        "--n-channels",
        type=int,
        default=int(os.environ.get("FLEXLB_N_CHANNELS", "8")),
        help="number of gRPC channels per target for channel pooling (default 8, "
        "env FLEXLB_N_CHANNELS)",
    )
    parser.add_argument(
        "--schedule-mode", choices=["auto", "batch", "direct", "queue"], default="batch"
    )
    parser.add_argument("--timeout-ms", type=int, default=120000)
    parser.add_argument(
        "--response-timeout",
        type=int,
        default=120,
        help="max seconds to wait for responses after sending completes (default 120)",
    )
    parser.add_argument("--sla-ttft-ms", type=float, default=500.0)
    parser.add_argument(
        "--zero-output-policy", choices=["skip", "one", "default100"], default="skip"
    )
    parser.add_argument(
        "--schedule-only",
        action="store_true",
        help="only call FlexLB Schedule; do not fetch engine stream",
    )
    parser.add_argument("--model", default="engine_service")
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--enable-fallback",
        action="store_true",
        help="on Schedule failure, try domain fallback direct to Engine",
    )
    parser.add_argument(
        "--endpoints-file",
        help="path to endpoints.json for prefill/decode engine addresses",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="loop trace replay until duration-s wall-clock or limit reached",
    )
    parser.add_argument(
        "--pushgateway-url",
        default="",
        help="pushgateway URL for metrics push (e.g. http://11.163.39.110:9091)",
    )
    return parser.parse_args()


class LoadClient:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pb2, self.pb2_grpc = ensure_proto_modules()
        self.schedule_pb2, self.schedule_pb2_grpc = ensure_schedule_proto_modules()
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.per_request_path = self.output_dir / "per_request.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.report_path = self.output_dir / "report.md"
        self._write_lock = asyncio.Lock()
        self._results: List[dict] = []
        self._channels: Dict[str, List[object]] = {}
        self._channel_rr: Dict[str, int] = {}
        self._n_channels: int = getattr(args, "n_channels", 8) or 8
        self._fallback_prefill_addrs: List[str] = []
        self._fallback_decode_addrs: List[str] = []
        self._fallback_prefill_rr = 0
        self._fallback_decode_rr = 0
        self._send_start: Optional[float] = None
        self._send_end: Optional[float] = None
        self._pushgateway_url: str = getattr(args, "pushgateway_url", "") or ""
        self._pushgateway_task: Optional[asyncio.Task] = None
        self._sent_count: int = 0
        self._actual_sent_count: int = 0
        self._inflight_count: int = 0
        if getattr(args, "endpoints_file", None):
            self._load_fallback_endpoints(args.endpoints_file)

    async def close(self) -> None:
        for pool in self._channels.values():
            for channel in pool:
                await channel.close()
        self._channels.clear()
        self._channel_rr.clear()

    async def run(self) -> None:
        # When loop is enabled, load all trace records (no duration/limit filter).
        # duration_s becomes a wall-clock timeout, limit becomes total sent cap.
        load_duration = 0.0 if self.args.loop else self.args.duration_s
        load_limit = 0 if self.args.loop else self.args.limit
        requests = load_replay_requests(
            self.args.trace,
            limit=load_limit,
            duration_s=load_duration,
            zero_output_policy=self.args.zero_output_policy,
            include_token_ids=True,
        )
        if not requests:
            raise RuntimeError("no replayable requests loaded")
        print(f"loaded {len(requests)} requests from {self.args.trace}", flush=True)
        if self.args.loop:
            print(
                f"loop mode: will replay trace repeatedly until "
                f"duration={self.args.duration_s}s or limit={self.args.limit}",
                flush=True,
            )

        self.per_request_path.write_text("", encoding="utf-8")
        _mc = (
            self.args.max_concurrency if self.args.max_concurrency > 0 else 999_999_999
        )
        sem = asyncio.Semaphore(_mc)
        started_at = time.monotonic()
        self._send_start = started_at
        # Start pushgateway metrics push loop
        if self._pushgateway_url:
            self._pushgateway_task = asyncio.create_task(self._pushgateway_loop())
            print(
                f"pushgateway metrics push enabled: {self._pushgateway_url}", flush=True
            )
        first_ts = requests[0].ts_ms
        last_ts = requests[-1].ts_ms
        trace_span_ms = max(last_ts - first_ts, 1)
        tasks: List[asyncio.Task] = []
        sent_count = 0
        loop_idx = 0

        while True:
            for req in requests:
                # Check wall-clock duration (loop mode)
                if self.args.loop and self.args.duration_s > 0:
                    if (time.monotonic() - started_at) >= self.args.duration_s:
                        break
                # Check total limit
                if self.args.limit > 0 and sent_count >= self.args.limit:
                    break

                # Calculate timing with loop offset
                if self.args.replay_speed > 0 and req.ts_ms > 0:
                    loop_offset_ms = loop_idx * trace_span_ms
                    due_s = (
                        (req.ts_ms - first_ts + loop_offset_ms)
                        / 1000.0
                        / self.args.replay_speed
                    )
                    sleep_s = due_s - (time.monotonic() - started_at)
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)

                # Check duration again after sleeping
                if self.args.loop and self.args.duration_s > 0:
                    if (time.monotonic() - started_at) >= self.args.duration_s:
                        break

                # For loop iterations > 0, create unique request to avoid ID conflicts
                loop_req = req
                if loop_idx > 0:
                    loop_req = self._make_loop_request(req, loop_idx, sent_count)

                tasks.append(
                    asyncio.create_task(self._handle_with_semaphore(loop_req, sem))
                )
                sent_count += 1
                self._sent_count = sent_count

            # Stop conditions
            if not self.args.loop:
                break
            if self.args.duration_s > 0 and (
                (time.monotonic() - started_at) >= self.args.duration_s
            ):
                break
            if self.args.limit > 0 and sent_count >= self.args.limit:
                break

            loop_idx += 1
            # Prune completed tasks periodically to control memory
            if len(tasks) >= 100_000:
                tasks = [t for t in tasks if not t.done()]
            print(
                f"loop replay: iteration {loop_idx} starting, "
                f"sent {sent_count} requests (actual_sent={self._actual_sent_count}), "
                f"elapsed {time.monotonic() - started_at:.1f}s",
                flush=True,
            )

        self._send_end = time.monotonic()
        self._sent_count = sent_count
        print(
            f"sending complete: sent={sent_count}, actual_sent={self._actual_sent_count} "
            f"requests dispatched in {self._send_end - started_at:.1f}s, "
            f"waiting for responses...",
            flush=True,
        )

        # Progress monitor: log response count every 10s
        async def _log_progress():
            while True:
                await asyncio.sleep(10)
                done = len(self._results)
                elapsed = time.monotonic() - started_at
                print(
                    f"  progress: {done}/{sent_count} responses received, "
                    f"sent={sent_count}, actual_sent={self._actual_sent_count}, "
                    f"elapsed {elapsed:.1f}s",
                    flush=True,
                )
                if done >= sent_count:
                    break

        progress_task = asyncio.create_task(_log_progress())
        response_timeout = getattr(self.args, "response_timeout", 120)
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=response_timeout)
        except asyncio.TimeoutError:
            done = len(self._results)
            print(
                f"  response timeout: {response_timeout}s reached, "
                f"{done}/{sent_count} responses received, "
                f"cancelling remaining...",
                flush=True,
            )
            for t in tasks:
                t.cancel()
            await asyncio.sleep(1)
        progress_task.cancel()
        elapsed = time.monotonic() - started_at
        await self._write_summary(elapsed)
        # Stop pushgateway loop and do final push
        if self._pushgateway_task:
            self._pushgateway_task.cancel()
            try:
                await self._pushgateway_task
            except asyncio.CancelledError:
                pass
            await self._final_push()

    def _make_loop_request(
        self, req: ReplayRequest, loop_idx: int, sent_count: int
    ) -> ReplayRequest:
        """Create a copy of req with unique IDs for loop iteration."""
        new_source_rid = f"{req.source_rid}_L{loop_idx}"
        new_request_id = to_signed_int64(stable_request_id(new_source_rid))
        return ReplayRequest(
            request_id=new_request_id,
            source_rid=new_source_rid,
            trace_id=f"{req.trace_id}_L{loop_idx}" if req.trace_id else "",
            ts_ms=req.ts_ms,
            input_len=req.input_len,
            output_len=req.output_len,
            block_keys=req.block_keys,
            token_ids=req.token_ids,
            prod_prefill=req.prod_prefill,
            prod_decode=req.prod_decode,
            prod_ttfb_ms=req.prod_ttfb_ms,
            prod_total_ms=req.prod_total_ms,
        )

    async def _handle_with_semaphore(
        self, req: ReplayRequest, sem: asyncio.Semaphore
    ) -> None:
        started = time.monotonic()

        # Phase 1: Schedule RPC (within semaphore)
        async with sem:
            self._inflight_count += 1
            try:
                self._actual_sent_count += 1
                result, input_pb, schedule_response = await self._do_schedule(
                    req, started
                )
            finally:
                self._inflight_count -= 1

        # Phase 2: Fetch response or fallback (outside semaphore).
        # _read_engine_stream fetches directly from the engine, not via Master,
        # so it doesn't need the semaphore.  Engine-side concurrency is
        # controlled by per-engine semaphores.
        if schedule_response is not None:
            try:
                first_frame_s, terminal_s = await self._read_engine_stream(
                    input_pb, schedule_response
                )
                end = terminal_s or time.monotonic()
                if first_frame_s:
                    result["ttft_ms"] = round((first_frame_s - started) * 1000.0, 3)
                result["total_ms"] = round((end - started) * 1000.0, 3)
                result["status"] = "ok"
                result["route_path"] = "master"
                result["wall_clock_ts"] = time.time()
            except Exception as exc:
                if self.args.enable_fallback and self._fallback_prefill_addrs:
                    try:
                        result = await self._try_fallback(req, result, started)
                    except Exception as fb_exc:
                        result["status"] = "exception"
                        result["error"] = f"fetch={exc!r}; fallback={fb_exc!r}"
                        result["route_path"] = "fallback"
                        result["total_ms"] = round(
                            (time.monotonic() - started) * 1000.0, 3
                        )
                        result["wall_clock_ts"] = time.time()
                else:
                    result["status"] = "exception"
                    result["error"] = repr(exc)
                    result["total_ms"] = round((time.monotonic() - started) * 1000.0, 3)
        elif result["status"] in ("exception", "schedule_error"):
            # Schedule failed — try fallback outside semaphore
            if self.args.enable_fallback and self._fallback_prefill_addrs:
                schedule_exc = result.pop("_schedule_exc", None)
                original_error = result.get("error", "")
                try:
                    result = await self._try_fallback(req, result, started)
                except Exception as fb_exc:
                    result["status"] = "exception"
                    if schedule_exc is not None:
                        result["error"] = (
                            f"master={schedule_exc!r}; fallback={fb_exc!r}"
                        )
                    else:
                        result["error"] = (
                            f"master={original_error}; fallback={fb_exc!r}"
                        )
                    result["route_path"] = "fallback"
                    result["total_ms"] = round((time.monotonic() - started) * 1000.0, 3)
                    result["wall_clock_ts"] = time.time()

        # Clean up internal fields before writing
        result.pop("_schedule_exc", None)

        async with self._write_lock:
            self._results.append(result)
            with self.per_request_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, separators=(",", ":")) + "\n")

    def _fetch_response_enabled(self) -> bool:
        """Check whether fetch-response (engine stream reading) is enabled.

        Controlled by --schedule-only flag and FLEXLB_EXPECT_FETCH_RESPONSE
        env var.  When --schedule-only is set, fetch is always disabled.
        When the env var is explicitly "0"/"false"/"no", fetch is disabled.
        Otherwise (env var unset or truthy), fetch is enabled.
        """
        if self.args.schedule_only:
            return False
        val = os.environ.get("FLEXLB_EXPECT_FETCH_RESPONSE", "")
        return val.lower() not in ("0", "false", "no")

    async def _do_schedule(
        self, req: ReplayRequest, started: float
    ) -> tuple[dict, Optional[object], Optional[object]]:
        """Execute the Schedule RPC only (no fetch response).

        Returns (result, input_pb, schedule_response).  schedule_response is
        not None only when the schedule succeeded *and* fetch response is
        needed.  Otherwise the result dict is fully populated.
        """
        input_pb = self._build_generate_input(req)
        schedule_req = self.schedule_pb2.FlexlbScheduleRequestPB(
            request_id=req.request_id,
            generate_input=input_pb.SerializeToString(),
            block_cache_keys=req.block_keys,
            seq_len=req.input_len,
            generate_timeout=self.args.timeout_ms,
            request_time_ms=int(time.time() * 1000),
            max_new_tokens=max(1, req.output_len),
            num_beams=1,
            force_disable_sp_run=False,
            model=self.args.model,
            api_key=self.args.api_key,
            schedule_mode=self._schedule_mode_pb(),
            cache_key_block_size=1024,
        )

        result = {
            "rid": req.source_rid,
            "trace_id": req.trace_id,
            "request_id": req.request_id,
            "ts": req.ts_ms,
            "input_len": req.input_len,
            "output_len": req.output_len,
            "status": "unknown",
            "schedule_ms": 0.0,
            "ttft_ms": 0.0,
            "total_ms": 0.0,
            "enqueued_by_master": False,
            "prefill": "",
            "decode": "",
            "error": "",
            "route_path": "master",
            "wall_clock_ts": 0.0,
        }

        try:
            schedule_start = time.monotonic()
            flexlb_stub = self.schedule_pb2_grpc.FlexlbServiceStub(
                await self._channel(self._flexlb_target())
            )
            response = await flexlb_stub.Schedule(
                schedule_req, timeout=self.args.timeout_ms / 1000.0
            )
            result["schedule_ms"] = round(
                (time.monotonic() - schedule_start) * 1000.0, 3
            )
            result["enqueued_by_master"] = bool(response.enqueued_by_master)

            if response.code != 200 or not response.success:
                result["status"] = "schedule_error"
                result["error"] = response.error_message or f"code={response.code}"
                result["total_ms"] = round((time.monotonic() - started) * 1000.0, 3)
                return result, None, None

            self._copy_role_addrs(input_pb, response)
            result["prefill"] = self._role_addr(response, "PREFILL")
            result["decode"] = self._role_addr(response, "DECODE")

            if not self._fetch_response_enabled():
                result["status"] = "scheduled"
                result["total_ms"] = round((time.monotonic() - started) * 1000.0, 3)
                return result, None, None

            return result, input_pb, response
        except Exception as exc:
            result["status"] = "exception"
            result["error"] = repr(exc)
            result["_schedule_exc"] = exc
            result["total_ms"] = round((time.monotonic() - started) * 1000.0, 3)
            return result, None, None

    def _load_fallback_endpoints(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        prefill_domain = data.get("prefill_domain", "")
        decode_domain = data.get("decode_domain", "")
        env = data.get("env", {})

        prefill_key = f"DOMAIN_ADDRESS:{prefill_domain}"
        decode_key = f"DOMAIN_ADDRESS:{decode_domain}"
        if prefill_key in env:
            self._fallback_prefill_addrs = []
            for a in env[prefill_key].split(","):
                a = a.strip()
                if not a:
                    continue
                # DOMAIN_ADDRESS contains HTTP port; convert to gRPC port
                # (gRPC port = HTTP port + 1, per CommonConstants.GRPC_PORT_OFFSET)
                host, port = a.rsplit(":", 1)
                self._fallback_prefill_addrs.append(f"{host}:{int(port) + 1}")
        if decode_key in env:
            self._fallback_decode_addrs = []
            for a in env[decode_key].split(","):
                a = a.strip()
                if not a:
                    continue
                # DOMAIN_ADDRESS contains HTTP port; convert to gRPC port
                host, port = a.rsplit(":", 1)
                self._fallback_decode_addrs.append(f"{host}:{int(port) + 1}")

        if not self._fallback_prefill_addrs:
            self._fallback_prefill_addrs = [
                e["grpc_addr"]
                for e in data.get("engines", [])
                if e.get("role") == "prefill" and e.get("grpc_addr")
            ]
        if not self._fallback_decode_addrs:
            self._fallback_decode_addrs = [
                e["grpc_addr"]
                for e in data.get("engines", [])
                if e.get("role") == "decode" and e.get("grpc_addr")
            ]

        if self._fallback_prefill_addrs:
            print(
                f"fallback prefill addrs: {self._fallback_prefill_addrs}",
                flush=True,
            )
        if self._fallback_decode_addrs:
            print(
                f"fallback decode addrs: {self._fallback_decode_addrs}",
                flush=True,
            )

    def _round_robin_prefill_addr(self) -> str:
        addrs = self._fallback_prefill_addrs
        if not addrs:
            raise RuntimeError("no fallback prefill addresses available")
        addr = addrs[self._fallback_prefill_rr % len(addrs)]
        self._fallback_prefill_rr += 1
        return addr

    def _round_robin_decode_addr(self) -> str:
        addrs = self._fallback_decode_addrs
        if not addrs:
            return ""
        addr = addrs[self._fallback_decode_rr % len(addrs)]
        self._fallback_decode_rr += 1
        return addr

    async def _try_fallback(
        self,
        req: ReplayRequest,
        result: dict,
        started: float,
    ) -> dict:
        prefill_addr = self._round_robin_prefill_addr()
        decode_addr = self._round_robin_decode_addr()

        fb_input = self._build_generate_input(req)
        del fb_input.generate_config.role_addrs[:]
        p_host, p_port = prefill_addr.rsplit(":", 1)
        fb_input.generate_config.role_addrs.add(
            role="PREFILL",
            role_type=self.pb2.ROLE_TYPE_PREFILL,
            ip=p_host,
            http_port=0,
            grpc_port=int(p_port),
        )
        if decode_addr:
            d_host, d_port = decode_addr.rsplit(":", 1)
            fb_input.generate_config.role_addrs.add(
                role="DECODE",
                role_type=self.pb2.ROLE_TYPE_DECODE,
                ip=d_host,
                http_port=0,
                grpc_port=int(d_port),
            )

        channel = await self._channel(prefill_addr)
        stub = self.pb2_grpc.RpcServiceStub(channel)
        stream = stub.GenerateStreamCall(
            fb_input, timeout=self.args.timeout_ms / 1000.0
        )

        first_frame_s = None
        terminal_s = None
        async for output in stream:
            now = time.monotonic()
            if first_frame_s is None:
                first_frame_s = now
            if output.flatten_output.finished and any(output.flatten_output.finished):
                terminal_s = now

        end = terminal_s or time.monotonic()
        result["schedule_ms"] = 0.0
        result["ttft_ms"] = (
            round((first_frame_s - started) * 1000.0, 3) if first_frame_s else 0.0
        )
        result["total_ms"] = round((end - started) * 1000.0, 3)
        result["status"] = "ok"
        result["prefill"] = prefill_addr
        result["decode"] = decode_addr
        result["route_path"] = "fallback"
        result["wall_clock_ts"] = time.time()
        return result

    async def _read_engine_stream(
        self, input_pb, schedule_response
    ) -> tuple[Optional[float], Optional[float]]:
        prefill_addr = self._role_addr(schedule_response, "PREFILL")
        pdfusion_addr = self._role_addr(schedule_response, "PDFUSION")
        target = prefill_addr or pdfusion_addr
        if not target:
            raise RuntimeError("schedule response has no PREFILL/PDFUSION address")
        channel = await self._channel(target)
        stub = self.pb2_grpc.RpcServiceStub(channel)
        if schedule_response.enqueued_by_master:
            stream = stub.FetchResponse(
                self.pb2.FetchRequestPB(request_id=input_pb.request_id),
                timeout=self.args.timeout_ms / 1000.0,
            )
        else:
            stream = stub.GenerateStreamCall(
                input_pb, timeout=self.args.timeout_ms / 1000.0
            )

        first_frame_s = None
        terminal_s = None
        async for output in stream:
            now = time.monotonic()
            if first_frame_s is None:
                first_frame_s = now
            if output.flatten_output.finished and any(output.flatten_output.finished):
                terminal_s = now
        return first_frame_s, terminal_s

    def _build_generate_input(self, req: ReplayRequest):
        meta = {
            "rid": req.source_rid,
            "trace_id": req.trace_id,
            "input_len": req.input_len,
            "output_len": req.output_len,
            "block_cache_keys": req.block_keys,
        }
        config = self.pb2.GenerateConfigPB(
            max_new_tokens=max(1, req.output_len),
            num_return_sequences=1,
            top_p=1.0,
            top_k=0,
            temperature=1.0,
            return_incremental=True,
            is_streaming=True,
            timeout_ms=self.args.timeout_ms,
            unique_key=encode_unique_key(meta),
        )
        info = self.pb2.RequestInfoPB(
            request_id=req.source_rid, trace_id=req.trace_id, source_role="flexlb_eval"
        )
        return self.pb2.GenerateInputPB(
            request_id=req.request_id,
            token_ids=req.token_ids or [],
            generate_config=config,
            client_id="flexlb_eval_client",
            start_time=int(time.time() * 1000),
            request_info=info,
        )

    def _copy_role_addrs(self, input_pb, response) -> None:
        del input_pb.generate_config.role_addrs[:]
        for status in response.server_status:
            input_pb.generate_config.role_addrs.add(
                role=status.role,
                role_type=getattr(
                    self.pb2, f"ROLE_TYPE_{status.role}", self.pb2.ROLE_TYPE_PDFUSION
                ),
                ip=status.server_ip,
                http_port=status.http_port,
                grpc_port=status.grpc_port,
            )

    def _role_addr(self, response, role: str) -> str:
        for status in response.server_status:
            if status.role == role and status.server_ip:
                return f"{status.server_ip}:{status.grpc_port}"
        return ""

    async def _channel(self, target: str):
        import grpc

        if target not in self._channels:
            pool = []
            for i in range(self._n_channels):
                pool.append(
                    grpc.aio.insecure_channel(
                        target,
                        options=[
                            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                            ("grpc.max_send_message_length", 64 * 1024 * 1024),
                            ("grpc.http2.initial_flow_control_window", 8388608),
                            ("grpc.max_concurrent_streams", 1000),
                            ("grpc.keepalive_time_ms", 30000),
                            ("grpc.keepalive_timeout_ms", 10000),
                            (
                                "grpc.primary_user_agent",
                                f"flexlb-load-client-{i}",
                            ),
                        ],
                    )
                )
            self._channels[target] = pool
            self._channel_rr[target] = 0
        rr = self._channel_rr[target]
        self._channel_rr[target] = (rr + 1) % self._n_channels
        return self._channels[target][rr]

    def _flexlb_target(self) -> str:
        if self.args.flexlb_grpc_target:
            return self.args.flexlb_grpc_target
        host, port = self.args.flexlb_http_addr.rsplit(":", 1)
        return f"{host}:{int(port) + 2}"

    def _schedule_mode_pb(self) -> int:
        return {
            "auto": self.schedule_pb2.FLEXLB_SCHEDULE_AUTO,
            "batch": self.schedule_pb2.FLEXLB_SCHEDULE_BATCH,
            "direct": self.schedule_pb2.FLEXLB_SCHEDULE_DIRECT,
            "queue": self.schedule_pb2.FLEXLB_SCHEDULE_QUEUE,
        }[self.args.schedule_mode]

    async def _write_summary(self, elapsed_s: float) -> None:
        ok = [r for r in self._results if r["status"] == "ok"]
        scheduled = [r for r in self._results if r["status"] in ("ok", "scheduled")]
        ttft = [r["ttft_ms"] for r in ok if r["ttft_ms"] > 0]
        total = [r["total_ms"] for r in ok if r["total_ms"] > 0]
        schedule = [r["schedule_ms"] for r in self._results if r["schedule_ms"] > 0]
        violations = [r for r in ok if r["ttft_ms"] > self.args.sla_ttft_ms]

        send_duration_s = (
            self._send_end - self._send_start
            if self._send_start is not None and self._send_end is not None
            else 0.0
        )
        summary = {
            "trace": self.args.trace,
            "max_concurrency": self.args.max_concurrency,
            "elapsed_s": round(elapsed_s, 3),
            "total_requests": len(self._results),
            "scheduled": len(scheduled),
            "completed": len(ok),
            "errors": len(self._results) - len(scheduled),
            "offered_qps": (
                round(len(self._results) / elapsed_s, 3) if elapsed_s > 0 else 0.0
            ),
            "completed_qps": round(len(ok) / elapsed_s, 3) if elapsed_s > 0 else 0.0,
            "send_duration_s": round(send_duration_s, 3),
            "sent_count": self._sent_count,
            "actual_sent_count": self._actual_sent_count,
            "send_qps": (
                round(len(self._results) / send_duration_s, 3)
                if send_duration_s > 0
                else 0.0
            ),
            "actual_send_qps": (
                round(self._actual_sent_count / send_duration_s, 3)
                if send_duration_s > 0
                else 0.0
            ),
            "n_channels": self._n_channels,
            "sla_ttft_ms": self.args.sla_ttft_ms,
            "sla_violations": len(violations),
            "sla_violation_rate": round(len(violations) / len(ok), 6) if ok else 0.0,
            "schedule_latency_ms": summarize_latencies(schedule),
            "ttft_ms": summarize_latencies(ttft),
            "total_ms": summarize_latencies(total),
            "prefill_balance": load_balance_summary(r["prefill"] for r in ok),
            "decode_balance": load_balance_summary(r["decode"] for r in ok),
            "status_counts": _count_by(self._results, "status"),
            "route_path_counts": _count_by(self._results, "route_path"),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_markdown_report(
            summary=summary, results=self._results, output_path=self.report_path
        )
        print(json.dumps(summary, indent=2), flush=True)
        print(f"report: {self.report_path}", flush=True)

    def _percentile(self, values: list, p: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p / 100.0)
        if idx >= len(sorted_vals):
            idx = len(sorted_vals) - 1
        return sorted_vals[idx]

    async def _pushgateway_loop(self) -> None:
        """Background loop: push metrics to pushgateway every 5 seconds."""
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                await asyncio.sleep(5)
                try:
                    await self._push_metrics(session)
                except Exception as exc:
                    print(f"pushgateway push error: {exc}", flush=True)

    async def _final_push(self) -> None:
        """Final metrics push after test completes."""
        if not self._pushgateway_url:
            return
        timeout = aiohttp.ClientTimeout(total=10)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                await self._push_metrics(session)
                print("pushgateway final push done", flush=True)
        except Exception as exc:
            print(f"pushgateway final push error: {exc}", flush=True)

    async def _push_metrics(self, session: aiohttp.ClientSession) -> None:
        """Collect stats from _results and push to pushgateway."""
        hostname = socket.gethostname()
        lines: list = []

        # Push send/completed counters (always, even if no results yet)
        lines.append(
            f'flexlb_client_send_total{{route_path="master"}} {self._sent_count}'
        )
        lines.append(
            f'flexlb_client_actual_send_total{{route_path="master"}} {self._actual_sent_count}'
        )
        lines.append(
            f'flexlb_client_completed_total{{route_path="master"}} {len(self._results)}'
        )

        # Semaphore inflight count
        lines.append(
            f'flexlb_client_inflight_count{{route_path="master"}} {self._inflight_count}'
        )

        # Semaphore max concurrency (for utilization calculation)
        lines.append(
            f'flexlb_client_max_concurrency{{route_path="master"}} {self.args.max_concurrency}'
        )

        # Semaphore utilization = inflight / max_concurrency
        if self.args.max_concurrency > 0:
            util = self._inflight_count / self.args.max_concurrency
            lines.append(
                f'flexlb_client_semaphore_utilization{{route_path="master"}} {util:.4f}'
            )

        if not self._results:
            body = "\n".join(lines) + "\n"
            url = (
                f"{self._pushgateway_url}/metrics/job/flexlb_client/instance/{hostname}"
            )
            headers = {"Content-Type": "text/plain; version=0.0.4"}
            async with session.put(
                url, data=body.encode("utf-8"), headers=headers
            ) as resp:
                if resp.status not in (200, 202):
                    text = await resp.text()
                    print(f"pushgateway push failed: {resp.status} {text}", flush=True)
            return

        groups: Dict[str, Dict[str, list]] = {}
        for r in self._results:
            rp = r.get("route_path", "unknown")
            if rp not in groups:
                groups[rp] = {"schedule_ms": [], "total_ms": [], "ttft_ms": []}
            if r.get("schedule_ms", 0) > 0:
                groups[rp]["schedule_ms"].append(r["schedule_ms"])
            if r.get("total_ms", 0) > 0:
                groups[rp]["total_ms"].append(r["total_ms"])
            if r.get("ttft_ms", 0) > 0:
                groups[rp]["ttft_ms"].append(r["ttft_ms"])

        for route_path, vals in groups.items():
            for metric_name in ("schedule_ms", "total_ms", "ttft_ms"):
                values = vals[metric_name]
                if not values:
                    continue
                count = len(values)
                avg = sum(values) / count
                p50 = self._percentile(values, 50)
                p99 = self._percentile(values, 99)
                mx = max(values)
                label = f'route_path="{route_path}"'
                lines.append(f"flexlb_client_{metric_name}_avg{{{label}}} {avg:.3f}")
                lines.append(f"flexlb_client_{metric_name}_p50{{{label}}} {p50:.3f}")
                lines.append(f"flexlb_client_{metric_name}_p99{{{label}}} {p99:.3f}")
                lines.append(f"flexlb_client_{metric_name}_max{{{label}}} {mx:.3f}")
                lines.append(f"flexlb_client_{metric_name}_count{{{label}}} {count}")

        body = "\n".join(lines) + "\n"
        url = f"{self._pushgateway_url}/metrics/job/flexlb_client/instance/{hostname}"
        headers = {"Content-Type": "text/plain; version=0.0.4"}
        async with session.put(url, data=body.encode("utf-8"), headers=headers) as resp:
            if resp.status not in (200, 202):
                text = await resp.text()
                print(f"pushgateway push failed: {resp.status} {text}", flush=True)


def _count_by(rows: List[dict], key: str) -> dict:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


async def main() -> None:
    args = parse_args()
    client = LoadClient(args)
    try:
        await client.run()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
