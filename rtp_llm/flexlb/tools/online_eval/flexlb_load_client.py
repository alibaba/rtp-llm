#!/usr/bin/env python3
"""Replay queries against a running SpringBoot flexlb-api and mock engines."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from online_eval.mock_engine import encode_unique_key
from online_eval.proto_utils import ensure_proto_modules
from online_eval.report import write_markdown_report
from online_eval.stats import load_balance_summary, summarize_latencies
from online_eval.trace_loader import ReplayRequest, load_replay_requests


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
        help="trace-time replay duration; 300 means first 5 minutes",
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        help="1=real inter-arrival, 10=10x faster, 0=as fast as possible",
    )
    parser.add_argument("--max-concurrency", type=int, default=1024)
    parser.add_argument(
        "--schedule-mode", choices=["auto", "batch", "direct", "queue"], default="batch"
    )
    parser.add_argument("--timeout-ms", type=int, default=3600000)
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
    return parser.parse_args()


class LoadClient:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pb2, self.pb2_grpc = ensure_proto_modules()
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.per_request_path = self.output_dir / "per_request.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.report_path = self.output_dir / "report.md"
        self._write_lock = asyncio.Lock()
        self._results: List[dict] = []
        self._channels: Dict[str, object] = {}
        self._fallback_prefill_addrs: List[str] = []
        self._fallback_decode_addrs: List[str] = []
        self._fallback_prefill_rr = 0
        self._fallback_decode_rr = 0
        if getattr(args, "endpoints_file", None):
            self._load_fallback_endpoints(args.endpoints_file)

    async def close(self) -> None:
        for channel in self._channels.values():
            await channel.close()
        self._channels.clear()

    async def run(self) -> None:
        requests = load_replay_requests(
            self.args.trace,
            limit=self.args.limit,
            duration_s=self.args.duration_s,
            zero_output_policy=self.args.zero_output_policy,
            include_token_ids=True,
        )
        if not requests:
            raise RuntimeError("no replayable requests loaded")
        print(f"loaded {len(requests)} requests from {self.args.trace}", flush=True)

        self.per_request_path.write_text("", encoding="utf-8")
        sem = asyncio.Semaphore(self.args.max_concurrency)
        started_at = time.monotonic()
        first_ts = requests[0].ts_ms
        tasks = []
        for req in requests:
            if self.args.replay_speed > 0 and req.ts_ms > 0:
                due_s = (req.ts_ms - first_ts) / 1000.0 / self.args.replay_speed
                sleep_s = due_s - (time.monotonic() - started_at)
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
            tasks.append(asyncio.create_task(self._handle_with_semaphore(req, sem)))
        await asyncio.gather(*tasks)
        elapsed = time.monotonic() - started_at
        await self._write_summary(elapsed)

    async def _handle_with_semaphore(
        self, req: ReplayRequest, sem: asyncio.Semaphore
    ) -> None:
        async with sem:
            result = await self._handle_request(req)
            async with self._write_lock:
                self._results.append(result)
                with self.per_request_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, separators=(",", ":")) + "\n")

    async def _handle_request(self, req: ReplayRequest) -> dict:
        started = time.monotonic()
        input_pb = self._build_generate_input(req)
        schedule_req = self.pb2.FlexlbScheduleRequestPB(
            request_id=req.request_id,
            generate_input=input_pb,
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
            flexlb_stub = self.pb2_grpc.FlexlbServiceStub(
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
                return result

            self._copy_role_addrs(input_pb, response)
            result["prefill"] = self._role_addr(response, self.pb2.ROLE_TYPE_PREFILL)
            result["decode"] = self._role_addr(response, self.pb2.ROLE_TYPE_DECODE)

            if self.args.schedule_only:
                result["status"] = "scheduled"
                result["total_ms"] = round((time.monotonic() - started) * 1000.0, 3)
                return result

            first_frame_s, terminal_s = await self._read_engine_stream(
                input_pb, response
            )
            end = terminal_s or time.monotonic()
            if first_frame_s:
                result["ttft_ms"] = round((first_frame_s - started) * 1000.0, 3)
            result["total_ms"] = round((end - started) * 1000.0, 3)
            result["status"] = "ok"
            result["route_path"] = "master"
            result["wall_clock_ts"] = time.time()
            return result
        except Exception as exc:
            if self.args.enable_fallback and self._fallback_prefill_addrs:
                try:
                    return await self._try_fallback(req, result, started)
                except Exception as fb_exc:
                    result["status"] = "exception"
                    result["error"] = f"master={exc!r}; fallback={fb_exc!r}"
                    result["route_path"] = "fallback"
                    result["total_ms"] = round((time.monotonic() - started) * 1000.0, 3)
                    result["wall_clock_ts"] = time.time()
                    return result
            result["status"] = "exception"
            result["error"] = repr(exc)
            result["total_ms"] = round((time.monotonic() - started) * 1000.0, 3)
            return result

    def _load_fallback_endpoints(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        prefill_domain = data.get("prefill_domain", "")
        decode_domain = data.get("decode_domain", "")
        env = data.get("env", {})

        prefill_key = f"DOMAIN_ADDRESS:{prefill_domain}"
        decode_key = f"DOMAIN_ADDRESS:{decode_domain}"
        if prefill_key in env:
            self._fallback_prefill_addrs = [
                a.strip() for a in env[prefill_key].split(",") if a.strip()
            ]
        if decode_key in env:
            self._fallback_decode_addrs = [
                a.strip() for a in env[decode_key].split(",") if a.strip()
            ]

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
        prefill_addr = self._role_addr(schedule_response, self.pb2.ROLE_TYPE_PREFILL)
        pdfusion_addr = self._role_addr(schedule_response, self.pb2.ROLE_TYPE_PDFUSION)
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
                role_type=status.role_type,
                ip=status.server_ip,
                http_port=status.http_port,
                grpc_port=status.grpc_port,
            )

    def _role_addr(self, response, role: int) -> str:
        for status in response.server_status:
            if status.role_type == role and status.server_ip:
                return f"{status.server_ip}:{status.grpc_port}"
        return ""

    async def _channel(self, target: str):
        import grpc

        if target not in self._channels:
            self._channels[target] = grpc.aio.insecure_channel(
                target,
                options=[
                    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                    ("grpc.max_send_message_length", 64 * 1024 * 1024),
                ],
            )
        return self._channels[target]

    def _flexlb_target(self) -> str:
        if self.args.flexlb_grpc_target:
            return self.args.flexlb_grpc_target
        host, port = self.args.flexlb_http_addr.rsplit(":", 1)
        return f"{host}:{int(port) + 2}"

    def _schedule_mode_pb(self) -> int:
        return {
            "auto": self.pb2.FLEXLB_SCHEDULE_AUTO,
            "batch": self.pb2.FLEXLB_SCHEDULE_BATCH,
            "direct": self.pb2.FLEXLB_SCHEDULE_DIRECT,
            "queue": self.pb2.FLEXLB_SCHEDULE_QUEUE,
        }[self.args.schedule_mode]

    async def _write_summary(self, elapsed_s: float) -> None:
        ok = [r for r in self._results if r["status"] == "ok"]
        scheduled = [r for r in self._results if r["status"] in ("ok", "scheduled")]
        ttft = [r["ttft_ms"] for r in ok if r["ttft_ms"] > 0]
        total = [r["total_ms"] for r in ok if r["total_ms"] > 0]
        schedule = [r["schedule_ms"] for r in self._results if r["schedule_ms"] > 0]
        violations = [r for r in ok if r["ttft_ms"] > self.args.sla_ttft_ms]

        summary = {
            "trace": self.args.trace,
            "elapsed_s": round(elapsed_s, 3),
            "total_requests": len(self._results),
            "scheduled": len(scheduled),
            "completed": len(ok),
            "errors": len(self._results) - len(scheduled),
            "offered_qps": (
                round(len(self._results) / elapsed_s, 3) if elapsed_s > 0 else 0.0
            ),
            "completed_qps": round(len(ok) / elapsed_s, 3) if elapsed_s > 0 else 0.0,
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
