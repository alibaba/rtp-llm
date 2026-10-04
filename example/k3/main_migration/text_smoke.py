#!/usr/bin/env python3
"""Four-layer preflight or complete 93-layer K3 PD smoke, selected by checkpoint.

Adapted from K3 dev 64c6aff3666402228950f1f09031e228c3734277.
This is not the original dev all suite. Runtime evidence is a separate gate.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import Any, Callable

@dataclass(frozen=True)
class Case:
    name: str
    prompt: str | list[dict[str, Any]]
    expected_regex: str
    reuse: str
    require_chunk: bool = False
    require_mtp: bool = False
    require_mtp_draft: bool = False
    require_multimodal: bool = False
    max_tokens: int | None = None
    timeout_s: int | None = None
    decode_owner_rank: int = 0
    expected_json: dict[str, str] | None = None
    expected_input_len: int | None = None
    expected_reuse_len: int | None = None
    cache_block_boundary: int | None = None
    cache_block_phase: str | None = None
    decode_crossings: tuple[int, ...] = ()
    expected_cache_tier: str | None = None
    allow_long_history: bool = False
    preparation_only: bool = False
    thinking_disabled: bool = False


class SmokeFailure(RuntimeError):
    pass


class TransportFailure(SmokeFailure):
    """Only connection failures and transient HTTP responses may be retried."""


class SmokeDeadline(SmokeFailure):
    """A formal request reached the user's five-minute wall-clock limit."""


def prefill_cache_tiers(aux: dict[str, Any]) -> dict[str, int]:
    """Use Prefill counters; generic local reuse includes lower tiers."""
    keys = ("prefill_total_reuse_len", "prefill_local_reuse_len",
            "prefill_memory_reuse_len", "prefill_disk_reuse_len")
    if any(key not in aux for key in keys):
        raise SmokeFailure("response lacks Prefill cache-tier counters")
    total, local, memory, disk = (int(aux[key]) for key in keys)
    device = local - memory - disk
    if min(total, local, memory, disk, device) < 0 or local > total:
        raise SmokeFailure(
            f"inconsistent Prefill cache-tier counters: "
            f"total={total} local={local} memory={memory} disk={disk}"
        )
    return {"total": total, "local": local, "device": device,
            "memory": memory, "disk": disk}


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON answer key: {key}")
        result[key] = value
    return result


def parse_decode_role_addr(value: str) -> dict[str, Any]:
    parts = value.rsplit(":", 2)
    if len(parts) != 3 or not parts[0]:
        raise argparse.ArgumentTypeError(
            "Decode role address must have IP:HTTP_PORT:GRPC_PORT form"
        )
    ip, http_port_text, grpc_port_text = parts
    try:
        http_port = int(http_port_text)
        grpc_port = int(grpc_port_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Decode role address ports must be integers"
        ) from exc
    if not (1 <= http_port <= 65535 and 1 <= grpc_port <= 65535):
        raise argparse.ArgumentTypeError(
            "Decode role address ports must be in [1, 65535]"
        )
    return {
        "role": "DECODE",
        "ip": ip,
        "http_port": http_port,
        "grpc_port": grpc_port,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Kimi K3 full-model PD cache/multi-batch smoke cases."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--decode-health-url", required=True)
    parser.add_argument(
        "--decode-role-addr",
        dest="decode_role_addrs",
        action="append",
        type=parse_decode_role_addr,
        default=[],
        help=(
            "ordered Decode DP owner endpoint in IP:HTTP_PORT:GRPC_PORT form; "
            "repeat once per DP rank"
        ),
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--prefill-event-dir", type=pathlib.Path,
                        help="Local Prefill LOG_PATH for the Host-load cancellation trigger")
    parser.add_argument("--prefill-engine-log", type=pathlib.Path,
                        help="Local Prefill C++ engine log for Host-load events")
    parser.add_argument("--prefill-rpc-runfiles", type=pathlib.Path,
                        help="Local Prefill Bazel runfiles for the PD-aware grouped cache case")
    parser.add_argument("--prefill-grpc-port", type=int,
                        help="Local Prefill group RPC port")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument(
        "--reuse-unit-tokens",
        type=int,
        default=0,
        help="cache reuse key span; 0 means one configured cache block, or use two blocks for a CP virtual key.",
    )
    parser.add_argument("--chunk-tokens", type=int, default=65536)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--identity-max-tokens", type=int, default=256)
    parser.add_argument("--single-exact-max-tokens", type=int, default=128)
    parser.add_argument("--mtp-chunk-max-tokens", type=int, default=128)
    parser.add_argument(
        "--rdma-prewarm-attempts",
        type=int,
        default=0,
        help=(
            "run this many bounded concurrent RDMA prewarm attempts before the "
            "formal all-suite cases; zero disables prewarm"
        ),
    )
    parser.add_argument(
        "--rdma-prewarm-timeout",
        type=int,
        default=300,
        help="per-request timeout for each bounded RDMA prewarm attempt",
    )
    parser.add_argument("--rdma-prewarm-backoff-s", type=float, default=5.0)
    parser.add_argument("--rdma-prewarm-settle-s", type=float, default=2.0)
    parser.add_argument("--checkpoint", "--long-prefix-checkpoint",
                        dest="long_prefix_checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    # The checkpoint chooses one of two fixed runs; phases and MTP cannot be
    # disabled from the command line.
    config = json.loads((args.long_prefix_checkpoint / "config.json").read_text())
    text_config = config.get("text_config", config)
    layers = text_config.get("num_hidden_layers")
    if layers not in (4, 93):
        parser.error("smoke requires a four-layer or 93-layer checkpoint")
    args.suite = "orthogonal-flow" if layers == 4 else "main-text-64k-capped"
    args.run_kind = "four-layer" if layers == 4 else "full-93"
    args.require_mtp = True
    args.orthogonal_phases = ("cache", "cancel", "page", "chunk", "decode")
    if (
        args.prefill_event_dir is None or not args.prefill_event_dir.is_dir()
    ):
        parser.error("orthogonal smoke needs an existing local --prefill-event-dir")
    if (
        args.prefill_engine_log is None or not args.prefill_engine_log.is_file()
    ):
        parser.error("orthogonal smoke needs an existing local --prefill-engine-log")
    if (
        args.prefill_rpc_runfiles is None or
        not (args.prefill_rpc_runfiles / "rtp_llm" / "rtp_llm" / "cpp" /
             "model_rpc" / "proto" / "model_rpc_service_pb2.py").is_file()
    ):
        parser.error("orthogonal smoke needs the local --prefill-rpc-runfiles")
    if (
        args.prefill_grpc_port is None or not 1 <= args.prefill_grpc_port <= 65535
    ):
        parser.error("orthogonal smoke needs --prefill-grpc-port")
    if args.batch_size < 4:
        parser.error(
            "--batch-size must be at least 4 to cover hit/partial-hit/miss mixing"
        )
    for key in (
        "block_size",
        "chunk_tokens",
        "max_tokens",
        "identity_max_tokens",
        "single_exact_max_tokens",
        "mtp_chunk_max_tokens",
        "timeout",
        "rdma_prewarm_timeout",
    ):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.rdma_prewarm_attempts < 0:
        parser.error("--rdma-prewarm-attempts must be non-negative")
    if len(args.decode_role_addrs) not in (1, 2):
        parser.error("smoke requires one or two ordered Decode owner addresses")
    args.decode_dp_size = len(args.decode_role_addrs)
    for key in ("rdma_prewarm_backoff_s", "rdma_prewarm_settle_s"):
        if getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be non-negative")
    if args.chunk_tokens != 65536:
        parser.error("smoke requires a 65536-token single-prefill budget")
    if args.block_size != 4096:
        parser.error("smoke requires 4096-token cache blocks")
    args.case_deadline_s = 300
    if args.reuse_unit_tokens and (
        args.reuse_unit_tokens < args.block_size
        or args.reuse_unit_tokens % args.block_size
        or args.chunk_tokens % args.reuse_unit_tokens
    ):
        parser.error(
            "reuse-unit-tokens must be a cache-block multiple within the chunk budget"
        )
    if args.chunk_tokens % args.block_size:
        parser.error("chunk budget must be a multiple of the configured cache block")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or (args.output.parent / "requests").exists():
        parser.error(
            "use a new artifact directory; existing request evidence must not be overwritten"
        )
    return args


def numbered_answer_pattern(value: int) -> str:
    return rf"\A\s*{value}\s*\Z"


def make_cache_prompt(
    namespace: str, case_name: str, value: int, repeats: int = 900
) -> str:
    marker = f"缓存测试标识：{namespace}/{case_name}。"
    filler = (
        "这是一段用于验证长上下文缓存边界的固定材料，请保持阅读但不要复述。"
        "每段材料彼此独立，最终只回答末尾的算术问题。"
    )
    return marker + filler * repeats + f"\n只回答数字：{value} 的平方是多少？"


def make_partial_prompt(
    namespace: str, common_name: str, suffix_name: str, value: int, repeats: int = 900
) -> str:
    marker = f"部分命中测试标识：{namespace}/{common_name}。"
    filler = (
        "这是两次请求共同拥有的前缀材料，用于验证完整缓存页能够被后续请求复用。"
        "请忽略材料内容并继续阅读。"
    )
    return (
        marker
        + filler * repeats
        + (f"\n分支标识：{suffix_name}。只回答数字：{value} 的平方是多少？")
    )


def make_whole_chunk_prompt(namespace: str, case_name: str, value: int) -> str:
    marker = f"整模型分块测试标识：{namespace}/{case_name}。"
    filler = "长上下文分块缓存验证材料，请勿复述，只需继续阅读直到末尾问题。"
    # Deliberately exceed 64K characters. The runtime assertion below uses
    # tokenizer-reported input_len, so coverage cannot silently be mislabelled.
    return marker + filler * 5000 + f"\n只回答数字：{value} 的平方是多少？"


def make_flow_prompt(namespace: str) -> str:
    """Build a modest multi-round prompt for the four-layer RDMA flow smoke."""
    marker = f"四层流程测试标识：{namespace}/chunkwise-rdma-flow。"
    filler = "这是用于验证分块计算与增量RDMA传输的固定材料，请继续读取。"
    return marker + filler * 256 + "\n请回复任意一个非空字符。"


def cache_block_boundaries(
    block_size: int, reuse_unit: int, chunk_tokens: int
) -> tuple[int, ...]:
    """Ordinary block/checkpoint edges and one/two chunk thresholds."""
    if block_size <= 0 or reuse_unit < block_size or reuse_unit % block_size:
        raise ValueError(
            "ordinary cache reuse span must be a positive multiple of page size"
        )
    if chunk_tokens < reuse_unit or chunk_tokens % reuse_unit:
        raise ValueError(
            "chunk budget must contain complete ordinary cache reuse spans"
        )
    return tuple(
        sorted(
            set(range(block_size, reuse_unit + 1, block_size))
            | {2 * reuse_unit, chunk_tokens, 2 * chunk_tokens}
        )
    )


class Runner:
    @property
    def reuse_unit_tokens(self) -> int:
        """Cache reuse granularity: one page, or one checkpoint span."""
        return int(getattr(self.args, "reuse_unit_tokens", 0)) or int(
            self.args.block_size
        )

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
        self.health_endpoint = args.base_url.rstrip("/") + "/health"
        self.decode_health_endpoint = args.decode_health_url
        self.decode_role_addrs = list(getattr(args, "decode_role_addrs", []))
        # The service is local to the Prefill host; never route smoke traffic
        # through inherited HTTP proxy settings.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.records: list[dict[str, Any]] = []
        self.stages: list[dict[str, Any]] = []
        self.rdma_prewarm_attempts: list[dict[str, Any]] = []
        self.started_at = time.time()
        self.failures: list[dict[str, Any]] = []
        self.skipped_cases: list[dict[str, Any]] = []
        self._record_lock = threading.Lock()
        self._artifact_counter = 0

    def save(self, passed: bool, error: str | None = None) -> None:
        payload = {
            "run_kind": getattr(self.args, "run_kind", "four-layer" if self.args.suite ==
                                "orthogonal-flow" else "full-93"),
            "suite": self.args.suite,
            "case_profile": "compact-64k-v1" if self.args.suite in
            ("main-text-64k-capped", "orthogonal-flow") else None,
            "orthogonal_phases": list(self.args.orthogonal_phases),
            "source_reference": "64c6aff3666402228950f1f09031e228c3734277",
            "profile": "orthogonal-pd-page-rr" if self.args.suite in
            ("main-text-64k-capped", "orthogonal-flow") else "tp8-ep8-sp-no-dcp-text",
            "deferred_by_user": (
                ["chunk prefill", "over-64K inputs", "chunk budget +1/+7", "110K seed and append"]
                if self.args.suite in ("main-text-64k", "main-text-64k-capped") else []
            ),
            "single_prefill_input_limit": 65536 if self.args.suite in ("main-text-64k", "main-text-64k-capped") else None,
            "formal_request_deadline_s": self.args.case_deadline_s,
            "full_original_suite_passed": self.args.suite == "main-text" and passed and not self.skipped_cases,
            "not_applicable": (
                ["multimodal", "KTP", "EAGLE3/DSpark"]
                if self.args.suite in ("main-text-64k-capped", "orthogonal-flow")
                else ["DCP", "PageRR owner", "DP multi-owner", "multimodal", "KTP", "EAGLE3/DSpark"]
            ),
            "runtime_evidence_gate": "separate all-rank audit required; HTTP results alone do not certify runtime paths",
            "formal_request_retries": 0,
            "namespace": self.args.namespace,
            "passed": passed,
            "error": error,
            "block_size": self.args.block_size,
            "chunk_tokens": self.args.chunk_tokens,
            "reuse_unit_tokens": self.reuse_unit_tokens,
            "batch_size": self.args.batch_size,
            "decode_role_addrs": self.decode_role_addrs,
            "max_tokens": self.args.max_tokens,
            "identity_max_tokens": self.args.identity_max_tokens,
            "single_exact_max_tokens": self.args.single_exact_max_tokens,
            "mtp_chunk_max_tokens": self.args.mtp_chunk_max_tokens,
            "rdma_prewarm": {
                "enabled": self.args.rdma_prewarm_attempts > 0,
                "target_decode_owners": min(
                    self.args.batch_size, len(self.decode_role_addrs)
                ),
                "request_timeout_s": self.args.rdma_prewarm_timeout,
                "attempts": self.rdma_prewarm_attempts,
            },
            "elapsed_s": round(time.time() - self.started_at, 3),
            "summary": {
                "case_count": len(self.records),
                "skipped_case_count": len(self.skipped_cases),
                "cache_block_boundary_case_count": sum(
                    r.get("cache_block_boundary") is not None for r in self.records
                ),
                "decode_crossing_case_count": sum(
                    bool(r.get("decode_crossings")) for r in self.records
                ),
                "hit_count": sum(
                    r.get("effective_reuse_len", 0) > 0 for r in self.records
                ),
                "miss_count": sum(
                    r.get("effective_reuse_len") == 0 for r in self.records
                ),
                "reasoning_count": sum(
                    bool(r.get("reasoning_content", "").strip()) for r in self.records
                ),
                "mtp_case_count": sum(bool(r.get("require_mtp")) for r in self.records),
                "mtp_draft_case_count": sum(
                    bool(r.get("require_mtp_draft")) for r in self.records
                ),
                "multimodal_case_count": sum(
                    bool(r.get("require_multimodal")) for r in self.records
                ),
                "concurrent_stages": sum(
                    s.get("concurrent", False) for s in self.stages
                ),
            },
            "stages": self.stages,
            "cases": self.records,
            "failures": self.failures,
            "skipped_cases": self.skipped_cases,
        }
        self.args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def health(self, stage: str) -> None:
        endpoints = (
            ("prefill", self.health_endpoint),
            ("decode", self.decode_health_endpoint),
        )
        for role, endpoint in endpoints:
            request = urllib.request.Request(endpoint, method="GET")
            try:
                with self.opener.open(
                    request, timeout=min(self.args.timeout, 10)
                ) as response:
                    if response.status != 200:
                        raise SmokeFailure(
                            f"{role} health before {stage} returned HTTP {response.status}"
                        )
            except Exception as exc:
                raise SmokeFailure(
                    f"{role} health check before {stage} failed: {exc}"
                ) from exc

    def request(self, case: Case, barrier=None) -> dict[str, Any]:
        with self._record_lock:
            self._artifact_counter += 1
            number = self._artifact_counter
        artifact = None
        if self.args.output is not None:
            directory = self.args.output.parent / "requests"
            directory.mkdir(parents=True, exist_ok=True)
            artifact = (
                directory
                / f"{number:04d}-{hashlib.sha256(case.name.encode()).hexdigest()[:12]}.json"
            )
        audit = {
            "name": case.name,
            "owner": case.decode_owner_rank,
            "passed": False,
            "phase": "preparation" if case.preparation_only else
                     "prewarm" if case.name.startswith("rdma_prewarm_") else "formal",
            "expected": {
                "regex": case.expected_regex,
                "json": case.expected_json,
                "input_len": case.expected_input_len,
                "reuse_len": case.expected_reuse_len,
                "cache_tier": case.expected_cache_tier,
            },
        }

        def persist():
            if artifact is not None:
                artifact.write_text(
                    json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
                )

        try:
            result = self._request(case, barrier, audit=audit, persist=persist)
            result["phase"] = audit["phase"]
            audit.update(passed=True, result=result)
            with self._record_lock:
                self.records.append(result)
            return result
        except SmokeDeadline as exc:
            audit["skipped"] = True
            audit["deadline_s"] = self.args.case_deadline_s
            audit["error"] = f"{type(exc).__name__}: {exc}"
            with self._record_lock:
                self.skipped_cases.append(
                    {"name": case.name, "reason": "five-minute request deadline", "sent": True,
                     "artifact": str(artifact) if artifact else None}
                )
            raise
        except Exception as exc:
            if barrier is not None:
                barrier.abort()
            audit["error"] = f"{type(exc).__name__}: {exc}"
            with self._record_lock:
                self.failures.append(
                    {
                        "name": case.name,
                        "error": audit["error"],
                        "phase": audit["phase"],
                        "retryable": isinstance(exc, TransportFailure),
                        "artifact": str(artifact) if artifact else None,
                    }
                )
            raise
        finally:
            persist()

    def _request(
        self,
        case: Case,
        barrier: threading.Barrier | None = None,
        *,
        audit: dict,
        persist: Callable,
    ) -> dict[str, Any]:
        if self.args.suite in ("orthogonal-flow", "main-text-64k", "main-text-64k-capped") and not case.allow_long_history:
            if not isinstance(case.prompt, str):
                raise SmokeFailure("64K profile requires a text prompt")
            input_ids = self.tokenize(case.prompt)
            self.save_token_fixture(case.prompt, input_ids)
            audit["verified_input_len"] = len(input_ids)
            persist()
            if len(input_ids) > 65536 or case.require_chunk:
                raise SmokeFailure(f"{case.name}: outside the non-chunk 64K profile")
        elif case.allow_long_history and case.require_chunk:
            raise SmokeFailure(f"{case.name}: model-level chunking is outside this profile")
        request_max_tokens = case.max_tokens or self.args.max_tokens
        payload = {
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": case.prompt}],
            "max_tokens": request_max_tokens,
            "temperature": 0,
            "top_k": 1,
            "top_p": 0.95,
            "seed": 0,
            "stream": False,
            "debug_info": True,
        }
        if case.preparation_only or case.thinking_disabled:
            payload["enable_thinking"] = False
        if self.decode_role_addrs:
            if not 0 <= case.decode_owner_rank < len(self.decode_role_addrs):
                raise SmokeFailure(
                    f"{case.name}: decode owner rank {case.decode_owner_rank} is outside "
                    f"the configured world size {len(self.decode_role_addrs)}"
                )
            # The OpenAI chat endpoint builds GenerateConfig exclusively from
            # request.extra_configs.  A top-level role_addrs field is ignored
            # by ChatCompletionRequest, which silently falls back to the
            # process-wide REMOTE_RPC_SERVER_IP and routes every request to the
            # first Decode rank.
            payload["extra_configs"] = {
                "role_addrs": [self.decode_role_addrs[case.decode_owner_rank]]
            }
        audit["request"] = payload
        persist()
        wire_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        if barrier is not None:
            # Token fixture construction can serialize on /tokenize. Release
            # the HTTP requests together only after every caller is ready.
            barrier.wait(timeout=max(30, self.args.timeout))
        started = time.time()
        request_timeout = case.timeout_s or self.args.timeout
        if self.args.case_deadline_s is not None:
            marker = b"\n__K3_HTTP_STATUS__"
            deadline = self.args.case_deadline_s
            command = ["curl", "--silent", "--show-error", "--noproxy", "*",
                       "--max-time", str(deadline), "--request", "POST",
                       "--header", "Content-Type: application/json",
                       "--header", f"X-K3-Smoke-Case: {case.name}",
                       "--data-binary", "@-",
                       "--write-out", marker.decode() + "%{http_code}", self.endpoint]
            try:
                response = subprocess.run(command, input=wire_payload, capture_output=True,
                                          timeout=deadline + 1, check=False)
            except subprocess.TimeoutExpired as exc:
                audit["response_body"] = (exc.output or b"").decode("utf-8", errors="replace")
                raise SmokeDeadline(f"{case.name}: request exceeded {deadline}s") from exc
            if response.returncode == 28:
                audit["response_body"] = response.stdout.decode("utf-8", errors="replace")
                raise SmokeDeadline(f"{case.name}: request exceeded {deadline}s")
            if response.returncode != 0:
                raise TransportFailure(f"{case.name}: curl exited {response.returncode}: "
                                       f"{response.stderr.decode(errors='replace')[:500]}")
            body, separator, status_bytes = response.stdout.rpartition(marker)
            if not separator or not status_bytes.isdigit():
                raise TransportFailure(f"{case.name}: missing HTTP status from capped request")
            status = int(status_bytes)
        else:
            request = urllib.request.Request(
                self.endpoint, data=wire_payload,
                headers={"Content-Type": "application/json", "X-K3-Smoke-Case": case.name},
                method="POST"
            )
            try:
                with self.opener.open(request, timeout=request_timeout) as response:
                    body = response.read()
                    status = response.status
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                audit.update(status=exc.code, response_body=detail)
                failure = TransportFailure if exc.code in (408, 429, 502, 503, 504) else SmokeFailure
                raise failure(f"{case.name}: HTTP {exc.code}: {detail[:1000]}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                raise TransportFailure(f"{case.name}: request failed: {exc}") from exc
        audit.update(
            status=status, response_body=body.decode("utf-8", errors="replace")
        )
        persist()
        if status != 200:
            failure = TransportFailure if status in (408, 429, 502, 503, 504) else SmokeFailure
            raise failure(f"{case.name}: HTTP {status}")
        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise SmokeFailure(f"{case.name}: invalid JSON: {body[:1000]!r}") from exc
        return self.validate(
            case,
            result,
            time.time() - started,
            request_max_tokens,
        )

    def validate(
        self,
        case: Case,
        response: dict[str, Any],
        elapsed_s: float,
        request_max_tokens: int,
    ) -> dict[str, Any]:
        try:
            choice = response["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason")
            content = message.get("content", "") or ""
            reasoning_content = message.get("reasoning_content", "") or ""
            aux = response["aux_info"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SmokeFailure(
                f"{case.name}: malformed response: {response!r}"
            ) from exc
        debug_info = response.get("debug_info") or {}
        output_ids = debug_info.get("output_ids")
        if not isinstance(content, str) or not isinstance(reasoning_content, str):
            raise SmokeFailure(f"{case.name}: malformed model response")
        diagnostic = case.preparation_only or self.args.suite in ("flow", "orthogonal-flow")
        if not diagnostic and ("\ufffd" in content or "\ufffd" in reasoning_content):
            raise SmokeFailure(f"{case.name}: Unicode replacement in model response")
        if not diagnostic and finish_reason in ("length", "content_filter"):
            raise SmokeFailure(
                f"{case.name}: incomplete model response finish_reason={finish_reason}"
            )
        # A four-layer checkpoint is only a transport preflight. With the
        # model's default reasoning mode enabled, all short output may remain
        # in reasoning_content. Keep full-model semantic checks pinned to the
        # final answer, but accept either non-empty channel for flow coverage.
        answer_text = content
        if diagnostic and not answer_text.strip():
            answer_text = reasoning_content
        if not answer_text.strip():
            raise SmokeFailure(f"{case.name}: empty model response")
        if aux.get("pd_sep") is not True:
            raise SmokeFailure(f"{case.name}: pd_sep={aux.get('pd_sep')!r}")
        output_len = int(aux.get("output_len", 0))
        if output_len <= 0:
            raise SmokeFailure(f"{case.name}: output_len={aux.get('output_len')!r}")
        if not (
            isinstance(output_ids, list)
            and output_ids
            and all(isinstance(ids, list) and ids for ids in output_ids)
        ):
            raise SmokeFailure(f"{case.name}: missing output token ids: {output_ids!r}")
        if case.expected_json is not None:
            try:
                actual = json.loads(
                    answer_text, object_pairs_hook=reject_duplicate_keys
                )
            except ValueError as exc:
                raise SmokeFailure(
                    f"{case.name}: invalid exact JSON answer: {answer_text!r}"
                ) from exc
            if actual != case.expected_json:
                raise SmokeFailure(
                    f"{case.name}: exact JSON mismatch: {actual!r}; expected {case.expected_json!r}"
                )
        elif re.search(case.expected_regex, answer_text, flags=re.IGNORECASE) is None:
            raise SmokeFailure(
                f"{case.name}: answer failed {case.expected_regex!r}: {answer_text!r}"
            )

        input_len = int(aux.get("input_len", 0))
        raw_reuse = int(aux.get("reuse_len", 0))
        prefill_reuse_value = aux.get("prefill_total_reuse_len")
        effective_reuse = (
            int(prefill_reuse_value) if prefill_reuse_value is not None else raw_reuse
        )
        tier_lengths = (
            prefill_cache_tiers(aux)
            if all(key in aux for key in (
                "prefill_total_reuse_len", "prefill_local_reuse_len",
                "prefill_memory_reuse_len", "prefill_disk_reuse_len"))
            else None
        )
        if case.expected_cache_tier is not None:
            if tier_lengths is None:
                raise SmokeFailure(f"{case.name}: missing Prefill cache-tier counters")
            tier = case.expected_cache_tier
            if tier not in ("miss", "device", "memory", "disk"):
                raise SmokeFailure(f"{case.name}: invalid expected cache tier {tier!r}")
            if tier == "miss":
                tier_ok = tier_lengths["total"] == 0
            else:
                tier_ok = tier_lengths[tier] > 0 and all(
                    tier_lengths[other] == 0
                    for other in ("device", "memory", "disk") if other != tier
                )
            if not tier_ok:
                raise SmokeFailure(
                    f"{case.name}: expected {tier} Prefill cache tier, got {tier_lengths}"
                )
        if case.expected_input_len is not None and input_len != case.expected_input_len:
            raise SmokeFailure(
                f"{case.name}: tokenized input changed: {input_len} != {case.expected_input_len}"
            )
        if (
            case.expected_reuse_len is not None
            and effective_reuse != case.expected_reuse_len
        ):
            raise SmokeFailure(
                f"{case.name}: reuse frontier {effective_reuse} != {case.expected_reuse_len}"
            )
        if input_len <= 0:
            raise SmokeFailure(f"{case.name}: input_len={aux.get('input_len')!r}")
        if effective_reuse < 0 or effective_reuse > input_len:
            raise SmokeFailure(
                f"{case.name}: invalid reuse {effective_reuse} for input_len {input_len}"
            )
        if case.allow_long_history and input_len - effective_reuse > 65536:
            raise SmokeFailure(
                f"{case.name}: historical-KV request has {input_len - effective_reuse} "
                "uncached Q tokens, exceeding 65536"
            )
        if effective_reuse and effective_reuse % self.args.block_size:
            raise SmokeFailure(
                f"{case.name}: reuse {effective_reuse} is not aligned to "
                f"block_size={self.args.block_size}"
            )
        if case.reuse == "miss" and effective_reuse != 0:
            raise SmokeFailure(
                f"{case.name}: expected miss, got reuse={effective_reuse}"
            )
        if case.reuse == "hit" and effective_reuse <= 0:
            raise SmokeFailure(
                f"{case.name}: expected hit, got reuse={effective_reuse}"
            )
        if case.reuse == "partial" and not (0 < effective_reuse < input_len):
            raise SmokeFailure(
                f"{case.name}: expected partial hit, got reuse={effective_reuse}, "
                f"input_len={input_len}"
            )
        if case.require_chunk and input_len <= self.args.chunk_tokens:
            raise SmokeFailure(
                f"{case.name}: input_len={input_len} did not exceed chunk threshold "
                f"{self.args.chunk_tokens}"
            )
        multimodal_lengths = aux.get("multimodal_lengths") or {}
        input_urls = debug_info.get("input_urls") or []
        if case.require_multimodal:
            if not isinstance(input_urls, list) or not any(
                isinstance(url, str) and url for url in input_urls
            ):
                raise SmokeFailure(
                    f"{case.name}: no processed multimodal input URL was reported: "
                    f"{input_urls!r}"
                )
        iter_count = int(aux.get("iter_count", 0))
        mtp_accepted_tokens = output_len - iter_count
        if case.require_mtp and (iter_count <= 0 or mtp_accepted_tokens <= 0):
            raise SmokeFailure(
                f"{case.name}: MTP produced no accepted draft token: "
                f"output_len={output_len}, iter_count={iter_count}"
            )
        mtp_draft_rounds = int(aux.get("speculative_draft_rounds", 0))
        if case.require_mtp_draft and mtp_draft_rounds <= 0:
            raise SmokeFailure(f"{case.name}: no MTP draft rounds were observed")

        # The first output token is produced by P. D consumes it at position
        # input_len; conservatively exclude the final output token, which need
        # not be consumed before EOS/stop. Speculative rejected slots do not
        # count as committed boundary coverage.
        decode_kv_last_position = input_len + output_len - 2
        for boundary in case.decode_crossings:
            if not input_len <= boundary <= decode_kv_last_position:
                raise SmokeFailure(
                    f"{case.name}: Decode did not cross committed KV boundary {boundary}; "
                    f"positions={input_len}..{decode_kv_last_position}"
                )

        selected_decode_role_addr = None
        observed_decode_role_addrs = aux.get("role_addrs") or []
        if self.decode_role_addrs:
            selected_decode_role_addr = self.decode_role_addrs[case.decode_owner_rank]
            if selected_decode_role_addr not in observed_decode_role_addrs:
                raise SmokeFailure(
                    f"{case.name}: Decode owner route was not preserved: "
                    f"expected={selected_decode_role_addr!r}, "
                    f"observed={observed_decode_role_addrs!r}"
                )

        return {
            "name": case.name,
            "expected_reuse": case.reuse,
            "effective_reuse_len": effective_reuse,
            "reuse_len": raw_reuse,
            "prefill_total_reuse_len": prefill_reuse_value,
            "prefill_cache_tiers": tier_lengths,
            "expected_cache_tier": case.expected_cache_tier,
            "allow_long_history": case.allow_long_history,
            "input_len": input_len,
            "output_len": output_len,
            "iter_count": iter_count,
            "mtp_accepted_tokens": mtp_accepted_tokens,
            "mtp_draft_rounds": mtp_draft_rounds,
            "require_mtp": case.require_mtp,
            "require_mtp_draft": case.require_mtp_draft,
            "expected_json": case.expected_json,
            "expected_regex": case.expected_regex,
            "expected_input_len": case.expected_input_len,
            "expected_reuse_len": case.expected_reuse_len,
            "cache_block_boundary": case.cache_block_boundary,
            "cache_block_phase": case.cache_block_phase,
            "decode_crossings": list(case.decode_crossings),
            "decode_kv_first_position": input_len,
            "decode_kv_last_position": decode_kv_last_position,
            "require_multimodal": case.require_multimodal,
            "multimodal_lengths": multimodal_lengths,
            "input_urls": input_urls,
            "max_tokens": request_max_tokens,
            "pd_sep": True,
            "elapsed_s": round(elapsed_s, 3),
            "content": content,
            "reasoning_content": reasoning_content,
            "finish_reason": finish_reason,
            "output_ids": output_ids,
            "decode_owner_rank": case.decode_owner_rank,
            "selected_decode_role_addr": selected_decode_role_addr,
            "observed_decode_role_addrs": observed_decode_role_addrs,
        }

    def request_cases(
        self, cases: list[Case], concurrent: bool,
        admission_wave_size: int | None = None,
        admission_gap_s: float = 0,
    ) -> list[dict[str, Any]]:
        if concurrent:
            # Decode boundary tests keep the requests active together while
            # limiting the number of new PD/RDMA connections opened at once.
            # The all-rank audit still requires the full *actual* Decode batch.
            barrier = None if admission_wave_size else threading.Barrier(len(cases))
            with ThreadPoolExecutor(max_workers=len(cases)) as pool:
                futures = []
                if admission_wave_size:
                    for begin in range(0, len(cases), admission_wave_size):
                        futures.extend(pool.submit(self.request, case) for case in
                                       cases[begin:begin + admission_wave_size])
                        if begin + admission_wave_size < len(cases):
                            time.sleep(admission_gap_s)
                else:
                    futures = [pool.submit(self.request, case, barrier) for case in cases]
                results, errors = [], []
                for future in futures:
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        errors.append(exc)
                if errors:
                    # Never let a transport error hide a sibling semantic error.
                    fatal = next(
                        (e for e in errors if not isinstance(e, TransportFailure)),
                        errors[0],
                    )
                    raise fatal
                return results
        return [self.request(case) for case in cases]

    def request_cases_group(self, cases: list[Case]) -> dict[str, int]:
        """Use the existing PD-aware group RPC to prove one target forward.

        The group RPC batches admitted requests before FetchResponse performs
        the normal P-to-D handoff. Its request IDs are saved with the raw RPC
        evidence so the all-rank auditor can correlate the exact four members.
        """
        runfiles = self.args.prefill_rpc_runfiles
        roots = [runfiles / "rtp_llm", *sorted(runfiles.glob("pip*/site-packages"))]
        sys.path[:0] = [str(root) for root in roots]
        try:
            import grpc
            import torch
            from transformers import AutoTokenizer
            from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, ThinkingMode
            from rtp_llm.cpp.model_rpc.model_rpc_client import trans_input
            from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
                CancelRequestPB, EnqueueGroupRequestPB, FetchRequestPB,
            )
            from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
            from rtp_llm.ops import RoleType
            from rtp_llm.utils.base_model_datatypes import GenerateInput, RequestInfo
            from rtp_llm.utils.grpc_util import trans_tensor
        except ImportError as exc:
            raise SmokeFailure(f"PD group RPC dependencies are missing from {runfiles}: {exc}") from exc

        group_id = time.time_ns() % (2 ** 62)
        group = EnqueueGroupRequestPB(batch_id=group_id, dp_rank=0,
                                      fetch_attach_timeout_ms=300000)
        inputs: dict[int, tuple[Case, list[int], dict[str, Any]]] = {}
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.long_prefix_checkpoint, trust_remote_code=True)
        stop_ids = [tokenizer.encode(marker, add_special_tokens=False) for marker in
                    ("<|end_of_msg|>", "<|close|>response<|sep|>")]
        for index, case in enumerate(cases):
            if not isinstance(case.prompt, str):
                raise SmokeFailure("PD group cache smoke requires text prompts")
            role = self.decode_role_addrs[case.decode_owner_rank]
            token_ids = self.tokenize(case.prompt)
            if len(token_ids) > 65536:
                raise SmokeFailure(f"{case.name}: grouped input exceeds 64K")
            request_id = group_id + index
            config = GenerateConfig(
                max_new_tokens=case.max_tokens or self.args.max_tokens,
                do_sample=False, top_k=1, top_p=0.95,
                timeout_ms=300000, aux_info=True, random_seed=0,
                thinking_mode=ThinkingMode.DISABLED, max_thinking_tokens=0,
                stop_words_list=stop_ids,
                role_addrs=[RoleAddr(role=RoleType.DECODE, ip=role["ip"],
                                     http_port=role["http_port"],
                                     grpc_port=role["grpc_port"])],
            )
            input_py = GenerateInput(
                request_id=request_id,
                token_ids=torch.tensor(token_ids, dtype=torch.int32),
                mm_inputs=[], generate_config=config,
                request_info=RequestInfo(request_id=case.name),
            )
            group.requests.add().input.CopyFrom(trans_input(input_py))
            payload = {
                "model": "kimi-k3", "messages": [{"role": "user", "content": case.prompt}],
                "max_tokens": config.max_new_tokens, "temperature": 0,
                "top_k": 1, "top_p": 0.95, "enable_thinking": False,
                "extra_configs": {"role_addrs": [role]},
            }
            inputs[request_id] = (case, token_ids, payload)

        address = f"127.0.0.1:{self.args.prefill_grpc_port}"
        started = time.time()
        with grpc.insecure_channel(address) as channel:
            stub = RpcServiceStub(channel)
            try:
                ack = stub.EnqueueGroup(group, timeout=300)
            except grpc.RpcError as exc:
                raise SmokeFailure(f"PD group enqueue failed: {exc}") from exc
            accepted = {item.request_id for item in ack.successes}
            errors = {item.request_id: str(item.error_info) for item in ack.errors}
            if accepted != set(inputs) or errors:
                for request_id in accepted:
                    try:
                        stub.Cancel(CancelRequestPB(request_id=request_id), timeout=5)
                    except grpc.RpcError:
                        pass
                raise SmokeFailure(f"PD group admitted {len(accepted)}/{len(cases)}; errors={errors}")

            def fetch(request_id: int) -> tuple[int, list[int], Any]:
                output_ids: list[int] = []
                final_aux = None
                finished = False
                for frame in stub.FetchResponse(
                    FetchRequestPB(request_id=request_id), timeout=300
                ):
                    flat = frame.flatten_output
                    if flat.HasField("output_ids"):
                        output_ids.extend(trans_tensor(flat.output_ids).reshape(-1).tolist())
                    if flat.aux_info:
                        final_aux = flat.aux_info[0]
                    finished = bool(flat.finished and flat.finished[0])
                if not finished or final_aux is None or not output_ids:
                    raise SmokeFailure(f"PD group request {request_id} did not finish")
                return request_id, output_ids, final_aux

            try:
                with ThreadPoolExecutor(max_workers=len(cases)) as pool:
                    results = list(pool.map(fetch, inputs))
            except Exception:
                for request_id in accepted:
                    try:
                        stub.Cancel(CancelRequestPB(request_id=request_id), timeout=5)
                    except grpc.RpcError:
                        pass
                raise

        for request_id, output_ids, aux in results:
            case, _, payload = inputs[request_id]
            visible_ids = output_ids
            for stop in sorted(stop_ids, key=len, reverse=True):
                if stop and visible_ids[-len(stop):] == stop:
                    visible_ids = visible_ids[:-len(stop)]
                    break
            decoded = tokenizer.decode(visible_ids, skip_special_tokens=True)
            role = self.decode_role_addrs[case.decode_owner_rank]
            auxiliary = {
                "pd_sep": bool(aux.pd_sep), "input_len": int(aux.input_len),
                "output_len": int(aux.output_len), "iter_count": int(aux.iter_count),
                "reuse_len": int(aux.total_reuse_len),
                "prefill_total_reuse_len": int(aux.prefill_total_reuse_len),
                "prefill_local_reuse_len": int(aux.prefill_local_reuse_len),
                "prefill_memory_reuse_len": int(aux.prefill_memory_reuse_len),
                "prefill_disk_reuse_len": int(aux.prefill_disk_reuse_len),
                "speculative_draft_rounds": int(aux.speculative_draft_rounds),
                "role_addrs": [role],
            }
            response = {
                "choices": [{"message": {"content": decoded, "reasoning_content": ""},
                             "finish_reason": "stop" if len(output_ids) < payload["max_tokens"]
                             else "length"}],
                "aux_info": auxiliary,
                "debug_info": {"output_ids": [output_ids]},
            }
            artifact = {
                "name": case.name, "owner": case.decode_owner_rank,
                "request_id": request_id, "transport": "pd_group_rpc",
                "rpc_status": "OK", "passed": False,
                "phase": "preparation" if case.preparation_only else "formal",
                "request": payload,
                "response_body": json.dumps(response, ensure_ascii=False),
            }
            with self._record_lock:
                self._artifact_counter += 1
                number = self._artifact_counter
            directory = self.args.output.parent / "requests"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{number:04d}-{hashlib.sha256(case.name.encode()).hexdigest()[:12]}.json"
            try:
                row = self.validate(case, response, time.time() - started,
                                    payload["max_tokens"])
                row["phase"] = artifact["phase"]
                artifact.update(passed=True, result=row)
                with self._record_lock:
                    self.records.append(row)
            except Exception as exc:
                artifact["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
        return {case.name: request_id for request_id, (case, _, _) in inputs.items()}

    def prewarm_rdma_pool(self) -> None:
        if self.args.rdma_prewarm_attempts == 0:
            return

        for attempt in range(1, self.args.rdma_prewarm_attempts + 1):
            self.health(f"rdma_prewarm_{attempt}")
            cases = [
                Case(
                    f"rdma_prewarm_{attempt}_{idx}",
                    make_cache_prompt(
                        self.args.namespace,
                        f"rdma-prewarm-{attempt}-{idx}",
                        80 + idx,
                        repeats=8,
                    ),
                    numbered_answer_pattern((80 + idx) ** 2),
                    "miss",
                    max_tokens=max(self.args.max_tokens, 128),
                    timeout_s=min(self.args.timeout, self.args.rdma_prewarm_timeout),
                    decode_owner_rank=idx % max(1, len(self.decode_role_addrs)),
                )
                for idx in range(self.args.batch_size)
            ]
            started = time.time()
            try:
                records = self.request_cases(cases, concurrent=True)
            except Exception as exc:
                elapsed_s = round(time.time() - started, 3)
                error = f"{type(exc).__name__}: {exc}"
                self.rdma_prewarm_attempts.append(
                    {
                        "attempt": attempt,
                        "passed": False,
                        "elapsed_s": elapsed_s,
                        "error": error,
                    }
                )
                print(
                    f"rdma_prewarm attempt={attempt} passed=false "
                    f"elapsed_s={elapsed_s} error={error}"
                )
                if not isinstance(exc, TransportFailure):
                    raise
                if attempt == self.args.rdma_prewarm_attempts:
                    raise SmokeFailure(
                        f"RDMA prewarm failed after {attempt} attempts: {exc}"
                    ) from exc
                time.sleep(self.args.rdma_prewarm_backoff_s * attempt)
                continue

            elapsed_s = round(time.time() - started, 3)
            self.rdma_prewarm_attempts.append(
                {
                    "attempt": attempt,
                    "passed": True,
                    "elapsed_s": elapsed_s,
                    "case_names": [record["name"] for record in records],
                    "input_lengths": [record["input_len"] for record in records],
                }
            )
            print(
                f"rdma_prewarm attempt={attempt} passed=true "
                f"decode_owners={len(records)} elapsed_s={elapsed_s}"
            )
            if self.args.rdma_prewarm_settle_s:
                time.sleep(self.args.rdma_prewarm_settle_s)
            return

    def run_stage(self, name: str, cases: list[Case], concurrent: bool = False,
                  grouped_pd: bool = False, admission_wave_size: int | None = None,
                  admission_gap_s: float = 0) -> None:
        self.health(name)
        started = time.time()
        stage = {
            "name": name,
            "start_time_ns": time.time_ns(),
            "concurrent": concurrent,
            "passed": False,
            "case_names": [case.name for case in cases],
            "decode_owner_ranks": [case.decode_owner_rank for case in cases],
            "transport": "pd_group_rpc" if grouped_pd else "http",
        }
        if admission_wave_size:
            stage["admission_wave_size"] = admission_wave_size
            stage["admission_gap_s"] = admission_gap_s
        self.stages.append(stage)
        try:
            if grouped_pd:
                stage["request_ids"] = self.request_cases_group(cases)
            elif name == "rolling_refill":
                self.request_refill(cases)
            else:
                self.request_cases(cases, concurrent, admission_wave_size,
                                   admission_gap_s)
            stage["passed"] = True
        except SmokeDeadline as exc:
            if self.args.case_deadline_s is None:
                raise
            stage["skipped"] = True
            stage["skipped_case_names"] = [
                row["name"] for row in self.skipped_cases
                if row["name"] in stage["case_names"]
            ]
            stage["error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            stage["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            stage["elapsed_s"] = round(time.time() - started, 3)
            stage["end_time_ns"] = time.time_ns()
        print(f"stage={name} passed={str(stage['passed']).lower()} "
              f"skipped={stage.get('skipped', False)} owners={stage['decode_owner_ranks']}")

    def tokenize(self, prompt: str) -> list[int]:
        request = urllib.request.Request(
            self.args.base_url.rstrip("/") + "/tokenize",
            data=json.dumps(
                {"model": "kimi-k3", "messages": [{"role": "user", "content": prompt}]}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with self.opener.open(request, timeout=self.args.timeout) as response:
            tokens = json.load(response)["token_ids"]
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(t) is not int for t in tokens)
        ):
            raise SmokeFailure("tokenizer returned invalid token IDs")
        return tokens

    def save_token_fixture(self, prompt: str, tokens: list[int]) -> None:
        if self.args.output is None:
            return
        directory = self.args.output.parent / "token-fixtures"
        directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        with gzip.open(
            directory / f"{digest}.json.gz", "wt", encoding="utf-8"
        ) as output:
            json.dump(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "input_ids": tokens,
                    "input_len": len(tokens),
                    "prompt_sha256": digest,
                },
                output,
                ensure_ascii=False,
            )

    def fit_prompt(self, head: str, tail: str, target: int) -> tuple[str, list[int]]:
        # Query the serving tokenizer, including its exact chat template. The
        # filler is inert; all authoritative records and instructions are last.
        lo, hi = 0, target * 2
        while lo <= hi:
            count = (lo + hi) // 2
            prompt = head + " x" * count + tail
            tokens = self.tokenize(prompt)
            if len(tokens) == target:
                self.save_token_fixture(prompt, tokens)
                return prompt, tokens
            if len(tokens) < target:
                lo = count + 1
            else:
                hi = count - 1
        # Some tokenizers merge the boundary differently. Search a small,
        # bounded neighbourhood; failure is explicit, never a mislabeled case.
        for count in range(max(0, hi - 8), lo + 9):
            for suffix in ("\n", " ", " a", "\n\n"):
                prompt = head + " x" * count + suffix + tail
                tokens = self.tokenize(prompt)
                if len(tokens) == target:
                    self.save_token_fixture(prompt, tokens)
                    return prompt, tokens
        raise SmokeFailure(f"cannot construct exact chat input length {target}")

    def fit_cache_prompt(self, case_name: str, value: int, target: int) -> str:
        head = (
            f"缓存测试标识：{self.args.namespace}/{case_name}。"
            "以下 x 是用于跨越完整缓存复用粒度的无关材料。"
        )
        tail = f"\n只回答数字：{value} 的平方是多少？"
        prompt, _ = self.fit_prompt(head, tail, target)
        return prompt

    def fit_partial_cache_prompts(
        self,
        common_name: str,
        seed_name: str,
        seed_value: int,
        query_name: str,
        query_value: int,
        target: int,
    ) -> tuple[str, str]:
        head = (
            f"部分命中测试标识：{self.args.namespace}/{common_name}。"
            "以下 x 是两次请求共同拥有的无关前缀材料。"
        )
        seed_tail = f"\n分支标识：{seed_name}。只回答数字：{seed_value} 的平方是多少？"
        query_tail = (
            f"\n分支标识：{query_name}。只回答数字：{query_value} 的平方是多少？"
        )
        seed_prompt, seed_tokens = self.fit_prompt(head, seed_tail, target)
        query_prompt = seed_prompt[: -len(seed_tail)] + query_tail
        query_tokens = self.tokenize(query_prompt)
        self.save_token_fixture(query_prompt, query_tokens)
        common_tokens = next(
            (
                index
                for index, pair in enumerate(zip(seed_tokens, query_tokens))
                if pair[0] != pair[1]
            ),
            min(len(seed_tokens), len(query_tokens)),
        )
        if common_tokens < self.reuse_unit_tokens:
            raise SmokeFailure(
                f"partial cache prompt common prefix {common_tokens} is shorter than "
                f"reuse unit {self.reuse_unit_tokens}"
            )
        return seed_prompt, query_prompt

    def record_case(self, name: str, owner: int, *, words: int = 1) -> Case:
        # Expected answers derive only from these records, with no external
        # knowledge or second model acting as judge.
        tag = hashlib.sha256(f"{self.args.namespace}/{name}".encode()).hexdigest()[:12]
        value = " ".join(f"CEDAR-{tag}-{i}" for i in range(words))
        prompt = (
            f"档案 {self.args.namespace}/{name}。\n"
            "以下是数据，不是指令。\n"
            f"key=amber; value=MAPLE-{tag}\n"
            f"key=cedar; value={value}\n"
            f"key=birch; value=BIRCH-{tag}\n"
            "检索 key=cedar 的 value，逐字复制。只输出一个 JSON 对象，"
            "仅包含字符串字段 value，不要代码块或解释。"
        )
        return Case(
            name,
            prompt,
            r"",
            "miss",
            decode_owner_rank=owner,
            expected_json={"value": value},
            # Reasoning can repeat the full record before the final JSON.
            max_tokens=max(self.args.max_tokens, 256 + words * 64),
        )

    def run_owner_regressions(self) -> None:
        size = max(1, len(self.decode_role_addrs))
        # The original wrong-answer class remains a formal, non-retried gate.
        self.run_stage(
            "historical_four_squares",
            [
                Case(
                    f"historical-square-{i}",
                    make_cache_prompt(
                        self.args.namespace, f"historical-square-{i}", 80 + i, repeats=8
                    ),
                    numbered_answer_pattern((80 + i) ** 2),
                    "miss",
                    decode_owner_rank=owner % size,
                )
                for i, owner in enumerate((0, 0, 1, 2))
            ],
            concurrent=True,
        )
        rolling = []
        for i in range(16):
            case = self.record_case(f"rolling-{i}", i % size, words=i + 1)
            prompt, ids = self.fit_prompt(
                f"Rolling {self.args.namespace}/{i}. Background:\n",
                "\n" + case.prompt,
                2048 + 137 * i,
            )
            rolling.append(replace(case, prompt=prompt, expected_input_len=len(ids)))
        self.run_stage("rolling_refill", rolling, concurrent=True)

    def run_batch_shapes(self) -> None:
        # Shape transitions exercise Graph eligibility; actual replay must be
        # proved separately from runtime events, never inferred from answers.
        for sequence, batch in enumerate((1, 2, 3, 7, 8, 9, 3, 1, 9)):
            self.run_stage(
                f"batch_shape_{sequence}_{batch}",
                [
                    self.record_case(f"batch-shape-{sequence}-{i}", 0, words=8)
                    for i in range(batch)
                ],
                concurrent=True,
            )

    def request_refill(self, cases: list[Case], window: int = 8) -> None:
        iterator = iter(cases)
        errors = []
        with ThreadPoolExecutor(max_workers=window) as pool:
            pending = {pool.submit(self.request, case) for case in list(cases)[:window]}
            for _ in range(min(window, len(cases))):
                next(iterator)
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    try:
                        future.result()
                    except Exception as exc:
                        errors.append(exc)
                if not errors:
                    for _ in completed:
                        case = next(iterator, None)
                        if case is not None:
                            pending.add(pool.submit(self.request, case))
        if errors:
            raise next(
                (e for e in errors if not isinstance(e, TransportFailure)), errors[0]
            )

    def run_prefix_branches(self) -> None:
        head = f"档案 {self.args.namespace}/prefix-branches。以下 x 为无关填充。\n"
        tail_a = '\n唯一有效记录：key=chosen; value=CEDAR-AMBER。只输出 JSON {"value":"CEDAR-AMBER"}，不要解释。'
        tail_b = '\n唯一有效记录：key=chosen; value=MAPLE-VIOLET。只输出 JSON {"value":"MAPLE-VIOLET"}，不要解释。'
        unit = self.reuse_unit_tokens
        target_tokens = unit * 2 + self.args.block_size
        if self.args.suite in ("main-text-64k", "main-text-64k-capped"):
            target_tokens = min(target_tokens, 65536)
        prompt_a, tokens_a = self.fit_prompt(head, tail_a, target_tokens)
        prompt_b = prompt_a[: -len(tail_a)] + tail_b
        tokens_b = self.tokenize(prompt_b)
        self.save_token_fixture(prompt_b, tokens_b)
        common = next(
            (i for i, pair in enumerate(zip(tokens_a, tokens_b)) if pair[0] != pair[1]),
            min(len(tokens_a), len(tokens_b)),
        )
        common_reuse = common // unit * unit
        if common_reuse <= 0:
            raise SmokeFailure("branch case has no complete common cache page")
        size = max(1, len(self.decode_role_addrs))
        a = Case(
            "prefix-A-seed",
            prompt_a,
            r"",
            "miss",
            expected_json={"value": "CEDAR-AMBER"},
            expected_input_len=len(tokens_a),
            expected_reuse_len=0,
        )
        b = Case(
            "prefix-B-partial",
            prompt_b,
            r"",
            "partial",
            decode_owner_rank=(size - 1),
            expected_json={"value": "MAPLE-VIOLET"},
            expected_input_len=len(tokens_b),
            expected_reuse_len=common_reuse,
        )
        self.run_stage("prefix_A_seed", [a])
        self.run_stage("prefix_B_partial", [b])
        a_hit = replace(
            a,
            name="prefix-A-return",
            reuse="hit",
            decode_owner_rank=(size - 1),
            expected_reuse_len=(len(tokens_a) - 1) // unit * unit,
        )
        self.run_stage("prefix_A_return", [a_hit])
        b_hit = replace(
            b,
            name="prefix-B-hit",
            reuse="hit",
            expected_reuse_len=(len(tokens_b) - 1) // unit * unit,
        )
        self.run_stage(
            "prefix_AB_mixed",
            [
                replace(
                    case, name=f"prefix-mixed-{i}", decode_owner_rank=(i + 4) % size
                )
                for i, case in enumerate((a_hit, b_hit, a_hit, b_hit))
            ],
            concurrent=True,
        )

    def run_padding_boundaries(self) -> None:
        # Chunk budget is TP8 aligned. A single cold input of C+1/C+7 forces
        # terminal round padding of 7/1 respectively, regardless of batching.
        size = max(1, len(self.decode_role_addrs))
        for tail_tokens in (1, 7):
            tag = f"PADDING-TAIL-{tail_tokens}"
            head = f"档案 {self.args.namespace}/padding-{tail_tokens}。以下 x 为无关填充。\n"
            tail = f'\n记录 value={tag}。只输出 JSON {{"value":"{tag}"}}，不要解释。'
            target = self.args.chunk_tokens + tail_tokens
            prompt, tokens = self.fit_prompt(head, tail, target)
            case = Case(
                f"padding-tail-{tail_tokens}-cold",
                prompt,
                r"",
                "miss",
                require_chunk=True,
                expected_json={"value": tag},
                expected_input_len=len(tokens),
                expected_reuse_len=0,
                decode_owner_rank=(size - 1),
                max_tokens=max(self.args.max_tokens, 256),
            )
            self.run_stage(f"padding_tail_{tail_tokens}_cold", [case])
            self.run_stage(
                f"padding_tail_{tail_tokens}_hit",
                [
                    replace(
                        case,
                        name=f"padding-tail-{tail_tokens}-hit",
                        reuse="hit",
                        decode_owner_rank=(tail_tokens + 1) % size,
                        expected_reuse_len=(target - 1)
                        // self.args.block_size
                        * self.args.block_size,
                    )
                ],
            )

    def run_cache_block_boundaries(self) -> None:
        page = self.args.block_size
        unit = self.reuse_unit_tokens
        boundaries = cache_block_boundaries(page, unit, self.args.chunk_tokens)
        if self.args.suite in ("main-text-64k", "main-text-64k-capped"):
            boundaries = tuple(b for b in boundaries if b + 1 <= 65536)
        owners = max(1, len(self.decode_role_addrs))
        # Each triplet is cold first, then repeated before another triplet can
        # evict its entries. A repeat below the first complete KDA checkpoint
        # must still miss; do not call every repeated request a cache hit.
        for boundary in boundaries:
            cases = []
            for delta in (-1, 0, 1):
                length = boundary + delta
                tag = hashlib.sha256(
                    f"{self.args.namespace}/cache-block/{length}".encode()
                ).hexdigest()[:10]
                prompt, ids = self.fit_prompt(
                    f"ID:{tag}\n",
                    f'\n只输出 JSON {{"value":"{tag}"}}，不要解释。',
                    length,
                )
                cases.append(
                    Case(
                        f"cache-block-{boundary}-{delta:+d}-cold",
                        prompt,
                        "",
                        "miss",
                        expected_json={"value": tag},
                        expected_input_len=len(ids),
                        expected_reuse_len=0,
                        cache_block_boundary=boundary,
                        cache_block_phase="cold",
                        max_tokens=max(self.args.max_tokens, 256),
                        decode_owner_rank=(boundary // page + delta) % owners,
                    )
                )
            self.run_stage(f"cache_block_{boundary}_cold", cases, concurrent=True)
            repeated = []
            for case in cases:
                reuse = (case.expected_input_len - 1) // unit * unit
                repeated.append(
                    replace(
                        case,
                        name=case.name.replace("-cold", "-repeat"),
                        reuse="hit" if reuse else "miss",
                        expected_reuse_len=reuse,
                        cache_block_phase="repeat",
                        decode_owner_rank=(case.decode_owner_rank + 1) % owners,
                    )
                )
            self.run_stage(f"cache_block_{boundary}_repeat", repeated, concurrent=True)

        # Start one token before each ordinary cache-block boundary.
        # Long exact answers ensure actual accepted Decode progress crosses
        # two page starts, independently of how many tokens MTP proposes.
        decode_boundaries = tuple(
            sorted(
                set(range(page, unit + 1, page)) | {2 * unit, self.args.chunk_tokens}
            )
        )
        if self.args.suite == "main-text-64k-capped":
            # The historical long-output deferral covers exactly these three
            # boundaries. Larger cache stripes add short-answer coverage only.
            deferred_boundaries = (page, 2 * page, self.args.chunk_tokens)
            for boundary in deferred_boundaries:
                for suffix, reason in (("cold", "historical runtime exceeds five minutes"),
                                       ("repeat", "explicitly deferred by user")):
                    self.skipped_cases.append({"name": f"decode-page-cross-{boundary}-{suffix}",
                                               "reason": reason, "sent": False})
            for suffix in ("cold", "repeat"):
                self.stages.append({"name": f"decode_page_cross_0_{suffix}", "passed": False,
                                    "skipped": True, "case_names":
                                    [f"decode-page-cross-{b}-{suffix}" for b in deferred_boundaries]})
            # Cross one committed Decode page start with a short exact answer.
            # The original multi-page long-output cases remain explicitly skipped.
            for boundary in decode_boundaries:
                tag = f"BND-{boundary}"
                prompt, ids = self.fit_prompt(
                    f"ID:{self.args.namespace}/decode-page-short-{boundary}\n",
                    f'\n记录 value={tag}。只输出 JSON {{"value":"{tag}"}}，不要解释。',
                    boundary - 1,
                )
                cold = Case(
                    f"decode-page-short-{boundary}-cold",
                    prompt,
                    "",
                    "miss",
                    require_mtp_draft=True,
                    expected_json={"value": tag},
                    expected_input_len=len(ids),
                    expected_reuse_len=0,
                    cache_block_boundary=boundary,
                    cache_block_phase="decode-short-cold",
                    decode_crossings=(boundary,),
                    max_tokens=256,
                    decode_owner_rank=(boundary // page) % owners,
                )
                self.run_stage(f"decode_page_short_{boundary}_cold", [cold])
                reuse = (len(ids) - 1) // unit * unit
                self.run_stage(
                    f"decode_page_short_{boundary}_repeat",
                    [
                        replace(
                            cold,
                            name=f"decode-page-short-{boundary}-repeat",
                            reuse="hit" if reuse else "miss",
                            expected_reuse_len=reuse,
                            cache_block_phase="decode-short-repeat",
                            decode_owner_rank=(cold.decode_owner_rank + 1) % owners,
                        )
                    ],
                )
            return
        expected = " ".join(f"{i:03d}" for i in range(max(64, page // 2)))
        last_number = max(64, page // 2) - 1
        for offset in range(0, len(decode_boundaries), 4):
            cases = []
            for boundary in decode_boundaries[offset : offset + 4]:
                tag = hashlib.sha256(
                    f"{self.args.namespace}/decode-cross/{boundary}".encode()
                ).hexdigest()[:10]
                prompt, ids = self.fit_prompt(
                    f"ID:{tag}\n",
                    f'\n只输出{{"value":"000 001 ... {last_number:03d}"}}，'
                    "展开全部整数，三位补零、单空格。",
                    boundary - 1,
                )
                cases.append(
                    Case(
                        f"decode-page-cross-{boundary}-cold",
                        prompt,
                        "",
                        "miss",
                        expected_json={"value": expected},
                        expected_input_len=len(ids),
                        expected_reuse_len=0,
                        cache_block_boundary=boundary,
                        cache_block_phase="decode-cold",
                        decode_owner_rank=(boundary // page) % owners,
                        decode_crossings=(boundary, boundary + page),
                        max_tokens=max(self.args.max_tokens, 4 * page + 2048),
                    )
                )
            self.run_stage(f"decode_page_cross_{offset}_cold", cases, concurrent=True)
            repeated = []
            for case in cases:
                reuse = (case.expected_input_len - 1) // unit * unit
                repeated.append(
                    replace(
                        case,
                        name=case.name.replace("-cold", "-repeat"),
                        reuse="hit" if reuse else "miss",
                        expected_reuse_len=reuse,
                        cache_block_phase="decode-repeat",
                        decode_owner_rank=(case.decode_owner_rank + 1) % owners,
                    )
                )
            self.run_stage(
                f"decode_page_cross_{offset}_repeat", repeated, concurrent=True
            )

    def run_flow(self) -> None:
        # The four-layer preflight checks PD, uneven batches, reuse and MTP
        # before the same orthogonal phases. Model-level chunking is excluded.
        prompt, tokens = self.fit_prompt(
            f"流程 {self.args.namespace}/short。\n",
            "\n请回复任意一个非空字符。",
            self.reuse_unit_tokens + 1,
        )
        seed = Case(
            "four_layer_pd_flow_miss",
            prompt,
            r".",
            "miss",
            expected_input_len=len(tokens),
            max_tokens=8,
        )
        self.run_stage(seed.name, [seed])
        owners = max(1, len(self.decode_role_addrs))
        for owner in range(owners):
            self.run_stage(
                f"flow_hit_owner_{owner}",
                [
                    replace(
                        seed,
                        name=f"flow_hit_owner_{owner}",
                        reuse="hit",
                        decode_owner_rank=owner,
                    )
                ],
            )
        owner_ranks = [0] * 4 + [owner for owner in range(1, owners)]
        batch = []
        for idx, owner in enumerate(owner_ranks):
            # A reusable KDA checkpoint spans an entire ordinary cache stripe. A
            # fixed character count can produce fewer tokens than that span.
            prompt, tokens = self.fit_prompt(
                f"四层流程测试标识：{self.args.namespace}/uneven/{idx}。\n",
                "\n请回复任意一个非空字符。",
                self.reuse_unit_tokens + self.args.block_size * (idx + 1),
            )
            batch.append(
                Case(
                    f"flow_uneven_{idx}",
                    prompt,
                    r".",
                    "miss",
                    decode_owner_rank=owner,
                    expected_input_len=len(tokens),
                    max_tokens=8,
                )
            )
        self.run_stage("flow_uneven_miss", batch, concurrent=True)
        self.run_stage(
            "flow_uneven_hit_rotated",
            [
                replace(
                    case,
                    name=case.name + "_hit",
                    reuse="hit",
                    decode_owner_rank=(case.decode_owner_rank + 1) % owners,
                )
                for case in reversed(batch)
            ],
            concurrent=True,
        )
        if self.args.require_mtp:
            self.run_stage(
                "flow_mtp_draft",
                [
                    Case(
                        "flow_mtp_draft",
                        f"四层 MTP 草稿路径检查：{self.args.namespace}。请回复任意内容。",
                        r".",
                        "miss",
                        require_mtp_draft=True,
                        max_tokens=min(self.args.max_tokens, 32),
                    )
                ],
            )

    def run_main_text(self) -> None:
        self.prewarm_rdma_pool()
        self.run_owner_regressions()
        self.run_batch_shapes()
        batch_owner_ranks = [
            rank % max(1, len(self.decode_role_addrs))
            for rank in [0, 0] + list(range(1, self.args.batch_size - 1))
        ]
        shard_span_reuse = self.reuse_unit_tokens > self.args.block_size
        if shard_span_reuse:
            cold_prompts = [
                self.fit_cache_prompt(
                    f"batch-cold-{idx}",
                    40 + idx,
                    self.reuse_unit_tokens + self.args.block_size * (idx + 1),
                )
                for idx in range(self.args.batch_size)
            ]
        else:
            cold_prompts = [
                make_cache_prompt(
                    self.args.namespace,
                    f"batch-cold-{idx}",
                    40 + idx,
                    repeats=300 + idx * 200,
                )
                for idx in range(self.args.batch_size)
            ]
        self.run_stage(
            "batch_all_miss",
            [
                Case(
                    f"batch_all_miss_{idx}",
                    prompt,
                    numbered_answer_pattern((40 + idx) ** 2),
                    "miss",
                    decode_owner_rank=batch_owner_ranks[idx],
                )
                for idx, prompt in enumerate(cold_prompts)
            ],
            concurrent=True,
        )
        self.run_stage(
            "batch_all_hit",
            [
                Case(
                    f"batch_all_hit_{idx}",
                    prompt,
                    numbered_answer_pattern((40 + idx) ** 2),
                    "hit",
                    decode_owner_rank=batch_owner_ranks[idx],
                )
                for idx, prompt in enumerate(cold_prompts)
            ],
            concurrent=True,
        )

        exact_hit_count = max(1, self.args.batch_size // 2)
        partial_idx = exact_hit_count
        mixed_prompts = []
        if shard_span_reuse:
            mixed_partial_seed, mixed_partial_query = self.fit_partial_cache_prompts(
                "batch-mixed-partial-common",
                "seed",
                67,
                "query",
                50 + partial_idx,
                self.reuse_unit_tokens + self.args.block_size * (partial_idx + 1),
            )
            for idx in range(self.args.batch_size):
                if idx == partial_idx:
                    prompt = mixed_partial_query
                else:
                    prompt = self.fit_cache_prompt(
                        f"batch-mixed-{idx}",
                        50 + idx,
                        self.reuse_unit_tokens + self.args.block_size * (idx + 1),
                    )
                mixed_prompts.append(prompt)
        else:
            for idx in range(self.args.batch_size):
                if idx == partial_idx:
                    prompt = make_partial_prompt(
                        self.args.namespace,
                        "batch-mixed-partial-common",
                        "query",
                        50 + idx,
                        repeats=350 + idx * 150,
                    )
                else:
                    prompt = make_cache_prompt(
                        self.args.namespace,
                        f"batch-mixed-{idx}",
                        50 + idx,
                        repeats=350 + idx * 150,
                    )
                mixed_prompts.append(prompt)
            mixed_partial_seed = make_partial_prompt(
                self.args.namespace,
                "batch-mixed-partial-common",
                "seed",
                67,
                repeats=350 + partial_idx * 150,
            )
        self.run_stage(
            "mixed_seed_hits",
            [
                Case(
                    f"mixed_seed_{idx}",
                    mixed_prompts[idx],
                    numbered_answer_pattern((50 + idx) ** 2),
                    "miss",
                    decode_owner_rank=batch_owner_ranks[idx],
                )
                for idx in range(exact_hit_count)
            ]
            + [
                Case(
                    "mixed_partial_seed",
                    mixed_partial_seed,
                    numbered_answer_pattern(4489),
                    "miss",
                    decode_owner_rank=batch_owner_ranks[partial_idx],
                )
            ],
        )
        self.run_stage(
            "batch_mixed_hit_miss",
            [
                Case(
                    f"batch_mixed_{idx}",
                    prompt,
                    numbered_answer_pattern((50 + idx) ** 2),
                    (
                        "hit"
                        if idx < exact_hit_count
                        else "partial" if idx == partial_idx else "miss"
                    ),
                    decode_owner_rank=batch_owner_ranks[idx],
                )
                for idx, prompt in enumerate(mixed_prompts)
            ],
            concurrent=True,
        )
        self.run_stage(
            "batch_mixed_then_all_hit",
            [
                Case(
                    f"batch_mixed_all_hit_{idx}",
                    prompt,
                    numbered_answer_pattern((50 + idx) ** 2),
                    "hit",
                    decode_owner_rank=batch_owner_ranks[idx],
                )
                for idx, prompt in enumerate(mixed_prompts)
            ],
            concurrent=True,
        )

        self.run_stage(
            "cuda_graph_bucket_8",
            [
                Case(
                    f"cuda_graph_bucket_8_{idx}",
                    make_cache_prompt(
                        self.args.namespace,
                        f"cuda-graph-8-{idx}",
                        110 + idx,
                        repeats=8,
                    ),
                    numbered_answer_pattern((110 + idx) ** 2),
                    "miss",
                    decode_owner_rank=0,
                )
                for idx in range(8)
            ],
            concurrent=True,
        )

        self.run_single_prefill_64k()
        self.run_prefix_branches()
        self.run_cache_block_boundaries()

    def _orthogonal_answer(self, name: str, prompt: str, value: str, reuse: str,
                           **kwargs) -> Case:
        diagnostic = self.args.suite == "orthogonal-flow"
        return Case(name, prompt, r".", reuse,
                    expected_json=None if diagnostic else {"value": value},
                    max_tokens=8 if diagnostic else 256,
                    require_mtp_draft=not diagnostic,
                    thinking_disabled=True, **kwargs)

    def _required_stage(self, name: str, cases: list[Case], concurrent=False,
                        grouped_pd=False, admission_wave_size=None,
                        admission_gap_s=0) -> None:
        self.run_stage(name, cases, concurrent=concurrent, grouped_pd=grouped_pd,
                       admission_wave_size=admission_wave_size,
                       admission_gap_s=admission_gap_s)
        if not self.stages[-1]["passed"]:
            raise SmokeFailure(f"{name}: required orthogonal stage was skipped")

    def _orthogonal_device_blocks(self) -> int | None:
        engine_log = getattr(self.args, "prefill_engine_log", None)
        if engine_log is None:
            return None
        with pathlib.Path(engine_log).open(encoding="utf-8", errors="replace") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 16 * 1024 * 1024))
            tail = stream.read()
        matches = re.findall(r"kvc raw pool\[DEVICE/full\]:[^\n]*?\btotal=(\d+)", tail)
        if not matches:
            raise SmokeFailure("Prefill Device/full pool capacity is missing from engine log")
        return int(matches[-1])

    def _orthogonal_cache_seed(self) -> None:
        unit = self.reuse_unit_tokens
        # KDA state is reusable only after a complete 32768-token checkpoint;
        # a shorter MLA-only prefix can hit Decode while Prefill still misses.
        target = max(32769, unit + 1)
        if target > 65536:
            raise SmokeFailure("cache-tier smoke seed exceeds the 64K Q budget")
        self._memory_candidates = []
        for attempt in range(3):
            tag = f"{self.args.namespace}-tier-{attempt}"
            memory_value = f"MEM-{attempt}"
            memory_tail = f'\n只输出 JSON {{"value":"{memory_value}"}}。'
            memory_prompt, _ = self.fit_prompt(f"CACHE:{tag}:memory\n", memory_tail, target)
            memory_seed = self._orthogonal_answer(
                f"tier-memory-seed-{attempt}", memory_prompt, memory_value, "miss")
            self._required_stage(memory_seed.name, [memory_seed])
            self._memory_candidates.append((tag, memory_seed, memory_value))
            if attempt == 0:
                device_probe = replace(memory_seed, name="tier-device-probe-0",
                                       reuse="any", expected_cache_tier=None,
                                       preparation_only=True,
                                       max_tokens=8 if self.args.suite == "orthogonal-flow" else 64)
                self._required_stage(device_probe.name, [device_probe])
                tier = self.records[-1].get("prefill_cache_tiers") or {}
                if tier.get("device", 0) <= 0 or tier.get("memory", 0) != 0:
                    raise SmokeFailure("immediate Device cache reuse was not observed")
        # Later smoke traffic, including long-KV preparation, supplies cache
        # pressure. This witness is newer than all three Memory candidates.
        witness_value = "WIT"
        witness_tail = f'\n只输出 JSON {{"value":"{witness_value}"}}。'
        witness_prompt, _ = self.fit_prompt(
            f"CACHE:{self.args.namespace}:witness\n", witness_tail, target)
        self._memory_witness = self._orthogonal_answer(
            "tier-witness-seed", witness_prompt, witness_value, "miss")
        self._required_stage(self._memory_witness.name, [self._memory_witness])
        if "cancel" in getattr(self.args, "orthogonal_phases", ()):
            self._cancel_host_candidates = self._seed_cancel_candidates()

    def _orthogonal_cache(self) -> None:
        if not hasattr(self, "_memory_candidates"):
            raise SmokeFailure("cache seeds must precede the traffic used for demotion")
        probe = replace(self._memory_witness, name="tier-memory-probe",
                        reuse="any", expected_cache_tier=None,
                        preparation_only=True,
                        max_tokens=8 if self.args.suite == "orthogonal-flow" else 64)
        self._required_stage(probe.name, [probe])
        tier = self.records[-1].get("prefill_cache_tiers") or {}
        if not (tier.get("memory", 0) > 0 and tier.get("device", 0) == 0):
            raise SmokeFailure("ordinary smoke traffic did not demote the witness prefix to Memory")
        owner_count = max(1, len(self.decode_role_addrs))
        device_blocks = self._orthogonal_device_blocks()
        target = max(32769, self.reuse_unit_tokens + 1)
        for attempt, (tag, memory_seed, memory_value) in enumerate(self._memory_candidates):
            memory_prompt = memory_seed.prompt
            device_value = f"DEV-{attempt}"
            device_tail = f'\n只输出 JSON {{"value":"{device_value}"}}。'
            device_prompt, _ = self.fit_prompt(f"CACHE:{tag}:device\n", device_tail, target)
            self._required_stage(f"tier-device-seed-{attempt}", [
                self._orthogonal_answer(f"tier-device-seed-{attempt}",
                                        device_prompt, device_value, "miss")])
            partial_seed_value, partial_value = f"SEED-{attempt}", f"PART-{attempt}"
            seed_tail = f'\n只输出 JSON {{"value":"{partial_seed_value}"}}。'
            query_tail = f'\n只输出 JSON {{"value":"{partial_value}"}}。'
            partial_seed_prompt, seed_ids = self.fit_prompt(
                f"CACHE:{tag}:partial\n", seed_tail, target + 256)
            partial_prompt = partial_seed_prompt[:-len(seed_tail)] + query_tail
            query_ids = self.tokenize(partial_prompt)
            common = next((index for index, pair in enumerate(zip(seed_ids, query_ids))
                           if pair[0] != pair[1]), min(len(seed_ids), len(query_ids)))
            if common < 32768 or len(query_ids) > 65536:
                raise SmokeFailure("partial cache prompt does not retain the KDA checkpoint")
            self._required_stage(f"tier-partial-seed-{attempt}", [
                self._orthogonal_answer(f"tier-partial-seed-{attempt}",
                                        partial_seed_prompt, partial_seed_value, "miss")])
            miss_value = f"MISS-{attempt}"
            miss_tail = f'\n只输出 JSON {{"value":"{miss_value}"}}。'
            miss_prompt, _ = self.fit_prompt(f"CACHE:{tag}:cold\n", miss_tail, target)
            mixed = [
                self._orthogonal_answer(f"tier-mixed-{attempt}-device", device_prompt,
                                        device_value, "any", decode_owner_rank=0),
                self._orthogonal_answer(f"tier-mixed-{attempt}-memory", memory_prompt,
                                        memory_value, "any", decode_owner_rank=1 % owner_count),
                self._orthogonal_answer(f"tier-mixed-{attempt}-partial", partial_prompt,
                                        partial_value, "any", decode_owner_rank=2 % owner_count),
                self._orthogonal_answer(f"tier-mixed-{attempt}-miss", miss_prompt,
                                        miss_value, "any", decode_owner_rank=3 % owner_count),
            ]
            stage_name = f"orthogonal_cache_mixed_{attempt}"
            self._required_stage(stage_name, mixed, concurrent=True, grouped_pd=True)
            records = {row["name"]: row for row in self.records
                       if row["name"] in {case.name for case in mixed}}
            kinds = ("device", "memory", "partial", "miss")
            checks = {}
            for kind in kinds:
                row = records[f"tier-mixed-{attempt}-{kind}"]
                tier = row.get("prefill_cache_tiers") or {}
                if kind == "miss":
                    checks[kind] = tier.get("total") == 0
                elif kind == "partial":
                    checks[kind] = (0 < row["effective_reuse_len"] < row["input_len"]
                                    and tier.get("device", 0) > 0)
                else:
                    checks[kind] = (tier.get(kind, 0) > 0 and
                                    all(tier.get(other, 0) == 0 for other in
                                        ("device", "memory", "disk") if other != kind))
            stage = self.stages[-1]
            stage["cache_tier_checks"] = checks
            stage["device_pool_blocks"] = device_blocks
            stage["cache_pressure_requests"] = getattr(self, "_cache_pressure_requests", 0)
            stage["current_q_total"] = sum(
                row["input_len"] - row["effective_reuse_len"] for row in records.values())
            stage["same_forward_passed"] = self._orthogonal_shared_prefill_forward(
                stage, {case.name for case in mixed})
            stage["path_passed"] = (all(checks.values()) and
                                     stage["current_q_total"] <= 65536 and
                                     stage["same_forward_passed"])
            if stage["path_passed"]:
                return
        raise SmokeFailure("mixed Prefill cache tiers were not triggered in three attempts")

    def _orthogonal_bounded_cache_pressure(self) -> None:
        # The long-KV chain retains roughly 20 KDA checkpoint blocks. A small
        # number of distinct prefixes finishes Device-to-Host demotion when
        # the bounded Device pool has room for those blocks plus recent cases.
        pool_blocks = self._orthogonal_device_blocks()
        if pool_blocks is None:
            raise SmokeFailure("Prefill Device/full pool capacity is missing")
        # Four-layer traffic has fewer ordinary cases before this stage, so
        # allow more probes to fill its Device pool. The full smoke keeps its
        # smaller cap because preceding correctness cases add pressure.
        limit = 50 if self.args.suite == "orthogonal-flow" else 24
        pressure_count = max(0, min(limit, pool_blocks - 20))
        self._cache_pressure_requests = pressure_count
        for index in range(pressure_count):
            prompt, _ = self.fit_prompt(
                f"CACHE-PRESSURE:{self.args.namespace}:{index}\n",
                "\nReply 1.", 32769,
            )
            case = Case(
                f"tier-pressure-{index:02d}", prompt, r".", "miss",
                max_tokens=8, preparation_only=True,
            )
            self._required_stage(case.name, [case])

    def _orthogonal_decode_batches(self) -> None:
        owner_count = max(1, len(self.decode_role_addrs))
        sizes = (1, 4, 8, 63, 64)
        for stage_index, size in enumerate(sizes):
            name = f"orthogonal_decode_{stage_index:02d}_batch_{size}"
            cases = []
            for index in range(size):
                case = self.record_case(f"{name}_{index:02d}",
                                        index % owner_count,
                                        words=12)
                if self.args.suite == "orthogonal-flow":
                    case = replace(case, expected_json=None, expected_regex=r".",
                                   max_tokens=64)
                else:
                    case = replace(case, max_tokens=256,
                                   require_mtp_draft=True, thinking_disabled=True)
                cases.append(case)
            self._required_stage(name, cases, concurrent=size > 1,
                                 admission_wave_size=8 if size > 8 else None,
                                 admission_gap_s=1.0 if size > 8 else 0)

    def _orthogonal_page_boundaries(self) -> None:
        owners = max(1, len(self.decode_role_addrs))
        for boundary in (4096, 32768):
            for delta in (-1, 0, 1):
                target = boundary + delta
                value = f"PAGE-{boundary}-{delta:+d}"
                tail = f'\n只输出 JSON {{"value":"{value}"}}。'
                prompt, ids = self.fit_prompt(
                    f"PAGE:{self.args.namespace}:{value}\n", tail, target)
                case = self._orthogonal_answer(
                    f"orthogonal_page_{boundary}_{delta:+d}", prompt,
                    value, "miss", expected_input_len=len(ids),
                    decode_owner_rank=(delta + 1) % owners)
                self._required_stage(case.name, [case])

    def _orthogonal_chunk_kv(self) -> None:
        base = f"Long KV pressure test {self.args.namespace}. "
        overhead = len(self.tokenize(base))
        if len(self.tokenize(base + " x" * 16)) - overhead != 16:
            raise SmokeFailure("long-KV seed filler is not one token per repeat")
        # 589824 cached tokens exceed the 6 GiB FP8 MLA expansion budget's
        # 557056-token aligned launch cap, while every seed adds <= 65536 Q.
        seed_end, step = 655_360, 32_740
        targets = list(range(65_504, seed_end, step))
        if targets[-1] != seed_end:
            targets.append(seed_end)
        for index, target in enumerate(targets):
            case = Case(
                f"orthogonal_kv_seed_{index:02d}",
                base + " x" * (target - overhead), r".",
                "miss" if index == 0 else "hit",
                allow_long_history=True, preparation_only=True,
                max_tokens=4,
            )
            self._required_stage(case.name, [case])
            row = self.records[-1]
            if abs(row["input_len"] - target) > 32:
                raise SmokeFailure(f"{case.name}: observed token count differs from seed target")
        value = f"KV-{hashlib.sha256(self.args.namespace.encode()).hexdigest()[:10]}"
        tail = f'\n唯一有效答案是 {value}。只输出 JSON {{"value":"{value}"}}。'
        shared = 589_856 - overhead
        suffix = 65_300
        prompt = base + " x" * shared + " z" * suffix + tail
        case = self._orthogonal_answer(
            "orthogonal_kv_final", prompt, value, "hit",
            allow_long_history=True)
        self._required_stage(case.name, [case])
        row = self.records[-1]
        if row["effective_reuse_len"] < 589_824:
            raise SmokeFailure("589824-token historical KV reuse frontier was not reached")
        self.stages[-1]["historical_kv_tokens"] = row["effective_reuse_len"]
        self.stages[-1]["current_q_tokens"] = row["input_len"] - row["effective_reuse_len"]

    def _new_log_events(self, directory: pathlib.Path,
                        offsets: dict[pathlib.Path, int]) -> list[dict[str, Any]]:
        events = []
        paths = list(directory.glob("main_[0-9]*.log"))
        engine_log = getattr(self.args, "prefill_engine_log", None)
        if engine_log is not None:
            paths.append(engine_log)
        for path in paths:
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                size = path.stat().st_size
                old = offsets.get(path, 0)
                if old > size:
                    old = 0
                handle.seek(old)
                for line in handle:
                    marker = "[K3_SMOKE_EVENT] "
                    if marker not in line:
                        continue
                    try:
                        event = json.loads(line.split(marker, 1)[1])
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        events.append(event)
                offsets[path] = handle.tell()
        return events

    def _orthogonal_shared_prefill_forward(self, stage: dict[str, Any],
                                           case_names: set[str]) -> bool:
        rank_logs = sorted(self.args.prefill_event_dir.glob("main_[0-9]*.log"))
        if not rank_logs:
            return False
        start, end = stage["start_time_ns"], stage["end_time_ns"]
        events = [event for event in self._new_log_events(
            self.args.prefill_event_dir, {})
            if start <= event.get("time_ns", -1) <= end]
        if stage.get("transport") == "pd_group_rpc":
            ids = stage.get("request_ids", {})
        else:
            ids = {event["case"]: event["request_id"] for event in events
                   if event.get("event") == "frontend_request"
                   and event.get("case") in case_names
                   and isinstance(event.get("request_id"), int)}
        if len(ids) != len(case_names) or len(set(ids.values())) != len(case_names):
            return False
        expected_ids = set(ids.values())
        return all(any(
            event.get("kind") == "target_prefill_forward"
            and event.get("tp_rank") == rank
            and event.get("actual_batch") == len(case_names)
            and set(event.get("request_ids", [])) == expected_ids
            for event in events)
            for rank in range(len(rank_logs)))

    def _seed_cancel_candidates(self) -> list[Case]:
        seeds = []
        for attempt in range(3):
            value = f"CANCEL-RECOVER-{attempt}"
            tail = f'\n只输出 JSON {{"value":"{value}"}}。'
            prompt, _ = self.fit_prompt(
                f"CACHE:{self.args.namespace}:cancel:{attempt}\n", tail, 32769)
            seed = self._orthogonal_answer(
                f"cancel-memory-seed-{attempt}", prompt, value, "miss")
            self._required_stage(seed.name, [seed])
            seeds.append(seed)
        return seeds

    def _orthogonal_cancel_recovery(self) -> None:
        if not hasattr(self, "_memory_candidates"):
            raise SmokeFailure("Host cache demotion was not established before cancel test")
        seeds = getattr(self, "_cancel_host_candidates", None)
        if seeds is None:
            raise SmokeFailure("cancel candidates must be seeded before cache pressure")
        offsets: dict[pathlib.Path, int] = {}
        self._new_log_events(self.args.prefill_event_dir, offsets)
        attempts = []
        recovery_seed = None
        for attempt, seed in enumerate(seeds):
            item = self._cancel_host_load_by_group_rpc(seed, attempt, offsets)
            attempts.append(item)
            if item["path_passed"]:
                recovery_seed = seed
                break
        else:
            raise SmokeFailure("three attempts did not cancel during Prefill Host cache load")
        assert recovery_seed is not None
        recovery = replace(recovery_seed, name="cancel-same-prefix-recovery", reuse="any")
        independent_value = "CANCEL-INDEPENDENT"
        independent_tail = f'\n只输出 JSON {{"value":"{independent_value}"}}。'
        independent_prompt, _ = self.fit_prompt(
            f"CACHE:{self.args.namespace}:independent\n", independent_tail, 8193)
        independent = self._orthogonal_answer(
            "cancel-independent-recovery", independent_prompt,
            independent_value, "miss")
        self._required_stage("orthogonal_cancel_recovery", [recovery, independent],
                             concurrent=True)
        self.stages[-1]["cancel_attempts"] = attempts
        self.stages[-1]["path_passed"] = True

    def _cancel_host_load_by_group_rpc(self, seed: Case, attempt: int,
                                       offsets: dict[pathlib.Path, int]) -> dict[str, Any]:
        """Cancel an admitted request while Host cache is loading.

        HTTP stream disconnect is polled after the load window. The existing
        Prefill batch RPC acknowledges Cancel synchronously and fences Fetch.
        No model output from this deliberately cancelled request counts as a
        successful answer.
        """
        runfiles = self.args.prefill_rpc_runfiles
        sys.path[:0] = [str(runfiles / "rtp_llm"),
                        *(str(path) for path in sorted(runfiles.glob("pip*/site-packages")))]
        try:
            import grpc
            import torch
            from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, ThinkingMode
            from rtp_llm.cpp.model_rpc.model_rpc_client import trans_input
            from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
                CancelRequestPB, EnqueueGroupRequestPB, FetchRequestPB,
            )
            from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
            from rtp_llm.ops import RoleType
            from rtp_llm.utils.base_model_datatypes import GenerateInput, RequestInfo
        except ImportError as exc:
            raise SmokeFailure(f"PD Cancel RPC dependencies missing: {exc}") from exc

        request_id = time.time_ns() % (2 ** 62)
        role = self.decode_role_addrs[0]
        config = GenerateConfig(
            max_new_tokens=64, do_sample=False, top_k=1, top_p=0.95,
            timeout_ms=300000, aux_info=True, random_seed=0,
            thinking_mode=ThinkingMode.DISABLED, max_thinking_tokens=0,
            role_addrs=[RoleAddr(role=RoleType.DECODE, ip=role["ip"],
                                 http_port=role["http_port"],
                                 grpc_port=role["grpc_port"])],
        )
        request = GenerateInput(
            request_id=request_id,
            token_ids=torch.tensor(self.tokenize(seed.prompt), dtype=torch.int32),
            mm_inputs=[], generate_config=config,
            request_info=RequestInfo(request_id=f"cancel-during-host-load-{attempt}"),
        )
        group = EnqueueGroupRequestPB(batch_id=request_id, dp_rank=0,
                                      fetch_attach_timeout_ms=300000)
        group.requests.add().input.CopyFrom(trans_input(request))
        self._new_log_events(self.args.prefill_event_dir, offsets)
        with grpc.insecure_channel(f"127.0.0.1:{self.args.prefill_grpc_port}") as channel:
            stub = RpcServiceStub(channel)
            ack = stub.EnqueueGroup(group, timeout=300)
            if ([item.request_id for item in ack.successes] != [request_id]
                    or ack.errors):
                raise SmokeFailure(f"Host-load cancel enqueue rejected: {ack}")
            cancel = stub.Cancel(CancelRequestPB(request_id=request_id), timeout=5)
            try:
                frames = list(stub.FetchResponse(
                    FetchRequestPB(request_id=request_id), timeout=5))
                fetch = {"finished": any(
                    frame.flatten_output.finished and frame.flatten_output.finished[0]
                    for frame in frames)}
            except grpc.RpcError as exc:
                fetch = {"code": str(exc.code()), "details": exc.details()}

        deadline = time.monotonic() + 5
        observed = []
        while time.monotonic() < deadline:
            observed.extend(event for event in self._new_log_events(
                self.args.prefill_event_dir, offsets)
                if event.get("request_id") == request_id)
            if any(event.get("event") == "prefill_priority_cancel_accepted"
                   for event in observed):
                break
            time.sleep(0.01)
        events = {event.get("event"): event for event in observed}
        started = events.get("host_cache_load_started")
        cancelled = events.get("prefill_priority_cancel_accepted")
        done = events.get("host_cache_load_done")
        path_passed = bool(
            int(cancel.status) == 1 and fetch.get("code") == "StatusCode.RESOURCE_EXHAUSTED"
            and "preempted" in fetch.get("details", "")
            and started and cancelled
            and started["time_ns"] <= cancelled["time_ns"]
            and (done is None or cancelled["time_ns"] < done["time_ns"])
        )
        return {"case": f"cancel-during-host-load-{attempt}",
                "request_id": request_id, "cancel_status": int(cancel.status),
                "fetch": fetch, "load_started": started, "cancelled": cancelled,
                "load_done": done, "path_passed": path_passed}

    def run_orthogonal_boundaries(self) -> None:
        phases = self.args.orthogonal_phases
        if "cache" in phases or "cancel" in phases:
            self._orthogonal_cache_seed()
        if "page" in phases:
            self._orthogonal_page_boundaries()
        if "chunk" in phases:
            self._orthogonal_chunk_kv()
        if "cache" in phases:
            self._orthogonal_bounded_cache_pressure()
        if "cache" in phases:
            self._orthogonal_cache()
        if "cancel" in phases:
            self._orthogonal_cancel_recovery()
        if "decode" in phases:
            self._orthogonal_decode_batches()

    def run_single_prefill_64k(self) -> None:
        # Exact rendered-token count, including the serving chat template.
        # Keep two concurrent long requests; capacity failures remain failures.
        for label, count in (("single", 1), ("batch", 2)):
            cases = []
            for index in range(count):
                tag = f"K64-{label}-{index}"
                prompt, ids = self.fit_prompt(
                    f"Archive {self.args.namespace}/{tag}. Background:\n",
                    f'\nRecord value={tag}. Return only JSON {{"value":"{tag}"}}.',
                    65536,
                )
                cases.append(Case(
                    f"prefill_64k_{label}_{index}_cold", prompt, "", "miss",
                    require_mtp=True, expected_json={"value": tag},
                    expected_input_len=len(ids), expected_reuse_len=0,
                    max_tokens=max(self.args.max_tokens, self.args.mtp_chunk_max_tokens),
                ))
            self.run_stage(f"prefill_64k_{label}_cold", cases, concurrent=count > 1)
            reuse_cases = [
                replace(case, name=case.name.replace("_cold", "_reuse"), reuse="hit",
                        expected_reuse_len=65535 // self.reuse_unit_tokens * self.reuse_unit_tokens)
                for case in cases
            ]
            reuse_stage = f"prefill_64k_{label}_reuse"
            if label == "single" and len(self.decode_role_addrs) == 2:
                # Reuse the existing request budget: cold state goes to DP0,
                # the same cached Prefill state then crosses to DP1.
                reuse_stage = "decode_dp_cross_owner_cached_64k"
                reuse_cases[0] = replace(reuse_cases[0], name=reuse_stage,
                                         decode_owner_rank=1)
            self.run_stage(reuse_stage, reuse_cases, concurrent=count > 1)

def main() -> int:
    args = parse_args()
    runner = Runner(args)
    try:
        if args.run_kind == "full-93":
            runner.run_main_text()
            # Seed after the broad correctness suite. On the 93-layer model,
            # its traffic can evict an early witness from Host as well as
            # Device; the PageRR and long-KV stages provide bounded pressure.
            runner._orthogonal_cache_seed()
            for phase in (runner._orthogonal_page_boundaries,
                          runner._orthogonal_chunk_kv,
                          runner._orthogonal_bounded_cache_pressure,
                          runner._orthogonal_cache,
                          runner._orthogonal_cancel_recovery,
                          runner._orthogonal_decode_batches):
                phase()
        else:
            runner.run_flow()
            runner.run_orthogonal_boundaries()
        runner.save(passed=True)
        print(
            f"PASS: run={args.run_kind} cases={len(runner.records)} artifacts={args.output}"
        )
        return 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        runner.save(passed=False, error=error)
        print(f"FAIL: {error}; partial artifacts={args.output}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
