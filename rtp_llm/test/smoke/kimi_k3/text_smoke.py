#!/usr/bin/env python3
"""K3.1 93-layer PDFUSION text smoke.

The fixed 93-layer text profile checks observable request and cache contracts.
Kernel implementation paths are outside this black-box smoke.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pathlib
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable

MAX_PREFILL_CONCURRENCY = 16
TOKEN_FIXTURE_CACHE_ENTRIES = 256
REQUEST_PROGRESS_PREFIX = "KIMI_K31_SMOKE "


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    expected_regex: str
    reuse: str
    require_mtp: bool = False
    require_mtp_draft: bool = False
    max_tokens: int | None = None
    timeout_s: int | None = None
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
    """A connection failure or transient HTTP response."""


class SmokeDeadline(SmokeFailure):
    """A request reached its wall-clock deadline."""


def cache_tiers(aux: dict[str, Any]) -> dict[str, int]:
    """The local reuse counter includes the host and disk tiers."""
    keys = (
        "reuse_len",
        "local_reuse_len",
        "memory_reuse_len",
        "disk_reuse_len",
    )
    if any(key not in aux for key in keys):
        raise SmokeFailure("response lacks local cache-tier counters")
    total, local, memory, disk = (int(aux[key]) for key in keys)
    device = local - memory - disk
    if min(total, local, memory, disk, device) < 0 or local > total:
        raise SmokeFailure(
            f"inconsistent local cache-tier counters: "
            f"total={total} local={local} memory={memory} disk={disk}"
        )
    return {
        "total": total,
        "local": local,
        "device": device,
        "memory": memory,
        "disk": disk,
    }


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON answer key: {key}")
        result[key] = value
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the K3.1 PDFUSION smoke.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--engine-log", required=True, type=pathlib.Path)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    config = json.loads((args.checkpoint / "config.json").read_text())
    if config.get("text_config", config).get("num_hidden_layers") != 93:
        parser.error("smoke requires a complete 93-layer checkpoint")
    if not args.engine_log.is_file():
        parser.error("smoke requires an existing engine log")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or (args.output.parent / "requests").exists():
        parser.error("use a new artifact directory")
    args.model_type = "kimi_k31"
    args.block_size = 4096
    args.reuse_unit_tokens = 32768
    args.chunk_tokens = 65536
    args.batch_size = 4
    args.max_tokens = 256
    args.mtp_chunk_max_tokens = 128
    args.timeout = 900
    args.case_deadline_s = 300
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
        # The service is on the local worker; never route smoke traffic
        # through inherited HTTP proxy settings.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.records: list[dict[str, Any]] = []
        self.stages: list[dict[str, Any]] = []
        self.started_at = time.time()
        self.started_monotonic_ns = time.monotonic_ns()
        self.failures: list[dict[str, Any]] = []
        self.skipped_cases: list[dict[str, Any]] = []
        self._record_lock = threading.Lock()
        self._artifact_counter = 0
        self._optimize_token_fixtures = getattr(args, "model_type", None) == "kimi_k31"
        self._fixture_lengths: OrderedDict[str, int] = OrderedDict()
        self._token_cache_hits = 0
        self._prompt_timing = {"fit": (0, 0), "tokenize": (0, 0)}

    @contextmanager
    def _measure_prompt_work(self, operation: str):
        start_ns = time.monotonic_ns()
        try:
            yield
        finally:
            elapsed_ns = time.monotonic_ns() - start_ns
            with self._record_lock:
                calls, total_ns = self._prompt_timing[operation]
                self._prompt_timing[operation] = (calls + 1, total_ns + elapsed_ns)

    def prompt_timing(self) -> dict[str, int]:
        with self._record_lock:
            fit_calls, fit_ns = self._prompt_timing["fit"]
            tokenize_calls, tokenize_ns = self._prompt_timing["tokenize"]
            return {
                "fit_calls": fit_calls,
                "fit_ms": fit_ns // 1_000_000,
                "tokenize_calls": tokenize_calls,
                "tokenize_ms": tokenize_ns // 1_000_000,
                "token_cache_hits": self._token_cache_hits,
            }

    def report_progress(
        self,
        event: str,
        elapsed_ms: int,
        passed: bool,
        case_count: int,
        stage: str | None = None,
    ) -> None:
        progress = {
            "event": event,
            "elapsed_ms": elapsed_ms,
            "passed": passed,
            "case_count": case_count,
            **self.prompt_timing(),
        }
        if stage is not None:
            progress["stage"] = stage
        print(
            REQUEST_PROGRESS_PREFIX + json.dumps(progress, separators=(",", ":")),
            flush=True,
        )

    def save(self, passed: bool, error: str | None = None) -> None:
        payload = {
            "run_kind": "full-93",
            "suite": "kimi-k31-pdfusion",
            "single_prefill_input_limit": self.args.chunk_tokens,
            "formal_request_deadline_s": self.args.case_deadline_s,
            "validation_scope": "request-contracts",
            "protocol_requests": getattr(self, "protocol_results", []),
            "namespace": self.args.namespace,
            "passed": passed,
            "error": error,
            "block_size": self.args.block_size,
            "chunk_tokens": self.args.chunk_tokens,
            "reuse_unit_tokens": self.reuse_unit_tokens,
            "batch_size": self.args.batch_size,
            "max_tokens": self.args.max_tokens,
            "mtp_chunk_max_tokens": self.args.mtp_chunk_max_tokens,
            "elapsed_s": round(time.time() - self.started_at, 3),
            "prompt_timing": self.prompt_timing(),
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
        with self.opener.open(self.health_endpoint, timeout=10) as response:
            if response.status != 200:
                raise SmokeFailure(
                    f"health before {stage} returned HTTP {response.status}"
                )

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
            "passed": False,
            "phase": "preparation" if case.preparation_only else "formal",
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
            audit["deadline_s"] = min(
                self.args.case_deadline_s, case.timeout_s or self.args.timeout
            )
            audit["error"] = f"{type(exc).__name__}: {exc}"
            with self._record_lock:
                self.skipped_cases.append(
                    {
                        "name": case.name,
                        "reason": "request deadline",
                        "sent": True,
                        "artifact": str(artifact) if artifact else None,
                    }
                )
            raise SmokeDeadline(
                f"case={case.name} " f"artifact={artifact}: {exc}"
            ) from exc
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
            failure_type = (
                TransportFailure if isinstance(exc, TransportFailure) else SmokeFailure
            )
            raise failure_type(
                f"case={case.name} " f"artifact={artifact}: {audit['error']}"
            ) from exc
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
        if not case.allow_long_history:
            enable_thinking = (
                False if case.preparation_only or case.thinking_disabled else None
            )
            input_len = self.verified_input_length(
                case.prompt, enable_thinking=enable_thinking
            )
            audit["verified_input_len"] = input_len
            persist()
            if input_len > self.args.chunk_tokens:
                raise SmokeFailure(f"{case.name}: outside the 64K prefill profile")
        request_max_tokens = case.max_tokens or self.args.max_tokens
        payload = {
            "model": "kimi_k31",
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
        audit["request"] = payload
        persist()
        wire_payload = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode()
        if barrier is not None:
            # Token fixture construction can serialize on /tokenize. Release
            # the HTTP requests together only after every caller is ready.
            barrier.wait(timeout=max(30, self.args.timeout))
        started = time.time()
        request_timeout = case.timeout_s or self.args.timeout
        if self.args.case_deadline_s is not None:
            marker = b"\n__K3_HTTP_STATUS__"
            deadline = min(self.args.case_deadline_s, request_timeout)
            command = [
                "curl",
                "--silent",
                "--show-error",
                "--noproxy",
                "*",
                "--max-time",
                str(deadline),
                "--request",
                "POST",
                "--header",
                "Content-Type: application/json",
                "--header",
                f"X-RTP-Smoke-Case: {case.name}",
                "--data-binary",
                "@-",
                "--write-out",
                marker.decode() + "%{http_code}",
                self.endpoint,
            ]
            try:
                response = subprocess.run(
                    command,
                    input=wire_payload,
                    capture_output=True,
                    timeout=deadline + 1,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                audit["response_body"] = (exc.output or b"").decode(
                    "utf-8", errors="replace"
                )
                raise SmokeDeadline(
                    f"{case.name}: request exceeded {deadline}s"
                ) from exc
            if response.returncode == 28:
                audit["response_body"] = response.stdout.decode(
                    "utf-8", errors="replace"
                )
                raise SmokeDeadline(f"{case.name}: request exceeded {deadline}s")
            if response.returncode != 0:
                raise TransportFailure(
                    f"{case.name}: curl exited {response.returncode}: "
                    f"{response.stderr.decode(errors='replace')[:500]}"
                )
            body, separator, status_bytes = response.stdout.rpartition(marker)
            if not separator or not status_bytes.isdigit():
                raise TransportFailure(
                    f"{case.name}: missing HTTP status from capped request"
                )
            status = int(status_bytes)
        else:
            request = urllib.request.Request(
                self.endpoint,
                data=wire_payload,
                headers={
                    "Content-Type": "application/json",
                    "X-RTP-Smoke-Case": case.name,
                },
                method="POST",
            )
            try:
                with self.opener.open(request, timeout=request_timeout) as response:
                    body = response.read()
                    status = response.status
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                audit.update(status=exc.code, response_body=detail)
                failure = (
                    TransportFailure
                    if exc.code in (408, 429, 502, 503, 504)
                    else SmokeFailure
                )
                raise failure(f"{case.name}: HTTP {exc.code}: {detail[:1000]}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                raise TransportFailure(f"{case.name}: request failed: {exc}") from exc
        audit.update(
            status=status, response_body=body.decode("utf-8", errors="replace")
        )
        persist()
        if status != 200:
            failure = (
                TransportFailure
                if status in (408, 429, 502, 503, 504)
                else SmokeFailure
            )
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
            verified_input_len=audit.get("verified_input_len"),
        )

    def validate(
        self,
        case: Case,
        response: dict[str, Any],
        elapsed_s: float,
        request_max_tokens: int,
        *,
        verified_input_len: int | None = None,
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
        diagnostic = case.preparation_only
        if not diagnostic and ("\ufffd" in content or "\ufffd" in reasoning_content):
            raise SmokeFailure(f"{case.name}: Unicode replacement in model response")
        if not diagnostic and finish_reason in ("length", "content_filter"):
            raise SmokeFailure(
                f"{case.name}: incomplete model response finish_reason={finish_reason}"
            )
        # Short cache-preparation requests may end in reasoning_content.
        # Formal answer checks always use the final content channel.
        answer_text = content
        if diagnostic and not answer_text.strip():
            answer_text = reasoning_content
        if not answer_text.strip():
            raise SmokeFailure(f"{case.name}: empty model response")
        if aux.get("pd_sep") is not False:
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
        if verified_input_len is not None and input_len != verified_input_len:
            raise SmokeFailure(
                f"{case.name}: rendered input length {input_len} differs from "
                f"/tokenize length {verified_input_len}"
            )
        effective_reuse = int(aux.get("reuse_len", 0))
        tier_lengths = (
            cache_tiers(aux)
            if all(
                key in aux
                for key in (
                    "reuse_len",
                    "local_reuse_len",
                    "memory_reuse_len",
                    "disk_reuse_len",
                )
            )
            else None
        )
        if case.expected_cache_tier is not None:
            if tier_lengths is None:
                raise SmokeFailure(f"{case.name}: missing local cache-tier counters")
            tier = case.expected_cache_tier
            if tier not in ("miss", "device", "memory", "disk"):
                raise SmokeFailure(f"{case.name}: invalid expected cache tier {tier!r}")
            if tier == "miss":
                tier_ok = tier_lengths["total"] == 0
            else:
                tier_ok = tier_lengths[tier] > 0 and all(
                    tier_lengths[other] == 0
                    for other in ("device", "memory", "disk")
                    if other != tier
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

        # Verify committed Decode progress, excluding the final sampled token.
        decode_kv_last_position = input_len + output_len - 2
        for boundary in case.decode_crossings:
            if not input_len <= boundary <= decode_kv_last_position:
                raise SmokeFailure(
                    f"{case.name}: Decode did not cross committed KV boundary {boundary}; "
                    f"positions={input_len}..{decode_kv_last_position}"
                )

        return {
            "name": case.name,
            "expected_reuse": case.reuse,
            "effective_reuse_len": effective_reuse,
            "cache_tiers": tier_lengths,
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
            "max_tokens": request_max_tokens,
            "pd_sep": False,
            "elapsed_s": round(elapsed_s, 3),
            "content": content,
            "reasoning_content": reasoning_content,
            "finish_reason": finish_reason,
            "output_ids": output_ids,
        }

    def request_cases(
        self,
        cases: list[Case],
        concurrent: bool,
    ) -> list[dict[str, Any]]:
        if concurrent:
            if len(cases) > MAX_PREFILL_CONCURRENCY:
                raise SmokeFailure("Prefill concurrent requests must not exceed 16")
            # Release the HTTP requests together once all callers are ready.
            barrier = threading.Barrier(len(cases))
            with ThreadPoolExecutor(max_workers=len(cases)) as pool:
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

    def run_stage(
        self,
        name: str,
        cases: list[Case],
        concurrent: bool = False,
    ) -> None:
        started_ns = time.monotonic_ns()
        stage = {
            "name": name,
            "start_time_ns": time.time_ns(),
            "concurrent": concurrent,
            "passed": False,
            "case_names": [case.name for case in cases],
        }
        self.stages.append(stage)
        self.report_progress("request_stage_start", 0, False, len(cases), stage=name)
        try:
            self.health(name)
            if name == "rolling_refill":
                self.request_refill(cases)
            else:
                self.request_cases(cases, concurrent)
            stage["passed"] = True
        except Exception as exc:
            stage["error"] = f"{type(exc).__name__}: {exc}"
            raise SmokeFailure(f"stage={name}: {stage['error']}") from exc
        finally:
            elapsed_ns = time.monotonic_ns() - started_ns
            stage["elapsed_s"] = round(elapsed_ns / 1_000_000_000, 3)
            stage["end_time_ns"] = time.time_ns()
            self.report_progress(
                "request_stage",
                elapsed_ns // 1_000_000,
                stage["passed"],
                len(cases),
                stage=name,
            )
            print(
                f"stage={name} passed={str(stage['passed']).lower()} "
                f"error={stage.get('error', '')}",
                flush=True,
            )

    def tokenize(
        self, prompt: str, *, enable_thinking: bool | None = None
    ) -> list[int]:
        payload = {
            "model": "kimi_k31",
            "messages": [{"role": "user", "content": prompt}],
        }
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        request = urllib.request.Request(
            self.args.base_url.rstrip("/") + "/tokenize",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with self._measure_prompt_work("tokenize"):
            with self.opener.open(request, timeout=self.args.timeout) as response:
                tokens = json.load(response)["token_ids"]
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(t) is not int for t in tokens)
        ):
            raise SmokeFailure("tokenizer returned invalid token IDs")
        return tokens

    def _fixture_path(self, prompt: str, enable_thinking: bool | None) -> pathlib.Path:
        digest = hashlib.sha256(f"{enable_thinking!r}\0{prompt}".encode()).hexdigest()
        return self.args.output.parent / "token-fixtures" / f"{digest}.json.gz"

    def verified_input_length(
        self, prompt: str, *, enable_thinking: bool | None
    ) -> int:
        if self._optimize_token_fixtures and self.args.output is not None:
            path = self._fixture_path(prompt, enable_thinking)
            with self._record_lock:
                length = self._fixture_lengths.get(path.name)
                if length is not None and path.is_file():
                    self._fixture_lengths.move_to_end(path.name)
                    self._token_cache_hits += 1
                    return length
        tokens = self.tokenize(prompt, enable_thinking=enable_thinking)
        self.save_token_fixture(prompt, tokens, enable_thinking=enable_thinking)
        return len(tokens)

    def save_token_fixture(
        self, prompt: str, tokens: list[int], *, enable_thinking: bool | None = None
    ) -> None:
        if self.args.output is None:
            return
        path = self._fixture_path(prompt, enable_thinking)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Concurrent requests may save the same fixture. Only publish a complete
        # gzip file, then make its length available to later K3.1 requests.
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            pending = pathlib.Path(output.name)
        try:
            with gzip.open(pending, "wt", encoding="utf-8") as output:
                json.dump(
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "input_ids": tokens,
                        "input_len": len(tokens),
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "enable_thinking": enable_thinking,
                    },
                    output,
                    ensure_ascii=False,
                )
            with self._record_lock:
                pending.replace(path)
                if self._optimize_token_fixtures:
                    self._fixture_lengths[path.name] = len(tokens)
                    self._fixture_lengths.move_to_end(path.name)
                    if len(self._fixture_lengths) > TOKEN_FIXTURE_CACHE_ENTRIES:
                        self._fixture_lengths.popitem(last=False)
        finally:
            pending.unlink(missing_ok=True)

    def fit_prompt(
        self,
        head: str,
        tail: str,
        target: int,
        *,
        enable_thinking: bool | None = None,
    ) -> tuple[str, list[int]]:
        with self._measure_prompt_work("fit"):
            prompt, tokens = self._fit_prompt(head, tail, target, enable_thinking)
            self.save_token_fixture(prompt, tokens, enable_thinking=enable_thinking)
            return prompt, tokens

    def _fit_prompt(
        self, head: str, tail: str, target: int, enable_thinking: bool | None
    ) -> tuple[str, list[int]]:
        # Query the serving tokenizer, including its exact chat template. The
        # filler is inert; all authoritative records and instructions are last.
        lo, hi = 0, target * 2
        try_correction = self._optimize_token_fixtures
        while lo <= hi:
            count = (lo + hi) // 2
            prompt = head + " x" * count + tail
            tokens = self.tokenize(prompt, enable_thinking=enable_thinking)
            if len(tokens) == target:
                return prompt, tokens
            if len(tokens) < target:
                lo = count + 1
            else:
                hi = count - 1
            if try_correction:
                try_correction = False
                # Usually one filler adds one token. Keep the exact search
                # bounds if this single speculative candidate misses.
                corrected_count = count + target - len(tokens)
                if lo <= corrected_count <= hi and corrected_count != (lo + hi) // 2:
                    corrected_prompt = head + " x" * corrected_count + tail
                    corrected_tokens = self.tokenize(
                        corrected_prompt, enable_thinking=enable_thinking
                    )
                    if len(corrected_tokens) == target:
                        return corrected_prompt, corrected_tokens
        # Some tokenizers merge the boundary differently. Search a small,
        # bounded neighbourhood; failure is explicit, never a mislabeled case.
        for count in range(max(0, hi - 8), lo + 9):
            for suffix in ("\n", " ", " a", "\n\n"):
                prompt = head + " x" * count + suffix + tail
                tokens = self.tokenize(prompt, enable_thinking=enable_thinking)
                if len(tokens) == target:
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

    def record_case(self, name: str, *, words: int = 1) -> Case:
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
            expected_json={"value": value},
            # Reasoning can repeat the full record before the final JSON.
            max_tokens=max(self.args.max_tokens, 256 + words * 64),
        )

    def run_request_regressions(self) -> None:
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
                )
                for i in range(4)
            ],
            concurrent=True,
        )
        rolling = []
        for i in range(16):
            case = self.record_case(f"rolling-{i}", words=i + 1)
            prompt, ids = self.fit_prompt(
                f"Rolling {self.args.namespace}/{i}. Background:\n",
                "\n" + case.prompt,
                2048 + 137 * i,
            )
            rolling.append(replace(case, prompt=prompt, expected_input_len=len(ids)))
        self.run_stage("rolling_refill", rolling, concurrent=True)

    def run_batch_shapes(self) -> None:
        # Validate answers across request-batch transitions. These response
        # checks do not establish whether CUDA Graph replay was selected.
        for sequence, batch in enumerate((1, 2, 3, 7, 8, 9, 3, 1, 9)):
            self.run_stage(
                f"batch_shape_{sequence}_{batch}",
                [
                    self.record_case(f"batch-shape-{sequence}-{i}", words=8)
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
        target_tokens = min(target_tokens, self.args.chunk_tokens)
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
                replace(case, name=f"prefix-mixed-{i}")
                for i, case in enumerate((a_hit, b_hit, a_hit, b_hit))
            ],
            concurrent=True,
        )

    def run_cache_block_boundaries(self) -> None:
        page = self.args.block_size
        unit = self.reuse_unit_tokens
        boundaries = cache_block_boundaries(page, unit, self.args.chunk_tokens)
        boundaries = tuple(b for b in boundaries if b + 1 <= self.args.chunk_tokens)
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
                    )
                )
            self.run_stage(f"cache_block_{boundary}_repeat", repeated, concurrent=True)

        # Start one token before each ordinary cache-block boundary.
        decode_boundaries = tuple(
            sorted(
                set(range(page, unit + 1, page)) | {2 * unit, self.args.chunk_tokens}
            )
        )
        # Cross one committed Decode page start with a short exact answer.
        # Long outputs are checked separately in the bounded batch matrix.
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
                    )
                ],
            )

    def run_main_text(self) -> None:
        self.run_request_regressions()
        self.run_batch_shapes()
        cold_prompts = [
            self.fit_cache_prompt(
                f"batch-cold-{idx}",
                40 + idx,
                self.reuse_unit_tokens + self.args.block_size * (idx + 1),
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
                )
                for idx, prompt in enumerate(cold_prompts)
            ],
            concurrent=True,
        )

        exact_hit_count = max(1, self.args.batch_size // 2)
        partial_idx = exact_hit_count
        mixed_prompts = []
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
        self.run_stage(
            "mixed_seed_hits",
            [
                Case(
                    f"mixed_seed_{idx}",
                    mixed_prompts[idx],
                    numbered_answer_pattern((50 + idx) ** 2),
                    "miss",
                )
                for idx in range(exact_hit_count)
            ]
            + [
                Case(
                    "mixed_partial_seed",
                    mixed_partial_seed,
                    numbered_answer_pattern(4489),
                    "miss",
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
                )
                for idx in range(8)
            ],
            concurrent=True,
        )

        self.run_single_prefill_64k()
        self.run_prefix_branches()
        self.run_cache_block_boundaries()

    def _orthogonal_answer(
        self, name: str, prompt: str, value: str, reuse: str, **kwargs
    ) -> Case:
        return Case(
            name,
            prompt,
            r".",
            reuse,
            expected_json={"value": value},
            max_tokens=256,
            require_mtp_draft=True,
            thinking_disabled=True,
            **kwargs,
        )

    def _orthogonal_device_blocks(self) -> int:
        pool_summary = re.compile(
            r"CoordinatorCacheManager pool summary: pools=\[[^\]]*?\bpool_name=full,"
            r"[^;\]]*?\btype=FULL,[^;\]]*?\bblocks=(\d+)"
        )
        legacy_pool = re.compile(r"kvc raw pool\[DEVICE/full\]:.*?\btotal=(\d+)")
        legacy_blocks = None
        with self.args.engine_log.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if match := pool_summary.search(line):
                    # The startup count includes the reserved block zero.
                    return int(match[1]) - 1
                if match := legacy_pool.search(line):
                    legacy_blocks = int(match[1])
        if legacy_blocks is None:
            raise SmokeFailure("Device/full pool capacity is missing from engine log")
        return legacy_blocks

    def _orthogonal_cache_seed(self) -> None:
        unit = self.reuse_unit_tokens
        # Seed at least one reusable 32768-token KDA checkpoint.
        target = max(32769, unit + 1)
        if target > 65536:
            raise SmokeFailure("cache-tier smoke seed exceeds the 64K Q budget")
        self._memory_candidates = []
        for attempt in range(3):
            tag = f"{self.args.namespace}-tier-{attempt}"
            memory_value = f"MEM-{attempt}"
            memory_tail = f'\n只输出 JSON {{"value":"{memory_value}"}}。'
            memory_prompt, _ = self.fit_prompt(
                f"CACHE:{tag}:memory\n", memory_tail, target, enable_thinking=False
            )
            memory_seed = self._orthogonal_answer(
                f"tier-memory-seed-{attempt}", memory_prompt, memory_value, "miss"
            )
            self.run_stage(memory_seed.name, [memory_seed])
            self._memory_candidates.append((tag, memory_seed, memory_value))
            if attempt == 0:
                device_probe = replace(
                    memory_seed,
                    name="tier-device-probe-0",
                    reuse="any",
                    expected_cache_tier=None,
                    preparation_only=True,
                    max_tokens=64,
                )
                self.run_stage(device_probe.name, [device_probe])
                tier = self.records[-1].get("cache_tiers") or {}
                if tier.get("device", 0) <= 0 or tier.get("memory", 0) != 0:
                    raise SmokeFailure("immediate Device cache reuse was not observed")
        # Later smoke traffic, including long-KV preparation, supplies cache
        # pressure. This witness is newer than all three Memory candidates.
        witness_value = "WIT"
        witness_tail = f'\n只输出 JSON {{"value":"{witness_value}"}}。'
        witness_prompt, _ = self.fit_prompt(
            f"CACHE:{self.args.namespace}:witness\n",
            witness_tail,
            target,
            enable_thinking=False,
        )
        self._memory_witness = self._orthogonal_answer(
            "tier-witness-seed", witness_prompt, witness_value, "miss"
        )
        self.run_stage(self._memory_witness.name, [self._memory_witness])

    def _orthogonal_cache(self) -> None:
        if not hasattr(self, "_memory_candidates"):
            raise SmokeFailure("cache seeds must precede the traffic used for demotion")
        probe = replace(
            self._memory_witness,
            name="tier-memory-probe",
            reuse="any",
            expected_cache_tier=None,
            preparation_only=True,
            max_tokens=64,
        )
        self.run_stage(probe.name, [probe])
        tier = self.records[-1].get("cache_tiers") or {}
        if not (tier.get("memory", 0) > 0 and tier.get("device", 0) == 0):
            raise SmokeFailure(
                "ordinary smoke traffic did not demote the witness prefix to Memory"
            )
        device_blocks = self._orthogonal_device_blocks()
        target = max(32769, self.reuse_unit_tokens + 1)
        for attempt, (tag, memory_seed, memory_value) in enumerate(
            self._memory_candidates
        ):
            memory_prompt = memory_seed.prompt
            device_value = f"DEV-{attempt}"
            device_tail = f'\n只输出 JSON {{"value":"{device_value}"}}。'
            device_prompt, _ = self.fit_prompt(
                f"CACHE:{tag}:device\n", device_tail, target, enable_thinking=False
            )
            self.run_stage(
                f"tier-device-seed-{attempt}",
                [
                    self._orthogonal_answer(
                        f"tier-device-seed-{attempt}",
                        device_prompt,
                        device_value,
                        "miss",
                    )
                ],
            )
            partial_seed_value, partial_value = f"SEED-{attempt}", f"PART-{attempt}"
            seed_tail = f'\n只输出 JSON {{"value":"{partial_seed_value}"}}。'
            query_tail = f'\n只输出 JSON {{"value":"{partial_value}"}}。'
            partial_seed_prompt, seed_ids = self.fit_prompt(
                f"CACHE:{tag}:partial\n",
                seed_tail,
                target + 256,
                enable_thinking=False,
            )
            partial_prompt = partial_seed_prompt[: -len(seed_tail)] + query_tail
            query_ids = self.tokenize(partial_prompt, enable_thinking=False)
            common = next(
                (
                    index
                    for index, pair in enumerate(zip(seed_ids, query_ids))
                    if pair[0] != pair[1]
                ),
                min(len(seed_ids), len(query_ids)),
            )
            if common < 32768 or len(query_ids) > 65536:
                raise SmokeFailure(
                    "partial cache prompt does not retain the KDA checkpoint"
                )
            self.run_stage(
                f"tier-partial-seed-{attempt}",
                [
                    self._orthogonal_answer(
                        f"tier-partial-seed-{attempt}",
                        partial_seed_prompt,
                        partial_seed_value,
                        "miss",
                    )
                ],
            )
            miss_value = f"MISS-{attempt}"
            miss_tail = f'\n只输出 JSON {{"value":"{miss_value}"}}。'
            miss_prompt, _ = self.fit_prompt(
                f"CACHE:{tag}:cold\n", miss_tail, target, enable_thinking=False
            )
            mixed = [
                self._orthogonal_answer(
                    f"tier-mixed-{attempt}-device",
                    device_prompt,
                    device_value,
                    "any",
                ),
                self._orthogonal_answer(
                    f"tier-mixed-{attempt}-memory",
                    memory_prompt,
                    memory_value,
                    "any",
                ),
                self._orthogonal_answer(
                    f"tier-mixed-{attempt}-partial",
                    partial_prompt,
                    partial_value,
                    "any",
                ),
                self._orthogonal_answer(
                    f"tier-mixed-{attempt}-miss",
                    miss_prompt,
                    miss_value,
                    "any",
                ),
            ]
            stage_name = f"orthogonal_cache_mixed_{attempt}"
            self.run_stage(stage_name, mixed, concurrent=True)
            records = {
                row["name"]: row
                for row in self.records
                if row["name"] in {case.name for case in mixed}
            }
            kinds = ("device", "memory", "partial", "miss")
            checks = {}
            for kind in kinds:
                row = records[f"tier-mixed-{attempt}-{kind}"]
                tier = row.get("cache_tiers") or {}
                if kind == "miss":
                    checks[kind] = tier.get("total") == 0
                elif kind == "partial":
                    checks[kind] = (
                        0 < row["effective_reuse_len"] < row["input_len"]
                        and tier.get("device", 0) > 0
                    )
                else:
                    checks[kind] = tier.get(kind, 0) > 0 and all(
                        tier.get(other, 0) == 0
                        for other in ("device", "memory", "disk")
                        if other != kind
                    )
            stage = self.stages[-1]
            stage["cache_tier_checks"] = checks
            stage["device_pool_blocks"] = device_blocks
            stage["cache_pressure_requests"] = getattr(
                self, "_cache_pressure_requests", 0
            )
            stage["current_q_total"] = sum(
                row["input_len"] - row["effective_reuse_len"]
                for row in records.values()
            )
            stage["functional_passed"] = (
                all(checks.values()) and stage["current_q_total"] <= 65536
            )
            if stage["functional_passed"]:
                return
        raise SmokeFailure(
            "mixed Prefill cache tiers were not triggered in three attempts"
        )

    def _orthogonal_bounded_cache_pressure(self) -> None:
        # The long-KV chain retains roughly 20 KDA checkpoint blocks. Scale the
        # remaining pressure with the actual Device pool to trigger Host demotion.
        pool_blocks = self._orthogonal_device_blocks()
        pressure_count = max(0, pool_blocks - 20)
        self._cache_pressure_requests = pressure_count
        # Independent prefixes can be in flight together. Drain each batch
        # before issuing more pressure or checking Host reuse.
        pressure_batch_size = 4
        for begin in range(0, pressure_count, pressure_batch_size):
            cases = []
            for index in range(begin, min(begin + pressure_batch_size, pressure_count)):
                prompt, _ = self.fit_prompt(
                    f"CACHE-PRESSURE:{self.args.namespace}:{index}\n",
                    "\nReply 1.",
                    32769,
                    enable_thinking=False,
                )
                cases.append(
                    Case(
                        f"tier-pressure-{index:02d}",
                        prompt,
                        r".",
                        "miss",
                        max_tokens=8,
                        preparation_only=True,
                    )
                )
            self.run_stage(cases[0].name, cases, concurrent=len(cases) > 1)

    def _orthogonal_decode_batches(self) -> None:
        # Revisit the boundary after shrinking the batch to exercise refill.
        sizes = (1, 4, 8, 15, 16, 1, 16, 15, 1)
        for stage_index, size in enumerate(sizes):
            name = f"orthogonal_decode_{stage_index:02d}_batch_{size}"
            cases = []
            for index in range(size):
                case = self.record_case(
                    f"{name}_{index:02d}",
                    words=112 if size == 15 else 56 if size == 16 else 12,
                )
                case = replace(
                    case,
                    max_tokens=2560 if size == 15 else 1280 if size == 16 else 256,
                    require_mtp_draft=True,
                    thinking_disabled=True,
                )
                cases.append(case)
            # Exercise concurrent generation and MTP across request group sizes.
            self.run_stage(name, cases, concurrent=size > 1)

        self._orthogonal_decode_long_output()

    def _orthogonal_decode_long_output(self) -> None:
        # Exercise concurrent long outputs at the Prefill admission limit.
        name = "orthogonal_decode_long_output_batch_16"
        cases = [
            replace(
                self.record_case(f"{name}_{index:02d}", words=112),
                max_tokens=2560,
                require_mtp_draft=True,
                thinking_disabled=True,
            )
            for index in range(MAX_PREFILL_CONCURRENCY)
        ]
        self.run_stage(name, cases, concurrent=True)

    def _orthogonal_page_boundaries(self) -> None:
        for boundary in (4096, 32768):
            for delta in (-1, 0, 1):
                target = boundary + delta
                value = f"PAGE-{boundary}-{delta:+d}"
                tail = f'\n只输出 JSON {{"value":"{value}"}}。'
                prompt, ids = self.fit_prompt(
                    f"PAGE:{self.args.namespace}:{value}\n",
                    tail,
                    target,
                    enable_thinking=False,
                )
                case = self._orthogonal_answer(
                    f"orthogonal_page_{boundary}_{delta:+d}",
                    prompt,
                    value,
                    "miss",
                    expected_input_len=len(ids),
                )
                self.run_stage(case.name, [case])

    def _orthogonal_chunk_kv(self) -> None:
        base = f"Long KV pressure test {self.args.namespace}. "
        overhead = len(self.tokenize(base, enable_thinking=False))
        if len(self.tokenize(base + " x" * 16, enable_thinking=False)) - overhead != 16:
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
                base + " x" * (target - overhead),
                r".",
                "miss" if index == 0 else "hit",
                allow_long_history=True,
                preparation_only=True,
                max_tokens=4,
            )
            self.run_stage(case.name, [case])
            row = self.records[-1]
            if abs(row["input_len"] - target) > 32:
                raise SmokeFailure(
                    f"{case.name}: observed token count differs from seed target"
                )
        value = f"KV-{hashlib.sha256(self.args.namespace.encode()).hexdigest()[:10]}"
        tail = f'\n唯一有效答案是 {value}。只输出 JSON {{"value":"{value}"}}。'
        shared = 589_856 - overhead
        suffix = 65_300
        prompt = base + " x" * shared + " z" * suffix + tail
        case = self._orthogonal_answer(
            "orthogonal_kv_final", prompt, value, "hit", allow_long_history=True
        )
        self.run_stage(case.name, [case])
        row = self.records[-1]
        if row["effective_reuse_len"] < 589_824:
            raise SmokeFailure(
                "589824-token historical KV reuse frontier was not reached"
            )
        self.stages[-1]["historical_kv_tokens"] = row["effective_reuse_len"]
        self.stages[-1]["current_q_tokens"] = (
            row["input_len"] - row["effective_reuse_len"]
        )

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
                cases.append(
                    Case(
                        f"prefill_64k_{label}_{index}_cold",
                        prompt,
                        "",
                        "miss",
                        require_mtp=True,
                        expected_json={"value": tag},
                        expected_input_len=len(ids),
                        expected_reuse_len=0,
                        max_tokens=max(
                            self.args.max_tokens, self.args.mtp_chunk_max_tokens
                        ),
                    )
                )
            self.run_stage(f"prefill_64k_{label}_cold", cases, concurrent=count > 1)
            reuse_cases = [
                replace(
                    case,
                    name=case.name.replace("_cold", "_reuse"),
                    reuse="hit",
                    expected_reuse_len=65535
                    // self.reuse_unit_tokens
                    * self.reuse_unit_tokens,
                )
                for case in cases
            ]
            reuse_stage = f"prefill_64k_{label}_reuse"
            self.run_stage(reuse_stage, reuse_cases, concurrent=count > 1)


def run_smoke(args: argparse.Namespace) -> None:
    from grammar_smoke import run_grammar_cases
    from logprobs_smoke import run_logprobs_cases
    from request_smoke import SmokeRequests, run_request_cases, run_tail_health

    runner = Runner(args)
    passed = False
    try:
        runner.run_main_text()
        runner._orthogonal_cache_seed()
        runner._orthogonal_page_boundaries()
        runner._orthogonal_chunk_kv()
        runner._orthogonal_bounded_cache_pressure()
        runner._orthogonal_cache()
        runner._orthogonal_decode_batches()
        protocol = SmokeRequests(args)
        try:
            run_request_cases(protocol)
            run_grammar_cases(protocol)
            run_logprobs_cases(protocol)
            run_tail_health(protocol)
        finally:
            runner.protocol_results = protocol.records
        runner.save(passed=True)
        passed = True
        print(f"PASS: cases={len(runner.records)} artifacts={args.output}", flush=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        runner.save(passed=False, error=error)
        print(f"FAIL: {error}; partial artifacts={args.output}")
        raise
    finally:
        runner.report_progress(
            "request_summary",
            (time.monotonic_ns() - runner.started_monotonic_ns) // 1_000_000,
            passed,
            len(runner.records),
        )


if __name__ == "__main__":
    run_smoke(parse_args())
