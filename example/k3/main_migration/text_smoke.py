#!/usr/bin/env python3
"""RTP main text-only smoke, adapted from K3 dev 64c6aff3666402228950f1f09031e228c3734277.

This is not the original dev all suite. Runtime evidence is a separate gate.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pathlib
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import Any, Callable

try:
    from .long_prefix_case import (
        DEFAULT_TARGET_TOKENS,
        LongPrefixCase,
        expanded_bytes_per_token,
    )
except ImportError:  # Direct script entry from the role launcher.
    from long_prefix_case import (
        DEFAULT_TARGET_TOKENS,
        LongPrefixCase,
        expanded_bytes_per_token,
    )


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str | list[dict[str, Any]]
    expected_regex: str
    reuse: str
    require_chunk: bool = False
    require_mtp: bool = False
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


class SmokeFailure(RuntimeError):
    pass


class TransportFailure(SmokeFailure):
    """Only connection failures and transient HTTP responses may be retried."""


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
    parser.add_argument(
        "--decode-dp-size",
        type=int,
        default=1,
        choices=(1,),
        help="number of Decode DP owners; TP-only uses 1",
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--suite",
        choices=("flow", "main-text", "main-text-64k"),
        default="main-text",
    )
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument(
        "--reuse-unit-tokens",
        type=int,
        default=0,
        help="ordinary main checkpoint granularity; 0 means one configured cache block (never TP times block).",
    )
    parser.add_argument("--chunk-tokens", type=int, default=65536)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--identity-max-tokens", type=int, default=256)
    parser.add_argument("--single-exact-max-tokens", type=int, default=128)
    parser.add_argument("--mtp-chunk-max-tokens", type=int, default=128)
    parser.add_argument(
        "--require-mtp",
        action="store_true",
        help="include the native MTP draft-acceptance case",
    )
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
    parser.add_argument("--long-prefix-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--long-prefix-tp-size", type=int, default=8)
    parser.add_argument(
        "--long-prefix-target-tokens", type=int, default=DEFAULT_TARGET_TOKENS
    )
    parser.add_argument("--long-prefix-kernel-page-size", type=int, default=128)
    parser.add_argument("--expanded-kv-budget-gib", type=float, default=6.0)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
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
        "long_prefix_tp_size",
        "long_prefix_target_tokens",
        "long_prefix_kernel_page_size",
        "rdma_prewarm_timeout",
    ):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.suite == "main-text" and (
        not math.isfinite(args.expanded_kv_budget_gib)
        or args.expanded_kv_budget_gib <= 0
    ):
        parser.error(
            "all suite needs a positive expansion budget for the long prefix case"
        )
    if args.rdma_prewarm_attempts < 0:
        parser.error("--rdma-prewarm-attempts must be non-negative")
    if args.decode_dp_size is not None and args.decode_dp_size <= 0:
        parser.error("--decode-dp-size must be positive")
    expected_owners = (
        (args.decode_dp_size,) if args.decode_dp_size is not None else (8, 16)
    )
    if args.suite in ("main-text", "main-text-64k") and len(args.decode_role_addrs) not in expected_owners:
        parser.error(
            f"--suite=all requires {expected_owners} ordered --decode-role-addr values"
        )
    for key in ("rdma_prewarm_backoff_s", "rdma_prewarm_settle_s"):
        if getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be non-negative")
    if args.suite in ("main-text", "main-text-64k"):
        config = json.loads((args.long_prefix_checkpoint / "config.json").read_text())
        config = config.get("text_config", config)
        if config.get("num_hidden_layers") != 93:
            parser.error(
                "main-text requires the full 93-layer checkpoint; use flow only for preflight"
            )
        if not args.require_mtp:
            parser.error("main-text requires --require-mtp (real acceptance)")
        if args.suite == "main-text" and args.long_prefix_target_tokens < DEFAULT_TARGET_TOKENS:
            parser.error("main-text cannot reduce the 110K long-prefix gate")
        if args.chunk_tokens < 65536:
            parser.error("main-text requires a chunk budget of at least 65536")
        if args.suite == "main-text-64k" and args.chunk_tokens != 65536:
            parser.error("main-text-64k requires a 65536-token single-prefill budget")
    if args.reuse_unit_tokens not in (0, args.block_size):
        parser.error(
            "main ordinary layout requires reuse-unit-tokens equal to block-size"
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
        self._record_lock = threading.Lock()
        self._artifact_counter = 0

    def save(self, passed: bool, error: str | None = None) -> None:
        payload = {
            "suite": self.args.suite,
            "source_reference": "64c6aff3666402228950f1f09031e228c3734277",
            "profile": "tp8-ep8-sp-no-dcp-text",
            "deferred_by_user": (
                ["chunk prefill", "over-64K inputs", "chunk budget +1/+7", "110K seed and append"]
                if self.args.suite == "main-text-64k" else []
            ),
            "single_prefill_input_limit": 65536 if self.args.suite == "main-text-64k" else None,
            "not_applicable": [
                "DCP",
                "PageRR owner",
                "DP multi-owner",
                "multimodal",
                "KTP",
                "EAGLE3/DSpark",
            ],
            "runtime_evidence_gate": "separate; this result alone does not certify RDMA, graph replay or precision",
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
            "phase": "prewarm" if case.name.startswith("rdma_prewarm_") else "formal",
            "expected": {
                "regex": case.expected_regex,
                "json": case.expected_json,
                "input_len": case.expected_input_len,
                "reuse_len": case.expected_reuse_len,
            },
        }

        def persist():
            if artifact is not None:
                artifact.write_text(
                    json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
                )

        try:
            result = self._request(case, barrier, audit=audit, persist=persist)
            result["phase"] = (
                "prewarm" if case.name.startswith("rdma_prewarm_") else "formal"
            )
            audit.update(passed=True, result=result)
            with self._record_lock:
                self.records.append(result)
            return result
        except Exception as exc:
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
        if barrier is not None:
            barrier.wait(timeout=30)
        if self.args.suite == "main-text-64k":
            if not isinstance(case.prompt, str):
                raise SmokeFailure("64K profile requires a text prompt")
            input_ids = self.tokenize(case.prompt)
            self.save_token_fixture(case.prompt, input_ids)
            audit["verified_input_len"] = len(input_ids)
            persist()
            if len(input_ids) > 65536 or case.require_chunk:
                raise SmokeFailure(f"{case.name}: outside the non-chunk 64K profile")
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
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.time()
        request_timeout = case.timeout_s or self.args.timeout
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
            raise SmokeFailure(f"{case.name}: HTTP {status}")
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
            message = response["choices"][0]["message"]
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
        # A four-layer checkpoint is only a transport preflight. With the
        # model's default reasoning mode enabled, all short output may remain
        # in reasoning_content. Keep full-model semantic checks pinned to the
        # final answer, but accept either non-empty channel for flow coverage.
        answer_text = content
        if self.args.suite == "flow" and not answer_text.strip():
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
            "input_len": input_len,
            "output_len": output_len,
            "iter_count": iter_count,
            "mtp_accepted_tokens": mtp_accepted_tokens,
            "require_mtp": case.require_mtp,
            "expected_json": case.expected_json,
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
            "output_ids": output_ids,
            "decode_owner_rank": case.decode_owner_rank,
            "selected_decode_role_addr": selected_decode_role_addr,
            "observed_decode_role_addrs": observed_decode_role_addrs,
        }

    def request_cases(
        self, cases: list[Case], concurrent: bool
    ) -> list[dict[str, Any]]:
        if concurrent:
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

    def run_stage(self, name: str, cases: list[Case], concurrent: bool = False) -> None:
        self.health(name)
        started = time.time()
        stage = {
            "name": name,
            "concurrent": concurrent,
            "passed": False,
            "case_names": [case.name for case in cases],
            "decode_owner_ranks": [case.decode_owner_rank for case in cases],
        }
        self.stages.append(stage)
        try:
            if name == "rolling_refill":
                self.request_refill(cases)
            else:
                self.request_cases(cases, concurrent)
            stage["passed"] = True
        except Exception as exc:
            stage["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            stage["elapsed_s"] = round(time.time() - started, 3)
        print(f"stage={name} passed=true owners={stage['decode_owner_ranks']}")

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
        prompt_a, tokens_a = self.fit_prompt(
            head, tail_a, unit * 2 + self.args.block_size
        )
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
        if self.args.suite == "main-text-64k":
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
        # Four layers only check flow, chunk boundaries and cache reuse.
        prompt, tokens = self.fit_prompt(
            f"流程 {self.args.namespace}/chunk。\n",
            "\n请回复任意一个非空字符。",
            self.args.chunk_tokens + 1,
        )
        seed = Case(
            "chunkwise_rdma_flow_miss",
            prompt,
            r".",
            "miss",
            require_chunk=True,
            expected_input_len=len(tokens),
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

        if self.args.suite == "main-text-64k":
            self.run_single_prefill_64k()
            self.run_prefix_branches()
            self.run_cache_block_boundaries()
            return

        single_prompt = make_whole_chunk_prompt(
            self.args.namespace, "whole-chunk-single", 61
        )
        self.run_stage(
            "whole_chunk_single_miss",
            [
                Case(
                    "whole_chunk_single_miss",
                    single_prompt,
                    numbered_answer_pattern(3721),
                    "miss",
                    require_chunk=True,
                    require_mtp=getattr(self.args, "require_mtp", False),
                    max_tokens=max(
                        self.args.max_tokens, self.args.mtp_chunk_max_tokens
                    ),
                )
            ],
        )
        self.run_stage(
            "whole_chunk_single_hit",
            [
                Case(
                    "whole_chunk_single_hit",
                    single_prompt,
                    numbered_answer_pattern(3721),
                    "hit",
                    require_chunk=True,
                    require_mtp=getattr(self.args, "require_mtp", False),
                    max_tokens=max(
                        self.args.max_tokens, self.args.mtp_chunk_max_tokens
                    ),
                )
            ],
        )

        chunk_batch_size = min(2, self.args.batch_size)
        chunk_prompts = [
            make_whole_chunk_prompt(
                self.args.namespace, f"whole-chunk-batch-{idx}", 70 + idx
            )
            for idx in range(chunk_batch_size)
        ]
        self.run_stage(
            "whole_chunk_batch_miss",
            [
                Case(
                    f"whole_chunk_batch_miss_{idx}",
                    prompt,
                    numbered_answer_pattern((70 + idx) ** 2),
                    "miss",
                    require_chunk=True,
                    require_mtp=getattr(self.args, "require_mtp", False),
                    max_tokens=max(
                        self.args.max_tokens, self.args.mtp_chunk_max_tokens
                    ),
                    decode_owner_rank=idx % max(1, len(self.decode_role_addrs)),
                )
                for idx, prompt in enumerate(chunk_prompts)
            ],
            concurrent=True,
        )
        self.run_stage(
            "whole_chunk_batch_hit",
            [
                Case(
                    f"whole_chunk_batch_hit_{idx}",
                    prompt,
                    numbered_answer_pattern((70 + idx) ** 2),
                    "hit",
                    require_chunk=True,
                    require_mtp=getattr(self.args, "require_mtp", False),
                    max_tokens=max(
                        self.args.max_tokens, self.args.mtp_chunk_max_tokens
                    ),
                    decode_owner_rank=idx % max(1, len(self.decode_role_addrs)),
                )
                for idx, prompt in enumerate(chunk_prompts)
            ],
            concurrent=True,
        )
        self.run_prefix_branches()
        self.run_padding_boundaries()
        self.run_cache_block_boundaries()
        self.run_long_prefix_case()

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
            self.run_stage(f"prefill_64k_{label}_reuse", [
                replace(case, name=case.name.replace("_cold", "_reuse"), reuse="hit",
                        expected_reuse_len=65535 // self.reuse_unit_tokens * self.reuse_unit_tokens)
                for case in cases
            ], concurrent=count > 1)

    def run_long_prefix_case(self) -> None:
        self.health("long_prefix_cached_dialog")
        stage = dict(name="long_prefix_cached_dialog", concurrent=False, passed=False)
        self.stages.append(stage)
        case = LongPrefixCase(
            self.args.base_url,
            self.args.output.parent / "long-prefix",
            self.args.namespace,
            timeout=self.args.timeout,
            budget=int(self.args.expanded_kv_budget_gib * 1024**3),
            page_size=self.args.block_size,
            kernel_page_size=self.args.long_prefix_kernel_page_size,
            bytes_per_token=expanded_bytes_per_token(
                self.args.long_prefix_checkpoint, self.args.long_prefix_tp_size
            ),
            target_tokens=self.args.long_prefix_target_tokens,
            reuse_unit_tokens=self.reuse_unit_tokens,
            decode_role_addrs=self.decode_role_addrs,
        )
        try:
            result = case.run()
            stage.update(
                passed=True,
                case_names=[row["name"] for row in result["cases"]],
                planned_prefix_blocks=result["planned_prefix_blocks"],
                evidence=str(case.output / "RESULT.json"),
            )
        finally:
            self.records.extend(case.records)
        self.health("long_prefix_cached_dialog")


def main() -> int:
    args = parse_args()
    runner = Runner(args)
    try:
        suites: dict[str, Callable[[], None]] = {
            "flow": runner.run_flow,
            "main-text": runner.run_main_text,
            "main-text-64k": runner.run_main_text,
        }
        suites[args.suite]()
        runner.save(passed=True)
        print(
            f"PASS: suite={args.suite} cases={len(runner.records)} artifacts={args.output}"
        )
        return 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        runner.save(passed=False, error=error)
        print(f"FAIL: {error}; partial artifacts={args.output}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
