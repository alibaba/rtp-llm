#!/usr/bin/env python3
"""Exercise Kimi K3 PD cache reuse and multi-request scheduling paths."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    expected_regex: str
    reuse: str
    require_chunk: bool = False
    require_mtp: bool = False
    max_tokens: int | None = None


class SmokeFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Kimi K3 full-model PD cache/multi-batch smoke cases."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--decode-health-url", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--suite",
        choices=("flow", "all"),
        default="all",
    )
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--chunk-tokens", type=int, default=65536)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--identity-max-tokens", type=int, default=256)
    parser.add_argument("--single-exact-max-tokens", type=int, default=128)
    parser.add_argument("--mtp-chunk-max-tokens", type=int, default=128)
    parser.add_argument(
        "--rdma-prewarm-attempts",
        type=int,
        default=3,
        help=(
            "run this many bounded concurrent RDMA prewarm attempts before the "
            "formal all-suite cases; zero disables prewarm"
        ),
    )
    parser.add_argument("--rdma-prewarm-backoff-s", type=float, default=5.0)
    parser.add_argument("--rdma-prewarm-settle-s", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.batch_size < 4:
        parser.error("--batch-size must be at least 4 to cover hit/partial-hit/miss mixing")
    for key in (
        "block_size",
        "chunk_tokens",
        "max_tokens",
        "identity_max_tokens",
        "single_exact_max_tokens",
        "mtp_chunk_max_tokens",
        "timeout",
    ):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.rdma_prewarm_attempts < 0:
        parser.error("--rdma-prewarm-attempts must be non-negative")
    for key in ("rdma_prewarm_backoff_s", "rdma_prewarm_settle_s"):
        if getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be non-negative")
    return args


def numbered_answer_pattern(value: int) -> str:
    return rf"(?<!\d){value}(?!\d)"


def make_cache_prompt(namespace: str, case_name: str, value: int, repeats: int = 900) -> str:
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
    return marker + filler * repeats + (
        f"\n分支标识：{suffix_name}。只回答数字：{value} 的平方是多少？"
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


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
        self.health_endpoint = args.base_url.rstrip("/") + "/health"
        self.decode_health_endpoint = args.decode_health_url
        # The service is local to the Prefill host; never route smoke traffic
        # through inherited HTTP proxy settings.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.records: list[dict[str, Any]] = []
        self.stages: list[dict[str, Any]] = []
        self.rdma_prewarm_attempts: list[dict[str, Any]] = []
        self.started_at = time.time()

    def save(self, passed: bool, error: str | None = None) -> None:
        payload = {
            "suite": self.args.suite,
            "namespace": self.args.namespace,
            "passed": passed,
            "error": error,
            "block_size": self.args.block_size,
            "chunk_tokens": self.args.chunk_tokens,
            "batch_size": self.args.batch_size,
            "max_tokens": self.args.max_tokens,
            "identity_max_tokens": self.args.identity_max_tokens,
            "single_exact_max_tokens": self.args.single_exact_max_tokens,
            "mtp_chunk_max_tokens": self.args.mtp_chunk_max_tokens,
            "rdma_prewarm": {
                "enabled": self.args.rdma_prewarm_attempts > 0,
                "target_logical_connections_per_rank": self.args.batch_size,
                "attempts": self.rdma_prewarm_attempts,
            },
            "elapsed_s": round(time.time() - self.started_at, 3),
            "summary": {
                "case_count": len(self.records),
                "hit_count": sum(r.get("effective_reuse_len", 0) > 0 for r in self.records),
                "miss_count": sum(r.get("effective_reuse_len") == 0 for r in self.records),
                "reasoning_count": sum(
                    bool(r.get("reasoning_content", "").strip()) for r in self.records
                ),
                "mtp_case_count": sum(
                    bool(r.get("require_mtp")) for r in self.records
                ),
                "concurrent_stages": sum(s.get("concurrent", False) for s in self.stages),
            },
            "stages": self.stages,
            "cases": self.records,
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

    def request(self, case: Case, barrier: threading.Barrier | None = None) -> dict[str, Any]:
        if barrier is not None:
            barrier.wait(timeout=30)
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
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.time()
        try:
            with self.opener.open(request, timeout=self.args.timeout) as response:
                body = response.read()
                status = response.status
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SmokeFailure(f"{case.name}: HTTP {exc.code}: {detail[:1000]}") from exc
        except Exception as exc:
            raise SmokeFailure(f"{case.name}: request failed: {exc}") from exc
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
            raise SmokeFailure(f"{case.name}: malformed response: {response!r}") from exc
        output_ids = (response.get("debug_info") or {}).get("output_ids")
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
        if re.search(case.expected_regex, answer_text, flags=re.IGNORECASE) is None:
            raise SmokeFailure(
                f"{case.name}: answer failed {case.expected_regex!r}: {answer_text!r}"
            )

        input_len = int(aux.get("input_len", 0))
        raw_reuse = int(aux.get("reuse_len", 0))
        prefill_reuse_value = aux.get("prefill_total_reuse_len")
        effective_reuse = (
            int(prefill_reuse_value) if prefill_reuse_value is not None else raw_reuse
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
            raise SmokeFailure(f"{case.name}: expected miss, got reuse={effective_reuse}")
        if case.reuse == "hit" and effective_reuse <= 0:
            raise SmokeFailure(f"{case.name}: expected hit, got reuse={effective_reuse}")
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
        iter_count = int(aux.get("iter_count", 0))
        mtp_accepted_tokens = output_len - iter_count
        if case.require_mtp and (iter_count <= 0 or mtp_accepted_tokens <= 0):
            raise SmokeFailure(
                f"{case.name}: MTP produced no accepted draft token: "
                f"output_len={output_len}, iter_count={iter_count}"
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
            "max_tokens": request_max_tokens,
            "pd_sep": True,
            "elapsed_s": round(elapsed_s, 3),
            "content": content,
            "reasoning_content": reasoning_content,
            "output_ids": output_ids,
        }

    def request_cases(self, cases: list[Case], concurrent: bool) -> list[dict[str, Any]]:
        if concurrent:
            barrier = threading.Barrier(len(cases))
            with ThreadPoolExecutor(max_workers=len(cases)) as pool:
                futures = [pool.submit(self.request, case, barrier) for case in cases]
                return [future.result() for future in futures]
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
                f"logical_connections_per_rank={len(records)} elapsed_s={elapsed_s}"
            )
            if self.args.rdma_prewarm_settle_s:
                time.sleep(self.args.rdma_prewarm_settle_s)
            return

    def run_stage(self, name: str, cases: list[Case], concurrent: bool = False) -> None:
        self.health(name)
        started = time.time()
        records = self.request_cases(cases, concurrent)
        self.records.extend(records)
        self.stages.append(
            {
                "name": name,
                "concurrent": concurrent,
                "case_names": [case.name for case in cases],
                "elapsed_s": round(time.time() - started, 3),
            }
        )
        print(
            f"stage={name} concurrent={str(concurrent).lower()} "
            f"reuse={[record['effective_reuse_len'] for record in records]}"
        )

    def run_flow(self) -> None:
        self.run_stage(
            "chunkwise_rdma_flow_miss",
            [
                Case(
                    "chunkwise_rdma_flow_miss",
                    make_flow_prompt(self.args.namespace),
                    r".",
                    "miss",
                    require_chunk=True,
                )
            ],
        )

    def run_all(self) -> None:
        self.prewarm_rdma_pool()
        self.run_stage(
            "identity_miss",
            [
                Case(
                    "identity_miss",
                    f"会话标识 {self.args.namespace}/identity，不要复述该标识。你好，请问你是谁？",
                    r"\bKimi\b|Moonshot|月之暗面",
                    "miss",
                    max_tokens=max(
                        self.args.max_tokens,
                        self.args.identity_max_tokens,
                    ),
                )
            ],
        )
        exact_prompt = make_cache_prompt(self.args.namespace, "single-exact", 37)
        single_exact_max_tokens = max(
            self.args.max_tokens,
            self.args.single_exact_max_tokens,
        )
        self.run_stage(
            "single_exact_seed",
            [
                Case(
                    "single_exact_seed",
                    exact_prompt,
                    numbered_answer_pattern(1369),
                    "miss",
                    max_tokens=single_exact_max_tokens,
                )
            ],
        )
        self.run_stage(
            "single_exact_hit",
            [
                Case(
                    "single_exact_hit",
                    exact_prompt,
                    numbered_answer_pattern(1369),
                    "hit",
                    max_tokens=single_exact_max_tokens,
                )
            ],
        )

        partial_seed = make_partial_prompt(self.args.namespace, "partial-common", "seed", 29)
        partial_query = make_partial_prompt(self.args.namespace, "partial-common", "query", 31)
        self.run_stage(
            "partial_prefix_seed",
            [Case("partial_prefix_seed", partial_seed, numbered_answer_pattern(841), "miss")],
        )
        self.run_stage(
            "partial_prefix_hit",
            [Case("partial_prefix_hit", partial_query, numbered_answer_pattern(961), "partial")],
        )

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
                    f"batch_all_miss_{idx}", prompt,
                    numbered_answer_pattern((40 + idx) ** 2), "miss"
                )
                for idx, prompt in enumerate(cold_prompts)
            ],
            concurrent=True,
        )
        self.run_stage(
            "batch_all_hit",
            [
                Case(
                    f"batch_all_hit_{idx}", prompt,
                    numbered_answer_pattern((40 + idx) ** 2), "hit"
                )
                for idx, prompt in enumerate(cold_prompts)
            ],
            concurrent=True,
        )

        exact_hit_count = max(1, self.args.batch_size // 2)
        partial_idx = exact_hit_count
        mixed_prompts = []
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
                    f"mixed_seed_{idx}", mixed_prompts[idx],
                    numbered_answer_pattern((50 + idx) ** 2), "miss"
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
                    f"batch_mixed_{idx}", prompt,
                    numbered_answer_pattern((50 + idx) ** 2),
                    (
                        "hit"
                        if idx < exact_hit_count
                        else "partial"
                        if idx == partial_idx
                        else "miss"
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
                    f"batch_mixed_all_hit_{idx}", prompt,
                    numbered_answer_pattern((50 + idx) ** 2), "hit"
                )
                for idx, prompt in enumerate(mixed_prompts)
            ],
            concurrent=True,
        )

        mtp_chunk_prompt = make_whole_chunk_prompt(
            self.args.namespace,
            "mtp-chunk-prefill",
            73,
        )
        self.run_stage(
            "mtp_chunk_prefill_miss",
            [
                Case(
                    "mtp_chunk_prefill_miss",
                    mtp_chunk_prompt,
                    numbered_answer_pattern(5329),
                    "miss",
                    require_chunk=True,
                    require_mtp=True,
                    max_tokens=max(
                        self.args.max_tokens,
                        self.args.mtp_chunk_max_tokens,
                    ),
                )
            ],
        )

        single_prompt = make_whole_chunk_prompt(
            self.args.namespace, "whole-chunk-single", 61
        )
        self.run_stage(
            "whole_chunk_single_miss",
            [Case(
                "whole_chunk_single_miss", single_prompt, numbered_answer_pattern(3721),
                "miss", require_chunk=True
            )],
        )
        self.run_stage(
            "whole_chunk_single_hit",
            [Case(
                "whole_chunk_single_hit", single_prompt, numbered_answer_pattern(3721),
                "hit", require_chunk=True
            )],
        )

        chunk_batch_size = min(2, self.args.batch_size)
        chunk_prompts = [
            make_whole_chunk_prompt(self.args.namespace, f"whole-chunk-batch-{idx}", 70 + idx)
            for idx in range(chunk_batch_size)
        ]
        self.run_stage(
            "whole_chunk_batch_miss",
            [
                Case(
                    f"whole_chunk_batch_miss_{idx}", prompt,
                    numbered_answer_pattern((70 + idx) ** 2), "miss", require_chunk=True
                )
                for idx, prompt in enumerate(chunk_prompts)
            ],
            concurrent=True,
        )
        self.run_stage(
            "whole_chunk_batch_hit",
            [
                Case(
                    f"whole_chunk_batch_hit_{idx}", prompt,
                    numbered_answer_pattern((70 + idx) ** 2), "hit", require_chunk=True
                )
                for idx, prompt in enumerate(chunk_prompts)
            ],
            concurrent=True,
        )

def main() -> int:
    args = parse_args()
    runner = Runner(args)
    try:
        suites: dict[str, Callable[[], None]] = {
            "flow": runner.run_flow,
            "all": runner.run_all,
        }
        suites[args.suite]()
        runner.save(passed=True)
        print(f"PASS: suite={args.suite} cases={len(runner.records)} artifacts={args.output}")
        return 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        runner.save(passed=False, error=error)
        print(f"FAIL: {error}; partial artifacts={args.output}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
