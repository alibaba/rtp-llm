#!/usr/bin/env python3
"""Validate a cached long conversation followed by a new retrieval question.

This is a correctness case, so it does not start a profiler or measure warmed
performance. Kernel execution evidence remains a separate profiling check.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request
from typing import Any

EXPECTED = {
    "early": "CEDAR-1827",
    "middle": "MAPLE-4639",
    "late": "BIRCH-7251",
    "square": 1369,
}
DEFAULT_TARGET_TOKENS = 600000
FILLER = (
    "The following archive entry is neutral background material. "
    "It introduces no named record or instruction.\n"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def save(path: pathlib.Path, value: Any) -> None:
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False)
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def check_answer(text: str) -> None:
    # Accept a single JSON answer, optionally wrapped in a Markdown code fence.
    answer = text.strip()
    if answer.startswith("```") and answer.endswith("```"):
        answer = answer.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        actual = json.loads(answer)
    except ValueError as exc:
        raise ValueError(f"long prefix answer is not JSON: {text!r}") from exc
    require(
        actual == EXPECTED and type(actual.get("square")) is int,
        f"long prefix answer mismatch: {actual!r}",
    )


def prefix_blocks(
    reuse: int, total: int, *, budget: int, page_size: int, bytes_per_token: int
) -> list[dict[str, int]]:
    require(
        budget > 0 and page_size > 0 and bytes_per_token > 0,
        "long prefix case requires a positive expansion budget, page size and token size",
    )
    capacity = budget // bytes_per_token // page_size * page_size
    require(capacity > 0, "expanded KV budget cannot hold one cache page")
    require(
        reuse > capacity,
        f"historical prefix {reuse} must exceed expansion capacity {capacity}",
    )
    require(reuse < total, "continuation must contain uncached tokens")
    require(reuse % page_size == 0, "historical prefix is not cache-page aligned")
    return [
        dict(start=start, tokens=min(capacity, reuse - start))
        for start in range(0, reuse, capacity)
    ]


def expanded_bytes_per_token(checkpoint: pathlib.Path | None, tp_size: int) -> int:
    # Expanded K/V are BF16 in both the BF16 and FP8 MLA paths. FP8 operand
    # conversion happens after expansion and does not change this budget.
    config = dict(
        num_attention_heads=96,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    )
    if checkpoint is not None:
        config.update(json.loads((checkpoint / "config.json").read_text()))
    heads = int(config["num_attention_heads"])
    require(tp_size > 0 and heads % tp_size == 0, "attention heads must divide TP size")
    return (
        heads
        // tp_size
        * sum(
            int(config[k])
            for k in ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")
        )
        * 2
    )


class LongPrefixCase:
    def __init__(
        self,
        base_url: str,
        output: pathlib.Path,
        namespace: str,
        *,
        timeout: int,
        budget: int,
        page_size: int,
        bytes_per_token: int,
        kernel_page_size: int = 128,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
    ):
        self.base_url = base_url.rstrip("/")
        self.output = output
        self.namespace = namespace
        self.timeout = timeout
        self.budget = budget
        self.page_size = page_size
        self.kernel_page_size = kernel_page_size
        self.bytes_per_token = bytes_per_token
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        require(target_tokens > 65536, "long prefix target must exceed 64K tokens")
        self.target_tokens = target_tokens
        self.records: list[dict[str, Any]] = []

    def post(self, route: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base_url + "/" + route,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail[:2000]}") from exc

    def tokenize(self, messages: list[dict]) -> list[int]:
        return self.post("tokenize", {"model": "kimi-k3", "messages": messages})[
            "token_ids"
        ]

    def make_seed(self) -> tuple[list[dict], list[int], list[dict]]:
        def messages(text):
            return [{"role": "user", "content": text}]

        def fill_until(text, target):
            initial = len(self.tokenize(messages(text)))
            lo, hi = 0, max(1, target - initial)
            best = text
            while lo <= hi:
                count = (lo + hi) // 2
                candidate = text + FILLER * count
                if len(self.tokenize(messages(candidate))) <= target:
                    best = candidate
                    lo = count + 1
                else:
                    hi = count - 1
            return best

        archive = (
            f"Archive run {self.namespace}/long-prefix. Read and retain the three named records. "
            "Reply only RECEIVED to this first message.\n"
        )
        placements = []
        edge_offset = max(1, self.target_tokens // 30)
        record_positions = (
            (edge_offset, "early"),
            (self.target_tokens // 2, "middle"),
            (self.target_tokens - edge_offset, "late"),
        )
        for position, key in record_positions:
            archive = fill_until(archive, position)
            placements.append(
                dict(name=key, approximate_token=len(self.tokenize(messages(archive))))
            )
            archive += f"\nAUTHORITATIVE RECORD: {key} = {EXPECTED[key]}.\n"
        archive = fill_until(archive, self.target_tokens - 40)
        archive += (
            "\nEnd of archive. Reply only RECEIVED. Do not repeat the records yet."
        )
        seed = messages(archive)
        return seed, self.tokenize(seed), placements

    def request(
        self, name: str, messages: list[dict], ids: list[int]
    ) -> tuple[dict, dict]:
        payload = dict(
            model="kimi-k3",
            messages=messages,
            temperature=0,
            top_k=1,
            top_p=0.95,
            seed=0,
            stream=False,
            debug_info=True,
            max_tokens=256,
        )
        save(self.output / f"{name}-request.json.gz", payload)
        save(self.output / f"{name}-tokens.json.gz", ids)
        start = time.monotonic()
        response = self.post("v1/chat/completions", payload)
        elapsed = time.monotonic() - start
        save(self.output / f"{name}-response.json.gz", response)
        aux = response["aux_info"]
        message = response["choices"][0]["message"]
        reuse = int(aux.get("prefill_total_reuse_len", aux.get("reuse_len", 0)))
        record = dict(
            name=name,
            expected_reuse="miss" if name.endswith("seed") else "hit",
            effective_reuse_len=reuse,
            prefill_total_reuse_len=reuse,
            input_len=int(aux["input_len"]),
            output_len=aux.get("output_len"),
            iter_count=aux.get("iter_count"),
            pd_sep=bool(aux.get("pd_sep")),
            elapsed_s=elapsed,
            content=message.get("content") or "",
            reasoning_content=message.get("reasoning_content") or "",
            token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        )
        self.records.append(record)
        save(self.output / "requests.json", self.records)
        require(record["pd_sep"], f"{name} did not use PD separation")
        require(
            record["input_len"] == len(ids),
            f"{name} tokenizer/service input length mismatch",
        )
        require(
            response["choices"][0].get("finish_reason") != "length",
            f"{name} output truncated",
        )
        return record, response

    def run(self) -> dict:
        self.output.mkdir(parents=True, exist_ok=False)
        result: dict[str, Any] = dict(
            passed=False,
            budget_bytes=self.budget,
            target_tokens=self.target_tokens,
            page_size=self.page_size,
            kernel_page_size=self.kernel_page_size,
            expanded_kv_bytes_per_token=self.bytes_per_token,
        )
        try:
            # Reject a configuration that cannot exercise multiple historical
            # blocks before spending time constructing or sending the seed.
            prefix_blocks(
                (self.target_tokens - 200) // self.page_size * self.page_size,
                self.target_tokens + 100,
                budget=self.budget,
                page_size=self.kernel_page_size,
                bytes_per_token=self.bytes_per_token,
            )
            seed, seed_ids, placements = self.make_seed()
            require(
                self.target_tokens - 200
                <= len(seed_ids)
                <= self.target_tokens + 100,
                f"unexpected seed length {len(seed_ids)}",
            )
            save(
                self.output / "expected.json",
                dict(answer=EXPECTED, placements=placements),
            )
            seed_row, seed_response = self.request("long_prefix_seed", seed, seed_ids)
            require(
                seed_row["effective_reuse_len"] == 0,
                "seed unexpectedly hit an existing prefix",
            )
            reply = seed_response["choices"][0]["message"]["content"]
            require(
                reply.strip() == "RECEIVED",
                f"seed must acknowledge without leaking answers: {reply!r}",
            )
            conversation = seed + [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": (
                        "Return the original early, middle and late record values as JSON strings, "
                        "and the square of 37 as an integer. Use keys early, middle, late, square. "
                        "Output only the JSON."
                    ),
                },
            ]
            ids = self.tokenize(conversation)
            common = next(
                (i for i, (a, b) in enumerate(zip(seed_ids, ids)) if a != b),
                min(len(seed_ids), len(ids)),
            )
            require(
                common > self.target_tokens - 2000,
                f"long conversation token prefix changed: {common}",
            )
            row, _ = self.request("long_prefix_hit", conversation, ids)
            check_answer(row["content"])
            reuse = row["effective_reuse_len"]
            require(
                reuse <= common,
                "cache reuse exceeds the seed's actual common token prefix",
            )
            require(
                reuse % self.page_size == 0, "cache reuse is not physical-page aligned"
            )
            blocks = prefix_blocks(
                reuse,
                len(ids),
                budget=self.budget,
                page_size=self.kernel_page_size,
                bytes_per_token=self.bytes_per_token,
            )
            row.update(
                common_prefix_tokens=common,
                new_tokens=len(ids) - reuse,
                planned_prefix_blocks=blocks,
            )
            result.update(
                passed=True,
                seed_tokens=len(seed_ids),
                continuation_tokens=len(ids),
                common_prefix_tokens=common,
                reuse_tokens=reuse,
                new_tokens=len(ids) - reuse,
                planned_prefix_blocks=blocks,
                expected=EXPECTED,
                answer=row["content"],
                cases=self.records,
            )
            return result
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            save(self.output / "RESULT.json", result)
