#!/usr/bin/env python3
"""Run a small, repeatable long-prefix cache pressure workload.

The normal K3 ``all`` suite remains unchanged.  This entry point is deliberately
separate: it uses exact 960K-token prefixes, fills several independent prefixes,
then revisits them after the pool has had an opportunity to evict older data.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

try:
    from .kimi_k3_long_prefix_case import (
        EXPECTED,
        LongPrefixCase,
        check_answer,
        expanded_bytes_per_token,
    )
except ImportError:
    from kimi_k3_long_prefix_case import (  # type: ignore
        EXPECTED,
        LongPrefixCase,
        check_answer,
        expanded_bytes_per_token,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--prefix-count", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--target-tokens", type=int, default=960000)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=4096)
    parser.add_argument("--kernel-page-size", type=int, default=128)
    parser.add_argument("--expanded-kv-budget-bytes", type=int, default=4 << 30)
    args = parser.parse_args()
    if args.prefix_count <= 0 or args.rounds <= 0 or args.max_tokens <= 0:
        parser.error("--prefix-count, --rounds and --max-tokens must be positive")
    if args.target_tokens <= 65536:
        parser.error("--target-tokens must exceed the chunk size")
    return args


def build_conversation(case: LongPrefixCase, seed, seed_ids):
    seed_reply = "RECEIVED"
    conversation = seed + [
        {"role": "assistant", "content": seed_reply},
        {
            "role": "user",
            "content": (
                "Return the original early, middle and late record values as JSON strings, "
                "and the square of 37 as an integer. Use keys early, middle, late, square. "
                "Output only the JSON."
            ),
        },
    ]
    return conversation, case.tokenize(conversation)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bytes_per_token = expanded_bytes_per_token(args.checkpoint, args.tp_size)
    cases = []
    for index in range(args.prefix_count):
        case_output = args.output / f"prefix-{index:03d}"
        case_output.mkdir(parents=True, exist_ok=True)
        cases.append(
            LongPrefixCase(
                args.base_url,
                case_output,
                f"{args.namespace}/prefix-{index:03d}",
                timeout=args.timeout,
                budget=args.expanded_kv_budget_bytes,
                page_size=args.page_size,
                bytes_per_token=bytes_per_token,
                kernel_page_size=args.kernel_page_size,
                target_tokens=args.target_tokens,
                max_tokens=args.max_tokens,
                expected={
                    key: f"{value}-{index:03d}" if isinstance(value, str) else value
                    for key, value in EXPECTED.items()
                },
            )
        )

    started = time.time()
    conversations = []
    for index, case in enumerate(cases):
        seed, seed_ids, _ = case.make_seed()
        seed_row, seed_response = case.request(
            f"pressure-{index:03d}-seed", seed, seed_ids
        )
        if seed_row["effective_reuse_len"] != 0:
            raise RuntimeError(f"prefix {index} unexpectedly hit before filling")
        if (seed_response["choices"][0]["message"].get("content") or "").strip() != "RECEIVED":
            raise RuntimeError(f"prefix {index} seed response was not RECEIVED")
        conversation, ids = build_conversation(case, seed, seed_ids)
        conversations.append((case, conversation, ids))

    records = []
    for round_index in range(args.rounds):
        for index, (case, conversation, ids) in enumerate(conversations):
            row, response = case.request(
                f"pressure-{index:03d}-round-{round_index:02d}",
                conversation,
                ids,
            )
            check_answer(response["choices"][0]["message"].get("content") or "", case.expected)
            records.append(
                {
                    "prefix": index,
                    "round": round_index,
                    "input_len": row["input_len"],
                    "reuse_len": row["effective_reuse_len"],
                    "elapsed_s": row["elapsed_s"],
                }
            )

    result = {
        "passed": True,
        "target_tokens": args.target_tokens,
        "prefix_count": args.prefix_count,
        "rounds": args.rounds,
        "elapsed_s": round(time.time() - started, 3),
        "records": records,
        "expected_by_prefix": [case.expected for case in cases],
    }
    (args.output / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
