#!/usr/bin/env python3
"""Generate reproducible independent-prefix batches without loading a model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Keep the canonical script usable both as a module and by file path.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from rtp_llm.test.perf_test.cache_grid.runner.workspace_budget import (
    validate_token_budget,
    workspace_capacity,
)

MAX_REQUEST_TOKENS = 256 * 1024


def round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def generate_grid(args: argparse.Namespace) -> dict:
    if args.num_cases <= 0 or not args.batch_sizes or min(args.batch_sizes) <= 0:
        raise ValueError("num-cases and batch-sizes must be positive")
    if args.input_alignment <= 0 or args.cache_alignment <= 0:
        raise ValueError("alignments must be positive")
    if not 0 < args.min_input_tokens <= args.max_input_tokens:
        raise ValueError("input bounds must satisfy 0 < min <= max")
    if args.commit_tail_tokens <= 0 or args.commit_tail_tokens % args.cache_alignment:
        raise ValueError(
            "commit-tail-tokens must be a positive multiple of cache-alignment"
        )
    if not 0 <= args.cold_probability <= 1:
        raise ValueError("cold-probability must be between 0 and 1")
    if args.max_batch_tokens <= 0 or (
        args.kv_budget_tokens is not None and args.kv_budget_tokens <= 0
    ):
        raise ValueError("token budgets must be positive")
    workspace = None
    if args.workspace_tokens is not None:
        workspace = workspace_capacity(args.workspace_tokens, args.cache_alignment)
        validate_token_budget(
            [],
            workspace_tokens=workspace,
            commit_tail=args.commit_tail_tokens,
            block=args.cache_alignment,
            token_budget=args.max_batch_tokens,
        )
    minimum = round_up(args.min_input_tokens, args.input_alignment)
    maximum = args.max_input_tokens // args.input_alignment * args.input_alignment
    block = args.cache_alignment
    minimum_kv = round_up(minimum, block)
    if minimum > maximum:
        raise ValueError("input range contains no aligned length")
    for batch in args.batch_sizes:
        if workspace is not None and batch * round_up(minimum + 1, block) > workspace:
            raise ValueError(
                f"batch={batch} cannot fit minimum inputs in workspace_tokens"
            )
        if batch * minimum > args.max_batch_tokens:
            raise ValueError(
                f"batch={batch} cannot fit minimum inputs in max-batch-tokens"
            )
        if (
            args.kv_budget_tokens is not None
            and batch * minimum_kv > args.kv_budget_tokens
        ):
            raise ValueError(
                f"batch={batch} cannot fit minimum inputs in kv-budget-tokens"
            )

    rng = random.Random(args.seed)
    cases = []
    seen = set()
    # Bounds retries when a tiny shape space cannot supply enough distinct cases.
    for _ in range(max(1000, args.num_cases * 100)):
        if len(cases) == args.num_cases:
            break
        batch = rng.choice(args.batch_sizes)
        groups = []
        input_sum = peak_kv = workspace_used = 0
        for index in range(batch):
            remaining = batch - index - 1
            upper = min(
                maximum,
                args.max_batch_tokens - input_sum - remaining * minimum,
            )
            if workspace is not None:
                available = (
                    workspace
                    - workspace_used
                    - remaining * round_up(minimum + 1, block)
                )
                upper = min(upper, available // block * block - 1)
            if args.kv_budget_tokens is not None:
                available = args.kv_budget_tokens - peak_kv - remaining * minimum_kv
                upper = min(upper, available // block * block)
            upper = upper // args.input_alignment * args.input_alignment
            input_len = rng.randrange(minimum, upper + 1, args.input_alignment)
            rounded_input = round_up(input_len, block)
            cache_step = args.commit_tail_tokens
            max_cache_blocks = max(
                0, (input_len - args.commit_tail_tokens) // cache_step
            )
            cache_allowed = args.kv_budget_tokens is None or (
                peak_kv
                + rounded_input
                + args.commit_tail_tokens
                + remaining * minimum_kv
                <= args.kv_budget_tokens
            )
            cache_len = 0
            if (
                cache_allowed
                and max_cache_blocks
                and rng.random() >= args.cold_probability
            ):
                cache_len = rng.randint(1, max_cache_blocks) * cache_step
            groups.append({"count": 1, "input_len": input_len, "cache_len": cache_len})
            input_sum += input_len
            workspace_used += round_up(input_len + 1, block)
            # Independent prefixes: include one divergent seed tail per hit request.
            peak_kv += rounded_input + (args.commit_tail_tokens if cache_len else 0)
        rng.shuffle(groups)
        signature = tuple((g["input_len"], g["cache_len"]) for g in groups)
        if signature in seen:
            continue
        seen.add(signature)
        cached = sum(g["cache_len"] for g in groups)
        cases.append(
            {
                "case_id": len(cases),
                "batch_size": batch,
                "prefix_policy": "independent",
                "request_groups": groups,
                "input_tokens_sum": input_sum,
                "cache_tokens_sum": cached,
                "new_prefill_tokens_sum": input_sum - cached,
                "estimated_peak_kv_tokens": peak_kv,
            }
        )
    if len(cases) != args.num_cases:
        raise ValueError(
            "not enough distinct batches; reduce num-cases or widen input bounds"
        )
    if workspace is not None:
        validate_token_budget(
            cases,
            workspace_tokens=workspace,
            commit_tail=args.commit_tail_tokens,
            block=block,
            token_budget=args.max_batch_tokens,
        )
    return {
        "schema_version": 2,
        "kind": "random_independent_cache_batch",
        "generator": {
            "name": "generate_random_batch_grid",
            "version": 4,
            "workspace_tokens": args.workspace_tokens,
            "seed": args.seed,
            "alignment": args.input_alignment,
            "cache_alignment": block,
            "parameters": {
                k: v
                for k, v in vars(args).items()
                if k not in ("output", "output_dir", "batch_input_limit")
            },
            "sampling": "budget-constrained sequential sampling with shuffled request slots",
            "kv_estimate_scope": "one active batch plus seed tails; excludes prior rounds, fixed/state pools and runtime memory",
        },
        "summary": {"case_count": len(cases), "max_request_tokens": maximum},
        "cases": cases,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path)
    output.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--batch-input-limit", action="append", default=[], metavar="B:TOKENS"
    )
    parser.add_argument(
        "--workspace-tokens",
        type=int,
        help="Optional fixed packed-token capacity including per-request padding/output reserve",
    )
    parser.add_argument("--num-cases", type=int, default=100)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--max-batch-tokens", type=int, default=1048576)
    parser.add_argument("--min-input-tokens", type=int, default=8192)
    parser.add_argument("--max-input-tokens", type=int, default=MAX_REQUEST_TOKENS)
    parser.add_argument("--input-alignment", type=int, default=256)
    parser.add_argument("--cache-alignment", type=int, default=4096)
    parser.add_argument("--commit-tail-tokens", type=int, default=4096)
    parser.add_argument("--cold-probability", type=float, default=0.2)
    parser.add_argument(
        "--kv-budget-tokens",
        type=int,
        default=None,
        help="Optional logical KV token budget, with headroom already deducted",
    )
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args(argv)


def build_plans(args):
    if args.output is not None:
        if args.batch_input_limit:
            raise ValueError("batch-input-limit requires output-dir")
        return [(args.output, generate_grid(args))]
    if not args.batch_sizes or min(args.batch_sizes) <= 0:
        raise ValueError("batch-sizes must be positive")
    if args.max_input_tokens <= 0:
        raise ValueError("max-input-tokens must be positive")
    limits = {}
    for item in args.batch_input_limit:
        try:
            batch, tokens = map(int, item.split(":"))
        except ValueError as exc:
            raise ValueError("batch-input-limit must have the form B:TOKENS") from exc
        if batch not in args.batch_sizes or batch in limits:
            raise ValueError(
                "batch-input-limit must name a unique requested batch size"
            )
        if not 0 < tokens <= args.max_input_tokens:
            raise ValueError("batch-input-limit must not exceed max-input-tokens")
        limits[batch] = tokens
    plans = []
    for batch in sorted(set(args.batch_sizes)):
        local = argparse.Namespace(**vars(args))
        local.batch_sizes = [batch]
        # Do not divide the budget equally: later slots use the remaining budget.
        minimum = round_up(args.min_input_tokens, args.input_alignment)
        local.max_input_tokens = min(
            limits.get(batch, args.max_input_tokens),
            args.max_batch_tokens - (batch - 1) * minimum,
        )
        local.max_input_tokens = (
            local.max_input_tokens // args.input_alignment * args.input_alignment
        )
        local.seed = args.seed + batch
        grid = generate_grid(local)
        path = (
            args.output_dir / f"batch_{batch:03d}_input_{local.max_input_tokens}.json"
        )
        plans.append((path, grid))
    return plans


def main(argv=None):
    args = parse_args(argv)
    try:
        plans = build_plans(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    # Check all destinations before writing, preserving existing resume plans.
    for path, _ in plans:
        if path.exists():
            raise SystemExit(f"Refusing to overwrite existing plan: {path}")
    for path, grid in plans:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(grid, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"Generated {len(grid['cases'])} cases: {path}")


if __name__ == "__main__":
    main()
