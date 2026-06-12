#!/usr/bin/env python3
"""Measure vLLM MTP acceptance on the GLM5 full-checkpoint smoke queries."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

DEFAULT_QUERY_FILE = Path(
    "/home/admin/zw193905/RTP-LLM/github-opensource/"
    "internal_source/rtp_llm/test/smoke/data/model/glm5/"
    "glm_5_fp8_full_q_r_h20_mtp_mega_moe_pd.json"
)
DEFAULT_MODEL = Path("/home/admin/zw193905/models/GLM-5-FP8")
DEFAULT_OUTPUT = Path(
    "/home/admin/zw193905/RTP-LLM/github-opensource/"
    "docs/glm5_eagle_vllm_acceptance_20260529.json"
)


def _top_k(query: dict) -> int:
    if query.get("top_k") is not None:
        return int(query["top_k"])
    return int((query.get("extend_fields") or {}).get("top_k", 1))


def _metrics_snapshot(llm: LLM) -> dict:
    counters: dict[str, int] = {}
    vectors: dict[str, list[int]] = {}
    for metric in llm.get_metrics():
        if isinstance(metric, Counter):
            counters[metric.name] = counters.get(metric.name, 0) + int(metric.value)
        elif isinstance(metric, Vector):
            cur = vectors.setdefault(metric.name, [0] * len(metric.values))
            if len(cur) < len(metric.values):
                cur.extend([0] * (len(metric.values) - len(cur)))
            for i, value in enumerate(metric.values):
                cur[i] += int(value)
    return {"counters": counters, "vectors": vectors}


def _diff_metrics(before: dict, after: dict) -> dict:
    names = set(before["counters"]) | set(after["counters"])
    counter_delta = {
        name: int(after["counters"].get(name, 0) - before["counters"].get(name, 0))
        for name in sorted(names)
    }
    vector_delta: dict[str, list[int]] = {}
    for name in set(before["vectors"]) | set(after["vectors"]):
        b = before["vectors"].get(name, [])
        a = after["vectors"].get(name, [])
        n = max(len(a), len(b))
        vector_delta[name] = [
            int((a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0))
            for i in range(n)
        ]
    return {"counters": counter_delta, "vectors": vector_delta}


def _acceptance_from_delta(delta: dict) -> dict:
    drafts = int(delta["counters"].get("vllm:spec_decode_num_drafts", 0))
    draft_tokens = int(delta["counters"].get("vllm:spec_decode_num_draft_tokens", 0))
    accepted = int(delta["counters"].get("vllm:spec_decode_num_accepted_tokens", 0))
    per_pos = delta["vectors"].get("vllm:spec_decode_num_accepted_tokens_per_pos", [])
    return {
        "num_drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "acceptance_rate": (accepted / draft_tokens) if draft_tokens else None,
        "acceptance_length": (1.0 + accepted / drafts) if drafts else None,
        "per_position_acceptance": [
            (value / drafts) if drafts else None for value in per_pos
        ],
        "per_position_accepted_counts": per_pos,
    }


def _finish_reason(reason) -> str:
    if reason in ("stop", "length", "tool_calls", "function_call"):
        return reason
    return "stop" if reason is None else str(reason)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-file", type=Path, default=DEFAULT_QUERY_FILE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tp", type=int, default=int(os.environ.get("VLLM_TP_SIZE", "8"))
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=0)
    args = parser.parse_args()

    with args.query_file.open(encoding="utf-8") as f:
        task = json.load(f)
    query_items = task["query_result"]
    if args.max_queries > 0:
        query_items = query_items[: args.max_queries]

    print(
        f"Loading vLLM model={args.model} tp={args.tp} "
        f"spec_tokens=3 max_model_len={args.max_model_len}",
        flush=True,
    )
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        enforce_eager=True,
        disable_log_stats=False,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 3,
            "draft_sample_method": "greedy",
        },
    )
    tokenizer = llm.get_tokenizer()

    before_all = _metrics_snapshot(llm)
    results = []
    for idx, item in enumerate(query_items):
        query = item["query"]
        prompt = tokenizer.apply_chat_template(
            query["messages"], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt)
        max_tokens = (
            args.max_tokens
            if args.max_tokens > 0
            else int(query.get("max_tokens", 1024))
        )
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=float(query.get("temperature", 0.0)),
            top_p=float(query.get("top_p", 1.0)),
            top_k=_top_k(query),
        )
        print(
            f"Query {idx}: prompt_tokens={len(prompt_ids)} max_tokens={params.max_tokens} "
            f"temperature={params.temperature} top_p={params.top_p} top_k={params.top_k}",
            flush=True,
        )
        before = _metrics_snapshot(llm)
        output = llm.generate([prompt], params)[0]
        after = _metrics_snapshot(llm)
        delta = _diff_metrics(before, after)
        acceptance = _acceptance_from_delta(delta)
        choice = output.outputs[0]
        print(
            f"Query {idx} done: completion_tokens={len(choice.token_ids)} "
            f"acceptance_length={acceptance['acceptance_length']} "
            f"accepted={acceptance['accepted_tokens']} drafts={acceptance['num_drafts']}",
            flush=True,
        )
        results.append(
            {
                "query_index": idx,
                "prompt_tokens": len(output.prompt_token_ids or prompt_ids),
                "completion_tokens": len(choice.token_ids),
                "finish_reason": _finish_reason(choice.finish_reason),
                "acceptance": acceptance,
                "metric_delta": delta,
                "output_token_ids": list(choice.token_ids),
                "output_preview": choice.text[:500],
            }
        )

    after_all = _metrics_snapshot(llm)
    all_delta = _diff_metrics(before_all, after_all)
    payload = {
        "engine": "vllm",
        "model": str(args.model),
        "query_file": str(args.query_file),
        "tensor_parallel_size": args.tp,
        "speculative_config": {
            "method": "mtp",
            "num_speculative_tokens": 3,
            "draft_sample_method": "greedy",
        },
        "overall_acceptance": _acceptance_from_delta(all_delta),
        "overall_metric_delta": all_delta,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
