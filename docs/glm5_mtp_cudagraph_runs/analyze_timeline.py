#!/usr/bin/env python3
"""
Inspect a Perfetto / Chrome Trace JSON emitted by RTP-LLM's GPU profiler
and report whether the three MTP-decode phases ran under CUDA-graph replay:

  - target verify     (MTP target model verify forward)
  - draft prefill     (MTP draft model prefill forward, the multi-token pass
                       inside each decode step)
  - draft decode      (MTP draft model decode loop iterations)

Usage:
  ./analyze_timeline.py <trace.json> [more.json ...]
"""

import json
import sys
from collections import Counter


def analyze(path: str) -> None:
    with open(path) as fp:
        data = json.load(fp)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    names: Counter = Counter()
    for ev in events:
        names[ev.get("name", "?")] += 1

    target_verify_count = names.get("executor.mtp.decode_step(target_model_verify)", 0)
    draft_prefill_total = sum(
        c
        for n, c in names.items()
        if n.startswith("executor.mtp.decode_step(draft_model_forward")
    )
    draft_decode_loop = sum(
        c
        for n, c in names.items()
        if n.startswith("executor.mtp.draft_model_decode(loop_iter=")
    )

    target_forward = names.get(
        "py_model.forward(cuda_graph=1,prefill_cg=0,model_id=0)", 0
    )
    draft_forward = names.get(
        "py_model.forward(cuda_graph=1,prefill_cg=0,model_id=1)", 0
    )
    draft_prefill_forward = sum(
        c
        for n, c in names.items()
        if n.startswith("py_model.forward(cuda_graph=1,prefill_cg=1,model_id=1)")
    )

    replay_decode = names.get("cuda_graph.forward(replayDecode)", 0)
    replay_prefill = names.get("cuda_graph.forward(replayPrefill)", 0)
    graph_launch = names.get("cudaGraphLaunch", 0)

    # Heuristic about sp_prefill_draft_model_ usage from the perf labels
    sp_labels = [
        n for n in names if n.startswith("executor.mtp.decode_step(draft_model_forward")
    ]
    sp_label = sp_labels[0] if sp_labels else "(missing)"

    print(f"=== {path} ===")
    print(f"  target verify          steps: {target_verify_count}")
    print(f"  draft prefill          steps: {draft_prefill_total}")
    print(f"  draft decode loop iter steps: {draft_decode_loop}")
    print()
    print(f"  py_model.forward(target, model_id=0)        : {target_forward}")
    print(f"  py_model.forward(draft  decode, model_id=1) : {draft_forward}")
    print(
        f"  py_model.forward(draft  prefill, model_id=1, prefill_cg=1) : {draft_prefill_forward}"
    )
    print()
    print(f"  cuda_graph.forward(replayDecode)  : {replay_decode}")
    print(f"  cuda_graph.forward(replayPrefill) : {replay_prefill}")
    print(f"  cudaGraphLaunch                    : {graph_launch}")
    print()
    print(f"  draft_model_forward label sample  : {sp_label}")

    # Sanity check
    expected_launches = target_verify_count + draft_prefill_total + draft_decode_loop
    print()
    if expected_launches == 0:
        print("  (no MTP decode events found — not a decode trace?)")
    elif graph_launch >= expected_launches:
        print(
            f"  ✓ cudaGraphLaunch ({graph_launch}) >= "
            f"target_verify + draft_prefill + draft_decode "
            f"({target_verify_count}+{draft_prefill_total}+{draft_decode_loop}"
            f"={expected_launches}) — all 3 phases captured."
        )
    else:
        deficit = expected_launches - graph_launch
        print(
            f"  ✗ cudaGraphLaunch={graph_launch}  expected>={expected_launches} "
            f"(target_verify={target_verify_count}, draft_prefill={draft_prefill_total}, "
            f"draft_decode={draft_decode_loop}) — deficit {deficit}."
        )
        if "sp_prefill_cg=0" in sp_label or "sp_cg=0" in sp_label:
            print(
                "    Hint: draft_model_forward label shows sp_cg=0/sp_prefill_cg=0 — "
                "MtpExecutor did not construct sp_prefill_draft_model_. "
                "Set RTP_LLM_FORCE_SP_PREFILL_CUDA_GRAPH=1 for the decode side."
            )
    print()


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    for p in sys.argv[1:]:
        analyze(p)


if __name__ == "__main__":
    main()
