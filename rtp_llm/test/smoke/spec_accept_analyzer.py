"""Analyze [sp_accept_trace] lines emitted by MtpExecutor when the server is
launched with RTP_SP_ACCEPT_TRACE=1.

Produces a summary of accept_len distribution, grouped by grammar kind, so we
can quantify the degradation of chain-spec accept_rate under strict grammar.

Usage:
    python spec_accept_analyzer.py <log_file_or_dir> [<log_file_or_dir> ...]

Input format (one line per stream per decode step):
    [sp_accept_trace] stream_id=<id> grammar=<kind> propose_step=<n> accept_len=<k>
Older logs without propose_step are also accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List

LINE_RE = re.compile(
    r"\[sp_accept_trace\]\s+"
    r"stream_id=(?P<stream_id>-?\d+)\s+"
    r"grammar=(?P<grammar>\S+)\s+"
    r"(?:propose_step=(?P<propose_step>\d+)\s+)?"
    r"accept_len=(?P<accept_len>\d+)"
)


def iter_log_files(paths: List[str]) -> Iterable[str]:
    for p in paths:
        if os.path.isfile(p):
            yield p
            continue
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for fn in files:
                    if fn.endswith((".log", ".txt", ".out", ".err")):
                        yield os.path.join(root, fn)


def parse_lines(paths: List[str]):
    # grammar_kind -> list[accept_len]
    by_grammar: Dict[str, List[int]] = defaultdict(list)
    # (grammar_kind, stream_id) -> list[accept_len]
    by_stream: Dict[tuple, List[int]] = defaultdict(list)
    propose_steps: set = set()
    total_lines = 0

    for path in iter_log_files(paths):
        try:
            with open(path, "r", errors="replace") as f:
                for line in f:
                    m = LINE_RE.search(line)
                    if not m:
                        continue
                    total_lines += 1
                    kind = m.group("grammar")
                    sid = int(m.group("stream_id"))
                    pstep = m.group("propose_step")
                    alen = int(m.group("accept_len"))
                    by_grammar[kind].append(alen)
                    by_stream[(kind, sid)].append(alen)
                    if pstep is not None:
                        propose_steps.add(int(pstep))
        except OSError as e:
            print(f"WARN: cannot read {path}: {e}", file=sys.stderr)

    return by_grammar, by_stream, propose_steps, total_lines


def summarize(vals: List[int]) -> dict:
    if not vals:
        return {"count": 0}
    s = sorted(vals)
    n = len(s)

    def pct(p: float) -> float:
        if n == 1:
            return float(s[0])
        k = max(0, min(n - 1, int(round(p * (n - 1)))))
        return float(s[k])

    return {
        "count": n,
        "mean": round(statistics.fmean(s), 3),
        "stdev": round(statistics.pstdev(s), 3) if n > 1 else 0.0,
        "min": s[0],
        "p50": pct(0.5),
        "p90": pct(0.9),
        "p95": pct(0.95),
        "max": s[-1],
        # Histogram by accept_len value — the key signal. For propose_step=4
        # we expect values in [1, 5]; concentration near 1 means grammar is
        # killing speculation.
        "histogram": dict(sorted(Counter(s).items())),
    }


def print_report(by_grammar, by_stream, propose_steps, total_lines):
    print("=" * 72)
    print(f"sp_accept_trace: parsed {total_lines} lines, "
          f"propose_steps observed = {sorted(propose_steps)}")
    print("=" * 72)

    kinds = sorted(by_grammar.keys())
    if not kinds:
        print("No [sp_accept_trace] lines found. "
              "Did the server run with RTP_SP_ACCEPT_TRACE=1?")
        return

    # Summary table
    print()
    print(f"{'grammar_kind':<16} {'count':>8} {'mean':>8} {'p50':>6} "
          f"{'p90':>6} {'p95':>6} {'min':>5} {'max':>5}")
    print("-" * 72)
    for kind in kinds:
        s = summarize(by_grammar[kind])
        print(f"{kind:<16} {s['count']:>8} {s['mean']:>8} {int(s['p50']):>6} "
              f"{int(s['p90']):>6} {int(s['p95']):>6} {s['min']:>5} {s['max']:>5}")

    # Histograms
    print()
    for kind in kinds:
        s = summarize(by_grammar[kind])
        print(f"[{kind}] accept_len histogram ({s['count']} samples)")
        total = s["count"]
        for alen, cnt in s["histogram"].items():
            bar = "#" * int(round(40 * cnt / total))
            pct_str = f"{100 * cnt / total:5.1f}%"
            print(f"  accept_len={alen:<3} | {cnt:>6} ({pct_str}) {bar}")
        print()

    # Comparison (baseline = grammar==none, target = every other)
    if "none" in by_grammar:
        baseline = summarize(by_grammar["none"])
        print("Grammar vs baseline (grammar=none) comparison:")
        print(f"  baseline mean accept_len = {baseline['mean']}")
        print("-" * 72)
        for kind in kinds:
            if kind == "none":
                continue
            cur = summarize(by_grammar[kind])
            if baseline["mean"] > 0:
                ratio = cur["mean"] / baseline["mean"]
            else:
                ratio = float("nan")
            verdict = _verdict(ratio)
            print(f"  {kind:<16}: mean={cur['mean']:>6}  ratio={ratio:.3f}  -> {verdict}")

    # Per-stream deep dive — useful when a few streams dominate
    print()
    print("Top 5 streams by sample count per grammar kind:")
    print("-" * 72)
    for kind in kinds:
        streams_for_kind = [(sid, vals) for (k, sid), vals in by_stream.items() if k == kind]
        streams_for_kind.sort(key=lambda x: -len(x[1]))
        for sid, vals in streams_for_kind[:5]:
            s = summarize(vals)
            print(f"  {kind:<12} stream_id={sid:<10}  steps={s['count']:<4}  "
                  f"mean={s['mean']:<6}  hist={s['histogram']}")


def _verdict(ratio: float) -> str:
    # Used to frame the decision for solution B / D / E.
    if ratio != ratio:  # NaN
        return "n/a"
    if ratio >= 0.8:
        return "OK (no action)"
    if ratio >= 0.5:
        return "DEGRADED (consider solution D top-k mask)"
    if ratio >= 0.3:
        return "BAD (solution B draft-side mask recommended)"
    return "COLLAPSE (solution B mandatory)"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help="Log files or directories to scan.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of the human report.")
    args = p.parse_args()

    by_grammar, by_stream, propose_steps, total_lines = parse_lines(args.paths)

    if args.json:
        out = {
            "total_lines": total_lines,
            "propose_steps": sorted(propose_steps),
            "by_grammar": {k: summarize(v) for k, v in by_grammar.items()},
            "by_stream": {
                f"{k}:{sid}": summarize(v)
                for (k, sid), v in by_stream.items()
            },
        }
        json.dump(out, sys.stdout, indent=2, default=str)
        print()
    else:
        print_report(by_grammar, by_stream, propose_steps, total_lines)


if __name__ == "__main__":
    main()
