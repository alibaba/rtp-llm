"""Compare verified replay totals over the same ordered input interval."""
import argparse
import heapq
import json
import struct
from pathlib import Path


def load(path):
    if not (path / "SUCCESS").is_file():
        raise ValueError(f"incomplete run: {path}")
    ident = json.loads((path / "identity.json").read_bytes())
    stats = json.loads((path / "native/summary.json").read_bytes())["total"]
    return ident, stats


def compare(paths):
    reference, first = load(paths[0])
    baseline = first["joint_tokens"]["sum"]
    rows = []
    for path in paths:
        ident, t = load(path)
        if (ident["trace_sha256"], ident["warmup"], t["requests"], t["input_tokens"]) != (
            reference["trace_sha256"], reference["warmup"], first["requests"], first["input_tokens"]
        ):
            raise ValueError(f"different trace or measurement range: {path}")
        full, joint = t["full_tokens"]["sum"], t["joint_tokens"]["sum"]
        rows.append(dict(run=str(path), revision=ident["revision"], requests=t["requests"], warmup=ident["warmup"],
                         input_tokens=t["input_tokens"], full_rate=full/t["input_tokens"],
                         joint_rate=joint/t["input_tokens"], lost_to_joint_tokens=full-joint,
                         attainment=joint/baseline if baseline else None,
                         evictions=t.get("evictions", {})))
    return rows


def cases(reference, candidate, limit):
    """Compare the facts written by the last invocation of each backend."""
    ident, total = load(reference)
    paths = [p / "native/match-lengths.u64" for p in (reference, candidate)]
    size = paths[0].stat().st_size
    if size != paths[1].stat().st_size or size % 16:
        raise ValueError("per-request files cover different stages; compare corresponding checkpoints")
    start = ident["warmup"] + total["requests"] - size // 16 + 1
    gains = losses = gained = lost = 0
    worst = []
    with paths[0].open("rb") as first, paths[1].open("rb") as second:
        ordinal = start
        while True:
            a, b = first.read(16 * 8192), second.read(16 * 8192)
            if not a:
                break
            for (f, j), (g, k) in zip(struct.iter_unpack("<QQ", a), struct.iter_unpack("<QQ", b)):
                if j > k:
                    losses += 1
                    lost += j - k
                    entry = (j - k, ordinal, f, j, g, k)
                    if len(worst) < limit:
                        heapq.heappush(worst, entry)
                    else:
                        heapq.heappushpop(worst, entry)
                elif k > j:
                    gains += 1
                    gained += k - j
                ordinal += 1
    return dict(first_ordinal=start, requests=size // 16, improved=gains, regressed=losses,
                gained_tokens=gained, lost_tokens=lost,
                worst=[dict(loss=d, request=i, baseline_full=f, baseline_joint=j,
                            candidate_full=g, candidate_joint=k) for d, i, f, j, g, k in sorted(worst, reverse=True)])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path, help="baseline first")
    p.add_argument("--cases", type=int, help="include the N largest per-request regressions")
    p.add_argument("--html", type=Path, help="write a standalone HTML report")
    p.add_argument("--theory-run", type=Path, help="aligned infinite-capacity full-retention reference for charts")
    args = p.parse_args()
    rows = compare(args.runs)
    if args.cases is not None:
        if args.cases < 1 or len(rows) < 2:
            p.error("--cases requires N >= 1 and at least two runs")
        for path, row in zip(args.runs[1:], rows[1:]):
            row["last_stage_cases"] = cases(args.runs[0], path, args.cases)
    if args.html and len(args.runs) == 1 and args.theory_run is None:
        from single_report import render_run
        render_run(args.runs[0], args.html)
    elif args.html:
        from report import render, with_charts
        import subprocess
        import sys
        reports = []
        for i, path in enumerate(args.runs, 1):
            dest = args.html.parent / (args.html.stem + f"-run-{i}")
            cmd = [sys.executable, str(Path(__file__).with_name("detailed.py")), str(path), "--output", str(dest)]
            if args.theory_run:
                cmd += ["--theory-run", str(args.theory_run)]
            subprocess.run(cmd, check=True)
            reports.append((str(path), dest))
        args.html.write_text(with_charts(render(rows), reports), encoding="utf-8")
    print(json.dumps(rows, indent=2))
