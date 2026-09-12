#!/usr/bin/env python3
"""QPS saturation ladder sweep runner (Task #42).

Drives the cache_affinity_guardrail scenario across a QPS-ladder x cap x seed
matrix, one cell at a time per lane (parallel_runner --parallel 1). Reuses the
golden 3-family A/B/C profile at a FLAT 1:1:1 split (per-family X = qps/3), so
the ladder is a ratio-constant continuation of the golden 3-QPS anchor.

Design decisions (see results/saturation-sweep-20260912/RUN.md):
  * profile   : flat A (low_a_qps == peak_a_qps == X), B == C == X, X = qps/3.
                Mirrors golden (low_a==peak_a==b==c==1) -> steady saturation.
  * window    : total <= 240 s, sustain >= 2000 requests (>=3000 in practice);
                warm-up (low+ramp) longer at low QPS so the KV pool reaches
                eviction steady state before the sustain measurement window.
  * capacity  : prefill_cache_blocks per QPS rung (golden baseline r768); the
                pilot bumps a rung to r_safe if r768 triggers admission rejects.
  * cap arm   : injected via V3_GUARDRAIL_CAP_MS (0 / 500 / 1e8), never in YAML.
  * seed      : seed_phase 0/1/2 -> the golden cross-family Bresenham phase LUT.

Each cell generates a scenario YAML, pins lane ports, runs parallel_runner,
verifies guardrail-summary.json, and appends a status record to a lane manifest.
Fails SOFT per cell: a failing cell is recorded honestly and the lane continues.

Cell naming: cell_q{qps}_cap{caplabel}_s{seed}   (caplabel: inf for 1e8)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── matrix spec ──────────────────────────────────────────────────────────────
QPS_LADDER = [30, 60, 90, 120, 150]        # authorized (Task #42)
# Beyond-authorized busy-band probe.  The authorized {30..150} ladder (all at
# golden r768) was measured: busy_mean climbs monotonically with QPS and with
# the guardrail cap, topping at q150 cap-inf = 0.523 -- still under the online
# comparable GPU-duty band 0.61-0.70 that Q2/Q4 ask about.  These rungs extend
# the SAME r768 curve upward to locate that band.  CRITICAL: capacity stays at
# r768 (NOT bumped to r_safe).  A first attempt bumped q180 to 1024 blocks and
# busy FELL back to ~0.53 (flat vs q150) because the extra blocks raise the
# cache hit rate and steal prefill compute -- the capacity bump cancels the QPS
# gain.  q150 at r768 is reject-free with decode_waiting_peak=0 (queue
# headroom), so r768 sustains higher QPS cleanly; any reject that does appear
# at these rungs is recorded honestly per the Task #42 capacity rule.  Kept in
# a SEPARATE out-dir (sweep_ext), labeled beyond-authorized.  Rungs MUST be
# divisible by 3 (flat 1:1:1 A/B/C split -> build_yaml guard) and per-family
# QPS <= 96 (case validation ceiling).  From the q150 cap-inf slope (~0.0016
# busy/QPS) busy 0.6-0.7 lands near q210-263, so {180,210,240} brackets it.
QPS_LADDER_EXT = [180, 210, 240]
CAPS = [0, 500, 100000000]
SEEDS = [0, 1, 2]

# (low_s, ramp_s, sustain_s, recover_s) per QPS rung.
# total <= 240 s; sustain requests = qps * sustain_s >= 3000 (>> 2000 floor).
WINDOW = {
    30:  (50, 50, 100, 40),   # total 240, sustain 3000 req
    60:  (40, 30,  60, 30),   # total 160, sustain 3600 req
    90:  (30, 25,  45, 25),   # total 125, sustain 4050 req
    120: (25, 20,  35, 20),   # total 100, sustain 4200 req
    150: (20, 20,  30, 20),   # total  90, sustain 4500 req
    # extension rungs (beyond authorized ladder): sustain >= 4500 req
    180: (20, 20,  25, 20),   # total 85, sustain 4500 req
    210: (20, 20,  22, 20),   # total 82, sustain 4620 req
    240: (20, 20,  20, 20),   # total 80, sustain 4800 req
}

# prefill_cache_blocks per QPS rung. Golden baseline r768; pilot raises a rung
# to r_safe (recorded in the manifest) if r768 causes admission rejections.
CAPACITY = {30: 768, 60: 768, 90: 768, 120: 768, 150: 768,
            # Extension rungs KEEP golden r768 (not bumped to r_safe): a larger
            # pool raises the cache hit rate and suppresses busy, cancelling the
            # QPS gain (measured: q180@1024 busy ~0.53 == q150@768).  Staying at
            # r768 keeps the ext rungs on the same busy curve as the authorized
            # ladder; q150@768 is reject-free with queue headroom.  Any reject at
            # these rungs is recorded honestly per the Task #42 capacity rule.
            180: 768, 210: 768, 240: 768}
DECODE_BLOCKS = 4096
HARMFUL_DELTA_MS = 100
INSTANCE = "cache_affinity_guardrail::normal::batch-window"
COMPOSITION_SUBDIR = "rtp_llm/flexlb/tools/composition_v3"


def cap_label(cap):
    return "inf" if cap >= 100000000 else str(cap)


def cell_name(qps, cap, seed):
    return "cell_q%d_cap%s_s%d" % (qps, cap_label(cap), seed)


def all_cells(ladder=None):
    ladder = QPS_LADDER if ladder is None else ladder
    return [(q, c, s) for q in ladder for c in CAPS for s in SEEDS]


def build_yaml(qps, cap, seed):
    """Scenario YAML: capacity + window + flat 1:1:1 QPS + seed_phase."""
    if qps % 3 != 0:
        raise ValueError("qps %d not divisible by 3 (1:1:1 split)" % qps)
    x = qps // 3
    low_s, ramp_s, sustain_s, recover_s = WINDOW[qps]
    blocks = CAPACITY[qps]
    # parameters live under the variant (matches the proven golden_sweep_lane.sh
    # template exactly; case_config merges top-level then variant parameters).
    return (
        "schema_version: 2\n"
        "case: cache_affinity_guardrail\n"
        "environment:\n"
        "  backend: java_mock\n"
        "  n_prefill: 4\n"
        "  n_decode: 2\n"
        "  prefill_cache_blocks: %d\n"
        "  decode_cache_blocks: %d\n"
        "  perf_preset: default\n"
        "execution:\n"
        "  timeout_s: 1800\n"
        "  stage_timeout_s: 1500\n"
        "  cleanup_timeout_s: 120\n"
        "variants:\n"
        "- id: normal\n"
        "  use: normal\n"
        "  parameters:\n"
        "    low_s: %d\n"
        "    ramp_s: %d\n"
        "    sustain_s: %d\n"
        "    recover_s: %d\n"
        "    low_a_qps: %d\n"
        "    peak_a_qps: %d\n"
        "    b_qps: %d\n"
        "    c_qps: %d\n"
        "    harmful_delta_ms: %d\n"
        "    seed_phase: %d\n"
    ) % (blocks, DECODE_BLOCKS, low_s, ramp_s, sustain_s, recover_s,
         x, x, x, x, HARMFUL_DELTA_MS, seed)


def run_cell(qps, cap, seed, args):
    name = cell_name(qps, cap, seed)
    out = Path(args.out_dir)
    cell_dir = out / name
    # Idempotent re-run: the harness setup does makedirs(.../master-logs)
    # without exist_ok, so a stale partial cell dir raises FileExistsError.
    # Clear any prior products for this cell before re-creating it.
    if cell_dir.exists():
        shutil.rmtree(cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out / ("%s.yaml" % name)
    yaml_path.write_text(build_yaml(qps, cap, seed))

    master_base = args.port_base + args.lane * 10
    # Per-lane harness footprint = master[m..m+5] + mock[b-1..b+151] (153 wide).
    # Lanes MUST be separated by >= 153 or their mock windows overlap and the
    # machine-level port-window lock (/tmp/flexlb_ft_portlocks) rejects the
    # loser instantly (EAGAIN -> every cell fails in 0.1s).  Stride 160 keeps
    # the whole 2-lane matrix inside the leased 400-port block:
    #   lane0: master 61000-61005, mock 61049-61201
    #   lane1: master 61010-61015, mock 61209-61361
    mock_base = args.port_base + 50 + args.lane * 160
    env = dict(os.environ)
    env["V3_MOCK_JAR"] = args.mock_jar
    env["V3_GUARDRAIL_CAP_MS"] = str(cap)
    env["FLEXLB_FT_PARALLEL_MASTER_BASE"] = str(master_base)
    env["FLEXLB_FT_PARALLEL_MOCK_BASE"] = str(mock_base)
    env["PYTHONPATH"] = ".:mechanism"

    comp_dir = Path(args.repo_dir) / COMPOSITION_SUBDIR
    cmd = [
        sys.executable, "-u", "parallel_runner.py",
        "--source", "yaml", "--case-dir", str(yaml_path),
        "--parallel", "1", "--shard", "case",
        "--instances", INSTANCE, "--out-dir", str(cell_dir),
    ]
    log_path = out / ("%s.log" % name)
    t0 = time.time()
    with open(log_path, "w") as lf:
        rc = subprocess.call(cmd, cwd=str(comp_dir), env=env,
                             stdout=lf, stderr=subprocess.STDOUT)
    wall = round(time.time() - t0, 1)

    summary = next(iter(sorted(cell_dir.glob("**/guardrail-summary.json"))), None)
    rec = {
        "cell": name, "qps": qps, "cap": cap, "cap_label": cap_label(cap),
        "seed": seed, "per_family_qps": qps // 3,
        "capacity_blocks": CAPACITY[qps],
        "window": dict(zip(("low_s", "ramp_s", "sustain_s", "recover_s"), WINDOW[qps])),
        "sustain_requests_planned": qps * WINDOW[qps][2],
        "master_base": master_base, "mock_base": mock_base,
        "rc": rc, "wall_s": wall,
        "status": "OK" if summary else "NO_SUMMARY",
        "summary": str(summary) if summary else None,
        "log": str(log_path),
    }
    return rec


def select_cells(args):
    ladder = QPS_LADDER_EXT if args.ladder == "ext" else QPS_LADDER
    cells = all_cells(ladder)
    if args.cells:
        want = set(args.cells.split(","))
        cells = [c for c in cells if cell_name(*c) in want]
        missing = want - {cell_name(*c) for c in cells}
        if missing:
            raise SystemExit("unknown cells: %s" % sorted(missing))
    elif args.shard_total > 1:
        cells = [c for i, c in enumerate(cells) if i % args.shard_total == args.shard_idx]
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="sweep output dir (cell dirs + manifest)")
    ap.add_argument("--repo-dir", required=True, help="repo root containing rtp_llm/...")
    ap.add_argument("--mock-jar", required=True, help="path to flexlb-mock-engine-*-all.jar")
    ap.add_argument("--port-base", type=int, required=True, help="lease port base for this machine")
    ap.add_argument("--lane", type=int, default=0, help="port lane offset on this machine (0 = single lane)")
    ap.add_argument("--shard-idx", type=int, default=0, help="cell shard index (which machine)")
    ap.add_argument("--shard-total", type=int, default=1, help="total cell shards (machines)")
    ap.add_argument("--ladder", choices=["authorized", "ext"], default="authorized",
                    help="authorized = Task #42 {30..150}; ext = beyond-authorized busy-band probe {180,210,240}")
    ap.add_argument("--cells", default=None, help="explicit comma-separated cell names (pilot)")
    ap.add_argument("--print-plan", action="store_true", help="print the cell plan and exit")
    args = ap.parse_args()

    cells = select_cells(args)
    if args.print_plan:
        for (q, c, s) in cells:
            print(cell_name(q, c, s), "cap_blocks=%d" % CAPACITY[q],
                  "window=%s" % (WINDOW[q],), "sustain_req=%d" % (q * WINDOW[q][2]))
        print("total cells: %d" % len(cells))
        return 0

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / ("lane%d_shard%d_manifest.jsonl" % (args.lane, args.shard_idx))
    print("[lane %d shard %d/%d] %d cells -> %s"
          % (args.lane, args.shard_idx, args.shard_total, len(cells), out), flush=True)
    n_ok = 0
    with open(manifest, "a") as mf:
        for (qps, cap, seed) in cells:
            name = cell_name(qps, cap, seed)
            print("[lane %d] START %s" % (args.lane, name), flush=True)
            try:
                rec = run_cell(qps, cap, seed, args)
            except Exception as exc:  # recorded honestly, lane continues
                rec = {"cell": name, "qps": qps, "cap": cap, "seed": seed,
                       "status": "EXCEPTION", "error": "%s: %s" % (type(exc).__name__, exc)}
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            n_ok += 1 if rec.get("status") == "OK" else 0
            print("[lane %d] DONE %s status=%s rc=%s wall=%ss"
                  % (args.lane, name, rec.get("status"), rec.get("rc"), rec.get("wall_s")),
                  flush=True)
    print("[lane %d] complete: %d/%d OK" % (args.lane, n_ok, len(cells)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
