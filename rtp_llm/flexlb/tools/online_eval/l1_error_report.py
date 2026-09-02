#!/usr/bin/env python3
"""l1_error_report.py — L1 time-grid error table: mock formulas vs real engine.

Inputs:
  --real   l1_real_medians.json     (l1_grid_runner.py extract)
  --mock   l1_mock_reference.json   (l1_mock_reference.py / L1MockReferenceMain)

For every grid cell the relative error is (mock_ms - real_ms) / real_ms.
Acceptance statistics use |rel_err| (P50 / P90 / max); the signed median is
reported alongside so systematic over-/under-prediction stays visible.

Calibers:
  --real-caliber    client | engine | auto (default auto: engine median
                    when the extract found engine_events rows, client
                    otherwise)
  --decode-caliber  budget | first_excl (default budget). 'budget' =
                    decodeMs(ol, batch) — the engine step-budget caliber
                    ceil(ol/tokensPerStep) x stepMs; 'first_excl' =
                    ceil((ol-1)/tokensPerStep) x stepMs, matching the
                    client total_ms - ttft_ms decode tail.

Acceptance gates (defaults follow the L1 spec):
  P50(|err|) <= 10%  and  P90(|err|) <= 20%, evaluated overall AND per axis
  (prefill / decode); the run passes only when every scope passes.
  --no-gate keeps the exit code 0 regardless (report still shows verdicts).

Structural-bias detection: within each prefill cache-mode group the signed
error is correlated against input_len via Spearman rank correlation
(ties get average ranks); |rho| > 0.7 with two-sided p < 0.05 (t
approximation, normal CDF via math.erfc) flags a monotone structural bias
— a missing shape term in the formula, not random noise.

Outputs: terminal table, --out JSON, optional --html heatmap page (offline,
inline CSS, no external assets).

Usage:
  python3 l1_error_report.py --real l1_real_medians.json \\
      --mock l1_mock_reference.json --out l1_error_report.json [--html ...] \\
      [--real-caliber auto] [--decode-caliber budget] \\
      [--gate-p50 0.10] [--gate-p90 0.20] [--no-gate]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ERR_DISPLAY_CAP = 999.0  # cap absurd ratios only for terminal display width


# ---------------------------------------------------------------------------
# math helpers (no numpy/scipy dependency — pure stdlib)


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy 'linear' method), 0..100."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _ranks(values: list[float]) -> list[float]:
    """Average ranks (ties share the mean of their rank span)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def spearman(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """Spearman rho + two-sided p (t statistic, normal-CDF approximation).

    The normal approximation of the t distribution is comfortable at the
    sample sizes L1 produces (n >= 30 per prefill cache-mode group).
    """
    n = len(xs)
    if n < 4:
        return None
    rho = _pearson(_ranks(xs), _ranks(ys))
    if rho >= 1.0:
        return 1.0, 0.0
    if rho <= -1.0:
        return -1.0, 0.0
    t = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return rho, p


def _stats_block(errs_signed: list[float]) -> dict:
    abs_errs = [abs(e) for e in errs_signed]
    if not abs_errs:
        return {"n": 0}
    return {
        "n": len(errs_signed),
        "p50_abs": round(percentile(abs_errs, 50), 4),
        "p90_abs": round(percentile(abs_errs, 90), 4),
        "max_abs": round(max(abs_errs), 4),
        "median_signed": round(statistics.median(errs_signed), 4),
        "min_signed": round(min(errs_signed), 4),
        "max_signed": round(max(errs_signed), 4),
    }


# ---------------------------------------------------------------------------
# cell join + error computation


def _mock_value(cell_ref: dict, decode_caliber: str) -> tuple[float | None, str]:
    axis = cell_ref.get("axis")
    if axis == "prefill":
        return cell_ref.get("mock_prefill_ms"), "prefillMs(batch)"
    if axis == "decode":
        if decode_caliber == "first_excl":
            return cell_ref.get("mock_decode_ms_first_excl"), "decodeSteps(ol-1)xstepMs"
        return cell_ref.get("mock_decode_ms"), "decodeMs(ol,batch)"
    return None, "unknown-axis"


def _real_value(cell_real: dict, real_caliber: str) -> tuple[float | None, str]:
    engine = cell_real.get("engine_median_ms")
    client = cell_real.get("median_ms")
    if real_caliber == "engine" or (real_caliber == "auto" and engine is not None):
        return engine, cell_real.get("engine_metric", "engine_exec_ms")
    return client, cell_real.get("metric", "client_ms")


def build_cells(
    real_doc: dict, mock_doc: dict, real_caliber: str, decode_caliber: str
) -> tuple[list[dict], list[str]]:
    """Join real and mock cells on grid_id; returns (cells, warnings)."""
    warnings: list[str] = []
    mock_by_id = {c["grid_id"]: c for c in mock_doc.get("cells", [])}
    cells: list[dict] = []
    for cell_real in real_doc.get("cells", []):
        grid_id = cell_real["grid_id"]
        cell_ref = mock_by_id.get(grid_id)
        base = {
            "grid_id": grid_id,
            "axis": cell_real.get("axis"),
            "input_len": cell_real.get("input_len"),
            "output_len": cell_real.get("output_len"),
            "batch_size": cell_real.get("batch_size"),
            "cache_mode": cell_real.get("cache_mode"),
        }
        if cell_ref is None:
            base.update({"missing": True, "reason": "no mock reference cell"})
            warnings.append(f"{grid_id}: no mock reference cell")
            cells.append(base)
            continue
        mock_ms, mock_src = _mock_value(cell_ref, decode_caliber)
        real_ms, real_src = _real_value(cell_real, real_caliber)
        base["real_ms"] = real_ms
        base["mock_ms"] = mock_ms
        base["real_source"] = real_src
        base["mock_source"] = mock_src
        if mock_ms is None:
            base.update({"missing": True, "reason": "no mock value"})
            warnings.append(f"{grid_id}: no mock value ({mock_src})")
        elif real_ms is None or real_ms <= 0:
            base.update(
                {
                    "missing": True,
                    "reason": f"no real samples (n={cell_real.get('n_samples', 0)})",
                }
            )
            warnings.append(f"{grid_id}: no real samples")
        else:
            rel = (mock_ms - real_ms) / real_ms
            base["missing"] = False
            base["reason"] = None
            base["rel_err"] = round(rel, 4)
            base["abs_rel_err"] = round(abs(rel), 4)
        cells.append(base)
    for grid_id in mock_by_id:
        if grid_id not in {c["grid_id"] for c in real_doc.get("cells", [])}:
            warnings.append(f"{grid_id}: mock cell has no real counterpart")
    return cells, warnings


# ---------------------------------------------------------------------------
# terminal rendering


def _fmt_err(rel: float | None) -> str:
    if rel is None:
        return "  --  "
    shown = max(min(rel * 100.0, ERR_DISPLAY_CAP), -ERR_DISPLAY_CAP)
    return f"{shown:+7.1f}%"


def _fmt_ms(v: float | None) -> str:
    if v is None:
        return "    --"
    return f"{v:8.1f}"


def print_table(
    cells: list[dict], stats: dict, gates: dict, monotonic: list[dict]
) -> None:
    print("=" * 88)
    print(
        "L1 TIME-GRID ERROR TABLE  (mock formula vs real engine; "
        "err = (mock - real) / real)"
    )
    print("=" * 88)
    header = (
        f"{'grid_id':<22}{'axis':<8}{'il':>6}{'ol':>6}{'b':>4}"
        f"{'mode':<6}{'real_ms':>10}{'mock_ms':>10}{'err':>10}"
    )
    print(header)
    print("-" * 88)
    for cell in cells:
        axis = cell.get("axis") or "?"
        il = cell.get("input_len")
        ol = cell.get("output_len")
        b = cell.get("batch_size")
        mode = cell.get("cache_mode") or ""
        if cell.get("missing"):
            err_str = "MISSING"
            reason = cell.get("reason") or ""
            print(
                f"{cell['grid_id']:<22}{axis:<8}{il:>6}{ol:>6}{b:>4}"
                f"{mode:<6}{'--':>10}{'--':>10}{err_str:>10}  {reason}"
            )
            continue
        print(
            f"{cell['grid_id']:<22}{axis:<8}{il:>6}{ol:>6}{b:>4}"
            f"{mode:<6}{_fmt_ms(cell['real_ms']):>10}"
            f"{_fmt_ms(cell['mock_ms']):>10}"
            f"{_fmt_err(cell['rel_err']):>10}"
        )
    print("-" * 88)

    def block(name: str) -> None:
        s = stats.get(name)
        if not s or s.get("n", 0) == 0:
            print(f"{name:<16} no samples")
            return
        print(
            f"{name:<16} n={s['n']:<4} |err| P50={s['p50_abs']*100:6.1f}%  "
            f"P90={s['p90_abs']*100:6.1f}%  max={s['max_abs']*100:6.1f}%  "
            f"signed median={s['median_signed']*100:+7.1f}%"
        )

    block("overall")
    block("prefill")
    block("prefill_zero")
    block("prefill_warm")
    block("decode")

    print("-" * 88)
    print(
        "MONOTONIC BIAS (Spearman of signed err vs input_len, "
        "flag when |rho|>0.7 and p<0.05):"
    )
    if not monotonic:
        print("  (no prefill groups to test)")
    for m in monotonic:
        flag = "FLAGGED" if m["flagged"] else "ok"
        print(
            f"  {m['scope']:<14} rho={m['spearman_rho']:+.3f}  "
            f"p={m['p_value']:.4f}  n={m['n']}  -> {flag}"
            + (
                "  [structural bias: error grows with input_len]"
                if m["flagged"]
                else ""
            )
        )

    print("-" * 88)
    print(
        f"GATES (P50 <= {gates['params']['p50_max']*100:.0f}%, "
        f"P90 <= {gates['params']['p90_max']*100:.0f}%):"
    )
    for scope in ("overall", "prefill", "decode"):
        g = gates["per_scope"][scope]
        print(
            f"  {scope:<8} {'PASS' if g['passed'] else 'FAIL'}"
            f"  (P50={g['p50_abs']*100:.1f}%  P90={g['p90_abs']*100:.1f}%)"
            + ("" if g["reason"] is None else f"  [{g['reason']}]")
        )
    print(f"  VERDICT: {'PASS' if gates['passed'] else 'FAIL'}")


# ---------------------------------------------------------------------------
# HTML rendering (offline, inline CSS only)


def _heat_color(abs_err: float) -> str:
    if abs_err <= 0.10:
        return "#2e7d32"
    if abs_err <= 0.20:
        return "#f9a825"
    if abs_err <= 0.50:
        return "#ef6c00"
    return "#c62828"


def render_html(
    doc: dict, cells: list[dict], stats: dict, gates: dict, monotonic: list[dict]
) -> str:
    def esc(s) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parts: list[str] = []
    parts.append(
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        "<title>L1 time-grid error report</title><style>"
        "body{font-family:ui-monospace,Menlo,Consolas,monospace;"
        "margin:24px auto;max-width:1100px;color:#1b1b1b;background:#fafafa}"
        "h1{font-size:20px}h2{font-size:15px;margin-top:28px}"
        "table{border-collapse:collapse;font-size:12px;margin:8px 0}"
        "td,th{border:1px solid #bbb;padding:3px 7px;text-align:right}"
        "th{background:#333;color:#fff;text-align:center}"
        "td.key,th.key{text-align:left}"
        "td.cell{color:#fff;font-weight:600;text-align:center;min-width:64px}"
        ".meta{color:#555;font-size:12px;white-space:pre-wrap}"
        ".pass{color:#2e7d32;font-weight:700}.fail{color:#c62828;font-weight:700}"
        ".legend span{display:inline-block;width:14px;height:14px;"
        "vertical-align:-2px;margin:0 3px 0 12px}"
        "</style></head><body>"
    )
    parts.append("<h1>L1 time-grid error report</h1>")
    parts.append(
        f"<p class=\"meta\">generated: {esc(doc['created'])}\n"
        f"real: {esc(doc['inputs']['real'])} "
        f"(caliber: {esc(doc['calibers']['real'])})\n"
        f"mock: {esc(doc['inputs']['mock'])} "
        f"(decode caliber: {esc(doc['calibers']['decode_mock'])})</p>"
    )

    parts.append("<h2>Acceptance gates</h2><table>")
    parts.append(
        '<tr><th class="key">scope</th><th>P50 |err|</th>'
        "<th>P90 |err|</th><th>verdict</th></tr>"
    )
    for scope in ("overall", "prefill", "decode"):
        g = gates["per_scope"][scope]
        cls = "pass" if g["passed"] else "fail"
        parts.append(
            f'<tr><td class="key">{scope}</td>'
            f"<td>{g['p50_abs']*100:.1f}%</td>"
            f"<td>{g['p90_abs']*100:.1f}%</td>"
            f"<td class=\"{cls}\">{'PASS' if g['passed'] else 'FAIL'}"
            f"</td></tr>"
        )
    verdict = "pass" if gates["passed"] else "fail"
    parts.append(
        f'</table><p class="{verdict}">VERDICT: '
        f"{'PASS' if gates['passed'] else 'FAIL'}</p>"
    )

    parts.append(
        "<h2>Monotonic bias (Spearman vs input_len)</h2><table>"
        '<tr><th class="key">scope</th><th>rho</th><th>p</th>'
        "<th>n</th><th>flag</th></tr>"
    )
    for m in monotonic:
        cls = "fail" if m["flagged"] else "pass"
        parts.append(
            f"<tr><td class=\"key\">{esc(m['scope'])}</td>"
            f"<td>{m['spearman_rho']:+.3f}</td>"
            f"<td>{m['p_value']:.4f}</td><td>{m['n']}</td>"
            f"<td class=\"{cls}\">{'FLAGGED' if m['flagged'] else 'ok'}"
            f"</td></tr>"
        )
    parts.append("</table>")

    parts.append(
        "<h2>Heatmaps (signed err%, cell color = |err| band)</h2>"
        '<p class="legend">bands:'
        '<span style="background:#2e7d32"></span>&le;10%'
        '<span style="background:#f9a825"></span>10-20%'
        '<span style="background:#ef6c00"></span>20-50%'
        '<span style="background:#c62828"></span>&gt;50%</p>'
    )

    def heatmap(title: str, rows: list, cols: list, key_fn, row_label: str) -> None:
        parts.append(
            f'<h2>{esc(title)}</h2><table><tr><th class="key">' f"{esc(row_label)}</th>"
        )
        for c in cols:
            parts.append(f"<th>batch {c}</th>")
        parts.append("</tr>")
        by_pos: dict = {}
        for cell in cells:
            if cell.get("missing"):
                continue
            by_pos[(key_fn(cell), cell["batch_size"])] = cell
        for r in rows:
            parts.append(f'<tr><td class="key">{r}</td>')
            for c in cols:
                cell = by_pos.get((r, c))
                if cell is None:
                    parts.append(
                        '<td class="cell" ' 'style="background:#9e9e9e">--</td>'
                    )
                else:
                    color = _heat_color(cell["abs_rel_err"])
                    parts.append(
                        f'<td class="cell" style="background:'
                        f"{color}\">{cell['rel_err']*100:+.1f}%"
                        f"</td>"
                    )
            parts.append("</tr>")
        parts.append("</table>")

    pf = [c for c in cells if c.get("axis") == "prefill"]
    if pf:
        ils = sorted({c["input_len"] for c in pf})
        batches = sorted({c["batch_size"] for c in pf})
        for mode in ("zero", "warm"):
            mode_cells = [c for c in pf if c.get("cache_mode") == mode]
            if mode_cells:
                heatmap(
                    f"prefill / {mode} (rows input_len)",
                    ils,
                    batches,
                    lambda c: c["input_len"],
                    "input_len",
                )
    dec = [c for c in cells if c.get("axis") == "decode"]
    if dec:
        ols = sorted({c["output_len"] for c in dec})
        batches = sorted({c["batch_size"] for c in dec})
        heatmap(
            "decode (rows output_len)",
            ols,
            batches,
            lambda c: c["output_len"],
            "output_len",
        )

    parts.append(
        "<h2>Per-cell detail</h2><table>"
        '<tr><th class="key">grid_id</th><th>axis</th><th>il</th>'
        '<th>ol</th><th>b</th><th class="key">mode</th>'
        "<th>real_ms</th><th>mock_ms</th><th>err</th></tr>"
    )
    for cell in cells:
        if cell.get("missing"):
            parts.append(
                f"<tr><td class=\"key\">{esc(cell['grid_id'])}</td>"
                f'<td colspan="8">MISSING — '
                f"{esc(cell.get('reason'))}</td></tr>"
            )
            continue
        color = _heat_color(cell["abs_rel_err"])
        parts.append(
            f"<tr><td class=\"key\">{esc(cell['grid_id'])}</td>"
            f"<td>{esc(cell['axis'])}</td><td>{cell['input_len']}"
            f"</td><td>{cell['output_len']}</td>"
            f"<td>{cell['batch_size']}</td>"
            f"<td class=\"key\">{esc(cell['cache_mode'])}</td>"
            f"<td>{cell['real_ms']:.1f}</td>"
            f"<td>{cell['mock_ms']:.1f}</td>"
            f'<td style="color:{color};font-weight:600">'
            f"{cell['rel_err']*100:+.1f}%</td></tr>"
        )
    parts.append("</table></body></html>")
    return "".join(parts)


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="l1_error_report.py",
        description="L1 error table: mock timing formulas vs real engine "
        "medians, with acceptance gates and monotone-bias "
        "detection.",
    )
    parser.add_argument(
        "--real", required=True, help="l1_real_medians.json (grid runner extract)"
    )
    parser.add_argument(
        "--mock", required=True, help="l1_mock_reference.json (mock reference)"
    )
    parser.add_argument("--out", required=True, help="output JSON report path")
    parser.add_argument(
        "--html", default=None, help="optional HTML heatmap report path"
    )
    parser.add_argument(
        "--real-caliber",
        choices=["client", "engine", "auto"],
        default="auto",
        help="real-side median source (default auto)",
    )
    parser.add_argument(
        "--decode-caliber",
        choices=["budget", "first_excl"],
        default="budget",
        help="mock decode caliber (default budget)",
    )
    parser.add_argument(
        "--gate-p50",
        type=float,
        default=0.10,
        help="acceptance: P50(|err|) max (default 0.10)",
    )
    parser.add_argument(
        "--gate-p90",
        type=float,
        default=0.20,
        help="acceptance: P90(|err|) max (default 0.20)",
    )
    parser.add_argument(
        "--no-gate", action="store_true", help="exit 0 even when gates fail"
    )
    args = parser.parse_args(argv)

    real_doc = json.loads(Path(args.real).read_text())
    mock_doc = json.loads(Path(args.mock).read_text())
    cells, warnings = build_cells(
        real_doc, mock_doc, args.real_caliber, args.decode_caliber
    )
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    def errs_of(pred) -> list[float]:
        return [c["rel_err"] for c in cells if not c.get("missing") and pred(c)]

    stats = {
        "overall": _stats_block(errs_of(lambda c: True)),
        "prefill": _stats_block(errs_of(lambda c: c["axis"] == "prefill")),
        "prefill_zero": _stats_block(
            errs_of(lambda c: c["axis"] == "prefill" and c["cache_mode"] == "zero")
        ),
        "prefill_warm": _stats_block(
            errs_of(lambda c: c["axis"] == "prefill" and c["cache_mode"] == "warm")
        ),
        "decode": _stats_block(errs_of(lambda c: c["axis"] == "decode")),
    }

    monotonic: list[dict] = []
    for mode in ("zero", "warm"):
        group = [
            c
            for c in cells
            if not c.get("missing")
            and c["axis"] == "prefill"
            and c["cache_mode"] == mode
        ]
        if len(group) < 4:
            continue
        result = spearman(
            [float(c["input_len"]) for c in group], [c["rel_err"] for c in group]
        )
        if result is None:
            continue
        rho, p = result
        monotonic.append(
            {
                "scope": f"prefill_{mode}",
                "spearman_rho": round(rho, 4),
                "p_value": round(p, 4),
                "n": len(group),
                "flagged": bool(abs(rho) > 0.7 and p < 0.05),
            }
        )

    def gate(s: dict) -> dict:
        if s.get("n", 0) == 0:
            return {
                "passed": False,
                "p50_abs": float("nan"),
                "p90_abs": float("nan"),
                "reason": "no samples",
            }
        ok_p50 = s["p50_abs"] <= args.gate_p50
        ok_p90 = s["p90_abs"] <= args.gate_p90
        reason = None
        if not ok_p50:
            reason = f"P50 {s['p50_abs']*100:.1f}% > {args.gate_p50*100:.0f}%"
        if not ok_p90:
            add = f"P90 {s['p90_abs']*100:.1f}% > {args.gate_p90*100:.0f}%"
            reason = f"{reason} + {add}" if reason else add
        return {
            "passed": ok_p50 and ok_p90,
            "p50_abs": s["p50_abs"],
            "p90_abs": s["p90_abs"],
            "reason": reason,
        }

    gates = {
        "params": {"p50_max": args.gate_p50, "p90_max": args.gate_p90},
        "per_scope": {
            scope: gate(stats[scope]) for scope in ("overall", "prefill", "decode")
        },
    }
    gates["passed"] = all(g["passed"] for g in gates["per_scope"].values())

    doc = {
        "tool": "l1_error_report",
        "version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inputs": {"real": str(args.real), "mock": str(args.mock)},
        "calibers": {"real": args.real_caliber, "decode_mock": args.decode_caliber},
        "cells": cells,
        "stats": stats,
        "monotonic_bias": monotonic,
        "gates": gates,
        "warnings": warnings,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    if args.html:
        html_path = Path(args.html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(render_html(doc, cells, stats, gates, monotonic))

    print_table(cells, stats, gates, monotonic)
    print(f"\nreport written: {out_path}")
    if args.html:
        print(f"html written: {args.html}")

    if args.no_gate:
        return 0
    return 0 if gates["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
