#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_golden_v3.py — 从零整体重生成 chart-report-golden-v3.html（SLO 口径重构版）。

设计要点：
  * 复用 v2 排版体系（栅格/卡片/深色仪器风/宽表滚动容器/minmax 防撑破），v2 产物保留不动；
  * 叙事流：口径为什么换(P1'/X3/X2) → 新旧背离(X4) → 全矩阵(P2/P3/P5) → verdict(T3) → 门禁(T1/T2) → 对照(X1 双口径)；
  * 数据层 = slo-analysis.json（核心）+ /tmp/slo-golden staging（X2 直方图逐请求重算）
    + v2 的 matrix/distributions/candidates/v1-comparison/cause/online/deployments（P2/P3/P5/X1 对照）；
  * 全部核心数字（M_cal/三臂反转/相变反转/join 覆盖率/verdict 无 HARMFUL）断言后注入。
"""
import json
import os
import re
import sys
from datetime import datetime

SWEEP = "/Users/wangziyi/code/codex-cache-v5/guardrail-v0-outer/github-opensource/results/golden-sweep-20260911"
BAL = "/Users/wangziyi/code/codex-cache-balance-run/results"
OUTDIR = BAL + "/golden-sweep-20260911"
OUT = OUTDIR + "/chart-report-golden-v3.html"
STAGING = "/tmp/slo-golden"
GOLD_DATA = "/tmp/golden_data.json"    # v2 已知good report-data（P2/P3/P5/X1 交叉核对基准）
TEMPLATE = "/tmp/v3_template.html"
V3_JS = "/tmp/v3_main.js"


def approx(a, b, tol=1e-4):
    return abs(a - b) <= tol


def arrapprox(xs, ys, tol=1e-4):
    return len(xs) == len(ys) and all(abs(x - y) <= tol for x, y in zip(xs, ys))


# ==================== 载入源数据 ====================
SLO = json.load(open(SWEEP + "/slo-analysis.json"))
M = json.load(open(SWEEP + "/matrix.json"))
D = json.load(open(SWEEP + "/per-metric-distributions.json"))
CAND = json.load(open(SWEEP + "/golden-candidates.json"))
V1C = json.load(open(BAL + "/guardrail-v1-20260910/v1-comparison.json"))
V1 = V1C["hi"]
CI = json.load(open(BAL + "/cause-investigation/summary.json"))
OA = json.load(open(BAL + "/mechanism-v4/online-affinity-summary.json"))
DP = json.load(open(BAL + "/alignment-v3/deployments.json"))["response"]["data"]
GD = json.load(open(GOLD_DATA))

CAPS = [0, 50, 100, 250, 500, 1000, 100000000]
CAPKEY = ["0", "50", "100", "250", "500", "1000", "100000000"]
LABELS = ["0", "50", "100", "250", "500", "1000", "\u221e"]
MC = SLO["m_calibration"]
GS = SLO["golden_sweep"]
V1A = SLO["v1_arms"]
V0A = SLO["v0_arms"]
ONC = SLO["old_new_consistency"]

# ==================== X2：r 分布直方图（staging 逐请求重算）====================
rpool = []
for s in (0, 1, 2):
    tr = {}
    with open(os.path.join(STAGING, "trace_cap250_s%d.jsonl" % s)) as f:
        for line in f:
            o = json.loads(line)
            tr[o["rid"]] = o["prefill_ms"]
    with open(os.path.join(STAGING, "ttft_cap250_s%d.jsonl" % s)) as f:
        for line in f:
            o = json.loads(line)
            rpool.append(o["ttft_ms"] / tr[o["rid"]])

BW = 0.05
lo = int(min(rpool) / BW) * BW
hi = (int(max(rpool) / BW) + 1) * BW
nb = int(round((hi - lo) / BW))
counts = [0] * nb
for r in rpool:
    counts[min(int((r - lo) / BW), nb - 1)] += 1
hist_centers = [round(lo + (i + 0.5) * BW, 4) for i in range(nb)]


def pctl(sorted_xs, q):
    k = (len(sorted_xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_xs) - 1)
    return sorted_xs[f] + (sorted_xs[c] - sorted_xs[f]) * (k - f)


rs = sorted(rpool)
p95_re, p99_re = pctl(rs, 0.95), pctl(rs, 0.99)

# 固定档 M 在黄金点的违约率（staging 重算）
fix_x, fix_y = [], []
for m, lab in [(1.5, "M1.5"), (2.0, "M2.0"), (MC["M_cal_p95"], "Mcal"), (3.0, "M3.0")]:
    v = sum(1 for r in rpool if r > m) / len(rpool)
    fix_x.append(round(m, 4))
    fix_y.append(round(v * 100, 2))
    if lab == "Mcal":
        mc_viol_re = v

# ==================== v2 复用数据（P2/P3/P5/c4 移除，X1 tri）====================
kv = lambda c: D[c]["kv_pool"]["prefill-0.eviction_age_avg_s"]
harmMean = [round(D[c]["hotspot_gt_threshold_ratio"]["mean"], 6) for c in CAPKEY]
bgMean = [round(D[c]["bg_hit_rate_avg"]["mean"], 6) for c in CAPKEY]
hotMean = [round(D[c]["hot_hit_rate"]["mean"], 6) for c in CAPKEY]
hotStd = [round(D[c]["hot_hit_rate"]["std"], 6) for c in CAPKEY]
ttftMean = [round(D[c]["sustain_ttft_p95_s"]["mean"], 6) for c in CAPKEY]
ttftStd = [round(D[c]["sustain_ttft_p95_s"]["std"], 6) for c in CAPKEY]
evMean = [round(kv(c)["mean"], 6) for c in CAPKEY]
evStd = [round(kv(c)["std"], 6) for c in CAPKEY]


def seedmat(field):
    out = [[0.0] * 7 for _ in range(3)]
    for c in M:
        out[c["seed"]][CAPS.index(c["cap"])] = c[field]
    return out


hotSeed = seedmat("hot_hit_rate")
ttftSeed = seedmat("sustain_ttft_p95_s")
_ev = [[0.0] * 7 for _ in range(3)]
for c in M:
    _ev[c["seed"]][CAPS.index(c["cap"])] = c["kv_pool_engines"]["prefill-0"]["eviction_age_avg_s"]
evSeed = [[round(v, 2) for v in row] for row in _ev]
base = round(D["250"]["sustain_ttft_p95_s"]["mean"], 6)
h1Band = {"lo": round(base * 0.99, 8), "hi": round(base * 1.01, 8), "base": base}
evBand = {"lo": 200, "hi": 300}

# ==================== SLO 核心数据 ====================
bycap = {b["cap"]: b for b in GS["by_cap"]}
base_bl = GS["baseline_cap250"]
BASE_P95 = base_bl["slo_viol_by_seed"]["M_cal_p95"]["p95"]
BL_TPS = base_bl["sustain_input_tps"]
BL_HOT = base_bl["hot_hit_rate"]
BL_BG = base_bl["bg_hit_rate_avg"]

# golden 7 档（P1' + X4 + T3）
old_pct = [round(ONC["golden_by_cap"][i]["old_harmful_delta_gt100"] * 100, 4) for i in range(7)]
new_pct = [round(bycap[c]["slo_viol"]["M_cal_p95"] * 100, 4) for c in CAPS]
new99_pct = [round(bycap[c]["slo_viol"]["M_cal_p99"] * 100, 4) for c in CAPS]
attr_pct = [round(bycap[c]["attr_viol"]["M_cal_p95"] * 100, 4) for c in CAPS]
ttft95 = [round(bycap[c]["ttft_ms_p95"], 2) for c in CAPS]
tpsR = [round(bycap[c]["input_tps"] / BL_TPS, 4) for c in CAPS]
hotR = [round(bycap[c]["hot_hit_rate"] / BL_HOT, 4) for c in CAPS]
bgR = [round(bycap[c]["bg_hit_rate_avg"] / BL_BG, 4) for c in CAPS]

# v1 三臂（X3 + X4 + T3）—— 顺序 hi/finite/zero（反转叙事序）
ARMS = ["hi", "finite", "zero"]
v1 = {}
for a in ARMS:
    s = V1A[a]["slo"]
    v1[a] = {
        "oldPct": round(s["old_harmful_delta_gt100"] * 100, 4),
        "newPct": round(s["slo_viol"]["M_cal_p95"] * 100, 4),
        "attrPct": round(s["attr_viol"]["M_cal_p95"] * 100, 4),
        "attr2Pct": round(s["attr_viol_ratio_cf"]["M_cal_p95"] * 100, 4),
        "ttft": round(s["ttft_ms"]["p95"], 1),
        "tpsR": round(V1A[a]["ratios_vs_zero"]["input_tps"], 5),
        "hotR": round(V1A[a]["ratios_vs_zero"]["hot_hit"], 5),
        "bgR": round(V1C[a]["hotspot_hit_bg"] / V1C["zero"]["hotspot_hit_bg"], 4),
    }

# 相变（P1'）
PT = ONC["phase_transition_250_to_500"]
ph = {
    "oldLo": round(PT["old_delta_gt100"]["cap250"] * 100, 2),
    "oldHi": round(PT["old_delta_gt100"]["cap500"] * 100, 2),
    "newLo": round(PT["new_slo_viol_M_cal_p95"]["cap250"] * 100, 2),
    "newHi": round(PT["new_slo_viol_M_cal_p95"]["cap500"] * 100, 2),
}

# X1 tri（v2 值 + 新口径扩展）
G = next(c for c in M if c["cap"] == 250 and c["seed"] == 0)
INF = [c for c in M if c["cap"] == 100000000]
v1_harm = V1["hotspot_delta"]["gt_100_ratio"]
inf_harm = sum(c["hotspot_gt_threshold_ratio"] for c in INF) / 3
tri = dict(GD["tri"])
tri.update({
    "v1NewSloPct": v1["hi"]["newPct"],
    "sweepInfNewSloPct": new_pct[6],
    "goldenNewSloPct": new_pct[3],
    "sloInfOldPct": old_pct[6],
})

DATA = {
    "caps": CAPS, "labels": LABELS,
    "harmMean": harmMean, "bgMean": bgMean,
    "hotMean": hotMean, "hotStd": hotStd,
    "ttftMean": ttftMean, "ttftStd": ttftStd,
    "evMean": evMean, "evStd": evStd,
    "hotSeed": hotSeed, "ttftSeed": ttftSeed, "evSeed": evSeed,
    "h1Band": h1Band, "evBand": evBand, "tri": tri,
    "oldPct": old_pct, "newPct": new_pct, "new99Pct": new99_pct,
    "attrPct": attr_pct, "ttft95": ttft95,
    "hist": {"centers": hist_centers, "counts": counts, "bw": BW},
    "mcal": {"p95": MC["M_cal_p95"], "p99": MC["M_cal_p99"],
             "mp95": MC["Mp_cal_p95"], "n": MC["n_pooled"]},
    "fixM": {"x": fix_x, "y": fix_y},
    "v1arms": {"arms": ARMS, "v": [v1[a] for a in ARMS]},
    "phase": ph,
}

# ==================== 断言 1：SLO 核心数字（用户口径逐项对账）====================
A = [
    ("M_cal P95 = 2.6397", approx(MC["M_cal_p95"], 2.6397, 1e-9)),
    ("M_cal P99 = 2.6528", approx(MC["M_cal_p99"], 2.6528, 1e-9)),
    ("M'_cal P95 = 1.0398", approx(MC["Mp_cal_p95"], 1.0398, 1e-9)),
    ("pooled n = 8100", MC["n_pooled"] == 8100),
    ("固定档 M1.5 误判 50.04%", approx(base_bl["slo_viol_by_seed"]["M1.5"]["p95"], 0.50037, 1e-6)),
    ("固定档 M2.0 误判 50.04%", approx(base_bl["slo_viol_by_seed"]["M2.0"]["p95"], 0.50037, 1e-6)),
    ("固定档 M3.0 全漏 0%", base_bl["slo_viol_by_seed"]["M3.0"]["p95"] == 0.0),
    ("v1 hi 旧有害 18.76%", approx(v1["hi"]["oldPct"], 18.7554, 1e-3)),
    ("v1 hi 新违约 29.88%", approx(v1["hi"]["newPct"], 29.8829, 1e-3)),
    ("v1 hi 归因违约 12.77%", approx(v1["hi"]["attrPct"], 12.7665, 1e-3)),
    ("v1 hi TTFT p95 682ms", approx(v1["hi"]["ttft"], 681.6, 0.05)),
    ("v1 finite 违约 59.81%", approx(v1["finite"]["newPct"], 59.809, 1e-3)),
    ("v1 finite TTFT 958ms", approx(v1["finite"]["ttft"], 958.39, 0.05)),
    ("v1 zero 旧 0% 新 64.98%", v1["zero"]["oldPct"] == 0.0 and approx(v1["zero"]["newPct"], 64.9784, 1e-3)),
    ("v1 zero TTFT 874ms", approx(v1["zero"]["ttft"], 874.51, 0.05)),
    ("反转：新违约 hi 三臂最少", v1["hi"]["newPct"] < v1["finite"]["newPct"] < v1["zero"]["newPct"]),
    ("反转：TTFT hi 三臂最低", v1["hi"]["ttft"] < v1["zero"]["ttft"] < v1["finite"]["ttft"]),
    ("反转：旧口径 hi 最有害 zero 最干净", v1["hi"]["oldPct"] > v1["finite"]["oldPct"] > v1["zero"]["oldPct"] == 0.0),
    ("v1 吞吐/命中三臂几乎相同", all(abs(v1[a]["tpsR"] - 1) < 0.002 and abs(v1[a]["hotR"] - 1) < 0.002 for a in ARMS)),
    ("相变旧 0→33.3%", ph["oldLo"] == 0.0 and approx(ph["oldHi"], 33.2963, 1e-2)),
    ("相变新 5.06→0.85%", approx(ph["newLo"], 5.0617, 5.1e-3) and approx(ph["newHi"], 0.8519, 5.1e-3)),
    ("相变 TTFT 持平", approx(PT["ttft_ms_p95"]["cap250"], 579.31, 0.01) and approx(PT["ttft_ms_p95"]["cap500"], 577.85, 0.01)),
    ("低压 TTFT 全档 ~578ms", all(577 < t < 580 for t in ttft95)),
    ("cap≤250 违约 ~5%", all(4 < p < 6 for p in new_pct[:4])),
    ("cap≥500 违约 ~1%", all(p < 1.3 for p in new_pct[4:])),
    ("verdict 无一格 HARMFUL", all(c["verdict"]["M_cal_p95"]["harmful_verdict"] is False for c in GS["cells"])),
    ("by_cap harmful_seeds 全 0", all(b["verdict_M_cal_p95"]["harmful_seeds"] == 0 for b in GS["by_cap"])),
    ("join golden 56700/56700 · 21 格全量", SLO["meta"]["join_coverage_summary"]["golden_total_joined"] == 56700
        and SLO["meta"]["join_coverage_summary"]["golden_expected"] == 56700
        and SLO["meta"]["join_coverage_summary"]["golden_cells_full"] == 21),
    ("join v1 每臂 16230/16230", all(V1A[a]["join_coverage"]["joined"] == V1A[a]["join_coverage"]["trace_rows"] == 16230
        and V1A[a]["join_coverage"]["coverage_ratio"] == 1.0 for a in ARMS)),
    ("join v0 每臂 16230", all(V0A[a]["join_coverage"]["joined"] == 16230 for a in ARMS)),
    ("consistency.v1_paradox == v1_arms", all(
        approx(p["old_harmful_delta_gt100"], V1A[p["arm"]]["slo"]["old_harmful_delta_gt100"], 1e-9)
        and approx(p["new_slo_viol_M_cal_p95"], V1A[p["arm"]]["slo"]["slo_viol"]["M_cal_p95"], 1e-9)
        and approx(p["client_ttft_p95_ms"], V1A[p["arm"]]["slo"]["ttft_ms"]["p95"], 1e-9)
        for p in ONC["v1_paradox"])),
    ("consistency.golden_by_cap == by_cap", all(
        approx(g["new_slo_viol_M_cal_p95"], bycap[g["cap"]]["slo_viol"]["M_cal_p95"], 1e-9)
        for g in ONC["golden_by_cap"])),
    ("相变值 == by_cap 250/500", approx(ph["newLo"] / 100, bycap[250]["slo_viol"]["M_cal_p95"], 1e-4)
        and approx(ph["newHi"] / 100, bycap[500]["slo_viol"]["M_cal_p95"], 1e-4)),
    ("hit 升 1.17×（250→500）", approx(bycap[500]["hot_hit_rate"] / bycap[250]["hot_hit_rate"], 1.1717, 1e-3)),
    ("hi 归因两口径接近 12.77/12.92", approx(v1["hi"]["attr2Pct"], 12.9205, 1e-3)),
    ("v0 hi 对照 18.52%/644ms", approx(V0A["hi"]["slo"]["slo_viol"]["M_cal_p95"], 0.185213, 1e-6)
        and approx(V0A["hi"]["slo"]["ttft_ms"]["p95"], 643.9971, 1e-3)),
    # X2 直方图 staging 重算自洽
    ("直方图重算 n=8100", sum(counts) == 8100 == len(rpool)),
    ("重算 P95 ≈ 2.6397（±0.002）", approx(p95_re, MC["M_cal_p95"], 2e-3)),
    ("重算 P99 ≈ 2.6528（±0.002）", approx(p99_re, MC["M_cal_p99"], 2e-3)),
    ("重算 M_cal 违约 ≈ 5%（tautology）", approx(mc_viol_re, 0.05, 6e-3)),
    ("重算 M1.5 违约 == JSON 50.04%", approx(fix_y[0] / 100, base_bl["slo_viol_by_seed"]["M1.5"]["p95"], 2e-3)),
    ("重算 M3.0 违约 == JSON 0%", approx(fix_y[3], 0.0, 0.01)),
    # v2 交叉核对（P2/P3/P5/tri 数据未变）
    ("harmMean == v2", arrapprox(harmMean, GD["harmMean"], 1e-9)),
    ("bgMean == v2", arrapprox(bgMean, GD["bgMean"], 1e-9)),
    ("hotSeed == v2", all(arrapprox(hotSeed[i], GD["hotSeed"][i], 1e-9) for i in range(3))),
    ("ttftSeed == v2", all(arrapprox(ttftSeed[i], GD["ttftSeed"][i], 1e-9) for i in range(3))),
    ("evSeed == v2", all(arrapprox(evSeed[i], GD["evSeed"][i], 1e-2) for i in range(3))),
    ("h1Band == v2", approx(h1Band["lo"], GD["h1Band"]["lo"], 1e-9) and approx(h1Band["hi"], GD["h1Band"]["hi"], 1e-9)),
    ("tri.v1HarmPct == 20.85%（旧口径另一分母）", approx(tri["v1HarmPct"], 20.85, 0.01)),
    ("slo 口径 inf 旧 33.30% ≈ matrix 33.32%", approx(old_pct[6], inf_harm * 100, 0.05)),
    ("X1 新口径 hi = v1_arms", approx(tri["v1NewSloPct"], v1["hi"]["newPct"], 1e-9)),
]
bad = [n for n, ok in A if not ok]
if bad:
    print("ASSERT FAIL:", bad)
    sys.exit(1)
print("[OK] 数据对账断言全部通过：%d 项（SLO 核心 + staging 重算 + v2 交叉）" % len(A))

# ==================== 呈现数值 ====================
p2 = lambda v: ("%.2f" % v).rstrip("0").rstrip(".")
fmtp = lambda v: p2(v) + "%"


def caplab(c):
    return "∞（1e8）" if c == 100000000 else str(c)


# ---- T3 verdict 表 ----
def vtag(labels):
    n = len(labels)
    nok = labels.count("OK")
    ncomp = labels.count("SLO-exceeded-but-compensated")
    nharm = labels.count("HARMFUL")
    assert nok + ncomp + nharm == n
    if nharm:
        return '<span class="tag hard">HARMFUL ×%d</span>' % nharm
    if ncomp == n:
        return '<span class="tag comp">compensated ×%d</span>' % ncomp
    if ncomp == 0:
        return '<span class="tag okc">OK ×%d</span>' % n
    return '<span class="tag okc">OK ×%d</span> <span class="tag comp">comp ×%d</span>' % (nok, ncomp)


def t3row(name, sub, viol, attr, tpsr, hotr, bgr, verdict_html, cls=""):
    return ('<tr%s><td class="mono">%s<span class="vsub">%s</span></td>'
            '<td class="mono">%s</td><td class="mono">%s</td>'
            '<td class="mono dim">%s</td><td class="mono dim">%s</td><td class="mono dim">%s</td>'
            '<td>%s</td></tr>' % (
                (' class="%s"' % cls) if cls else "",
                name, sub, viol, attr, tpsr, hotr, bgr, verdict_html))


T3 = ['<tr class="sechead"><td colspan="7">GOLDEN 21 格 · 3 QPS 低压 · 比值 vs cap250 黄金点基线（tps %.0f · hot %.4f · bg %.4f）· 违约门限 = 基线 seed P95 %s</td></tr>'
      % (BL_TPS, BL_HOT, BL_BG, fmtp(BASE_P95 * 100))]
for i, c in enumerate(CAPS):
    b = bycap[c]
    labels = [cell["verdict"]["M_cal_p95"]["label"] for cell in GS["cells"] if cell["cap"] == c]
    sub = "TTFT p95 %.0fms · 3 seed" % ttft95[i]
    if c == 250:
        sub += " · 基线（M_cal 校准点 · 违约≈5% 为 P95 tautology）"
    T3.append(t3row(
        "cap " + caplab(c), sub,
        fmtp(new_pct[i]) + '<span class="vsub">P99 口径 ' + fmtp(new99_pct[i]) + "</span>",
        fmtp(attr_pct[i]),
        "%.4f" % tpsR[i], "%.4f" % hotR[i], "%.4f" % bgR[i],
        vtag(labels)))
T3.append('<tr class="sechead"><td colspan="7">V1 三臂 · 154 QPS 高压 · 比值 vs zero 臂（regime 内对照 · 绝对违约排序 hi &lt; finite &lt; zero）· 违约门限 = zero 臂 64.98%（超基线定义不适用 → 无臂 HARMFUL）</td></tr>')
for a, cname in [("hi", "hi（cap=1e8）"), ("finite", "finite（cap=500）"), ("zero", "zero（cap=0 · 基线）")]:
    v = v1[a]
    comp = []
    if v["tpsR"] >= 1:
        comp.append("吞吐")
    if v["hotR"] >= 1:
        comp.append("hot")
    if v["bgR"] >= 1:
        comp.append("bg")
    vd = ('<span class="tag comp">compensated</span><span class="vsub">by ' + "/".join(comp) + "</span>") if a != "zero" \
        else '<span class="tag okc">OK</span><span class="vsub">基线臂</span>'
    T3.append(t3row(
        cname, "TTFT p95 %.0fms · 旧口径 %s · n=16230" % (v["ttft"], fmtp(v["oldPct"])),
        fmtp(v["newPct"]) + '<span class="vsub">归因 ' + fmtp(v["attrPct"]) + "</span>",
        fmtp(v["attrPct"]),
        "%.5f" % v["tpsR"], "%.5f" % v["hotR"], "%.4f" % v["bgR"],
        vd))
T3ROWS = "\n".join(T3)

# ---- T1 门禁表（S1 更新为 SLO 口径）----
s1_row = ('<tr><td class="mono">S1</td><td>有害判定（SLO 口径）</td><td><span class="tag soft">软</span></td>'
          '<td class="mono">违约率 &gt; 基线 seed P95（%s%%）∧ 无吞吐/命中补偿；归因违约盯梢</td>'
          '<td class="mono dim">旧「有害占比 ≤2%%（delta&gt;100ms）」废弃：高压判反（hi 旧 %s 最有害 → 新违约 %s 三臂最少）、低压相变伪影；'
          '21 格无一格 HARMFUL；低压 cap≤250 违约 ~5%% 为 M_cal tautology 仅盯相对变化；'
          'M 按 regime 黄金点数据校准（P95=%s），固定档 1.5/2.0/3.0 误判 50%%/50%%/全漏</td></tr>'
          ) % (fmtp(BASE_P95 * 100), fmtp(v1["hi"]["oldPct"]), fmtp(v1["hi"]["newPct"]), "%.4f" % MC["M_cal_p95"])
T1ROWS = "\n".join([
    '<tr><td class="mono">H1</td><td>TTFT p95</td><td><span class="tag hard">硬</span></td><td class="mono">±1%（0.5730–0.5845 s）</td><td class="mono dim">7 档 seed CV ≤ 0.17%（黄金点档基准 0.5787 s）</td></tr>',
    '<tr><td class="mono">H2</td><td>sustain 吞吐</td><td><span class="tag hard">硬</span></td><td class="mono">±4σ</td><td class="mono dim">7 档 input_tps seed CV ≤ 0.031%，σ ≪ 容差</td></tr>',
    s1_row,
    '<tr><td class="mono">S5</td><td>驱逐年龄（prefill-0）</td><td><span class="tag soft">软</span></td><td class="mono">200–300 s 带内</td><td class="mono dim">cap≥500 稳定 251.4 s 带内；cap≤250 波动 190–507 s 部分出带</td></tr>',
    '<tr><td class="mono">INVALID</td><td>拒绝数</td><td><span class="tag hard">硬</span></td><td class="mono">= 0</td><td class="mono dim">21 格 rejections 全 0，issued 56700/56700</td></tr>',
])

# ---- X1 三方对照行（复用 v2，有害行改双口径）----
f_infharm = "%.1f%%" % (inf_harm * 100)
R101 = CI["10.43.8.101"]["effective_reuse_from_wall_ratio"]["ratio_of_means"]
R171 = CI["10.43.10.171"]["effective_reuse_from_wall_ratio"]["ratio_of_means"]
R204 = CI["10.43.8.204"]["effective_reuse_from_wall_ratio"]["ratio_of_means"]
R155 = CI["10.43.12.155"]["effective_reuse_from_wall_ratio"]["ratio_of_means"]
L101 = OA["10.43.8.101"]["fractions"]["CACHE_LEADER"]
L171 = OA["10.43.10.171"]["fractions"]["CACHE_LEADER"]
mh = lambda host, mid: CI[host]["metrics"][mid]["mean"]
V8_TPS_IN = mh("10.43.8.101", "rtp_llm_context_wall_tps_with_cache") / mh("10.43.10.171", "rtp_llm_context_wall_tps_with_cache")
V8_TPS_CA = mh("10.43.8.101", "rtp_llm_context_tps") / mh("10.43.10.171", "rtp_llm_context_tps")
OS_TPS_IN = mh("10.43.8.204", "rtp_llm_context_wall_tps_with_cache") / mh("10.43.12.155", "rtp_llm_context_wall_tps_with_cache")
OS_TPS_CA = mh("10.43.8.204", "rtp_llm_context_tps") / mh("10.43.12.155", "rtp_llm_context_tps")
CAP250 = [c for c in M if c["cap"] == 250]
ev250 = sorted(c["kv_pool_engines"]["prefill-0"]["eviction_age_avg_s"] for c in CAP250)
bg250 = sorted(c["bg_hit_rate_avg"] for c in CAP250)
hot250 = sorted(c["hot_hit_rate"] for c in CAP250)
g_lead, g_issued = G["reasons"]["CACHE_LEADER"], G["issued"]


def td(*groups):
    parts = []
    for main, sub, ref in groups:
        h = '<span class="vmain">' + main + "</span>"
        if sub:
            h += '<span class="vsub">' + sub + "</span>"
        if ref:
            h += '<span class="srcref">[' + ref + "]</span>"
        parts.append(h)
    return "<td>" + "".join(parts) + "</td>"


def trow(metric, v8g, osg, gdg):
    return ('            <tr><td class="dim">' + metric + "</td>"
            + td(*v8g) + td(*osg) + td(*gdg) + "</tr>")


ROWS = "\n".join([
    trow("cap（maxExtraTtftMs）",
         [("1e8（无限让步）", "ESTIMATED_TTFT · minPrefixHitPercent=5", "部署")],
         [("无此概念", "非同类算法 · 无缓存亲和预算", "部署")],
         [("250", None, "摸测")]),
    trow("有害判定（旧 delta&gt;100ms / 新 SLO 违约）",
         [("20.85% → 29.88%", "154QPS 高压 v1-hi · 新 = SLO 违约（三臂最少）· 归因违约 12.77%", "v1-hi·SLO"),
          (f_infharm + " → " + fmtp(new_pct[6]), "3QPS 低压摸测 cap=1e8 三格 · 新为校准 tautology 邻域", "摸测·SLO")],
         [("N/A", "无该算法路径", None)],
         [("0% → " + fmtp(new_pct[3]), "cap≤250 旧全格 0% · 新 ≈5% 为 M_cal P95 tautology", "摸测·SLO")]),
    trow("客户端 TTFT p95（SLO 违约判定口径）",
         [("682 ms", "v1-hi 高压三臂最低（zero 874 · finite 958）", "SLO")],
         [("—", "线上无逐请求口径", None)],
         [("579 ms", "全档 577.5–579.6 平坦 · 无判别力", "摸测·SLO")]),
    trow("hot 命中率",
         [("0.750 / 0.683", "高压 v1-hi / 低压摸测 cap=1e8", "v1-hi·摸测")],
         [("—", "线上无逐请求命中口径", None)],
         [("0.683", "3 seed %.3f–%.3f · seed 敏感" % (hot250[0], hot250[-1]), "摸测")]),
    trow("bg 命中率形态",
         [("0.327", "bg 命中暴跌形态（高压）", "v1-hi")],
         [("—", "线上无逐请求命中口径", None)],
         [("0.494", "3 seed %.3f–%.3f · 无暴跌" % (bg250[0], bg250[-1]), "摸测")]),
    trow("有效复用占比（同口径 wall TPS）",
         [("%.1f%% / %.1f%%" % (R101 * 100, R171 * 100), "101 / 171 两机 · 76min 窗", "线上观测")],
         [("%.1f%% / %.1f%%" % (R204 * 100, R155 * 100), "204 / 155 两机接近", "线上观测")],
         [("N/A", "mock 无 wall TPS 口径", None)]),
    trow("路由分支 CACHE_LEADER 占比",
         [("%.1f%% / %.1f%%" % (L101 * 100, L171 * 100), "101 / 171 两机", "线上观测")],
         [("—", "SHORTEST_TTFT 无此分支", None)],
         [("%.1f%%" % (g_lead / g_issued * 100), "2697 / 2700 请求", "摸测")]),
    trow("跨机吞吐比（输入 / 计算 TPS）",
         [("%.2f× / %.2f×" % (V8_TPS_IN, V8_TPS_CA), "101 vs 171 · 悬殊", "线上观测")],
         [("%.2f× / %.2f×" % (OS_TPS_IN, OS_TPS_CA), "204 vs 155 · 接近", "线上观测")],
         [("N/A", "无双机对照（单 mock 集群）", None)]),
    trow("驱逐年龄 ev_age（prefill-0）",
         [("—", "线上无此口径", None)],
         [("—", None, None)],
         [("%.0f–%.0f s" % (ev250[0], ev250[-1]), "3 seed 波动 · 部分出带（200–300s）· mean %.1fs" % (sum(ev250) / 3), "摸测")]),
])

# ==================== 模板 token ====================
JOING = SLO["meta"]["join_coverage_summary"]["golden_total_joined"]
TOKENS = {
    "@@MCAL@@": "%.4f" % MC["M_cal_p95"],
    "@@MCAL99@@": "%.4f" % MC["M_cal_p99"],
    "@@MPcal@@": "%.4f" % MC["Mp_cal_p95"],
    "@@NPOOLED@@": str(MC["n_pooled"]),
    "@@M15@@": fmtp(base_bl["slo_viol_by_seed"]["M1.5"]["p95"] * 100),
    "@@M20@@": fmtp(base_bl["slo_viol_by_seed"]["M2.0"]["p95"] * 100),
    "@@M30@@": fmtp(0.0),
    "@@V1HIOLD@@": fmtp(v1["hi"]["oldPct"]),
    "@@V1HINEW@@": fmtp(v1["hi"]["newPct"]),
    "@@V1HIATTR@@": fmtp(v1["hi"]["attrPct"]),
    "@@V1HIATTR2@@": fmtp(v1["hi"]["attr2Pct"]),
    "@@V1HITTFT@@": "%.0f" % v1["hi"]["ttft"],
    "@@V1FINNEW@@": fmtp(v1["finite"]["newPct"]),
    "@@V1ZERONEW@@": fmtp(v1["zero"]["newPct"]),
    "@@V1HITPS@@": "%.5f" % v1["hi"]["tpsR"],
    "@@PHOLDLO@@": fmtp(ph["oldLo"]), "@@PHOLDHI@@": fmtp(ph["oldHi"]),
    "@@PHNEWLO@@": fmtp(ph["newLo"]), "@@PHNEWHI@@": fmtp(ph["newHi"]),
    "@@BASEP95@@": fmtp(BASE_P95 * 100),
    "@@TTFTFLAT@@": "578",
    "@@JOINGOLDEN@@": str(JOING),
    "@@JOINV1@@": "16230",
    "@@NOCOUNT@@": "21",
    "@@X1V1OLD@@": "20.85%", "@@X1V1NEW@@": fmtp(v1["hi"]["newPct"]),
    "@@X1INFOLD@@": f_infharm, "@@X1INFNEW@@": fmtp(new_pct[6]),
    "@@T3ROWS@@": T3ROWS, "@@T1ROWS@@": T1ROWS, "@@ROWS@@": ROWS,
    "@@DATA@@": json.dumps(DATA, ensure_ascii=False, separators=(",", ":")),
    "@@JS@@": open(V3_JS).read().strip("\n"),
    "@@TS@@": datetime.now().strftime("%Y-%m-%d %H:%M"),
}

HTML = open(TEMPLATE).read()
for k, v in TOKENS.items():
    HTML = HTML.replace(k, v)
assert "@@" not in HTML, "存在未替换占位符: %s" % re.findall(r"@@\w+@@", HTML)[:5]

with open(OUT, "w") as f:
    f.write(HTML)

# ==================== 写回自检 ====================
o = open(OUT).read()
CHECKS = [
    ('<canvas id="c11"', 1), ('<canvas id="c8"', 1), ('<canvas id="c9"', 1),
    ('<canvas id="c7"', 1), ('<canvas id="c10"', 1), ('<canvas id="c6"', 1),
    ('<canvas id="c2"', 1), ('<canvas id="c3"', 1), ('<canvas id="c5"', 1),
    ('id="report-data"', 1), ('class="x1grid"', 1), ('class="subgrid"', 1),
    ('class="tblwrap"', 4), ('minmax(0,', 2),
    ('2.6397', 2), ('29.88%', 3), ('64.98%', 2), ('18.76%', 2),
    ('0.85%', 1), ('5.06%', 1), ('33.3%', 2), ('20.85%', 2),
    ('T3', 1), ('compensated', 3), ('HARMFUL', 2), ('12.77%', 2),
    ('slo-analysis.json', 1), ('RUN-slo.md', 1), ('/tmp/slo-golden', 1),
    ('P1&prime;', 1), ('X2', 1), ('X3', 1), ('X4', 1),
]
for key, want in CHECKS:
    got = o.count(key)
    assert got >= want, "check %r: want>=%d got=%d" % (key, want, got)
json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>', o, re.S).group(1))
print("[OK] 写回完成：%s（%d bytes）" % (OUT, len(o)))
print("[OK] %d 项 HTML 结构自检 + report-data JSON 有效性校验通过" % len(CHECKS))
