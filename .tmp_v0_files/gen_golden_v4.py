#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_golden_v4.py — 从零整体重生成 chart-report-golden-v4.html（纯 SLO 违约指标体系版）。

设计要点：
  * 只呈现新指标（SLO 违约率）的结论 —— 无历史叙事、无双口径对照（用户明确修正）；
  * 复用既有排版体系（栅格/卡片/深色仪器风/宽表滚动容器/minmax 防撑破），既有版本报告保留不动；
  * 叙事流：指标定义(S1) → 高压结果(S2) → 全矩阵(S3) → verdict(S4) → 场景健康(S5) → 门禁(T1/T2) → 对照(X1)；
  * 数据层 = slo-analysis.json（主）+ matrix.json（S5/X1 摸测列）+ v1-comparison.json（bg 比）
    + /tmp/slo-golden staging（S1 直方图逐请求重算）+ cause/online/deployments（X1 线上列）；
  * report-data JSON 不含任何历史口径字段（生成后跑禁词扫描零命中作为验收项）。
"""
import json
import os
import re
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = "/Users/wangziyi/code/codex-cache-v5/guardrail-v0-outer/github-opensource/results/golden-sweep-20260911"
BAL = "/Users/wangziyi/code/codex-cache-balance-run/results"
OUTDIR = BAL + "/golden-sweep-20260911"
OUT = OUTDIR + "/chart-report-golden-v4.html"
STAGING = "/tmp/slo-golden"
TEMPLATE = os.path.join(HERE, "golden_v4_template.html")
V4_JS = os.path.join(HERE, "golden_v4_main.js")


def approx(a, b, tol=1e-4):
    return abs(a - b) <= tol


def arrapprox(xs, ys, tol=1e-4):
    return len(xs) == len(ys) and all(abs(x - y) <= tol for x, y in zip(xs, ys))


# ==================== 载入源数据（全部只读） ====================
SLO = json.load(open(SWEEP + "/slo-analysis.json"))
M = json.load(open(SWEEP + "/matrix.json"))
D = json.load(open(SWEEP + "/per-metric-distributions.json"))
V1C = json.load(open(BAL + "/guardrail-v1-20260910/v1-comparison.json"))
CI = json.load(open(BAL + "/cause-investigation/summary.json"))
OA = json.load(open(BAL + "/mechanism-v4/online-affinity-summary.json"))
DP = json.load(open(BAL + "/alignment-v3/deployments.json"))["response"]["data"]

CAPS = [0, 50, 100, 250, 500, 1000, 100000000]
CAPKEY = ["0", "50", "100", "250", "500", "1000", "100000000"]
LABELS = ["0", "50", "100", "250", "500", "1000", "\u221e"]
MC = SLO["m_calibration"]
GS = SLO["golden_sweep"]
V1A = SLO["v1_arms"]
V0A = SLO["v0_arms"]

# ==================== S1：r 分布直方图（staging 逐请求重算）====================
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

# 固定档 M 在黄金点的违约率（staging 重算 · 仅作「M 必须校准」注记）
fix_x, fix_y = [], []
for m, lab in [(1.5, "M1.5"), (2.0, "M2.0"), (MC["M_cal_p95"], "Mcal"), (3.0, "M3.0")]:
    v = sum(1 for r in rpool if r > m) / len(rpool)
    fix_x.append(round(m, 4))
    fix_y.append(round(v * 100, 2))
    if lab == "Mcal":
        mc_viol_re = v

# ==================== S5 场景健康数据（matrix.json 逐格重算 mean/std/seed）====================
import statistics


def seedmat(field, getter=None):
    out = [[0.0] * 7 for _ in range(3)]
    for c in M:
        out[c["seed"]][CAPS.index(c["cap"])] = getter(c) if getter else c[field]
    return out


ttftSeed = seedmat("sustain_ttft_p95_s")
evSeed = seedmat(None, lambda c: round(c["kv_pool_engines"]["prefill-0"]["eviction_age_avg_s"], 2))


def colstat(mat, i):
    col = [mat[s][i] for s in range(3)]
    return round(statistics.fmean(col), 6), round(statistics.stdev(col), 6)  # 样本标准差（n-1，与 distributions 同口径）


ttftMean = [colstat(ttftSeed, i)[0] for i in range(7)]
ttftStd = [colstat(ttftSeed, i)[1] for i in range(7)]
evMean = [colstat(evSeed, i)[0] for i in range(7)]
evStd = [colstat(evSeed, i)[1] for i in range(7)]
base = round(statistics.fmean([ttftSeed[s][3] for s in range(3)]), 6)  # cap250 三 seed 均值
h1Band = {"lo": round(base * 0.99, 8), "hi": round(base * 1.01, 8), "base": base}
evBand = {"lo": 200, "hi": 300}

# ==================== SLO 核心数据 ====================
bycap = {b["cap"]: b for b in GS["by_cap"]}
base_bl = GS["baseline_cap250"]
BASE_P95 = base_bl["slo_viol_by_seed"]["M_cal_p95"]["p95"]
BL_TPS = base_bl["sustain_input_tps"]
BL_HOT = base_bl["hot_hit_rate"]
BL_BG = base_bl["bg_hit_rate_avg"]

new_pct = [round(bycap[c]["slo_viol"]["M_cal_p95"] * 100, 4) for c in CAPS]
new99_pct = [round(bycap[c]["slo_viol"]["M_cal_p99"] * 100, 4) for c in CAPS]
attr_pct = [round(bycap[c]["attr_viol"]["M_cal_p95"] * 100, 4) for c in CAPS]
ttft95 = [round(bycap[c]["ttft_ms_p95"], 2) for c in CAPS]
tpsR = [round(bycap[c]["input_tps"] / BL_TPS, 4) for c in CAPS]
hotR = [round(bycap[c]["hot_hit_rate"] / BL_HOT, 4) for c in CAPS]
bgR = [round(bycap[c]["bg_hit_rate_avg"] / BL_BG, 4) for c in CAPS]

# S3：21 格逐 seed 违约率（%）
violSeed = [[0.0] * 7 for _ in range(3)]
for cell in GS["cells"]:
    violSeed[cell["seed"]][CAPS.index(cell["cap"])] = round(cell["slo"]["slo_viol"]["M_cal_p95"] * 100, 4)

# 高压三臂（S2 + S4）—— 顺序 hi/finite/zero（违约率升序 = 结论序）
ARMS = ["hi", "finite", "zero"]
v1 = {}
for a in ARMS:
    s = V1A[a]["slo"]
    v1[a] = {
        "newPct": round(s["slo_viol"]["M_cal_p95"] * 100, 4),
        "attrPct": round(s["attr_viol"]["M_cal_p95"] * 100, 4),
        "attr2Pct": round(s["attr_viol_ratio_cf"]["M_cal_p95"] * 100, 4),
        "ttft": round(s["ttft_ms"]["p95"], 1),
        "tpsR": round(V1A[a]["ratios_vs_zero"]["input_tps"], 5),
        "hotR": round(V1A[a]["ratios_vs_zero"]["hot_hit"], 5),
        "bgR": round(V1C[a]["hotspot_hit_bg"] / V1C["zero"]["hotspot_hit_bg"], 4),
    }

# X1 摸测列（matrix · cap=1e8 三格 / cap250）
G = next(c for c in M if c["cap"] == 250 and c["seed"] == 0)

DATA = {
    "caps": CAPS, "labels": LABELS,
    "ttftMean": ttftMean, "ttftStd": ttftStd,
    "evMean": evMean, "evStd": evStd,
    "ttftSeed": ttftSeed, "evSeed": evSeed,
    "h1Band": h1Band, "evBand": evBand,
    "violSeed": violSeed, "violMean": new_pct,
    "baseP95Pct": round(BASE_P95 * 100, 4),
    "hist": {"centers": hist_centers, "counts": counts, "bw": BW},
    "mcal": {"p95": MC["M_cal_p95"], "p99": MC["M_cal_p99"],
             "mp95": MC["Mp_cal_p95"], "n": MC["n_pooled"]},
    "fixM": {"x": fix_x, "y": fix_y},
    "v1arms": {"arms": ARMS, "v": [v1[a] for a in ARMS]},
}

# ==================== 断言 1：SLO 核心数字（用户验收口径逐项对账）====================
A = [
    ("M_cal P95 = 2.6397", approx(MC["M_cal_p95"], 2.6397, 1e-9)),
    ("M_cal P99 = 2.6528", approx(MC["M_cal_p99"], 2.6528, 1e-9)),
    ("M'_cal P95 = 1.0398", approx(MC["Mp_cal_p95"], 1.0398, 1e-9)),
    ("pooled n = 8100", MC["n_pooled"] == 8100),
    ("固定档 M1.5 误判 50.04%", approx(base_bl["slo_viol_by_seed"]["M1.5"]["p95"], 0.50037, 1e-6)),
    ("固定档 M2.0 误判 50.04%", approx(base_bl["slo_viol_by_seed"]["M2.0"]["p95"], 0.50037, 1e-6)),
    ("固定档 M3.0 全漏 0%", base_bl["slo_viol_by_seed"]["M3.0"]["p95"] == 0.0),
    ("基线 seed P95 门限 = 5.43%", approx(BASE_P95, 0.054333, 1e-6)),
    ("v1 hi 违约 29.88%（三臂最少）", approx(v1["hi"]["newPct"], 29.8829, 1e-3)),
    ("v1 finite 违约 59.81%", approx(v1["finite"]["newPct"], 59.809, 1e-3)),
    ("v1 zero 违约 64.98%（三臂最多）", approx(v1["zero"]["newPct"], 64.9784, 1e-3)),
    ("违约排序 hi < finite < zero", v1["hi"]["newPct"] < v1["finite"]["newPct"] < v1["zero"]["newPct"]),
    ("v1 hi TTFT p95 682ms（三臂最低）", approx(v1["hi"]["ttft"], 681.6, 0.05)),
    ("v1 finite TTFT 958ms", approx(v1["finite"]["ttft"], 958.39, 0.05)),
    ("v1 zero TTFT 875ms", approx(v1["zero"]["ttft"], 874.51, 0.05)),
    ("TTFT 排序 hi < zero < finite", v1["hi"]["ttft"] < v1["zero"]["ttft"] < v1["finite"]["ttft"]),
    ("v1 hi 归因违约 12.77%", approx(v1["hi"]["attrPct"], 12.7665, 1e-3)),
    ("v1 finite 归因违约 3.73%", approx(v1["finite"]["attrPct"], 3.7338, 1e-3)),
    ("v1 zero 归因违约 0%", v1["zero"]["attrPct"] == 0.0),
    ("hi 归因两口径接近 12.77/12.92", approx(v1["hi"]["attr2Pct"], 12.9205, 1e-3)),
    ("v1 吞吐/命中三臂几乎相同（±0.2%）", all(abs(v1[a]["tpsR"] - 1) < 0.002 and abs(v1[a]["hotR"] - 1) < 0.002 for a in ARMS)),
    ("v1 hi 吞吐比 1.00096", approx(v1["hi"]["tpsR"], 1.00096, 1e-5)),
    ("v1 hi hot 命中比 0.99868", approx(v1["hi"]["hotR"], 0.998677, 1e-5)),
    ("v0 hi 对照 18.52%/644ms", approx(V0A["hi"]["slo"]["slo_viol"]["M_cal_p95"], 0.185213, 1e-6)
        and approx(V0A["hi"]["slo"]["ttft_ms"]["p95"], 643.9971, 1e-3)),
    ("低压 TTFT 全档 ~578ms", all(577 < t < 580 for t in ttft95)),
    ("cap≤250 违约 ~5%", all(4 < p < 6 for p in new_pct[:4])),
    ("cap250 违约 5.06%（校准点）", approx(new_pct[3], 5.0617, 1e-3)),
    ("cap500 违约 0.85%", approx(new_pct[4], 0.8519, 1e-3)),
    ("cap≥500 违约 ~1%", all(p < 1.3 for p in new_pct[4:])),
    ("cap1e8 违约 0.73%", approx(new_pct[6], 0.7284, 1e-3)),
    ("hit 升 1.17×（250→500）", approx(bycap[500]["hot_hit_rate"] / bycap[250]["hot_hit_rate"], 1.1717, 1e-3)),
    ("violSeed 21 格逐 seed 与 by_cap 均值一致", all(
        approx(sum(violSeed[s][i] for s in range(3)) / 3, new_pct[i], 0.02) for i in range(7))),
    ("violSeed cap250 三 seed 含贴门限值", any(v >= BASE_P95 * 100 - 0.15 for v in [violSeed[s][3] for s in range(3)])),
    ("verdict 无一格 HARMFUL", all(c["verdict"]["M_cal_p95"]["harmful_verdict"] is False for c in GS["cells"])),
    ("by_cap harmful_seeds 全 0", all(b["verdict_M_cal_p95"]["harmful_seeds"] == 0 for b in GS["by_cap"])),
    ("join golden 56700/56700 · 21 格全量", SLO["meta"]["join_coverage_summary"]["golden_total_joined"] == 56700
        and SLO["meta"]["join_coverage_summary"]["golden_expected"] == 56700
        and SLO["meta"]["join_coverage_summary"]["golden_cells_full"] == 21),
    ("join v1 每臂 16230/16230", all(V1A[a]["join_coverage"]["joined"] == V1A[a]["join_coverage"]["trace_rows"] == 16230
        and V1A[a]["join_coverage"]["coverage_ratio"] == 1.0 for a in ARMS)),
    ("join v0 每臂 16230", all(V0A[a]["join_coverage"]["joined"] == 16230 for a in ARMS)),
    # S1 直方图 staging 重算自洽
    ("直方图重算 n=8100", sum(counts) == 8100 == len(rpool)),
    ("重算 P95 ≈ 2.6397（±0.002）", approx(p95_re, MC["M_cal_p95"], 2e-3)),
    ("重算 P99 ≈ 2.6528（±0.002）", approx(p99_re, MC["M_cal_p99"], 2e-3)),
    ("重算 M_cal 违约 ≈ 5%（tautology）", approx(mc_viol_re, 0.05, 6e-3)),
    ("重算 M1.5 违约 == JSON 50.04%", approx(fix_y[0] / 100, base_bl["slo_viol_by_seed"]["M1.5"]["p95"], 2e-3)),
    ("重算 M3.0 违约 == JSON 0%", approx(fix_y[3], 0.0, 0.01)),
    # S5 数据 matrix 重算 × distributions 交叉核对
    ("ttftMean == distributions", arrapprox(ttftMean, [D[c]["sustain_ttft_p95_s"]["mean"] for c in CAPKEY], 1e-6)),
    ("ttftStd == distributions", arrapprox(ttftStd, [D[c]["sustain_ttft_p95_s"]["std"] for c in CAPKEY], 1e-6)),
    ("evMean == distributions", arrapprox(evMean, [D[c]["kv_pool"]["prefill-0.eviction_age_avg_s"]["mean"] for c in CAPKEY], 1e-2)),
    ("h1Band 基准 0.5787s", approx(h1Band["base"], 0.578738, 1e-5)),
]
bad = [n for n, ok in A if not ok]
if bad:
    print("ASSERT FAIL:", bad)
    sys.exit(1)
print("[OK] 数据对账断言全部通过：%d 项（SLO 核心 + staging 重算 + 交叉核对）" % len(A))

# ==================== 呈现数值 ====================
p2 = lambda v: ("%.2f" % v).rstrip("0").rstrip(".")
fmtp = lambda v: p2(v) + "%"
fmtt = lambda v: str(int(v + 0.5)) if v >= 0 else str(-int(-v + 0.5))  # floor(v+0.5) 显示口径


def caplab(c):
    return "∞（1e8）" if c == 100000000 else str(c)


# ---- S4 verdict 表（golden 7 档段 + 高压三臂段）----
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
        return '<span class="tag okc">OK ×%d</span>' % nok
    return '<span class="tag okc">OK ×%d</span> <span class="tag comp">comp ×%d</span>' % (nok, ncomp)


def s4row(name, sub, viol, attr, tpsr, hotr, bgr, verdict_html, cls=""):
    return ('<tr%s><td class="mono">%s<span class="vsub">%s</span></td>'
            '<td class="mono">%s</td><td class="mono">%s</td>'
            '<td class="mono dim">%s</td><td class="mono dim">%s</td><td class="mono dim">%s</td>'
            '<td>%s</td></tr>' % (
                (' class="%s"' % cls) if cls else "",
                name, sub, viol, attr, tpsr, hotr, bgr, verdict_html))


S4 = ['<tr class="sechead"><td colspan="7">GOLDEN 21 格 · 3 QPS 低压 · 比值 vs cap250 黄金点基线（tps %.0f · hot %.4f · bg %.4f）· 违约门限 = 基线 seed P95 %s</td></tr>'
      % (BL_TPS, BL_HOT, BL_BG, fmtp(BASE_P95 * 100))]
for i, c in enumerate(CAPS):
    labels = [cell["verdict"]["M_cal_p95"]["label"] for cell in GS["cells"] if cell["cap"] == c]
    sub = "TTFT p95 %.0fms · 3 seed" % ttft95[i]
    if c == 250:
        sub += " · 基线（M_cal 校准点 · 违约≈5% 为 P95 tautology）"
    S4.append(s4row(
        "cap " + caplab(c), sub,
        fmtp(new_pct[i]) + '<span class="vsub">P99 口径 ' + fmtp(new99_pct[i]) + "</span>",
        fmtp(attr_pct[i]),
        "%.4f" % tpsR[i], "%.4f" % hotR[i], "%.4f" % bgR[i],
        vtag(labels)))
S4.append('<tr class="sechead"><td colspan="7">高压三臂 · 154 QPS · 比值 vs zero 臂（regime 内对照 · 绝对违约排序 hi &lt; finite &lt; zero）· 违约门限 = zero 臂基线 seed 分布 P95（超门限且有补偿 = compensated）</td></tr>')
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
    S4.append(s4row(
        cname, "TTFT p95 %sms · n=16230" % fmtt(v["ttft"]),
        fmtp(v["newPct"]) + '<span class="vsub">P99 口径 ' + fmtp(round(V1A[a]["slo"]["slo_viol"]["M_cal_p99"] * 100, 4)) + "</span>",
        fmtp(v["attrPct"]),
        "%.5f" % v["tpsR"], "%.5f" % v["hotR"], "%.4f" % v["bgR"],
        vd))
S4ROWS = "\n".join(S4)

# ---- T1 门禁表（S1 = SLO 违约率 + 归因违约佐证）----
s1_row = ('<tr><td class="mono">S1</td><td>有害判定（SLO 违约率）</td><td><span class="tag soft">软</span></td>'
          '<td class="mono">违约率 &gt; 基线 seed 分布 P95（%s）∧ 无吞吐/命中补偿；归因违约盯梢佐证</td>'
          '<td class="mono dim">21 格无一格 HARMFUL；高压三臂 hi 违约 %s 三臂最少、TTFT p95 %sms 三臂最低（归因违约 %s 佐证亲和让步占比）；'
          '低压 cap≤250 违约 ~5%% 为 M_cal P95 tautology 仅盯相对变化；'
          'M 按 regime 黄金点数据校准（P95=%s），固定档 1.5/2.0/3.0 误判 50%%/50%%/全漏</td></tr>'
          ) % (fmtp(BASE_P95 * 100), fmtp(v1["hi"]["newPct"]), fmtt(v1["hi"]["ttft"]), fmtp(v1["hi"]["attrPct"]), "%.4f" % MC["M_cal_p95"])
T1ROWS = "\n".join([
    '<tr><td class="mono">H1</td><td>TTFT p95</td><td><span class="tag hard">硬</span></td><td class="mono">±1%（0.5730–0.5845 s）</td><td class="mono dim">7 档 seed CV ≤ 0.17%（黄金点档基准 0.5787 s）</td></tr>',
    '<tr><td class="mono">H2</td><td>sustain 吞吐</td><td><span class="tag hard">硬</span></td><td class="mono">±4σ</td><td class="mono dim">7 档 input_tps seed CV ≤ 0.031%，σ ≪ 容差</td></tr>',
    s1_row,
    '<tr><td class="mono">S5</td><td>驱逐年龄（prefill-0）</td><td><span class="tag soft">软</span></td><td class="mono">200–300 s 带内</td><td class="mono dim">cap≥500 稳定 251.4 s 带内；cap≤250 波动 190–507 s 部分出带</td></tr>',
    '<tr><td class="mono">INVALID</td><td>拒绝数</td><td><span class="tag hard">硬</span></td><td class="mono">= 0</td><td class="mono dim">21 格 rejections 全 0，issued 56700/56700</td></tr>',
])

# ---- X1 三方对照行（有害行 = SLO 违约口径 · 线上列 N/A 如实）----
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
    trow("有害判定（SLO 违约率 · M_cal P95）",
         [("N/A", "线上无逐请求 TTFT⋈prefill join 能力，违约率不可重算 · 高压同算法实测见下行", None),
          (fmtp(v1["hi"]["newPct"]), "154QPS 高压 v1-hi 实测（同 cap=1e8 算法）· 三臂最少 · 归因违约 " + fmtp(v1["hi"]["attrPct"]), "v1-hi·SLO")],
         [("N/A", "无该算法路径", None)],
         [(fmtp(new_pct[3]), "cap250 = M_cal 校准点（P95 tautology 邻域）· cap≥500 降至 " + fmtp(new_pct[4]) + "–" + fmtp(new_pct[6]), "摸测·SLO")]),
    trow("客户端 TTFT p95（SLO 违约判定口径）",
         [("682 ms", "154QPS 高压 cap=1e8 实测 · 三臂最低（zero 875 · finite 958）", "v1-hi·SLO")],
         [("—", "线上无逐请求口径", None)],
         [("579 ms", "全档 577.5–579.6 平坦 · 低压无判别力", "摸测·SLO")]),
    trow("hot 命中率",
         [("0.750 / 0.683", "高压 v1-hi / 低压摸测 cap=1e8", "v1-hi·摸测")],
         [("—", "线上无逐请求命中口径", None)],
         [("0.683", "3 seed %.3f–%.3f · seed 敏感" % (hot250[0], hot250[-1]), "摸测")]),
    trow("bg 命中率形态",
         [("0.327", "bg 命中形态（高压）", "v1-hi")],
         [("—", "线上无逐请求命中口径", None)],
         [("0.494", "3 seed %.3f–%.3f" % (bg250[0], bg250[-1]), "摸测")]),
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
    "@@V1HINEW@@": fmtp(v1["hi"]["newPct"]),
    "@@V1FINNEW@@": fmtp(v1["finite"]["newPct"]),
    "@@V1ZERONEW@@": fmtp(v1["zero"]["newPct"]),
    "@@V1HIATTR@@": fmtp(v1["hi"]["attrPct"]),
    "@@V1FINATTR@@": fmtp(v1["finite"]["attrPct"]),
    "@@V1ZEROATTR@@": fmtp(v1["zero"]["attrPct"]),
    "@@V1HIATTR2@@": fmtp(v1["hi"]["attr2Pct"]),
    "@@V1HITTFT@@": fmtt(v1["hi"]["ttft"]),
    "@@V1FINTTFT@@": fmtt(v1["finite"]["ttft"]),
    "@@V1ZEROTTFT@@": fmtt(v1["zero"]["ttft"]),
    "@@V1HITPS@@": "%.5f" % v1["hi"]["tpsR"],
    "@@V1HIHOT@@": "%.5f" % v1["hi"]["hotR"],
    "@@BASEP95@@": fmtp(BASE_P95 * 100),
    "@@TTFTFLAT@@": "578",
    "@@PHNEWLO@@": fmtp(new_pct[3]),
    "@@PHNEWHI@@": fmtp(new_pct[4]),
    "@@N1000PCT@@": fmtp(new_pct[6]),
    "@@JOINGOLDEN@@": str(JOING),
    "@@JOINV1@@": "16230",
    "@@NOCOUNT@@": "21",
    "@@S4ROWS@@": S4ROWS, "@@T1ROWS@@": T1ROWS, "@@ROWS@@": ROWS,
    "@@DATA@@": json.dumps(DATA, ensure_ascii=False, separators=(",", ":")),
    "@@JS@@": open(V4_JS).read().strip("\n"),
    "@@TS@@": datetime.now().strftime("%Y-%m-%d %H:%M"),
}

HTML = open(TEMPLATE).read()
for k, v in TOKENS.items():
    HTML = HTML.replace(k, v)
assert "@@" not in HTML, "存在未替换占位符: %s" % re.findall(r"@@\w+@@", HTML)[:5]

with open(OUT, "w") as f:
    f.write(HTML)

# ==================== 写回自检（结构 + 禁词零命中）====================
o = open(OUT).read()
CHECKS = [
    ('<canvas id="c1"', 1), ('<canvas id="c2"', 1), ('<canvas id="c3"', 1),
    ('<canvas id="c4"', 1), ('<canvas id="c5"', 1), ('<canvas id="c6"', 1),
    ('id="report-data"', 1), ('class="subgrid"', 3), ('class="formula"', 1),
    ('class="deflist"', 2), ('class="tblwrap"', 4), ('minmax(0,', 2),
    ('2.6397', 2), ('29.88%', 3), ('59.81%', 2), ('64.98%', 2),
    ('682', 3), ('875', 1), ('958', 1),
    ('0.85%', 2), ('5.06%', 1), ('0.73%', 1), ('12.77%', 3), ('3.73%', 1),
    ('50.04%', 2), ('1.0398', 2), ('5.43%', 3), ('1.00096', 2), ('0.99868', 2),
    ('S1', 1), ('S2', 1), ('S3', 1), ('S4', 1), ('S5', 1), ('N1', 1), ('T1', 1), ('X1', 1),
    ('compensated', 3), ('HARMFUL', 2),
    ('slo-analysis.json', 1), ('RUN-slo.md', 1), ('/tmp/slo-golden', 1),
    ('v1-comparison.json', 1), ('matrix.json', 1),
]
for key, want in CHECKS:
    got = o.count(key)
    assert got >= want, "check %r: want>=%d got=%d" % (key, want, got)

BANNED = ["delta&gt;100", "delta>100", "旧口径", "新口径", "反转", "判反", "伪影",
          "18.76", "新旧", "旧指标", "旧有害", "旧 delta", "相变", "20.85", "判反实锤",
          "oldPct", "old_pct", "old_harmful", "phase", "tri\":", "phaseLine", "diagRef", "armLabels",
          "P1&prime;", "一致性散点", 'id="c7"', 'id="c8"', 'id="c9"', 'id="c10"', 'id="c11"',
          "v1/v2/v3", "演进"]
hits = [(w, o.count(w)) for w in BANNED if w in o]
assert not hits, "禁词命中: %s" % hits

json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>', o, re.S).group(1))
print("[OK] 写回完成：%s（%d bytes）" % (OUT, len(o)))
print("[OK] %d 项 HTML 结构自检 + %d 词禁词扫描零命中 + report-data JSON 有效性校验通过" % (len(CHECKS), len(BANNED)))
