import glob
import json
import sys


def pct(v, p):
    if not v:
        return float("nan")
    v = sorted(v)
    i = min(len(v) - 1, max(0, int(round(p / 100.0 * (len(v) - 1)))))
    return v[i]


LONG = 16384


def analyze(path):
    ok = fail = lok = lfail = sok = sfail = 0
    tpot, ttft, dq, step, clat = [], [], [], [], []
    ltpot, lttft = [], []
    codes = {}
    elapsed = None
    for line in open(path):
        try:
            d = json.loads(line)
        except Exception:
            continue
        il = (d.get("source_metrics") or {}).get("input_len", 0)
        long_ = il >= LONG
        if d.get("status") != "ok":
            fail += 1
            lfail += long_
            sfail += not long_
            try:
                c = json.loads(d["error"])["error_code_str"]
            except Exception:
                c = str(d.get("error"))[:30]
            k = ("long" if long_ else "short", c)
            codes[k] = codes.get(k, 0) + 1
            continue
        ok += 1
        lok += long_
        sok += not long_
        dm = d.get("derived_metrics") or {}
        ai = d.get("aux_info") or {}
        ol = dm.get("returned_output_len") or ai.get("output_len") or 0
        if ol and ol > 4 and dm.get("tpot_by_output_len") is not None:
            tpot.append(dm["tpot_by_output_len"])
            if long_:
                ltpot.append(dm["tpot_by_output_len"])
        if ol and ol > 4 and dm.get("decode_step_time") is not None:
            step.append(dm["decode_step_time"])
        if dm.get("ttft") is not None:
            ttft.append(dm["ttft"])
            if long_:
                lttft.append(dm["ttft"])
        if dm.get("wait_time") is not None:
            dq.append(dm["wait_time"])
        cl = (d.get("replay") or {}).get("client_latency_ms")
        if cl is not None:
            clat.append(cl)
    return dict(
        ok=ok,
        fail=fail,
        lok=lok,
        lfail=lfail,
        sok=sok,
        sfail=sfail,
        tpot=tpot,
        ltpot=ltpot,
        ttft=ttft,
        lttft=lttft,
        dq=dq,
        step=step,
        clat=clat,
        codes=codes,
    )


def show(label, r):
    tot = r["ok"] + r["fail"]
    lt = r["lok"] + r["lfail"]
    st = r["sok"] + r["sfail"]
    print(f"== {label}")
    print(
        f"  总: {tot} ok={r['ok']} fail={r['fail']} 失败率={r['fail']/max(tot,1)*100:.1f}%"
    )
    print(
        f"  长(>=16k): {lt} 个, ok={r['lok']} 失败率={r['lfail']/max(lt,1)*100:.1f}% | "
        f"短: {st} 个, ok={r['sok']} 失败率={r['sfail']/max(st,1)*100:.1f}%"
    )
    print(
        f"  TPOT(ms): p50={pct(r['tpot'],50):.0f} p90={pct(r['tpot'],90):.0f} p99={pct(r['tpot'],99):.0f} | "
        f"长TPOT p50={pct(r['ltpot'],50):.0f} p99={pct(r['ltpot'],99):.0f}"
    )
    print(
        f"  TTFT(ms): p50={pct(r['ttft'],50):.0f} p99={pct(r['ttft'],99):.0f} | "
        f"长TTFT p50={pct(r['lttft'],50):.0f} p99={pct(r['lttft'],99):.0f}"
    )
    print(
        f"  decode排队(ms): p50={pct(r['dq'],50):.0f} p90={pct(r['dq'],90):.0f} p99={pct(r['dq'],99):.0f}"
    )
    print(f"  端到端(ms): p50={pct(r['clat'],50):.0f} p99={pct(r['clat'],99):.0f}")
    top = sorted(r["codes"].items(), key=lambda x: -x[1])[:4]
    if top:
        print(f"  失败码 top: {top}")


base = "/home/admin/workspace/aop_lab/app_source/latency_analysis/2026-08-06_rtp-llm-decode-routing-compare/operator_runs"
runs = [
    ("v32a0m-20260819T070834Z*", "A0m  | A原版        | rank0-only"),
    ("v32b0m-20260819T061029Z*", "B0m  | B收缩式offload | rank0-only"),
    ("v32br0m-*", "Br0m | B环准入(16k窗) | rank0-only"),
    ("v32am8-*", "Am8  | A原版        | +8rank分发  [对齐A基线]"),
    ("v32brm8-*", "Brm8 | B环准入(16k窗) | +8rank分发"),
    ("v32brm9-*", "Brm9 | B环准入(16k窗) | +8rank分发+块感知LB+环限流"),
]
for pat, label in runs:
    fs = glob.glob(f"{base}/{pat}/*/3to1/results.jsonl")
    if fs:
        show(label, analyze(fs[0]))
    else:
        print(f"== {label}: results not found ({pat})")
