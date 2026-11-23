"""Generate the detailed chart report from native windows; optionally align an unlimited reference."""
import argparse
import heapq
import json
import struct
import subprocess
import sys
from pathlib import Path

from analyze import compare, load


def read(p):
    return json.loads(p.read_bytes())


def summarize(run, theory):
    identity, total = load(run)
    doc = read(run / "native/summary.json")
    windows = sorted(doc["timeline"], key=lambda w: w["start_ns"])
    reference = None
    if theory:
        compare([run, theory])
        ref = read(theory / "native/summary.json")
        reference = {w["start_ns"]: w["metrics"] for w in ref["timeline"]}
        if set(reference) != {w["start_ns"] for w in windows}:
            raise ValueError("incompatible reference windows")
    first = identity["warmup"] + 1
    rows, cumulative = [], []
    end = inputs = full = joint = theoretical = 0
    for w in windows:
        t, start = w["metrics"], w["start_ns"]
        f, j, den = t["full_tokens"]["sum"], t["joint_tokens"]["sum"], t["input_tokens"]
        upper = None
        if reference is not None:
            u = reference[start]
            if (u["requests"], u["input_tokens"]) != (t["requests"], den):
                raise ValueError("reference window denominator mismatch")
            upper = u["joint_tokens"]["sum"]
            theoretical += upper
        end += t["requests"]
        inputs += den
        full += f
        joint += j
        row = dict(t, end_request=end, first_request=end-t["requests"]+1,
                   first_ns=start, last_ns=start+doc["window_ns"]-1,
                   full_hit_rate=f/den, joint_hit_rate=j/den,
                   joint_full_ratio=j/f if f else None, theory_hit_rate=upper/den if upper is not None else None,
                   attainment=j/upper if upper else None)
        rows.append(row)
        cumulative.append(dict(end_request=end, full_hit_rate=full/inputs, joint_hit_rate=joint/inputs,
                               theory_hit_rate=theoretical/inputs if theory else None))
    if (end,inputs,full,joint)!=(total["requests"],total["input_tokens"],total["full_tokens"]["sum"],total["joint_tokens"]["sum"]):
        raise ValueError("timeline totals differ from native summary")
    all_rows = dict(total, **cumulative[-1], attainment=joint/theoretical if theoretical else None,
                    joint_full_ratio=joint/full if full else None, first_ns=rows[0]["first_ns"], last_ns=rows[-1]["last_ns"],
                    fallback_requests=None)
    facts = run / "native/match-lengths.u64"
    size = facts.stat().st_size
    if size % 16 or size // 16 > end:
        raise ValueError("invalid per-request facts")
    offset = end - size // 16
    examples = []
    fallback = zero = 0
    with facts.open("rb") as stream:
        ordinal = offset
        for block in iter(lambda: stream.read(16*8192), b""):
            for f, j in struct.iter_unpack("<QQ", block):
                ordinal += 1
                if f > j:
                    fallback += 1
                    zero += j == 0
                    entry = (f-j, ordinal, f, j)
                    if len(examples) < 20:
                        heapq.heappush(examples, entry)
                    else:
                        heapq.heappushpop(examples, entry)
    if offset == 0:
        all_rows["fallback_requests"] = fallback
    sample_rows = [dict(request=i, source_request=first+i-1, full_tokens=f,
                        joint_tokens=j, gap_tokens=gap) for gap,i,f,j in sorted(examples,reverse=True)]
    return dict(windows=rows, cumulative=cumulative,total=all_rows,request_step=0,
                window_note=f"按日志时间每 {doc['window_ns']/1e9:g} 秒分段",
                theory_note="用户指定的同区间无限容量全保留基准" if theory else "未提供对齐的无限容量基准",
                identity=dict(identity,source_first_request=first,source_last_request=first+end-1),
                fallback_examples=sample_rows, native_examples=True,
                fact_scope=f"逐请求文件覆盖正式请求 {offset+1:,}–{end:,}。该段回退 {fallback:,} 条，其中回退到零 {zero:,} 条。", parameters=read(run/"parameters.json"),verification={"分段与累计值一致":True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--theory-run",type=Path)
    a=p.parse_args()
    data=summarize(a.run,a.theory_run)
    a.output.mkdir(parents=True,exist_ok=False)
    metrics=a.output/"report-metrics.json"
    metrics.write_text(json.dumps(data))
    subprocess.run([sys.executable,str(Path(__file__).with_name("charts.py")),str(metrics),str(a.output/"report")],check=True)


if __name__=="__main__":
    main()
