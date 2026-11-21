"""Compare genuine OldFST/NewFST ARM smoke ZIPs. No raw payload is printed."""
import argparse
import io
import json
import math
import re
from zipfile import ZipFile

SUITES = ("next_moe_nvfp4_cudagraph_tp2_sm100", "next_moe_nvfp4_deepep_ll_cudagraph_dp2_sm100")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def read_case(outer, suite, side, diagnostic):
    label = suite + ("_sample_diag" if diagnostic else "")
    found = [p for p in outer.namelist() if p.endswith("/" + label + "/test.outputs/outputs.zip")]
    need(len(found) == 1, "missing or ambiguous suite: " + label)
    location = found[0]
    test = outer.read(location.replace("test.outputs/outputs.zip", "test.log")).decode(errors="replace")
    scores = set(re.findall(r"total count:\[(\d+)\], suc count:\[(\d+)\], compare diff count:\[(\d+)\], "
                            r"visit_failed_count:\[(\d+)\], other_count: \[(\d+)\]", test))
    need(len(scores) == 1, "no unique original golden summary")
    total, good, diff, visit, other = map(int, scores.pop())
    need(total == 2 and good + diff == 2 and visit == other == 0 and (side == "new" or good == 2),
         "incomplete requests, or OldFST golden not 2/2")
    with ZipFile(io.BytesIO(outer.read(location))) as worker:
        names = worker.namelist()
        need(len(names) == len(set(names)), "duplicate artifact members")
        owners = {}
        for name in names:
            m = re.fullmatch(r"main_logs/batch_record/owner-(\d+)-\d+/manifest.json", name)
            if m:
                owners[m[1]] = json.loads(worker.read(name))["world_rank"]
        need(sorted(owners.values()) == [0, 1], "missing two-rank owners")
        lines = worker.read("main_logs/process.log").decode(errors="replace").splitlines()
        pattern = (r"finally choose load method: (\w+)" if side == "old" else
                   r"model loaded successfully: model_type=qwen35_moe selected=(\w+)")
        for pid in owners:
            text = "\n".join(l for l in lines if l.startswith("[process-" + pid + "]"))
            need(set(re.findall(pattern, text)) == {"fastsafetensors"}, "worker did not use expected FST")
            if side == "new":
                need("Streamed checkpoint tensors with fastsafetensors" in text, "no actual NewFST reader")
        rows = []
        for name, expected in (("single", 1), ("batch", 3)):
            prefix = "next-prefix-v1-" + name
            need(prefix + ".duplicate" not in names and prefix + ".json" in names, "missing/ambiguous fixture capture")
            data = json.loads(worker.read(prefix + ".json"))
            need(set(data) == {"version", "rows"} and data["version"] == 1 and len(data["rows"]) == expected,
                 "invalid emitted-prefix schema")
            for row in data["rows"]:
                need(isinstance(row, list) and 1 <= len(row) <= 20 and
                     all(isinstance(x, str) and re.fullmatch("[0-9a-f]{64}", x) for x in row), "invalid digest")
                rows.append(row)
        samples = {}
        for line in lines:
            if "NEXT_SMOKE_AUDIT_V1 " not in line:
                continue
            need(diagnostic, "ordinary golden unexpectedly sampled GPU logits")
            rank = re.search(r"\[(\d+):\d+\]\s*\[RANK (\d+)\]", line)
            hit = re.search(r"NEXT_SMOKE_AUDIT_V1 slot=([0-3]) step=(\d+) pick=([012]) margin=(\S+)", line)
            need(rank and hit and rank[1] in owners and int(rank[2]) == owners[rank[1]], "unbound log row")
            slot, step, pick = map(int, hit.groups()[:3])
            margin = float(hit[4])
            need(int(rank[2]) == int("dp2" in suite and slot == 0), "unexpected TP/DP serving rank")
            need(0 <= step < len(rows[slot]) and (slot, step) not in samples and math.isfinite(margin) and margin >= 0,
                 "invalid, duplicate or non-finite sample")
            samples[slot, step] = {"top_rank": pick, "top2_margin": margin}
        if diagnostic:
            need(len(samples) == sum(map(len, rows)), "missing diagnostic steps (no partial evidence accepted)")
    return good, rows, samples


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                None if len(a) == len(b) else min(len(a), len(b)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", required=True); p.add_argument("--new", required=True)
    p.add_argument("--old-diag"); p.add_argument("--new-diag")
    args = p.parse_args()
    need(bool(args.old_diag) == bool(args.new_diag), "need both diagnostic ZIPs or neither")
    result = {"same_physical_gpu_pair": "not guaranteed by remote CI", "suites": {}}
    for suite in SUITES:
        obs = {}
        for side in ("old", "new"):
            with ZipFile(getattr(args, side)) as archive:
                obs[side] = read_case(archive, suite, side, False)
            diag = getattr(args, side + "_diag")
            if diag:
                with ZipFile(diag) as archive:
                    g, rows, samples = read_case(archive, suite, side, True)
                need((g, rows) == obs[side][:2], "instrumentation changed ordinary outputs; do not attribute cause")
                obs[side] = g, rows, samples
        out = result["suites"][suite] = {"old_golden": obs["old"][0], "new_golden": obs["new"][0], "rows": []}
        for slot, (a, b) in enumerate(zip(obs["old"][1], obs["new"][1])):
            k = first_difference(a, b)
            item = {"slot": slot, "first_emitted_step0": k}
            if k is not None and k == min(len(a), len(b)):
                item["result"] = "length-only difference"
            elif k is not None and args.old_diag:
                o, n = (obs[s][2][slot, k] for s in ("old", "new"))
                item.update(old=o, new=n)
                item["different_pre_sampler_winner"] = (
                    True if all(x["top_rank"] == 1 and x["top2_margin"] > 0 for x in (o, n)) else "not established"
                )
            out["rows"].append(item)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
