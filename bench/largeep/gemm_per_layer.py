"""Per-layer routed group-GEMM time per rank from a kineto trace.
EP16: DeepGEMM masked kernels (sm120_fp8_fp4_gemm_1d1d_impl<0u, 4096u, ...>) between k_d2 and k_c1 on the rail stream.
TP4: FlashInfer cutlass_fused_moe grouped kernels (GemmUniversal<GroupProblemShape...>), fc1 + fc2 per layer.
Forwards are split at gaps > 1.5 ms; verify forwards have 43 MoE layers, the MTP draft 1.
usage: gemm_per_layer.py ep16|tp4 trace.json out.json"""

import collections
import json
import re
import statistics as S
import sys

mode, f, out = sys.argv[1], sys.argv[2], sys.argv[3]
ev = json.load(open(f))["traceEvents"]
ks = sorted(
    (e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"),
    key=lambda e: e["ts"],
)
nm = (
    lambda k: re.sub(r"<.*", "", k["name"].split("(")[0])
    .replace("void ", "")
    .split("::")[-1]
)
layers = []  # (ts, [gemm durs], glue_dur)
if mode == "ep16":
    d2 = [i for i, k in enumerate(ks) if nm(k) == "k_d2"]
    c1 = [i for i, k in enumerate(ks) if nm(k) == "k_c1"]
    for a, b in zip(d2, c1):
        st = ks[a].get("args", {}).get("stream")
        g = [
            ks[i]["dur"]
            for i in range(a + 1, b)
            if ks[i].get("args", {}).get("stream") == st
            and "sm120_fp8_fp4_gemm_1d1d_impl<0u, 4096u" in ks[i]["name"]
        ]
        other = [
            ks[i]["dur"]
            for i in range(a + 1, b)
            if ks[i].get("args", {}).get("stream") == st
            and "gemm_1d1d" not in ks[i]["name"]
        ]
        layers.append((ks[a]["ts"], g, sum(other)))
else:
    gi = [i for i, k in enumerate(ks) if "GroupProblemShape" in k["name"]]
    for j in range(0, len(gi) - 1, 2):
        a, b = gi[j], gi[j + 1]
        layers.append((ks[a]["ts"], [ks[a]["dur"], ks[b]["dur"]], 0.0))
fw = [[layers[0]]]
for x, y in zip(layers, layers[1:]):
    if y[0] - x[0] > 1500:
        fw.append([y])
    else:
        fw[-1].append(y)
pos = collections.defaultdict(list)
draft = []
for fwd in fw:
    if len(fwd) >= 40:
        for j, L in enumerate(fwd):
            pos[j].append(L)
    elif len(fwd) == 1:
        draft.append(fwd[0])
res = {"verify_forwards": sum(1 for x in fw if len(x) >= 40), "layers": {}}
for j, v in sorted(pos.items()):
    res["layers"][j] = {
        "gemm_us": S.median(sum(L[1]) for L in v),
        "up_us": S.median(L[1][0] for L in v),
        "down_us": (
            S.median(L[1][1] for L in v) if all(len(L[1]) > 1 for L in v) else None
        ),
        "glue_us": S.median(L[2] for L in v),
        "n": len(v),
    }
res["draft_gemm_us"] = S.median(sum(L[1]) for L in draft) if draft else None
json.dump(res, open(out, "w"))
print(
    mode,
    f.split("/")[-1],
    "verify fwds",
    res["verify_forwards"],
    "layers",
    len(res["layers"]),
    "sum gemm over layers %.2f ms"
    % (sum(v["gemm_us"] for v in res["layers"].values()) / 1e3),
    "draft %s" % res["draft_gemm_us"],
)
