"""Only qualify observations from real same-input paired first-proposal operands."""

import json
import re
import sys
from pathlib import Path


def need(b, s):
    if not b:
        raise ValueError(s)


def hex64(s):
    return type(s) is str and re.fullmatch("[a-f0-9]{64}", s)


def read(directory, role):
    def get(name, stage):
        d = json.loads((Path(directory) / (name + ".json")).read_text())
        need(
            d["family"] == "v41draft"
            and d["role"] == role
            and d["stage"] == stage
            and re.fullmatch("[a-f0-9]{32}", d["nonce"]),
            "identity",
        )
        return d

    pp = [get("v41draft_p" + str(i), "p_commit") for i in range(2)]
    d = get("v41draft_d0", "d_propose")
    s = get("v41draft_scores", "d_score")
    need(
        len({v["nonce"] for v in [*pp, d, s]}) == 1
        and all(p["rank"] == str(i) for i, p in enumerate(pp)),
        "mixed pair",
    )
    need(
        sorted(pp[0]["positions"] + pp[1]["positions"]) == list(range(512, 783))
        and all(hex64(p["natural"]) and hex64(p["used"]) for p in pp),
        "incomplete features",
    )
    need(
        d["actual_hit"] == 512
        and d["prompt_digest"]
        == "d570954fff2202dfde49e873ece0fc1adcb2df1c6a9095cb98fe6601ac4645c4",
        "request",
    )
    need(
        s["anchor"] == d["query_ids"][0]
        and d["query_ids"][1] == d["query_ids"][2]
        and hex64(d["used_history_sha"])
        and hex64(d["natural_history_sha"]),
        "history",
    )
    need(
        set(d["layers"]) == set("012")
        and all(
            all(hex64(v[k]) for k in ("used", "natural", "attn_input", "query_kv"))
            for v in d["layers"].values()
        ),
        "cache coverage",
    )
    need(
        hex64(d["graph_head"])
        and hex64(d["native_head"]["kernel"])
        and d["verify"]["draft"][0] == s["mapped"],
        "native consumer",
    )
    need(set(d["ep_weights"]) == {"0", "1"}, "EP coverage")
    for rank, w in d["ep_weights"].items():
        need(
            w["nonce"] == d["nonce"]
            and w["role"] == role
            and w["rank"] == rank
            and w["stage"] == "d_weights"
            and w["slots"]
            and isinstance(w["missing"], dict),
            "weight coverage",
        )
        need(all(hex64(v["sha"]) for v in w["slots"].values()), "weight SHA")
    need(
        all(
            hex64(s[k])
            for k in ("base", "bias", "pre_sampler", "probabilities", "w1_used", "w2")
        )
        and (hex64(s["d2t"]) or s["d2t"] == "identity")
        and len(s["top"]) == 4,
        "sampler",
    )
    return pp, d, s


def compare(old, new):
    a, x, s = old
    b, y, t = new
    need(
        x["nonce"] == y["nonce"]
        and x["used_history_sha"] == y["used_history_sha"]
        and x["query_ids"] == y["query_ids"]
        and x["prompt_digest"] == y["prompt_digest"]
        and s["temperature"] == t["temperature"],
        "different prompt, history or sampling treatment",
    )
    need(
        all(
            p["positions"] == q["positions"] and p["used"] == q["used"]
            for p, q in zip(a, b)
        ),
        "different committed target features",
    )
    need(
        all(x["layers"][i]["used"] == y["layers"][i]["used"] for i in "012"),
        "different committed draft cache",
    )
    paths = {
        i: x["ep_weights"][i]["paths"] == y["ep_weights"][i]["paths"] for i in "01"
    }
    missing = {
        i: {k: v["ep_weights"][i]["missing"] for k, v in (("old", x), ("new", y))}
        for i in "01"
    }
    differences = {}
    for i in "01":
        l = x["ep_weights"][i]["slots"]
        r = y["ep_weights"][i]["slots"]
        differences[i] = sorted(k for k in set(l) | set(r) if l.get(k) != r.get(k))
    return dict(
        valid=True,
        same_effective_inputs=True,
        nonce=x["nonce"],
        same_natural_target_features=[
            p["natural"] == q["natural"] for p, q in zip(a, b)
        ],
        same_natural_history=x["natural_history_sha"] == y["natural_history_sha"],
        cache={
            i: {
                k: x["layers"][i][k] == y["layers"][i][k]
                for k in ("natural", "used", "attn_input", "query_kv")
            }
            for i in "012"
        },
        same_head_hidden=x["graph_head"] == y["graph_head"],
        same_lm_head=x["native_head"] == y["native_head"],
        same_expert_paths=paths,
        missing_weight_slots=missing,
        different_weight_slots=differences,
        score_equal={
            k: s[k] == t[k]
            for k in (
                "base",
                "bias",
                "pre_sampler",
                "probabilities",
                "w1_used",
                "w2",
                "d2t",
            )
        },
        first_proposal={
            k: {
                "top_before_sampling": v["top"],
                "id": v["mapped"],
                "verify": d["verify"],
            }
            for k, v, d in (("oldfst", s, x), ("newfst", t, y))
        },
        weight_attribution_qualified=not any(differences.values())
        and all(paths.values())
        and not any(m for values in missing.values() for m in values.values())
        and x["native_head"] == y["native_head"]
        and all(s[k] == t[k] for k in ("w1_used", "w2", "d2t")),
    )


if __name__ == "__main__":
    try:
        need(len(sys.argv) == 3, "need Old and New report directories")
        result = compare(read(sys.argv[1], "oldfst"), read(sys.argv[2], "newfst"))
    except (ValueError, OSError, TypeError, IndexError, KeyError) as e:
        print(
            json.dumps(
                {
                    "valid": False,
                    "reason": "missing or unmatched paired first proposal evidence",
                }
            )
        )
        sys.exit(2)
    print(json.dumps(result, separators=(",", ":")))
