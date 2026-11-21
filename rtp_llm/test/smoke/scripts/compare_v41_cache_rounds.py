"""Compare fixed V4.1 fixture rounds. Exit 0 means valid evidence, not a green smoke."""

import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

PHASES = ("cold", "fill", "hit")
DIGEST = "d570954fff2202dfde49e873ece0fc1adcb2df1c6a9095cb98fe6601ac4645c4"


def require(ok, field):
    if not ok:
        raise ValueError("invalid V4.1 cache evidence: " + field)


def hash_ids(ids):
    return hashlib.sha256(struct.pack("<" + "i" * len(ids), *ids)).hexdigest()


def validate(doc):
    require(
        doc["schema"] == 1
        and doc["input_ids_sha256"] == DIGEST
        and doc["input_len"] == 783,
        "fixture",
    )
    require(re.fullmatch("[a-f0-9]{32}", doc["run_id"]) is not None, "run_id")
    require(
        doc["loader"] in ("oldfst", "newfst") and doc["reader"] == "fastsafetensors",
        "reader",
    )
    require(
        type(doc["vocab_size"]) is int and 0 < doc["vocab_size"] <= 2**31, "vocab_size"
    )
    require(set(doc["phases"]) == set(PHASES), "phases")

    def tokens(values, lo, hi):
        return (
            type(values) is list
            and lo <= len(values) <= hi
            and all(type(v) is int and 0 <= v < doc["vocab_size"] for v in values)
        )

    for phase in PHASES:
        data = doc["phases"][phase]
        out, expected = data["output_ids"], data["expected_ids"]
        require(
            data["pd_sep"] is True
            and data["prefill_reuse_len"] == (512 if phase == "hit" else 0),
            phase + " route/reuse",
        )
        require(
            tokens(out, 2, 24)
            and tokens(expected, 1, 24)
            and data["first_token"] == out[0],
            phase + " emitted ids",
        )
        require(data["matches_golden"] is (out == expected), phase + " golden")
        rounds = data["rounds"]
        require(
            isinstance(rounds, list)
            and 0 < len(rounds) <= 24
            and data["draft_rounds"] == len(rounds),
            phase + " rounds",
        )
        offset = 1
        for i, row in enumerate(rounds):
            n = row["accept_len"]
            require(
                type(n) is int
                and 1 <= n <= 4
                and row["offset"] == offset
                and offset < len(out),
                phase + " accept/offset",
            )
            require(
                row["run_id"] == doc["run_id"] and row["phase"] == phase,
                phase + " identity",
            )
            require(
                re.fullmatch("[0-9a-f]{64}", row["history_sha256_le_i32"]) is not None,
                phase + " history",
            )
            require(
                row["committed_sha256_le_i32"] == hash_ids(out[:offset]),
                phase + " history/output",
            )
            require(
                tokens(row["draft"], 3, 3)
                and tokens(row["target"], 4, 4)
                and tokens(row["result"], n, n),
                phase + " tensor widths",
            )
            if i == 0:
                scores = row["first_verify_logits"]
                require(
                    scores["draft_id"] == row["draft"][0]
                    and scores["sampled_target_id"] == row["target"][0],
                    phase + " score columns",
                )
                require(
                    all(
                        type(scores[k]) in (float, int) and math.isfinite(scores[k])
                        for k in ("draft_id_target_logit", "sampled_id_target_logit")
                    ),
                    phase + " raw target logits",
                )
                if scores["draft_id"] == scores["sampled_target_id"]:
                    require(
                        scores["draft_id_target_logit"]
                        == scores["sampled_id_target_logit"],
                        phase + " shared score column",
                    )
            else:
                require("first_verify_logits" not in row, phase + " unexpected logits")
            require(
                row["accepted_draft_ids"]
                == row["draft"][: n - 1]
                == row["result"][: n - 1],
                phase + " accepted proposals",
            )
            consumed = min(n, len(out) - offset)
            if consumed < n:
                require(
                    i == len(rounds) - 1
                    and (out[-1] == doc["eos_id"] or len(out) == 24),
                    phase + " final truncation",
                )
            require(
                row["result"][:consumed] == out[offset : offset + consumed],
                phase + " dispatch",
            )
            offset += consumed
        require(offset == len(out), phase + " coverage")
    return doc


def compare(old, new):
    for d in (old, new):
        validate(d)
    require(old["loader"] == "oldfst" and new["loader"] == "newfst", "loader order")
    require(
        old["vocab_size"] == new["vocab_size"] and old["eos_id"] == new["eos_id"],
        "vocabulary",
    )
    result = {}
    for phase in PHASES:
        a, b = (d["phases"][phase] for d in (old, new))
        require(a["expected_ids"] == b["expected_ids"], "golden revision")
        diffs = {}
        if a["first_token"] != b["first_token"]:
            result[phase] = {
                "first_diff": {"prefill": [a["first_token"], b["first_token"]]},
                "compared_rounds": 0,
            }
            continue
        count = 0
        for x, y in zip(a["rounds"], b["rounds"]):
            if (
                x["offset"] != y["offset"]
                or a["output_ids"][: x["offset"]] != b["output_ids"][: y["offset"]]
            ):
                break
            require(
                x["history_sha256_le_i32"] == y["history_sha256_le_i32"],
                phase + " mismatched native contexts",
            )
            count += 1
            if count == 1:
                p, q = x["first_verify_logits"], y["first_verify_logits"]
                p_scores = {
                    p["draft_id"]: p["draft_id_target_logit"],
                    p["sampled_target_id"]: p["sampled_id_target_logit"],
                }
                q_scores = {
                    q["draft_id"]: q["draft_id_target_logit"],
                    q["sampled_target_id"]: q["sampled_id_target_logit"],
                }
                if p_scores != q_scores:
                    diffs["first_verify_logits"] = {
                        "offset": x["offset"],
                        "old": p_scores,
                        "new": q_scores,
                        "comparable_deltas_new_minus_old": {
                            t: q_scores[t] - p_scores[t]
                            for t in p_scores.keys() & q_scores.keys()
                        },
                    }
            for field in (
                "draft",
                "target",
                "accept_len",
                "accepted_draft_ids",
                "result",
            ):
                left, right = x[field], y[field]
                # A verify target row at position j is conditioned on draft[:j].
                # Scores/IDs after the first proposal difference are not comparable.
                if field == "target":
                    common = next(
                        (
                            i
                            for i, (p, q) in enumerate(zip(x["draft"], y["draft"]))
                            if p != q
                        ),
                        3,
                    )
                    left, right = left[: common + 1], right[: common + 1]
                if left != right and field not in diffs:
                    diffs[field] = {"offset": x["offset"], "old": left, "new": right}
        result[phase] = {
            "first_diff": diffs,
            "compared_rounds": count,
            "old_matches_golden": a["matches_golden"],
            "new_matches_golden": b["matches_golden"],
        }
    return {"schema": 1, "input_ids_sha256": DIGEST, "phases": result}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: compare_v41_cache_rounds.py OLD_JSON NEW_JSON")
    try:
        print(
            json.dumps(
                compare(*(json.loads(Path(p).read_text()) for p in sys.argv[1:])),
                allow_nan=False,
            )
        )
    except (ValueError, KeyError, TypeError, OSError, struct.error) as exc:
        sys.exit("invalid paired V4.1 evidence: " + type(exc).__name__)
