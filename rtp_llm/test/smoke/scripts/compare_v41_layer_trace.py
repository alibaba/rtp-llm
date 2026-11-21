"""Find the earliest observed V4.1 divergence in matched OldFST/NewFST ARM artifacts."""

import argparse
import io
import json
import re
from zipfile import ZipFile

from compare_v41_long_logits import FIXTURE, TARGET, compare, unique_file

RECORD = "v41_long_layer_trace.json"
RANKS = ("0", "1")
CANDIDATES = {"28", "1606"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and bool(re.fullmatch("[0-9a-f]{64}", value))


def load_trace(path, role):
    suffix = "/" + TARGET + "/test.outputs/outputs.zip"
    with ZipFile(path) as outer:
        data = unique_file(outer, lambda n: ("/" + n).endswith(suffix), 128 << 20)
    with ZipFile(io.BytesIO(data)) as inner:
        return json.loads(unique_file(inner, lambda n: n == RECORD, 2 << 20))


def validate(doc, role):
    require(
        (doc.get("schema"), doc.get("loader"), doc.get("input_ids_sha256"))
        == (2, role, FIXTURE),
        "Not the fixed 783-token " + role + " trace",
    )
    ranks = doc.get("ranks", {})
    require(set(ranks) == set(RANKS), "Need both TP ranks")
    count = ranks["0"].get("embedding", {}).get("layers")
    require(type(count) is int and 0 < count <= 256, "Invalid layer count")
    stages = [
        "embedding",
        *["layer_" + str(i) for i in range(count)],
        "hc_reduce",
        "norm",
    ]
    for k in RANKS:
        group = ranks[k]
        require(
            set(group) == set(stages) | {"lm_head"},
            f"Missing or unexpected stages on rank {k}",
        )
        for s in stages:
            row = group[s]
            require(
                (row.get("stage"), row.get("rank"), row.get("layers"))
                == (s, int(k), count),
                f"Stage identity mismatch: {k}/{s}",
            )
            shape = row.get("valid_shape")
            require(
                isinstance(shape, list)
                and bool(shape)
                and all(type(x) is int and x > 0 for x in shape),
                f"Invalid shape: {k}/{s}",
            )
            require(
                isinstance(row.get("dtype"), str)
                and bool(row["dtype"])
                and digest(row.get("positions_sha"))
                and digest(row.get("valid_sha"))
                and "last_sha" in row
                and (row["last_sha"] is None or digest(row["last_sha"])),
                f"Invalid hidden record: {k}/{s}",
            )
        head = group["lm_head"]
        require(
            (head.get("stage"), head.get("rank"), head.get("world"))
            == ("lm_head", int(k), 2)
            and digest(head.get("last_sha"))
            and digest(head.get("head_sha")),
            "Invalid native LM head witness for TP rank " + k,
        )
        require(
            all(
                isinstance(head.get(a), str) and head[a]
                for a in ("head_dtype", "hidden_dtype")
            )
            and isinstance(head.get("head_shape"), list)
            and len(head["head_shape"]) == 2
            and all(type(x) is int and x > 0 for x in head["head_shape"]),
            "Missing LM head layout",
        )
        cand = head.get("candidate_sha")
        require(
            isinstance(cand, dict)
            and set(cand) <= CANDIDATES
            and all(digest(v) for v in cand.values()),
            "Invalid candidate-weight rows",
        )
    for s in stages:
        owners = [k for k in RANKS if ranks[k][s]["last_sha"] is not None]
        require(len(owners) == 1, "Expected one owner of the last valid token for " + s)
    for token in CANDIDATES:
        require(
            sum(token in ranks[k]["lm_head"]["candidate_sha"] for k in RANKS) == 1,
            "Expected one owner of candidate head row: " + token,
        )
    return stages


def compare_traces(old, new):
    stages = validate(old, "oldfst")
    require(stages == validate(new, "newfst"), "Different layer topology")
    first_valid = first_last = None
    a, b = old["ranks"], new["ranks"]
    for s in stages:
        diffs, last_diffs = [], []
        for k in RANKS:
            x, y = a[k][s], b[k][s]
            require(
                x["positions_sha"] == y["positions_sha"]
                and (x["last_sha"] is None) == (y["last_sha"] is None),
                f"Different token selection: {k}/{s}",
            )
            layout_changed = (x["dtype"], x["valid_shape"]) != (
                y["dtype"],
                y["valid_shape"],
            )
            if layout_changed or x["valid_sha"] != y["valid_sha"]:
                diffs.append(int(k))
            if x["last_sha"] is not None and (
                layout_changed or x["last_sha"] != y["last_sha"]
            ):
                last_diffs.append(int(k))
        if first_valid is None and diffs:
            first_valid = {"stage": s, "ranks": diffs}
        if first_last is None and last_diffs:
            first_last = {"stage": s, "ranks": last_diffs}
    head = {}
    for k in RANKS:
        x, y = a[k]["lm_head"], b[k]["lm_head"]
        require(
            set(x["candidate_sha"]) == set(y["candidate_sha"]),
            "Different LM head row placement",
        )
        head[k] = {
            "weight_equal": (x["head_dtype"], x["head_shape"], x["head_sha"])
            == (y["head_dtype"], y["head_shape"], y["head_sha"]),
            "last_hidden_equal": (x["hidden_dtype"], x["last_sha"])
            == (y["hidden_dtype"], y["last_sha"]),
            "candidate_rows_equal": {
                t: x["head_dtype"] == y["head_dtype"] and x["candidate_sha"][t] == v
                for t, v in sorted(y["candidate_sha"].items())
            },
        }
    return {
        "first_valid_rows_difference": first_valid,
        "first_last_token_difference": first_last,
        "lm_head_by_rank": head,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oldfst")
    parser.add_argument("newfst")
    args = parser.parse_args()
    logits = compare(args.oldfst, args.newfst)
    stages = compare_traces(
        load_trace(args.oldfst, "oldfst"), load_trace(args.newfst, "newfst")
    )
    print(json.dumps({"prefill": logits, **stages}, indent=2, allow_nan=False))
