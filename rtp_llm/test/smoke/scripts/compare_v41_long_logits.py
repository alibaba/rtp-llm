"""Compare two Aone smoke-sm10x-arm archives. Does not extract model/server logs."""

import argparse
import io
import json
import math
from zipfile import ZipFile

TARGET = "deepseek_v41_long_prefill_logits_sm100_arm"
FIXTURE = "d570954fff2202dfde49e873ece0fc1adcb2df1c6a9095cb98fe6601ac4645c4"
RECORD = "v41_long_target_logits.json"


def unique_file(archive, predicate, limit):
    entries = [e for e in archive.infolist() if predicate(e.filename)]
    if len(entries) != 1 or entries[0].file_size > limit:
        raise ValueError("Missing, duplicate, or oversized V4.1 diagnostic artifact")
    return archive.read(entries[0])


def read_record(path, role):
    suffix = "/" + TARGET + "/test.outputs/outputs.zip"
    with ZipFile(path) as outer:
        nested = unique_file(
            outer, lambda name: ("/" + name).endswith(suffix), 128 << 20
        )
    with ZipFile(io.BytesIO(nested)) as inner:
        record = json.loads(unique_file(inner, lambda name: name == RECORD, 16384))
    if not (
        record["schema"] == 1
        and record["loader"] == role
        and record["reader"] == "fastsafetensors"
        and record["input_ids_sha256"] == FIXTURE
        and record["input_len"] == 783
        and record["pd_sep"] is False
        and record["reuse_len"] == 0
    ):
        raise ValueError(f"{role}: not the expected cold, local-prefill FST fixture")
    a, b = (record["logit_1606"], record["logit_28"])
    if not all(type(x) in (int, float) and math.isfinite(x) for x in (a, b)):
        raise ValueError(f"{role}: missing or nonfinite candidate logits")
    if not all(
        type(record[k]) is int and record[k] >= 0
        for k in ("argmax", "http_first_token")
    ):
        raise ValueError(f"{role}: invalid token IDs")
    return {
        "logit_1606": a,
        "logit_28": b,
        "margin_1606_minus_28": a - b,
        "argmax": record["argmax"],
        "http_first_token": record["http_first_token"],
    }


def compare(old_path, new_path):
    old = read_record(old_path, "oldfst")
    new = read_record(new_path, "newfst")
    a, b = (r["margin_1606_minus_28"] for r in (old, new))
    return {
        "oldfst": old,
        "newfst": new,
        "candidate_order_changed": (a > 0) - (a < 0) != (b > 0) - (b < 0),
        "argmax_changed": old["argmax"] != new["argmax"],
        "http_first_token_changed": old["http_first_token"] != new["http_first_token"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oldfst", help="OldFST control Aone ZIP")
    parser.add_argument("newfst", help="NewLoader ARM Aone ZIP")
    args = parser.parse_args()
    print(json.dumps(compare(args.oldfst, args.newfst), indent=2, allow_nan=False))
