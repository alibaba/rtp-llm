"""Compare the same first verify row in the dedicated Old/New target jobs."""

import json
import math
import re
import sys
from pathlib import Path

PROMPT = "d570954fff2202dfde49e873ece0fc1adcb2df1c6a9095cb98fe6601ac4645c4"
HISTORY = "564fe056b843df0ce51d71487dab3d014f749f12470e02c661bbd76307585659"


def validate(d, role):
    if (
        d["role"] != role
        or not re.fullmatch("[0-9a-f]{32}", d["nonce"])
        or d["history"] != HISTORY
        or d["prompt"] != PROMPT
        or d["prefix"] != 512
        or d["anchor"] != 1606
        or d["candidates"] != [11556, 2521, 304]
        or set(d["scores"]) != {"305", "16"}
        or not all(
            type(v) in (int, float) and math.isfinite(v) for v in d["scores"].values()
        )
        or type(d["target"]) != int
        or d["target"] != d["first_committed"]
    ):
        raise ValueError("invalid target control")


def compare(a, b):
    validate(a, "oldfst")
    validate(b, "newfst")
    if a["nonce"] != b["nonce"] or a["history"] != b["history"]:
        raise ValueError("not the same invocation and history")
    if a["target"] != 305 or a["scores"]["305"] < a["scores"]["16"]:
        raise ValueError("OldFST reference failed to reproduce historical first verify")
    return dict(
        fixed_candidates=a["candidates"],
        old=a["scores"],
        new=b["scores"],
        new_minus_old={k: b["scores"][k] - v for k, v in a["scores"].items()},
        margin_305_minus_16={
            d["role"]: d["scores"]["305"] - d["scores"]["16"] for d in (a, b)
        },
        target_ids=[a["target"], b["target"]],
        same_target=a["target"] == b["target"],
    )


if __name__ == "__main__":
    try:
        result = compare(*(json.loads(Path(p).read_bytes()) for p in sys.argv[1:]))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        reasons = {
            "invalid target control",
            "not the same invocation and history",
            "OldFST reference failed to reproduce historical first verify",
        }
        print(
            json.dumps(
                dict(
                    valid=False,
                    reason=(
                        str(exc) if str(exc) in reasons else "incomplete target control"
                    ),
                )
            )
        )
        sys.exit(2)
    print(json.dumps(result, allow_nan=False))
