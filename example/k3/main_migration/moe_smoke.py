"""Check the actual SiTU arguments selected when the serving model loads."""

import json
import math

from request_smoke import require


def check_situ(record):
    require(
        record["activation"] == "situ" and record["backend"],
        "missing actual SiTU backend",
    )
    gate, up = record["gate_beta"], record["up_beta"]
    require(
        math.isfinite(gate)
        and gate > 0
        and (up is None or (math.isfinite(up) and up > 0)),
        "invalid SiTU scales",
    )
    args = record["kwargs"]
    if {"activation_beta", "activation_linear_beta"} <= args.keys():
        mapped = args["activation_beta"], args["activation_linear_beta"]
    elif {"situ_beta", "situ_linear_beta"} <= args.keys():
        mapped = args["situ_beta"], args["situ_linear_beta"]
    else:
        require(
            {"activation_alpha", "activation_beta"} <= args.keys(),
            "unsupported SiTU interface",
        )
        mapped = args["activation_alpha"], args["activation_beta"] or None
    require(mapped == (gate, up), "gate/up SiTU scales were swapped or dropped")


def audit_situ(prefill_dir, decode_dir):
    result = {}
    for role, directory in (("prefill", prefill_dir), ("decode", decode_dir)):
        for rank in range(8):
            records = []
            for line in (
                (directory / f"main_{rank}.log")
                .read_text(errors="replace")
                .splitlines()
            ):
                if "[RTP_MOE_ACTIVATION] " in line:
                    item = json.loads(line.split("[RTP_MOE_ACTIVATION] ", 1)[1])
                    if item["activation"] == "situ":
                        check_situ(item)
                        records.append(item)
            require(records, f"no real SiTU configuration for {role} rank {rank}")
            result[f"{role}_{rank}"] = records
    return result
