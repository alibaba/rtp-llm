"""Audit the actual SiTU arguments recorded while each serving rank loads MoE."""

import json
import math


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_situ(record):
    require(
        record.get("activation") == "situ"
        and isinstance(record.get("backend"), str)
        and bool(record["backend"]),
        "missing actual SiTU backend",
    )
    gate, up = record.get("gate_beta"), record.get("up_beta")

    def positive_scale(value):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        )

    require(
        positive_scale(gate)
        and "up_beta" in record
        and (up is None or positive_scale(up)),
        "invalid SiTU scales",
    )
    kwargs = record.get("kwargs")
    require(
        isinstance(kwargs, dict)
        and {"activation_beta", "activation_linear_beta"} <= kwargs.keys(),
        "SiTU evidence requires activation_beta and activation_linear_beta",
    )
    require(
        positive_scale(kwargs["activation_beta"])
        and (
            kwargs["activation_linear_beta"] is None
            or positive_scale(kwargs["activation_linear_beta"])
        ),
        "invalid actual SiTU keyword scales",
    )
    require(
        (kwargs["activation_beta"], kwargs["activation_linear_beta"]) == (gate, up),
        "gate/up SiTU scales were swapped or dropped",
    )


def audit_situ(prefill_dir, decode_dir, *, prefill_ranks=8, decode_ranks=8):
    result = {}
    for role, directory, ranks in (
        ("prefill", prefill_dir, prefill_ranks),
        ("decode", decode_dir, decode_ranks),
    ):
        require(ranks > 0, f"invalid {role} rank count: {ranks}")
        for rank in range(ranks):
            path = directory / f"main_{rank}.log"
            records = []
            for line_number, line in enumerate(
                path.read_text(errors="replace").splitlines(), 1
            ):
                marker = "[RTP_MOE_ACTIVATION] "
                if marker not in line:
                    continue
                try:
                    item = json.loads(line.split(marker, 1)[1])
                    require(isinstance(item, dict), "SiTU evidence is not an object")
                    if item.get("activation") == "situ":
                        check_situ(item)
                        records.append(item)
                except (AssertionError, ValueError) as error:
                    raise AssertionError(f"{path}:{line_number}: {error}") from error
            require(records, f"no real SiTU configuration for {role} rank {rank}")
            result[f"{role}_{rank}"] = records
    return result
