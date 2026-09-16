"""Reproducible offline request plans; Java hashes and sends the exact tokens."""

import hashlib
import json
import random
from pathlib import Path


def write_trace(path, specification, namespace):
    required = {"seed", "count", "interval_ms", "block_size", "families"}
    if set(specification) != required:
        raise ValueError("trace requires exactly " + str(sorted(required)))
    if (
        type(specification["seed"]) is not int
        or type(specification["count"]) is not int
        or specification["count"] < 1
    ):
        raise ValueError("trace seed/count must be explicit integers, count positive")
    if (
        type(specification["interval_ms"]) is not int
        or specification["interval_ms"] < 0
    ):
        raise ValueError("trace interval_ms must be nonnegative integer")
    if specification["block_size"] != 1024:
        raise ValueError("Java client currently supports only 1024-token cache blocks")
    families = specification["families"]
    if not isinstance(families, list) or not families:
        raise ValueError("trace needs at least one request family")
    for family in families:
        if set(family) != {
            "name",
            "prefix_tokens",
            "suffix_length",
            "token_max",
            "output_len",
            "priority",
        }:
            raise ValueError("trace family fields must be explicit")
        if not family["name"] or not isinstance(family["prefix_tokens"], list):
            raise ValueError("invalid family name or prefix")
        if any(
            type(x) is not int or x < 0 or x > 2147483647
            for x in family["prefix_tokens"]
        ):
            raise ValueError("prefix must contain exact nonnegative int32 token ids")
        for key in ("suffix_length", "token_max", "output_len", "priority"):
            if type(family[key]) is not int:
                raise ValueError("family numeric fields must be integers")
        if (
            family["suffix_length"] < 0
            or not 1 <= family["token_max"] <= 2147483647
            or family["output_len"] < 1
            or not 1 <= family["priority"] <= 100
        ):
            raise ValueError("invalid family request shape")
        if len(family["prefix_tokens"]) + family["suffix_length"] <= 0:
            raise ValueError("empty input is not replayable")
    rng = random.Random(specification["seed"])
    path = Path(path)
    with path.open("w") as out:
        for index in range(specification["count"]):
            family = families[index % len(families)]
            tokens = list(family["prefix_tokens"]) + [
                rng.randrange(family["token_max"])
                for _ in range(family["suffix_length"])
            ]
            row = dict(
                rid=f"{namespace}:{index}",
                ts=index * specification["interval_ms"],
                il=len(tokens),
                ol=family["output_len"],
                input_ids=tokens,
                priority=family["priority"],
                cache_key_block_size=specification["block_size"],
                family=family["name"],
            )
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = dict(
        specification=specification,
        namespace=namespace,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return path
