"""Versioned offline traffic sources. No source participates in live sending."""

import hashlib
import json
from pathlib import Path

from .synthetic_trace import write_trace


def _families(path, parameters, namespace, base_dir):
    return write_trace(path, parameters, namespace)


def _recorded(path, parameters, namespace, base_dir):
    if (
        set(parameters) != {"path", "sha256", "identity"}
        or parameters["identity"] != "namespace"
    ):
        raise ValueError(
            "recorded trace requires path, sha256 and explicit namespace identity policy"
        )
    source = (Path(base_dir) / parameters["path"]).resolve()
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != parameters["sha256"]:
        raise ValueError("recorded trace checksum mismatch")
    with Path(path).open("w") as output:
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            original = row.get("rid")
            if not isinstance(original, str) or not original:
                raise ValueError("recorded trace requires explicit string rid")
            row["original_rid"] = original
            row["rid"] = namespace + ":" + original
            output.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    return Path(path)


# Trusted code registry; YAML cannot import modules or executable expressions.
SOURCES = {
    ("synthetic", "prefix_families", "1"): _families,
    ("trace", "recorded", "1"): _recorded,
}


def validate_plan(path):
    """Validate the canonical token plan; Java remains authority for supplied KV keys."""
    ids, previous, count = set(), -1, 0
    with Path(path).open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) & {"request_id", "request_id_int"}:
                raise ValueError(
                    "canonical plans use rid only; alternate IDs bypass namespace isolation"
                )
            if (
                "bh" in row
                and "block_cache_keys" in row
                and row["bh"] != row["block_cache_keys"]
            ):
                raise ValueError("conflicting cache key fields")
            rid, ts = row.get("rid"), row.get("ts")
            tokens = row.get("input_ids")
            if not isinstance(rid, str) or not rid or rid in ids:
                raise ValueError("request identities must be nonempty and unique")
            if type(ts) is not int or not previous <= ts <= 9223372036854775807:
                raise ValueError(
                    "arrival times must be ordered nonnegative int64 milliseconds"
                )
            if ts < 0 or not isinstance(tokens, list) or not tokens:
                raise ValueError("canonical arrival and exact input_ids required")
            if any(type(t) is not int or not 0 <= t <= 2147483647 for t in tokens):
                raise ValueError("invalid token id")
            if type(row.get("il")) is not int or row["il"] != len(tokens):
                raise ValueError("input length differs from tokens")
            if type(row.get("ol")) is not int or not 1 <= row["ol"] <= 2147483647:
                raise ValueError("invalid output length")
            if type(row.get("priority")) is not int or not 1 <= row["priority"] <= 100:
                raise ValueError("invalid priority")
            if (
                type(row.get("cache_key_block_size")) is not int
                or row["cache_key_block_size"] != 1024
            ):
                raise ValueError("unsupported cache block size")
            # Untrusted supplied keys must not bypass the Java token hash check.
            for key in ("bh", "block_cache_keys"):
                if key in row and not isinstance(row[key], list):
                    raise ValueError("cache keys must be an array")
            ids.add(rid)
            previous = ts
            count += 1
    if not count:
        raise ValueError("empty request plan")
    return count


def materialize(path, specification, namespace, base_dir):
    if not isinstance(specification, dict) or set(specification) != {
        "kind",
        "model",
        "version",
        "parameters",
    }:
        raise ValueError("source requires kind, model, version and parameters")
    identity = tuple(specification[k] for k in ("kind", "model", "version"))
    if not all(isinstance(v, str) for v in identity) or identity not in SOURCES:
        raise ValueError("unknown source model/version")
    path = Path(path)
    if path.exists():
        raise ValueError("request plan destination must be fresh")
    temporary = path.with_suffix(path.suffix + ".pending")
    try:
        SOURCES[identity](temporary, specification["parameters"], namespace, base_dir)
        count = validate_plan(temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        temporary.replace(path)
        manifest = dict(
            schema_version=1,
            source=specification,
            base_dir=str(Path(base_dir).resolve()),
            namespace=namespace,
            sha256=digest,
            request_count=count,
            reproducibility="DETERMINISTIC_INPUT",
            realism="NOT_VALIDATED",
            key_validation="Java token hashing before sending",
        )
        path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False)
        )
        return path
    finally:
        temporary.unlink(missing_ok=True)
        temporary.with_suffix(".manifest.json").unlink(missing_ok=True)
