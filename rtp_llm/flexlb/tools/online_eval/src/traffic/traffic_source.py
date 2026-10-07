"""Versioned offline traffic sources. No source participates in live sending."""

import hashlib
import json
from pathlib import Path

from traffic.realistic import write_trace as write_realistic
from traffic.prefix_lineage import write_trace as write_lineage
from traffic.prefix_lineage_v3 import write_trace as write_lineage_v3


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Only these versioned sources participate in scenario materialization.
SOURCES = {
    ("synthetic", "realistic", "1"): write_realistic,
    ("trace", "prefix_lineage", "2"): write_lineage,
    ("trace", "prefix_lineage", "3"): write_lineage_v3,
}


def validate_plan(path, namespace=None):
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
            compact = "input_token_blocks" in row
            if compact:
                if "input_ids" in row:
                    raise ValueError("ambiguous token encoding")
                tokens = row["input_token_blocks"]
            if not isinstance(rid, str) or not rid or rid in ids:
                raise ValueError("request identities must be nonempty and unique")
            if type(ts) is not int or not previous <= ts <= 9223372036854775807:
                raise ValueError(
                    "arrival times must be ordered nonnegative int64 milliseconds"
                )
            if ts < 0 or not isinstance(tokens, list) or not tokens:
                raise ValueError("canonical arrival and exact input_ids required")
            block_size = row.get("cache_key_block_size")
            def valid_token(t):
                return type(t) is int and 0 <= t <= 2147483647
            if any(not (valid_token(t) or (compact and isinstance(t,list) and len(t)==block_size and all(valid_token(v) for v in t))) for t in tokens):
                raise ValueError("invalid token id")
            block_size = row.get("cache_key_block_size")
            if type(row.get("il")) is not int or not 1 <= row["il"] <= 2147483647:
                raise ValueError("invalid input length")
            if compact and (type(block_size) is not int or block_size not in (64,128,256,512,1024,2048,4096)):
                raise ValueError("invalid compact block size")
            expected_count = (row["il"] + block_size - 1) // block_size if compact else row["il"]
            if expected_count != len(tokens):
                raise ValueError("input length differs from tokens")
            if type(row.get("ol")) is not int or not 1 <= row["ol"] <= 2147483647:
                raise ValueError("invalid output length")
            if type(row.get("priority")) is not int or not 1 <= row["priority"] <= 100:
                raise ValueError("invalid priority")
            if type(row.get("cache_key_block_size")) is not int or row[
                "cache_key_block_size"
            ] not in (64, 128, 256, 512, 1024, 2048, 4096):
                raise ValueError("unsupported cache block size")
            # Untrusted supplied keys must not bypass the Java token hash check.
            for key in ("bh", "block_cache_keys"):
                if key in row and not isinstance(row[key], list):
                    raise ValueError("cache keys must be an array")
            if namespace is not None and not rid.startswith(namespace + ":"):
                raise ValueError("canonical trace request namespace mismatch")
            ids.add(rid)
            previous = ts
            count += 1
    if not count:
        raise ValueError("empty request plan")
    return count


def materialize(path, specification, namespace, base_dir, *, max_requests=None):
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
    if max_requests is not None and (type(max_requests) is not int or max_requests < 1):
        raise ValueError("max_requests must be a positive integer")
    path = Path(path)
    if path.exists():
        raise ValueError("request plan destination must be fresh")
    temporary = path.with_suffix(path.suffix + ".pending")
    try:
        semantics = SOURCES[identity](temporary, specification["parameters"], namespace, base_dir,
                                      max_requests=max_requests)
        count = validate_plan(temporary, namespace)
        digest = sha256_file(temporary)
        temporary.replace(path)
        manifest = dict(
            schema_version=1,
            source=specification,
            base_dir=str(Path(base_dir).resolve()),
            namespace=namespace,
            sha256=digest,
            request_count=count,
            projection=dict(max_requests=max_requests, selected_requests=count),
            reproducibility="DETERMINISTIC_INPUT",
            **semantics,
            key_validation="Java token hashing before sending",
        )
        path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False)
        )
        return path
    finally:
        temporary.unlink(missing_ok=True)
        temporary.with_suffix(".manifest.json").unlink(missing_ok=True)
