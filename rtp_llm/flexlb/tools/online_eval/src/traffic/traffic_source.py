"""Versioned offline traffic sources. No source participates in live sending."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from traffic.contracts import PRIORITY, JAVA_LENGTH

from traffic.synthetic.generator import write_trace as write_realistic


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Generator versions describe algorithms; capture codecs come from their data.
SOURCES = {
    ("synthetic", "realistic", "1"): write_realistic,
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
            try:
                JAVA_LENGTH.validate(row.get("il"), "il")
            except ValueError as exc:
                raise ValueError("invalid input length") from exc
            if compact and (type(block_size) is not int or block_size not in (64,128,256,512,1024,2048,4096)):
                raise ValueError("invalid compact block size")
            expected_count = (row["il"] + block_size - 1) // block_size if compact else row["il"]
            if expected_count != len(tokens):
                raise ValueError("input length differs from tokens")
            try:
                JAVA_LENGTH.validate(row.get("ol"), "ol")
            except ValueError as exc:
                raise ValueError("invalid output length") from exc
            try:
                PRIORITY.validate(row.get("priority"), "priority")
            except ValueError as exc:
                raise ValueError("invalid priority") from exc
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


@dataclass(frozen=True)
class SourceSpecification:
    kind: str
    model: str
    parameters: dict
    version: str | None = None

    @classmethod
    def read(cls, value):
        required = {"kind", "model", "parameters"}
        if (not isinstance(value, dict) or not required <= set(value)
                or set(value) - required - {"version"}):
            raise ValueError("source requires kind, model and parameters; version is an optional pin")
        result = cls(**value)
        if (type(result.kind) is not str or type(result.model) is not str
                or not isinstance(result.parameters, dict)
                or ("version" in value and type(result.version) is not str)):
            raise ValueError("invalid source specification")
        if result.is_lineage:
            if result.version is not None:
                from traffic.codecs import codec_for_version
                if not result.version.isdecimal():
                    raise ValueError("unknown source model/version")
                codec_for_version(int(result.version))
        elif (result.kind, result.model, result.version) not in SOURCES:
            raise ValueError("unknown source model/version")
        return result

    @property
    def is_lineage(self):
        return (self.kind, self.model) == ("trace", "prefix_lineage")

    def write_trace(self, path, namespace, base_dir, *, max_requests=None):
        if self.is_lineage:
            from traffic.codecs import decode_model
            from traffic.lineage_transforms import LineageProjector
            return LineageProjector(lambda raw: decode_model(raw, expected_version=self.version)).write_trace(
                path, self.parameters, namespace, base_dir, max_requests=max_requests)
        return SOURCES[(self.kind, self.model, self.version)](
            path, self.parameters, namespace, base_dir, max_requests=max_requests)


def materialize(path, specification, namespace, base_dir, *, max_requests=None):
    source = SourceSpecification.read(specification)
    if max_requests is not None and (type(max_requests) is not int or max_requests < 1):
        raise ValueError("max_requests must be a positive integer")
    path = Path(path)
    if path.exists():
        raise ValueError("request plan destination must be fresh")
    temporary = path.with_suffix(path.suffix + ".pending")
    try:
        semantics = source.write_trace(temporary, namespace, base_dir, max_requests=max_requests)
        resolved_source = dict(specification)
        if source.is_lineage:
            resolved_source["version"] = str(semantics["source_codec"]["version"])
        count = validate_plan(temporary, namespace)
        digest = sha256_file(temporary)
        temporary.replace(path)
        manifest = dict(
            request_plan_manifest_schema_version=1,
            source=resolved_source,
            requested_source=specification,
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
