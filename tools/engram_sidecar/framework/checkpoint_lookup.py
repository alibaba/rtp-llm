#!/usr/bin/env python3
"""Validate real checkpoint Engram tables through the RTP Triton lookup path.

prepare: CPU-only extraction of a few raw FP8/E8M0 rows from the real Owner
checkpoint, bound to its private publication and real tokenizer/hash constants.
run: import the real allocation once, verify bootstrap samples, run eager
HostEngramEmbedding/Triton lookup, and compare exact BF16 against an independent
CPU byte decoder. This does not run projections, the complete model or HTTP.

The operator transfers bundle/manifest/capabilities and oracle through an
authenticated private channel and keeps Owner alive until this process exits.
No capability or raw oracle row is printed. Engine CUDA Graph is not exercised.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
from uuid import UUID, uuid4


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
MAX_PRIVATE_BYTES = 4 << 20
MAX_SELECTED_ROWS = 512
KIND = "real-checkpoint-engram-lookup-v1"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
PROMPT = "你好，请用一句话介绍自己。"


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def emit(status, **values):
    print(json.dumps(dict(status=status, scope="real_checkpoint_lookup_only",
                          full_model_http=False, **values), sort_keys=True, allow_nan=False), flush=True)


def unique_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("non-finite JSON value")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def private_read(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in [path] + list(path.parents)):
        raise ValueError("private path must be absolute without symlinks")
    parent = path.parent.stat()
    if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o077:
        raise ValueError("private parent must be owned by current UID and mode 0700")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077 or info.st_size > MAX_PRIVATE_BYTES):
            raise ValueError("unsafe or oversized private input")
        raw = stream.read(MAX_PRIVATE_BYTES + 1)
    if len(raw) > MAX_PRIVATE_BYTES:
        raise ValueError("private input exceeds limit")
    return unique_json(raw)


def private_write_new(path, document):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in [path] + list(path.parents)):
        raise ValueError("private output must be absolute without symlinks")
    if not path.parent.exists():
        path.parent.mkdir(mode=0o700)
    parent = path.parent.stat()
    if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o077:
        raise ValueError("private output parent is not private/current-UID")
    raw = json.dumps(document, sort_keys=True, allow_nan=False).encode()
    if len(raw) > MAX_PRIVATE_BYTES:
        raise ValueError("oracle exceeds bounded private input limit")
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and refuses an existing output.
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_config(path, require_no_tables=False):
    from tools.engram_sidecar.framework.stage_checkpoint import _canonical, _small_file, _header, _snapshot

    checkpoint = _canonical(path)
    if not checkpoint.is_absolute() or not checkpoint.is_dir():
        raise ValueError("existing absolute checkpoint required")
    config_raw, _ = _small_file(checkpoint / "config.json", 1 << 20)
    raw = unique_json(config_raw)
    if not isinstance(raw, dict) or raw.get("fixture_component_only"):
        raise ValueError("a real model checkpoint, not a component fixture, is required")
    config = {**raw, **raw.get("text_config", {})}
    config.pop("text_config", None)
    config["checkpoint_path"] = str(checkpoint)
    tokenizer_raw, _ = _small_file(checkpoint / "tokenizer.json", 16 << 20)
    index_raw, _ = _small_file(checkpoint / "model.safetensors.index.json", 32 << 20)
    index = unique_json(index_raw)["weight_map"]
    if not isinstance(index, dict) or not index:
        raise ValueError("checkpoint index is empty")
    if require_no_tables:
        if any(".engram.embed." in name for name in index):
            raise ValueError("consumer index still contains Engram tables")
        filenames = set(index.values())
        if {p.name for p in checkpoint.glob("*.safetensors")} != filenames:
            raise ValueError("unindexed consumer tensor files")
        for filename in filenames:
            if Path(filename).name != filename or not filename.endswith(".safetensors"):
                raise ValueError("unsafe checkpoint shard name")
            path = checkpoint / filename
            header, _, _ = _header(path, _snapshot(path))
            names = set(header) - {"__metadata__"}
            if names != {name for name, shard in index.items() if shard == filename}:
                raise ValueError("consumer header/index mismatch")
            if any(".engram.embed." in name for name in names):
                raise ValueError("consumer tensor file contains an Engram table")
    return config, {"config_sha256": digest(config_raw), "tokenizer_sha256": digest(tokenizer_raw)}, index


def load_package(shard_json, checkpoint):
    """Build the in-memory package from published shard descriptors."""
    from rtp_llm.models_py.modules.dsv41.engram_startup import package_from_descriptors

    return package_from_descriptors([str(path) for path in shard_json], str(checkpoint))


def package_identity(package):
    from rtp_llm.models_py.modules.dsv41.engram_manifest import parse_manifest

    manifest = parse_manifest(package["raw"])
    result = {"dataset_id": package["dataset_id"], "generation": package["generation"],
              "model_revision": package["model_revision"], "hash_abi": manifest.hash_abi,
              "manifest_sha256": digest(package["raw"])}
    for key in ("dataset_id", "hash_abi", "manifest_sha256"):
        if not isinstance(result[key], str) or not DIGEST.fullmatch(result[key]):
            raise ValueError("invalid package digest")
    if str(UUID(result["generation"])) != result["generation"]:
        raise ValueError("invalid package generation")
    if not isinstance(result["model_revision"], str) or not 0 < len(result["model_revision"]) <= 512:
        raise ValueError("invalid model revision")
    return result


def windows_for_tokens(tokens, depth):
    if not tokens or len(tokens) > 16 or type(depth) is not int or not 2 <= depth <= 8:
        raise ValueError("bounded token list and window depth required")
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("invalid tokenizer ID")
    return [[tokens[i - j] if i >= j else -1 for j in range(depth)] for i in range(len(tokens))]


def boundary_ids(rows, dim):
    if type(rows) is not int or rows < 2 or type(dim) is not int or dim < 32:
        raise ValueError("invalid table geometry")
    selected = [0, 1, rows // 2, rows - 1]
    boundary = (1 << 32) // dim
    selected += [row for row in (boundary - 1, boundary, boundary + 1) if 0 <= row < rows]
    return list(dict.fromkeys(selected)) + [-1, 0]


def layout_metadata(layout):
    return {"layers": [{"layer_id": layer, "rows": rows} for layer, rows in zip(layout.layer_ids, layout.num_embeddings)],
            "head_dim": layout.head_dim, "hash_columns": layout.n_hash_cols,
            "window_depth": layout.max_ngram_size,
            "logical_table_bytes": sum(rows * (layout.head_dim + layout.head_dim // 32) for rows in layout.num_embeddings)}


def selected_ids(cases, layout):
    result = {str(layer): set() for layer in layout.layer_ids}
    limits = dict(zip(map(str, layout.layer_ids), layout.num_embeddings))
    if not isinstance(cases, list) or not 1 <= len(cases) <= 8:
        raise ValueError("invalid case count")
    names = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"name", "kind", "token_windows", "ids"}:
            raise ValueError("invalid case fields")
        if not isinstance(case["name"], str) or not re.fullmatch(r"[a-z0-9_]{1,64}", case["name"]) or case["name"] in names:
            raise ValueError("invalid/duplicate case name")
        names.add(case["name"])
        if case["kind"] not in ("direct", "hash") or not isinstance(case["ids"], dict) or set(case["ids"]) != set(result):
            raise ValueError("case layers or kind mismatch")
        windows = case["token_windows"]
        if case["kind"] == "hash":
            if (not isinstance(windows, list) or not 1 <= len(windows) <= 16
                    or any(not isinstance(row, list) or len(row) != layout.max_ngram_size
                           or any(type(x) is not int or x < -1 for x in row) for row in windows)):
                raise ValueError("invalid token windows")
        elif windows is not None:
            raise ValueError("direct case cannot supply token windows")
        for layer, matrix in case["ids"].items():
            if not isinstance(matrix, list) or not 1 <= len(matrix) <= 16:
                raise ValueError("invalid ID matrix")
            columns = len(matrix[0]) if isinstance(matrix[0], list) else 0
            if not 1 <= columns <= 128 or any(not isinstance(row, list) or len(row) != columns for row in matrix):
                raise ValueError("ragged or oversized ID matrix")
            if case["kind"] == "hash" and (len(matrix) != len(windows) or columns != layout.n_hash_cols):
                raise ValueError("hash result shape mismatch")
            for row in matrix:
                for value in row:
                    if type(value) is not int or not -1 <= value < limits[layer]:
                        raise ValueError("row ID out of bounds")
                    if value >= 0:
                        result[layer].add(value)
            if len(result[layer]) > MAX_SELECTED_ROWS:
                raise ValueError("selected-row budget exceeded")
    return result


def fp8_e4m3_value(raw):
    if type(raw) is not int or not 0 <= raw <= 255:
        raise ValueError("FP8 byte out of bounds")
    sign, exponent, mantissa = (-1.0 if raw & 128 else 1.0), (raw >> 3) & 15, raw & 7
    if exponent == 15 and mantissa == 7:
        raise ValueError("non-finite FP8 checkpoint sample")
    return sign * (math.ldexp(mantissa, -9) if exponent == 0 else math.ldexp(1.0 + mantissa / 8.0, exponent - 7))


def decode_row(weight, scale):
    """Independent IEEE decoder matching the reference E8M0 exponent-bit rule.

    Byte zero denotes FP32 zero in the RTP lookup's bit interpretation. No GPU,
    Triton, production dequantizer or FP8 Torch conversion is used here.
    """
    if not weight or len(weight) % 32 or len(scale) != len(weight) // 32:
        raise ValueError("row geometry mismatch")
    scales = [struct.unpack("<f", struct.pack("<I", byte << 23))[0] for byte in scale]
    if not all(math.isfinite(value) for value in scales):
        raise ValueError("non-finite E8M0 checkpoint sample")
    return [fp8_e4m3_value(value) * scales[column // 32] for column, value in enumerate(weight)]


def validate_oracle(document, *, layout, metadata, identity, token_count=None):
    fields = {"schema_version", "kind", "identity", "checkpoint_metadata", "layout", "cases", "rows", "source_headers"}
    if not isinstance(document, dict) or set(document) != fields or type(document["schema_version"]) is not int or document["schema_version"] != 1 or document["kind"] != KIND:
        raise ValueError("invalid oracle schema")
    if (document["identity"] != identity or document["checkpoint_metadata"] != metadata
            or json.dumps(document["layout"], sort_keys=True, allow_nan=False) != json.dumps(layout_metadata(layout), sort_keys=True)):
        raise ValueError("oracle model/generation/hash/layout mismatch")
    expected = selected_ids(document["cases"], layout)
    if token_count is not None:
        for case in document["cases"]:
            if case["kind"] == "hash" and any(token >= token_count for row in case["token_windows"] for token in row):
                raise ValueError("token window ID exceeds tokenizer vocabulary")
    if not isinstance(document["rows"], dict) or set(document["rows"]) != set(expected):
        raise ValueError("oracle layers mismatch")
    if not isinstance(document["source_headers"], dict) or not 1 <= len(document["source_headers"]) <= 32:
        raise ValueError("source header evidence missing")
    for name, sha in document["source_headers"].items():
        if Path(name).name != name or not isinstance(sha, str) or not DIGEST.fullmatch(sha):
            raise ValueError("invalid source header evidence")
    parsed = {}
    for layer, records in document["rows"].items():
        if not isinstance(records, list) or len(records) > MAX_SELECTED_ROWS:
            raise ValueError("oversized oracle rows")
        parsed[layer] = {}
        for row in records:
            if not isinstance(row, dict) or set(row) != {"row_id", "fp8_hex", "e8m0_hex"} or type(row["row_id"]) is not int or row["row_id"] in parsed[layer]:
                raise ValueError("invalid or duplicate oracle row")
            if not isinstance(row["fp8_hex"], str) or len(row["fp8_hex"]) != 2 * layout.head_dim or not isinstance(row["e8m0_hex"], str) or len(row["e8m0_hex"]) != 2 * (layout.head_dim // 32):
                raise ValueError("oracle row byte count mismatch")
            weight, scale = bytes.fromhex(row["fp8_hex"]), bytes.fromhex(row["e8m0_hex"])
            if len(weight) != layout.head_dim or len(scale) != layout.head_dim // 32:
                raise ValueError("noncanonical oracle hex")
            decode_row(weight, scale)
            parsed[layer][row["row_id"]] = (weight, scale)
        if set(parsed[layer]) != expected[layer]:
            raise ValueError("oracle has missing or unrequested rows")
    return parsed


def verify_package_samples(parsed, package, layout):
    for layer, rows in zip(layout.layer_ids, layout.num_embeddings):
        selected = sorted({0, rows // 2, rows - 1})
        for position, role in enumerate(("weight", "scale")):
            sample = package["samples"]["layers.{}.engram.embed.{}".format(layer, role)]
            if sample["row_ids"] != selected or digest(b"".join(parsed[str(layer)][row][position] for row in selected)) != sample["sha256"]:
                raise ValueError("oracle/source differs from published bootstrap samples")


def read_selected_rows(checkpoint, index, layout, selected):
    from tools.engram_sidecar.framework.stage_checkpoint import _basename, _snapshot, _header, _source_file

    checkpoint, headers, records = Path(checkpoint), {}, {}
    for layer, count in zip(layout.layer_ids, layout.num_embeddings):
        if len(selected[str(layer)]) > MAX_SELECTED_ROWS or any(type(row) is not int or not 0 <= row < count for row in selected[str(layer)]):
            raise ValueError("invalid bounded source row selection")
        by_role = {}
        for role, width, dtypes in (("weight", layout.head_dim, {"F8_E4M3", "F8_E4M3FN"}),
                                    ("scale", layout.head_dim // 32, {"F8_E8M0", "U8"})):
            name = "layers.{}.engram.embed.{}".format(layer, role)
            filename = _basename(index[name])
            path = checkpoint / filename
            snapshot = _snapshot(path)
            header, data_start, header_sha = _header(path, snapshot)
            tensor = header[name]
            if tensor["dtype"] not in dtypes or tensor["shape"] != [count, width]:
                raise ValueError("source table dtype/shape mismatch")
            headers[filename] = "sha256:" + header_sha
            by_role[role] = {}
            with _source_file(path, snapshot) as stream:
                for row in sorted(selected[str(layer)]):
                    stream.seek(data_start + tensor["data_offsets"][0] + row * width)
                    raw = stream.read(width)
                    if len(raw) != width:
                        raise ValueError("short selected row read")
                    by_role[role][row] = raw
        records[str(layer)] = [{"row_id": row, "fp8_hex": by_role["weight"][row].hex(),
                                "e8m0_hex": by_role["scale"][row].hex()} for row in sorted(selected[str(layer)])]
    return records, headers


def prepare(args):
    # Oracle extraction is CPU-only, even if the launcher normally exposes GPUs.
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise ValueError("prepare must start without an initialized CUDA context")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoTokenizer
    from rtp_llm.models_py.modules.dsv41.engram import EngramLayout, NgramHashState
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import hash_state_digest

    config, metadata, index = checkpoint_config(args.checkpoint)
    package = load_package(args.shard_json, args.checkpoint)
    identity = package_identity(package)
    layout = EngramLayout(config)
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), trust_remote_code=True, local_files_only=True)
    state = NgramHashState(layout, tokenizer=tokenizer)
    if hash_state_digest(state) != identity["hash_abi"]:
        raise ValueError("source tokenizer/hash differs from Owner publication")
    tokens = tokenizer.encode(PROMPT, add_special_tokens=False)[:8]
    windows = windows_for_tokens(tokens, layout.max_ngram_size)
    cases = [{"name": "boundary_and_4gib", "kind": "direct", "token_windows": None,
              "ids": {str(layer): [boundary_ids(rows, layout.head_dim)] for layer, rows in zip(layout.layer_ids, layout.num_embeddings)}}]
    for name, selected_windows in (("tokenizer_prefill", windows), ("decode_history", windows[-1:])):
        hashed = state.hash_token_windows(torch.tensor(selected_windows, dtype=torch.int64))
        cases.append({"name": name, "kind": "hash", "token_windows": selected_windows,
                      "ids": {str(layer): hashed[:, i].tolist() for i, layer in enumerate(layout.layer_ids)}})
    rows, headers = read_selected_rows(args.checkpoint, index, layout, selected_ids(cases, layout))
    document = {"schema_version": 1, "kind": KIND, "identity": identity,
                "checkpoint_metadata": metadata, "layout": layout_metadata(layout),
                "cases": cases, "rows": rows, "source_headers": headers}
    parsed = validate_oracle(document, layout=layout, metadata=metadata, identity=identity, token_count=len(state.token_map))
    verify_package_samples(parsed, package, layout)
    if torch.cuda.is_initialized():
        raise AssertionError("CPU oracle preparation initialized CUDA")
    private_write_new(args.private_inputs, document)
    emit("ORACLE_PREPARED", logical_table_bytes=document["layout"]["logical_table_bytes"],
         selected_rows={layer: len(values) for layer, values in parsed.items()},
         cases=len(cases), cuda_initialized=False, published_samples_matched=True)


def run(args):
    config, metadata, _ = checkpoint_config(args.checkpoint, require_no_tables=True)
    package, oracle = load_package(args.shard_json, args.checkpoint), private_read(args.private_inputs)
    identity = package_identity(package)
    import torch
    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise ValueError("authorized CUDA device is not available")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    from rtp_llm.models_py.modules.dsv41.engram import EngramLayout, NgramHashState, HostEngramEmbedding
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import attach_from_package, hash_state_digest

    layout = EngramLayout(config)
    state = NgramHashState(layout, str(args.checkpoint))
    if hash_state_digest(state) != identity["hash_abi"]:
        raise ValueError("consumer tokenizer/hash mismatch")
    parsed = validate_oracle(oracle, layout=layout, metadata=metadata, identity=identity, token_count=len(state.token_map))
    verify_package_samples(parsed, package, layout)
    spec = importlib.util.find_spec("engram_native_dev_492ecf45")
    if spec is None or not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
        raise RuntimeError("prebuilt native bridge required; no Python shim")
    native = importlib.import_module("engram_native_dev_492ecf45")
    if any(native.resource_counters().values()):
        raise RuntimeError("lookup process must begin with zero native resources")
    storage, embeddings, records = None, {}, []
    try:
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        storage = attach_from_package(package, layout=layout, hash_state=state, device=device)
        after = torch.cuda.memory_allocated(device)
        info = storage.info()
        for layer in layout.layer_ids:
            embedding = HostEngramEmbedding.from_storage(storage, layer)
            if embedding.weight is not None or embedding.scales is not None:
                raise AssertionError("consumer allocated private complete tables")
            embeddings[str(layer)] = embedding
        with torch.inference_mode():
            for case in oracle["cases"]:
                if case["kind"] == "hash":
                    windows = torch.tensor(case["token_windows"], dtype=torch.int64, device=device)
                    gpu_hash = state.hash_token_windows(windows).cpu()
                    for i, layer in enumerate(layout.layer_ids):
                        torch.testing.assert_close(gpu_hash[:, i], torch.tensor(case["ids"][str(layer)], dtype=torch.int64), rtol=0, atol=0)
                for layer in layout.layer_ids:
                    key = str(layer)
                    matrix = case["ids"][key]
                    reference = [[decode_row(*parsed[key][row]) if row >= 0 else [0.0] * layout.head_dim for row in batch] for batch in matrix]
                    expected = torch.tensor(reference, dtype=torch.float32).to(torch.bfloat16)
                    if not torch.isfinite(expected).all():
                        raise AssertionError("non-finite CPU reference")
                    for dtype in (torch.int64, torch.int32):
                        if dtype == torch.int32 and max(max(row) for row in matrix) >= 1 << 31:
                            continue
                        indices = torch.tensor(matrix, dtype=dtype, device=device)
                        output = embeddings[key](indices, device)
                        actual = output.cpu()
                        if actual.dtype != torch.bfloat16 or not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
                            raise AssertionError("real checkpoint lookup differs from independent BF16 oracle")
                        records.append({"case": case["name"], "layer": layer, "index_dtype": str(dtype),
                                        "shape": list(actual.shape), "exact_bf16_bits": True})
            torch.cuda.synchronize(device)
        emit("PASS", logical_table_bytes=oracle["layout"]["logical_table_bytes"],
             physical_bytes=info["physical_bytes"], allocation_count=info["allocation_count"],
             dataset_id=info["dataset_id"], generation=info["generation"],
             bootstrap_first_middle_last_verified=True, no_consumer_checkpoint_tables=True,
             torch_attach_allocated_bytes=after - before,
             torch_peak_increment_bytes=torch.cuda.max_memory_allocated(device) - before,
             cases=records)
    finally:
        torch.cuda.synchronize(device)
        for embedding in embeddings.values():
            embedding.release_external_views()
        embeddings.clear()
        gc.collect()
        if storage is not None:
            storage.close()
        counters = native.resource_counters()
        if any(counters.values()):
            raise AssertionError("native lookup resources remain")
        emit("CLOSED", counters=counters)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shard-json", type=Path, action="append", required=True)
    parser.add_argument("--private-inputs", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    if any(not path.is_absolute() for path in (args.checkpoint, args.private_inputs)) or args.device < 0:
        parser.error("absolute input paths and nonnegative device required")
    try:
        (prepare if args.action == "prepare" else run)(args)
        return 0
    except BaseException as error:
        emit("FAILED", error_type=type(error).__name__, code=getattr(error, "code", None))
        return 1


if __name__ == "__main__":
    sys.exit(main())
