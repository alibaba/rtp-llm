#!/usr/bin/env python3
"""Stage an owner/consumer split without loading tensors (Python >= 3.6).

Default is a read-only plan. --execute writes to DEST.staging and publishes via
an atomic directory rename only after every file has passed verification.
Consumer mixed shards are rebuilt from selected byte ranges; Engram tables are
never read/copied into that destination. This is a checkpoint preparation tool,
not a downloader, model loader or CUDA program.
"""
import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import sys
import time


INDEX = "model.safetensors.index.json"
SCHEMA = 1
BUFFER = 8 * 1024 * 1024
MAX_JSON = 64 * 1024 * 1024
MAX_HEADER = 32 * 1024 * 1024
DEFAULT_METADATA = (
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "tokenizer.model", "vocab.json", "merges.txt", "chat_template.jinja",
    "encoding.py", "encoding/encoding.py", "LICENSE", "README.md",
)
EMBED = re.compile(r"(?:^|\.)engram\.embed\.(weight|scale)$")
DTYPE_BITS = {
    "BOOL": 8, "U8": 8, "I8": 8, "I16": 16, "U16": 16,
    "I32": 32, "U32": 32, "I64": 64, "U64": 64,
    "F16": 16, "BF16": 16, "F32": 32, "F64": 64,
    "F8_E4M3": 8, "F8_E4M3FN": 8, "F8_E5M2": 8, "F8_E8M0": 8,
    "F8_E4M3FNUZ": 8, "F8_E5M2FNUZ": 8, "F4": 4,
}


class StageError(RuntimeError):
    pass


def _fail(message):
    raise StageError(message)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate JSON key: " + key)
        result[key] = value
    return result


def _decode(raw):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _basename(name):
    if not isinstance(name, str) or not name or name in (".", "..") or "/" in name or "\\" in name or "\0" in name:
        _fail("only plain filenames are allowed: " + repr(name))
    return name


def _metadata_name(name):
    if not isinstance(name, str) or len(name.split("/")) > 2:
        _fail("metadata path is limited to one subdirectory")
    for part in name.split("/"):
        _basename(part)
    return name


def _canonical(path):
    path = Path(os.path.abspath(str(path)))
    # Reject every symlink component, including a dangling destination link.
    for ancestor in [path] + list(path.parents):
        if ancestor.is_symlink():
            _fail("symlink path is not allowed: " + str(ancestor))
    if str(path.resolve()) != str(path):
        _fail("non-canonical path: " + str(path))
    return path


def _snapshot(path):
    s = os.stat(str(path), follow_symlinks=False)
    if not stat.S_ISREG(s.st_mode):
        _fail("source is not a regular file: " + str(path))
    return {"device": s.st_dev, "inode": s.st_ino, "bytes": s.st_size,
            "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns}


def _unchanged(path, expected):
    if _snapshot(path) != expected:
        _fail("source changed: " + str(path))


@contextlib.contextmanager
def _source_file(path, expected):
    _unchanged(path, expected)
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        s = os.fstat(stream.fileno())
        actual = {"device": s.st_dev, "inode": s.st_ino, "bytes": s.st_size,
                  "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns}
        if actual != expected:
            _fail("source changed during open: " + str(path))
        yield stream
        s = os.fstat(stream.fileno())
        actual = {"device": s.st_dev, "inode": s.st_ino, "bytes": s.st_size,
                  "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns}
        if actual != expected:
            _fail("open source changed while reading: " + str(path))
    _unchanged(path, expected)


def _small_file(path, maximum=MAX_JSON):
    snapshot = _snapshot(path)
    if snapshot["bytes"] > maximum:
        _fail("metadata exceeds bounded read limit: " + str(path))
    with _source_file(path, snapshot) as stream:
        raw = stream.read(maximum + 1)
    if len(raw) != snapshot["bytes"]:
        _fail("short metadata read: " + str(path))
    return raw, snapshot


def _header(path, snapshot):
    with _source_file(path, snapshot) as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            _fail("short safetensors prefix: " + str(path))
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= MAX_HEADER or length + 8 > snapshot["bytes"]:
            _fail("invalid or oversized safetensors header: " + str(path))
        raw = stream.read(length)
    if len(raw) != length:
        _fail("short safetensors header: " + str(path))
    header = _decode(raw)
    if not isinstance(header, dict):
        _fail("safetensors header must be an object")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        _fail("invalid safetensors metadata")
    intervals = []
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data_offsets"}:
            _fail("invalid tensor descriptor: " + name)
        shape, offsets, dtype = value["shape"], value["data_offsets"], value["dtype"]
        if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
            _fail("invalid tensor shape: " + name)
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int or x < 0 for x in offsets):
            _fail("invalid tensor offsets: " + name)
        if dtype not in DTYPE_BITS:
            _fail("unsupported safetensors dtype: " + str(dtype))
        count = 1
        for dimension in shape:
            count *= dimension
        if offsets[1] - offsets[0] != (count * DTYPE_BITS[dtype] + 7) // 8:
            _fail("shape/dtype byte size mismatch: " + name)
        intervals.append((offsets[0], offsets[1], name))
    cursor = 0
    for begin, end, name in sorted(intervals):
        if begin != cursor or end < begin:
            _fail("non-contiguous or overlapping tensor ranges: " + name)
        cursor = end
    if cursor + length + 8 != snapshot["bytes"]:
        _fail("safetensors payload length mismatch: " + str(path))
    return header, length + 8, _digest(prefix + raw)


def _is_embed(name):
    if ".engram.embed." in name or name.startswith("engram.embed."):
        if not EMBED.search(name):
            _fail("unknown Engram embedding tensor: " + name)
        return True
    return False


def _subset_header(header, keys):
    result = {}
    if "__metadata__" in header:
        result["__metadata__"] = header["__metadata__"]
    ranges, offset = [], 0
    for key in sorted(keys, key=lambda key: (header[key]["data_offsets"][0], key)):
        tensor = header[key]
        begin, end = tensor["data_offsets"]
        result[key] = {"dtype": tensor["dtype"], "shape": tensor["shape"],
                       "data_offsets": [offset, offset + end - begin]}
        ranges.append({"tensor": key, "offset": begin, "bytes": end - begin})
        offset += end - begin
    raw = json.dumps(result, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw, ranges, offset


def _expected(value):
    result = {}
    for name, digest in (value or {}).items():
        _metadata_name(name)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            _fail("expected SHA256 must have 64 hex characters: " + name)
        result[name] = digest.lower()
    return result


def make_plan(source, destination, mode, expected_sha256=None, metadata_files=()):
    source, destination = _canonical(source), _canonical(destination)
    if mode not in ("owner", "consumer"):
        _fail("mode must be owner or consumer")
    if not source.is_dir() or not destination.parent.is_dir():
        _fail("source and destination parent must already exist")
    if os.path.commonpath([str(source), str(destination)]) in (str(source), str(destination)):
        _fail("source/destination must not contain one another")
    if destination.exists():
        _fail("destination already exists; never overwrite it")
    stage = _canonical(str(destination) + ".staging")
    if stage.exists() and not stage.is_dir():
        _fail("staging path is not a directory")
    expected_sha256 = _expected(expected_sha256)
    raw, index_snapshot = _small_file(source / INDEX)
    index_sha = _digest(raw)
    if INDEX in expected_sha256 and index_sha != expected_sha256[INDEX]:
        _fail("source index SHA256 does not match expected")
    index = _decode(raw)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        _fail("empty or invalid checkpoint weight_map")
    per_shard = {}
    embedding_keys = []
    for key, filename in weight_map.items():
        _basename(filename)
        if not filename.endswith(".safetensors"):
            _fail("index references a non-safetensors file")
        per_shard.setdefault(filename, []).append(key)
        if _is_embed(key):
            embedding_keys.append(key)
    if not embedding_keys:
        _fail("checkpoint has no Engram embedding tables")
    for key in embedding_keys:
        base = key.rsplit(".", 1)[0]
        if base + ".weight" not in embedding_keys or base + ".scale" not in embedding_keys:
            _fail("Engram weight/scale pair is incomplete: " + base)
    sources = {INDEX: {"snapshot": index_snapshot, "sha256": index_sha}}
    outputs, rewritten_map, tensor_bytes = [], {}, 0
    for filename, keys in sorted(per_shard.items()):
        snapshot = _snapshot(source / filename)
        header, data_start, header_sha = _header(source / filename, snapshot)
        if set(header) - {"__metadata__"} != set(keys):
            _fail("index/header tensor mismatch: " + filename)
        sources[filename] = {"snapshot": snapshot, "header_sha256": header_sha,
                             "expected_full_sha256": expected_sha256.get(filename)}
        embed = [key for key in keys if _is_embed(key)]
        chosen = keys if mode == "owner" and embed else ([] if mode == "owner" else [key for key in keys if not _is_embed(key)])
        if not chosen:
            continue
        tensor_bytes += sum(header[key]["data_offsets"][1] - header[key]["data_offsets"][0] for key in chosen)
        rewritten_map.update({key: filename for key in chosen})
        if set(chosen) == set(keys):
            outputs.append({"name": filename, "kind": "copy", "bytes": snapshot["bytes"],
                            "expected_sha256": expected_sha256.get(filename)})
        else:
            prefix, ranges, payload_bytes = _subset_header(header, chosen)
            outputs.append({"name": filename, "kind": "subset", "bytes": len(prefix) + payload_bytes,
                            "prefix_hex": prefix.hex(), "source_data_start": data_start, "ranges": ranges})
    names = set(DEFAULT_METADATA)
    names.update(_metadata_name(name) for name in metadata_files)
    if INDEX in names or any(name.endswith(".safetensors") for name in names):
        _fail("metadata names cannot include index or tensor shards")
    if any(name in (".lock", "stage_state.json", "checkpoint_provenance.json") or name.endswith((".part", ".tmp")) for name in names):
        _fail("reserved staging metadata filename")
    for name in sorted(names):
        path = _canonical(source / name)
        if path.is_symlink():
            _fail("metadata symlink is forbidden: " + str(path))
        if not path.exists():
            if name in metadata_files or name == "config.json":
                _fail("required metadata missing: " + name)
            continue
        raw, snapshot = _small_file(path)
        sha = _digest(raw)
        if name in expected_sha256 and expected_sha256[name] != sha:
            _fail("source metadata SHA256 does not match expected: " + name)
        sources[name] = {"snapshot": snapshot, "sha256": sha}
        outputs.append({"name": name, "kind": "copy", "bytes": snapshot["bytes"], "expected_sha256": sha})
    index_metadata = dict(index.get("metadata", {}))
    index_metadata["total_size"] = tensor_bytes
    new_index = {"metadata": index_metadata, "weight_map": rewritten_map}
    new_raw = _json_bytes(new_index)
    outputs.append({"name": INDEX, "kind": "generated", "bytes": len(new_raw),
                    "content_hex": new_raw.hex(), "expected_sha256": _digest(new_raw)})
    plan = {"schema": SCHEMA, "source": str(source), "destination": str(destination),
            "stage": str(stage), "mode": mode, "sources": sources, "outputs": outputs,
            "embedding_keys": sorted(embedding_keys), "output_tensor_count": len(rewritten_map),
            "output_tensor_bytes": tensor_bytes, "output_bytes": sum(x["bytes"] for x in outputs)}
    plan["plan_sha256"] = _digest(_json_bytes(plan))
    return plan


def _hash_file(path, buffer_size):
    snapshot = _snapshot(path)
    digest = hashlib.sha256()
    with _source_file(path, snapshot) as stream:
        while True:
            chunk = stream.read(buffer_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rename_noreplace(source, destination):
    """Publish atomically without replacing even an empty competing directory."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "linux":
        if hasattr(library, "renameat2"):
            result = library.renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
        else:
            # Older glibc (the development hosts use glibc 2.17) lacks the wrapper.
            machine = os.uname().machine
            number = {"aarch64": 276, "x86_64": 316}.get(machine)
            if number is None:
                _fail("no safe no-replace rename on this architecture")
            result = library.syscall(number, -100, os.fsencode(source), -100, os.fsencode(destination), 1)
    elif sys.platform == "darwin" and hasattr(library, "renamex_np"):
        result = library.renamex_np(os.fsencode(source), os.fsencode(destination), 4)
    else:
        _fail("atomic no-replace rename requires Linux or macOS")
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _atomic_json(path, document):
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))
    _sync_dir(path.parent)


def _copy_range(source, output, count, buffer_size, digest, range_digest):
    remaining = count
    while remaining:
        chunk = source.read(min(buffer_size, remaining))
        if not chunk:
            _fail("unexpected EOF while streaming source")
        output.write(chunk)
        digest.update(chunk)
        if range_digest is not None:
            range_digest.update(chunk)
        remaining -= len(chunk)


def _write_output(plan, item, stage, buffer_size, before_publish=None):
    name = item["name"]
    target = stage / name
    temporary = stage / (name + ".part")
    if target.parent != stage:
        _canonical(target.parent)
        if not target.parent.exists():
            target.parent.mkdir(mode=0o700)
            _sync_dir(stage)
    if target.exists() or target.is_symlink() or temporary.exists() or temporary.is_symlink():
        _fail("unexpected existing output; refusing overwrite: " + name)
    fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    digest, selected = hashlib.sha256(), []
    with os.fdopen(fd, "wb") as output:
        if item["kind"] == "generated":
            raw = bytes.fromhex(item["content_hex"])
            output.write(raw)
            digest.update(raw)
        else:
            source = Path(plan["source"]) / name
            source_info = plan["sources"][name]
            with _source_file(source, source_info["snapshot"]) as stream:
                if item["kind"] == "copy":
                    _copy_range(stream, output, item["bytes"], buffer_size, digest, None)
                else:
                    prefix = bytes.fromhex(item["prefix_hex"])
                    output.write(prefix)
                    digest.update(prefix)
                    for interval in item["ranges"]:
                        stream.seek(item["source_data_start"] + interval["offset"])
                        range_digest = hashlib.sha256()
                        _copy_range(stream, output, interval["bytes"], buffer_size, digest, range_digest)
                        selected.append(dict(interval, sha256=range_digest.hexdigest()))
        output.flush()
        os.fsync(output.fileno())
    sha = digest.hexdigest()
    if temporary.stat().st_size != item["bytes"]:
        _fail("output size mismatch: " + name)
    if item.get("expected_sha256") and sha != item["expected_sha256"]:
        _fail("streamed SHA256 does not match expected: " + name)
    record = {"bytes": item["bytes"], "sha256": sha, "kind": item["kind"]}
    if item["kind"] == "copy":
        record["source_full_sha256"] = sha
        record["expected_sha256_matched"] = bool(item.get("expected_sha256"))
    elif item["kind"] == "subset":
        record["source_digest_scope"] = "header plus selected tensor ranges only; full source shard was not hashed"
        record["source_header_sha256"] = plan["sources"][name]["header_sha256"]
        record["source_ranges"] = selected
    if before_publish:
        before_publish(name, record)
    _rename_noreplace(temporary, target)
    _sync_dir(target.parent)
    return record


def _check_all_sources(plan):
    for name, source in plan["sources"].items():
        _unchanged(Path(plan["source"]) / name, source["snapshot"])


def execute_plan(plan, resume=False, buffer_size=BUFFER, reserve_bytes=1024 ** 3, progress=None):
    if type(buffer_size) is not int or not 1 <= buffer_size <= 64 * 1024 * 1024:
        _fail("buffer size must be between 1 byte and 64 MiB")
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        _fail("reserve_bytes must be nonnegative")
    destination, stage = _canonical(plan["destination"]), _canonical(plan["stage"])
    if destination.exists():
        _fail("destination already exists; never overwrite it")
    _check_all_sources(plan)
    state_path = stage / "stage_state.json"
    existing = stage.exists()
    if existing and not resume:
        _fail("staging directory exists; use --resume after inspection")
    if existing:
        s = stage.stat()
        if s.st_uid != os.geteuid() or stat.S_IMODE(s.st_mode) & 0o077:
            _fail("staging directory must be private and owned by current UID")
    else:
        free = shutil.disk_usage(str(destination.parent)).free
        if free < plan["output_bytes"] + reserve_bytes:
            _fail("insufficient destination space before staging")
        stage.mkdir(mode=0o700)
    lockfd = os.open(str(stage / ".lock"), os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("another staging process holds the lock")
        state = {"schema": SCHEMA, "plan_sha256": plan["plan_sha256"], "completed": {}, "pending": {}}
        if existing:
            raw, _ = _small_file(state_path)
            state = _decode(raw)
            if state.get("plan_sha256") != plan["plan_sha256"]:
                _fail("source/options changed; resume plan SHA256 mismatch")
        else:
            _atomic_json(state_path, state)
        allowed = {x["name"] for x in plan["outputs"]}
        by_name = {x["name"]: x for x in plan["outputs"]}
        if not set(state["completed"]).issubset(allowed):
            _fail("unknown output in resume state")
        state.setdefault("pending", {})
        if not set(state["pending"]).issubset(allowed):
            _fail("unknown pending output in resume state")
        # A durable journal entry precedes each file rename. Recover either side
        # of a crash between file rename and the final completed-state update.
        for name, record in list(state["pending"].items()):
            item = by_name[name]
            target, partial = stage / name, stage / (name + ".part")
            if name in state["completed"] or target.exists() == partial.exists():
                _fail("ambiguous pending output; inspect staging: " + name)
            candidate = target if target.exists() else partial
            if record["bytes"] != item["bytes"] or _snapshot(candidate)["bytes"] != item["bytes"]:
                _fail("pending output size mismatch: " + name)
            if _hash_file(candidate, buffer_size) != record["sha256"]:
                _fail("pending output SHA256 mismatch: " + name)
            if item.get("expected_sha256") and item["expected_sha256"] != record["sha256"]:
                _fail("pending expected SHA256 mismatch: " + name)
            if candidate == partial:
                _rename_noreplace(partial, target)
                _sync_dir(target.parent)
            state["completed"][name] = record
            del state["pending"][name]
        if existing:
            temporary_state = stage / "stage_state.json.tmp"
            if temporary_state.exists():
                if temporary_state.is_symlink() or not temporary_state.is_file():
                    _fail("unsafe temporary state")
                temporary_state.unlink()
            _atomic_json(state_path, state)
        directories = {str(Path(name).parent) for name in allowed if "/" in name}
        paths = []
        for path in stage.iterdir():
            if path.is_dir() and not path.is_symlink():
                if path.name not in directories:
                    _fail("untracked staging directory: " + path.name)
                paths.extend(path.iterdir())
            else:
                paths.append(path)
        for path in paths:
            relative = str(path.relative_to(stage))
            if relative in (".lock", "stage_state.json"):
                if not path.is_file() or path.is_symlink():
                    _fail("unsafe state file")
                continue
            if relative.endswith(".part") and relative[:-5] in allowed:
                if path.is_symlink() or not path.is_file():
                    _fail("unsafe partial output")
                # Only incomplete per-file work is discarded on explicit resume.
                path.unlink()
                continue
            if relative == "stage_state.json.tmp":
                if path.is_symlink() or not path.is_file():
                    _fail("unsafe temporary state")
                path.unlink()
                continue
            if relative == "checkpoint_provenance.json":
                raw, _ = _small_file(path)
                if _decode(raw).get("plan_sha256") != plan["plan_sha256"]:
                    _fail("unexpected provenance in staging")
                path.unlink()
                continue
            if relative not in state["completed"]:
                _fail("untracked file in staging; inspect before resuming: " + relative)
        for item in plan["outputs"]:
            record = state["completed"].get(item["name"])
            if record is None:
                continue
            target = stage / item["name"]
            if record["bytes"] != item["bytes"] or _snapshot(target)["bytes"] != item["bytes"]:
                _fail("resume output size mismatch: " + item["name"])
            if _hash_file(target, buffer_size) != record["sha256"]:
                _fail("resume output SHA256 mismatch: " + item["name"])
            if item.get("expected_sha256") and record["sha256"] != item["expected_sha256"]:
                _fail("resume expected SHA256 mismatch: " + item["name"])
        needed = sum(item["bytes"] for item in plan["outputs"] if item["name"] not in state["completed"])
        free = shutil.disk_usage(str(stage)).free
        if free < needed + reserve_bytes:
            _fail("insufficient destination space: free={}, need={} plus reserve={}".format(free, needed, reserve_bytes))
        for item in plan["outputs"]:
            if item["name"] in state["completed"]:
                if progress:
                    progress("verified_resume", item["name"], item["bytes"])
                continue
            if progress:
                progress("start", item["name"], item["bytes"])
            def before_publish(name, record):
                state["pending"][name] = record
                _atomic_json(state_path, state)
            record = _write_output(plan, item, stage, buffer_size, before_publish)
            state["completed"][item["name"]] = record
            del state["pending"][item["name"]]
            _atomic_json(state_path, state)
            if progress:
                progress("complete", item["name"], item["bytes"])
        _check_all_sources(plan)
        provenance = {"schema": SCHEMA, "plan_sha256": plan["plan_sha256"],
                      "source": plan["source"], "mode": plan["mode"],
                      "created_unix_time": time.time(), "source_files": plan["sources"],
                      "outputs": state["completed"], "excluded_embedding_keys": plan["embedding_keys"] if plan["mode"] == "consumer" else [],
                      "verification": "streaming SHA256; complete files rehashed only on resume",
                      "source_immutability": "inode, size, mtime_ns, ctime_ns before/after; full-copy SHA256 and selected-range hashes"}
        _atomic_json(stage / "checkpoint_provenance.json", provenance)
        if destination.exists() or destination.is_symlink():
            _fail("destination appeared during staging; refusing publish")
        _rename_noreplace(stage, destination)
        _sync_dir(destination.parent)
        return provenance
    finally:
        os.close(lockfd)


def summary(plan):
    return {"plan_sha256": plan["plan_sha256"], "source": plan["source"],
            "destination": plan["destination"], "mode": plan["mode"],
            "output_bytes": plan["output_bytes"], "output_tensor_count": plan["output_tensor_count"],
            "embedding_tensor_count": len(plan["embedding_keys"]),
            "files": [{k: item[k] for k in ("name", "kind", "bytes")} for item in plan["outputs"]],
            "free_destination_bytes": shutil.disk_usage(str(Path(plan["destination"]).parent)).free}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--mode", choices=("owner", "consumer"), required=True)
    parser.add_argument("--expected-sha256", help="JSON object: plain source filename -> expected SHA256")
    parser.add_argument("--metadata-file", action="append", default=[])
    parser.add_argument("--execute", action="store_true", help="write files; without this flag only read headers and print a plan")
    parser.add_argument("--resume", action="store_true", help="rehash and reuse complete files from matching private .staging directory")
    parser.add_argument("--buffer-mib", type=int, default=8)
    parser.add_argument("--reserve-gib", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        expected = None
        if args.expected_sha256:
            raw, _ = _small_file(Path(args.expected_sha256))
            expected = _decode(raw)
            if not isinstance(expected, dict):
                _fail("expected SHA256 input must be an object")
        plan = make_plan(args.source, args.destination, args.mode, expected, args.metadata_file)
        print(json.dumps(summary(plan), sort_keys=True, indent=2), flush=True)
        if args.execute:
            def progress(event, filename, size):
                print(json.dumps({"event": event, "file": filename, "bytes": size, "time": time.time()}), flush=True)
            result = execute_plan(plan, args.resume, args.buffer_mib * 1024 * 1024,
                                  args.reserve_gib * 1024 ** 3, progress)
            print(json.dumps({"event": "published", "destination": plan["destination"],
                              "plan_sha256": result["plan_sha256"]}), flush=True)
        return 0
    except (StageError, OSError, ValueError, KeyError, TypeError) as error:
        print("checkpoint staging failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
