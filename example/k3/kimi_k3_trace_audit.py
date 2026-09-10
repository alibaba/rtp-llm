"""Verify persisted K3 trace integrity without claiming model coverage."""

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import torch


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frame(path):
    # PyTorch mmap requires uncompressed tensor storage ZIP members.
    with zipfile.ZipFile(path) as archive:
        mmap = all(m.compress_type == zipfile.ZIP_STORED for m in archive.infolist())
    return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)


def audit_recorder(directory, *, require_closed=True):
    directory = Path(directory)
    require(not (directory / "incomplete.json").exists(), "recorder reported failure")
    require(not list(directory.glob("*.part")), "unfinished artifact files")
    identity = json.loads((directory / "identity.json").read_text())
    rows = [
        json.loads(line)
        for line in (directory / "index.jsonl").read_text().splitlines()
    ]
    closed = directory / "recorder_closed.json"
    if require_closed:
        require(closed.exists(), "recorder has no shutdown flush marker")
    if closed.exists():
        require(
            json.loads(closed.read_text())["frames_written"] == len(rows),
            "shutdown count differs from index",
        )
    expected_paths = {f"frame-{i:08d}.pt" for i in range(len(rows))}
    require(
        {p.name for p in directory.glob("frame-*.pt")} == expected_paths,
        "missing or unindexed frame files",
    )
    counts = Counter()
    byte_counts = Counter()
    dtypes = {}
    observations = set()
    previous = None
    next_fragment = 0
    final_seen = True
    tensor_bytes = 0
    for sequence, row in enumerate(rows):
        filename = f"frame-{sequence:08d}.pt"
        require(row["sequence"] == sequence, "non-contiguous index sequence")
        require(row["path"] == filename, "unexpected frame path")
        path = directory / filename
        require(path.stat().st_size == row["bytes"], f"size mismatch: {filename}")
        require(sha256(path) == row["sha256"], f"digest mismatch: {filename}")
        frame = load_frame(path)
        require(frame["schema_version"] == 1, "unsupported frame schema")
        require(frame["sequence"] == sequence, "payload sequence differs from index")
        metadata = frame["metadata"]
        require(metadata == row["metadata"], "payload metadata differs from index")
        observation = metadata["observation_id"]
        fragment = metadata["trace_fragment"]
        if observation != previous:
            require(final_seen, "observation is missing its final fragment")
            require(
                observation not in observations, "observation reappeared after final"
            )
            observations.add(observation)
            previous = observation
            next_fragment = 0
            final_seen = False
        require(not final_seen, "fragment appears after final marker")
        require(fragment["index"] == next_fragment, "missing or reordered fragment")
        require(type(fragment["final"]) is bool, "invalid final marker")
        next_fragment += 1
        final_seen = fragment["final"]
        require(len(frame["tensors"]) == row["tensor_count"], "tensor count mismatch")
        for item in frame["tensors"]:
            value, tensor_meta, name = item["value"], item["metadata"], item["name"]
            require(isinstance(value, torch.Tensor), "non-tensor payload")
            require(
                list(value.shape) == tensor_meta["shape"], f"shape mismatch: {name}"
            )
            require(str(value.dtype) == tensor_meta["dtype"], f"dtype mismatch: {name}")
            count = value.numel() * value.element_size()
            tensor_bytes += count
            counts[name] += 1
            byte_counts[name] += count
            dtypes.setdefault(name, set()).add(str(value.dtype))
        del frame
    require(final_seen, "last observation is missing its final fragment")
    return {
        "identity": identity,
        "integrity_verified": True,
        "coverage_verified": False,
        "recorder_closed": closed.exists(),
        "fragment_count": len(rows),
        "observation_count": len(observations),
        "tensor_count": sum(counts.values()),
        "tensor_bytes": tensor_bytes,
        "file_bytes": sum(row["bytes"] for row in rows),
        "outputs": {
            name: {
                "count": counts[name],
                "bytes": byte_counts[name],
                "dtypes": sorted(dtypes[name]),
            }
            for name in sorted(counts)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--allow-open",
        action="store_true",
        help="Inspect a quiescent live recorder; never declares the run complete",
    )
    args = parser.parse_args()
    identities = sorted(args.root.rglob("identity.json"))
    require(bool(identities), "no recorders found")
    reports = {}
    errors = {}
    for identity in identities:
        directory = identity.parent
        try:
            reports[str(directory)] = audit_recorder(
                directory, require_closed=not args.allow_open
            )
        except Exception as exc:
            errors[str(directory)] = f"{type(exc).__name__}: {exc}"
    report = {
        "integrity_verified": not errors,
        "coverage_verified": False,
        "note": "Layer, module, rank, request and token coverage require a separate execution contract.",
        "recorders": reports,
        "errors": errors,
    }
    with args.report.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"recorders": len(reports), "errors": errors}))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
