"""Inspect this run's published cache; never restart or erase a serving cache."""

import tarfile
import time
from hashlib import sha256
from pathlib import Path

from request_smoke import require


def compilation_keys(root):
    """Fingerprint native/Triton cubins in one run's private cache only."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError(f"JIT cache must not be a symlink: {root}")
    keys = {}
    for name in ("deep-gemm", "triton"):
        component = root / name
        if component.is_symlink():
            raise ValueError(f"JIT component must not be a symlink: {component}")
        if not component.is_dir():
            raise FileNotFoundError(f"Missing private JIT component: {component}")
        for path in sorted(component.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"JIT entry must not be a symlink: {path}")
            if path.suffix == ".cubin":
                if not path.is_file() or not path.stat().st_size:
                    raise ValueError(f"Incomplete JIT binary: {path}")
                keys[path.relative_to(root).as_posix()] = sha256(path.read_bytes()).hexdigest()
    return keys


def compilation_changes(before, after):
    return {
        "added": sorted(after.keys() - before.keys()),
        "rewritten": sorted(key for key in before.keys() & after.keys() if before[key] != after[key]),
        "removed": sorted(before.keys() - after.keys()),
    }


def check_publication(root, started_ns, timeout=180):
    import zstandard

    require(
        root is not None and root.is_dir(),
        "JIT smoke requires a configured snapshot directory",
    )
    require(
        isinstance(started_ns, int) and 0 < started_ns <= time.time_ns(),
        "JIT smoke requires this service startup's timestamp",
    )
    deadline = time.monotonic() + timeout
    inspected = set()
    binaries = {}
    snapshots = []
    while time.monotonic() < deadline:
        candidates = sorted(root.glob("v2/*/*.jit_snapshot.tar.zst"), reverse=True)
        for path in candidates:
            if path in inspected or path.stat().st_mtime_ns < started_ns:
                continue
            binaries = {}
            with path.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(
                source
            ) as stream:
                with tarfile.open(fileobj=stream, mode="r|") as archive:
                    for member in archive:
                        if (
                            member.isfile()
                            and member.size > 0
                            and member.name.endswith((".cubin", ".so", ".o"))
                        ):
                            component = member.name.split("/", 1)[0]
                            binaries[component] = binaries.get(component, 0) + 1
            inspected.add(path)
            snapshots.append({"path": str(path), "bytes": path.stat().st_size})
            if {"deep_gemm", "triton"} <= binaries.keys():
                return {
                    "passed": True,
                    "compiled_binary_counts": binaries,
                    "snapshots": [snapshots[-1]],
                    "restore_tested": False,
                    "storage_backend": "file",
                }
        time.sleep(2)
    raise AssertionError(
        f"this run did not publish DeepGEMM and Triton binaries: {binaries}"
    )
