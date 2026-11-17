"""Stage the shared FP8 PLE table for the Qwen4 M11 perf test."""

import ctypes
import hashlib
import json
import logging
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path

PLE_SHARD = re.compile(
    r"model\.language_model\.layers\.\d+\.ple\.ple_embedding\."
    r"ngram_embedding\.shard_\d+\.weight"
)


def _memfd():
    if hasattr(os, "memfd_create"):
        return os.memfd_create("qwen4-m11-ple")
    # Some Linux Python 3.10 builds omit the wrapper.
    create = ctypes.CDLL(None, use_errno=True).memfd_create
    fd = create(b"qwen4-m11-ple", 1)  # MFD_CLOEXEC
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


@contextmanager
def shared_ple(checkpoint: Path, output: Path):
    existing = os.environ.get("QWEN4_PLE_SHARED_TABLE")
    if existing:
        manifest = Path(existing).resolve(strict=True)
        Path(json.loads(manifest.read_text())["path"]).stat()
        yield manifest
        return

    index = (checkpoint / "model.safetensors.index.json").read_bytes()
    config = (checkpoint / "config.json").read_bytes()
    shards = {}
    for name, filename in json.loads(index)["weight_map"].items():
        if PLE_SHARD.fullmatch(name):
            shards.setdefault(filename, []).append(name)
    if sum(map(len, shards.values())) != 128:
        raise ValueError("Expected 128 Qwen4 M11 PLE shards in the checkpoint")

    ranges = []
    for filename, names in shards.items():
        path = checkpoint / filename
        with path.open("rb") as source:
            header_size = int.from_bytes(source.read(8), "little")
            header = json.loads(source.read(header_size))
        for name in names:
            tensor = header[name]
            begin, end = tensor["data_offsets"]
            shape = tensor["shape"]
            if (
                tensor["dtype"] != "F8_E4M3"
                or begin < 0
                or end - begin != math.prod(shape)
            ):
                raise ValueError(f"Invalid FP8 PLE shard for {name}")
            ranges.append((path, name, 8 + header_size + begin, end - begin, shape))

    table_size = sum(length for _, _, _, length, _ in ranges)
    meminfo = Path("/proc/meminfo").read_text().splitlines()
    available = next(
        int(line.split()[1]) * 1024
        for line in meminfo
        if line.startswith("MemAvailable:")
    )
    limit = Path("/sys/fs/cgroup/memory.max")
    if limit.is_file() and (value := limit.read_text().strip()) != "max":
        used = int(limit.with_name("memory.current").read_text())
        available = min(available, int(value) - used)
    if available < table_size + 96 * (1 << 30):
        raise RuntimeError(
            "Shared PLE staging requires the table size plus 96 GiB of available RAM"
        )

    logging.info("Staging %.1f GiB of shared PLE data", table_size / (1 << 30))
    manifest = output / "ple_manifest.json"
    with os.fdopen(_memfd(), "wb") as table:
        entries = {}
        for path, name, offset, length, shape in ranges:
            entries[name] = [table.tell(), length, shape]
            with path.open("rb") as source:
                source.seek(offset)
                remaining = length
                while remaining:
                    data = source.read(min(remaining, 16 << 20))
                    if not data:
                        raise EOFError(f"Truncated PLE shard {name} in {path}")
                    table.write(data)
                    remaining -= len(data)
        table.flush()
        manifest.write_text(
            json.dumps(
                {
                    "path": f"/proc/{os.getpid()}/fd/{table.fileno()}",
                    "entries": entries,
                    "checkpoint_index_sha256": hashlib.sha256(index).hexdigest(),
                    "checkpoint_config_sha256": hashlib.sha256(config).hexdigest(),
                }
            )
        )
        try:
            yield manifest
        finally:
            manifest.unlink()
