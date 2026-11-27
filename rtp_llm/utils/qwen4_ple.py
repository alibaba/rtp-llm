"""Stage one shared PLE table that every local rank of a Qwen4 model maps.

The released n-gram memory is the bulk of a qwen4_exp checkpoint -- around a
hundred gigabytes -- and it is frozen for the whole of an RL run. Staging it
once into a memfd and letting every local rank map it read-only costs one copy
of host memory instead of one per rank, which is what makes a DP topology
loadable at all: the table is selected per TP rank, so at TP1 every rank would
otherwise hold the whole thing.

Both the MXFP4 checkpoint's FP8 table and the BF16 checkpoint's BF16 table are
staged by the same byte-range copy; only the element size differs.
"""

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


def _memfd(name: str):
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name)
    # Some Linux Python 3.10 builds omit the wrapper.
    create = ctypes.CDLL(None, use_errno=True).memfd_create
    fd = create(name.encode(), 1)  # MFD_CLOEXEC
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


# Checkpoint dtypes the table may use, and their element size in bytes.
PLE_DTYPES = {"F8_E4M3": 1, "BF16": 2}


def _ple_layout(checkpoint: Path):
    index = (checkpoint / "model.safetensors.index.json").read_bytes()
    config = (checkpoint / "config.json").read_bytes()
    identity = hashlib.sha256(os.fsencode(checkpoint.resolve()))
    identity.update(hashlib.sha256(index).digest())
    identity.update(hashlib.sha256(config).digest())
    shards = {}
    for name, filename in json.loads(index)["weight_map"].items():
        if PLE_SHARD.fullmatch(name):
            shards.setdefault(filename, []).append(name)
    if sum(map(len, shards.values())) != 128:
        raise ValueError("Expected 128 Qwen4 PLE shards in the checkpoint")

    # The staging process and all loading ranks independently use this order.
    ranges = []
    entries = {}
    table_size = 0
    for filename, names in sorted(shards.items()):
        path = checkpoint / filename
        with path.open("rb") as source:
            header_size = int.from_bytes(source.read(8), "little")
            header_data = source.read(header_size)
        identity.update(hashlib.sha256(header_data).digest())
        header = json.loads(header_data)
        for name in sorted(names, key=lambda name: header[name]["data_offsets"][0]):
            tensor = header[name]
            begin, end = tensor["data_offsets"]
            shape = tensor["shape"]
            length = end - begin
            item_size = PLE_DTYPES.get(tensor["dtype"])
            if (
                item_size is None
                or begin < 0
                or length <= 0
                or length != math.prod(shape) * item_size
            ):
                raise ValueError(
                    f"Invalid PLE shard for {name}: dtype {tensor['dtype']}, "
                    f"{length} bytes for shape {shape}"
                )
            ranges.append((path, name, 8 + header_size + begin, length))
            entries[name] = (table_size, length, shape)
            table_size += length

    return ranges, entries, table_size, f"qwen4-ple-v1-{identity.hexdigest()}"


def resolve_shared_ple_table(path: str, checkpoint: Path):
    _, entries, table_size, name = _ple_layout(checkpoint)
    if os.readlink(path) != f"/memfd:{name} (deleted)":
        raise ValueError("Qwen4 shared PLE table does not match checkpoint")
    if os.stat(path).st_size != table_size:
        raise ValueError("Qwen4 shared PLE table has an invalid size")
    return table_size, entries


def _check_staging_memory(table_size: int):
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


@contextmanager
def shared_ple(checkpoint: Path, *, shutdown_requested=None):
    ranges, _, table_size, name = _ple_layout(checkpoint)
    # The timeline driver passes its already staged table across exec via pass_fds.
    for path in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(path)
        except FileNotFoundError:
            continue
        fd = int(path.name)
        if target == f"/memfd:{name} (deleted)" and os.get_inheritable(fd):
            with os.fdopen(fd, "rb") as table:
                os.set_inheritable(fd, False)
                if os.fstat(fd).st_size != table_size:
                    raise ValueError("Qwen4 shared PLE table has an invalid size")
                yield table.fileno()
            return

    _check_staging_memory(table_size)
    logging.info("Staging %.1f GiB of shared PLE data", table_size / (1 << 30))
    with os.fdopen(_memfd(name), "wb") as table:
        for path, tensor_name, offset, length in ranges:
            with path.open("rb") as source:
                source.seek(offset)
                remaining = length
                while remaining:
                    if shutdown_requested is not None and shutdown_requested():
                        raise InterruptedError(
                            "Shutdown requested while staging Qwen4 PLE"
                        )
                    data = source.read(min(remaining, 16 << 20))
                    if not data:
                        raise EOFError(f"Truncated PLE shard {tensor_name} in {path}")
                    table.write(data)
                    remaining -= len(data)
        table.flush()
        yield table.fileno()


def shares_ple_table(model_args, dp_size: int) -> bool:
    """Decide whether this run should map one shared table instead of copies.

    MXFP4 keeps its existing behaviour unconditionally. A BF16 checkpoint only
    stages when DP would otherwise replicate the table, because at TP == world
    size the per-rank shards already sum to one copy and staging would just move
    that memory earlier.
    """
    if model_args.model_type != "qwen4_exp":
        return False
    checkpoint = Path(model_args.ckpt_path)
    try:
        config = json.loads((checkpoint / "config.json").read_text())
    except OSError:
        return False
    if (config.get("quantization_config") or {}).get("quant_method") == "mxfp4":
        return True
    return dp_size > 1


@contextmanager
def stage_qwen4_ple_for_server(model_args, *, dp_size=1, shutdown_requested):
    """Expose a single staged table to the backend launcher and its ranks."""
    if not shares_ple_table(model_args, dp_size):
        yield None
        return

    checkpoint = Path(model_args.ckpt_path)

    with shared_ple(checkpoint, shutdown_requested=shutdown_requested) as fd:
        yield f"/proc/{os.getpid()}/fd/{fd}"
