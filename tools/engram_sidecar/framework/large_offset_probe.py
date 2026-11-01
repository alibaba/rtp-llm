#!/usr/bin/env python3
"""Real >4GiB EGM address/lookup probe; no checkpoint or LLM required.

Only native prebuilt imports are permitted. The ordinary Triton lookup may JIT.
The operator owns authenticated publication transfer and consumer-exit proof
before creating Owner STOP. No stdout contains the private Fabric capability.
"""
from __future__ import annotations

import argparse
import base64
import gc
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys
import threading
import time
from uuid import UUID, uuid4

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tools.engram_sidecar.framework.checkpoint_support import emit, private_write
from rtp_llm.models_py.modules.dsv41.engram_bootstrap import _json, _private_read

ROWS = (1 << 24) + 257
DIM = 256
SCALE_COLUMNS = DIM // 32
WEIGHT_BYTES = ROWS * DIM
LOGICAL_BYTES = ROWS * (DIM + SCALE_COLUMNS)
BOUNDARY_ROW = 1 << 24
SAMPLE_ROWS = (0, BOUNDARY_ROW - 1, BOUNDARY_ROW, ROWS - 1)
FP8_BYTES = (0x38, 0xB8, 0x40, 0xC0, 0x30, 0xB0, 0x48, 0xC8)
FP8_VALUES = (1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 4.0, -4.0)
MAX_PRIVATE_BYTES = 32 << 10


def row_pattern(slot):
    """Small exact FP8 bit pattern and independently interpreted float values."""
    if type(slot) is not int or not 0 <= slot < len(SAMPLE_ROWS):
        raise ValueError("invalid sample slot")
    weights = bytes(FP8_BYTES[(column + 3 * slot) % len(FP8_BYTES)] for column in range(DIM))
    scales = bytes(125 + (group + slot) % 5 for group in range(SCALE_COLUMNS))
    expected = [math.ldexp(FP8_VALUES[(column + 3 * slot) % len(FP8_VALUES)], scales[column // 32] - 127)
                for column in range(DIM)]
    return weights, scales, expected


def sample_metadata():
    return [{"row": row, "fp8_bytes": list(row_pattern(slot)[0]),
             "e8m0_bytes": list(row_pattern(slot)[1])}
            for slot, row in enumerate(SAMPLE_ROWS)]


def metadata(allocation_bytes, granularity, generation):
    return {"schema_version": 1, "kind": "engram-large-offset-probe-v1",
            "rows": ROWS, "dim": DIM, "scale_columns": SCALE_COLUMNS,
            "weight_bytes": WEIGHT_BYTES, "logical_bytes": LOGICAL_BYTES,
            "allocation_bytes": allocation_bytes, "granularity": granularity,
            "generation": generation, "samples": sample_metadata()}


def validate_metadata(document):
    """Fixed bounded experiment geometry, before any native import/allocation."""
    expected_fields = set(metadata(LOGICAL_BYTES, 1, "")) | {"handle"}
    if type(document) is not dict or set(document) != expected_fields:
        raise ValueError("INVALID_PROBE_FIELDS")
    allocation, granularity, generation = (document[k] for k in ("allocation_bytes", "granularity", "generation"))
    if (type(allocation) is not int or type(granularity) is not int
            or not 0 < granularity <= 128 << 30
            or allocation != LOGICAL_BYTES + (-LOGICAL_BYTES) % granularity
            or allocation > 128 << 30):
        raise ValueError("INVALID_PROBE_ALLOCATION")
    try:
        if str(UUID(generation)) != generation:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ValueError("INVALID_PROBE_GENERATION") from None
    actual = {key: value for key, value in document.items() if key != "handle"}
    # Canonical JSON equality also rejects bool-for-int substitutions and NaNs.
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
            metadata(allocation, granularity, generation), sort_keys=True):
        raise ValueError("PROBE_GEOMETRY_OR_PATTERN_MISMATCH")
    encoded = document["handle"]
    if type(encoded) is not str or not 0 < len(encoded) <= 1024:
        raise ValueError("INVALID_PROBE_CAPABILITY")
    try:
        handle = base64.b64decode(encoded, validate=True)
    except ValueError:
        raise ValueError("INVALID_PROBE_CAPABILITY") from None
    if not handle:
        raise ValueError("INVALID_PROBE_CAPABILITY")
    return handle


def read_publication(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("absolute private publication path required")
    parent = path.parent.stat()
    if (path.parent.is_symlink() or not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o077):
        raise ValueError("UNSAFE_PROBE_DIRECTORY")
    document = _json(_private_read(path, MAX_PRIVATE_BYTES))
    handle = validate_metadata(document)
    return document, handle


def native_and_torch(device):
    import torch
    if not torch.cuda.is_available() or not 0 <= device < torch.cuda.device_count():
        raise RuntimeError("requested CUDA device is not available")
    torch.cuda.set_device(device)
    # No load_native.py/build/JIT extension path exists in this probe.
    spec = importlib.util.find_spec("engram_native_dev_492ecf45")
    if (spec is None or not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader)
            or not spec.origin or not Path(spec.origin).is_file()):
        raise RuntimeError("prebuilt native extension required; Python shim/JIT loader is forbidden")
    native = importlib.import_module("engram_native_dev_492ecf45")
    if native.max_allocation_bytes < LOGICAL_BYTES:
        raise RuntimeError("prebuilt native bridge has insufficient allocation capacity")
    if any(native.resource_counters().values()):
        raise RuntimeError("native probe must start with zero live resources")
    return native, torch


def run_owner(args):
    args.publication.mkdir(mode=0o700, parents=True, exist_ok=False)
    native, torch, owner = None, None, None
    published, failed = False, False

    def hold():
        deadline, reported = time.monotonic() + args.report_timeout, False
        while not (args.publication / "STOP").is_file():
            if not reported and time.monotonic() >= deadline:
                emit("WAITING_FOR_STOP", scope="large_offset_probe", action="backing_retained")
                reported = True
            threading.Event().wait(1)

    def interrupt(signum, frame):
        nonlocal failed
        if not published:
            raise SystemExit(128 + signum)
        failed = True
        emit("SIGNAL_WHILE_PUBLISHED", signal=signum, action="backing_retained_until_STOP")

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        native, torch = native_and_torch(args.device)
        owner = native.Owner(size=LOGICAL_BYTES, numa=args.numa, device=args.device)
        for slot, row in enumerate(SAMPLE_ROWS):
            weights, scales, _ = row_pattern(slot)
            owner.load(weights, offset=row * DIM)
            owner.load(scales, offset=WEIGHT_BYTES + row * SCALE_COLUMNS)
        info = owner.info()
        document = metadata(info["allocation_bytes"], info["granularity"], str(uuid4()))
        document["handle"] = base64.b64encode(owner.export_handle()).decode("ascii")
        validate_metadata(document)
        # Treat an attempted publication as visible, even if file write errors.
        published = True
        private_write(args.publication / "probe.json", json.dumps(document, sort_keys=True).encode())
        del document
        emit("READY", scope="large_offset_probe", logical_bytes=LOGICAL_BYTES,
             allocation_bytes=info["allocation_bytes"], granularity=info["granularity"],
             sampled_rows=list(SAMPLE_ROWS), weight_byte_offsets=[row * DIM for row in SAMPLE_ROWS],
             scale_offset=WEIGHT_BYTES, device=args.device, numa=args.numa,
             counters=native.resource_counters())
        hold()
    except BaseException as error:
        failed = True
        emit("FAILED", scope="large_offset_probe", error_type=type(error).__name__)
        if published:
            hold()
    finally:
        if published and not (args.publication / "STOP").is_file():
            hold()
        if owner is not None:
            try:
                owner.close()
            except BaseException as error:
                failed = True
                emit("CLEANUP_FAILED", error_type=type(error).__name__)
        if native is not None:
            counters = native.resource_counters()
            failed |= any(counters.values())
            emit("CLOSED" if not any(counters.values()) else "RESOURCES_REMAIN", counters=counters)
        (args.publication / "probe.json").unlink(missing_ok=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return int(failed)


def run_consumer(args):
    document, handle = read_publication(args.bundle)
    native, torch = native_and_torch(args.device)
    if len(handle) != native.fabric_handle_bytes:
        raise ValueError("INVALID_PROBE_CAPABILITY_LENGTH")
    from rtp_llm.models_py.modules.dsv41._engram_triton import lookup_host_rows

    device = torch.device("cuda", args.device)
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    free_before, _ = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    consumer, weight, scale, ids, output = None, None, None, None, None
    results = []
    try:
        consumer = native.Consumer(handle, document["allocation_bytes"], document["granularity"], args.device)
        del handle, document
        weight = consumer.tensor_view([ROWS, DIM], 0)
        scale = consumer.tensor_view([ROWS, SCALE_COLUMNS], WEIGHT_BYTES)
        after_attach = torch.cuda.memory_allocated(device)
        if after_attach != before:
            raise AssertionError("mapping unexpectedly allocated PyTorch HBM")
        if not weight.is_contiguous() or not scale.is_contiguous():
            raise AssertionError("native table views are not contiguous")
        selected = list(SAMPLE_ROWS) + [-1, ROWS, (1 << 31) - 1]
        expected_rows = [row_pattern(slot)[2] for slot in range(len(SAMPLE_ROWS))]
        expected_rows += [[0.0] * DIM for _ in range(len(selected) - len(SAMPLE_ROWS))]
        expected = torch.tensor([expected_rows], dtype=torch.bfloat16, device="cpu")
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        with torch.inference_mode():
            for dtype in (torch.int64, torch.int32):
                ids = torch.tensor([selected], dtype=dtype, device=device)
                output = lookup_host_rows(weight, scale, ids, num_sms)
                actual = output.cpu()
                if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
                    raise AssertionError("Triton BF16 bytes differ from independent CPU FP8/scale pattern")
                results.append({"index_dtype": str(dtype), "rows": selected,
                                "exact_bf16_bytes": True, "invalid_rows_zero": True})
        torch.cuda.synchronize(device)
        peak_delta = torch.cuda.max_memory_allocated(device) - before
        free_after, _ = torch.cuda.mem_get_info(device)
        # Views allocate no table HBM. Only tiny indices/outputs are managed by
        # PyTorch; the independent driver counter bounds hidden allocations too.
        if peak_delta > 64 << 20 or free_before - free_after > 512 << 20:
            raise AssertionError("HBM growth is inconsistent with a selected-row-only probe")
        emit("PASS", scope="large_offset_native_triton_only", cases=results,
             logical_shared_bytes=LOGICAL_BYTES, torch_attach_allocated_bytes=after_attach - before,
             torch_peak_delta_bytes=peak_delta, driver_free_delta_bytes=free_before - free_after,
             no_full_table_hbm_copy=True)
    finally:
        torch.cuda.synchronize(device)
        weight = scale = ids = output = None
        gc.collect()
        if consumer is not None:
            consumer.close()
        counters = native.resource_counters()
        emit("CLOSED" if not any(counters.values()) else "RESOURCES_REMAIN", counters=counters)
        if any(counters.values()):
            raise AssertionError("native consumer resources remain")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("owner", "consumer"))
    parser.add_argument("--publication", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--numa", type=int, default=0)
    parser.add_argument("--report-timeout", type=float, default=600)
    args = parser.parse_args(argv)
    if args.device < 0 or args.numa < 0 or not 0 < args.report_timeout < float("inf"):
        parser.error("valid device/NUMA ordinals and finite positive timeout required")
    target = args.publication if args.role == "owner" else args.bundle
    if target is None or not target.is_absolute():
        parser.error("owner needs absolute --publication; consumer needs absolute --bundle")
    return run_owner(args) if args.role == "owner" else run_consumer(args)


if __name__ == "__main__":
    raise SystemExit(main())
