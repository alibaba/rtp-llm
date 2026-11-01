"""Publish two sealed EGM fixture allocations; only STOP authorizes teardown.

Run from a checkout at tools/engram_sidecar/integration/owner_fixture.py.
The coordinator must stop all consumers before creating directory/STOP.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from uuid import uuid4

import fixture


def emit(status, **fields):
    # A lost SSH/stdout pipe must never trigger teardown of a published owner.
    try:
        print(json.dumps(dict(status=status, **fields), sort_keys=True), flush=True)
    except Exception:
        pass


def atomic_json(path, value, mode):
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def wait_for_stop(directory, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    reported = False
    sleeper = threading.Event()
    while not (directory / "STOP").is_file():
        if not reported and time.monotonic() >= deadline:
            emit("FAIL_TIMEOUT", action="retaining_owner_until_STOP")
            reported = True
        sleeper.wait(1.0)
    return reported


def create_manifest(native, owners, device, numa):
    from rtp_llm.models_py.modules.dsv41.engram_manifest import (
        LAYOUT_ABI, compute_dataset_id, parse_manifest, validate_for_attach,
    )

    generation, incarnation = str(uuid4()), str(uuid4())
    document = dict(schema_version=1, model_revision=fixture.MODEL_REVISION,
                    generation=generation, layout_abi=LAYOUT_ABI,
                    hash_abi=fixture.HASH_ABI, backend="fabric_egm",
                    state="READY", tensors=[])
    capabilities, descriptions = {}, []
    for layer, rows in fixture.LAYERS:
        weight, scale = fixture.make_table(rows)
        weight_bytes = bytes(weight.reshape(-1).tolist())
        scale_bytes = bytes(scale.reshape(-1).tolist())
        scale_offset = rows * fixture.DIM + 128
        owner = native.Owner(size=scale_offset + len(scale_bytes), numa=numa, device=device)
        owners.append(owner)
        owner.load(weight_bytes, offset=0)
        owner.load(scale_bytes, offset=scale_offset)
        info = owner.info()
        if scale_offset % info["granularity"] == 0:
            raise RuntimeError("fixture scale offset unexpectedly aligns to VMM granularity")
        allocation_id = str(uuid4())
        handle_ref = "fixture-capability-" + allocation_id
        for role, width, offset, payload in (
                ("weight", fixture.DIM, 0, weight_bytes),
                ("scale", fixture.DIM // 32, scale_offset, scale_bytes)):
            document["tensors"].append(dict(
                name=f"layers.{layer}.engram.embed.{role}", shape=[rows, width],
                dtype="float8_e4m3fn" if role == "weight" else "float8_e8m0fnu",
                storage_dtype="uint8", row_bytes=width, quant_block=32,
                shards=[dict(shard_id=str(uuid4()), row_begin=0, row_end=rows,
                             logical_bytes=len(payload), byte_offset=offset,
                             allocation_bytes=info["allocation_bytes"],
                             allocation_granularity=info["granularity"],
                             owner_id="engram-integration-owner", owner_incarnation=incarnation,
                             allocation_id=allocation_id, handle_ref=handle_ref,
                             sha256="sha256:" + hashlib.sha256(payload).hexdigest())]))
        # Both complete tensors are loaded and hashed before the owner is sealed.
        capabilities[allocation_id] = base64.b64encode(owner.export_handle()).decode("ascii")
        descriptions.append(dict(allocation_id=allocation_id,
                                 logical_bytes=info["logical_bytes"],
                                 allocation_bytes=info["allocation_bytes"]))
    document["dataset_id"] = compute_dataset_id(document)
    parsed = parse_manifest(json.dumps(document).encode())
    validate_for_attach(parsed, fixture.expected_layout(), document["dataset_id"], generation)
    return document, capabilities, descriptions


def main(manifest_factory=create_manifest):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--numa", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    args = parser.parse_args()
    if (not args.directory.is_absolute() or args.directory.is_symlink()
            or args.device < 0 or args.numa < 0
            or not 0 < args.timeout_seconds < float("inf")):
        parser.error("directory must be an absolute nonsymlink; device/numa >=0; timeout finite and >0")
    args.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if args.directory.stat().st_uid != os.getuid():
        parser.error("directory must belong to the current user")
    os.chmod(args.directory, 0o700)
    for name in ("STOP", "manifest.json", "capabilities.json"):
        if (args.directory / name).exists():
            parser.error("use a fresh publication directory without STOP/manifest/capabilities")
    lock_path = args.directory / ".owner.lock"
    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(lock_fd)

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parents[2]))  # checkout/tools/engram_sidecar/integration
    sys.path.insert(0, str(here.parent / "native"))
    owners, published, native, failed = [], False, None, False

    def stop_signal(signum, frame):
        nonlocal failed
        if not published:
            raise SystemExit(128 + signum)
        failed = True
        emit("FAIL_SIGNAL", signal=signum, action="retaining_owner_until_STOP")

    previous_handlers = {sig: signal.signal(sig, stop_signal)
                         for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        from load_native import load_native
        native = load_native()
        document, capabilities, descriptions = manifest_factory(native, owners, args.device, args.numa)
        # Mark potentially published before the first capability file can appear.
        published = True
        atomic_json(args.directory / "capabilities.json", capabilities, 0o600)
        atomic_json(args.directory / "manifest.json", document, 0o644)
        emit("READY", generation=document["generation"], allocations=descriptions,
             counters=native.resource_counters())
        failed = wait_for_stop(args.directory, args.timeout_seconds) or failed
    except Exception as error:
        failed = True
        emit("FAIL_OWNER", error_type=type(error).__name__)
        if published:
            wait_for_stop(args.directory, args.timeout_seconds)
    finally:
        # Publication makes STOP mandatory. A timeout never reaches this cleanup.
        if published and not (args.directory / "STOP").is_file():
            failed = True
            emit("FAIL_PENDING_STOP", action="retaining_owner_until_STOP")
            wait_for_stop(args.directory, args.timeout_seconds)
        cleanup_failed = False
        for owner in reversed(owners):
            try:
                owner.close()
            except Exception as error:
                cleanup_failed = True
                emit("FAIL_CLEANUP", error_type=type(error).__name__)
        if native is not None:
            counters = native.resource_counters()
            zero = all(value == 0 for value in counters.values())
            emit("CLOSED" if zero and not cleanup_failed else "FAIL_COUNTERS", counters=counters)
            failed = failed or cleanup_failed or not zero
        (args.directory / "capabilities.json").unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
