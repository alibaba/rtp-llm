#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Small single-idle-GPU Python stack probe; no EGM, model, NCCL or collectives.

Run with an external timeout. Selection is an idle snapshot, not a reservation.
"""
import argparse
import csv
import importlib
import json
import os
import subprocess
import sys


def emit(status, stage, **fields):
    print(json.dumps({"status": status, "stage": stage, **fields}, ensure_ascii=False), flush=True)


def gpu_inventory():
    command = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
    if completed.returncode:
        raise RuntimeError(f"nvidia-smi failed: rc={completed.returncode}, stderr={completed.stderr[:1024]}")
    devices = []
    for row in csv.reader(completed.stdout.splitlines()):
        if len(row) != 4:
            raise RuntimeError("unexpected nvidia-smi query format")
        index, uuid, memory, util = [field.strip() for field in row]
        try:
            memory, util = float(memory), float(util)
        except ValueError:
            memory, util = None, None
        devices.append(dict(nvidia_smi_index=index, uuid=uuid, memory_used_mib=memory,
                            utilization_percent=util,
                            idle=memory is not None and 0 <= memory < 64 and util == 0))
    return devices


def select_idle_gpu():
    devices = gpu_inventory()
    emit("INFO", "gpu_idle_inventory", devices=devices, threshold="memory.used < 64 MiB and utilization.gpu == 0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
        if not entries or "-1" in entries:
            return None, "CUDA_VISIBLE_DEVICES disables GPU access"
        # Numeric CUDA ordinals are not guaranteed to equal nvidia-smi indexes.
        # Fail closed rather than widen an existing caller-imposed GPU scope.
        if not all(entry.startswith("GPU-") for entry in entries):
            return None, "existing numeric/MIG CUDA_VISIBLE_DEVICES cannot be safely matched to nvidia-smi UUIDs"
        allowed = set()
        for entry in entries:
            matches = [d["uuid"] for d in devices if d["uuid"].startswith(entry)]
            if len(matches) != 1:
                return None, "CUDA_VISIBLE_DEVICES UUID is unavailable or ambiguous"
            allowed.add(matches[0])
        devices = [d for d in devices if d["uuid"] in allowed]
    for device in devices:
        if device["idle"]:
            return device, None
    return None, "no eligible GPU has memory.used < 64 MiB and utilization.gpu == 0"


def _add_1024(a, b, output):
    # tl is populated before this function is passed to triton.jit.
    offsets = tl.arange(0, 1024)
    left = tl.load(a + offsets)
    right = tl.load(b + offsets)
    tl.store(output + offsets, left + right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-only", action="store_true", help="import/version checks only; never query/init CUDA")
    args = parser.parse_args()
    stage = "select_idle_gpu"
    candidate, skip_reason = None, "--import-only requested"
    if not args.import_only:
        try:
            candidate, skip_reason = select_idle_gpu()
        except Exception as exc:
            skip_reason = f"idle query unavailable: {type(exc).__name__}: {exc}"
        if candidate is not None:
            # Done before importing torch/Triton or initializing CUDA.
            os.environ["CUDA_VISIBLE_DEVICES"] = candidate["uuid"]
            emit("INFO", "gpu_selected", uuid=candidate["uuid"], cuda_ordinal=0,
                 nvidia_smi_index=candidate["nvidia_smi_index"], scope="one UUID only")

    torch = None
    cuda_started = False
    try:
        stage = "imports"
        torch = importlib.import_module("torch")
        triton = importlib.import_module("triton")
        globals()["tl"] = importlib.import_module("triton.language")
        versions = dict(torch=torch.__version__, triton=triton.__version__, torch_cuda=torch.version.cuda,
                        python=sys.version.split()[0])
        emit("PASS", stage, **versions)
        stage = "versions"
        if versions["torch"] != "2.11.0+cu130" or versions["triton"] != "3.6.0":
            raise RuntimeError("installed torch/Triton differs from main 492ecf45 ARM lock")
        emit("PASS", stage, expected_main_commit="492ecf45b3ef55b22fd4882622abfdb2e2f664f2", **versions)

        if candidate is None:
            emit("GPU_TEST_SKIPPED", "gpu_test", reason=skip_reason, import_passed=True)
            return 0

        # Import can take seconds. Check the same physical card again before any
        # torch.cuda call; never silently switch to a different GPU after import.
        stage = "gpu_idle_recheck"
        try:
            current = next((d for d in gpu_inventory() if d["uuid"] == candidate["uuid"]), None)
        except Exception as exc:
            emit("GPU_TEST_SKIPPED", stage, reason=f"recheck unavailable: {type(exc).__name__}: {exc}", import_passed=True)
            return 0
        if current is None or not current["idle"]:
            emit("GPU_TEST_SKIPPED", stage, reason="selected GPU became busy or disappeared", observed=current, import_passed=True)
            return 0
        emit("PASS", stage, **current)

        stage = "cuda_initialize"
        cuda_started = True
        torch.cuda.set_device(0)
        if torch.cuda.device_count() != 1:
            raise RuntimeError("expected exactly one GPU after UUID isolation")
        properties = torch.cuda.get_device_properties(0)
        emit("PASS", stage, uuid=candidate["uuid"], name=properties.name,
             capability=[properties.major, properties.minor])

        stage = "triton_eager"
        add = triton.jit(_add_1024)
        x = torch.arange(1024, device="cuda:0", dtype=torch.float32)
        y = torch.full((1024,), 2, device="cuda:0", dtype=torch.float32)
        output = torch.empty_like(x)
        add[(1,)](x, y, output, num_warps=4)
        torch.cuda.synchronize()
        expected = torch.arange(1024, dtype=torch.float32) + 2
        if not torch.equal(output.cpu(), expected):
            raise RuntimeError("eager add result differs from the CPU reference")
        emit("PASS", stage, elements=1024, dtype="float32", exact_match=True)

        stage = "graph_warmup"
        current_stream = torch.cuda.current_stream()
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                add[(1,)](x, y, output, num_warps=4)
        current_stream.wait_stream(warmup_stream)
        torch.cuda.synchronize()
        emit("PASS", stage, warmup_iterations=3)

        stage = "graph_capture"
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            add[(1,)](x, y, output, num_warps=4)
        emit("PASS", stage)

        stage = "graph_replay"
        for iteration in range(5):
            # Mutate values, retain addresses, and verify each replay recomputes.
            x.fill_(iteration)
            y.fill_(iteration + 1)
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.full((1024,), 2 * iteration + 1, dtype=torch.float32)
            if not torch.equal(output.cpu(), expected):
                raise RuntimeError(f"graph replay {iteration} differs from the CPU reference")
            emit("PASS", stage, iteration=iteration, elements=1024, exact_match=True)

        stage = "cleanup"
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(0)
        del graph  # Graph released before captured tensor addresses.
        del output, x, y, warmup_stream
        torch.cuda.empty_cache()
        emit("PASS", "complete", scope="python_stack_only", uuid=candidate["uuid"],
             eager_passed=True, graph_replays=5, logical_tensor_bytes=3 * 1024 * 4,
             torch_peak_allocated_bytes=peak,
             note="CUDA context/compiler overhead is additional; no EGM, model, NCCL or collective was tested")
        return 0
    except Exception as exc:
        emit("FAIL", stage, error_type=type(exc).__name__, message=str(exc))
        if cuda_started and torch is not None:
            try:
                torch.cuda.synchronize()
                emit("PASS", "failure_path_drain")
            except Exception as drain_exc:
                emit("CLEANUP_DEFERRED", "failure_path_drain", error_type=type(drain_exc).__name__,
                     message=str(drain_exc), action="standalone process exit")
        return 1


if __name__ == "__main__":
    sys.exit(main())
