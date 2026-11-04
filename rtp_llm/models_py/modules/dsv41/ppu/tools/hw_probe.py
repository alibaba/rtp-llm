"""Standalone 8-rank hardware/collective probe for the D machine (L3).

Run inside the PPU SDK container on the D machine::

    /opt/conda312/bin/python rtp_llm/models_py/modules/dsv41/ppu/tools/hw_probe.py \
        --output /tmp/rtp-llm-v41-runs/hw-probe-<date>/probe.json

One process per cuda:N (8 x ZW-M890P) joined by
``torch.distributed.init_process_group(backend="nccl")`` (PCCL on this
stack, the same path sglang-for-sail uses). The default launcher is
self-spawned subprocesses with ``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` +
``MASTER_ADDR``/``MASTER_PORT`` on a free port; launching through
``torchrun --nproc_per_node=8`` is also supported (the worker detects the
environment). The parent bounds the whole run with a timeout and kills
ONLY the processes it spawned.

Probes (each pass/fail/exception, in rank order with barriers):

(a) collectives on device tensors: all_reduce bf16/fp32,
    all_gather_into_tensor bf16 AND int64, broadcast, barrier;
(b) host pinned memory: ``torch.empty(..., pin_memory=True)`` + H2D copy;
(c) import + symbol probes: deep_gemm (m_grouped_gemm_w4a16_nopad,
    preprocess_mxfp4_scales), deep_ep (Buffer), flash_mla
    (flash_mla_sparse_fwd), hadamard;
(d) tiny device bf16 matmul + fp8->bf16 dequant sanity (finite check).

The script is dependency-free apart from torch and the probed wheels; it
never imports rtp_llm. Total runtime is bounded well under 5 minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

PROBE_TIMEOUT_DEFAULT = 240  # seconds, parent-side hard bound


# ---------------------------------------------------------------------------
# Probe helpers (worker side)
# ---------------------------------------------------------------------------


def _result(status: str, detail: str = "") -> Dict[str, str]:
    return {"status": status, "detail": detail}


def _run_probe(probes: Dict[str, Any], name: str, fn) -> None:
    try:
        detail = fn()
        probes[name] = _result("pass", detail or "ok")
    except Exception as error:  # noqa: BLE001 - a probe failure is data
        probes[name] = _result("exception", f"{type(error).__name__}: {error}"[:500])


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def worker_main(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda", local_rank)
    probes: Dict[str, Any] = {}
    dist_initialized = False

    def _device_visible() -> str:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        return (
            f"cuda_count={torch.cuda.device_count()} "
            f"name={torch.cuda.get_device_name(device)}"
        )

    _run_probe(probes, "device_visible", _device_visible)

    def _init():
        nonlocal dist_initialized
        dist.init_process_group(backend="nccl", init_method="env://")
        dist_initialized = True
        return f"rank={rank}/{world_size} backend=nccl device={device}"

    _run_probe(probes, "init_process_group_nccl", _init)

    if dist_initialized:

        def _barrier():
            dist.barrier()
            return "ok"

        # ---- (a) collectives on device tensors --------------------------
        def _all_reduce_bf16():
            tensor = torch.full(
                (1024,), float(rank + 1), dtype=torch.bfloat16, device=device
            )
            dist.all_reduce(tensor)
            value = float(tensor[0].item())
            expected = world_size * (world_size + 1) / 2
            if abs(value - expected) > 0.5:
                raise RuntimeError(f"bf16 all_reduce value {value} != {expected}")
            return f"sum={value}"

        def _all_reduce_fp32():
            tensor = torch.full(
                (1024,), float(rank + 1), dtype=torch.float32, device=device
            )
            dist.all_reduce(tensor)
            expected = float(world_size * (world_size + 1) / 2)
            if not torch.allclose(tensor, torch.full_like(tensor, expected)):
                raise RuntimeError("fp32 all_reduce mismatch")
            return f"sum={expected}"

        def _all_gather_into_tensor_bf16():
            local = torch.full(
                (256,), float(rank + 1), dtype=torch.bfloat16, device=device
            )
            out = torch.empty(world_size * 256, dtype=torch.bfloat16, device=device)
            dist.all_gather_into_tensor(out, local)
            for other in range(world_size):
                block = out[other * 256 : (other + 1) * 256].float()
                if not torch.allclose(block, torch.full_like(block, float(other + 1))):
                    raise RuntimeError(f"bf16 gather block {other} mismatch")
            return "rank blocks verified"

        def _all_gather_into_tensor_int64():
            local = torch.arange(64, dtype=torch.int64, device=device) + rank * 1000
            out = torch.empty(world_size * 64, dtype=torch.int64, device=device)
            dist.all_gather_into_tensor(out, local)
            for other in range(world_size):
                block = out[other * 64 : (other + 1) * 64]
                expected = (
                    torch.arange(64, dtype=torch.int64, device=device) + other * 1000
                )
                if not torch.equal(block, expected):
                    raise RuntimeError(f"int64 gather block {other} mismatch")
            return "rank blocks verified exactly"

        def _broadcast():
            tensor = torch.full(
                (128,), 7.5 if rank == 0 else -1.0, dtype=torch.float32, device=device
            )
            dist.broadcast(tensor, src=0)
            if not torch.allclose(tensor, torch.full_like(tensor, 7.5)):
                raise RuntimeError("broadcast mismatch")
            return "root=0 value verified"

        for name, fn in (
            ("all_reduce_bf16", _all_reduce_bf16),
            ("all_reduce_fp32", _all_reduce_fp32),
            ("all_gather_into_tensor_bf16", _all_gather_into_tensor_bf16),
            ("all_gather_into_tensor_int64", _all_gather_into_tensor_int64),
            ("broadcast", _broadcast),
            ("barrier", _barrier),
        ):
            dist.barrier()  # rank-ordered probe entry
            _run_probe(probes, name, fn)
            dist.barrier()

    # ---- per-rank probes, executed strictly in rank order ----------------
    if dist_initialized:
        import torch.distributed as dist

        for other in range(world_size):
            if other == rank:
                _probe_local_ranks(probes, torch, device)
            dist.barrier()

    # ---- (c) import / symbol probes (every rank; cheap) ------------------
    _probe_imports(probes)

    if dist_initialized:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()

    payload = {
        "rank": rank,
        "local_rank": local_rank,
        "device": str(device),
        "torch": torch.__version__,
        "probes": probes,
        "all_pass": all(p["status"] == "pass" for p in probes.values()),
    }
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    rank_path = os.path.join(out_dir, f"hw_probe.rank{rank}.json")
    with open(rank_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[hw_probe rank {rank}] wrote {rank_path}", flush=True)

    # torchrun mode: rank 0 aggregates the sibling files after the last
    # barrier so a single --output document exists either way.
    if rank == 0 and os.environ.get("HW_PROBE_LAUNCHER") == "torchrun-env":
        _aggregate(args, world_size, wait=True)
    return 0


def _probe_local_ranks(probes, torch, device) -> None:
    """Pinned memory + compute sanity, in rank order."""

    def _pin_memory():
        host = torch.empty(1 << 20, dtype=torch.uint8, pin_memory=True)
        if not host.is_pinned():
            raise RuntimeError("pin_memory=True tensor reports is_pinned() False")
        host[:64] = torch.arange(64, dtype=torch.uint8)
        device_copy = torch.empty(1 << 20, dtype=torch.uint8, device=device)
        device_copy.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        if not torch.equal(device_copy[:64].cpu(), host[:64].cpu()):
            raise RuntimeError("H2D pinned copy mismatch")
        return "pinned=1MiB h2d ok"

    def _bf16_matmul():
        a = torch.randn(64, 64, dtype=torch.bfloat16, device=device)
        b = torch.randn(64, 64, dtype=torch.bfloat16, device=device)
        c = a @ b
        if not torch.isfinite(c.float()).all():
            raise RuntimeError("bf16 matmul produced non-finite values")
        return "64x64 finite"

    def _fp8_dequant():
        raw = torch.randint(-120, 120, (64,), dtype=torch.int32).to(torch.float32)
        fp8 = raw.to(torch.float8_e4m3fn)
        bf16 = fp8.to(torch.bfloat16)
        if not torch.isfinite(bf16.float()).all():
            raise RuntimeError("fp8->bf16 dequant produced non-finite values")
        return "e4m3->bf16 finite"

    _run_probe(probes, "pin_memory_h2d", _pin_memory)
    _run_probe(probes, "bf16_matmul_finite", _bf16_matmul)
    _run_probe(probes, "fp8_to_bf16_finite", _fp8_dequant)


def _probe_imports(probes) -> None:
    import importlib

    def _module_symbols(module_name: str, symbols: List[str]) -> Any:
        module = importlib.import_module(module_name)
        found = [
            symbol for symbol in symbols if callable(getattr(module, symbol, None))
        ]
        missing = [symbol for symbol in symbols if symbol not in found]
        if missing:
            raise RuntimeError(f"{module_name} missing symbols {missing}")
        return f"symbols={found}"

    _run_probe(
        probes,
        "deep_gemm_symbols",
        lambda: _module_symbols(
            "deep_gemm", ["m_grouped_gemm_w4a16_nopad", "preprocess_mxfp4_scales"]
        ),
    )
    _run_probe(
        probes, "deep_ep_symbols", lambda: _module_symbols("deep_ep", ["Buffer"])
    )
    _run_probe(
        probes,
        "flash_mla_symbols",
        lambda: _module_symbols("flash_mla", ["flash_mla_sparse_fwd"]),
    )

    def _hadamard():
        # The A0-installed wheel is `fast_hadamard_transform` (backed by the
        # `fast_hadamard_transform_cuda` extension); plain `hadamard` is the
        # upstream name some builds use, so both candidates are probed.
        last_error: Optional[Exception] = None
        for module_name in ("hadamard", "fast_hadamard_transform"):
            try:
                module = importlib.import_module(module_name)
            except Exception as error:  # noqa: BLE001 - recorded per candidate
                last_error = error
                continue
            candidates = [
                name
                for name in (
                    "get_hadamard_matrix",
                    "hadamard_matrix",
                    "create_hadamard_matrix",
                    "hadamard_transform",
                    "hadamard_transform_1d",
                    "hadamard_transform_2d",
                )
                if callable(getattr(module, name, None))
            ]
            return f"module={module_name} symbols={candidates}"
        raise RuntimeError(f"no hadamard module importable: {last_error}")

    _run_probe(probes, "hadamard_symbols", _hadamard)


# ---------------------------------------------------------------------------
# Parent / aggregation
# ---------------------------------------------------------------------------


def _aggregate(
    args: argparse.Namespace, world_size: int, wait: bool = False
) -> Dict[str, Any]:
    out_dir = os.path.dirname(os.path.abspath(args.output))
    deadline = time.time() + 30
    payloads: List[Dict[str, Any]] = []
    while True:
        payloads = []
        for rank in range(world_size):
            path = os.path.join(out_dir, f"hw_probe.rank{rank}.json")
            if not os.path.exists(path):
                break
            with open(path, "r", encoding="utf-8") as handle:
                payloads.append(json.load(handle))
        if len(payloads) == world_size or not wait or time.time() > deadline:
            break
        time.sleep(1.0)

    by_probe: Dict[str, Dict[str, int]] = {}
    failed: List[str] = []
    for payload in payloads:
        for name, result in payload.get("probes", {}).items():
            bucket = by_probe.setdefault(
                name, {"pass": 0, "exception": 0, "missing": 0}
            )
            status = result.get("status", "missing")
            key = status if status in bucket else "exception"
            bucket[key] += 1
            if status != "pass":
                failed.append(
                    f"rank{payload.get('rank')}:{name} ({result.get('detail', '')[:120]})"
                )
    document = {
        "launcher": os.environ.get("HW_PROBE_LAUNCHER", "self-spawn-subprocess"),
        "world_size": world_size,
        "timeout_sec": int(args.timeout),
        "ranks": payloads,
        "summary": {
            "ranks_reported": len(payloads),
            "all_pass": bool(payloads) and all(p.get("all_pass") for p in payloads),
            "failed": failed,
            "by_probe": by_probe,
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    _print_summary(document, args.output)
    return document


def _print_summary(document: Dict[str, Any], output_path: str) -> None:
    print("=" * 72, flush=True)
    print(
        f"HW probe summary ({document['launcher']}, world_size={document['world_size']})",
        flush=True,
    )
    print(f"output: {output_path}", flush=True)
    summary = document["summary"]
    print(
        f"ranks reported: {summary['ranks_reported']}/{document['world_size']}",
        flush=True,
    )
    for name, counts in sorted(summary["by_probe"].items()):
        total = sum(counts.values())
        status = "PASS" if counts.get("pass", 0) == total else "FAIL"
        print(f"  [{status}] {name}: {counts.get('pass', 0)}/{total} ranks", flush=True)
    for line in summary["failed"][:40]:
        print(f"    ! {line}", flush=True)
    print(f"ALL PASS: {summary['all_pass']}", flush=True)
    print("=" * 72, flush=True)


def parent_main(args: argparse.Namespace) -> int:
    """Self-spawn one worker process per rank; bound with a timeout."""
    master_port = _free_port()
    env = dict(os.environ)
    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "WORLD_SIZE": str(args.nproc),
            "HW_PROBE_LAUNCHER": "self-spawn-subprocess",
        }
    )
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    processes: List[subprocess.Popen] = []
    for rank in range(args.nproc):
        rank_env = dict(env)
        rank_env["RANK"] = str(rank)
        rank_env["LOCAL_RANK"] = str(rank)
        command = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            "--output",
            os.path.abspath(args.output),
        ]
        log_path = os.path.join(out_dir, f"hw_probe.rank{rank}.log")
        log = open(log_path, "w", encoding="utf-8")
        processes.append(
            subprocess.Popen(
                command, env=rank_env, stdout=log, stderr=subprocess.STDOUT
            )
        )
        log.close()
    deadline = time.time() + int(args.timeout)
    try:
        for process in processes:
            remaining = max(0.0, deadline - time.time())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print(
                    f"[hw_probe] timeout after {args.timeout}s; terminating spawned workers",
                    file=sys.stderr,
                    flush=True,
                )
                break
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    document = _aggregate(args, args.nproc)
    return 0 if document["summary"]["ranks_reported"] == args.nproc else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="final JSON path")
    parser.add_argument("--nproc", type=int, default=8, help="ranks to launch")
    parser.add_argument(
        "--timeout",
        type=int,
        default=PROBE_TIMEOUT_DEFAULT,
        help="parent-side bound (s)",
    )
    parser.add_argument("--worker", action="store_true", help="internal: worker mode")
    args = parser.parse_args()

    in_distributed_env = (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and "MASTER_ADDR" in os.environ
    )
    if args.worker or in_distributed_env:
        if not in_distributed_env:
            parser.error(
                "--worker requires RANK/WORLD_SIZE/MASTER_ADDR in the environment"
            )
        if os.environ.get("HW_PROBE_LAUNCHER") is None:
            os.environ["HW_PROBE_LAUNCHER"] = "torchrun-env"
        return worker_main(args)
    return parent_main(args)


if __name__ == "__main__":
    sys.exit(main())
