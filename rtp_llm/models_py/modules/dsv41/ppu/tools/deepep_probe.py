#!/usr/bin/env python
"""DeepEP dispatch/combine weight-semantics probe (L3, wave-2 mission C).

Resolves the L1 open question in ``ppu/moe.py``'s module docstring: does
``deep_ep.Buffer.combine`` re-apply the top-k dispatch weights (making
``ep_gather``'s explicit weight application a DOUBLE-apply), or does the
weight flow end at the local compute?

Method (EP8, one process per rank, tiny tensors, identity "experts"):

* every rank owns M tokens with ONE routed expert each (8-slot layout, slot
  0 valid, the rest ``-1``/``0.0`` padding, exactly the dispatch format
  ``pad_topk_for_dispatch`` produces);
* Experiment A — ``y_recv = x_recv`` (raw identity): the combined output is
  ``x`` if combine never scales, ``w * x`` if combine applies the weight;
* Experiment B — ``y_recv = w_recv * x_recv`` (the moe.py convention: local
  compute applies the received weight, as ``ep_gather`` does): the combined
  output is ``w * x`` if combine does not re-scale (moe.py CORRECT), ``w^2 *
  x`` if it double-applies;
* the received rows themselves are checked for dispatch-side scaling
  (``x_recv == x`` vs ``w * x``).

Run on the D machine inside the PPU container::

    /opt/conda312/bin/python rtp_llm/models_py/modules/dsv41/ppu/tools/deepep_probe.py \
        --output /tmp/rtp-llm-v41-runs/<run>/deepep_probe.json

Self-spawns one worker per rank (hw_probe.py launcher pattern) and bounds
the run with a parent-side timeout.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List

PROBE_TIMEOUT_DEFAULT = 300

HIDDEN = 5120
NUM_EXPERTS = 384
SLOTS = 8
TOKENS_PER_RANK = 8


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _result(status: str, detail: str = "") -> Dict[str, str]:
    return {"status": status, "detail": detail}


def _classify(out: Any, x: Any, weight: float) -> str:
    """Which scaling hypothesis ``out`` matches (relative error < 1e-2)."""
    base = x.float()
    value = out.float()
    hypotheses = {
        "unscaled": base,
        "weight_once": base * weight,
        "weight_squared": base * weight * weight,
    }
    best, best_error = None, None
    for name, expected in hypotheses.items():
        error = ((value - expected) / expected.clamp_min(1e-6)).abs().max().item()
        if best_error is None or error < best_error:
            best, best_error = name, error
    return (
        f"{best} (max rel err {best_error:.2e})"
        if best_error < 1e-2
        else (
            f"no hypothesis (errors vs unscaled/once/squared: "
            f"{((value - hypotheses['unscaled']) / hypotheses['unscaled'].clamp_min(1e-6)).abs().max().item():.2e}/"
            f"{((value - hypotheses['weight_once']) / hypotheses['weight_once'].clamp_min(1e-6)).abs().max().item():.2e}/"
            f"{((value - hypotheses['weight_squared']) / hypotheses['weight_squared'].clamp_min(1e-6)).abs().max().item():.2e})"
        )
    )


def worker_main(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl", init_method="env://")
    probes: Dict[str, Any] = {}

    try:
        import deep_ep

        buffer_cls = deep_ep.Buffer

        def _make_buffer():
            hidden_bytes = HIDDEN * 2
            num_nvl_bytes = num_rdma_bytes = 0
            for config in (
                buffer_cls.get_dispatch_config(world_size),
                buffer_cls.get_combine_config(world_size),
            ):
                num_nvl_bytes = max(
                    config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
                    num_nvl_bytes,
                )
                num_rdma_bytes = max(
                    config.get_rdma_buffer_size_hint(hidden_bytes, world_size),
                    num_rdma_bytes,
                )
            return buffer_cls(
                dist.group.WORLD,
                num_nvl_bytes,
                num_rdma_bytes,
                low_latency_mode=False,
                num_qps_per_rank=24,
            )

        def _run_experiment(buffer, mode: str) -> Dict[str, Any]:
            generator = torch.Generator().manual_seed(1234 + rank)
            x = (
                torch.randn(TOKENS_PER_RANK, HIDDEN, generator=generator)
                .to(torch.bfloat16)
                .to(device)
            )
            weights = (
                0.25 + 0.5 * torch.rand(TOKENS_PER_RANK, generator=generator)
            ).to(torch.float32)
            # One routed expert per token (slot 0), padded to 8 slots. All
            # dispatch-side tensors live ON DEVICE (deep_ep kernels read them
            # directly; CPU tensors caused hggcErrorIllegalAddress).
            experts = (
                torch.arange(TOKENS_PER_RANK) * (world_size * 3) + rank
            ) % NUM_EXPERTS
            ids = torch.full(
                (TOKENS_PER_RANK, SLOTS), -1, dtype=torch.int64, device=device
            )
            ids[:, 0] = experts.to(device)
            weights_p = torch.zeros(
                TOKENS_PER_RANK, SLOTS, dtype=torch.float32, device=device
            )
            weights_p[:, 0] = weights.to(device)

            (
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                num_tokens_per_expert,
                is_token_in_rank,
                _,
            ) = buffer.get_dispatch_layout(ids, NUM_EXPERTS)
            (
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert,
                handle,
                _,
            ) = buffer.dispatch(
                x,
                None,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                ids,
                weights_p,
                expert_alignment=1,
            )
            # What does the recv side carry for the first received row?
            recv_weight_scale = None
            if recv_x.size(0):
                slot = int(recv_topk_idx[0, 0].item())
                recv_weight_scale = {
                    "recv_topk_idx_row0": [int(v) for v in recv_topk_idx[0].tolist()],
                    "recv_topk_weights_row0": [
                        float(v) for v in recv_topk_weights[0].tolist()
                    ],
                    "recv_x_row0_absmax": float(recv_x[0].float().abs().max()),
                    "token0_absmax": float(x[0].float().abs().max()),
                }
            if mode == "identity":
                y_local = recv_x.clone()
            elif mode == "constant_scale":
                # Control: a plain constant scale must come back as 0.5 * x.
                y_local = recv_x * 0.5
            else:  # "weights_applied": the moe.py ep_gather convention
                y_local = torch.zeros_like(recv_x)
                for row in range(recv_x.size(0)):
                    valid = recv_topk_idx[row] >= 0
                    w = recv_topk_weights[row, valid].to(recv_x.dtype)
                    y_local[row] = recv_x[row] * w.sum()
            y_combined, _, _ = buffer.combine(y_local.to(torch.bfloat16), handle)
            classification = _classify(y_combined[0], x[0], float(weights[0]))
            return {
                "recv_rows": int(recv_x.size(0)),
                "recv_side": recv_weight_scale,
                "combined_shape": list(y_combined.shape),
                "classification": classification,
            }

        def _probe_all():
            # ONE buffer reused across experiments: creating several Buffers
            # per rank corrupted the QP state on this build (garbage combine
            # outputs), so every experiment shares a single buffer.
            buffer = _make_buffer()
            results = {}
            for mode, name in (
                ("identity", "experiment_a_identity"),
                ("constant_scale", "experiment_b_constant_scale"),
                ("weights_applied", "experiment_c_weights_applied_locally"),
            ):
                try:
                    results[name] = dict(
                        _result("pass"), **_run_experiment(buffer, mode)
                    )
                except Exception as error:  # noqa: BLE001
                    results[name] = _result(
                        "exception", f"{type(error).__name__}: {error}"[:500]
                    )
                dist.barrier()
            return results

        try:
            probes.update(_probe_all())
        except Exception as error:  # noqa: BLE001
            probes["setup"] = _result(
                "exception", f"{type(error).__name__}: {error}"[:500]
            )

    except Exception as error:  # noqa: BLE001
        probes["setup"] = _result("exception", f"{type(error).__name__}: {error}"[:500])
    finally:
        dist.barrier()
        dist.destroy_process_group()

    payload = {
        "rank": rank,
        "world_size": world_size,
        "device": str(device),
        "probes": probes,
    }
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    rank_path = os.path.join(out_dir, f"deepep_probe.rank{rank}.json")
    with open(rank_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[deepep_probe rank {rank}] wrote {rank_path}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Parent / aggregation
# ---------------------------------------------------------------------------


def _aggregate(output: str, world_size: int) -> Dict[str, Any]:
    out_dir = os.path.dirname(os.path.abspath(output))
    deadline = time.time() + 30
    payloads: List[Dict[str, Any]] = []
    while True:
        payloads = []
        for rank in range(world_size):
            path = os.path.join(out_dir, f"deepep_probe.rank{rank}.json")
            if not os.path.exists(path):
                break
            with open(path, "r", encoding="utf-8") as handle:
                payloads.append(json.load(handle))
        if len(payloads) == world_size or time.time() > deadline:
            break
        time.sleep(1.0)

    classifications = {
        "experiment_a_identity": [],
        "experiment_b_constant_scale": [],
        "experiment_c_weights_applied_locally": [],
    }
    recv_scales: List[Any] = []
    failed: List[str] = []
    for payload in payloads:
        for name, probe in payload.get("probes", {}).items():
            if probe.get("status") != "pass":
                failed.append(
                    f"rank{payload.get('rank')}:{name} ({probe.get('detail', '')[:120]})"
                )
                continue
            if name in classifications:
                classifications[name].append(probe["classification"])
            if probe.get("recv_side"):
                recv_scales.append(probe["recv_side"])

    def _consensus(entries: List[str]) -> str:
        if not entries:
            return "no data"
        distinct = {entry.split(" (")[0] for entry in entries}
        if len(distinct) == 1:
            return f"ALL ranks: {entries[0]}"
        return f"MIXED: {entries}"

    conclusion_a = _consensus(classifications["experiment_a_identity"])
    conclusion_b = _consensus(classifications["experiment_b_constant_scale"])
    conclusion_c = _consensus(classifications["experiment_c_weights_applied_locally"])
    # Decision rule for ppu/moe.py: experiment C is exactly what moe.py does
    # with identity experts (weights applied in the local compute / ep_gather).
    if "weight_squared" in conclusion_c:
        verdict = (
            "DOUBLE-APPLY: deep_ep combine re-applies the dispatch weights; "
            "ppu/moe.py's ep_gather weight application double-scales and needs "
            "a fix"
        )
    elif "weight_once" in conclusion_c:
        verdict = (
            "SINGLE-APPLY: combine does not re-scale; ppu/moe.py's ep_gather "
            "weight application is CORRECT (weights applied exactly once)"
        )
    else:
        verdict = f"INCONCLUSIVE (experiment C: {conclusion_c}; B: {conclusion_b})"

    document = {
        "world_size": world_size,
        "hidden": HIDDEN,
        "num_experts": NUM_EXPERTS,
        "ranks": payloads,
        "summary": {
            "ranks_reported": len(payloads),
            "failed": failed,
            "experiment_a_identity": conclusion_a,
            "experiment_b_constant_scale": conclusion_b,
            "experiment_c_weights_applied_locally": conclusion_c,
            "recv_side_samples": recv_scales[:4],
            "verdict": verdict,
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    print(json.dumps(document["summary"], indent=2), flush=True)
    return document


def parent_main(args: argparse.Namespace) -> int:
    master_port = _free_port()
    env = dict(os.environ)
    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "WORLD_SIZE": str(args.nproc),
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
        log_path = os.path.join(out_dir, f"deepep_probe.rank{rank}.log")
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
                    f"[deepep_probe] timeout after {args.timeout}s; terminating",
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
    document = _aggregate(args.output, args.nproc)
    reported = document["summary"]["ranks_reported"]
    return 0 if reported == args.nproc and not document["summary"]["failed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="final JSON path")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=PROBE_TIMEOUT_DEFAULT)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker_main(args)
    return parent_main(args)


if __name__ == "__main__":
    sys.exit(main())
