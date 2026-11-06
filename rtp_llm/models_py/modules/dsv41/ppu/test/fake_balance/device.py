"""Eight PPU ranks: actual grouped/MegaMoE receive loads and Graph replay.

Uses small synthetic expert weights with real 384-expert/top-6 EP8 geometry,
the production PPUMoE adapter, existing native fake-balance op and real native
communication/GEMMs. This qualifies routing, not model output accuracy or TPS.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv41.ppu.moe import PPUMoE

WORLD, EXPERTS, LOCAL, TOP_K, HIDDEN, INTERMEDIATE = 8, 384, 48, 6, 512, 128


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_engine(rank, backend, enabled):
    os.environ["RTP_V41_PPU_MOE_BACKEND"] = backend
    os.environ["RTP_V41_PPU_MOE_PRECISION"] = "w4a4" if backend == "megamoe" else "w4a8"
    value = PPUMoE(
        num_routed_experts=EXPERTS,
        num_local_experts=LOCAL,
        ep_rank=rank,
        ep_size=WORLD,
        hidden_size=HIDDEN,
        inter_dim=INTERMEDIATE,
        group=dist.group.WORLD,
    )
    value.configure_fake_balance(enabled, top_k=TOP_K, dp_rank=rank, dp_size=WORLD)
    generator = torch.Generator().manual_seed(730 + rank)
    pairs = []
    for n, k in ((2 * INTERMEDIATE, HIDDEN), (HIDDEN, INTERMEDIATE)):
        packed = torch.randint(
            256, (LOCAL, n, k // 2), generator=generator, dtype=torch.uint8
        ).cuda()
        scales = torch.full((LOCAL, n, k // 32), 120, dtype=torch.uint8, device="cuda")
        pairs.append((packed, scales))
    value.bind_expert_weights(0, *pairs)
    return value


def inputs(rows, live):
    x = torch.full((rows, HIDDEN), 0.125, dtype=torch.bfloat16, device="cuda")
    ids = torch.full((rows, 8), -1, dtype=torch.int64, device="cuda")
    ids[:, :TOP_K] = torch.arange(TOP_K, device="cuda")
    weights = torch.zeros((rows, 8), dtype=torch.float32, device="cuda")
    weights[:, :TOP_K] = 0.25
    ids[live:] = -1
    weights[live:] = 0
    return x, ids, weights


def receive_counts(engine, grouped_counts):
    if engine.backend == "megamoe":
        return engine._mega_recv_stats.clone().long()
    return grouped_counts.clone()


def validate_counts(engine, counts, name, enabled, live_counts):
    gathered = [torch.empty_like(counts) for _ in range(WORLD)]
    dist.all_gather(gathered, counts)
    per_expert = torch.cat(gathered).cpu()
    per_rank = per_expert.reshape(WORLD, LOCAL).sum(1)
    assignments = sum(live_counts) * TOP_K
    assert int(per_expert.sum()) == assignments, (name, per_expert.tolist())
    if enabled:
        assert int(per_rank.max() - per_rank.min()) <= 1, (name, per_rank.tolist())
        assert int(per_expert.max() - per_expert.min()) <= 1, (
            name,
            per_expert.tolist(),
        )
    else:
        # All unmodified inputs route to experts 0..5 on owner rank zero.
        assert per_rank.tolist() == [assignments] + [0] * 7, (name, per_rank.tolist())
        assert per_expert[:TOP_K].tolist() == [sum(live_counts)] * TOP_K
    return {
        "case": name,
        "backend": engine.backend,
        "precision": engine.precision,
        "fake_balance_expert": enabled,
        "live_rows_per_source_rank": live_counts,
        "received_assignments_per_rank": per_rank.tolist(),
        "received_assignments_per_expert": per_expert.tolist(),
        "total_assignments": assignments,
        "status": "PASS",
    }


def worker(args):
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=args.rendezvous,
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=300),
    )
    results = []
    try:
        for backend in ("grouped", "megamoe"):
            for enabled in (False, True):
                engine = make_engine(rank, backend, enabled)
                grouped_counts = torch.zeros(LOCAL, dtype=torch.int64, device="cuda")
                local_compute = engine._local_compute

                def counted_compute(x, ids, weights, counts, layer_id):
                    # Count the actual DeepEP receive result without replacing
                    # communication or computation. Diagnostics are outside TPS.
                    grouped_counts.add_(torch.as_tensor(counts, device="cuda"))
                    return local_compute(x, ids, weights, counts, layer_id)

                engine._local_compute = counted_compute
                try:
                    for rows, one_busy in (
                        (8, False),
                        (16, False),
                        (64, False),
                        (256, False),
                        (8, True),
                    ):
                        live = rows if not one_busy or rank == 0 else 0
                        x, ids, weights = inputs(live, live)
                        before = (
                            receive_counts(engine, grouped_counts)
                            if engine._mega_recv_stats is not None
                            or backend == "grouped"
                            else torch.zeros_like(grouped_counts)
                        )
                        output = engine(x, ids, weights, layer_id=0)
                        torch.cuda.synchronize()
                        assert torch.isfinite(
                            output
                        ).all(), "non-finite native MoE output"
                        counts = receive_counts(engine, grouped_counts) - before
                        live_counts = [rows] + [0] * 7 if one_busy else [rows] * WORLD
                        results.append(
                            validate_counts(
                                engine,
                                counts,
                                f"eager-b{rows}-idle{one_busy}",
                                enabled,
                                live_counts,
                            )
                        )
                    if backend == "megamoe":
                        # A static 16-row capture must honor masks changed on
                        # replay. Warm up all native operations before capture.
                        x, ids, weights = inputs(16, 16)
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(2):
                                engine(x, ids, weights, layer_id=0)
                        torch.cuda.current_stream().wait_stream(stream)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            graph_output = engine(x, ids, weights, layer_id=0)
                        for live in (8, 16, 0, 8):
                            new_x, new_ids, new_weights = inputs(16, live)
                            x.copy_(new_x)
                            ids.copy_(new_ids)
                            weights.copy_(new_weights)
                            before = engine._mega_recv_stats.clone()
                            graph.replay()
                            torch.cuda.synchronize()
                            assert torch.isfinite(graph_output).all()
                            assert torch.count_nonzero(graph_output[live:]) == 0
                            counts = engine._mega_recv_stats.long() - before.long()
                            results.append(
                                validate_counts(
                                    engine,
                                    counts,
                                    f"graph-b16-live{live}",
                                    enabled,
                                    [live] * WORLD,
                                )
                            )
                finally:
                    engine.close()
                    dist.barrier()
        write(
            args.output / f"rank-{rank}.json",
            {
                "rank": rank,
                "status": "PASS",
                "cases": results,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(rank),
            },
        )
    finally:
        PPUMoE.close_mega_resources()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rendezvous")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if (
        torch.cuda.device_count() != WORLD
        or "ZW-M890P" not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("this test requires eight allocated ZW-M890P devices")
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    children, logs = [], []
    started = time.time()
    try:
        for rank in range(WORLD):
            log = (args.output / f"rank-{rank}.log").open("x")
            logs.append(log)
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        __spec__.name,
                        "--worker",
                        "--output",
                        str(args.output),
                        "--rendezvous",
                        f"tcp://127.0.0.1:{port}",
                    ],
                    env={**os.environ, "RANK": str(rank), "WORLD_SIZE": str(WORLD)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        deadline = time.monotonic() + 1500
        codes = [
            child.wait(timeout=max(1, deadline - time.monotonic()))
            for child in children
        ]
        if any(codes):
            raise RuntimeError(f"PPU rank failures: {codes}; see rank logs")
        ranks = [
            json.loads((args.output / f"rank-{rank}.json").read_text())
            for rank in range(WORLD)
        ]
        assert all(value["status"] == "PASS" for value in ranks)
        assert all(value["cases"] == ranks[0]["cases"] for value in ranks)
        write(
            args.output / "summary.json",
            {
                "status": "PASS",
                "ranks": WORLD,
                "cases": ranks[0]["cases"],
                "elapsed_seconds": time.time() - started,
                "workload": "real native EP8 MoE; synthetic small weights, 384 experts/top-6; not full-model performance",
            },
        )
        print(
            json.dumps(
                {"status": "PASS", "ranks": WORLD, "cases": len(ranks[0]["cases"])}
            )
        )
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
