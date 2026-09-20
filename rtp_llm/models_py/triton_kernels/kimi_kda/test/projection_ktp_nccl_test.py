"""Manual real-NCCL integration, launched with GPU_COUNT=1,2,4,8.

Uses real production collective wrappers and rank-tagged projection callbacks
instead of model weights. M is local physical rows; gathered M is M * ranks.
Rank 1 exercises wrapper copy/identity behavior (no network NCCL operation).
"""

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed import collective_torch as collectives
from rtp_llm.models_py.modules.kimi_k3.projection_ktp import (
    KtpProjectionWorkspace,
    project_kda_inputs_ktp,
)

assert_exact = partial(torch.testing.assert_close, rtol=0, atol=0)


def expected_fields(local, ranks):
    """Independent owner-major oracle, never invokes either pack/unpack path."""
    heads, dim, latent = 96 // ranks, 128, 128
    width = heads * dim
    fields = {key: [] for key in ("q", "k", "v", "output_gate", "raw_gate", "raw_beta")}
    for source in range(ranks):
        for name, section in (("q", 0), ("k", 1), ("v", 2), ("output_gate", 3)):
            offsets = (torch.arange(section * width, (section + 1) * width) % 19).to(
                torch.bfloat16
            )
            fields[name].append(local[:, :1] + offsets + source * 32)
        forget_first = local[:, :1] + ((4 * width) % 19) + source * 32
        fields["raw_gate"].append(
            forget_first + (torch.arange(width) % 13).to(torch.bfloat16)
        )
        beta_start = 4 * width + latent + source * heads
        fields["raw_beta"].append(
            local[:, :1]
            + (torch.arange(beta_start, beta_start + heads) % 19).to(torch.bfloat16)
            + source * 32
        )
    return {name: torch.cat(parts, dim=1) for name, parts in fields.items()}


@contextmanager
def managed_graph(stream):
    """Release NCCL graph references before the enclosing group shutdown."""
    graph = torch.cuda.CUDAGraph()
    try:
        yield graph
    finally:
        try:
            stream.synchronize()
        finally:
            # Explicitly destroy the graph even when capture, replay or a
            # correctness assertion raises; Python references may still exist.
            graph.reset()


def worker(rank, ranks, rendezvous):
    # Eight workers otherwise oversubscribe CPU fixture/oracle arithmetic.
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=ranks,
        timeout=timedelta(seconds=120),
        device_id=torch.device("cuda", rank),
    )
    collectives._parallelism_config = SimpleNamespace(
        tp_size=1, dp_size=ranks, world_size=ranks
    )
    collectives._group_map[collectives.Group.KTP] = dist.group.WORLD
    collectives._initialized = True
    try:
        for rows in (1, 2, 4, 8, 16, 32):
            width, latent = (96 // ranks) * 128, 128
            column = (torch.arange(4 * width + latent + 96, device="cuda") % 19).to(
                torch.bfloat16
            )
            gate_column = (torch.arange(width, device="cuda") % 13).to(torch.bfloat16)

            def project(hidden):
                return hidden[:, :1] + column + rank * 32

            def forget(hidden):
                return hidden[:, :1] + gate_column

            local = torch.empty((rows, 7168), device="cuda", dtype=torch.bfloat16)
            # The workspace contract intentionally requires KTP > 1.
            workspaces = [
                (
                    KtpProjectionWorkspace(
                        [rows],
                        ktp_size=ranks,
                        total_heads=96,
                        head_dim=128,
                        device=local.device,
                    )
                    if ranks > 1
                    else None
                )
                for _ in range(2)
            ]
            for padded in (False, True):
                for optimize in (False, True):

                    def run():
                        return project_kda_inputs_ktp(
                            local,
                            project,
                            forget,
                            total_heads=96,
                            head_dim=128,
                            forget_latent_size=latent,
                            ktp_size=ranks,
                            ktp_rank=rank,
                            workspace=workspaces[int(optimize)],
                            optimize=optimize,
                        )

                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        local.fill_(rank)
                        for _ in range(3):
                            run()
                    stream.synchronize()
                    dist.barrier()
                    with managed_graph(stream) as graph:
                        with torch.cuda.graph(graph, stream=stream):
                            captured = run()
                        for replay in range(4):
                            cpu_local = (
                                (
                                    (torch.arange(rows) % 3).to(torch.bfloat16)
                                    + rank * 4
                                    + replay * 0.25
                                )[:, None]
                                .expand(rows, 7168)
                                .clone()
                            )
                            if padded:
                                cpu_local[max(0, rows - 1) :].zero_()
                            expected = expected_fields(cpu_local, ranks)
                            with torch.cuda.stream(stream):
                                local.copy_(cpu_local)
                                graph.replay()
                            stream.synchronize()
                            # The stream is complete; both executions see the changed inputs.
                            for actual in (captured, run()):
                                for name, want in expected.items():
                                    assert_exact(getattr(actual, name).cpu(), want)
                    dist.barrier()
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "ranks": ranks,
                            "local_M": rows,
                            "gathered_M": rows * ranks,
                            "valid_and_padded": True,
                            "variants": 2,
                            "replays_each": 4,
                            "eager_and_graph": "PASS",
                        }
                    ),
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


class ProjectionKtpNcclTest(unittest.TestCase):
    def test_real_transport_and_replay(self):
        ranks = int(os.environ.get("GPU_COUNT", "1"))
        self.assertIn(ranks, (1, 2, 4, 8))
        self.assertGreaterEqual(
            torch.cuda.device_count(), ranks, "manual test requires its declared GPUs"
        )
        with tempfile.TemporaryDirectory(prefix="k3-ktp-nccl-") as directory:
            rendezvous = Path(directory, "store").as_uri()
            mp.spawn(worker, args=(ranks, rendezvous), nprocs=ranks, join=True)


if __name__ == "__main__":
    unittest.main()
