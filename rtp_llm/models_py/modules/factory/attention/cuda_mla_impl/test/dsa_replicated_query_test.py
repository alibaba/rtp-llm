"""Distributed BF16 Q-projection/absorption/Gather A/B, excluding Indexer/SFA."""

import hashlib
import json
import math
import os
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_page_rr_impl import (
    PageRRSparseMlaOp,
    SparseMlaPageRRImpl,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.ops import NcclCommConfig, ParallelismConfig
from rtp_llm.test.utils.port_util import PortsContext
from rtp_llm.utils.model_weight import W


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _measure(graph, samples=15, repeat=10):
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    for _ in range(samples):
        dist.barrier()
        start.record()
        for _ in range(repeat):
            graph.replay()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end) / repeat)
    per_rank = [None] * dist.get_world_size()
    dist.all_gather_object(per_rank, elapsed)
    maxima = [max(ranks) for ranks in zip(*per_rank)]
    return {
        "per_rank_ms": per_rank,
        "max_rank_ms": maxima,
        "p50_ms": _percentile(maxima, 0.5),
        "p90_ms": _percentile(maxima, 0.9),
    }


def _worker(rank, world, port, output):
    torch.cuda.set_device(rank)
    parallel = ParallelismConfig()
    parallel.world_rank, parallel.local_rank = rank, rank
    parallel.world_size, parallel.tp_size, parallel.dp_size = world, world, 1
    init_distributed_environment(
        parallel,
        NcclCommConfig(
            nccl_ip="127.0.0.1",
            tp_nccl_port=port + 1,
            dp_tp_nccl_port=port + 2,
            ffn_tp_nccl_port=port + 3,
        ),
        port,
        disable_custom_all_reduce=True,
    )
    try:
        torch.manual_seed(1731)
        device, dtype = torch.device("cuda", rank), torch.bfloat16
        heads, q_rank, nope, latent, suffix = 96, 1536, 128, 512, 64
        local = heads // world
        full_weight = (
            torch.randn(q_rank, heads * (nope + suffix), device=device)
            / math.sqrt(q_rank)
        ).to(dtype)
        full_kc = (
            torch.randn(heads, nope, latent, device=device) / math.sqrt(nope)
        ).to(dtype)
        selected = slice(rank * local, (rank + 1) * local)
        projections = {
            "sharded": CudaF16Linear(
                full_weight.view(q_rank, heads, nope + suffix)[:, selected]
                .reshape(q_rank, -1)
                .contiguous()
            ),
            "replicated": CudaF16Linear(full_weight),
        }
        adapters = {}
        for name in projections:
            adapter = object.__new__(SparseMlaPageRRImpl)
            adapter.num_heads = local
            adapter.nope_head_dim, adapter.rope_head_dim = nope, suffix
            adapter.kv_lora_rank = latent
            adapter.page_rr_params = SimpleNamespace(is_prefill=False, query_cp=False)
            adapter.workspace_owner = SimpleNamespace(heads=heads)
            adapter.weights = [
                {
                    W.mla_kc: (
                        full_kc
                        if name == "replicated"
                        else full_kc[selected].contiguous()
                    )
                }
            ]
            adapters[name] = adapter
        op = PageRRSparseMlaOp(
            local,
            latent,
            suffix,
            nope,
            128,
            1.0,
            2048,
            parallelism_config=parallel,
        )
        results = []
        for rows in (1, 64, 128, 256, 512, 640):
            inputs = torch.randn(rows, q_rank, device=device, dtype=dtype)
            if rows > 1:
                inputs[-1].zero_()

            def forward(name):
                count = heads if name == "replicated" else local
                query = projections[name](inputs).view(rows, count, nope + suffix)
                query = op._decode(adapters[name]._apply_input_bmm(query, 0))
                return F.pad(query, (0, 0, 0, op.kernel_heads - heads)).contiguous()

            baseline = forward("sharded").clone()
            candidate = forward("replicated").clone()
            torch.testing.assert_close(candidate, baseline, atol=0.03125, rtol=0.02)
            reference_q = (
                (inputs.float() @ full_weight.float())
                .to(dtype)
                .view(rows, heads, nope + suffix)
            )
            reference = candidate.new_zeros(rows, op.kernel_heads, latent + suffix)
            reference[:, :heads, latent:] = reference_q[..., nope:]
            reference[:, :heads, :latent] = (
                torch.bmm(
                    reference_q[..., :nope].float().transpose(0, 1), full_kc.float()
                )
                .transpose(0, 1)
                .to(dtype)
            )
            errors = {}
            for name, value in (("sharded", baseline), ("replicated", candidate)):
                assert value.shape == (rows, op.kernel_heads, latent + suffix)
                assert value.is_contiguous()
                assert torch.isfinite(value).all().item()
                torch.testing.assert_close(value, reference, atol=0.03125, rtol=0.02)
                errors[name] = (value.float() - reference.float()).abs().max().item()
            combined = []
            lengths = torch.ones(rows, dtype=torch.int32, device=device)
            lse = torch.zeros(rows, heads, dtype=torch.float32, device=device)
            for value in (baseline, candidate):
                combined.append(
                    op.communicator.combine(
                        (value[:, :heads, :128] * (rank + 1)).contiguous(),
                        lse,
                        lengths,
                        output_dim=128,
                    )
                )
            assert combined[0].shape == (local, rows, 128)
            torch.testing.assert_close(combined[1], combined[0], atol=0.125, rtol=0.02)
            dispatch = {}
            graphs, outputs = {}, {}
            for name in ("sharded", "replicated"):
                with patch.object(
                    op.communicator, "query_gather", wraps=op.communicator.query_gather
                ) as gather:
                    forward(name)
                    dispatch[name] = gather.call_count
                assert dispatch[name] == int(name == "sharded")
                for _ in range(3):
                    forward(name)
                torch.cuda.synchronize()
                dist.barrier()
                graphs[name] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[name]):
                    outputs[name] = forward(name)
            # Graph replay must consume fresh inputs, including changing null rows.
            inputs.copy_(torch.randn_like(inputs))
            if rows > 1:
                inputs[0].zero_()
            expected = forward("sharded").clone()
            for name in graphs:
                graphs[name].replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    outputs[name], expected, atol=0.03125, rtol=0.02
                )
            rounds = []
            for order in (("sharded", "replicated"), ("replicated", "sharded")):
                measured = {name: _measure(graphs[name]) for name in order}
                rounds.append(
                    {
                        "order": order,
                        "paths": measured,
                        "speedup": measured["sharded"]["p50_ms"]
                        / measured["replicated"]["p50_ms"],
                    }
                )
            results.append(
                {
                    "rows": rows,
                    "max_abs_vs_fp32": errors,
                    "query_gather_calls": dispatch,
                    "rounds": rounds,
                }
            )
            if rank == 0:
                print(
                    json.dumps(
                        {"rows": rows, "correctness": "passed", "rounds": rounds}
                    ),
                    flush=True,
                )
            del graphs, outputs, reference, reference_q, baseline, candidate
        manifest = {
            "source_revision": os.environ.get("SOURCE_REVISION", "unknown"),
            "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rank": rank,
            "tp_size": world,
            "gpu": torch.cuda.get_device_name(),
            "host": socket.gethostname(),
            "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
            "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "dtype": "bf16",
            "heads": heads,
            "q_rank": q_rank,
            "qk_nope_dim": nope,
            "latent_dim": latent,
            "suffix_dim": suffix,
            "timing": "CUDA-event Graph replay, max(rank_elapsed) per sample; milliseconds",
            "boundary": "Q-B projection -> latent absorption -> Gather/bypass -> contiguous padded SFA Q",
            "excluded": "SP input AllGather, Indexer, sparse attention, DCP output combine, allocation/JIT/capture",
            "synthetic": True,
            "results": results,
        }
        Path(output, f"rank{rank}.json").write_text(json.dumps(manifest, indent=2))
        if rank == 0:
            print(json.dumps(manifest), flush=True)
        dist.barrier()
    finally:
        destroy_distributed_environment()


class DsaReplicatedQueryTest(unittest.TestCase):
    def test_distributed_query_path(self):
        world = int(os.environ.get("GPU_COUNT", "2"))
        self.assertGreaterEqual(torch.cuda.device_count(), world)
        with tempfile.TemporaryDirectory() as temporary, PortsContext(
            num_ports=12
        ) as ports:
            output = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", temporary)
            mp.spawn(_worker, args=(world, ports[0], output), nprocs=world, join=True)


if __name__ == "__main__":
    unittest.main()
