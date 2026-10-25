"""Real peer kernels against logical KV arrays, including mutable graph replay."""

import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed import collective_torch as collectives
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.fia2a_plan import (
    Fia2aA2ABackend,
    Fia2aFusionMode,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_fia2a_backend import (
    PageRRFIA2AMLABackend,
)


def _merge(values, logs):
    logs = torch.stack(logs)
    maximum = logs.max(0).values
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    weights = torch.exp(logs - maximum)
    denominator = weights.sum(0)
    value = (torch.stack(values).float() * weights[..., None]).sum(0)
    return value / denominator.clamp_min(1e-30)[..., None], torch.logsumexp(logs, 0)


def _split_reference(query, keys, visible, scale):
    """Model K128 online probability rounding independently of page addresses."""
    queries, heads, _ = query.shape
    if keys.shape[0] == 0:
        return query.new_zeros(queries, heads, 512, dtype=torch.float32), torch.full(
            (queries, heads), -torch.inf, device=query.device,
        )
    scores = (query.float().reshape(queries * heads, -1) @ keys.float().T)
    scores = scores.view(queries, heads, -1) * scale
    mask = torch.arange(keys.shape[0], device=query.device)[None, None, :] >= visible[:, None, None]
    scores = scores.masked_fill(mask, -torch.inf)
    tiles = torch.nn.functional.pad(scores, (0, -keys.shape[0] % 128), value=-torch.inf)
    tiles = tiles.view(queries, heads, -1, 128)
    maxima = tiles.amax(-1).cummax(-1).values
    maxima = torch.where(torch.isfinite(maxima), maxima, 0.0)
    last = maxima[..., -1]
    rounded = torch.exp(tiles - maxima[..., None]).to(query.dtype).float()
    weights = (rounded * torch.exp(maxima - last[..., None])[..., None]).flatten(-2)
    denominator = torch.exp(scores - last[..., None]).sum(-1)
    value = weights[..., : keys.shape[0]] @ keys[:, :512].float()
    return value / denominator.clamp_min(1e-30)[..., None], torch.log(denominator) + last


def _reference(query, keys, cutoffs, backend, flattened):
    batch, queries, heads, _ = query.shape
    positions = torch.arange(keys.shape[1], device=query.device)
    owners = positions // 128 % backend.communicator.size
    results = []
    for row in range(batch):
        for query_begin in (range(queries) if flattened else (0,)):
            query_end = query_begin + 1 if flattened else queries
            q = query[row, query_begin:query_end]
            limits = cutoffs[row, query_begin:query_end]
            peer_values, peer_logs = [], []
            for owner in range(backend.communicator.size):
                logical = keys[row][(owners == owner) & (positions < limits[-1])]
                visible = ((positions[None, :] < limits[:, None]) & (owners == owner)).sum(-1)
                width = math.ceil(math.ceil(logical.shape[0] / 128) / backend.splits) * 128
                values, logs = [], []
                for split in range(backend.splits):
                    begin = min(split * width, logical.shape[0])
                    end = min(begin + width, logical.shape[0])
                    value, log = _split_reference(q, logical[begin:end], (visible - begin).clamp(0, end - begin), backend.softmax_scale)
                    values.append(value)
                    logs.append(log)
                if backend.fused and backend.splits > 1:
                    peer_values.extend(values)
                    peer_logs.extend(logs)
                else:
                    value, log = _merge(values, logs)
                    peer_values.append(value.bfloat16())
                    peer_logs.append(log)
            results.append(_merge(peer_values, peer_logs)[0])
    result = torch.cat(results).bfloat16().view(batch * queries, heads, 512)
    rank = backend.communicator.rank
    local_heads = backend.communicator.local_heads
    return result[:, rank * local_heads:(rank + 1) * local_heads].transpose(0, 1).contiguous()


def _worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    try:
        device = torch.device("cuda", rank)
        configs = SimpleNamespace(kv_lora_rank=512, rope_head_dim=64, kernel_tokens_per_block=128)
        retained = []
        with patch.object(collectives, "get_process_group", return_value=dist.group.WORLD), patch.object(
            collectives, "_get_group", return_value=dist.group.WORLD
        ):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                communicator = collectives.MlaDcpCommunicator(48, 576, 512, dtype, device)
                if not communicator.peer_memory_supported:
                    raise RuntimeError("FIA2A protocol regression requires direct peer-memory access")
                for batch, flattened, mode, transport in (
                    (1, False, Fia2aFusionMode.UNFUSED, Fia2aA2ABackend.NCCL),
                    (1, False, Fia2aFusionMode.UNFUSED, Fia2aA2ABackend.CUSTOM),
                    (1, False, Fia2aFusionMode.FUSED, Fia2aA2ABackend.AUTO),
                    (32, False, Fia2aFusionMode.AUTO, Fia2aA2ABackend.AUTO),
                    (8, True, Fia2aFusionMode.AUTO, Fia2aA2ABackend.AUTO),
                ):
                    torch.manual_seed(193)
                    query = (torch.randn(batch, 4, 96, 576, device=device) * 0.3).to(dtype)
                    keys = torch.randn(batch, 512, 576, device=device).to(dtype)
                    owned = torch.arange(512, device=device) // 128 % 2 == rank
                    cache = keys[:, owned].reshape(-1, 128, 576).clone()
                    prefixes = torch.tensor([0, 127, 255, 508] * ((batch + 3) // 4), device=device)[:batch]
                    cutoffs = prefixes[:, None] + torch.arange(1, 5, device=device)
                    lengths = ((torch.arange(512, device=device)[None, None, :] < cutoffs[..., None]) & owned).sum(-1).int()
                    table = torch.arange(batch * 2, device=device, dtype=torch.int32).view(batch, 2)
                    metadata = SimpleNamespace(
                        block_tables=table, query_block_tables=table.repeat_interleave(4, 0),
                        local_causal_lens=lengths.reshape(-1, 1).clone() if flattened else lengths.clone(),
                    )
                    backend = PageRRFIA2AMLABackend(configs, communicator, dtype, mode, transport, token_major=True)
                    backend.prepare(metadata, 1 / math.sqrt(576), 1.0)
                    marker = torch.ones(1, device=device)
                    for _ in range(2):
                        backend.forward(query, cache)
                        dist.all_reduce(marker)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = backend.forward(query, cache)
                        dist.all_reduce(marker)
                    for shift in (0, 1):
                        query.copy_((query.float() * 0.9).to(dtype))
                        cutoffs = prefixes[:, None] + torch.arange(1, 5, device=device) - shift
                        lengths = ((torch.arange(512, device=device)[None, None, :] < cutoffs[..., None]) & owned).sum(-1).int()
                        metadata.local_causal_lens.copy_(lengths.reshape(-1, 1) if flattened else lengths)
                        graph.replay()
                        torch.cuda.synchronize()
                        expected = _reference(query, keys, cutoffs, backend, flattened)
                        torch.testing.assert_close(output, expected, atol=0.002 if dtype == torch.float8_e4m3fn else 0.001, rtol=0.015)
                    retained.append((backend, graph, output, expected, query, cache, metadata, marker))
                # Eager arena growth must not invalidate earlier captured pointers.
                for backend, graph, output, expected, *_ in retained:
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(output, expected, atol=0.002, rtol=0.015)
    finally:
        dist.destroy_process_group()


class Fia2aBackendTest(unittest.TestCase):
    def test_distributed_graphs(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("requires two SM103 GPUs")
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=("file://" + os.path.join(directory, "rendezvous"),), nprocs=2)


if __name__ == "__main__":
    unittest.main()
