"""Real two-rank IPC, vocabulary and primitive Graph tests for all N=1..64.

These are primitive Graphs, not additional model capture buckets.
"""

import gc
import os
import socket
import unittest
import weakref
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _embedding(owners, reducer, compute=None):
    from rtp_llm.models_py.modules.qwen4_exp.decode_communication import (
        DecodeVocabEmbedding,
    )
    from rtp_llm.models_py.modules.qwen4_exp.decode_vocab_weights import (
        prepare_decode_vocab,
    )

    prepared = prepare_decode_vocab(
        source_weights=owners,
        compute_weights=owners if compute is None else compute,
        group=reducer.workspace.group,
    )
    return DecodeVocabEmbedding(prepared, reducer)


def _worker(rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=180),
    )
    from rtp_llm.models_py.distributed import collective_torch
    from rtp_llm.models_py.modules.qwen4_exp.decode_communication import (
        Qwen4DecodeReduce,
    )

    collective_torch._group_map[collective_torch.Group.TP] = dist.group.WORLD
    collective_torch._group_map[collective_torch.Group.DP_AND_TP] = dist.group.WORLD
    collective_torch._initialized = True
    collective_torch._parallelism_config = SimpleNamespace(
        tp_size=2, dp_size=1, world_size=2
    )
    group = collective_torch.Group.TP
    device = torch.device("cuda", rank)
    reducer = Qwen4DecodeReduce(SimpleNamespace(group=dist.group.WORLD, device=device))
    graphs = []
    try:
        # Converted owners must release hidden shards, even while the model
        # and its shared draft embedding remain alive. Reserved allocator
        # memory is intentionally not used as evidence of tensor lifetime.
        from rtp_llm.models_py.modules.qwen4_exp.weight_views import PreparedWeightDict
        from rtp_llm.utils.model_weight import W

        owners = {
            W.embedding: torch.ones(
                4096 * 1280, device=device, dtype=torch.bfloat16
            ).view(4096, 1280)
        }
        compute = PreparedWeightDict(owners)
        base_ref = weakref.ref(owners[W.embedding]._base)
        source_ref = weakref.ref(owners[W.embedding])
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        retired = _embedding(owners, reducer, compute)
        gc.collect()
        torch.cuda.synchronize()
        assert source_ref() is None, "old hidden-sharded embedding must be released"
        assert base_ref() is None, "view-backed hidden storage must also be released"
        assert owners[W.embedding] is compute[W.embedding] is retired.weight
        assert (
            torch.cuda.memory_allocated() - allocated_before < 1024 * 1024
        ), "repartition must not retain a second 10 MiB vocabulary shard"
        converted_ptr = retired.weight.data_ptr()
        if rank == 1:
            del retired
            gc.collect()
        shared_owner = _embedding(owners, reducer)
        assert shared_owner.weight.data_ptr() == converted_ptr
        torch.testing.assert_close(
            shared_owner(torch.tensor([0, 2048, 4095], device=device)),
            torch.ones(3, 2560, device=device, dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )
        inconsistent = owners[W.embedding] if rank == 0 else owners[W.embedding].clone()
        try:
            _embedding({W.embedding: inconsistent}, reducer)
        except ValueError as exc:
            assert "matching" in str(exc)
        else:
            raise AssertionError(
                "mixed hidden/vocabulary layouts must fail on every rank"
            )
        del inconsistent, shared_owner, owners, compute
        if rank == 0:
            del retired
        gc.collect()
        for rows in range(1, 65):
            value = torch.full(
                (rows, 2560), rank + 1, device=device, dtype=torch.bfloat16
            )
            original = value.clone()
            eager = reducer(value, group)
            torch.testing.assert_close(eager, torch.full_like(value, 3), rtol=0, atol=0)
            torch.testing.assert_close(value, original, rtol=0, atol=0)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = reducer(value, group)
            reducer.finish_graph_capture()
            graphs.append((graph, value, out))
        # Different sizes repeatedly reuse one workspace; inputs stay live and
        # are changed between replays to detect stale registration/staging.
        for sequence in (range(64), reversed(range(64)), range(0, 64, 3)):
            for index in sequence:
                graph, value, out = graphs[index]
                value.fill_(rank + index % 7)
                graph.replay()
                torch.testing.assert_close(
                    out, torch.full_like(out, 2 * (index % 7) + 1), rtol=0, atol=0
                )
        # Both oversized and wrong-dtype calls preserve the NCCL fallback.
        for rows, dtype in ((65, torch.bfloat16), (8, torch.float32)):
            value = torch.full((rows, 2560), rank + 1, device=device, dtype=dtype)
            torch.testing.assert_close(
                reducer(value, group), torch.full_like(value, 3), rtol=0, atol=0
            )

        for fp8 in (False, True):
            vocab, half_width = 129, 1280
            full = (
                (
                    torch.arange(vocab * half_width * 2, device=device).reshape(
                        vocab, half_width * 2
                    )
                    % 15
                )
                - 7
            ).to(torch.bfloat16)
            weight = full[:, rank * half_width : (rank + 1) * half_width].contiguous()
            scale = None
            if fp8:
                weight = weight.to(torch.float8_e4m3fn)
                scale = torch.tensor([0.7 if rank == 0 else 0.3], device=device)
                expected_weight = torch.cat(
                    [
                        (
                            full[:, :half_width]
                            * torch.tensor(0.7, device=device, dtype=torch.bfloat16)
                        ).to(torch.bfloat16),
                        (
                            full[:, half_width:]
                            * torch.tensor(0.3, device=device, dtype=torch.bfloat16)
                        ).to(torch.bfloat16),
                    ],
                    dim=1,
                )
            else:
                expected_weight = full
            from rtp_llm.utils.fp8_constants import FP8_STORAGE_SCALE_SUFFIX

            owners = {W.embedding: weight}
            if scale is not None:
                owners[W.embedding + FP8_STORAGE_SCALE_SUFFIX] = scale
            embedding = _embedding(owners, reducer)
            shared = _embedding(owners, reducer)
            assert (
                shared.shard is embedding.shard
            ), "target/draft must share vocabulary storage"
            assert weight._rtp_requires_restart_for_update
            for rows in range(1, 65):
                ids = (
                    torch.arange(rows, device=device, dtype=torch.int64)
                    * 17
                    % (vocab + 3)
                    - 1
                )
                mask = torch.arange(rows, device=device) % 5 != 0
                wanted = expected_weight[ids.clamp(0, vocab - 1)].clone()
                wanted[(ids < 0) | (ids >= vocab) | ~mask] = 0
                actual = embedding(ids, text_tokens_mask=mask)
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = embedding(ids, text_tokens_mask=mask)
                reducer.finish_graph_capture()
                for shift in (0, 1, 63, 64, 128):
                    ids.fill_(shift)
                    graph.replay()
                    wanted = expected_weight[ids].clone()
                    wanted[~mask] = 0
                    torch.testing.assert_close(output, wanted, rtol=0, atol=0)
                graphs.append((graph, ids, mask, output, embedding))
            empty = embedding(torch.empty(0, device=device, dtype=torch.int64))
            assert empty.shape == (0, 2560)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(
                "PASS: N=1..64 BF16 reduction, BF16/FP8 embedding, repeated Graph replay, boundary/fallback/shared-weight cases",
                flush=True,
            )
    finally:
        # Graphs keep registered pointers alive until both ranks finish.
        torch.cuda.synchronize()
        dist.barrier()
        graphs.clear()
        reducer.workspace.destroy()
        dist.destroy_process_group()


@unittest.skipUnless(
    torch.cuda.device_count() >= 2 and torch.cuda.get_device_capability(0) == (12, 0),
    "requires two SM120 GPUs",
)
class DecodeCommunicationSm120Test(unittest.TestCase):
    def test_two_rank_decode_communication(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(_worker, args=(port,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
