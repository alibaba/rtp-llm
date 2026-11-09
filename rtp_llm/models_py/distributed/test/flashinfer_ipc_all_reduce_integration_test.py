import datetime
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, store):
    import flashinfer.comm

    from rtp_llm.config.collective_config import CollectiveConfig
    from rtp_llm.config.server_config_setup import set_parallelism_config
    from rtp_llm.models_py.distributed import all_reduce as all_reduce_module
    from rtp_llm.ops import ParallelismConfig

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + store,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=90),
    )
    group = dist.group.WORLD
    pc = ParallelismConfig()
    pc.tp_size = pc.ep_size = pc.world_size = pc.local_world_size = 2
    pc.world_rank = rank
    set_parallelism_config(pc)

    def initialize(large=False, backend="sm120_ipc"):
        all_reduce_module.destroy_all_reduce()
        return all_reduce_module.init_all_reduce(
            group,
            device,
            pc,
            CollectiveConfig(backend, large),
        )

    def prepare(max_tokens=32768):
        all_reduce_module.prepare_collective_workspaces(
            [SimpleNamespace(hidden_size=2560, compute_dtype=torch.bfloat16)],
            SimpleNamespace(
                runtime_config=SimpleNamespace(
                    fifo_scheduler_config=SimpleNamespace(
                        max_batch_tokens_size=max_tokens
                    ),
                    reserve_runtime_mem_mb=1024,
                ),
                kv_cache_config=SimpleNamespace(kv_cache_mem_mb=0),
            ),
        )

    from flashinfer.comm import vllm_ar

    original = vllm_ar.init_custom_ar
    if rank == 1:
        del vllm_ar.init_custom_ar
    try:
        assert initialize() is None
    finally:
        vllm_ar.init_custom_ar = original
    with patch.object(
        torch.cuda,
        "get_device_capability",
        return_value=(12, 0) if rank == 0 else (8, 9),
    ):
        assert initialize() is None
    try:
        initialize(large=bool(rank))
    except ValueError as error:
        assert "differs across TP ranks" in str(error)
    else:
        raise AssertionError("rank-local tuning did not fail consistently")
    with patch.object(
        torch.cuda,
        "get_device_capability",
        side_effect=RuntimeError("injected probe failure") if rank == 1 else None,
        return_value=(12, 0),
    ):
        assert initialize() is None
    original_empty = torch.empty

    def allocate(*args, **kwargs):
        if rank == 1 and args and args[0] == 8 * 1024 * 1024:
            raise torch.OutOfMemoryError("injected local IPC allocation failure")
        return original_empty(*args, **kwargs)

    with patch.object(torch, "empty", side_effect=allocate):
        assert initialize() is None
    x = torch.full((4, 2560), rank + 1, dtype=torch.bfloat16, device=device)
    dist.all_reduce(x)
    torch.testing.assert_close(x, torch.full_like(x, 3), rtol=0, atol=0)
    from flashinfer.comm import vllm_ar

    original_register = vllm_ar.register_buffer

    def register(*args):
        if rank == 1:
            raise RuntimeError("injected local IPC registration failure")
        return original_register(*args)

    with patch.object(vllm_ar, "register_buffer", side_effect=register):
        assert initialize() is None
    x.fill_(rank + 1)
    dist.all_reduce(x)
    torch.testing.assert_close(x, torch.full_like(x, 3), rtol=0, atol=0)

    communicator = initialize()
    assert communicator is not None and communicator.cpp_small_enabled
    x = torch.full((4, 2560), rank + 1, dtype=torch.bfloat16, device=device)
    assert communicator.should_use(x)
    actual = communicator.all_reduce(x)
    torch.testing.assert_close(actual, torch.full_like(actual, 3), rtol=0, atol=0)
    large_prefill = initialize(large=True)
    assert large_prefill is not None
    assert not large_prefill.large_prefill
    prepare()
    for rows in (4093, 4094, 4095, 4096, 8192, 12288, 16384):
        value = torch.full((rows, 2560), rank + 1, dtype=torch.bfloat16, device=device)
        assert large_prefill.should_use(value)
        torch.testing.assert_close(
            large_prefill.all_reduce(value), torch.full_like(value, 3), rtol=0, atol=0
        )
    communicator = initialize(large=None, backend="auto")
    assert communicator is not None
    small_workspace = communicator._workspace
    value = torch.full((4096, 2560), rank + 1, dtype=torch.bfloat16, device=device)
    with patch.object(
        torch.cuda,
        "mem_get_info",
        side_effect=AssertionError("request-time allocation"),
    ):
        assert not communicator.should_use(value)
    prepare(8192)
    assert communicator.should_use(value)
    assert communicator._prefill_workspace.max_bytes == 8192 * 2560 * 2
    too_large = torch.empty((12288, 2560), dtype=torch.bfloat16, device=device)
    assert not communicator.should_use(too_large)
    assert communicator._workspace is small_workspace
    assert communicator._prefill_workspace is not small_workspace
    torch.testing.assert_close(
        communicator.all_reduce(value), torch.full_like(value, 3), rtol=0, atol=0
    )
    captures = []
    for rows in (4, 7):
        value = torch.full((rows, 2560), rank + 1, dtype=torch.bfloat16, device=device)
        communicator.all_reduce(value)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        all_reduce_module.begin_graph_capture()
        with torch.cuda.graph(graph):
            assert communicator.should_use(value)
            output = communicator.all_reduce(value)
        all_reduce_module.end_graph_capture()
        assert not communicator.allow_ipc_capture
        captures.append((value, output, graph))
    for value, output, graph in captures:
        value.fill_(rank + 2)
        graph.replay()
        torch.testing.assert_close(output, torch.full_like(output, 5), rtol=0, atol=0)
    x.fill_(rank + 3)
    communicator.all_reduce_into(x, x)
    torch.testing.assert_close(x, torch.full_like(x, 7), rtol=0, atol=0)
    # Only one peer has an unaligned contiguous view. It must keep the same
    # collective backend and stage locally instead of falling back alone.
    storage = torch.full(
        (4 * 2560 + rank,), rank + 1, dtype=torch.bfloat16, device=device
    )
    value = storage[rank:].view(4, 2560)
    assert communicator.should_use(value)
    original = value.clone()
    all_reduce_module.begin_graph_capture()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = communicator.all_reduce(value)
    all_reduce_module.end_graph_capture()
    graph.replay()
    torch.testing.assert_close(result, torch.full_like(result, 3), rtol=0, atol=0)
    torch.testing.assert_close(value, original, rtol=0, atol=0)
    communicator.all_reduce_into(value, value)
    torch.testing.assert_close(value, torch.full_like(value, 3), rtol=0, atol=0)
    torch.cuda.synchronize()
    all_reduce_module.destroy_all_reduce()
    dist.destroy_process_group()


class FlashInferIpcAllReduceIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.device_count() == 2 and torch.version.hip is None,
        "requires two CUDA GPUs",
    )
    def test_policy_agreement_graph_registration_and_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _worker, args=(str(Path(directory) / "store"),), nprocs=2, join=True
            )


if __name__ == "__main__":
    unittest.main()
