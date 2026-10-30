import datetime
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, store):
    import flashinfer.comm

    from rtp_llm.models_py.distributed.flashinfer_all_reduce import FlashInferAllReduce

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
    # A missing optional API must also participate in the same consensus.
    original = flashinfer.comm.allreduce_fusion
    if rank == 1:
        del flashinfer.comm.allreduce_fusion
    try:
        communicator = FlashInferAllReduce(
            group, device, single_node=True, disable_custom_all_reduce=False
        )
        assert communicator.disabled and not communicator.cpp_small_enabled
    finally:
        flashinfer.comm.allreduce_fusion = original
    # Neither a heterogeneous capability nor rank-local tuning may select
    # different collectives. Both ranks must disable IPC before allocation.
    with patch.object(
        torch.cuda,
        "get_device_capability",
        return_value=(12, 0) if rank == 0 else (8, 9),
    ):
        communicator = FlashInferAllReduce(
            group, device, single_node=True, disable_custom_all_reduce=False
        )
        assert communicator.disabled and not communicator.cpp_small_enabled
    with patch.dict(os.environ, {"QWEN4_TP_LARGE_PREFILL": str(rank)}):
        communicator = FlashInferAllReduce(
            group, device, single_node=True, disable_custom_all_reduce=False
        )
        assert communicator.disabled and not communicator.cpp_small_enabled
    with patch.object(
        torch.cuda,
        "get_device_capability",
        side_effect=(
            RuntimeError("injected local capability failure") if rank == 1 else None
        ),
        return_value=(12, 0),
    ):
        communicator = FlashInferAllReduce(
            group, device, single_node=True, disable_custom_all_reduce=False
        )
        assert communicator.disabled
    original_empty = torch.empty

    def allocate(*args, **kwargs):
        if rank == 1 and args and args[0] == 8 * 1024 * 1024:
            raise torch.OutOfMemoryError("injected local IPC allocation failure")
        return original_empty(*args, **kwargs)

    communicator = FlashInferAllReduce(
        group, device, single_node=True, disable_custom_all_reduce=False
    )
    x = torch.full((4, 2560), rank + 1, dtype=torch.bfloat16, device=device)
    with patch.object(torch, "empty", side_effect=allocate):
        assert not communicator.should_use(x)
    assert communicator.disabled
    dist.all_reduce(x)
    torch.testing.assert_close(x, torch.full_like(x, 3), rtol=0, atol=0)
    from flashinfer.comm import vllm_ar

    communicator = FlashInferAllReduce(
        group, device, single_node=True, disable_custom_all_reduce=False
    )
    original_register = vllm_ar.register_buffer

    def register(*args):
        if rank == 1:
            raise RuntimeError("injected local IPC registration failure")
        return original_register(*args)

    with patch.object(vllm_ar, "register_buffer", side_effect=register):
        assert not communicator.should_use(x)
    assert communicator.disabled
    x.fill_(rank + 1)
    dist.all_reduce(x)
    torch.testing.assert_close(x, torch.full_like(x, 3), rtol=0, atol=0)

    with patch.dict(os.environ, {"QWEN4_TP_LARGE_PREFILL": "0"}):
        communicator = FlashInferAllReduce(
            group, device, single_node=True, disable_custom_all_reduce=False
        )
    assert not communicator.disabled and communicator.cpp_small_enabled
    x = torch.full((4, 2560), rank + 1, dtype=torch.bfloat16, device=device)
    assert communicator.should_use(x)
    actual = communicator.all_reduce(x)
    torch.testing.assert_close(actual, torch.full_like(actual, 3), rtol=0, atol=0)
    communicator.allow_ipc_capture = True
    captures = []
    for rows in (4, 7):
        value = torch.full((rows, 2560), rank + 1, dtype=torch.bfloat16, device=device)
        communicator.all_reduce(value)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = communicator.all_reduce(value)
        communicator.finish_graph_capture()
        captures.append((value, output, graph))
    for value, output, graph in captures:
        value.fill_(rank + 2)
        graph.replay()
        torch.testing.assert_close(output, torch.full_like(output, 5), rtol=0, atol=0)
    x.fill_(rank + 3)
    communicator.all_reduce_into(x, x)
    torch.testing.assert_close(x, torch.full_like(x, 7), rtol=0, atol=0)
    torch.cuda.synchronize()
    communicator.destroy()
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
