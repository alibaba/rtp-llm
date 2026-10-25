"""Small scale-shard correctness cases; GPU execution is caller-owned.

The guarded torchrun/Bazel driver calls run_distributed_cases() on EVERY rank.
Run separately at world sizes 1, 2 and 3. This module does not initialize CUDA,
distributed groups, or a model on import. CPU tests are metadata/oracle checks,
not claims of hardware support or graph lifetime integration in a full model.
"""

import importlib
import math
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch


def expected_partition(table_rows, peers):
    """Independent row-coverage oracle, also checked against real descriptors."""
    if table_rows <= 0 or peers <= 0:
        raise ValueError("Positive table rows and group size are required")
    capacity = (table_rows + peers - 1) // peers
    return capacity, [
        (min(rank * capacity, table_rows), min((rank + 1) * capacity, table_rows))
        for rank in range(peers)
    ]


def byte_oracle(weight, scales, ids):
    valid = (ids >= 0) & (ids < weight.shape[0])
    safe = ids.clamp(0, weight.shape[0] - 1).reshape(-1)
    values = weight.view(torch.uint8).index_select(0, safe).reshape(-1, 24, 256)
    factors = scales.index_select(0, safe).reshape(-1, 24, 8)
    values[~valid] = 0
    factors[~valid] = 127
    factors = factors.reshape(ids.shape[0], 48, 4).to(torch.int64)
    words = sum(factors[..., i] << (8 * i) for i in range(4)).to(torch.int32)
    storage = torch.full(
        (48, math.ceil(ids.shape[0] / 4) * 4), 0x7F7F7F7F, dtype=torch.int32
    )
    storage[:, : ids.shape[0]] = words.T
    return values.reshape(ids.shape[0], 6144), storage


def _provider():
    return importlib.import_module(
        "rtp_llm.models_py.modules.dsv41._engram_scale_shards"
    )


@contextmanager
def forbid_forward_collectives():
    """Startup exchanges are allowed, lookup/capture/replay cannot call these."""
    names = (
        "all_gather",
        "all_gather_into_tensor",
        "all_gather_object",
        "all_reduce",
        "broadcast",
        "broadcast_object_list",
        "barrier",
        "new_group",
        "all_to_all_single",
    )
    with ExitStack() as stack:
        for name in names:
            if hasattr(torch.distributed, name):
                stack.enter_context(
                    patch.object(
                        torch.distributed,
                        name,
                        side_effect=AssertionError(
                            f"Unexpected forward collective: {name}"
                        ),
                    )
                )
        yield


class ScaleShardCPUTest(unittest.TestCase):
    def test_existing_groups_never_initialize_the_runtime(self):
        from rtp_llm.models_py.distributed import collective_torch

        with patch.object(collective_torch, "_initialized", False), patch.object(
            collective_torch,
            "_get_group",
            side_effect=AssertionError("implicit group initialization"),
        ) as get_group:
            self.assertEqual(_provider()._existing_local_groups(), ())
            get_group.assert_not_called()

    def test_partition_oracle_counts_one_through_eight_and_empty_tails(self):
        for peers in range(1, 9):
            for rows in (1, 2, 3, 7, 8, 9, 257):
                with self.subTest(rows=rows, peers=peers):
                    capacity, bounds = expected_partition(rows, peers)
                    covered = [r for begin, end in bounds for r in range(begin, end)]
                    self.assertEqual(covered, list(range(rows)))
                    self.assertEqual(len(bounds), peers)
                    self.assertLess(peers * capacity - rows, peers)
                    for index in range(rows):
                        owner, offset = divmod(index, capacity)
                        self.assertLess(owner, peers)
                        self.assertEqual(bounds[owner][0] + offset, index)

    def test_oracle_unsigned_scale_bytes_invalid_ids_and_padding(self):
        weight = torch.arange(256, dtype=torch.int64).to(torch.uint8).view(1, 256)
        scales = torch.tensor([[0, 1, 128, 255, 127, 126, 125, 124]], dtype=torch.uint8)
        ids = torch.zeros((1, 24), dtype=torch.int64)
        ids[0, 1:4] = torch.tensor([-1, 1, 2**62])
        values, storage = byte_oracle(weight, scales, ids)
        self.assertEqual(storage[0, 0].item(), -8388352)
        self.assertTrue(torch.all(storage[2:8, 0] == 0x7F7F7F7F))
        self.assertTrue(torch.all(storage[:, 1:] == 0x7F7F7F7F))
        self.assertFalse(values.reshape(1, 24, 256)[0, 1:4].any())

    def test_disabled_selection_does_not_enter_collectives(self):
        with forbid_forward_collectives(), patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=AssertionError("Disabled sharding must not probe CUDA"),
        ):
            selected = _provider().select_scale_group(
                "cuda:0", "cpu-disabled-test", enabled=False
            )
        self.assertIsNone(selected)


def _ids(rows, tokens, generation, peers):
    ids = (
        torch.arange(tokens * 24, dtype=torch.int64).reshape(tokens, 24) * 37
        + generation * 53
    ) % rows
    capacity, bounds = expected_partition(rows, peers)
    # Explicitly cover every nonempty owner even for M=1; the remaining entries
    # include invalid and extreme IDs while preserving those sentinel columns.
    for rank, (begin, end) in enumerate(bounds):
        if begin < end:
            ids[0, rank] = begin if generation % 2 else end - 1
    ids[:, -4:] = torch.tensor([-1, rows, 2**62, -(2**62)])
    return ids


def _assert_output(outputs, weight, scales, cpu_ids):
    values, storage = byte_oracle(weight, scales, cpu_ids)
    if not torch.equal(outputs.values.view(torch.uint8).cpu(), values):
        raise AssertionError("FP8 source bytes differ from independent CPU oracle")
    if not torch.equal(outputs.scale_storage.cpu(), storage):
        raise AssertionError(
            "Packed scales or initialized padding differ from CPU oracle"
        )


@torch.inference_mode()
def _run_group(group, device, label):
    from rtp_llm.models_py.modules.dsv41 import _engram_quantized_triton as direct
    from rtp_llm.models_py.modules.dsv41.engram import _PinnedSharedTable

    dist = torch.distributed
    provider = _provider()
    peers, rank = dist.get_world_size(group), dist.get_rank(group)
    global_ranks = dist.get_process_group_ranks(group)
    receipt = {
        "group": label,
        "global_ranks": global_ranks,
        "group_rank": rank,
        "global_rank": dist.get_rank(),
        "device": str(device),
        "cases": [],
    }
    for rows in dict.fromkeys((257, max(1, peers - 1))):
        row = torch.arange(rows, dtype=torch.int64)[:, None]
        weight = ((row * 17 + torch.arange(256)[None, :] * 29) % 256).to(torch.uint8)
        scales = ((row * 13 + torch.arange(8)[None, :] * 31) % 256).to(torch.uint8)
        identity = f"engram-scale-test-{global_ranks}-N{rows}"
        descriptor = provider.create_sharded_scales(scales, group, device, identity)
        capacity, bounds = expected_partition(rows, peers)
        if peers == 1:
            if not isinstance(descriptor, torch.Tensor):
                raise AssertionError("P1 must retain the local Tensor provider")
            local = descriptor
        else:
            if not isinstance(descriptor, provider.ShardedScaleDescriptor):
                raise AssertionError(
                    "Requested shared small-table test silently fell back"
                )
            if (
                descriptor.group_size,
                descriptor.rank,
                descriptor.table_rows,
                descriptor.rows_per_shard,
            ) != (peers, rank, rows, capacity):
                raise AssertionError("Descriptor partition metadata differs from group")
            if descriptor.validate_metadata(device, rows) is False:
                raise AssertionError("Descriptor metadata rejected")
            if (
                descriptor.peer_ptrs.dtype != torch.uint64
                or descriptor.peer_ptrs.numel() != peers
            ):
                raise AssertionError("Invalid peer-pointer table")
            local = descriptor.local_scales
        if (
            local.shape != (capacity, 8)
            or local.dtype != torch.uint8
            or local.device != device
        ):
            raise AssertionError("Unexpected local scale storage")
        begin, end = bounds[rank]
        if not torch.equal(local[: end - begin].cpu(), scales[begin:end]):
            raise AssertionError("Owner shard holds the wrong source rows")
        if peers > 1:
            again = provider.create_sharded_scales(scales, group, device, identity)
            if again.local_scales.data_ptr() != local.data_ptr():
                raise AssertionError(
                    "Same live source generation did not reuse local allocation"
                )
        # Keep the actual native host layout and register only this tiny table.
        pinned = _PinnedSharedTable(
            weight.view(torch.float8_e4m3fn), scales, identity, device
        )
        weight_uva = pinned.cuda_view(pinned.weight)
        capture_stream = torch.cuda.Stream(device=device)
        aux = torch.cuda.Stream(device=device)
        completed = torch.cuda.Event()
        completed.record(torch.cuda.current_stream(device))
        pinned._lookup_events.append(completed)
        graphs = []
        try:
            for tokens in (1, 37, 384, 385):
                backing = torch.empty(
                    (tokens * 2 + 2, 50), dtype=torch.int64, device=device
                )
                live = backing[1 : 1 + tokens * 2 : 2, 1:49:2]
                live.copy_(_ids(rows, tokens, 0, peers))
                outputs = direct.allocate_outputs(live)
                direct.prepare_lookup(weight_uva, descriptor, live, outputs)
                torch.cuda.synchronize(device)
                for asynchronous in (False, True):
                    graph = torch.cuda.CUDAGraph()
                    with forbid_forward_collectives(), torch.cuda.graph(
                        graph, stream=capture_stream
                    ):
                        if asynchronous:
                            aux.wait_stream(capture_stream)
                            with torch.cuda.stream(aux):
                                direct.launch_epilogue(
                                    weight_uva, descriptor, live, outputs
                                )
                                descriptor.record_stream(aux)
                            capture_stream.wait_stream(aux)
                            outputs.values.record_stream(capture_stream)
                            outputs.scales.record_stream(capture_stream)
                        else:
                            direct.launch_epilogue(
                                weight_uva, descriptor, live, outputs
                            )
                            descriptor.record_stream(capture_stream)
                    graphs.append((graph, live, outputs, tokens, asynchronous))
            # Revisit all graph sizes in reverse order after capturing larger
            # batches, with live IDs and poisoned buffers changed each replay.
            for graph, live, outputs, tokens, asynchronous in reversed(graphs):
                pointers = (outputs.values.data_ptr(), outputs.scales.data_ptr())
                for generation in (1, 2, 3):
                    cpu_ids = _ids(rows, tokens, generation + rank, peers)
                    live.copy_(cpu_ids)
                    outputs.values.view(torch.uint8).fill_(0xA5)
                    outputs.scale_storage.fill_(-1)
                    with forbid_forward_collectives():
                        graph.replay()
                    torch.cuda.synchronize(device)
                    _assert_output(outputs, weight, scales, cpu_ids)
                    if pointers != (
                        outputs.values.data_ptr(),
                        outputs.scales.data_ptr(),
                    ):
                        raise AssertionError("Captured output addresses changed")
                receipt["cases"].append(
                    {
                        "table_rows": rows,
                        "M": tokens,
                        "async": asynchronous,
                        "changed_replays": 3,
                        "status": "PASS",
                        "allocated_scale_bytes_group": peers * capacity * 8,
                        "replicated_scale_bytes_group": peers * rows * 8,
                    }
                )
        finally:
            torch.cuda.synchronize(device)
            completed.record(torch.cuda.current_stream(device))
            completed.synchronize()
            for graph, *_ in graphs:
                graph.reset()
            # Every peer must finish replay before any descriptor/table owner
            # can be dropped. Framework reload/lease logic is a separate test.
            dist.barrier(group=group)
    return receipt


def run_distributed_cases(device=None, group=None):
    """Caller initializes dist and selects/locks devices; call on every rank."""
    dist = torch.distributed
    if not dist.is_initialized():
        raise RuntimeError("Caller must initialize torch.distributed, including P1")
    device = torch.device(
        device if device is not None else f"cuda:{torch.cuda.current_device()}"
    )
    if device.type != "cuda" or device.index is None:
        raise ValueError("Pass an explicit local CUDA device")
    group = dist.group.WORLD if group is None else group
    receipts = [_run_group(group, device, "primary")]
    if group is dist.group.WORLD and dist.get_world_size() == 3:
        # All ranks create the subgroup in the same order; rank1 does not enter
        # its collectives. The final WORLD barrier joins both member branches.
        subgroup = dist.new_group(ranks=[0, 2], backend="nccl")
        if dist.get_rank() in (0, 2):
            receipts.append(_run_group(subgroup, device, "noncontiguous_0_2"))
        dist.barrier()
    return {
        "status": "PASS",
        "scope": "tiny actual distributed FP8 lookup, no WKV/model/performance",
        "groups": receipts,
        "framework_capture_lease_reload": "NOT_RUN; test retains descriptor owners",
        "cpu_layout": "unchanged native rows with real UVA registration",
    }


def _distributed_worker(rank, world_size, store_path):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        "nccl",
        init_method="file://" + store_path,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
        device_id=device,
    )
    try:
        result = run_distributed_cases(device)
        if result["status"] != "PASS":
            raise AssertionError(result)
    finally:
        torch.distributed.destroy_process_group()


class ScaleShardDistributedCUDA(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 3,
        "requires three peer-accessible GPUs",
    )
    def test_one_two_three_ranks_and_noncontiguous_subgroup(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory(prefix="engram-scale-shards-") as directory:
            for world_size in (1, 2, 3):
                with self.subTest(world_size=world_size):
                    mp.spawn(
                        _distributed_worker,
                        args=(world_size, str(Path(directory) / f"store{world_size}")),
                        nprocs=world_size,
                        join=True,
                    )


if __name__ == "__main__":
    unittest.main()
