"""CP2/4/8 GPU regression tests for the DSV4.1 CP prefill symmetric workspace."""

import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

MIB = 1024 * 1024


def _exchange_splits(world_size):
    # Rank 1 sends no data, one destination receives no data for CP4/8,
    # and rank 0's first segment spans the six-row tile in both layouts.
    splits = [
        [(source + 2 * dest) % 5 for dest in range(world_size)]
        for source in range(world_size)
    ]
    splits[0][0] = 7
    splits[1] = [0] * world_size
    if world_size > 2:
        for row in splits:
            row[2] = 0
    return tuple(map(tuple, splits))


def _assert_constant(tensor, value):
    # Bound assertion scratch when exercising a multi-tile gather.
    for part in tensor.flatten().split(MIB):
        assert (part == value).all().item(), f"expected {value} in {tensor.shape}"


def _gather_inputs(
    device, rank, world_size, size, dtype, alias, base=0, output_offset=0
):
    count = size // dtype.itemsize
    output = torch.empty(
        (world_size * count + output_offset,), dtype=dtype, device=device
    )[output_offset:]
    if alias:
        shard = output.narrow(0, rank * count, count)
    else:
        # Offset differs by rank, while the collective shape agrees everywhere.
        shard = torch.empty((count + rank,), dtype=dtype, device=device)[rank:]
    midpoint = count // 2
    shard[:midpoint].fill_(base + 3 * rank + 1)
    shard[midpoint:].fill_(base + 3 * rank + 2)
    return shard, output


def _check_gather(output, world_size, base=0):
    for rank, shard in enumerate(output.chunk(world_size)):
        midpoint = shard.numel() // 2
        _assert_constant(shard[:midpoint], base + 3 * rank + 1)
        _assert_constant(shard[midpoint:], base + 3 * rank + 2)


def _exchange_inputs(device, rank, splits, dtype, width=12):
    sent = torch.empty((sum(splits[rank]), width), dtype=dtype, device=device)
    expected_splits = [row[rank] for row in splits]
    received = torch.empty((sum(expected_splits), width), dtype=dtype, device=device)
    for peer, rows in enumerate(sent.split(splits[rank])):
        rows.fill_(16 * rank + peer + 1)
    # The backend now gathers directly into its symmetric staging window.
    rows = torch.arange(sent.shape[0] - 1, -1, -1, device=device)
    return sent.flip(0), rows, received, expected_splits


def _check_exchange(received, sizes, rank):
    for peer, chunk in enumerate(received.split(sizes)):
        _assert_constant(chunk, 16 * peer + rank + 1)


def _check_borrowed_windows(cp, group, workspace, device, rank, world_size):
    # 64 total bytes pass the old CP4 test but leave only eight bytes per
    # local shard on CP8. The multicast kernel derives width from the output.
    with cp.projected_gather_buffer(
        workspace.buffer, group, numel=64, dtype=torch.uint8
    ) as projected:
        assert (projected is None) == (world_size == 8)
    count = 12_288
    reference = torch.empty((0,), device=device)
    projected_stream = torch.cuda.Stream()
    main_stream = torch.cuda.current_stream()
    projected_stream.wait_stream(main_stream)
    with (
        torch.cuda.stream(projected_stream),
        cp.projected_gather_buffer(
            reference, group, numel=world_size * count, dtype=torch.float32
        ) as projected,
    ):
        assert projected is not None
        assert (
            projected.untyped_storage().data_ptr()
            == workspace.projected.buffer.data_ptr()
        )
        local = projected.narrow(0, rank * count, count)
        local.fill_(rank + 64)
        cp.gather_inplace(projected, local, group)
        with (
            torch.cuda.stream(main_stream),
            cp.borrowed_gather_buffer(
                reference, group, numel=world_size * count, dtype=torch.bfloat16
            ) as history,
        ):
            assert history is not None
            assert history.untyped_storage().data_ptr() == workspace.buffer.data_ptr()
            local = history.narrow(0, rank * count, count)
            local.fill_(rank + 32)
            cp.gather_inplace(history, local, group)
            for owner, part in enumerate(history.chunk(world_size)):
                _assert_constant(part, owner + 32)
        for owner, part in enumerate(projected.chunk(world_size)):
            _assert_constant(part, owner + 64)
    main_stream.wait_stream(projected_stream)


def _check_commit_session(cp, workspace, device, rank, world_size):
    left = torch.full((4,), rank, dtype=torch.int32, device=device)
    right = torch.full_like(left, rank + 16)
    kv = torch.full((16, 512), rank + 32, dtype=torch.bfloat16, device=device)
    assert cp.try_commit_all_gather(left) is None
    with cp.reuse_prefill_cp_collectives(device, cp_size=world_size + 1):
        assert cp.try_commit_all_gather(left) is None
    with cp.reuse_prefill_cp_collectives(device, cp_size=world_size):
        left_all = cp.try_commit_all_gather(left, retain=True)
        right_all = cp.try_commit_all_gather(right, retain=True)
        first = cp.try_commit_all_gather(kv)
        assert left_all is not None and right_all is not None and first is not None
        for view in (left_all, right_all, first):
            assert view.untyped_storage().data_ptr() == workspace.buffer.data_ptr()
        assert left_all.data_ptr() < right_all.data_ptr() < first.data_ptr()
        for owner, part in enumerate(first.chunk(world_size)):
            _assert_constant(part, owner + 32)
        kv.fill_(rank + 64)
        second = cp.try_commit_all_gather(kv)
        assert second is not None and second.data_ptr() == first.data_ptr()
        for owner, part in enumerate(second.chunk(world_size)):
            _assert_constant(part, owner + 64)
        # Private draft KV uses peer batch with an offset past both retained
        # metadata arrays. Check the real CUDA call as well as the results.
        local_bytes = workspace.buffer.nbytes // world_size + 16
        large_kv = torch.empty(local_bytes // 2, dtype=torch.bfloat16, device=device)
        midpoint = large_kv.numel() // 2
        large_kv[:midpoint].fill_(73 + 3 * rank)
        large_kv[midpoint:].fill_(74 + 3 * rank)
        runtime = workspace.runtime
        with patch.object(
            runtime, "cudaMemcpyBatchAsync", wraps=runtime.cudaMemcpyBatchAsync
        ) as batch:
            large = cp.try_commit_all_gather(large_kv)
        assert large is not None
        assert large.untyped_storage().data_ptr() != workspace.buffer.data_ptr()
        batch.assert_called_once()
        offset = 2 * world_size * left.nbytes
        peers = [*range(rank + 1, world_size), *range(rank + 1)]
        assert batch.call_args.args[0] == [
            large.data_ptr() + peer * large_kv.nbytes for peer in peers
        ]
        assert batch.call_args.args[1] == [
            workspace.peer_addresses[peer] + offset for peer in peers
        ]
        assert batch.call_args.args[2] == [large_kv.nbytes] * world_size
        _check_gather(large, world_size, base=72)
        for owner, part in enumerate(left_all.chunk(world_size)):
            _assert_constant(part, owner)
        for owner, part in enumerate(right_all.chunk(world_size)):
            _assert_constant(part, owner + 16)
    assert cp.try_commit_all_gather(left) is None


def _check_ptds_stress(cp, group, workspace, device, rank, world_size):
    main = torch.cuda.current_stream(device)
    side = torch.cuda.Stream(device)
    runtime = workspace.runtime
    status, flags = runtime.cudaStreamGetFlags(side.cuda_stream)
    assert int(status) == 0 and flags & runtime.cudaStreamNonBlocking
    count = 84_483
    iterations = 144
    local = [torch.empty(count, dtype=torch.uint8, device=device) for _ in range(2)]
    out = [
        torch.empty(world_size * count, dtype=torch.uint8, device=device)
        for _ in range(2)
    ]
    observed = torch.empty(iterations, world_size, 3, dtype=torch.uint8, device=device)
    errors = torch.empty(iterations, world_size, dtype=torch.bool, device=device)
    assert main.cuda_stream == 0
    for i in range(iterations):
        slot = i % 2
        stream = (main, side)[slot]
        with torch.cuda.stream(stream):
            torch.cuda._sleep(50_000 * (1 + (rank + i) % 4))
            local[slot].fill_((i + 29 * rank) % 251)
            assert cp.try_all_gather_into(out[slot], local[slot], group)
            torch.cuda._sleep(40_000 * (1 + (2 * rank + i) % 4))
            rows = out[slot].view(world_size, count)
            observed[i, :, 0].copy_(rows[:, 0])
            observed[i, :, 1].copy_(rows[:, count // 2])
            observed[i, :, 2].copy_(rows[:, -1])
            for peer, row in enumerate(rows):
                errors[i, peer].copy_((row != (i + 29 * peer) % 251).any())
            # Immediately recycle private output on its consumer stream.
            out[slot].zero_()
    main.wait_stream(side)
    expected = (
        torch.arange(iterations, device=device)[:, None]
        + 29 * torch.arange(world_size, device=device)[None, :]
    ) % 251
    assert torch.equal(observed, expected[:, :, None].expand_as(observed))
    assert not errors.any().item()


def _worker(rank, rendezvous, world_size, *, local_rank=None):
    from rtp_llm.models_py.modules.dsv41 import _cp_collectives as cp

    device = torch.device("cuda", rank if local_rank is None else local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method="env://" if rendezvous is None else "file://" + rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        group = dist.group.WORLD
        splits_matrix = _exchange_splits(world_size)
        cold_free, total = torch.cuda.mem_get_info(device)
        warm = torch.ones(128, device=device)
        dist.all_reduce(warm)
        torch.cuda.synchronize(device)
        del warm
        torch.cuda.empty_cache()
        before_workspace, _ = torch.cuda.mem_get_info(device)
        model = SimpleNamespace(_cp_size=world_size, _cp_rank=rank)
        with patch.object(cp, "_get_group", return_value=group):
            cp.initialize_prefill_cp_collectives(
                model, device, cp_size=world_size, cp_rank=rank
            )
            with cp.prefill_cp_collectives(model, device, eager=True):
                workspace = cp._workspace(group)
                assert workspace is not None, "symmetric backend did not enroll"
                assert workspace.size == world_size
                assert workspace.buffer.nbytes == 144 * MIB
                assert workspace.projected is not None
                assert workspace.projected.buffer.nbytes == 80 * MIB
                assert workspace.projected.handle.multicast_ptr
                assert workspace.handle.multicast_ptr
                assert (
                    workspace.handle.signal_pad_ptrs
                    != workspace.projected.handle.signal_pad_ptrs
                )
                torch.cuda.synchronize(device)
                after_workspace, _ = torch.cuda.mem_get_info(device)

                # Legacy barriers order the batch on the per-thread default stream.
                assert torch.cuda.current_stream(device).cuda_stream == 0
                local, gathered = _gather_inputs(
                    device,
                    rank,
                    world_size,
                    84_483,
                    torch.uint8,
                    False,
                    output_offset=1,
                )
                runtime = workspace.runtime
                with (
                    patch.object(
                        runtime,
                        "cudaMemcpyBatchAsync",
                        wraps=runtime.cudaMemcpyBatchAsync,
                    ) as batch,
                    patch.object(
                        runtime, "cudaMemcpyAsync", wraps=runtime.cudaMemcpyAsync
                    ) as copy,
                ):
                    assert cp.try_all_gather_into(gathered, local, group)
                batch.assert_called_once()
                assert batch.call_args.args[-1] == runtime.cudaStreamPerThread
                copy.assert_called_once()
                assert all(call.args[-1] == 0 for call in copy.call_args_list)
                _check_gather(gathered, world_size)
                del local, gathered

                for size, dtype, alias in (
                    (5_406_720, torch.bfloat16, True),
                    (65_536, torch.float32, True),
                ):
                    local, gathered = _gather_inputs(
                        device, rank, world_size, size, dtype, alias
                    )
                    assert cp.try_all_gather_into(gathered, local, group)
                    _check_gather(gathered, world_size)
                    del local, gathered

                # Exercise both ownership contracts using aligned sizes and
                # rank-dependent source pointer alignment.
                local, _ = _gather_inputs(
                    device, rank, world_size, 4096, torch.uint8, False
                )
                private = cp.try_all_gather(local, group)
                assert private is not None
                with cp.borrowed_all_gather(local, group) as borrowed:
                    assert borrowed is not None
                    assert (
                        borrowed.untyped_storage().data_ptr()
                        == workspace.buffer.data_ptr()
                    )
                    _check_gather(borrowed, world_size)
                _check_gather(private, world_size)
                with cp.borrowed_all_gather(local[:-1], group) as unaligned:
                    assert unaligned is None
                del local, private, borrowed

                # Force the peer route with a short view of the real window.
                with patch.object(workspace, "buffer", workspace.buffer[: 32 * 1024]):
                    local, gathered = _gather_inputs(
                        device, rank, world_size, 16 * 1024 + 4, torch.uint8, True
                    )
                    assert cp.try_all_gather_into(gathered, local, group)
                    _check_gather(gathered, world_size)
                    del local, gathered

                # Keep the real allocation and shorten the tile in this test.
                # Each 16 KiB tile holds six rows. Source rank 0 has a peer
                # segment crossing that boundary in both layouts.
                with patch.object(cp, "_WORKSPACE_BYTES", 16 * 1024):
                    for transpose in (False, True):
                        splits = (
                            tuple(zip(*splits_matrix)) if transpose else splits_matrix
                        )
                        sent, rows, received, sizes = _exchange_inputs(
                            device, rank, splits, torch.bfloat16, width=1300
                        )
                        assert cp.try_all_to_all_into(
                            received,
                            sent,
                            group,
                            splits_matrix,
                            rows=rows,
                            transpose=transpose,
                        )
                        _check_exchange(received, sizes, rank)
                empty = torch.empty((0, 12), dtype=torch.uint8, device=device)
                empty_rows = torch.empty(0, dtype=torch.long, device=device)
                assert cp.try_all_to_all_into(
                    empty,
                    empty,
                    group,
                    ((0,) * world_size,) * world_size,
                    rows=empty_rows,
                )

                for rows in (1, 4, 32):
                    contribution = torch.full(
                        (rows, 1024),
                        rank + 1 if rank in (0, 2) else 0,
                        dtype=torch.float32,
                        device=device,
                    )
                    assert cp.try_all_reduce_sum_(contribution, group)
                    _assert_constant(contribution, 4 if world_size > 2 else 1)

                stream0, stream1 = torch.cuda.Stream(), torch.cuda.Stream()
                local0, gathered0 = _gather_inputs(
                    device,
                    rank,
                    world_size,
                    40 * MIB + 3,
                    torch.uint8,
                    False,
                    output_offset=1,
                )
                # A full-capacity tile and an odd tail use the same addresses
                # and must preserve the local shard already in the output.
                local1, gathered1 = _gather_inputs(
                    device, rank, world_size, 144 * MIB + 3, torch.uint8, True, base=32
                )
                sent, rows, received, sizes = _exchange_inputs(
                    device, rank, splits_matrix, torch.bfloat16
                )
                contribution = torch.full(
                    (1, 1024), rank + 1, dtype=torch.float32, device=device
                )
                producer = torch.cuda.current_stream()
                stream0.wait_stream(producer)
                stream1.wait_stream(producer)
                with torch.cuda.stream(stream0):
                    assert cp.try_all_gather_into(gathered0, local0, group)
                with torch.cuda.stream(stream1):
                    assert cp.try_all_to_all_into(
                        received, sent, group, splits_matrix, rows=rows
                    )
                with torch.cuda.stream(stream0):
                    assert cp.try_all_reduce_sum_(contribution, group)
                with (
                    torch.cuda.stream(stream1),
                    patch.object(
                        runtime,
                        "cudaMemcpyBatchAsync",
                        wraps=runtime.cudaMemcpyBatchAsync,
                    ) as batch,
                ):
                    assert cp.try_all_gather_into(gathered1, local1, group)
                assert [call.args[2] for call in batch.call_args_list] == [
                    [144 * MIB] * (world_size - 1),
                    [3] * (world_size - 1),
                ]
                peers = [*range(rank + 1, world_size), *range(rank)]
                for start, call in zip((0, 144 * MIB), batch.call_args_list):
                    assert call.args[0] == [
                        gathered1.data_ptr() + peer * local1.nbytes + start
                        for peer in peers
                    ]
                    assert call.args[1] == [
                        workspace.peer_addresses[peer] for peer in peers
                    ]
                assert all(
                    call.args[-1] == stream1.cuda_stream
                    for call in batch.call_args_list
                )
                producer.wait_stream(stream0)
                producer.wait_stream(stream1)
                _check_gather(gathered0, world_size)
                _check_exchange(received, sizes, rank)
                _assert_constant(contribution, world_size * (world_size + 1) // 2)
                _check_gather(gathered1, world_size, base=32)
                del local0, gathered0, local1, gathered1
                _check_ptds_stress(cp, group, workspace, device, rank, world_size)
                _check_borrowed_windows(cp, group, workspace, device, rank, world_size)

            assert cp._workspace(group) is None
            stream0.wait_stream(producer)
            with torch.cuda.stream(stream0):
                _check_commit_session(cp, workspace, device, rank, world_size)
            producer.wait_stream(stream0)
            with cp.prefill_cp_collectives(model, device, eager=False):
                assert cp._workspace(group) is None
            with cp.prefill_cp_collectives(model, device, eager=True):
                assert cp._workspace(group) is workspace
        receipt_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if receipt_dir:
            receipt = {
                "rank": rank,
                "cp_size": world_size,
                "cold_device_used": total - cold_free,
                "device_used_before_workspace": total - before_workspace,
                "device_used_after_workspace": total - after_workspace,
                "workspace_increment": before_workspace - after_workspace,
                "both_signal_pads_distinct": True,
            }
            (
                Path(receipt_dir) / f"cp{world_size}-workspace-rank{rank}.json"
            ).write_text(json.dumps(receipt))
        dist.barrier()
    finally:
        dist.destroy_process_group()


class V41CPCollectivesTest(unittest.TestCase):
    def test_workspace_gather_exchange_reduce_and_cross_stream_reuse(self):
        # Keep the existing four-GPU Bazel target as the default. An isolated
        # host/container can opt into CP2 or CP8 with the same assertions.
        from rtp_llm.models_py.modules.dsv41._cp_collectives import SUPPORTED_CP_SIZES

        world_size = int(os.environ.get("DSV41_CP_TEST_WORLD_SIZE", "4"))
        self.assertIn(world_size, SUPPORTED_CP_SIZES)
        self.assertGreaterEqual(torch.cuda.device_count(), world_size)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as tmp:
            mp.spawn(
                _worker,
                args=(str(Path(tmp) / "rendezvous"), world_size),
                nprocs=world_size,
            )


class V41CPCollectivesInitializationTest(unittest.TestCase):
    def test_fabric_domain_preserves_binary_uuid_and_partition(self):
        import ctypes

        import pynvml

        from rtp_llm.models_py.modules.dsv41 import _cp_collectives as cp

        cluster = bytes(range(16))
        info = pynvml.c_nvmlGpuFabricInfoV_t()
        ctypes.memmove(
            ctypes.addressof(info) + type(info).clusterUuid.offset,
            cluster,
            len(cluster),
        )
        info.cliqueId = 7
        info.status = pynvml.NVML_SUCCESS
        info.state = pynvml.NVML_GPU_FABRIC_STATE_COMPLETED
        sm = Mock()
        sm.get_backend.return_value = "CUDA"
        with (
            patch.object(pynvml, "nvmlInit"),
            patch.object(pynvml, "nvmlShutdown"),
            patch.object(pynvml, "nvmlDeviceGetHandleByUUID"),
            patch.object(pynvml, "c_nvmlGpuFabricInfoV_t", return_value=info),
            patch.object(pynvml, "nvmlDeviceGetGpuFabricInfoV"),
        ):
            self.assertEqual(cp._fabric_domain(sm, "cuda", "local"), (cluster, 7))
            info.state = 0
            self.assertIsNone(cp._fabric_domain(sm, "cuda", "local"))

    def test_inaccessible_fabric_skips_rendezvous(self):
        from rtp_llm.models_py.modules.dsv41 import _cp_collectives as cp

        sm, group = Mock(), Mock()
        with (
            patch.object(cp, "_all", return_value=[True, True]),
            patch.object(cp, "_can_import_fabric_peers", return_value=False) as probe,
        ):
            self.assertEqual(
                cp._allocate_window(sm, group, "cuda", 1024, check_fabric=True),
                (None, None),
            )
        probe.assert_called_once_with(group, sm.empty.return_value)
        sm.rendezvous.assert_not_called()

    def test_rendezvous_failure_votes_before_falling_back(self):
        from rtp_llm.models_py.modules.dsv41 import _cp_collectives as cp

        for local_failure in (False, True):
            sm, group = Mock(), Mock()
            sm.rendezvous.side_effect = (
                RuntimeError("rendezvous unavailable") if local_failure else None
            )
            sm.rendezvous.return_value = SimpleNamespace(multicast_ptr=1)
            with (
                self.subTest(local_failure=local_failure),
                patch.object(
                    cp, "_all", side_effect=[[True, True], [not local_failure, False]]
                ) as vote,
            ):
                self.assertEqual(
                    cp._allocate_window(sm, group, "cuda", 1024), (None, None)
                )
                self.assertEqual(vote.call_count, 2)
                self.assertEqual(vote.call_args.args[1], not local_failure)

    def test_target_creation_registers_the_retained_model(self):
        from rtp_llm.models.deepseek_v41 import DeepSeekV41, DeepSeekV41DSpark

        owner = SimpleNamespace(
            model_config=object(),
            parallelism_config=object(),
            weight=object(),
            moe_config=object(),
            max_generate_batch_size=16,
            fmha_config=object(),
            hw_kernel_config=object(),
            device="cuda:3",
        )
        with patch(
            "rtp_llm.models_py.model_desc.deepseek_v41_model.DeepSeekV41Model"
        ) as create:
            DeepSeekV41._create_python_model(owner)
        self.assertIs(owner.py_model, create.return_value)
        owner.py_model.initialize_prefill_cp_collectives.assert_called_once_with(
            torch.device("cuda", 3)
        )
        with patch(
            "rtp_llm.models_py.model_desc.deepseek_v41_dspark_model.DeepSeekV41DSparkModel"
        ) as create_draft:
            DeepSeekV41DSpark._create_python_model(owner)
        self.assertIs(owner.py_model, create_draft.return_value)
        owner.py_model.initialize_prefill_cp_collectives.assert_not_called()

    def test_model_registration_only_runs_for_the_v41_prefill_target(self):
        from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
        from rtp_llm.models_py.modules.dsv41 import _cp_collectives as cp
        from rtp_llm.ops import RoleType

        device = torch.device("cuda", 3)
        for role, cp_size, world_size, eligible in (
            (RoleType.PREFILL, 2, 2, True),
            (RoleType.PREFILL, 4, 4, True),
            (RoleType.PREFILL, 8, 8, True),
            (RoleType.DECODE, 4, 4, False),
            (RoleType.PDFUSION, 4, 4, False),
            (RoleType.PREFILL, 1, 1, False),
            (RoleType.PREFILL, 3, 3, False),
            (RoleType.PREFILL, 4, 8, False),
        ):
            with self.subTest(role=role, cp_size=cp_size, world_size=world_size):
                owner = SimpleNamespace(
                    _prefill_cp_size=cp_size,
                    parallelism_config=SimpleNamespace(
                        role_type=role, tp_rank=1, world_size=world_size
                    ),
                )
                with patch.object(cp, "initialize_prefill_cp_collectives") as init:
                    DeepSeekV41Model.initialize_prefill_cp_collectives(owner, device)
                if eligible:
                    init.assert_called_once_with(
                        owner, device, cp_size=cp_size, cp_rank=1
                    )
                else:
                    init.assert_not_called()

    def test_forward_scope_only_uses_registered_workspace(self):
        from rtp_llm.models_py.modules.dsv41 import _cp_collectives as cp

        device = torch.device("cuda", 3)
        group = Mock()
        group.size.return_value = 4
        for available in (True, False):
            with self.subTest(available=available):
                workspace = (
                    Mock(
                        device=device,
                        group=group,
                        size=4,
                        buffer=SimpleNamespace(nbytes=144 * MIB),
                        projected=None,
                    )
                    if available
                    else None
                )
                model = SimpleNamespace(_cp_size=4)
                with (
                    patch.object(cp.dist, "is_initialized", return_value=True),
                    patch.object(cp.dist, "get_world_size", return_value=4),
                    patch.object(cp, "_get_group", return_value=group) as get_group,
                    patch.object(cp, "_initialize", return_value=workspace) as init,
                    patch.object(torch.cuda, "device", return_value=nullcontext()),
                    patch.object(torch.cuda, "current_device", return_value=3),
                ):
                    with cp.prefill_cp_collectives(model, device, eager=True):
                        self.assertIsNone(cp._workspace(group))
                    get_group.assert_not_called()
                    init.assert_not_called()
                    for _ in range(2):
                        cp.initialize_prefill_cp_collectives(
                            model, device, cp_size=4, cp_rank=2
                        )
                    for eager, cp_size, forward_device, expected in (
                        (True, 4, device, workspace),
                        (False, 4, device, None),
                        (True, 1, device, None),
                        (True, 4, torch.device("cuda", 0), None),
                    ):
                        model._cp_size = cp_size
                        with cp.prefill_cp_collectives(model, forward_device, eager):
                            self.assertIs(cp._workspace(group), expected)
                        self.assertIsNone(cp._workspace(group))
                    get_group.assert_called_once_with(cp.Group.TP)
                    init.assert_called_once_with(group, device, cp_rank=2)


if __name__ == "__main__":
    unittest.main()
