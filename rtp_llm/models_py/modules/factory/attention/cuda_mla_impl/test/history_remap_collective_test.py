import os
from datetime import timedelta
from unittest import TestCase, main
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.test.utils.port_util import PortManager


def _assert_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ), (actual.shape, actual.dtype)


def _registered_worker(rank: int, world: int, port: int) -> None:
    os.environ["GLM5_CP_OPT"] = "1"
    os.environ["GLM5_CP_PYNCCL_VALIDATE"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=180),
        device_id=device,
    )
    from flash_mla import flash_mla_sparse_fwd

    from rtp_llm.models_py.distributed import pynccl_cp
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.cp_topk_offsets_triton import (
        offset_cp_topk,
    )

    pg = dist.group.WORLD
    pynccl_cp.warmup(pg, device)
    key = (pg, rank, "mla_history", torch.bfloat16)
    assert pynccl_cp.enabled() and pynccl_cp._PYNCCL_VALIDATE
    assert (
        key in pynccl_cp._SYMM_BASES
    ), "Registered MLA history window unavailable; fallback is forbidden"
    base = pynccl_cp._SYMM_BASES[key]
    assert (pg, rank, base.data_ptr()) in pynccl_cp._SYMM_WINDOWS
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    try:
        if os.environ.get("GLM5_CP_PROJECTION_BENCHMARK") == "1":
            from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.cp_projection_benchmark import (
                benchmark_projection,
            )

            benchmark_projection(rank, world)
        with torch.cuda.stream(stream):
            local = torch.empty((256, 576), device=device, dtype=torch.bfloat16)
            independent = torch.empty(
                (world * 256, 576), device=device, dtype=torch.bfloat16
            )
            from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
                cp_bounded_gather,
            )

            # Force multiple chunks while retaining equal padded send shapes on
            # every rank, including a rank whose entire payload is padding.
            projection = torch.arange(64, device=device, dtype=torch.bfloat16).view(
                8, 8
            )
            projection = projection + rank * 100
            if rank == world - 1:
                projection.zero_()
            full_projection = torch.empty(
                (world * 8, 8), device=device, dtype=torch.bfloat16
            )
            pynccl_cp.all_gather_into_tensor(full_projection, projection, group=pg)
            projection_map = torch.tensor(
                [0, 9, 2, 15, 7], device=device, dtype=torch.int64
            )

            def gather_chunk(output, source, **kwargs):
                pynccl_cp.all_gather_into_tensor(output, source, group=pg)

            with patch.object(
                cp_bounded_gather, "_PROJECTION_SCRATCH_BYTES", (world + 1) * 3 * 8 * 2
            ):
                for mapping in (None, projection_map):
                    actual_projection = cp_bounded_gather.gather_projection_chunks(
                        projection[:, :4],
                        projection[:, 4:],
                        world,
                        mapping,
                        gather_chunk,
                    )
                    expected_projection = (
                        full_projection
                        if mapping is None
                        else full_projection.index_select(0, mapping)
                    )
                    _assert_bits(actual_projection, expected_projection)

            # Real NCCL capture/replay: mixed dtypes are separate collectives,
            # and replay refreshes values and the fixed-address restore map.
            for rope_dtype in (torch.float32, torch.float64):
                projection_rope = projection[:, :3].to(rope_dtype) + 0.000123
                full_rope = torch.empty((world * 8, 3), device=device, dtype=rope_dtype)
                scratch_budget = (
                    (world + 1)
                    * 3
                    * max(
                        4 * projection.element_size(),
                        3 * projection_rope.element_size(),
                    )
                )
                with patch.object(
                    cp_bounded_gather, "_PROJECTION_SCRATCH_BYTES", scratch_budget
                ):
                    cp_bounded_gather.gather_projection_components(
                        projection[:, :4],
                        projection_rope,
                        world,
                        projection_map,
                        gather_chunk,
                    )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured_ckv, captured_rope = (
                            cp_bounded_gather.gather_projection_components(
                                projection[:, :4],
                                projection_rope,
                                world,
                                projection_map,
                                gather_chunk,
                            )
                        )
                    pointers = (captured_ckv.data_ptr(), captured_rope.data_ptr())
                    for ids in ([0, 9, 2, 15, 7], [15, 0, 7, 9, 2]):
                        projection.add_(1)
                        projection_rope.add_(0.000321)
                        if rank == world - 1:
                            projection.zero_()
                            projection_rope.zero_()
                        projection_map.copy_(
                            torch.tensor(ids, device=device, dtype=torch.int64)
                        )
                        graph.replay()
                        pynccl_cp.all_gather_into_tensor(
                            full_projection, projection, group=pg
                        )
                        pynccl_cp.all_gather_into_tensor(
                            full_rope, projection_rope, group=pg
                        )
                        _assert_bits(
                            captured_ckv,
                            full_projection[:, :4].index_select(0, projection_map),
                        )
                        _assert_bits(
                            captured_rope, full_rope.index_select(0, projection_map)
                        )
                        assert pointers == (
                            captured_ckv.data_ptr(),
                            captured_rope.data_ptr(),
                        )

            for empty_last_rank in (False, True):
                rows = 0 if empty_last_rank and rank == world - 1 else 4
                q = (
                    torch.randn((rows, 64, 576), device=device, dtype=torch.bfloat16)
                    * 0.1
                )
                mapping = torch.empty(world * 112, device=device, dtype=torch.int64)
                offsets = torch.zeros(rows, device=device, dtype=torch.int32)
                topk = torch.arange(128, device=device, dtype=torch.int32).repeat(
                    rows, 1
                )
                if rows:
                    topk[:, 3] = -1
                    topk[:, 7] = mapping.numel() + 1

                def update(layer: int) -> None:
                    local.normal_(mean=rank * 0.1 + layer * 0.03125, std=0.1)
                    left, right = (48, 64) if layer % 2 == 0 else (64, 48)
                    host_map = torch.cat(
                        [
                            torch.cat(
                                (torch.arange(left), torch.arange(64, 64 + right))
                            )
                            + r * 256
                            for r in range(world)
                        ]
                    )
                    mapping.copy_(host_map.to(device))
                    if rows:
                        offsets.copy_(
                            torch.tensor(
                                [0, 112, 0, 112], device=device, dtype=torch.int32
                            )
                        )

                def candidate() -> tuple[torch.Tensor, ...]:
                    raw = pynccl_cp.all_gather(local, group=pg, role="mla_history")
                    assert (
                        raw.data_ptr() == base.data_ptr()
                    ), "AllGather silently bypassed registered window"
                    if rows == 0:
                        return ()
                    indices = offset_cp_topk(topk, offsets, mapping, raw.shape[0])
                    return flash_mla_sparse_fwd(
                        q, raw.unsqueeze(1), indices.unsqueeze(1), 576**-0.5
                    )

                def reference() -> tuple[torch.Tensor, ...]:
                    dist.all_gather_into_tensor(independent, local, group=pg)
                    if rows == 0:
                        return ()
                    indices = topk + offsets[:, None]
                    indices.masked_fill_(topk < 0, -1)
                    return flash_mla_sparse_fwd(
                        q,
                        independent[mapping].unsqueeze(1),
                        indices.unsqueeze(1),
                        576**-0.5,
                    )

                pending = []
                for layer in range(3):
                    update(layer)
                    expected = reference()
                    actual = candidate()
                    pending.append((actual, expected))
                stream.synchronize()
                for actual, expected in pending:
                    for got, want in zip(actual, expected):
                        _assert_bits(got, want)

                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                stream.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    captured = candidate()
                for layer in (3, 4, 5):
                    update(layer)
                    expected = reference()
                    graph.replay()
                    stream.synchronize()
                    for got, want in zip(captured, expected):
                        _assert_bits(got, want)
                del graph
        torch.cuda.synchronize(device)
        dist.barrier()
        print(
            f"REGISTERED_HISTORY_PASS rank={rank} world={world} eager_layers=6 graph_replays=6 empty_rank={rank == world - 1}",
            flush=True,
        )
    finally:
        pynccl_cp.destroy()
        dist.destroy_process_group()


class HistoryRemapCollectiveTest(TestCase):
    @patch.dict(os.environ, {"GLM5_CP_OPT": "1", "GLM5_CP_PYNCCL_VALIDATE": "1"})
    def test_registered_history_lifetime_and_graph(self) -> None:
        world = int(os.environ.get("RTP_HISTORY_TEST_WORLD_SIZE", "2"))
        self.assertGreaterEqual(torch.cuda.device_count(), world)
        manager = PortManager()
        ports, locks = manager.get_consecutive_ports(1)
        try:
            mp.spawn(
                _registered_worker, args=(world, ports[0]), nprocs=world, join=True
            )
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)


if __name__ == "__main__":
    main()
