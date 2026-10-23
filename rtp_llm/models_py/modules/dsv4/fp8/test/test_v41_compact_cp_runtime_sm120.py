"""Real CP4 transport, cache replication and forward-owned failure lifetimes."""

import os
import socket
import unittest
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_compact_cp_runtime as runtime
from rtp_llm.models_py.modules.dsv4.fp8._v41_compact_cp_plan import CompactCPPlan
from rtp_llm.models_py.modules.dsv4.fp8._v41_compact_cp_transport import (
    prepare_metadata,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import _linear_bf16_bf16_fp32
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache
from rtp_llm.ops.compute_ops import rtp_llm_ops


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
    )


def setup(rank, fixture, weights, mode, execution_round):
    device = torch.device("cuda", rank)
    ratio, layer = fixture["ratio"], fixture["layer"]
    tensors = fixture["tensors"]
    start = int(tensors["positions"][0])
    plans = tuple(
        CompactCPPlan(
            mode, -91, 2, 3, execution_round, start // 4096, start, peer, layer, ratio
        )
        for peer in range(4)
    )
    metadata = tuple(prepare_metadata(plan, device) for plan in plans)
    positions = torch.cat([meta.positions for meta in metadata])
    local_positions = (
        torch.cat(
            [
                torch.arange(begin, end, device=device)
                for begin, end in plans[rank].intervals
            ]
        )
        - start
    )
    order = (
        torch.cat(
            [
                torch.cat(
                    [torch.arange(a, b, device=device) for a, b in plan.intervals]
                )
                for plan in plans
            ]
        )
        - start
    )
    cp = SimpleNamespace(
        cp_rank=rank,
        cp_size=4,
        chunk_length=1024,
        seq_len_full=4096,
        unpad_restore_is_prefix=False,
        unpad_restore=order.argsort(),
    )
    dispatch = SimpleNamespace(cp=cp, input_ids=local_positions)
    owner = SimpleNamespace(**weights)
    frequencies = torch.zeros(
        start + 4096, fixture["rope_dim"] // 2, device=device, dtype=torch.complex64
    )
    frequencies[start : start + 4096 : ratio] = tensors["freqs"]
    attention = SimpleNamespace(
        _owner=lambda: owner,
        freqs_cis=frequencies,
        eps=fixture["eps"],
        rope_head_dim=fixture["rope_dim"],
        _shared_attention={},
    )
    caches, slots, expected_raw = [], [], []
    main_tag = "csa_kv" if ratio == 2 else "hca_kv"
    relative = positions - start
    row_indices = relative // ratio
    expected_index = torch.cat(
        (
            tensors["index_payload"].view(torch.uint8),
            tensors["index_scales"].view(torch.uint8).reshape(-1, 4),
        ),
        dim=1,
    )
    for tag, expected, shift in (
        (main_tag, tensors["main_bytes"], rank),
        ("indexer_kv", expected_index, rank + 5),
    ):
        layout = dsv41_page_layouts(0)[tag]
        raw = torch.full(
            (19, layout.stride_bytes), 173, device=device, dtype=torch.uint8
        )
        cache = V41PagedCache(raw, tag, 0)
        page = 1 + ((relative // 256) * 7 + shift) % 17
        slot = (page * layout.entries + (relative % 256) // ratio).contiguous()
        reference = raw.clone()
        V41PagedCache(reference, tag, 0).scatter(
            expected[row_indices], slot, backend="eager"
        )
        caches.append(cache)
        slots.append(slot)
        expected_raw.append(reference)
    state = torch.full((7, 1024), -17.0, device=device) if ratio == 2 else None
    state_slots = torch.tensor([rank, rank + 3], device=device) if ratio == 2 else None
    producer = runtime.Producer(
        attention, plans, metadata, *caches, *slots, state, state_slots
    )
    projection = rtp_llm_ops.V41CompactCpProjection(rank)
    forward = runtime.CompactCPForward(
        mode, "eligible", dispatch, projection, (producer,)
    )
    worker = SimpleNamespace(
        v4=SimpleNamespace(layers=[SimpleNamespace(attn=attention)])
    )
    return worker, forward, producer, tensors["x"][local_positions], expected_raw


def run_rank(rank, port, root):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    group = dist.group.WORLD
    try:
        paths = sorted(Path(root).glob("layer*-chunk*.pt"))
        assert len(paths) == 8
        with patch.object(collective_torch, "_get_group", return_value=group):
            for path in paths:
                fixture = torch.load(
                    path, map_location=f"cuda:{rank}", weights_only=True
                )
                weights = torch.load(
                    Path(root) / f'weights-layer{fixture["layer"]}.pt',
                    map_location=f"cuda:{rank}",
                    weights_only=True,
                )
                for mode in ("projected", "packed"):
                    for iteration in range(3):
                        worker, forward, p, x, expected = setup(
                            rank, fixture, weights, mode, iteration
                        )
                        torch.cuda.synchronize()
                        before = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with (
                            torch.cuda.stream(stream),
                            runtime.execution_scope(worker, forward),
                        ):
                            if rank == iteration % 4:
                                torch.cuda._sleep(2_000_000)
                            forward.start(p.attention, x)
                            pressure = torch.empty(2**20, device=x.device).fill_(
                                iteration
                            )
                            result = forward.finish(p.attention)
                            del pressure
                        torch.cuda.current_stream().wait_stream(stream)
                        if mode == "projected":
                            reference = _linear_bf16_bf16_fp32(
                                fixture["tensors"]["x"], weights["global_wkv"]
                            )
                            if fixture["ratio"] == 2:
                                reference = torch.cat(
                                    (
                                        reference,
                                        _linear_bf16_bf16_fp32(
                                            fixture["tensors"]["x"],
                                            weights["global_wgate"],
                                        ),
                                    ),
                                    dim=1,
                                )
                            exact(result, reference)
                        else:
                            for cache, reference in zip(
                                (p.main_cache, p.index_cache), expected
                            ):
                                exact(cache.raw, reference)
                            if p.state_pool is not None:
                                reference = torch.full_like(p.state_pool, -17)
                                reference[p.state_slots] = fixture["tensors"][
                                    "state_tail"
                                ]
                                exact(p.state_pool, reference)
                        assert forward.closed and forward.active is None
                        assert worker._compact_cp_last_stats["drained"]
                        assert runtime.SHARED_KEY not in p.attention._shared_attention
                        assert (
                            torch.cuda.max_memory_allocated() - before < 128 * 1024**2
                        )
                        with unittest.TestCase().assertRaises(RuntimeError):
                            forward.start(p.attention, x)
                        dist.barrier()

            # Inject recoverable Python failures after a real collective was
            # enqueued. Other peers complete, and each rank reaches the barrier.
            fixture = torch.load(
                paths[0], map_location=f"cuda:{rank}", weights_only=True
            )
            weights = torch.load(
                Path(root) / f'weights-layer{fixture["layer"]}.pt',
                map_location=f"cuda:{rank}",
                weights_only=True,
            )
            for failure in ("cancel", "install", "wait"):
                worker, forward, p, x, _ = setup(rank, fixture, weights, "packed", 20)
                expected_error = unittest.TestCase().assertRaisesRegex(
                    RuntimeError, "injected"
                )
                with expected_error if rank == 1 else nullcontext():
                    with runtime.execution_scope(worker, forward):
                        forward.start(p.attention, x)
                        if rank == 1 and failure == "cancel":
                            raise RuntimeError("injected cancellation")
                        if rank == 1 and failure == "wait":
                            original_wait = forward.active.wait
                            forward.active.wait = lambda: (_ for _ in ()).throw(
                                RuntimeError("injected wait failure")
                            )
                        scope = (
                            patch.object(
                                V41PagedCache,
                                "scatter",
                                side_effect=RuntimeError("injected install failure"),
                            )
                            if rank == 1 and failure == "install"
                            else nullcontext()
                        )
                        with scope:
                            forward.finish(p.attention)
                if rank == 1 and failure == "wait":
                    assert worker._compact_cp_failed is forward
                    assert not worker._compact_cp_last_stats["drained"]
                    with unittest.TestCase().assertRaisesRegex(
                        RuntimeError, "undrained"
                    ):
                        runtime.prepare(worker, None, 21)
                    forward.active.wait = original_wait
                    forward.drain()
                else:
                    assert forward.active is None and forward.closed
                assert runtime.SHARED_KEY not in p.attention._shared_attention
                torch.cuda.synchronize()
                dist.barrier()

            from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
            from rtp_llm.models_py.modules.dsv4 import native_pp_execution

            for failure in ("allocation", "mode", "provider"):
                mode = "projected" if rank == 1 and failure == "mode" else "packed"
                worker, forward, p, x, _ = setup(rank, fixture, weights, mode, 30)
                if rank == 1 and failure == "provider":
                    forward = runtime.CompactCPForward(
                        "off", "projection-provider", forward.dispatch
                    )
                worker._dp_group = group
                worker._stage_agreement = SimpleNamespace(group_size=4)
                worker._engram_cache_dir = Path("/compact-cp-admission-test")
                worker._agreement_config = lambda: {"test": "compact-cp-admission"}
                worker._worker_state = "ready"
                worker._weight_device = lambda: x.device
                worker._agree_dp = lambda *a, **kw: DeepSeekV41Model._agree_dp(
                    worker, *a, **kw
                )
                worker._clear_forward_state = lambda: None
                worker.v4._propagate_cp_ctx = lambda *a: None
                worker.v4.set_cp_info = lambda *a: None
                forward.dispatch.schedule = ("same-schedule",)

                def prepare(*_args):
                    if rank == 1 and failure == "allocation":
                        raise MemoryError("injected compact scratch allocation failure")
                    return forward

                with (
                    patch.object(
                        native_pp_execution,
                        "_validate_execution",
                        return_value=(forward.dispatch, None),
                    ),
                    patch.object(runtime, "prepare", side_effect=prepare),
                    patch.object(
                        runtime.PendingGather,
                        "launch",
                        side_effect=AssertionError("payload launched before admission"),
                    ) as launch,
                    unittest.TestCase().assertRaises(RuntimeError),
                ):
                    native_pp_execution.execute_native_pp_prefill(
                        worker, None, None, 30
                    )
                launch.assert_not_called()
                assert worker._worker_state == "failed"
                assert forward.active is None
                dist.barrier()
    finally:
        dist.destroy_process_group()


class CompactCPRuntime(unittest.TestCase):
    def test_cp4_real_cache_and_lifetime(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 4)
        self.assertIn("DSV41_CP_FIXTURES", os.environ)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(
            run_rank, args=(port, os.environ["DSV41_CP_FIXTURES"]), nprocs=4, join=True
        )


if __name__ == "__main__":
    unittest.main()
