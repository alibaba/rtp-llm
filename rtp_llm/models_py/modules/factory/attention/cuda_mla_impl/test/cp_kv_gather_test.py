"""Main-KV packing and Q overlap: exact values, real NCCL and optional timing.

Run with two visible GPUs. Set CP_KV_BENCHMARK=1 for before/after measurements;
these include packing, restore, Q projection, allocations and the stream join.
"""

import contextlib
import json
import os
import tempfile
import statistics
import traceback
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed import collective_torch, pynccl_cp, symm_mem
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashmla_sparse_cp_impl as cp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
    SparseMlaImpl,
)
from rtp_llm.utils.model_weight import W


def _op(indices, size=2, graph=False):
    op = cp.SparseMlaFp8CPOp.__new__(cp.SparseMlaFp8CPOp)
    op.kv_restore_unpad_indices = indices
    op.prefill_cp_size = size
    op.use_cuda_graph = graph
    op._kv_gather_stream = None
    return op


def _join(result):
    ckv, kpe, stream = result
    if stream is not None:
        consumer = torch.cuda.current_stream(ckv.device)
        consumer.wait_stream(stream)
        ckv.record_stream(consumer)
        kpe.record_stream(consumer)
    return ckv, kpe


def _legacy(op, ckv, kpe):
    full_ckv = cp.all_gather(ckv.contiguous(), group=cp.Group.TP, role="mla_ckv")
    full_kpe = cp.all_gather(kpe.contiguous(), group=cp.Group.TP, role="mla_kpe")
    ids = op.kv_restore_unpad_indices
    return full_ckv[ids], full_kpe[ids]


def _reference(op, ckv, kpe):
    # Use plain NCCL, leaving the production symmetric window/stream untouched.
    values = []
    for tensor in (ckv, kpe):
        source = tensor.contiguous()
        output = torch.empty(
            (source.shape[0] * 2, source.shape[-1]),
            dtype=source.dtype,
            device=source.device,
        )
        dist.all_gather_into_tensor(output, source)
        values.append(output[op.kv_restore_unpad_indices])
    return tuple(values)


class PackedKVGatherTest(unittest.TestCase):
    def test_strided_payload_restore_and_dtype_fallback(self):
        for rows in (0, 1, 7, 129):
            for mixed_dtype in (False, True):
                with self.subTest(rows=rows, mixed_dtype=mixed_dtype):
                    # Both sources are views, as in fused QKV output/RoPE.
                    base = torch.arange(rows * 13, dtype=torch.float32).reshape(
                        rows, 13
                    )
                    ckv, kpe = base[:, 1:9], base[:, 9:12]
                    if mixed_dtype:
                        kpe = kpe.to(torch.float64)
                    ids = torch.arange(rows * 2 - 1, -1, -1)[::2]
                    op = _op(ids)
                    with patch.object(
                        cp,
                        "all_gather",
                        side_effect=lambda t, **_: torch.cat((t, t + 1)),
                    ) as gather:
                        actual_ckv, actual_kpe, stream = op.gather_kv(ckv, kpe)
                    self.assertIsNone(stream)
                    self.assertEqual(gather.call_count, 2 if mixed_dtype else 1)
                    torch.testing.assert_close(
                        actual_ckv, torch.cat((ckv, ckv + 1))[ids], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        actual_kpe, torch.cat((kpe, kpe + 1))[ids], rtol=0, atol=0
                    )
                    self.assertEqual(actual_ckv.dtype, ckv.dtype)
                    self.assertEqual(actual_kpe.dtype, kpe.dtype)

    def test_empty_q_consumes_prestarted_gather_without_second_collective(self):
        op = _op(torch.tensor([3, 0, 2]))
        op.kv_cache_write_op = SimpleNamespace(forward=Mock())
        op.write_cache_store_impl = None
        op.attn_inputs = op.mla_params = None
        op.kv_cache_sharded = False
        op.total_local_ids = torch.empty(0, dtype=torch.long)
        ckv, kpe = torch.randn(2, 8), torch.randn(2, 4)
        with (
            patch.object(
                cp, "all_gather", side_effect=lambda t, **_: torch.cat((t, t))
            ) as gather,
            patch.object(cp.common, "apply_write_cache_store"),
        ):
            gathered = op.gather_kv(ckv, kpe)
            result = op.forward(
                None, ckv, kpe, None, None, object(), gathered_kv=gathered
            )
        self.assertIsNone(result)
        self.assertEqual(gather.call_count, 1)
        written_ckv, written_kpe = op.kv_cache_write_op.forward.call_args.args[:2]
        torch.testing.assert_close(written_ckv, gathered[0], rtol=0, atol=0)
        torch.testing.assert_close(written_kpe, gathered[1], rtol=0, atol=0)


def _distributed_worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=180),
        device_id=device,
    )
    collective_torch._initialized = True
    collective_torch._parallelism_config = SimpleNamespace(
        tp_size=2, dp_size=1, world_size=2
    )
    collective_torch._group_map[collective_torch.Group.DP_AND_TP] = dist.group.WORLD
    original_opt = pynccl_cp._OPT_ENABLED
    graph = None
    try:
        default_symm = symm_mem.init_symm_mem_communicator(dist.group.WORLD)
        if rank == 0:
            print(f"CP_KV_TORCH_SYMM enabled={default_symm is not None}", flush=True)
        symm_counter = (
            patch.object(default_symm, "all_gather", wraps=default_symm.all_gather)
            if default_symm is not None
            else contextlib.nullcontext(None)
        )
        with (
            symm_counter as default_symm_gather,
            patch.object(
                pynccl_cp,
                "_pynccl_symm_all_gather",
                wraps=pynccl_cp._pynccl_symm_all_gather,
            ) as direct_symm_gather,
        ):
            for direct in (False, True):
                pynccl_cp._OPT_ENABLED = direct
                if direct:
                    pynccl_cp.warmup(group=dist.group.WORLD, device=device)
                for rows in (0, 1, 7, 129, 4096):
                    if direct and rows == 0:
                        # Existing pynccl symmetric windows reject zero-size
                        # views (data_ptr=0 is not a registered window).
                        continue
                    torch.manual_seed(801 + rank)
                    base = torch.randn(rows, 1600, dtype=torch.bfloat16, device=device)
                    ckv, kpe = base[:, 1024:1536], base[:, 1536:1600]
                    ids = torch.arange(rows * 2 - 1, -1, -1, device=device)[::2]
                    op = _op(ids)
                    expected = _reference(op, ckv, kpe)
                    for overlap in (True, False):
                        before = direct_symm_gather.call_count
                        before_default = (
                            default_symm_gather.call_count if default_symm_gather else 0
                        )
                        result = op.gather_kv(ckv, kpe, overlap=overlap)
                        if direct and overlap and pynccl_cp._SYMM_BASES:
                            assert direct_symm_gather.call_count == before + 1
                        if not direct and default_symm_gather and rows <= 1024:
                            assert default_symm_gather.call_count == before_default + 1
                        actual = _join(result)
                        for actual_t, expected_t in zip(actual, expected):
                            torch.testing.assert_close(
                                actual_t, expected_t, rtol=0, atol=0
                            )
                    # Exercise allocator reuse after producer temporaries die.
                    temporary = base.clone()
                    result = op.gather_kv(
                        temporary[:, 1024:1536], temporary[:, 1536:1600], overlap=True
                    )
                    del temporary
                    junk = torch.empty_like(base).fill_(float("nan"))
                    for actual_t, expected_t in zip(_join(result), expected):
                        torch.testing.assert_close(actual_t, expected_t, rtol=0, atol=0)
                    del junk
                    # A later layer/op must reuse the stream (and symmetric
                    # window), while waiting for the previous consumer's reads.
                    other_op = _op(ids)
                    first = _join(op.gather_kv(ckv, kpe, overlap=True))
                    snapshots = tuple(t.clone() for t in first)
                    second = _join(other_op.gather_kv(ckv + 1, kpe + 1, overlap=True))
                    assert other_op._kv_gather_stream is op._kv_gather_stream
                    for saved, ref, following in zip(snapshots, expected, second):
                        torch.testing.assert_close(saved, ref, rtol=0, atol=0)
                        torch.testing.assert_close(following, ref + 1, rtol=0, atol=0)

                # Graph path stays on the capture stream; replay reads new data.
                rows = 17
                ckv = torch.full(
                    (rows, 512), rank + 1, dtype=torch.bfloat16, device=device
                )
                kpe = torch.full(
                    (rows, 64), rank + 5, dtype=torch.bfloat16, device=device
                )
                op = _op(torch.arange(rows * 2, device=device).flip(0), graph=True)
                warmup = torch.cuda.Stream(device=device)
                warmup.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup):
                    op.gather_kv(ckv, kpe, overlap=True)
                torch.cuda.current_stream().wait_stream(warmup)
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = op.gather_kv(ckv, kpe, overlap=True)
                assert captured[2] is None
                for increment in (1, 2):
                    ckv.add_(increment)
                    kpe.add_(increment)
                    graph.replay()
                    expected = _reference(op, ckv, kpe)
                    for actual_t, expected_t in zip(captured[:2], expected):
                        torch.testing.assert_close(actual_t, expected_t, rtol=0, atol=0)
                # Eager -> capture -> eager must retain the registered window
                # without resetting its stream ownership between phases.
                op.use_cuda_graph = False
                before = direct_symm_gather.call_count
                result = _join(op.gather_kv(ckv, kpe, overlap=True))
                for actual_t, expected_t in zip(result, expected):
                    torch.testing.assert_close(actual_t, expected_t, rtol=0, atol=0)
                if direct and pynccl_cp._SYMM_BASES:
                    assert direct_symm_gather.call_count == before + 1
                # ProcessGroupNCCL shutdown waits for captured graph references.
                graph = None
                dist.barrier()
                if rank == 0:
                    print(
                        f"CP_KV_EXACT_PASS direct_nccl={direct} rows={'1,7,129,4096' if direct else '0,1,7,129,4096'} graph_replays=2",
                        flush=True,
                    )
        if os.environ.get("CP_KV_BENCHMARK") == "1":
            # Function-call counters above are correctness checks only. Timing
            # uses the unwrapped production group lookup and collectives.
            for direct in (False, True):
                pynccl_cp._OPT_ENABLED = direct
                _benchmark(rank, device, direct)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        graph = None
        pynccl_cp.destroy()
        pynccl_cp._OPT_ENABLED = original_opt
        dist.destroy_process_group()


def _benchmark(rank, device, direct):
    cpus = os.environ.get("CP_KV_BENCH_CPUS")
    if cpus:
        os.sched_setaffinity(0, {int(cpus.split(",")[rank])})
    for rows in (64, 2048, 8192, 32768):
        torch.manual_seed(803 + rank)
        base = torch.randn(rows, 1600, dtype=torch.bfloat16, device=device)
        ckv, kpe = base[:, 1024:1536], base[:, 1536:1600]
        q = torch.randn(rows, 64, 256, dtype=torch.bfloat16, device=device)
        weight = torch.randn(64, 192, 512, dtype=torch.bfloat16, device=device)
        impl = SparseMlaImpl.__new__(SparseMlaImpl)
        impl.num_heads, impl.nope_head_dim, impl.rope_head_dim, impl.kv_lora_rank = (
            64,
            192,
            64,
            512,
        )
        impl.weights = [{W.mla_kc: weight}]
        op = _op(torch.arange(rows * 2, device=device).flip(0))

        def old():
            projected = impl._apply_input_bmm(q, 0)
            return (*_legacy(op, ckv, kpe), projected)

        def packed():
            projected = impl._apply_input_bmm(q, 0)
            return (*_join(op.gather_kv(ckv, kpe)), projected)

        def overlapped():
            gathered = op.gather_kv(ckv, kpe, overlap=True)
            projected = impl._apply_input_bmm(q, 0)
            return (*_join(gathered), projected)

        # Also measure a process whose role window was already bound on the
        # main stream (e.g. capture/warmup) before enabling side-stream overlap.
        functions = (old, packed, overlapped, overlapped)
        expected = old()
        for index, fn in enumerate(functions):
            torch.cuda.synchronize()
            pynccl_cp._SYMM_ROLE_STREAMS.clear()
            if index == 3:
                packed()
            actual = fn()
            for got, ref in zip(actual, expected):
                torch.testing.assert_close(got, ref, rtol=0, atol=0)
            for _ in range(5):
                fn()
        samples = [[] for _ in functions]
        for repeat in range(11):
            for index in (range(4) if repeat % 2 == 0 else reversed(range(4))):
                dist.barrier()
                torch.cuda.synchronize()
                pynccl_cp._SYMM_ROLE_STREAMS.clear()
                if index == 3:
                    packed()
                functions[index]()
                torch.cuda.synchronize()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                begin.record()
                for _ in range(50):
                    result = functions[index]()
                end.record()
                end.synchronize()
                elapsed = torch.tensor(
                    begin.elapsed_time(end) * 1000 / 50, device=device
                )
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                samples[index].append(elapsed.item())
                del result
        if rank == 0:
            print(
                "CP_KV_BENCHMARK "
                + json.dumps(
                    {
                        "rows_per_rank": rows,
                        "world_size": 2,
                        "direct_nccl": direct,
                        "old_us": statistics.median(samples[0]),
                        "packed_us": statistics.median(samples[1]),
                        "overlap_us": statistics.median(samples[2]),
                        "overlap_after_main_binding_us": statistics.median(samples[3]),
                        "samples_us": samples,
                    }
                ),
                flush=True,
            )


class DistributedPackedKVGatherTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
    def test_real_nccl_exact_and_graph(self):
        with tempfile.TemporaryDirectory(prefix="cp-kv-test-") as directory:
            mp.spawn(
                _distributed_worker,
                args=(directory + "/rendezvous",),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
