"""Main-KV packing and Q overlap: exact values, real NCCL and optional timing.

Run with two visible GPUs. Set CP_KV_BENCHMARK=1 for before/after measurements;
these include packing, restore, Q projection, allocations and the stream join.
"""

import contextlib
import json
import os
import statistics
import tempfile
import traceback
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

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
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.rope_emb_new import (
    NewMlaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention
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
    def test_early_kv_context_covers_projection_and_indexer_failures(self):
        for failure, host_backed, cross_layer in [
            (f, h, c)
            for f in (None, "projection", "indexer")
            for h in (False, True)
            for c in (False, True)
        ]:
            events = []
            cache, prepared = object(), object()

            @contextlib.contextmanager
            def cache_context(*_):
                events.append("cache_enter")
                try:
                    yield cache
                finally:
                    events.append("cache_exit")

            @contextlib.contextmanager
            def prepare(ckv, kpe, actual_cache):
                self.assertIs(actual_cache, cache)
                events.append("kv_submit")
                try:
                    yield prepared
                finally:
                    events.append("kv_join")

            @contextlib.contextmanager
            def new_gather(ckv, kpe):
                events.append("new_submit")
                try:
                    yield prepared
                finally:
                    events.append("new_event_join")

            @contextlib.contextmanager
            def joint_prepare(q, ckv, kpe, actual_cache, layer_id=0, new_kv=None):
                if cross_layer:
                    self.assertIs(new_kv, prepared)
                    events.append("write_then_history")
                    yield prepared
                else:
                    events.append("joint_rope")
                    with prepare(ckv, kpe, actual_cache) as value:
                        yield value

            def project(q):
                events.append("projection")
                if failure == "projection":
                    raise ValueError("projection")
                return torch.ones(2, 4, dtype=torch.bfloat16)

            def indexer(*_):
                events.append("indexer")
                if failure == "indexer":
                    raise ValueError("indexer")
                return None

            def attend(*_, **kwargs):
                self.assertIs(kwargs["prepared_kv"], prepared)
                events.append("attention")
                return torch.ones(2, 1, 2, dtype=torch.bfloat16)

            impl = SimpleNamespace(
                cross_layer_new_kv_before_q=cross_layer,
                gather_cross_layer_kv_before_q=new_gather,
                overlap_kv_before_q=True,
                overlap_kv=True,
                layered_prefill_cache=(
                    SimpleNamespace(attention=cache_context) if host_backed else None
                ),
                prepare_kv=joint_prepare,
                prepare_kv_before_q=prepare,
                apply_query_rope=lambda _: events.append("q_rope"),
                forward=attend,
            )
            layer = SimpleNamespace(
                q_lora_rank=2,
                kv_lora_rank=2,
                qk_rope_head_dim=2,
                num_heads=1,
                q_head_dim=4,
                v_head_dim=2,
                layer_idx=0,
                _fuse_q_a_norm_mode="off",
                _fuse_kv_a_norm=False,
                fused_qkv_a_proj=lambda _: torch.ones(2, 6, dtype=torch.bfloat16),
                q_a_layernorm=lambda x: x,
                q_b_proj=project,
                kv_a_layernorm=lambda x: x,
                o_proj=lambda x: x,
                parallelism_config=SimpleNamespace(get_attn_tp_size=lambda: 1),
                _run_sparse_indexer=indexer,
            )
            if failure:
                with self.assertRaisesRegex(ValueError, failure):
                    MlaAttention.forward(
                        layer, torch.ones(2, 4, dtype=torch.bfloat16), impl, cache
                    )
            else:
                MlaAttention.forward(
                    layer, torch.ones(2, 4, dtype=torch.bfloat16), impl, cache
                )
            if cross_layer:
                expected = ["new_submit", "projection"]
                if failure != "projection":
                    expected += ["q_rope"]
                    if host_backed:
                        expected += ["cache_enter"]
                    expected += ["write_then_history", "indexer"]
                    if failure is None:
                        expected += ["attention"]
                    if host_backed:
                        expected += ["cache_exit"]
                expected += ["new_event_join"]
                self.assertEqual(events, expected)
                continue
            expected = ["projection"] if host_backed else ["kv_submit", "projection"]
            if failure != "projection":
                expected += (
                    ["cache_enter", "joint_rope", "kv_submit"]
                    if host_backed
                    else ["q_rope"]
                )
                expected += ["indexer"]
            if failure is None:
                expected += ["attention"]
            if not host_backed or failure != "projection":
                expected += ["kv_join"]
            if host_backed and failure != "projection":
                expected += ["cache_exit"]
            self.assertEqual(events, expected)

    def test_rank_major_slots_preserve_padding_and_rr_ownership(self):
        torch.manual_seed(2301)
        for size in (1, 2, 4):
            for owner_size in (64, 512):
                for rank in range(size):
                    # Two requests, prefix offsets, partial owner blocks and
                    # a rank-major permutation with interspersed padding.
                    positions = torch.cat(
                        (torch.arange(61, 170), torch.arange(509, 1100))
                    )
                    batches = torch.cat((torch.zeros(109), torch.ones(591))).long()
                    table = torch.arange(64).reshape(2, 32).int()
                    slots = cp._cp_sharded_slot_mapping(
                        positions,
                        table,
                        batches,
                        64,
                        size,
                        rank,
                        owner_tokens_per_block=owner_size,
                    )
                    rows = (slots.numel() + size - 1) // size * size + size * 3
                    restore = torch.randperm(rows)[: slots.numel()]
                    mapped = cp._rank_major_slot_mapping(slots, restore, rows)
                    torch.testing.assert_close(mapped[restore], slots, rtol=0, atol=0)
                    padding = torch.ones(rows, dtype=torch.bool)
                    padding[restore] = False
                    self.assertTrue(torch.all(mapped[padding] == -1))
                    self.assertEqual(int((mapped >= 0).sum()), int((slots >= 0).sum()))
        empty = cp._rank_major_slot_mapping(
            torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), 0
        )
        self.assertEqual(empty.numel(), 0)
        with self.assertRaises(ValueError):
            cp._rank_major_slot_mapping(torch.tensor([1, 2]), torch.tensor([0]), 2)
        with self.assertRaises(ValueError):
            cp._rank_major_slot_mapping(
                torch.tensor([1]), torch.tensor([0], dtype=torch.int32), 2
            )

    def test_rank_major_gather_and_writer_use_matching_order(self):
        for mixed in (False, True):
            base = torch.arange(4 * 13).reshape(4, 13).float()
            ckv, kpe = base[:, 1:9], base[:, 9:12]
            if mixed:
                kpe = kpe.double()
            restore = torch.tensor([7, 0, 3, 4, 1])
            logical_slots = torch.tensor([12, -1, 3, 5, -1])
            op = _op(restore)
            op.rank_major_slot_mapping = cp._rank_major_slot_mapping(
                logical_slots, restore, 8
            )
            op.kv_cache_sharded = True
            op.sharded_slot_mapping = logical_slots
            op._rank_major_source_slots = logical_slots
            op.mla_params = object()
            op.kv_cache_write_op = SimpleNamespace(forward=Mock())
            with patch.object(
                cp, "all_gather", side_effect=lambda t, **_: torch.cat((t, t + 1))
            ):
                gathered = op.gather_kv(ckv, kpe)
                op.write_kv(ckv, kpe, object(), gathered_kv=gathered)
            actual_ckv, actual_kpe = op.kv_cache_write_op.forward.call_args.args[:2]
            torch.testing.assert_close(
                actual_ckv, torch.cat((ckv, ckv + 1)), rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual_kpe, torch.cat((kpe, kpe + 1)), rtol=0, atol=0
            )
            self.assertIs(
                op.kv_cache_write_op.forward.call_args.kwargs["slot_mapping_override"],
                op.rank_major_slot_mapping,
            )
            # Match layered_prefill_cache.attention(): temporarily remap the
            # physical cache pages, then restore the original slot object.
            staged_slots = torch.tensor([2, -1, 0, 1, -1])
            op.sharded_slot_mapping = staged_slots
            self.assertIsNone(op._rank_major_write_slots())
            with patch.object(
                cp, "all_gather", side_effect=lambda t, **_: torch.cat((t, t + 1))
            ):
                restored = op.gather_kv(ckv, kpe)
                op.write_kv(ckv, kpe, object(), gathered_kv=restored)
            torch.testing.assert_close(restored[0], actual_ckv[restore], rtol=0, atol=0)
            self.assertIs(
                op.kv_cache_write_op.forward.call_args.kwargs["slot_mapping_override"],
                staged_slots,
            )
            op.sharded_slot_mapping = logical_slots
            self.assertIs(op._rank_major_write_slots(), op.rank_major_slot_mapping)
            op.rank_major_slot_mapping = torch.full((7,), -1)
            with self.assertRaisesRegex(ValueError, "gather rows"):
                op.gather_kv(ckv, kpe)

    def test_pack_precedes_producer_event_and_payload_lifetime_is_recorded(self):
        # CPU protocol test: exercise the actual method while modelling only
        # CUDA streams. Real NCCL/allocator/capture checks are below.
        for mixed_dtype in (False, True):
            with self.subTest(mixed_dtype=mixed_dtype):
                base = torch.arange(26, dtype=torch.float32).reshape(2, 13)
                ckv, kpe = base[:, 1:9], base[:, 9:12]
                if mixed_dtype:
                    kpe = kpe.double()
                op = _op(torch.tensor([3, 0, 2]))
                events, recorded = [], []
                producer, side = Mock(name="producer"), Mock(name="side")
                state = {"stream": producer}
                cat, contiguous = torch.cat, torch.Tensor.contiguous

                def pack(*args, **kwargs):
                    self.assertIs(state["stream"], producer)
                    events.append("pack")
                    return cat(*args, **kwargs)

                def copy_payload(tensor, *args, **kwargs):
                    self.assertIs(state["stream"], producer)
                    events.append("copy")
                    return contiguous(tensor, *args, **kwargs)

                def record(tensor, stream):
                    self.assertIs(stream, side)
                    recorded.append(tensor)

                @contextlib.contextmanager
                def stream_context(stream):
                    self.assertIs(stream, side)
                    state["stream"] = side
                    try:
                        yield
                    finally:
                        state["stream"] = producer

                def gather(tensor, **kwargs):
                    self.assertIs(state["stream"], side)
                    self.assertTrue(tensor.is_contiguous())
                    self.assertTrue(any(tensor is t for t in recorded))
                    self.assertTrue(
                        any(op.kv_restore_unpad_indices is t for t in recorded)
                    )
                    events.append("gather")
                    return cat((tensor, tensor + 1))

                side.wait_stream.side_effect = lambda s: (
                    self.assertIs(s, producer),
                    events.append("wait"),
                )
                with (
                    patch.object(
                        torch.Tensor,
                        "is_cuda",
                        new_callable=PropertyMock,
                        return_value=True,
                    ),
                    patch.object(
                        torch.cuda, "is_current_stream_capturing", return_value=False
                    ),
                    patch.object(torch.cuda, "current_stream", return_value=producer),
                    patch.object(torch.cuda, "Stream", return_value=side),
                    patch.object(torch.cuda, "stream", side_effect=stream_context),
                    patch.object(cp, "_KV_GATHER_STREAMS", {}),
                    patch.object(torch, "cat", side_effect=pack),
                    patch.object(torch.Tensor, "contiguous", copy_payload),
                    patch.object(torch.Tensor, "record_stream", record),
                    patch.object(cp, "all_gather", side_effect=gather),
                ):
                    actual_ckv, actual_kpe, actual_stream = op.gather_kv(
                        ckv, kpe, overlap=True
                    )
                self.assertIs(actual_stream, side)
                self.assertEqual(
                    events,
                    (
                        ["copy", "copy", "wait", "gather", "gather"]
                        if mixed_dtype
                        else ["pack", "wait", "gather"]
                    ),
                )
                producer.wait_stream.assert_not_called()
                for got, tensor in ((actual_ckv, ckv), (actual_kpe, kpe)):
                    torch.testing.assert_close(
                        got,
                        cat((tensor, tensor + 1))[op.kv_restore_unpad_indices],
                        rtol=0,
                        atol=0,
                    )

    def test_capture_and_single_rank_do_not_create_side_stream(self):
        for size, graph, capturing in (
            (1, False, False),
            (2, True, False),
            (2, False, True),
        ):
            with self.subTest(size=size, graph=graph, capturing=capturing):
                ckv, kpe = torch.randn(2, 8), torch.randn(2, 4)
                op = _op(torch.arange(2), size=size, graph=graph)
                with (
                    patch.object(
                        torch.Tensor,
                        "is_cuda",
                        new_callable=PropertyMock,
                        return_value=True,
                    ),
                    patch.object(
                        torch.cuda,
                        "is_current_stream_capturing",
                        return_value=capturing,
                    ),
                    patch.object(torch.cuda, "Stream") as stream,
                    patch.object(
                        torch.cuda,
                        "stream",
                        side_effect=lambda s: contextlib.nullcontext(),
                    ),
                    patch.object(cp, "all_gather", side_effect=lambda t, **_: t),
                ):
                    got_ckv, got_kpe, side = op.gather_kv(ckv, kpe, overlap=True)
                stream.assert_not_called()
                self.assertIsNone(side)
                torch.testing.assert_close(got_ckv, ckv, rtol=0, atol=0)
                torch.testing.assert_close(got_kpe, kpe, rtol=0, atol=0)

    def test_torch_fallback_fully_overwrites_uninitialized_output(self):
        # Poison the allocation to catch partial writes/shape mistakes. Neither
        # the ROCm capture nor symmetric-memory dispatch is changed.
        for size in (1, 2, 4):
            for rows in (0, 1, 7):
                for dtype in (torch.bfloat16, torch.float32, torch.int64):
                    with self.subTest(size=size, rows=rows, dtype=dtype):
                        source = torch.arange(rows * 5).reshape(rows, 5).to(dtype)
                        expected = torch.cat([source + rank for rank in range(size)])
                        output = torch.full_like(expected, -123)
                        group = object()

                        def gather(out, inp, *, group):
                            self.assertIs(out, output)
                            self.assertIs(inp, source)
                            for rank in range(size):
                                out[rank * rows : (rank + 1) * rows].copy_(inp + rank)

                        with (
                            patch.object(
                                collective_torch, "_get_rocm_rccl", return_value=None
                            ),
                            patch.object(
                                collective_torch,
                                "_get_symm_mem",
                                return_value=SimpleNamespace(
                                    get_symm_mem_communicator=lambda: None
                                ),
                            ),
                            patch.object(
                                collective_torch, "_get_group", return_value=group
                            ),
                            patch.object(dist, "get_world_size", return_value=size),
                            patch.object(
                                dist, "all_gather_into_tensor", side_effect=gather
                            ) as collective,
                            patch.object(
                                torch, "empty", return_value=output
                            ) as allocate,
                            patch.object(
                                torch,
                                "zeros",
                                side_effect=AssertionError("redundant zero fill"),
                            ),
                        ):
                            result = collective_torch.all_gather(
                                source, collective_torch.Group.TP
                            )
                        allocate.assert_called_once_with(
                            [size * rows, 5], device=source.device, dtype=dtype
                        )
                        collective.assert_called_once_with(output, source, group=group)
                        torch.testing.assert_close(result, expected, rtol=0, atol=0)

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


def _gloo_worker(rank, rendezvous, world_size):
    # Real collective coverage without allocating a CUDA context. This tests
    # the changed C10D fallback, not CUDA stream ordering or overlap speed.
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        with (
            patch.object(collective_torch, "_get_rocm_rccl", return_value=None),
            patch.object(
                collective_torch,
                "_get_symm_mem",
                return_value=SimpleNamespace(get_symm_mem_communicator=lambda: None),
            ),
            patch.object(collective_torch, "_get_group", return_value=dist.group.WORLD),
        ):
            for rows in (0, 1, 7, 129):
                for dtype in (torch.bfloat16, torch.float32, torch.int64):
                    source = torch.arange(rows * 5).reshape(rows, 5).to(dtype)
                    expected = torch.cat([source + r for r in range(world_size)])
                    actual = collective_torch.all_gather(
                        source + rank, collective_torch.Group.TP
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class CpuCollectiveKVGatherTest(unittest.TestCase):
    def test_precomputed_scatter_rejects_invalid_metadata(self):
        actual, padded = torch.ones(3, 5), torch.zeros(8, 5)
        for indices in (
            torch.tensor([[0, 1, 2]]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1, 2], dtype=torch.int32),
        ):
            with self.assertRaisesRegex(ValueError, "precomputed KV scatter"):
                cp._scatter_actual_to_padded(
                    actual=actual,
                    padded=padded,
                    per_req_actual_local_kv_lens=torch.tensor([3]),
                    per_req_padded_local_kv_lens=torch.tensor([8]),
                    dst_indices=indices,
                )

    def test_precomputed_scatter_matches_original_with_padding_and_empty_rows(self):
        for actual_lens, padded_lens in (
            ([0], [4]),
            ([3], [8]),
            ([2, 3], [2, 3]),
            ([0, 3, 0, 7], [8, 8, 8, 16]),
            ([129, 7, 0, 513], [512, 512, 0, 1024]),
        ):
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                actual = torch.arange(sum(actual_lens) * 5).reshape(-1, 5).to(dtype)
                expected = torch.zeros(sum(padded_lens), 5, dtype=dtype)
                got = expected.clone()
                offsets, start = [], 0
                for count, padded in zip(actual_lens, padded_lens):
                    offsets.extend(range(start, start + count))
                    start += padded
                args = dict(
                    actual=actual,
                    per_req_actual_local_kv_lens=torch.tensor(actual_lens),
                    per_req_padded_local_kv_lens=torch.tensor(padded_lens),
                )
                cp._scatter_actual_to_padded(padded=expected, **args)
                with patch.object(
                    torch,
                    "repeat_interleave",
                    side_effect=AssertionError("rebuilt indices"),
                ):
                    cp._scatter_actual_to_padded(
                        padded=got,
                        dst_indices=torch.tensor(offsets, dtype=torch.int64),
                        **args,
                    )
                torch.testing.assert_close(got, expected, rtol=0, atol=0)

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_real_four_rank_fallback_exact(self):
        with tempfile.TemporaryDirectory(prefix="cp-kv-cpu-test-") as directory:
            mp.spawn(
                _gloo_worker,
                args=(directory + "/rendezvous", 4),
                nprocs=4,
                join=True,
            )


def _check_split_rope_exact(device):
    for dtype in (torch.bfloat16, torch.float16):
        for neox in (False, True):
            for contiguous in ("0", "1"):
                for rows, heads in ((1, 64), (17, 64), (4096, 64), (65544, 4)):
                    positions = torch.arange(
                        rows, device=device, dtype=torch.int64
                    ).remainder(1024)
                    phase = torch.randn(1024, 32, device=device)
                    cos_sin = torch.cat((phase.cos(), phase.sin()), dim=-1)
                    original_q = torch.randn(
                        rows, heads, 256, device=device, dtype=dtype
                    )
                    original_kv = torch.randn(rows, 576, device=device, dtype=dtype)
                    q_joint, q_split = original_q.clone(), original_q.clone()
                    kv_joint, kv_split = original_kv.clone(), original_kv.clone()
                    impl = cp.SparseMlaCpImpl.__new__(cp.SparseMlaCpImpl)
                    impl.nope_head_dim = 192
                    impl.fmha_impl = SimpleNamespace(full_rope_pos_ids=positions)
                    impl.rope_params = None
                    impl.rope_impl = NewMlaRotaryEmbeddingOp(cos_sin, neox)
                    with patch.dict(
                        os.environ, {"RTP_CP_CONTIGUOUS_ROPE_KPE": contiguous}
                    ):
                        joint_k = impl._apply_rope(q_joint, kv_joint[:, 512:])
                        key = kv_split[:, 512:]
                        split_k = impl._apply_rope_parts(
                            key.unsqueeze(1)[:, :0, :], key
                        )
                        impl.apply_query_rope(q_split)
                    torch.testing.assert_close(q_split, q_joint, rtol=0, atol=0)
                    torch.testing.assert_close(split_k, joint_k, rtol=0, atol=0)
    print(
        "CP_SPLIT_ROPE_EXACT_PASS bf16,fp16 neox,interleave strided,contiguous rows=1,17,4096,65544",
        flush=True,
    )


def _check_async_kv_pipeline(rank, device):
    # Compare the complete side-stream pipeline to the original writer +
    # history gather, including prefix hits, B2, RR ownership and padding.
    stream = pynccl_cp.warmup_kv_overlap(cp.Group.TP)
    for owner_size in (64, 512):
        lengths = [owner_size * 2 + 7, owner_size + 5]
        new_lens = [11, 7]
        prefixes = [n - a for n, a in zip(lengths, new_lens)]
        total = sum(lengths)
        local_rows = 11
        torch.manual_seed(819)
        restore = torch.randperm(local_rows * 2, device=device)[: sum(new_lens)]
        global_ids = torch.where(restore // local_rows == rank)[0]
        local_ids = restore[global_ids] % local_rows
        cp_info = SimpleNamespace(
            prefill_cp_chunk_lengths=[6, 5],
            prefill_qkv_padding_mask=torch.cat((torch.ones(18), torch.zeros(4))).int(),
            prefill_qkv_restore_indice=torch.cat(
                (
                    restore.cpu(),
                    torch.tensor(
                        [i for i in range(22) if i not in restore.cpu().tolist()]
                    ),
                )
            ),
            prefill_actual_input_lengths_cpu=torch.tensor(new_lens, dtype=torch.int32),
        )
        inputs = SimpleNamespace(
            context_parallel_info=cp_info,
            prefix_lengths=torch.tensor(prefixes, dtype=torch.int32),
            is_prefill=True,
        )
        positions = torch.cat(
            [torch.arange(p, n, device=device) for p, n in zip(prefixes, lengths)]
        ).int()
        batches = torch.cat(
            [torch.full((n,), i, device=device) for i, n in enumerate(new_lens)]
        ).int()
        indptr = torch.tensor([0, lengths[0], total], device=device, dtype=torch.int32)
        params = SimpleNamespace(
            fill_cp_plan_params=lambda *_: None,
            cp_kv_restore_unpad_indices=restore,
            cp_total_global_ids=global_ids,
            cp_total_local_ids=local_ids,
            cp_cu_kv_seqlens_global=indptr,
            cp_total_kv_len=total,
            batch_indice_d=batches,
            positions_d=positions,
            prefill_ragged_kv_len_indptr_d=indptr,
        )
        op = cp.SparseMlaFp8CPOp(
            64,
            512,
            64,
            512,
            64,
            1.0,
            128,
            parallelism_config=SimpleNamespace(
                tp_size=2,
                tp_rank=rank,
                prefill_cp_config=SimpleNamespace(kv_cache_sharded=True),
            ),
        )
        op.kv_owner_tokens_per_block = owner_size
        columns = ((max(lengths) + owner_size * 2 - 1) // (owner_size * 2)) * (
            owner_size // 64
        )
        table = torch.arange(2 * columns, device=device, dtype=torch.int32).reshape(
            2, columns
        )
        with (
            patch.object(op, "_refresh_fp8_kernel_metadata"),
            patch.dict(os.environ, {"RTP_LLM_CP_KV_INDEX_OPT": "1"}),
        ):
            op.plan(params, table, inputs)
        writer = MlaKVCacheWriteOp.__new__(MlaKVCacheWriteOp)
        writer.kv_cache_type = "fp8_ds_mla"
        writer.scale = torch.tensor(1.0, device=device)
        op.kv_cache_write_op = writer
        impl = cp.SparseMlaCpImpl.__new__(cp.SparseMlaCpImpl)
        impl.fmha_impl, impl._cp_size, impl._kv_overlap_stream = op, 2, stream
        full_positions = torch.cat(
            [torch.arange(n, device=device) for n in lengths]
        ).int()
        full_batches = torch.cat(
            [torch.full((n,), i, device=device) for i, n in enumerate(lengths)]
        ).int()
        full_slots = cp._cp_sharded_slot_mapping(
            full_positions,
            table,
            full_batches,
            64,
            2,
            rank,
            owner_tokens_per_block=owner_size,
        )
        logical_rows = torch.cat(
            (
                torch.arange(prefixes[0], lengths[0], device=device),
                torch.arange(lengths[0] + prefixes[1], total, device=device),
            )
        )
        full_data = torch.randn(total, 576, device=device, dtype=torch.bfloat16)
        prefix_slots = full_slots.clone()
        prefix_slots[logical_rows] = -1
        seed_cache = torch.full(
            (2 * columns, 64, 656), 37, device=device, dtype=torch.uint8
        )
        writer.forward(
            full_data[:, :512],
            full_data[:, 512:],
            SimpleNamespace(kv_cache_base=seed_cache),
            None,
            slot_mapping_override=prefix_slots,
        )
        expected_cache, actual_cache = seed_cache.clone(), seed_cache.clone()
        for iteration in range(3):
            new_data = full_data[logical_rows] + iteration
            raw = torch.full(
                (local_rows * 2, 576), float("nan"), device=device, dtype=torch.bfloat16
            )
            raw.index_copy_(0, restore, new_data)
            ckv = raw[rank * local_rows : (rank + 1) * local_rows, :512]
            kpe = raw[rank * local_rows : (rank + 1) * local_rows, 512:]
            writer.forward(
                new_data[:, :512],
                new_data[:, 512:],
                SimpleNamespace(kv_cache_base=expected_cache),
                None,
                slot_mapping_override=op.sharded_slot_mapping,
            )
            expected = op._allocate_fused_kv()
            # Independent oracle: rebuild the original per-layer scatter
            # rather than letting both sides consume the optimized plan.
            with (
                patch.object(op, "indexer_copy_dst_idx", None),
                patch.object(op, "use_cuda_graph", True),
            ):
                op._gather_sharded_kv_cache(
                    SimpleNamespace(kv_cache_base=expected_cache), expected
                )
            try:
                with impl._prepare_same_layer_kv_async(
                    ckv, kpe, SimpleNamespace(kv_cache_base=actual_cache)
                ) as (got, ready):
                    # Exercise both the usual consumer event and exceptional
                    # context exit (which must still join all queued writes).
                    if iteration == 1:
                        raise RuntimeError("intentional consumer failure")
                    torch.cuda.current_stream().wait_event(ready)
                    saved = got.clone()
            except RuntimeError as exc:
                assert str(exc) == "intentional consumer failure"
                saved = got.clone()
            torch.testing.assert_close(saved, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)
    if rank == 0:
        print(
            "CP_KV_ASYNC_PIPELINE_EXACT_PASS B2 prefix RR64,512 padded repeated exception",
            flush=True,
        )


def _check_rank_major_plan_lifecycle(rank, device):
    op = cp.SparseMlaFp8CPOp(
        num_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=512,
        page_size=64,
        softmax_extra_scale=1.0,
        top_k=128,
        parallelism_config=SimpleNamespace(
            tp_size=2,
            tp_rank=rank,
            prefill_cp_config=SimpleNamespace(kv_cache_sharded=True),
        ),
    )
    restore = torch.tensor([3, 0, 2, 1], device=device)
    global_ids = torch.tensor([1, 3] if rank == 0 else [2, 0], device=device)
    info = SimpleNamespace(
        prefill_cp_chunk_lengths=[1, 1],
        prefill_qkv_padding_mask=torch.ones(4, dtype=torch.int32),
        prefill_qkv_restore_indice=restore.cpu(),
        prefill_actual_input_lengths_cpu=torch.tensor([4], dtype=torch.int32),
    )
    inputs = SimpleNamespace(
        context_parallel_info=info,
        prefix_lengths=torch.tensor([2], dtype=torch.int32),
        is_prefill=True,
    )
    params = SimpleNamespace(
        fill_cp_plan_params=lambda *_: None,
        cp_kv_restore_unpad_indices=restore,
        cp_total_global_ids=global_ids,
        cp_total_local_ids=torch.arange(2, device=device),
        cp_cu_kv_seqlens_global=torch.tensor([0, 6], device=device, dtype=torch.int32),
        cp_total_kv_len=6,
        batch_indice_d=torch.zeros(4, device=device, dtype=torch.int32),
        positions_d=torch.arange(2, 6, device=device, dtype=torch.int32),
        prefill_ragged_kv_len_indptr_d=torch.tensor(
            [0, 6], device=device, dtype=torch.int32
        ),
    )
    table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    with (
        patch.object(op, "_refresh_fp8_kernel_metadata"),
        patch.dict(os.environ, {"USE_GATHER_PATH": "1"}),
    ):
        # Refresh, disable, capture policy, then re-enable: never retain a
        # previous plan's rank-major mapping when the writer changes layout.
        for flag, graph, sharded in [
            ("1", False, True),
            ("0", False, True),
            ("1", False, True),
            ("1", True, True),
            ("1", False, True),
            ("1", False, False),
        ]:
            op.use_cuda_graph = graph
            op.kv_cache_sharded = sharded
            with patch.dict(os.environ, {"RTP_LLM_CP_KV_INDEX_OPT": flag}):
                op.plan(params, table, inputs)
            if flag == "1" and not graph and sharded:
                assert op.rank_major_slot_mapping.numel() == 4
                torch.testing.assert_close(
                    op.rank_major_slot_mapping[restore],
                    op.sharded_slot_mapping,
                    rtol=0,
                    atol=0,
                )
            else:
                assert op.rank_major_slot_mapping is None
    if rank == 0:
        print("CP_KV_RANK_MAJOR_PLAN_LIFECYCLE_PASS", flush=True)


def _check_rank_major_cache_bytes(rank, device):
    # Exercise the real gather -> stream join -> cache writer, not a replacement
    # quantizer. Repeated plans/layers must not read a reused symmetric window.
    for dtype in (torch.bfloat16, torch.float16):
        for cache_type in ("fp8_ds_mla", "auto"):
            for rows in (1, 129, 1025):
                total = 2 * rows
                restore = torch.randperm(total, device=device)[: max(1, total - 3)]
                logical_slots = torch.arange(restore.numel(), device=device)
                logical_slots[logical_slots.remainder(2) != rank] = -1
                writer = MlaKVCacheWriteOp.__new__(MlaKVCacheWriteOp)
                writer.kv_cache_type = cache_type
                writer.scale = torch.tensor(1.0, dtype=torch.float32, device=device)
                cache_dtype = torch.uint8 if cache_type == "fp8_ds_mla" else dtype
                width = 656 if cache_type == "fp8_ds_mla" else 576
                reference_cache = torch.full(
                    ((total + 63) // 64, 64, width),
                    37,
                    dtype=cache_dtype,
                    device=device,
                )
                candidate_cache = reference_cache.clone()
                old = _op(restore)
                old.kv_cache_sharded = True
                old.sharded_slot_mapping = logical_slots
                old.mla_params = None
                old.kv_cache_write_op = writer
                new = _op(restore)
                new.kv_cache_sharded = True
                new.sharded_slot_mapping = logical_slots
                new._rank_major_source_slots = logical_slots
                new.rank_major_slot_mapping = cp._rank_major_slot_mapping(
                    logical_slots, restore, total
                )
                new.mla_params = None
                new.kv_cache_write_op = writer
                for layer in range(3):
                    torch.manual_seed(1701 + rank + layer)
                    base = torch.randn(rows, 1600, dtype=dtype, device=device)
                    ckv, kpe = base[:, 1024:1536], base[:, 1536:1600]
                    reference = _reference(old, ckv, kpe)
                    writer.forward(
                        *reference,
                        SimpleNamespace(kv_cache_base=reference_cache),
                        None,
                        slot_mapping_override=logical_slots,
                    )
                    result = new.gather_kv(ckv, kpe, overlap=True)
                    new.write_kv(
                        ckv,
                        kpe,
                        SimpleNamespace(kv_cache_base=candidate_cache),
                        gathered_kv=result,
                    )
                    # Enqueue consumers before the next gather reuses its
                    # registered output window. Check after all three layers.
                    if layer == 0:
                        first_reference = reference_cache.clone()
                        first_candidate = candidate_cache.clone()
                torch.testing.assert_close(
                    first_candidate, first_reference, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    candidate_cache, reference_cache, rtol=0, atol=0
                )
                staged_slots = torch.where(
                    logical_slots >= 0, total - 1 - logical_slots, logical_slots
                )
                new.sharded_slot_mapping = staged_slots
                stage_ref = torch.full_like(reference_cache, 37)
                stage_actual = stage_ref.clone()
                writer.forward(
                    *reference,
                    SimpleNamespace(kv_cache_base=stage_ref),
                    None,
                    slot_mapping_override=staged_slots,
                )
                staged_gather = new.gather_kv(ckv, kpe, overlap=True)
                new.write_kv(
                    ckv,
                    kpe,
                    SimpleNamespace(kv_cache_base=stage_actual),
                    gathered_kv=staged_gather,
                )
                torch.testing.assert_close(stage_actual, stage_ref, rtol=0, atol=0)
                new.sharded_slot_mapping = logical_slots
                writer.forward(
                    *reference,
                    SimpleNamespace(kv_cache_base=stage_ref),
                    None,
                    slot_mapping_override=logical_slots,
                )
                restored_gather = new.gather_kv(ckv, kpe, overlap=True)
                new.write_kv(
                    ckv,
                    kpe,
                    SimpleNamespace(kv_cache_base=stage_actual),
                    gathered_kv=restored_gather,
                )
                torch.testing.assert_close(stage_actual, stage_ref, rtol=0, atol=0)
                if dtype == torch.bfloat16:
                    # Real communication + real quantizing cache writer for
                    # the combined new-KV/Q and cross-layer dispatch. History
                    # math is independently covered by cp_kv_prefetch_gpu_test.
                    wrapper = cp.SparseMlaCpImpl.__new__(cp.SparseMlaCpImpl)
                    wrapper.fmha_impl = new
                    wrapper._cp_size = 2
                    wrapper._kv_overlap_stream = pynccl_cp.warmup_kv_overlap(
                        cp.Group.TP
                    )
                    wrapper._apply_rope_parts = lambda _q, key: key
                    pipeline = Mock()
                    pipeline.prepare.return_value = (None, None)
                    wrapper._cross_layer_prefetch = pipeline
                    for staged in (False, True):
                        new.sharded_slot_mapping = logical_slots
                        ref = torch.full_like(reference_cache, 37)
                        actual = torch.full_like(reference_cache, 37)
                        expected_slots = staged_slots if staged else logical_slots
                        writer.forward(
                            *reference,
                            SimpleNamespace(kv_cache_base=ref),
                            None,
                            slot_mapping_override=expected_slots,
                        )
                        with wrapper.gather_cross_layer_kv_before_q(ckv, kpe) as new_kv:
                            # Match the real order: Q can run before the
                            # host-layer context changes the physical slots.
                            torch.cuda._sleep(10000)
                            if staged:
                                new.sharded_slot_mapping = staged_slots
                            with wrapper.prepare_kv(
                                ckv,
                                ckv,
                                kpe,
                                SimpleNamespace(kv_cache_base=actual),
                                layer_id=1,
                                new_kv=new_kv,
                            ):
                                snapshot = actual.clone()
                        torch.testing.assert_close(snapshot, ref, rtol=0, atol=0)
                        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
                    assert pipeline.prepare.call_count == 2
    if rank == 0:
        print(
            "CP_KV_RANK_MAJOR_CACHE_BYTES_PASS bf16,fp16 fp8_ds_mla,auto rows=1,129,1025 layers=3",
            flush=True,
        )


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
        _check_split_rope_exact(device)
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
                        # Eligibility includes empty payload and ownership of
                        # the scratch stream, not just the number of rows.
                        target = torch.cuda.current_stream(device)
                        if overlap:
                            if device.index not in cp._KV_GATHER_STREAMS:
                                cp._KV_GATHER_STREAMS[device.index] = torch.cuda.Stream(
                                    device=device
                                )
                            target = cp._KV_GATHER_STREAMS[device.index]
                            op._kv_gather_stream = target
                        probe = torch.empty(
                            (rows, 576), dtype=base.dtype, device=device
                        )
                        with torch.cuda.stream(target):
                            expect_torch = bool(
                                default_symm is not None
                                and (not direct or pynccl_cp._TORCH_SYMM_GATHER)
                                and default_symm.should_torch_symm_mem_allgather(
                                    probe, max_gathered_rows=None if direct else 2048
                                )
                            )
                        before = direct_symm_gather.call_count
                        before_default = (
                            default_symm_gather.call_count if default_symm_gather else 0
                        )
                        result = op.gather_kv(ckv, kpe, overlap=overlap)
                        if (
                            direct
                            and overlap
                            and pynccl_cp._SYMM_BASES
                            and not expect_torch
                        ):
                            assert direct_symm_gather.call_count == before + 1
                        if default_symm_gather:
                            assert (
                                default_symm_gather.call_count
                                == before_default + int(expect_torch)
                            ), (
                                direct,
                                overlap,
                                rows,
                                expect_torch,
                            )
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

                _check_rank_major_plan_lifecycle(rank, device)
                _check_rank_major_cache_bytes(rank, device)
                _check_async_kv_pipeline(rank, device)

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
                before_torch = (
                    default_symm_gather.call_count if default_symm_gather else 0
                )
                result = _join(op.gather_kv(ckv, kpe, overlap=True))
                for actual_t, expected_t in zip(result, expected):
                    torch.testing.assert_close(actual_t, expected_t, rtol=0, atol=0)
                if direct and pynccl_cp._SYMM_BASES:
                    # Eager can select Torch multicast before the registered
                    # NCCL path. Either optimized backend must still execute
                    # exactly one gather; captured outputs were checked above.
                    torch_calls = (
                        (default_symm_gather.call_count - before_torch)
                        if default_symm_gather
                        else 0
                    )
                    assert direct_symm_gather.call_count - before + torch_calls == 1
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
            for index in range(4) if repeat % 2 == 0 else reversed(range(4)):
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
