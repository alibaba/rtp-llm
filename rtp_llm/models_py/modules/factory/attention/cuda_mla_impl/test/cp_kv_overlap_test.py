"""CPU protocol tests; real multi-rank NCCL coverage lives in the GPU test."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashmla_sparse_cp_impl as cp,
)
from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention


class CpKvOverlapTest(unittest.TestCase):
    def setUp(self):
        gather = patch.object(
            cp,
            "all_gather_kv_overlap",
            side_effect=lambda _input, _group, *, output: output.zero_(),
        )
        gather.start()
        self.addCleanup(gather.stop)

    def test_forward_wrappers_share_the_allocator_stream(self):
        from rtp_llm.models_py.distributed import pynccl_cp

        group = object()
        with patch.object(pynccl_cp, "_OVERLAP_STREAMS", {}), patch.object(
            pynccl_cp, "_resolve_process_group", return_value=group
        ), patch.object(pynccl_cp, "_get_comm"), patch.object(
            torch.cuda, "current_device", return_value=0
        ), patch.object(
            torch.cuda, "Stream"
        ) as stream:
            first = pynccl_cp.warmup_kv_overlap(group)
            for _ in range(100):
                self.assertIs(pynccl_cp.warmup_kv_overlap(group), first)
            stream.assert_called_once_with(device=torch.device("cuda", 0), priority=-1)

    def make_wrapper(self):
        wrapper = object.__new__(cp.SparseMlaCpImpl)
        wrapper._glm53_prefill_workspace = None
        wrapper._cp_size = 2
        wrapper._kv_overlap_stream = Mock(name="producer")
        wrapper._apply_rope = Mock()
        wrapper._apply_input_bmm = Mock(side_effect=lambda q, *_a, **_kw: q)
        wrapper._apply_output_bmm = Mock(side_effect=lambda q, *_a: q)
        wrapper.fmha_params = SimpleNamespace(batch_indice_d=None)
        wrapper.fmha_impl = Mock()
        wrapper.fmha_impl.use_cuda_graph = False
        wrapper.fmha_impl._allocate_fused_kv.return_value = torch.empty(13, 576)
        wrapper.fmha_impl._pack_sharded_kv_cache.return_value = torch.empty(7, 576)
        wrapper.fmha_impl.sharded_kv_restore_indices = torch.arange(13)
        wrapper.fmha_impl.forward.return_value = torch.ones(2, 1, 512)
        return wrapper

    def test_cross_layer_keeps_history_pipeline_with_same_layer_fast_path_available(
        self,
    ):
        for fail_consumer in (False, True):
            with self.subTest(fail_consumer=fail_consumer):
                wrapper = self.make_wrapper()
                wrapper.overlap_kv = True
                wrapper.fmha_impl._rank_major_write_slots.return_value = torch.arange(2)
                wrapper._prepare_same_layer_kv_async = Mock(
                    side_effect=AssertionError("must not replace cross-layer history")
                )
                pipeline = Mock()
                prepared = (object(), object())
                pipeline.prepare.return_value = prepared
                wrapper._cross_layer_prefetch = pipeline
                order = []
                wrapper.fmha_impl.write_kv.side_effect = lambda *_a: order.append(
                    "write"
                )
                pipeline.prepare.side_effect = lambda *_a: (
                    order.append("prefetch"),
                    prepared,
                )[1]
                compute = Mock()
                q = torch.ones(2, 1, 576, dtype=torch.bfloat16)
                k = torch.ones(2, 512, dtype=torch.bfloat16)
                pe = torch.ones(2, 64, dtype=torch.bfloat16)
                wrapper._apply_rope.return_value = pe
                cache = object()
                with patch.object(torch.cuda, "current_stream", return_value=compute):
                    self.assertFalse(wrapper.overlap_kv_before_q)
                    try:
                        with wrapper.prepare_kv(q, k, pe, cache, layer_id=3) as result:
                            self.assertIs(result, prepared)
                            if fail_consumer:
                                raise ValueError("consumer failed")
                    except ValueError:
                        if not fail_consumer:
                            raise
                self.assertEqual(order, ["write", "prefetch"])
                pipeline.prepare.assert_called_once_with(3, cache)
                wrapper._prepare_same_layer_kv_async.assert_not_called()
                wrapper.fmha_impl._pack_sharded_kv_cache.assert_not_called()
                # Cross-layer cleanup and the MoE fence own this join, not
                # the layer-exit context (which would serialize lookahead).
                compute.wait_stream.assert_not_called()
                cp.all_gather_kv_overlap.assert_not_called()

    def test_cross_layer_activation_requires_real_cache_prerequisites(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
            cp_kv_prefetch as kv,
        )

        for enabled in (False, True):
            for overlap in (False, True):
                for layerwise in (False, True):
                    for slots in (0, 1, 2):
                        for prefixes in ([0, 0], [0, 64]):
                            with self.subTest(
                                enabled=enabled,
                                overlap=overlap,
                                layerwise=layerwise,
                                slots=slots,
                                prefixes=prefixes,
                            ):
                                wrapper = self.make_wrapper()
                                wrapper.overlap_kv = overlap
                                wrapper._cross_layer_prefetch = None
                                # prepare() may replace the op inputs with an owned
                                # host-prefix snapshot. The wrapper's original input
                                # must not decide whether this forward has history.
                                wrapper.attn_inputs = SimpleNamespace(
                                    prefix_lengths=torch.tensor(
                                        [64, 64] if not any(prefixes) else [0, 0]
                                    )
                                )
                                wrapper.fmha_impl.attn_inputs = SimpleNamespace(
                                    prefix_lengths=torch.tensor(prefixes)
                                )
                                cache = SimpleNamespace(
                                    layerwise=layerwise, staging=[object()] * slots
                                )
                                expected = (
                                    enabled
                                    and overlap
                                    and layerwise
                                    and slots >= 2
                                    and any(prefixes)
                                )
                                with patch.dict(
                                    cp.os.environ,
                                    {"RTP_LLM_CP_KV_CROSS_LAYER": str(int(enabled))},
                                ), patch.object(kv, "CrossLayerKvPrefetch") as factory:
                                    result = wrapper.begin_cross_layer_prefetch(cache)
                                if expected:
                                    factory.assert_called_once_with(wrapper, cache)
                                    self.assertIs(result, factory.return_value)
                                    self.assertIs(wrapper._cross_layer_prefetch, result)
                                else:
                                    factory.assert_not_called()
                                    self.assertIsNone(result)
                                    self.assertIsNone(wrapper._cross_layer_prefetch)

    def test_new_kv_context_joins_only_its_event_not_future_history(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                wrapper = self.make_wrapper()
                wrapper._apply_rope_parts = Mock(side_effect=lambda _q, k: k)
                compute, ready = Mock(), Mock()
                k = torch.ones(2, 512, dtype=torch.bfloat16)
                pe = torch.ones(2, 64, dtype=torch.bfloat16)
                with patch.object(
                    torch.cuda, "current_stream", return_value=compute
                ), patch.object(
                    torch.cuda, "stream", side_effect=lambda _: nullcontext()
                ), patch.object(
                    torch.cuda, "Event", return_value=ready
                ):
                    try:
                        with wrapper.gather_cross_layer_kv_before_q(k, pe) as (
                            raw,
                            event,
                        ):
                            self.assertEqual(tuple(raw.shape), (4, 576))
                            self.assertIs(event, ready)
                            compute.wait_event.assert_not_called()
                            # A future history submission can follow this event.
                            if fail:
                                raise ValueError("Q projection failed")
                    except ValueError:
                        if not fail:
                            raise
                compute.wait_event.assert_called_once_with(ready)
                compute.wait_stream.assert_not_called()
                ready.record.assert_called_once_with(wrapper._kv_overlap_stream)

    def test_composed_new_kv_selects_writer_layout_after_staging_swap(self):
        for rank_major in (False, True):
            for fail in (False, True):
                with self.subTest(rank_major=rank_major, fail=fail):
                    wrapper = self.make_wrapper()
                    pipeline = Mock()
                    prepared = (object(), object())
                    pipeline.prepare.return_value = prepared
                    wrapper._cross_layer_prefetch = pipeline
                    op = wrapper.fmha_impl
                    op._rank_major_write_slots.return_value = (
                        torch.arange(4) if rank_major else None
                    )
                    op.kv_restore_unpad_indices = torch.tensor([3, 0, 2])
                    raw = torch.arange(4 * 576).reshape(4, 576).to(torch.bfloat16)
                    ready, compute = Mock(), Mock()
                    order = []
                    compute.wait_event.side_effect = lambda _e: order.append("ready")
                    op.write_kv.side_effect = lambda *_a, **_kw: order.append("write")
                    pipeline.prepare.side_effect = lambda *_a: (
                        order.append("history"),
                        prepared,
                    )[1]
                    k = torch.ones(2, 512, dtype=torch.bfloat16)
                    pe = torch.ones(2, 64, dtype=torch.bfloat16)
                    cache = object()
                    with patch.object(
                        torch.cuda, "current_stream", return_value=compute
                    ):
                        try:
                            with wrapper.prepare_kv(
                                k, k, pe, cache, layer_id=2, new_kv=(raw, ready)
                            ) as result:
                                self.assertIs(result, prepared)
                                if fail:
                                    raise ValueError("attention failed")
                        except ValueError:
                            if not fail:
                                raise
                    self.assertEqual(order, ["ready", "write", "history"])
                    ckv, kpe, stream = op.write_kv.call_args.kwargs["gathered_kv"]
                    expected = raw if rank_major else raw[[3, 0, 2]]
                    torch.testing.assert_close(
                        torch.cat((ckv, kpe), dim=-1), expected, rtol=0, atol=0
                    )
                    self.assertIsNone(stream)
                    wrapper._apply_rope.assert_not_called()
                    pipeline.prepare.assert_called_once_with(2, cache)
                    compute.wait_stream.assert_not_called()

    def test_producer_joined_after_indexer_exception(self):
        wrapper = self.make_wrapper()
        compute = Mock()
        q = torch.ones(2, 1, 576)
        with patch.object(
            torch.cuda, "current_stream", return_value=compute
        ), patch.object(
            torch.cuda, "stream", side_effect=lambda _: nullcontext()
        ), patch.object(
            torch.cuda, "Event"
        ):
            with self.assertRaisesRegex(ValueError, "indexer failed"):
                with wrapper.prepare_kv(q, q, q, object()):
                    cp.all_gather_kv_overlap.assert_called_once()
                    compute.wait_stream.assert_not_called()
                    raise ValueError("indexer failed")
        wrapper._kv_overlap_stream.wait_stream.assert_called_once_with(compute)
        compute.wait_stream.assert_called_once_with(wrapper._kv_overlap_stream)

    def test_producer_joined_when_submission_fails(self):
        wrapper = self.make_wrapper()
        cp.all_gather_kv_overlap.side_effect = RuntimeError("gather failed")
        compute = Mock()
        q = torch.ones(2, 1, 576)
        with patch.object(
            torch.cuda, "current_stream", return_value=compute
        ), patch.object(torch.cuda, "stream", side_effect=lambda _: nullcontext()):
            with self.assertRaisesRegex(RuntimeError, "gather failed"):
                with wrapper.prepare_kv(q, q, q, object()):
                    self.fail("failed producer must not yield")
        compute.wait_stream.assert_called_once_with(wrapper._kv_overlap_stream)

    def test_ready_wait_is_after_q_bmm_before_attention_and_rope_once(self):
        wrapper = self.make_wrapper()
        compute = Mock()
        events = []
        wrapper._apply_input_bmm.side_effect = lambda q, *_a, **_kw: (
            events.append("q"),
            q,
        )[1]
        compute.wait_event.side_effect = lambda _: events.append("wait")
        wrapper.fmha_impl.forward.side_effect = lambda *_a, **_kw: (
            events.append("attention"),
            None,
        )[1]
        q = torch.ones(2, 1, 576)
        with patch.object(
            torch.cuda, "current_stream", return_value=compute
        ), patch.object(
            torch.cuda, "stream", side_effect=lambda _: nullcontext()
        ), patch.object(
            torch.cuda, "Event"
        ):
            with wrapper.prepare_kv(q, q, q, object()) as prepared:
                wrapper.forward(q, q, q, object(), 0, None, prepared_kv=prepared)
        self.assertEqual(events, ["q", "wait", "attention"])
        wrapper._apply_rope.assert_called_once()
        wrapper.fmha_impl.write_kv.assert_called_once()
        self.assertIs(
            wrapper.fmha_impl.forward.call_args.kwargs["prepared_kv"], prepared[0]
        )

    def test_empty_q_rank_uses_prepared_collective_without_resubmission(self):
        op = object.__new__(cp.SparseMlaFp8CPOp)
        op.use_cuda_graph = False
        op.write_cache_store_impl = None
        op.attn_inputs = SimpleNamespace(is_prefill=True, cache_store_inputs=None)
        op.write_kv = Mock()
        op._gather_sharded_kv_cache = Mock()
        op.kv_cache_sharded = True
        op._gather = object()
        op.total_local_ids = torch.empty(0, dtype=torch.int64)
        self.assertIsNone(
            op.forward(None, None, None, None, None, prepared_kv=torch.empty(1, 576))
        )
        op.write_kv.assert_not_called()
        op._gather_sharded_kv_cache.assert_not_called()

    def test_empty_q_serial_rank_still_participates(self):
        op = object.__new__(cp.SparseMlaFp8CPOp)
        op.use_cuda_graph = False
        op.write_cache_store_impl = None
        op.attn_inputs = SimpleNamespace(is_prefill=True, cache_store_inputs=None)
        op.write_kv = Mock()
        op._gather_sharded_kv_cache = Mock()
        op._allocate_fused_kv = Mock(return_value=torch.empty(1, 576))
        op.kv_cache_sharded = True
        op._gather = object()
        self.assertIsNone(op.forward(None, None, None, None, None))
        op.write_kv.assert_called_once()
        op._gather_sharded_kv_cache.assert_called_once()

    def run_cache_publication(
        self,
        overlap,
        cross_layer=False,
        empty_q=False,
        indexer_fails=False,
        host_backed=False,
        dtype=torch.float32,
    ):
        cache = SimpleNamespace(
            kv_cache_base=torch.zeros(2, 2),
            kv_scale_base=torch.full((2,), -7),
        )
        published = []
        writer = Mock(
            side_effect=lambda kv: published.append(
                (kv.kv_cache_base.clone(), kv.kv_scale_base.clone())
            )
        )
        wrapper = self.make_wrapper()
        wrapper.overlap_kv = overlap
        wrapper._apply_rope.side_effect = lambda _q, k: k
        wrapper._cross_layer_prefetch = None
        if cross_layer:
            wrapper._cross_layer_prefetch = Mock()
            wrapper._cross_layer_prefetch.prepare.return_value = (
                torch.empty(2, 4),
                Mock(),
            )
        op = object.__new__(cp.SparseMlaFp8CPOp)
        op.use_cuda_graph = False
        op.full_rope_pos_ids = None
        op.kv_restore_unpad_indices = torch.arange(2)
        op.kv_cache_write_op = Mock()
        op.kv_cache_write_op.forward.side_effect = (
            lambda *_a, **_kw: cache.kv_cache_base.fill_(11)
        )
        op.mla_params = None
        op.kv_cache_sharded = True
        op.sharded_slot_mapping = torch.arange(2)
        # The layered-cache context suppresses the writer on host-backed
        # layers; only its writeback path may publish their original buffers.
        op.write_cache_store_impl = None if host_backed else writer
        op.attn_inputs = SimpleNamespace(is_prefill=True, cache_store_inputs=object())
        op._gather = object()
        op.total_local_ids = torch.arange(0 if empty_q else 2)
        op.total_local_ids_is_identity = True
        op._allocate_fused_kv = Mock(return_value=torch.empty(2, 4))
        op._pack_sharded_kv_cache = Mock(return_value=torch.empty(2, 4))
        op._gather_sharded_kv_cache = Mock()
        op.sharded_kv_restore_indices = torch.arange(2)
        op._attend_gather = Mock(side_effect=lambda q, *_a: q)
        wrapper.fmha_impl = op

        def indexer(*_args):
            # Exercise the real MlaAttention -> prepare_kv -> CP forward order.
            # A decoder consuming immediately at publication must see new KV.
            writer.assert_not_called()
            if indexer_fails:
                raise ValueError("indexer failed")
            cache.kv_scale_base.fill_(42)
            return None if empty_q else torch.zeros(2, 1, dtype=torch.int32)

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
            fused_qkv_a_proj=lambda _: torch.ones(2, 6, dtype=dtype),
            q_a_layernorm=lambda x: x,
            q_b_proj=lambda _: torch.ones(2, 4, dtype=dtype),
            kv_a_layernorm=lambda x: x,
            o_proj=lambda x: x,
            parallelism_config=SimpleNamespace(get_attn_tp_size=lambda: 1),
            _run_sparse_indexer=indexer,
        )
        with patch.object(
            torch.cuda, "current_stream", return_value=Mock()
        ), patch.object(
            torch.cuda, "stream", side_effect=lambda _: nullcontext()
        ), patch.object(
            cp, "all_gather", side_effect=lambda tensor, **_kw: tensor
        ), patch.object(
            torch.cuda, "Event"
        ):
            if indexer_fails:
                with self.assertRaisesRegex(ValueError, "indexer failed"):
                    MlaAttention.forward(layer, torch.ones(2, 4), wrapper, cache)
            else:
                MlaAttention.forward(layer, torch.ones(2, 4), wrapper, cache)
        if indexer_fails or host_backed:
            writer.assert_not_called()
        else:
            writer.assert_called_once_with(cache)
            torch.testing.assert_close(published[0][0], torch.full((2, 2), 11.0))
            torch.testing.assert_close(published[0][1], torch.full((2,), 42))
        op.kv_cache_write_op.forward.assert_called_once()
        if overlap:
            op._gather_sharded_kv_cache.assert_not_called()

    def test_composed_bf16_real_forward_publication_and_failure(self):
        for host in (False, True):
            for empty in (False, True):
                self.run_cache_publication(
                    True, True, empty_q=empty, host_backed=host, dtype=torch.bfloat16
                )
            self.run_cache_publication(
                True, True, indexer_fails=True, host_backed=host, dtype=torch.bfloat16
            )

    def test_cache_store_sees_both_kv_buffers_after_indexer(self):
        for overlap, cross_layer in ((False, False), (True, False), (True, True)):
            for empty_q in (False, True):
                with self.subTest(
                    overlap=overlap, cross_layer=cross_layer, empty_q=empty_q
                ):
                    self.run_cache_publication(overlap, cross_layer, empty_q)

    def test_indexer_failure_does_not_publish_prepared_cache(self):
        for cross_layer in (False, True):
            with self.subTest(cross_layer=cross_layer):
                self.run_cache_publication(True, cross_layer, indexer_fails=True)

    def test_host_backed_cache_publication_remains_deferred(self):
        for cross_layer in (False, True):
            with self.subTest(cross_layer=cross_layer):
                self.run_cache_publication(True, cross_layer, host_backed=True)

    def test_padding_and_restore_are_identical_with_independent_gather(self):
        op = object.__new__(cp.SparseMlaFp8CPOp)
        op.use_cuda_graph = False
        op.indexer_copy_dst_idx = torch.tensor([0, 1, 2, 8, 9])
        op._gather = SimpleNamespace(batch_size=3)
        op.sharded_local_kv_lens = torch.tensor([4, 4, 4], dtype=torch.int32)
        op.sharded_actual_local_kv_lens = torch.tensor([3, 0, 2], dtype=torch.int32)
        op.sharded_actual_workspace_starts = torch.tensor([0, 3, 3], dtype=torch.int32)
        op.sharded_total_local_kv_len = 12
        op.sharded_actual_total_local_kv_len = 5
        op.sharded_kv_restore_indices = torch.tensor([0, 8, 3, 4, 10, 2, 9])
        op.block_table = torch.zeros(3, 1, dtype=torch.int32)
        cache = SimpleNamespace(
            kv_cache_base=torch.empty(1, 64, 656, dtype=torch.uint8)
        )
        actual = torch.arange(5 * 576).reshape(5, 576).to(torch.bfloat16)
        expected = torch.zeros(12, 576, dtype=torch.bfloat16)
        expected[:3] = actual[:3]
        expected[8:10] = actual[3:]
        seen = []

        def gather(tensor, **_kwargs):
            torch.testing.assert_close(tensor, expected, atol=0, rtol=0)
            seen.append(tensor)
            return tensor

        def unpack(_src, dst, *_args):
            dst.copy_(actual)

        serial = torch.empty(7, 576, dtype=torch.bfloat16)
        asynchronous = torch.empty_like(serial)
        graph_fallback = torch.empty_like(serial)
        with patch.object(
            cp.rtp_llm_ops,
            "cp_gather_and_upconvert_fp8_kv_cache_v2",
            side_effect=unpack,
        ), patch.object(cp, "all_gather", side_effect=gather):
            with patch.object(op, "indexer_copy_dst_idx", None):
                op._gather_sharded_kv_cache(cache, serial)
            op._gather_sharded_kv_cache(cache, asynchronous, gather=gather)
            # Graph replay must not consume eager-only Indexer mappings.
            op.use_cuda_graph = True
            op.indexer_copy_dst_idx = torch.full((5,), 99, dtype=torch.int64)
            op._gather_sharded_kv_cache(cache, graph_fallback, gather=gather)
        self.assertEqual(len(seen), 3)
        torch.testing.assert_close(
            serial, expected[op.sharded_kv_restore_indices], atol=0, rtol=0
        )
        torch.testing.assert_close(asynchronous, serial, atol=0, rtol=0)
        torch.testing.assert_close(graph_fallback, serial, atol=0, rtol=0)

    def test_rank_major_cache_publication_and_empty_q(self):
        with patch.dict("os.environ", {"RTP_CP_REMAP_HISTORY_TOPK": "1"}):
            for host_backed in (False, True):
                for empty_q in (False, True):
                    self.run_cache_publication(
                        True, empty_q=empty_q, host_backed=host_backed
                    )
            self.run_cache_publication(True, indexer_fails=True)

    def test_same_layer_received_buffer_reused_after_cache_write(self):
        wrapper = self.make_wrapper()
        op = wrapper.fmha_impl
        op._rank_major_write_slots.return_value = torch.arange(4)
        op.sharded_total_local_kv_len = 7
        op.sharded_actual_total_local_kv_len = 6
        op._allocate_fused_kv.return_value = torch.empty(13, 576, dtype=torch.bfloat16)
        op._pack_sharded_kv_cache.side_effect = (
            lambda *a, local_fused, **kw: local_fused
        )
        phases = []

        def gather(src, _group, *, output):
            phases.append(("gather", tuple(src.shape)))
            output.fill_(1 if len(phases) == 1 else 2)
            return output

        def write(_ckv, _kpe, _cache, *, gathered_kv):
            a, b, stream = gathered_kv
            self.assertTrue((a == 1).all())
            self.assertTrue((b == 1).all())
            self.assertIsNone(stream)
            phases.append(("write", tuple(a.shape)))

        op.write_kv.side_effect = write
        ckv, kpe = torch.ones(2, 512, dtype=torch.bfloat16), torch.ones(
            2, 64, dtype=torch.bfloat16
        )
        compute = Mock()
        with patch.dict("os.environ", {"RTP_CP_REMAP_HISTORY_TOPK": "1"}), patch.object(
            torch.cuda, "current_stream", return_value=compute
        ), patch.object(
            torch.cuda, "stream", side_effect=lambda _: nullcontext()
        ), patch.object(
            torch.cuda, "Event"
        ), patch.object(
            cp, "all_gather_kv_overlap", side_effect=gather
        ), patch.object(
            torch, "index_select", wraps=torch.index_select
        ) as select:
            with wrapper._prepare_same_layer_kv_async(ckv, kpe, object()) as (
                kv,
                ready,
            ):
                self.assertIsInstance(kv, cp._RankMajorKv)
                self.assertEqual(kv.values.shape, (14, 576))
                self.assertTrue((kv.values == 2).all())
                ready.record.assert_called_once_with(wrapper._kv_overlap_stream)
                compute.wait_stream.assert_not_called()
                select.assert_not_called()
                op._allocate_fused_kv.assert_not_called()
                pack_args, pack_kw = op._pack_sharded_kv_cache.call_args
                self.assertEqual(
                    pack_args[1].data_ptr(), pack_kw["local_fused"].data_ptr()
                )
                self.assertIsNone(pack_kw["actual_fused"])
        self.assertEqual(
            phases, [("gather", (2, 576)), ("write", (4, 512)), ("gather", (7, 576))]
        )
        compute.wait_stream.assert_called_once_with(wrapper._kv_overlap_stream)


class PreparedRankMajorKvTest(unittest.TestCase):
    def test_missing_destination_only_allowed_for_rank_major(self):
        history = torch.arange(24).reshape(6, 4).float()
        indices = torch.tensor([3, 0], dtype=torch.int64)
        for enabled, graph in ((0, False), (1, False), (1, True)):
            with self.subTest(enabled=enabled, graph=graph), patch.dict(
                "os.environ", {"RTP_CP_REMAP_HISTORY_TOPK": str(enabled)}
            ):
                if enabled and not graph:
                    got = cp._prepare_gathered_cp_kv(
                        history, indices, None, use_cuda_graph=graph
                    )
                    self.assertIs(got.values, history)
                else:
                    with self.assertRaisesRegex(ValueError, "requires a destination"):
                        cp._prepare_gathered_cp_kv(
                            history, indices, None, use_cuda_graph=graph
                        )

    def test_layout_selection_empty_ragged_padding_and_alias_lifetime(self):
        # Physical slots include inter-request/rank padding. The logical map
        # contains only valid rows, including an empty request in the middle.
        for indices in ([], [7], [7, 0, 8, 3, 12]):
            for enabled, graph in ((False, False), (True, False), (True, True)):
                with self.subTest(indices=indices, enabled=enabled, graph=graph):
                    mapping = torch.tensor(indices, dtype=torch.int64)
                    history = torch.arange(15 * 576).reshape(15, 576).float()
                    destination = torch.empty(len(indices), 576)
                    expected = history.index_select(0, mapping)
                    with patch.dict(
                        "os.environ", {"RTP_CP_REMAP_HISTORY_TOPK": str(int(enabled))}
                    ), patch.object(
                        torch, "index_select", wraps=torch.index_select
                    ) as select:
                        prepared = cp._prepare_gathered_cp_kv(
                            history, mapping, destination, use_cuda_graph=graph
                        )
                    if enabled and not graph:
                        self.assertIsInstance(prepared, cp._RankMajorKv)
                        self.assertIs(prepared.values, history)
                        self.assertIs(prepared.restore_indices, mapping)
                        select.assert_not_called()
                        # The prepared value retains ownership after locals go away.
                        del history
                        actual = prepared.values.index_select(
                            0, prepared.restore_indices
                        )
                    else:
                        self.assertIs(prepared, destination)
                        select.assert_called_once()
                        actual = prepared
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_ready_and_exception_join_preserved_with_rank_major_view(self):
        fixture = CpKvOverlapTest()
        wrapper = fixture.make_wrapper()
        compute, ready = Mock(), Mock()
        q = torch.ones(2, 1, 576)
        wrapper.fmha_impl.sharded_kv_restore_indices = torch.arange(12, -1, -1)
        raw = None
        with patch.dict("os.environ", {"RTP_CP_REMAP_HISTORY_TOPK": "1"}), patch.object(
            torch.cuda, "current_stream", return_value=compute
        ), patch.object(
            torch.cuda, "stream", side_effect=lambda _: nullcontext()
        ), patch.object(
            torch.cuda, "Event", return_value=ready
        ), patch.object(
            cp,
            "all_gather_kv_overlap",
            side_effect=lambda _x, _g, *, output: output.zero_(),
        ), patch.object(
            torch, "index_select", wraps=torch.index_select
        ) as select:
            with self.assertRaisesRegex(RuntimeError, "consumer failed"):
                with wrapper.prepare_kv(q, q, q, object()) as (prepared, event):
                    self.assertIsInstance(prepared, cp._RankMajorKv)
                    self.assertIs(event, ready)
                    ready.record.assert_called_once_with(wrapper._kv_overlap_stream)
                    compute.wait_stream.assert_not_called()
                    select.assert_not_called()
                    raw = prepared.values
                    raise RuntimeError("consumer failed")
        compute.wait_stream.assert_called_once_with(wrapper._kv_overlap_stream)
        self.assertEqual(raw.shape, (14, 576))


class StagingRankMajorProtocolTest(unittest.TestCase):
    def test_real_plan_and_writer_guard_follow_compact_slots(self):
        # Exercise the production metadata path with real CPU tensors. Only
        # CUDA stream/copy plumbing is mocked; GPU byte tests remain separate.
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.layered_prefill_cache import (
            _PrefillStaging,
        )
        from rtp_llm.ops import compute_ops

        for cp_size in (4, 8):
            for empty in (False, True):
                for failed in (False, True):
                    with self.subTest(cp=cp_size, empty=empty, failed=failed):
                        slots = torch.tensor([508, 131, 200, -1], dtype=torch.int64)
                        if empty:
                            slots = slots[:0]
                        table = torch.tensor(
                            [[7, 2, 9, -1], [7, 3, 0, 0]], dtype=torch.int32
                        )
                        restore = torch.tensor(
                            [cp_size * 2 - 1, 0, cp_size, 1], dtype=torch.int64
                        )[: slots.numel()]
                        plan = cp._rank_major_slot_mapping(slots, restore, cp_size * 2)
                        op = SimpleNamespace(
                            block_table=table,
                            kv_cache_sharded=True,
                            sharded_slot_mapping=slots,
                            rank_major_slot_mapping=plan,
                            _rank_major_source_slots=slots,
                            kv_restore_unpad_indices=restore,
                            write_cache_store_impl=object(),
                        )
                        fmha = SimpleNamespace(
                            fmha_impl=op,
                            overlap_kv=True,
                            attn_inputs=SimpleNamespace(
                                is_prefill=True, kv_cache_kernel_block_id_host=table
                            ),
                            write_cache_store_impl=object(),
                        )
                        store, inner_store = (
                            fmha.write_cache_store_impl,
                            op.write_cache_store_impl,
                        )
                        cache = object.__new__(_PrefillStaging)
                        cache.active = False
                        cache.capacity = 8
                        cache.page_size = 64
                        cache.device = torch.device("cpu")
                        cache.backing = [torch.zeros(16, 64, 656, dtype=torch.uint8)]
                        cache.staging = [torch.zeros(8, 64, 656, dtype=torch.uint8)]
                        cache._fetch_plan = Mock(return_value=[])
                        cache._acquire = Mock(return_value=0)
                        calls = []
                        cache._wait_ready = Mock(
                            side_effect=lambda *args: calls.append("h2d_ready")
                        )
                        cache._writeback = Mock(
                            side_effect=lambda *args: calls.append("writeback")
                        )
                        with patch.object(
                            torch.cuda,
                            "is_current_stream_capturing",
                            return_value=False,
                        ):
                            cache.begin_forward(fmha)
                        expected_slots = torch.tensor(
                            [252, 67, 136, -1], dtype=torch.int64
                        )[: slots.numel()]
                        torch.testing.assert_close(
                            cache.slots, expected_slots, rtol=0, atol=0
                        )
                        expected_plan = torch.full_like(plan, -1)
                        expected_plan[restore] = expected_slots
                        torch.testing.assert_close(
                            cache.rank_major_slots, expected_plan, rtol=0, atol=0
                        )
                        with patch.object(
                            torch.cuda, "current_stream", return_value=object()
                        ), patch.object(
                            compute_ops, "LayerKVCache", side_effect=SimpleNamespace
                        ):
                            try:
                                with cache.attention(1, fmha, object()) as staged:
                                    self.assertIs(
                                        staged.kv_cache_base, cache.staging[0]
                                    )
                                    self.assertIs(
                                        cp.SparseMlaFp8CPOp._rank_major_write_slots(op),
                                        cache.rank_major_slots,
                                    )
                                    self.assertIsNone(fmha.write_cache_store_impl)
                                    self.assertIsNone(op.write_cache_store_impl)
                                    calls.append("attention")
                                    if failed:
                                        raise RuntimeError("injected")
                            except RuntimeError as exc:
                                self.assertTrue(failed)
                                self.assertEqual(str(exc), "injected")
                        self.assertEqual(
                            calls,
                            ["h2d_ready", "attention"]
                            + ([] if failed else ["writeback"]),
                        )
                        self.assertIs(op.sharded_slot_mapping, slots)
                        self.assertIs(op.rank_major_slot_mapping, plan)
                        self.assertIs(
                            cp.SparseMlaFp8CPOp._rank_major_write_slots(op), plan
                        )
                        self.assertIs(op.block_table, table)
                        self.assertIs(fmha.write_cache_store_impl, store)
                        self.assertIs(op.write_cache_store_impl, inner_store)


if __name__ == "__main__":
    unittest.main()
