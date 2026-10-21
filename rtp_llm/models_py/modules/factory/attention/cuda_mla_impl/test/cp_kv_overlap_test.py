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
            stream.assert_called_once_with(device=torch.device("cuda", 0))

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
        wrapper.fmha_impl._allocate_fused_kv.return_value = torch.empty(13, 576)
        wrapper.fmha_impl._pack_sharded_kv_cache.return_value = torch.empty(7, 576)
        wrapper.fmha_impl.sharded_kv_restore_indices = torch.arange(13)
        wrapper.fmha_impl.forward.return_value = torch.ones(2, 1, 512)
        return wrapper

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
        op.attn_inputs = SimpleNamespace(
            is_prefill=True, cache_store_inputs=object()
        )
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
            fused_qkv_a_proj=lambda _: torch.ones(2, 6),
            q_a_layernorm=lambda x: x,
            q_b_proj=lambda _: torch.ones(2, 4),
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
        with patch.object(
            cp.rtp_llm_ops,
            "cp_gather_and_upconvert_fp8_kv_cache_v2",
            side_effect=unpack,
        ), patch.object(cp, "all_gather", side_effect=gather):
            op._gather_sharded_kv_cache(cache, serial)
            op._gather_sharded_kv_cache(cache, asynchronous, gather=gather)
        self.assertEqual(len(seen), 2)
        torch.testing.assert_close(
            serial, expected[op.sharded_kv_restore_indices], atol=0, rtol=0
        )
        torch.testing.assert_close(asynchronous, serial, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
