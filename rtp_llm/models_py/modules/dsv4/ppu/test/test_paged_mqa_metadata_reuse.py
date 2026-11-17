"""Forward-owned paged-MQA planner reuse contracts."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.kernels.ppu.jit.ppu_fp4_indexer import paged_score
from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.decode.decode_attn_metadata import (
    get_or_build_paged_mqa_plan,
)


def _metadata():
    return SimpleNamespace(paged_mqa_plan_cache={})


class PagedMqaMetadataReuseTest(unittest.TestCase):
    def test_shared_base_metadata_gets_forward_owned_cache_lazily(self):
        metadata = SimpleNamespace()
        source = torch.ones((1, 1), dtype=torch.int32)
        schedule = object()
        deep_gemm = SimpleNamespace(
            get_paged_mqa_logits_metadata=Mock(return_value=schedule)
        )
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            first = get_or_build_paged_mqa_plan(
                metadata,
                source_lengths=source,
                batch_size=1,
                q_len=1,
                batched_queries=False,
                query_index=None,
                page_size=64,
                num_sms=132,
            )
            second = get_or_build_paged_mqa_plan(
                metadata,
                source_lengths=source,
                batch_size=1,
                q_len=1,
                batched_queries=False,
                query_index=None,
                page_size=64,
                num_sms=132,
            )
        self.assertIs(first[1], schedule)
        self.assertIs(second[1], schedule)
        self.assertEqual(deep_gemm.get_paged_mqa_logits_metadata.call_count, 1)
        self.assertEqual(len(metadata.paged_mqa_plan_cache), 1)

    def test_attention_threads_shared_plans_to_every_indexer_layer(self):
        try:
            import deep_gemm
            from rtp_llm.models_py.modules.dsv4.ppu.attention.ppu_fp4_indexer import (
                PpuFP4Attention,
            )
        except ModuleNotFoundError as error:
            self.skipTest(f"full RTP runtime is unavailable: {error}")

        for q_len, batched, expected_builds in ((1, False, 1), (6, True, 1), (6, False, 6)):
            with self.subTest(q_len=q_len, batched=batched):
                attention = PpuFP4Attention.__new__(PpuFP4Attention)
                torch.nn.Module.__init__(attention)
                indexer = SimpleNamespace(
                    _batch_mtp_queries=batched,
                    _kv_eb=64,
                    forward_decode_vectorized=Mock(return_value=None),
                )
                attention.indexer = indexer
                attention._decode_indexer_streams = None
                attention._decode_compressor_meta_from_metadata = Mock(
                    return_value=object()
                )
                batch = 4
                metadata = SimpleNamespace(
                    compressed_lens_per_token={
                        4: torch.arange(
                            1, batch * q_len + 1, dtype=torch.int32
                        ).reshape(batch, q_len)
                    },
                    paged_mqa_plan_cache={},
                    topk_buffer_compressed=torch.empty(
                        (batch, q_len, 8), dtype=torch.int32
                    ),
                )
                builder = Mock(side_effect=lambda lengths, *_: lengths.clone())
                with (
                    patch.object(deep_gemm, "get_num_sms", return_value=132),
                    patch.object(
                        deep_gemm,
                        "get_paged_mqa_logits_metadata",
                        builder,
                    ),
                ):
                    for _ in range(21):
                        attention._decode_update_indexer(
                            torch.empty((batch, q_len, 1)),
                            torch.empty((batch, q_len, 1)),
                            batch,
                            q_len,
                            torch.zeros(batch, dtype=torch.int32),
                            torch.zeros((batch, q_len), dtype=torch.int32),
                            metadata,
                        )
                self.assertEqual(builder.call_count, expected_builds)
                plans = [
                    call.kwargs["paged_mqa_plans"]
                    for call in indexer.forward_decode_vectorized.call_args_list
                ]
                if expected_builds == 1:
                    self.assertEqual(len({plan[1].data_ptr() for plan in plans}), 1)
                else:
                    self.assertTrue(all(len(plan) == q_len for plan in plans))
                    self.assertEqual(
                        len(
                            {
                                tuple(item[1].data_ptr() for item in plan)
                                for plan in plans
                            }
                        ),
                        1,
                    )

    def test_attention_overlap_waits_for_metadata_producer(self):
        try:
            import deep_gemm
            from rtp_llm.models_py.modules.dsv4.ppu.attention.ppu_fp4_indexer import (
                PpuFP4Attention,
            )
        except ModuleNotFoundError as error:
            self.skipTest(f"full RTP runtime is unavailable: {error}")

        attention = PpuFP4Attention.__new__(PpuFP4Attention)
        torch.nn.Module.__init__(attention)
        indexer = SimpleNamespace(
            _batch_mtp_queries=False,
            _kv_eb=64,
            forward_decode_vectorized=Mock(return_value=None),
        )
        attention.indexer = indexer
        attention._decode_indexer_streams = {"q": object(), "weights": object()}
        attention._decode_compressor_meta_from_metadata = Mock(return_value=object())
        metadata = SimpleNamespace(
            compressed_lens_per_token={
                4: torch.arange(1, 5, dtype=torch.int32).reshape(4, 1)
            },
            paged_mqa_plan_cache={},
            topk_buffer_compressed=torch.empty((4, 1, 8), dtype=torch.int32),
        )
        producer, consumer = object(), Mock()
        with (
            patch.object(deep_gemm, "get_num_sms", return_value=132),
            patch.object(
                deep_gemm,
                "get_paged_mqa_logits_metadata",
                side_effect=lambda lengths, *_: lengths.clone(),
            ),
            patch("torch.cuda.current_stream", return_value=consumer),
        ):
            attention._decode_update_indexer(
                torch.empty((4, 1, 1)),
                torch.empty((4, 1, 1)),
                4,
                1,
                torch.zeros(4, dtype=torch.int32),
                torch.zeros((4, 1), dtype=torch.int32),
                metadata,
                q_producer_stream=producer,
            )
        consumer.wait_stream.assert_called_once_with(producer)
        self.assertIs(
            indexer.forward_decode_vectorized.call_args.kwargs["decode_streams"],
            attention._decode_indexer_streams,
        )

    def test_all_b_q_buckets_build_once_for_twenty_one_layers(self):
        builds = []
        deep_gemm = SimpleNamespace(
            get_paged_mqa_logits_metadata=lambda lengths, page, sms: (
                builds.append((lengths.clone(), page, sms)) or torch.tensor([len(builds)])
            )
        )
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            for batch in (1, 4, 8, 16, 32):
                for q_len in range(1, 7):
                    for batched in ((False, True) if q_len > 1 else (False,)):
                        with self.subTest(batch=batch, q_len=q_len, batched=batched):
                            metadata = _metadata()
                            source = torch.arange(
                                1, batch * q_len + 1, dtype=torch.int32
                            ).reshape(batch, q_len)
                            indices = (
                                (None,)
                                if q_len == 1 or batched
                                else tuple(range(q_len))
                            )
                            before = len(builds)
                            first = []
                            for _ in range(21):
                                current = [
                                    get_or_build_paged_mqa_plan(
                                        metadata,
                                        source_lengths=source,
                                        batch_size=batch,
                                        q_len=q_len,
                                        batched_queries=batched,
                                        query_index=index,
                                        page_size=64,
                                        num_sms=132,
                                    )
                                    for index in indices
                                ]
                                if not first:
                                    first = current
                                else:
                                    self.assertEqual(
                                        [x[0].data_ptr() for x in current],
                                        [x[0].data_ptr() for x in first],
                                    )
                                    self.assertEqual(
                                        [x[1].data_ptr() for x in current],
                                        [x[1].data_ptr() for x in first],
                                    )
                            self.assertEqual(len(builds) - before, len(indices))

    def test_fresh_eager_metadata_isolated_and_source_swap_rejected(self):
        deep_gemm = SimpleNamespace(
            get_paged_mqa_logits_metadata=Mock(
                side_effect=lambda *args: torch.tensor([1])
            )
        )
        source = torch.arange(8, dtype=torch.int32).reshape(4, 2)
        kwargs = dict(
            source_lengths=source,
            batch_size=4,
            q_len=2,
            batched_queries=True,
            query_index=None,
            page_size=64,
            num_sms=132,
        )
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            get_or_build_paged_mqa_plan(_metadata(), **kwargs)
            get_or_build_paged_mqa_plan(_metadata(), **kwargs)
            self.assertEqual(deep_gemm.get_paged_mqa_logits_metadata.call_count, 2)
            metadata = _metadata()
            get_or_build_paged_mqa_plan(metadata, **kwargs)
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                get_or_build_paged_mqa_plan(
                    metadata, **{**kwargs, "source_lengths": source.clone()}
                )

    def test_capture_transition_discards_warmup_plan_once(self):
        deep_gemm = SimpleNamespace(
            get_paged_mqa_logits_metadata=Mock(
                side_effect=lambda lengths, *_: lengths.clone()
            )
        )
        metadata = _metadata()
        source = torch.arange(4, dtype=torch.int32).reshape(4, 1)
        kwargs = dict(
            metadata=metadata,
            source_lengths=source,
            batch_size=4,
            q_len=1,
            batched_queries=False,
            query_index=None,
            page_size=64,
            num_sms=132,
        )
        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "torch.cuda.is_current_stream_capturing",
                side_effect=(False, True, True, False),
            ),
        ):
            warmup = get_or_build_paged_mqa_plan(**kwargs)
            captured = get_or_build_paged_mqa_plan(**kwargs)
            again = get_or_build_paged_mqa_plan(**kwargs)
            eager_again = get_or_build_paged_mqa_plan(**kwargs)
        self.assertEqual(deep_gemm.get_paged_mqa_logits_metadata.call_count, 3)
        self.assertNotEqual(warmup[1].data_ptr(), captured[1].data_ptr())
        self.assertEqual(captured[1].data_ptr(), again[1].data_ptr())
        self.assertNotEqual(again[1].data_ptr(), eager_again[1].data_ptr())

    def test_paged_score_uses_explicit_schedule_or_fallback(self):
        explicit = torch.tensor([7])
        fallback = torch.tensor([8])
        logits = torch.tensor([9])
        deep_gemm = SimpleNamespace(
            get_num_sms=Mock(return_value=132),
            get_paged_mqa_logits_metadata=Mock(return_value=fallback),
            fp8_fp4_paged_mqa_logits=Mock(return_value=logits),
        )
        tensors = dict(
            q=torch.zeros((1, 1, 1, 64), dtype=torch.int8),
            q_scales=torch.ones((1, 1, 1), dtype=torch.int32),
            weights=torch.ones((1, 1), dtype=torch.float32),
            cache=torch.zeros((2, 64, 68), dtype=torch.uint8),
            block_table=torch.zeros((1, 1), dtype=torch.int32),
            lengths=torch.ones((1, 1), dtype=torch.int32),
            max_context=64,
        )
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            self.assertIs(paged_score(**tensors, schedule=explicit), logits)
            self.assertIs(
                deep_gemm.fp8_fp4_paged_mqa_logits.call_args.args[5], explicit
            )
            deep_gemm.get_paged_mqa_logits_metadata.assert_not_called()
            self.assertIs(paged_score(**tensors), logits)
            deep_gemm.get_paged_mqa_logits_metadata.assert_called_once()
            self.assertIs(
                deep_gemm.fp8_fp4_paged_mqa_logits.call_args.args[5], fallback
            )


if __name__ == "__main__":
    unittest.main()
