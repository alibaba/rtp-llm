import itertools
from unittest import SkipTest, TestCase, main, mock

import torch
from torch import dtype as _dtype

from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalDeepstackInjector,
    MultimodalEmbeddingInjector,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


class MultimodalEmbeddingTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    SEQUENCE_LENGTH = [1, 5, 10, 100, 1024, 2048, 4096]
    NUM_FEATURES = [0, 1, 5]
    HIDDEN_SIZES = [768, 2560, 8192]
    LAYER_IDS = [0, 1, 5]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_multimodal_embedding_test(
        self, seq_len: int, num_features: int, hidden_size: int, dtype: _dtype
    ):
        if seq_len < num_features:
            return
        torch.manual_seed(0)
        embeddings = torch.randn(seq_len, hidden_size, device="cuda", dtype=dtype)
        expected = embeddings.clone()

        features = []
        locs = []
        for i in range(num_features):
            feature_len = min(seq_len, max(1, (i % 4) + 1))
            max_start = max(0, seq_len - feature_len)
            start = 0 if max_start == 0 else (i * 3) % (max_start + 1)
            feature = torch.randn(feature_len, hidden_size, device="cuda", dtype=dtype)
            features.append(feature)
            locs.append(start)
            expected[start : start + feature_len] = feature

        injector = MultimodalEmbeddingInjector().cuda()
        loc_tensor = (
            torch.tensor(locs, device="cuda", dtype=torch.int32)
            if locs
            else torch.empty(0, device="cuda", dtype=torch.int32)
        )
        output = injector(embeddings.clone(), features, loc_tensor)
        self.assertTrue(torch.allclose(output, expected))

    def _run_deepstack_embedding_test(
        self,
        seq_len: int,
        num_features: int,
        hidden_size: int,
        layer_id: int,
        dtype: _dtype,
    ):
        torch.manual_seed(1)
        hidden = torch.randn(seq_len, hidden_size, device="cuda", dtype=dtype)
        expected = hidden.clone()

        deepstack_tensors = []
        locs = []
        for i in range(num_features):
            layers = (i % 4) + 1
            token_len = min(seq_len, max(1, (i % 3) + 1))
            max_start = max(0, seq_len - token_len)
            start = 0 if max_start == 0 else (i * 5) % (max_start + 1)
            tensor = torch.randn(
                layers, token_len, hidden_size, device="cuda", dtype=dtype
            )
            deepstack_tensors.append(tensor)
            locs.append(start)
            if layer_id < layers:
                expected[start : start + token_len] += tensor[layer_id]

        injector = MultimodalDeepstackInjector().cuda()
        loc_tensor = (
            torch.tensor(locs, device="cuda", dtype=torch.int32)
            if locs
            else torch.empty(0, device="cuda", dtype=torch.int32)
        )
        output = injector(hidden.clone(), deepstack_tensors, loc_tensor, layer_id)
        self.assertTrue(torch.allclose(output, expected))

    def test_multimodal_embedding(self):
        for params in itertools.product(
            self.SEQUENCE_LENGTH,
            self.NUM_FEATURES,
            self.HIDDEN_SIZES,
            self.DTYPES,
        ):
            with self.subTest(
                seq_len=params[0],
                num_features=params[1],
                hidden_size=params[2],
                dtype=params[3],
            ):
                self._run_multimodal_embedding_test(*params)

    def test_multimodal_deepstack_embedding(self):
        for params in itertools.product(
            self.SEQUENCE_LENGTH,
            self.NUM_FEATURES,
            self.HIDDEN_SIZES,
            self.LAYER_IDS,
            self.DTYPES,
        ):
            with self.subTest(
                seq_len=params[0],
                num_features=params[1],
                hidden_size=params[2],
                layer_id=params[3],
                dtype=params[4],
            ):
                self._run_deepstack_embedding_test(*params)

    def test_rejects_negative_multimodal_locations(self):
        embeddings = torch.zeros(4, 2, dtype=torch.half)
        feature = torch.ones(2, 2, dtype=torch.half)
        locations = torch.tensor([-1], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "loc must be non-negative"):
            MultimodalEmbeddingInjector()(embeddings, [feature], locations)

        deepstack = torch.ones(1, 2, 2, dtype=torch.half)
        with self.assertRaisesRegex(ValueError, "loc must be non-negative"):
            MultimodalDeepstackInjector()(
                embeddings,
                [deepstack],
                locations,
                layer_id=0,
            )

    def test_injects_upstream_cropped_feature_at_zero(self):
        embeddings = torch.zeros(4, 2, dtype=torch.half)
        feature = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.half,
        )

        output = MultimodalEmbeddingInjector()(
            embeddings,
            [feature],
            torch.tensor([0], dtype=torch.int32),
        )

        torch.testing.assert_close(output[:2], feature)
        torch.testing.assert_close(
            output[2:],
            torch.zeros(2, 2, dtype=torch.half),
        )

    def test_injector_preserves_noncontiguous_destination_storage(self):
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                backing = torch.arange(48, dtype=dtype, device="cuda").reshape(6, 8)
                expected = backing.clone()
                embeddings = backing[1:5, ::2]
                feature = torch.full((2, 4), -7.0, dtype=dtype, device="cuda")
                expected[2:4, ::2] = feature
                self.assertFalse(embeddings.is_contiguous())

                with mock.patch.object(rtp_llm_ops, "fused_multimodal_copy_") as fused:
                    output = MultimodalEmbeddingInjector()(
                        embeddings, [feature], torch.tensor([1], dtype=torch.int32)
                    )

                fused.assert_not_called()
                self.assertIs(output, embeddings)
                torch.testing.assert_close(backing, expected, rtol=0, atol=0)

    def test_injector_fuses_noncontiguous_sources_for_disjoint_ranges(self):
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                embeddings = torch.zeros((5, 4), dtype=dtype, device="cuda")
                feature = torch.arange(8, dtype=dtype, device="cuda").reshape(4, 2).t()
                self.assertFalse(feature.is_contiguous())
                features = [feature, feature + 10]
                expected = embeddings.clone()
                expected[0:2] = features[0]
                expected[3:5] = features[1]

                with mock.patch.object(
                    rtp_llm_ops,
                    "fused_multimodal_copy_",
                    wraps=rtp_llm_ops.fused_multimodal_copy_,
                ) as fused:
                    output = MultimodalEmbeddingInjector()(
                        embeddings, features, torch.tensor([0, 3], dtype=torch.int32)
                    )

                fused.assert_called_once()
                self.assertTrue(
                    all(src.is_contiguous() for src in fused.call_args.args[1])
                )
                self.assertIs(output, embeddings)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_fast_copy_validates_entire_ordered_batch_before_writing(self):
        dst = torch.zeros((8, 4), dtype=torch.bfloat16, device="cuda")
        sources = torch.arange(16, dtype=dst.dtype, device=dst.device).reshape(2, 2, 4)
        first, last = sources.unbind()
        for rejected_source, offsets in (
            (last, [0, 1]),
            (last, [3, 0]),
            (last, [0, 7]),
            (last.float(), [0, 4]),
            (dst[:2], [0, 4]),
        ):
            self.assertFalse(
                rtp_llm_ops.try_fused_multimodal_copy_(
                    dst, [first, rejected_source], offsets
                )
            )
            self.assertEqual(dst.count_nonzero().item(), 0)
        self.assertTrue(
            rtp_llm_ops.try_fused_multimodal_copy_(dst, [first, last], [0, 4])
        )
        torch.testing.assert_close(dst[:2], first)
        torch.testing.assert_close(dst[4:6], last)

    def test_aliased_features_read_in_input_order(self):
        for step in (1, 2):
            with self.subTest(destination_stride=step):
                backing = torch.arange(32, device="cuda", dtype=torch.float16).view(
                    8, 4
                )
                dst = backing[::step]
                # The second source observes the first write. The first copy may
                # also partially overlap its own destination.
                expected = backing.clone()
                reference = expected[::step]
                sources = [backing[1:3], backing[:1]]
                reference[:2].copy_(expected[1:3].clone())
                reference[3:4].copy_(expected[:1].clone())
                result = MultimodalEmbeddingInjector()(
                    dst, sources, torch.tensor([0, 3], dtype=torch.int32, device="cpu")
                )
                self.assertIs(result, dst)
                torch.testing.assert_close(backing, expected, rtol=0, atol=0)

    def test_injector_preserves_last_writer_for_overlapping_ranges(self):
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                embeddings = torch.zeros((5, 4), dtype=dtype, device="cuda")
                first = torch.full((3, 4), 2.0, dtype=dtype, device="cuda")
                last = torch.full((2, 4), 9.0, dtype=dtype, device="cuda")
                expected = embeddings.clone()
                expected[1:4] = first
                expected[2:4] = last

                with mock.patch.object(rtp_llm_ops, "fused_multimodal_copy_") as fused:
                    output = MultimodalEmbeddingInjector()(
                        embeddings,
                        [first, last],
                        torch.tensor([1, 2], dtype=torch.int32),
                    )

                fused.assert_not_called()
                self.assertIs(output, embeddings)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_injector_rejects_feature_location_count_mismatch(self):
        injector = MultimodalEmbeddingInjector().cuda()
        embeddings = torch.zeros(4, 4, device="cuda", dtype=torch.half)
        features = [
            torch.ones((1, 4), device="cuda", dtype=torch.half),
            torch.ones((1, 4), device="cuda", dtype=torch.half),
        ]

        with self.assertRaisesRegex(ValueError, "1 entries but 2 features"):
            injector(
                embeddings,
                features,
                torch.tensor([0], device="cuda", dtype=torch.int32),
            )

        # Producers may leave the tensor undefined, which arrives here as None.
        with self.assertRaisesRegex(ValueError, "must be provided with features"):
            injector(embeddings, features, None)

    def test_deepstack_injector_normalizes_tensor_sequence_and_none_locations(self):
        injector = MultimodalDeepstackInjector().cuda()
        hidden = torch.zeros(4, 4, device="cuda", dtype=torch.half)
        stacks = [torch.ones((1, 1, 4), device="cuda", dtype=torch.half)]

        tensor_output = injector(
            hidden.clone(),
            stacks,
            torch.tensor([1], device="cuda", dtype=torch.int32),
            0,
        )
        sequence_output = injector(hidden.clone(), stacks, [1], 0)
        torch.testing.assert_close(tensor_output, sequence_output)

        with self.assertRaisesRegex(ValueError, "must be provided with deepstack"):
            injector(hidden.clone(), stacks, None, 0)
        with self.assertRaisesRegex(ValueError, "0 entries but 1 deepstack"):
            injector(hidden.clone(), stacks, [], 0)

    def test_injector_rejects_location_beyond_embeddings(self):
        injector = MultimodalEmbeddingInjector().cuda()
        embeddings = torch.zeros(4, 4, device="cuda", dtype=torch.half)
        feature = torch.ones((1, 4), device="cuda", dtype=torch.half)

        with self.assertRaisesRegex(IndexError, "cannot be placed at loc 4"):
            injector(
                embeddings,
                [feature],
                torch.tensor([4], device="cuda", dtype=torch.int32),
            )

    def test_injector_moves_host_features_to_embeddings_device(self):
        injector = MultimodalEmbeddingInjector().cuda()
        embeddings = torch.zeros(4, 4, device="cuda", dtype=torch.half)
        cpu_feature = torch.tensor(
            [[4.0, 3.0, 2.0, 1.0]], dtype=torch.half, device="cpu"
        )
        cpu_locs = torch.tensor([1], dtype=torch.int32, device="cpu")
        self.assertEqual(cpu_feature.device.type, "cpu")
        self.assertEqual(cpu_locs.device.type, "cpu")

        injector(embeddings, [cpu_feature], cpu_locs)
        torch.testing.assert_close(embeddings[1:2], cpu_feature.cuda())

    def test_injector_rejects_dtype_mismatch(self):
        injector = MultimodalEmbeddingInjector().cuda()
        embeddings = torch.zeros(4, 4, device="cuda", dtype=torch.half)
        feature = torch.ones((1, 4), device="cuda", dtype=torch.float32)

        with self.assertRaisesRegex(TypeError, "dtype mismatch"):
            injector(
                embeddings,
                [feature],
                torch.tensor([0], device="cuda", dtype=torch.int32),
            )


if __name__ == "__main__":
    main()
