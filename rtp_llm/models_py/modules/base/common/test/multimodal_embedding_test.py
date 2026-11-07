import itertools
from unittest import SkipTest, TestCase, main
from unittest.mock import patch

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

    def test_rejects_negative_multimodal_location(self):
        embeddings = torch.zeros(4, 2, dtype=torch.half)
        feature = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.half)
        locations = torch.tensor([-1], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "loc must be non-negative"):
            MultimodalEmbeddingInjector()(embeddings.clone(), [feature], locations)

    def test_rejects_negative_deepstack_location(self):
        embeddings = torch.zeros(4, 2, dtype=torch.half)
        locations = torch.tensor([-1], dtype=torch.int32)
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

    def test_noncontiguous_embeddings_use_sequential_copy(self):
        for enabled, dtype in itertools.product([False, True], self.DTYPES):
            with self.subTest(enable_fused_copy=enabled, dtype=dtype):
                base = torch.zeros((4, 4), device="cuda", dtype=dtype)
                embeddings = base[:, ::2]
                self.assertFalse(embeddings.is_contiguous())
                feature = torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=dtype
                )
                expected_base = base.clone()
                expected_base[1:3, ::2] = feature

                with patch.object(
                    rtp_llm_ops,
                    "fused_multimodal_copy_",
                    side_effect=AssertionError("noncontiguous dst must use fallback"),
                ) as fused_copy:
                    output = MultimodalEmbeddingInjector(enable_fused_copy=enabled)(
                        embeddings,
                        [feature],
                        torch.tensor([1], device="cpu", dtype=torch.int32),
                    )
                fused_copy.assert_not_called()
                self.assertIs(output, embeddings)
                torch.testing.assert_close(base, expected_base, rtol=0, atol=0)

    def test_fused_copy_contiguous_destination_and_chunk_boundaries(self):
        # The binding flushes a batch at MAX_FUSED_D2D_COPIES (currently 64).
        for num_features, dtype in itertools.product([1, 64, 65, 129], self.DTYPES):
            with self.subTest(num_features=num_features, dtype=dtype):
                embeddings = torch.zeros(
                    (num_features * 2 + 1, 8), device="cuda", dtype=dtype
                )
                # Strided sources are made contiguous by the injector; they do
                # not require fallback when the destination is contiguous.
                features = [
                    torch.full((1, 16), i + 1, device="cuda", dtype=dtype)[:, ::2]
                    for i in range(num_features)
                ]
                self.assertFalse(features[0].is_contiguous())
                locs = list(range(1, num_features * 2, 2))
                expected = embeddings.clone()
                for feature, loc in zip(features, locs):
                    expected[loc : loc + 1] = feature

                with patch.object(
                    rtp_llm_ops,
                    "fused_multimodal_copy_",
                    wraps=rtp_llm_ops.fused_multimodal_copy_,
                ) as fused_copy:
                    output = MultimodalEmbeddingInjector(enable_fused_copy=True)(
                        embeddings,
                        features,
                        torch.tensor(locs, device="cpu", dtype=torch.int32),
                    )
                fused_copy.assert_called_once()
                self.assertIs(output, embeddings)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_fused_copy_overlapping_ranges_use_sequential_copy(self):
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                embeddings = torch.zeros((6, 8), device="cuda", dtype=dtype)
                features = [
                    torch.full((3, 8), 1, device="cuda", dtype=dtype),
                    torch.full((2, 8), 2, device="cuda", dtype=dtype),
                ]
                # Reversed offsets ensure fallback preserves input order,
                # rather than sorting copies and changing the last writer.
                locs = [2, 1]
                expected = embeddings.clone()
                for feature, loc in zip(features, locs):
                    expected[loc : loc + feature.size(0)] = feature
                with patch.object(
                    rtp_llm_ops,
                    "fused_multimodal_copy_",
                    side_effect=AssertionError("overlapping ranges must use fallback"),
                ) as fused_copy:
                    output = MultimodalEmbeddingInjector(enable_fused_copy=True)(
                        embeddings,
                        features,
                        torch.tensor(locs, device="cpu", dtype=torch.int32),
                    )
                fused_copy.assert_not_called()
                self.assertIs(output, embeddings)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == "__main__":
    main()
