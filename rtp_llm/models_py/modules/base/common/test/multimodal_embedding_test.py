import itertools
from types import SimpleNamespace
from unittest import SkipTest, TestCase, main
from unittest.mock import Mock

import torch
from torch import dtype as _dtype

from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalDeepstackInjector,
    MultimodalEmbedding,
    MultimodalEmbeddingInjector,
)


class DraftMultimodalEmbeddingTest(TestCase):
    def make_inputs(self):
        return SimpleNamespace(
            input_ids=torch.tensor([2, -100, 3, 100000, 1, 4], device="cpu"),
            embedding_inputs=SimpleNamespace(
                text_tokens_mask=torch.tensor([1, 0, 0, 0, 1, 1], device="cpu"),
                combo_tokens_type_ids=None,
            ),
            multimodal_inputs=SimpleNamespace(
                multimodal_features=[
                    torch.tensor([[10.0, 11], [12, 13]], device="cpu"),
                    torch.tensor([[14.0, 15]], device="cpu"),
                ],
                mm_features_locs=torch.tensor([1, 3], device="cpu"),
            ),
        )

    def test_no_features_preserves_original_call(self):
        inputs = self.make_inputs()
        for features in ([], None):
            inputs.multimodal_inputs.multimodal_features = features
            embedding = Mock(return_value=object())
            self.assertIs(
                MultimodalEmbedding(embedding)(inputs.input_ids, model_inputs=inputs),
                embedding.return_value,
            )
            embedding.assert_called_once_with(inputs.input_ids)

    def test_masked_and_custom_embedding_replace_all_hash_rows(self):
        for supports_mask in (True, False):
            with self.subTest(supports_mask=supports_mask):
                inputs = self.make_inputs()
                original_ids = inputs.input_ids.clone()
                original_mask = inputs.embedding_inputs.text_tokens_mask.clone()
                original_features = [
                    f.clone() for f in inputs.multimodal_inputs.multimodal_features
                ]
                table = torch.arange(10, dtype=torch.float32, device="cpu").reshape(
                    5, 2
                )

                def embed(
                    ids, position_ids=None, token_types=None, text_tokens_mask=None
                ):
                    if supports_mask:
                        self.assertIsNotNone(text_tokens_mask)
                        ids = ids.masked_fill(text_tokens_mask == 0, 0)
                    return table[ids]

                embedding = MultimodalEmbedding(
                    Mock(side_effect=embed), supports_mask=supports_mask
                )
                output = embedding(inputs.input_ids, model_inputs=inputs)
                expected = torch.tensor(
                    [[4.0, 5], [10, 11], [12, 13], [14, 15], [2, 3], [8, 9]],
                    device="cpu",
                )
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                self.assertTrue(torch.equal(inputs.input_ids, original_ids))
                self.assertTrue(
                    torch.equal(inputs.embedding_inputs.text_tokens_mask, original_mask)
                )
                for original, feature in zip(
                    original_features, inputs.multimodal_inputs.multimodal_features
                ):
                    self.assertTrue(torch.equal(original, feature))

    def test_chunk_boundaries_match_full_embedding(self):
        inputs = self.make_inputs()
        table = torch.arange(10, dtype=torch.float32, device="cpu").reshape(5, 2)

        def embed(ids):
            return table[ids]

        embedding = MultimodalEmbedding(Mock(side_effect=embed), supports_mask=False)
        expected = embedding(inputs.input_ids, model_inputs=inputs)
        for chunk_size in (1, 2, 3):
            chunks = [
                embedding(
                    inputs.input_ids[start : start + chunk_size],
                    model_inputs=inputs,
                    token_offset=start,
                )
                for start in range(0, inputs.input_ids.numel(), chunk_size)
            ]
            torch.testing.assert_close(torch.cat(chunks), expected, rtol=0, atol=0)

    def test_custom_embedding_zeros_cp_padding(self):
        inputs = self.make_inputs()
        inputs.input_ids[-1] = 0
        inputs.embedding_inputs.text_tokens_mask[-1] = 0
        embedding = MultimodalEmbedding(
            Mock(side_effect=lambda ids: torch.ones((ids.numel(), 2), device="cpu")),
            supports_mask=False,
        )
        output = embedding(inputs.input_ids, model_inputs=inputs)
        self.assertTrue(torch.equal(output[-1], torch.zeros(2, device="cpu")))

    def test_missing_or_misaligned_mask_fails_before_lookup(self):
        for mask in (None, torch.ones(2, device="cpu")):
            inputs = self.make_inputs()
            inputs.embedding_inputs.text_tokens_mask = mask
            embed = Mock()
            with self.assertRaisesRegex(ValueError, "aligned text_tokens_mask"):
                MultimodalEmbedding(embed)(inputs.input_ids, model_inputs=inputs)
            embed.assert_not_called()


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


if __name__ == "__main__":
    main()
