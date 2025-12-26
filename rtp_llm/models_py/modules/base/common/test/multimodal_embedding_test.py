import itertools
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype

from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalDeepstackInjector,
    MultimodalEmbeddingInjector,
)


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


if __name__ == "__main__":
    main()
