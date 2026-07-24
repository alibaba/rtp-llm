from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalDeepstackInjector,
    MultimodalEmbeddingInjector,
)


class MultimodalPrefixCacheInjectorTest(TestCase):
    def test_mixed_cached_and_uncached_features_remain_independent(self):
        embeddings = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        first = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 100
        second = torch.arange(6, dtype=torch.float32).reshape(3, 2) + 200
        locations = torch.tensor([-2, 4], dtype=torch.int32)

        expected_embeddings = embeddings.clone()
        expected_embeddings[:2] = first[2:]
        expected_embeddings[4:7] = second
        actual_embeddings = MultimodalEmbeddingInjector()(
            embeddings.clone(), [first, second], locations
        )
        torch.testing.assert_close(actual_embeddings, expected_embeddings)

        hidden = embeddings.clone()
        first_stack = torch.stack((first, first + 10))
        second_stack = torch.stack((second, second + 10))
        expected_hidden = hidden.clone()
        expected_hidden[:2] += first_stack[1, 2:]
        expected_hidden[4:7] += second_stack[1]
        actual_hidden = MultimodalDeepstackInjector()(
            hidden, [first_stack, second_stack], locations, layer_id=1
        )
        torch.testing.assert_close(actual_hidden, expected_hidden)

    def test_embedding_injector_drops_cached_feature_prefix_on_cpu(self):
        embeddings = torch.arange(18, dtype=torch.float32).reshape(6, 3)
        feature = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 100
        injector = MultimodalEmbeddingInjector()

        for cached_rows in (2, 4, 6):
            with self.subTest(cached_rows=cached_rows):
                expected = embeddings.clone()
                if cached_rows < feature.size(0):
                    tail = feature[cached_rows:]
                    expected[: tail.size(0)] = tail

                actual = injector(
                    embeddings.clone(),
                    [feature],
                    torch.tensor([-cached_rows], dtype=torch.int32),
                )
                torch.testing.assert_close(actual, expected)
                self.assertEqual(actual.device.type, "cpu")

    def test_deepstack_injector_drops_cached_feature_prefix_on_cpu(self):
        hidden = torch.arange(18, dtype=torch.float32).reshape(6, 3)
        stack = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) + 100
        injector = MultimodalDeepstackInjector()
        layer_id = 1

        for cached_rows in (2, 4, 6):
            with self.subTest(cached_rows=cached_rows):
                expected = hidden.clone()
                if cached_rows < stack.size(1):
                    tail = stack[layer_id, cached_rows:]
                    expected[: tail.size(0)] += tail

                actual = injector(
                    hidden.clone(),
                    [stack],
                    torch.tensor([-cached_rows], dtype=torch.int32),
                    layer_id,
                )
                torch.testing.assert_close(actual, expected)
                self.assertEqual(actual.device.type, "cpu")


if __name__ == "__main__":
    main()
