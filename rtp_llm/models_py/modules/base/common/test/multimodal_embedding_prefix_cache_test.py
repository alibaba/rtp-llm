from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalDeepstackInjector,
    MultimodalEmbeddingInjector,
)


class MultimodalPrefixCacheInjectorTest(TestCase):
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
