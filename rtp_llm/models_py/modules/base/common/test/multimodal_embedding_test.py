from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalEmbeddingInjector,
)


class MultimodalEmbeddingInjectorTest(TestCase):
    def test_replaces_placeholder_rows_at_cpp_locations(self):
        embeddings = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        original = embeddings.clone()
        features = [
            torch.full((2, 4), 100.0),
            torch.full((1, 4), 200.0),
        ]

        result = MultimodalEmbeddingInjector()(
            embeddings, features, torch.tensor([1, 4], dtype=torch.int32)
        )

        self.assertIs(result, embeddings)
        torch.testing.assert_close(result[0], original[0])
        torch.testing.assert_close(result[1:3], features[0])
        torch.testing.assert_close(result[3], original[3])
        torch.testing.assert_close(result[4:5], features[1])
        torch.testing.assert_close(result[5], original[5])

    def test_rejects_feature_location_mismatch(self):
        with self.assertRaisesRegex(ValueError, "feature/location mismatch"):
            MultimodalEmbeddingInjector()(
                torch.zeros(3, 4),
                [torch.zeros(1, 4)],
                torch.tensor([], dtype=torch.int32),
            )


if __name__ == "__main__":
    main()
