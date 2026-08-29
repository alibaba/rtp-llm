from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalEmbeddingInjector,
    prepare_mtp_multimodal_inputs,
)


class MtpMultimodalEmbeddingTest(TestCase):
    IMAGE_A = torch.tensor([[1000.0, 1001.0], [2000.0, 2001.0]])
    IMAGE_B = torch.tensor([[3000.0, 3001.0], [4000.0, 4001.0]])
    EMBEDDING = torch.arange(64 * 2, dtype=torch.float32).reshape(64, 2)

    def _embed(self, input_ids, features, locations, cu_seqlens):
        ids, shifted_features, shifted_locs = prepare_mtp_multimodal_inputs(
            torch.tensor(input_ids, dtype=torch.int32),
            features,
            torch.tensor(locations, dtype=torch.int32),
            torch.tensor(cu_seqlens, dtype=torch.int32),
        )
        embeddings = self.EMBEDDING.index_select(0, ids.to(torch.long))
        return MultimodalEmbeddingInjector()(
            embeddings, shifted_features, shifted_locs
        )

    def test_shifts_features_and_masks_hash_ids(self):
        output = self._embed(
            [-101, -102, 11, 12], [self.IMAGE_A], [1], [0, 4]
        )
        self.assertTrue(torch.equal(output[0:2], self.IMAGE_A))
        self.assertTrue(torch.equal(output[2], self.EMBEDDING[11]))
        self.assertTrue(torch.equal(output[3], self.EMBEDDING[12]))

    def test_drops_feature_row_shifted_before_request(self):
        output = self._embed([-102, 11, 12, 13], [self.IMAGE_A], [0], [0, 4])
        self.assertTrue(torch.equal(output[0], self.IMAGE_A[1]))
        self.assertTrue(torch.equal(output[1], self.EMBEDDING[11]))
        self.assertTrue(torch.equal(output[2], self.EMBEDDING[12]))
        self.assertTrue(torch.equal(output[3], self.EMBEDDING[13]))

    def test_shift_stays_within_each_packed_request(self):
        output = self._embed(
            [-101, -102, 20, -202, 22, 21],
            [self.IMAGE_A, self.IMAGE_B],
            [1, 3],
            [0, 3, 6],
        )
        self.assertTrue(torch.equal(output[0:2], self.IMAGE_A))
        self.assertTrue(torch.equal(output[2], self.EMBEDDING[20]))
        self.assertTrue(torch.equal(output[3], self.IMAGE_B[1]))
        self.assertTrue(torch.equal(output[4], self.EMBEDDING[22]))
        self.assertTrue(torch.equal(output[5], self.EMBEDDING[21]))


if __name__ == "__main__":
    main()
