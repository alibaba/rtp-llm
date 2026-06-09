from unittest import SkipTest, TestCase, main

import torch
from torch.nn import functional as F

from rtp_llm.models_py.modules.base.common.embedding import EmbeddingTorch


class EmbeddingTorchTest(TestCase):
    HIDDEN_DIM = 64
    VOCAB_SIZE = 100

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.weight = torch.randn(self.VOCAB_SIZE, self.HIDDEN_DIM)

    def test_normal_lookup(self):
        emb = EmbeddingTorch(self.weight)
        input_ids = torch.tensor([0, 5, 10])
        result = emb(input_ids)
        expected = F.embedding(input_ids, self.weight)
        self.assertTrue(torch.equal(result, expected))

    def test_preserves_dtype(self):
        weight_bf16 = self.weight.to(torch.bfloat16)
        emb = EmbeddingTorch(weight_bf16)
        result = emb(torch.tensor([1, 2]))
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_output_shape(self):
        emb = EmbeddingTorch(self.weight)
        input_ids = torch.tensor([1, 2, 3, 4])
        result = emb(input_ids)
        self.assertEqual(list(result.shape), [4, self.HIDDEN_DIM])

    def test_weight_accessible(self):
        emb = EmbeddingTorch(self.weight)
        self.assertTrue(torch.equal(emb.weight, self.weight))


class EmbeddingTorchCudaTest(TestCase):
    HIDDEN_DIM = 64
    VOCAB_SIZE = 100

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.manual_seed(42)
        self.weight = torch.randn(self.VOCAB_SIZE, self.HIDDEN_DIM, device="cuda")

    def test_cuda_lookup(self):
        emb = EmbeddingTorch(self.weight)
        input_ids = torch.tensor([0, 5, 10], device="cuda")
        result = emb(input_ids)
        expected = F.embedding(input_ids, self.weight)
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.device.type, "cuda")


if __name__ == "__main__":
    main()
