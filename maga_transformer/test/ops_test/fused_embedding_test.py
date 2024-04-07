import torch
from unittest import TestCase, main
from typing import Any
from maga_transformer.ops.comm.embedding_op import EmbeddingOp

class EmbeddingTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.embedding1 = torch.nn.Embedding(10, 1024, dtype=torch.half, device='cuda:0')
        self.embedding2 = torch.nn.Embedding(10, 1024, dtype=torch.half, device='cuda:0')
        self.embedding3 = torch.nn.Embedding(10, 1024, dtype=torch.half, device='cuda:0')
        self.fused_embedding = EmbeddingOp(self.embedding1.weight, self.embedding2.weight, self.embedding3.weight, False)

    def test_embedding_lookup(self):
        tt1 = torch.tensor([4], dtype=torch.int).cuda()
        tt2 = torch.tensor([5], dtype=torch.int).cuda()
        tt3 = torch.tensor([6], dtype=torch.int).cuda()

        out1 = self.embedding1(tt1) + self.embedding2(tt2) + self.embedding3(tt3)
        out2 = self.fused_embedding.forward(tt1, tt2, tt3)
        self.assertTrue(all(torch.isclose(out1, out2).reshape(-1)))

    def test_input_check(self):
        tt1 = torch.tensor([4], dtype=torch.int).cuda()
        tt2 = torch.tensor([5], dtype=torch.int).cuda()
        with self.assertRaisesRegex(RuntimeError, "token_type_ids should not be nullptr"):
            out2 = self.fused_embedding.forward(tt1, tt2, None)

main()