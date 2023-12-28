import os
import unittest
import torch
from transformers.models.t5.modeling_t5 import T5LayerNorm

class TestT5LayerNorm(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
    def test_logn_attention(self):
        torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/maga_transformer/tests/libtest_ops.so")

        eps = 1e-6
        for batch_size in range(32, 4096, 512):
            for hidden_units in range(1024, 4096, 1024):

                self.gamm = torch.rand(hidden_units).cuda()
                self.T5LayerNormOp = torch.classes.unittest.T5LayerNormOp(eps)
                self.T5LayerNorm = T5LayerNorm(hidden_units, eps)
                self.T5LayerNorm.weight = torch.nn.Parameter(self.gamm)

                hidden_states = torch.rand(batch_size, hidden_units).cuda()
                result = self.T5LayerNorm(hidden_states)
                test = self.T5LayerNormOp.forward(hidden_states, self.gamm)
                torch.testing.assert_close(test, result)


if __name__ == '__main__':
    unittest.main()