import math
import os
import unittest

import torch


def logn_attention_test(x: torch.Tensor, seq_length: int):

    logn_list = [
        math.log(i, seq_length) if i > seq_length else 1 for i in range(1, 32768)
    ]
    logn_tensor = torch.Tensor(logn_list)[None, :, None, None]
    logn_tensor = logn_tensor[:, 0 : x.size(1), :, :].cuda()
    x = x * logn_tensor.expand_as(x)
    return x


class TestRAPE(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_logn_attention(self):
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/libtest_ops.so"
        )

        for logn_seq_len in range(1024, 2048, 512):
            self.LognAttentionOp = torch.classes.unittest.LognAttentionOp(logn_seq_len)
            for len in range(1024, 8096, 1024):
                data = torch.rand([32, len, 2, 128]).cuda()
                result = self.LognAttentionOp.forward(data)
                test = logn_attention_test(data, logn_seq_len)
                torch.testing.assert_close(test, result, rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
