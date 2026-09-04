import os
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.pad_query_heads import (
    maybe_pad_query_heads,
)


class PadQueryHeadsTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and Triton")
    def test_fused_copy_and_zero_padding(self):
        torch.manual_seed(31)
        query = torch.randn(
            37, 64, 576, dtype=torch.bfloat16, device=torch.device("cuda")
        )

        with torch.no_grad():
            padded = maybe_pad_query_heads(query, 128)

        self.assertIsNotNone(padded)
        assert padded is not None
        self.assertEqual(tuple(padded.shape), (37, 128, 576))
        self.assertTrue(padded.is_contiguous())
        torch.testing.assert_close(padded[:, :64], query, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(padded[:, 64:]).item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and Triton")
    def test_env_gate_preserves_fallback(self):
        query = torch.randn(
            2, 64, 576, dtype=torch.bfloat16, device=torch.device("cuda")
        )
        with mock.patch.dict(
            os.environ, {"RTP_LLM_FUSE_SPARSE_QUERY_PADDING": "0"}
        ):
            self.assertIsNone(maybe_pad_query_heads(query, 128))


if __name__ == "__main__":
    unittest.main()
