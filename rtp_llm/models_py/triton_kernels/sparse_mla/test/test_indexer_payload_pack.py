import os
import unittest
from unittest.mock import patch

import torch
from rtp_llm.models_py.triton_kernels.sparse_mla.fused_indexer_cp_assemble import (
    _pack_indexer_k_payload,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class IndexerPayloadPackTest(unittest.TestCase):
    def test_bits_and_layout(self):
        for n in (0, 1, 17, 25024):
            for width, scales in ((128, 4), (65, 3)):
                for strided in (False, True):
                    k = torch.randint(
                        0, 256, (n, width * 2), device="cuda", dtype=torch.uint8
                    )
                    s = torch.randint(
                        0, 256, (n, scales * 2), device="cuda", dtype=torch.uint8
                    )
                    k = k[:, ::2] if strided else k[:, :width].contiguous()
                    s = s[:, ::2] if strided else s[:, :scales].contiguous()
                    with patch.dict(
                        os.environ, {"RTP_INDEXER_FUSED_PAYLOAD_PACK": "0"}
                    ):
                        ref, qw, sw = _pack_indexer_k_payload(k, s)
                    with patch.dict(
                        os.environ, {"RTP_INDEXER_FUSED_PAYLOAD_PACK": "1"}
                    ):
                        got, gqw, gsw = _pack_indexer_k_payload(k, s)
                        cached, _, _ = _pack_indexer_k_payload(k, s)
                    self.assertEqual((qw, sw), (gqw, gsw))
                    self.assertTrue(torch.equal(ref, got))
                    self.assertTrue(torch.equal(ref, cached))


if __name__ == "__main__":
    unittest.main()
