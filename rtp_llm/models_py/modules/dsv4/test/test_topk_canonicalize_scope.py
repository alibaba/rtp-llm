"""Keep the integer fusion restricted to the intended prefill scope."""

import os
import unittest
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.dsv4 import forward_metadata
from rtp_llm.models_py.modules.dsv4.fp8 import indexer
from rtp_llm.models_py.triton_kernels.common import canonicalize_topk


def reference(out):
    sentinel = torch.iinfo(out.dtype).max
    values = torch.sort(torch.where(out >= 0, out, sentinel), dim=-1).values
    return torch.where(values == sentinel, -1, values)


class TestCanonicalizeScope(unittest.TestCase):
    def test_scope_and_large_shape_fallback(self):
        with patch.dict(
            os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": "1"}
        ), patch.object(
            indexer, "_FUSED_PREFILL_TOPK_CANONICALIZE", True
        ), patch.object(
            canonicalize_topk,
            "canonicalize_topk_",
            wraps=canonicalize_topk.canonicalize_topk_,
        ) as fused:
            small = torch.randint(
                -3, 90000, (81, 2048), device="cuda", dtype=torch.int32
            )
            expected = reference(small)
            original = small.clone()
            indexer.canonicalize_topk_output(small)
            torch.testing.assert_close(small, expected, rtol=0, atol=0)
            fused.assert_not_called()
            token = forward_metadata._FORWARD_METADATA.set({})
            try:
                indexer.canonicalize_topk_output(original)
                torch.testing.assert_close(original, expected, rtol=0, atol=0)
                fused.assert_called_once()
                for width in (512, 1024, 2048):
                    for rows in (4093, 32768):
                        value = torch.randint(
                            -3, 90000, (rows, width), device="cuda", dtype=torch.int32
                        )
                        expected = reference(value)
                        indexer.canonicalize_topk_output(value)
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)
                self.assertEqual(fused.call_count, 7)
                for rows, width, dtype in (
                    (32769, 512, torch.int32),
                    (32769, 1024, torch.int32),
                    (32769, 2048, torch.int32),
                    (1025, 1537, torch.int32),
                    (13, 4096, torch.int32),
                    (13, 2048, torch.int64),
                ):
                    value = torch.randint(
                        -3, 90000, (rows, width), device="cuda", dtype=dtype
                    )
                    expected = reference(value)
                    indexer.canonicalize_topk_output(value)
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                    self.assertEqual(fused.call_count, 7)
                with patch.dict(os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": "0"}):
                    unchanged = torch.randint(
                        -3, 90000, (81, 2048), device="cuda", dtype=torch.int32
                    )
                    expected = unchanged.clone()
                    indexer.canonicalize_topk_output(unchanged)
                    torch.testing.assert_close(unchanged, expected, rtol=0, atol=0)
                    self.assertEqual(fused.call_count, 7)
            finally:
                forward_metadata._FORWARD_METADATA.reset(token)


if __name__ == "__main__":
    unittest.main()
