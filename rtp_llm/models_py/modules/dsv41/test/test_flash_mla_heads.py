"""Native FlashMLA tiling must not restrict model TP head partitions."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location(
    "v41_flash_mla", Path(__file__).parents[1] / "flash_mla.py"
)
FLASH = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FLASH)


class FlashMLAHeadTest(unittest.TestCase):
    def test_prefill_padding_keeps_real_heads_and_sink_values(self):
        for heads in (1, 8, 16, 32, 64, 128):
            q = torch.randn(3, heads, 512, dtype=torch.bfloat16)
            sink = torch.randn(heads)

            def native(padded, kv, indices, scale, **kwargs):
                self.assertIn(padded.shape[1], (64, 128))
                torch.testing.assert_close(padded[:, :heads], q)
                torch.testing.assert_close(kwargs["attn_sink"][:heads], sink)
                self.assertTrue(torch.isposinf(kwargs["attn_sink"][heads:]).all())
                return padded, padded.sum(-1), padded.mean(-1)

            with patch.dict(
                sys.modules, {"flash_mla": SimpleNamespace(flash_mla_sparse_fwd=native)}
            ):
                output, maximum, lse = FLASH.flash_mla_sparse_fwd(
                    q, None, None, 1.0, attn_sink=sink
                )
            torch.testing.assert_close(output, q)
            self.assertEqual(maximum.shape, (3, heads))
            self.assertEqual(lse.shape, (3, heads))

    def test_decode_slices_lse_on_head_axis(self):
        q = torch.randn(2, 6, 16, 512, dtype=torch.bfloat16)

        def native(*, q, attn_sink, **kwargs):
            return q, q.sum(-1).transpose(1, 2)

        with patch.dict(
            sys.modules, {"flash_mla": SimpleNamespace(flash_mla_with_kvcache=native)}
        ):
            output, lse = FLASH.flash_mla_with_kvcache(q=q)
        torch.testing.assert_close(output, q)
        torch.testing.assert_close(lse, q.sum(-1).transpose(1, 2))


if __name__ == "__main__":
    unittest.main()
