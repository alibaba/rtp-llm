"""Raw Meta archives must finish GQA/rotary conversion before online FP8."""

import tempfile
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.llama_checkpoint_formats_test import load, write_meta_source


class LlamaMetaOnlineFp8Test(unittest.TestCase):
    def test_real_meta_fp8_after_tp_rotary(self):
        self.assertEqual(torch.cuda.get_device_capability(), (9, 0))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_meta_source(root)
            q = source["layers.0.attention.wq.weight"][[4, 6, 5, 7]]
            k = source["layers.0.attention.wk.weight"][[0, 2, 1, 3]]
            v = source["layers.0.attention.wv.weight"]
            logical = torch.cat((q, k, v)).bfloat16().float()
            expected_scale = logical.abs().amax() / 448
            codes = (logical / expected_scale).to(torch.float8_e4m3fn)
            for force_cpu in (False, True):
                with self.subTest(force_cpu=force_cpu):
                    model = load(
                        root,
                        tp=2,
                        rank=1,
                        dtype=torch.bfloat16,
                        device="cuda:0",
                        online_quant="fp8_online",
                        force_cpu=force_cpu,
                    )
                    leaf = model.layers[0].self_attn.qkv_proj
                    torch.testing.assert_close(
                        leaf.weight_scale.cpu().reshape(()), expected_scale
                    )
                    torch.testing.assert_close(
                        leaf.weight.cpu().view(torch.uint8), codes.view(torch.uint8)
                    )


if __name__ == "__main__":
    unittest.main()
