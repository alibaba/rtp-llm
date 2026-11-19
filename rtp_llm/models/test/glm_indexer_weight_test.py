import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models.deepseek_v2 import PackedIndexerWeight, pack_indexer_weight
from rtp_llm.utils.model_weight import CkptWeightInfo


class GlmIndexerWeightTest(unittest.TestCase):
    def test_checkpoint_scale_precision_and_block_order(self):
        generator = torch.Generator().manual_seed(37)
        raw = torch.randn(128, 384, generator=generator).to(torch.float8_e4m3fn)
        gate = torch.randn(32, 384, generator=generator).bfloat16()
        for scale in (
            torch.tensor([[0.01567, 0.02453, 0.03591]]),
            torch.tensor([[0.015625, 0.03125, 0.0625]]).to(torch.float8_e8m0fnu),
        ):
            with self.subTest(dtype=scale.dtype):
                tensors = {"wk": raw, "scale": scale, "gate": gate}
                source = SimpleNamespace(
                    load_tensor=lambda name, dtype: [tensors[name].to(dtype)]
                )
                packed = PackedIndexerWeight(
                    "packed",
                    [CkptWeightInfo(n) for n in tensors],
                    pack_indexer_weight,
                    data_type=torch.float32,
                )
                result = packed._load_raw_tensor(
                    source, 0, "cpu", SimpleNamespace(compute_dtype=torch.bfloat16)
                )["packed"]
                expected = torch.cat(
                    [
                        block.double() * s
                        for block, s in zip(raw.split(128, dim=1), scale.double()[0])
                    ],
                    dim=1,
                ).bfloat16()
                self.assertEqual(result.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(result[:128], expected))
                self.assertTrue(torch.equal(result[128:], gate))


if __name__ == "__main__":
    unittest.main()
