"""A CPU conversion policy must retain online NVFP4 weight loading."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.kimi_k3.nvfp4_tensor import OnlineNvfp4Projection
from rtp_llm.utils.model_weight import W


class KimiK31CpuWeightConversionTest(unittest.TestCase):
    def test_cpu_weight_conversion_matches_cuda_conversion(self):
        self.assertIn(torch.cuda.get_device_capability(), ((10, 0), (10, 3)))
        torch.manual_seed(311)
        projection = OnlineNvfp4Projection(
            AtomicWeight(W.mla_q_b_w, []), module="dsa.query"
        )
        config = SimpleNamespace(force_cpu_load_weights=True)
        for rows in (32, 1152):
            with self.subTest(rows=rows):
                # More than one staging tile must retain the whole weight's
                # global scale, including its swizzled scale-tile boundaries.
                matrix = torch.randn((64, rows), dtype=torch.bfloat16)
                reference = projection._postprocess(
                    {projection.name: matrix.cuda()}, "cuda", config
                )
                for destination in ("cpu", "cuda"):
                    converted = projection._postprocess(
                        {projection.name: matrix}, destination, config
                    )
                    self.assertEqual(set(converted), set(reference))
                    for name, actual in converted.items():
                        with self.subTest(destination=destination, name=name):
                            self.assertEqual(actual.device.type, destination)
                            torch.testing.assert_close(
                                actual.cpu(), reference[name].cpu(), rtol=0, atol=0
                            )


if __name__ == "__main__":
    unittest.main()
