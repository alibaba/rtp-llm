"""CPU file loading and negative preflight; not an M890P execution test."""

import os
import tempfile
import unittest

import torch

from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.new_models.deepseek_v4.ppu_linear import PpuOwnedFP8Linear
from rtp_llm.models_py.weight_mapper import get_all_weights


class PpuLinearOwnerTest(unittest.TestCase):
    def test_raw_shards_and_cpu_preflight(self):
        weights = {
            "weight": (torch.arange(256 * 512).reshape(256, 512) % 251)
            .byte()
            .view(torch.float8_e4m3fn),
            "scale": torch.arange(125, 133, dtype=torch.uint8)
            .reshape(2, 4)
            .view(torch.float8_e8m0fnu),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pt")
            torch.save(weights, path)
            for partition, axis in (("replicated", None), ("column", 0), ("row", 1)):
                for rank in range(2):
                    with self.subTest(partition=partition, rank=rank):
                        layer = PpuOwnedFP8Linear(
                            512,
                            256,
                            platform_provider=object(),
                            tp_size=2,
                            tp_rank=rank,
                            partition=partition,
                        )
                        layer.load_weights(
                            get_all_weights(
                                [path], read_spec_resolver=layer.checkpoint_read_spec
                            )
                        )
                        NewModelLoader._validate_loaded_weights(layer)
                        for name, full in weights.items():
                            expected = (
                                full if axis is None else full.chunk(2, axis)[rank]
                            )
                            torch.testing.assert_close(
                                getattr(layer, name).view(torch.uint8),
                                expected.view(torch.uint8),
                                atol=0,
                                rtol=0,
                            )
                        with self.assertRaisesRegex(RuntimeError, "M890P"):
                            layer.bind_runtime()


if __name__ == "__main__":
    unittest.main()
