"""Characterize native DEVICE_INPUT metadata without loading a full K3 model."""

import unittest

from rtp_llm.cpp.models.test.pywrapped_model_cache_store_integration_test import (
    CacheStoreForwardModel,
    run_scenario,
)


class K3DeviceInputMetadataReproTest(unittest.TestCase):
    def test_native_device_lengths_omit_k3_required_host_mirrors(self):
        class InspectHostMirrors(CacheStoreForwardModel):
            def __init__(self):
                super().__init__()
                self.mirrors = []

            def _forward_one(self, inputs):
                attn = inputs.attention_inputs
                if isinstance(attn, dict):
                    attn = next(iter(attn.values()))
                self.mirrors.append(
                    {
                        "host_cu": attn.cu_seqlens.tolist(),
                        "device_cu": attn.cu_seqlens_device.cpu().tolist(),
                        "prefix_device": attn.prefix_lengths.device.type,
                        "table_device": attn.kv_cache_kernel_block_id.device.type,
                    }
                )
                return super()._forward_one(inputs)

        model = InspectHostMirrors()
        run_scenario(model, "cuda_lengths_device_prefix")
        first = model.mirrors[0]
        self.assertEqual(first["host_cu"], [])
        self.assertEqual(first["device_cu"], [0, 1, 4, 8])
        self.assertEqual(first["prefix_device"], "cuda")
        self.assertEqual(first["table_device"], "cuda")
        print("native DEVICE_INPUT metadata:", first)


if __name__ == "__main__":
    unittest.main()
