"""Characterize native DEVICE_INPUT metadata without loading a full K3 model."""

import unittest

import torch

from rtp_llm.cpp.models.test.pywrapped_model_cache_store_integration_test import (
    CacheStoreForwardModel,
    PyModelOutputs,
    run_scenario,
)


class K3DeviceInputMetadataReproTest(unittest.TestCase):
    def test_no_sp_padding_still_publishes_physical_counts(self):
        class InspectCounts(CacheStoreForwardModel):
            requires_sequence_parallel_padding = False

            def __init__(self):
                super().__init__()
                self.counts = []

            def _forward_one(self, inputs):
                attn = inputs.attention_inputs
                if isinstance(attn, dict):
                    attn = next(iter(attn.values()))
                self.counts.append((attn.physical_token_count, attn.physical_request_count))
                return super()._forward_one(inputs)

        model = InspectCounts()
        run_scenario(model, "multi_tag")
        self.assertTrue(model.counts)
        self.assertEqual(model.counts[0], (8, 3))

    def test_native_cacheless_warmup_publishes_attention_marker(self):
        class InspectWarmup(CacheStoreForwardModel):
            def __init__(self):
                super().__init__()
                self.markers = []

            def _forward_one(self, inputs):
                attention = inputs.attention_inputs
                if isinstance(attention, dict):
                    attention = next(iter(attention.values()))
                self.markers.append(getattr(attention, "is_warmup", False))
                return PyModelOutputs(
                    torch.zeros(
                        (inputs.input_ids.numel(), 1),
                        dtype=torch.float16,
                        device=inputs.input_ids.device,
                    )
                )

        model = InspectWarmup()
        run_scenario(model, "cacheless_warmup")
        self.assertTrue(model.markers)
        self.assertTrue(all(model.markers))

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
                        "is_warmup": getattr(attn, "is_warmup", False),
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
        self.assertFalse(first["is_warmup"])
        print("native DEVICE_INPUT metadata:", first)


if __name__ == "__main__":
    unittest.main()
