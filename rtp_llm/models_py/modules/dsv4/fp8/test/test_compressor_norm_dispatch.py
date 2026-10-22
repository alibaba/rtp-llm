"""Preserve the CUDA compressor math while isolating PPU compatibility."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.device.device_type import DeviceType
from rtp_llm.models_py.modules.dsv4.fp8 import _compressor_vllm_triton as compressor


class CompressorNormDispatchTest(unittest.TestCase):
    def test_reciprocal_sqrt_is_ppu_only(self):
        for head_dim, kernel_name in (
            (128, "_fused_kv_compress_norm_rope_insert_indexer_attn"),
            (512, "_fused_kv_compress_norm_rope_insert_sparse_attn"),
        ):
            for device_type in (DeviceType.Cuda, DeviceType.Ppu, DeviceType.ROCm):
                with self.subTest(head_dim=head_dim, device_type=device_type):
                    positions = torch.tensor([3], dtype=torch.int64)
                    with (
                        patch.object(compressor, "get_device_type", return_value=device_type),
                        patch.object(compressor, kernel_name) as kernel,
                        patch.object(compressor, "validate_slot_mapping"),
                        patch.object(compressor, "_validate_fused_state_block_table"),
                    ):
                        compressor.run_fused_compress_kv_write(
                            state_cache=torch.zeros(1, 16, 4 * head_dim),
                            token_to_req_indices=torch.zeros(1, dtype=torch.int32),
                            positions=positions,
                            slot_mapping=positions,
                            block_table=torch.zeros(1, 1, dtype=torch.int32),
                            rms_norm_weight=torch.ones(head_dim, dtype=torch.bfloat16),
                            rms_norm_eps=1e-6,
                            cos_sin_cache=torch.zeros(4, 64),
                            kv_cache=torch.zeros(1, 32, 584 if head_dim == 512 else 132, dtype=torch.uint8),
                            kv_slot_mapping=torch.zeros(1, dtype=torch.int64),
                            kv_raw=torch.zeros(4, 2 * head_dim),
                            score_raw=torch.zeros(4, 2 * head_dim),
                            ape=torch.zeros(4, 2 * head_dim),
                            seq_start=0,
                            head_dim=head_dim,
                            rope_head_dim=64,
                            compress_ratio=4,
                            overlap=True,
                            state_tokens_per_block=256,
                        )
                    kernel.__getitem__.assert_called_once_with((1,))
                    launch = kernel.__getitem__.return_value
                    launch.assert_called_once()
                    self.assertEqual(
                        launch.call_args.kwargs["PPU_RMS_COMPAT"],
                        device_type == DeviceType.Ppu,
                    )


if __name__ == "__main__":
    unittest.main()
