import random
import sys
import unittest

import torch

# Add RTP-LLM to path if needed
sys.path.append("/data2/baowending.bwd/new/RTP-LLM/github-opensource/")

import numpy as np

from rtp_llm.models_py.bindings.cuda.test.concat_and_cache_mla.util import (
    create_mla_cache,
)
from rtp_llm.ops import compute_ops


class TestConcatAndCacheDSMLA(unittest.TestCase):
    """Test suite for concat_and_cache_ds_mla operation (DeepSeek MLA)"""

    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        cls.device = "cuda:0"
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.cuda.set_device(cls.device)

    def _run_concat_and_cache_ds_mla(
        self,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        num_tokens: int,
        block_size: int,
        num_blocks: int,
        dtype: torch.dtype,
        seed: int,
    ) -> None:
        """
        Helper method to run concat_and_cache_ds_mla test

        This test verifies the DeepSeek MLA variant which:
        1. Uses tile-wise dynamic quantization for NoPE part (4 tiles of 128 elements)
        2. Keeps RoPE part in original precision
        3. Stores scale values for each tile
        """
        if dtype.itemsize != 2:
            self.skipTest("ds_mla only supports 16-bit input (fp16/bf16)")
        import torch

        kv_cache_dtype = "fp8_ds_mla"
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.set_default_device(self.device)
        torch.cuda.set_device(self.device)

        # Create random slot mapping
        total_slots = num_blocks * block_size
        slot_mapping_lst = random.sample(range(total_slots), num_tokens)
        slot_mapping = torch.tensor(
            slot_mapping_lst, dtype=torch.long, device=self.device
        )

        # Create input tensors
        kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=self.device)
        k_pe = torch.randn(
            num_tokens, qk_rope_head_dim, dtype=dtype, device=self.device
        )

        # For fp8_ds_mla: entry_size = kv_lora_rank + (4 * 4) + (2 * qk_rope_head_dim)
        # kv_lora_rank bytes for NoPE (quantized to FP8, 1 byte per element),
        # 16 bytes for 4 scales, 2*qk_rope_head_dim bytes for RoPE
        entry_size = kv_lora_rank + (4 * 4) + (2 * qk_rope_head_dim)

        scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        kv_cache = create_mla_cache(
            num_blocks,
            block_size,
            entry_size,
            dtype=torch.uint8,
            kv_cache_dtype=kv_cache_dtype,
            device=self.device,
        )

        # Call the concat_and_cache_mla kernel with fp8_ds_mla
        compute_ops.concat_and_cache_mla(
            kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
        )

        # For DeepSeek MLA, just verify the kernel runs without error
        # Full numerical verification requires exact matching of the FP8 quantization logic
        print(
            f"✓ DeepSeek MLA kernel executed successfully: "
            f"kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}, "
            f"num_tokens={num_tokens}, dtype={dtype}"
        )

        # Correctness verification based on kernel implementation
        # Reference: rtp_llm/cpp/kernels/indexer_k_quant_kernel.cu::concat_and_cache_ds_mla_kernel
        #
        # Memory layout in kv_cache (per entry):
        # [NoPE_quantized (kv_lora_rank bytes)] [4 scales (16 bytes)] [RoPE (qk_rope_head_dim * 2 bytes)]
        #
        # Details:
        # - NoPE: 4 tiles of 128 elements each, quantized to FP8 E4M3 (1 byte per element) = 512 bytes total
        # - Scales: 4 float32 values (4 bytes each) for the 4 tiles = 16 bytes
        # - RoPE: qk_rope_head_dim fp16/bf16 values (2 bytes each), direct copy from k_pe = 128 bytes

        for i in range(num_tokens):
            slot_idx = slot_mapping[i].item()
            block_idx = slot_idx // block_size
            block_offset = slot_idx % block_size

            kv_cache_slice = kv_cache[block_idx, block_offset]

            # 1. Verify RoPE part (direct copy from k_pe)
            # From kernel line 394: const int64_t dst_idx = kv_lora_rank / 2 + 8 + pe_idx_start;
            # RoPE starts at byte offset: kv_lora_rank (NoPE) + 16 (scales) = kv_lora_rank + 16 bytes
            # When viewing as scalar_t (16-bit), offset is: (kv_lora_rank + 16) / 2
            rope_offset_in_bytes = kv_lora_rank + 16
            rope_offset_in_scalar = rope_offset_in_bytes // 2
            kv_cache_as_dtype = kv_cache_slice.view(dtype)
            kv_rope_cached = kv_cache_as_dtype[
                rope_offset_in_scalar : rope_offset_in_scalar + qk_rope_head_dim
            ]
            k_pe_original = k_pe[i]

            # RoPE should match exactly (direct copy, no quantization)
            torch.testing.assert_close(
                kv_rope_cached,
                k_pe_original,
                rtol=1e-5,
                atol=1e-5,
                msg=f"RoPE mismatch for token {i}",
            )

            # 2. Verify scale values
            # From kernel line 431: const uint64_t dst_idx = kv_lora_rank / 4 + tile_idx;
            # But wait - this is for when kv_cache_32bit = reinterpret_cast<float*>(&kv_cache[dst_idx_start])
            # So dst_idx is in terms of the reinterpreted pointer, which starts at byte 0
            # But looking more carefully: the comment says "first kv_lora_rank/2 bytes",
            # and dst_idx = kv_lora_rank/4 means kv_lora_rank/4 * 4 = kv_lora_rank bytes offset
            # So scales start at byte offset kv_lora_rank
            kv_cache_as_float32 = kv_cache_slice.view(torch.float32)
            scales_offset_in_float32 = kv_lora_rank // 4
            scales = kv_cache_as_float32[
                scales_offset_in_float32 : scales_offset_in_float32 + 4
            ]

            # Each scale should be positive and reasonable
            for tile_idx in range(4):
                self.assertGreater(
                    scales[tile_idx].item(),
                    0,
                    f"Scale {tile_idx} for token {i} should be positive",
                )

            # 3. Verify NoPE quantization and scale computation
            # Each tile has 128 elements quantized to FP8 (1 byte each)
            kv_c_original = kv_c[i]  # [kv_lora_rank]
            # NoPE data occupies first kv_lora_rank bytes (512 bytes for 512 FP8 elements)
            # kv_cache_slice is already uint8, shape is [entry_size]
            kv_nope_quantized = kv_cache_slice[:kv_lora_rank]

            for tile_idx in range(4):
                tile_start = tile_idx * 128
                tile_end = tile_start + 128

                # Get original values for this tile
                tile_values_original = kv_c_original[tile_start:tile_end].float()

                # Get quantized values for this tile (as uint8)
                tile_quantized_uint8 = kv_nope_quantized[tile_start:tile_end]

                # Get scale for this tile
                tile_scale = scales[tile_idx].item()

                # Verify scale computation
                # From kernel lines 413-426:
                # max_abs = max over 16 threads, each with 8 elements
                # tile_scale = max_abs / 448.0
                # tile_scale = max(tile_scale, FLT_MIN)
                max_abs = tile_values_original.abs().max().item()
                expected_scale = max(max_abs / 448.0, 1.1754944e-38)  # FLT_MIN

                # Scale should match expected value (with small tolerance for float precision)
                self.assertAlmostEqual(
                    tile_scale,
                    expected_scale,
                    places=6,
                    msg=f"Scale mismatch for tile {tile_idx} of token {i}: "
                    f"got {tile_scale}, expected {expected_scale}",
                )

                # Dequantize: convert FP8 back to float
                # The kernel quantizes as: scaled_convert<uint8_t>(vals[i], tile_scale)
                # Which internally does: __nv_cvt_float_to_fp8(val / scale, __NV_SATFINITE, __NV_E4M3)
                # To dequantize, we need to convert FP8 back to float and multiply by scale

                # For FP8 E4M3, we can approximate by converting uint8 to float8_e4m3fn
                # PyTorch 2.1+ supports float8_e4m3fn type
                try:
                    tile_quantized_fp8 = tile_quantized_uint8.view(torch.float8_e4m3fn)
                    tile_dequantized = tile_quantized_fp8.float() * tile_scale
                except AttributeError:
                    # Fallback if float8_e4m3fn is not available
                    # This is less accurate but gives us a rough approximation
                    print(
                        f"Warning: torch.float8_e4m3fn not available, using approximation"
                    )
                    tile_dequantized = tile_quantized_uint8.float() * (
                        tile_scale / 128.0
                    )

                # Calculate quantization error
                abs_error = (tile_dequantized - tile_values_original).abs()
                rel_error = abs_error / (tile_values_original.abs() + 1e-8)

                # Calculate statistics
                mean_abs_error = abs_error.mean().item()
                max_abs_error = abs_error.max().item()
                mean_rel_error = rel_error.mean().item()
                max_rel_error = rel_error.max().item()

                # Print error statistics for debugging
                if i == 0:  # Only print for first token to avoid spam
                    print(
                        f"  Tile {tile_idx}: mean_abs_err={mean_abs_error:.6f}, "
                        f"max_abs_err={max_abs_error:.6f}, "
                        f"mean_rel_err={mean_rel_error:.4f}, "
                        f"max_rel_err={max_rel_error:.4f}"
                    )

                # Verify quantization error is reasonable for FP8
                # FP8 E4M3 has limited precision, so we expect some error
                # But mean relative error should be < 10% and max < 50% for most values
                self.assertLess(
                    mean_rel_error,
                    0.05,  # 15% mean relative error threshold
                    f"Mean relative error too high for tile {tile_idx} of token {i}: "
                    f"{mean_rel_error:.4f}",
                )

        print(f"✓ DeepSeek MLA correctness checks passed")

    def test_concat_and_cache_ds_mla_bfloat16(self):
        """Test concat_and_cache_ds_mla with bfloat16"""
        self._run_concat_and_cache_ds_mla(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            num_tokens=42,
            block_size=16,
            num_blocks=8,
            dtype=torch.bfloat16,
            seed=0,
        )

    def test_concat_and_cache_ds_mla_float16(self):
        """Test concat_and_cache_ds_mla with float16"""
        self._run_concat_and_cache_ds_mla(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            num_tokens=42,
            block_size=16,
            num_blocks=8,
            dtype=torch.float16,
            seed=0,
        )

    def test_concat_and_cache_ds_mla_large(self):
        """Test concat_and_cache_ds_mla with larger configuration"""
        self._run_concat_and_cache_ds_mla(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            num_tokens=128,
            block_size=32,
            num_blocks=16,
            dtype=torch.bfloat16,
            seed=42,
        )


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
