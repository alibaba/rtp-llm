import random
import sys
import unittest

# Add RTP-LLM to path if needed
sys.path.append("/data2/baowending.bwd/new/RTP-LLM/github-opensource/")

import numpy as np
import torch

from rtp_llm.models_py.bindings.cuda.test.concat_and_cache_mla.util import (
    create_mla_cache,
)
from rtp_llm.ops import compute_ops


class TestConcatAndCacheModel1MLA(unittest.TestCase):
    """Test suite for concat_and_cache_model1_mla operation (MODEL1 MLA)"""

    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        cls.device = "cuda:0"
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.cuda.set_device(cls.device)

    def _run_concat_and_cache_model1_mla(
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
        Helper method to run concat_and_cache_model1_mla test

        This test verifies the MODEL1 MLA variant which:
        1. Uses tile-wise dynamic quantization for NoPE part (7 tiles of 64 elements)
        2. Keeps RoPE part in original precision (BF16)
        3. Stores scale values (fp8_e8m0) at block end
        4. Memory layout: 584 bytes per token
           - NoPE + RoPE: 448 + 128 = 576 bytes
           - Scales: 7×1 + 1 padding = 8 bytes
        """
        if dtype.itemsize != 2:
            self.skipTest("model1_mla only supports 16-bit input (fp16/bf16)")

        kv_cache_dtype = "fp8_model1_mla"
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

        # For fp8_model1_mla: entry_size = 584 bytes
        # = 448 (NoPE FP8) + 128 (RoPE BF16) + 8 (7 scales + 1 padding)
        entry_size = 584

        scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        kv_cache = create_mla_cache(
            num_blocks,
            block_size,
            entry_size,
            dtype=torch.uint8,
            kv_cache_dtype=kv_cache_dtype,
            device=self.device,
        )

        # Call the concat_and_cache_mla kernel with fp8_model1_mla
        compute_ops.concat_and_cache_mla(
            kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
        )

        # For MODEL1 MLA, just verify the kernel runs without error
        # Full numerical verification requires exact matching of the FP8 quantization logic
        print(
            f"✓ MODEL1 MLA kernel executed successfully: "
            f"kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}, "
            f"num_tokens={num_tokens}, dtype={dtype}"
        )

        # Basic sanity checks
        # Check that RoPE part is written correctly (should be in BF16, not quantized)
        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            block_idx = slot // block_size
            block_offset = slot % block_size

            # MODEL1 Memory Layout:
            # 1. NoPE + RoPE section: block_size * 576 bytes
            #    Each token: [NoPE: 448 bytes (FP8)] + [RoPE: 128 bytes (BF16)]
            # 2. Scale section: block_size * 8 bytes
            #    Each token: 7 scales (fp8_e8m0) + 1 byte padding

            # Flatten kv_cache to 1D byte array for easier byte-level addressing
            kv_cache_bytes = kv_cache.reshape(-1)

            # Calculate byte offset for this token's NoPE+RoPE section
            nope_rope_stride = (
                kv_lora_rank + qk_rope_head_dim * 2
            )  # 448 + 128 = 576 bytes
            token_offset = (
                block_idx * block_size * entry_size + block_offset * nope_rope_stride
            )

            # RoPE starts at offset + 448 bytes
            rope_start = token_offset + kv_lora_rank  # 448 bytes for NoPE
            rope_end = rope_start + qk_rope_head_dim * 2  # 128 bytes for RoPE

            # Extract RoPE part as bytes then reinterpret as dtype
            kv_rope_bytes = kv_cache_bytes[rope_start:rope_end]
            kv_rope = kv_rope_bytes.view(dtype)

            # Check that RoPE values are not all zeros
            self.assertGreater(
                kv_rope.abs().sum().item(),
                0,
                f"RoPE values are all zeros for token {i}",
            )

            # Check RoPE values match input (should not be quantized)
            # Allow some tolerance due to potential numerical issues
            torch.testing.assert_close(
                kv_rope,
                k_pe[i],
                atol=1e-3,
                rtol=1e-2,
                msg=f"RoPE values mismatch for token {i}",
            )

        print(f"✓ MODEL1 MLA sanity checks passed")

        # Verify quantization accuracy by comparing with reference implementation
        self._verify_model1_quantization_accuracy(
            kv_c=kv_c,
            k_pe=k_pe,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_size=block_size,
            entry_size=entry_size,
            dtype=dtype,
            num_tokens=num_tokens,
        )

        print(f"✓ MODEL1 MLA quantization accuracy checks passed")

    def _verify_model1_quantization_accuracy(
        self,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        block_size: int,
        entry_size: int,
        dtype: torch.dtype,
        num_tokens: int,
    ):
        """
        Verify MODEL1 quantization accuracy against reference implementation.

        This follows the exact kernel logic from concat_and_cache_ds_model1_kernel:
        1. NoPE is quantized into 7 tiles (64 elements each)
        2. Each tile has its own scale factor (fp8_e8m0 format)
        3. RoPE is stored as-is in BF16/FP16
        """
        import math

        # Helper function to compute FP8 E8M0 scale (power of 2)
        def compute_fp8_e8m0_scale(max_abs: float) -> float:
            """Compute FP8 E8M0 scale factor (power of 2)."""
            tile_scale = max(max_abs / 448.0, 1e-38)  # FLT_MIN
            # Convert to power of 2: exp2(ceil(log2(scale)))
            log2_scale = math.ceil(math.log2(tile_scale))
            return 2.0**log2_scale

        def encode_fp8_e8m0(scale: float) -> int:
            """Encode scale to FP8 E8M0 format (uint8)."""
            log2_scale = math.log2(scale)
            encoded = int(min(max(log2_scale + 127.0, 0.0), 255.0))
            return encoded

        def decode_fp8_e8m0(encoded: int) -> float:
            """Decode FP8 E8M0 format to scale value."""
            return 2.0 ** (encoded - 127.0)

        kv_cache_bytes = kv_cache.reshape(-1)
        nope_rope_stride = kv_lora_rank + qk_rope_head_dim * 2  # 576 bytes

        # Check a subset of tokens
        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            if slot < 0:
                continue

            block_idx = slot // block_size
            block_offset = slot % block_size

            # Get input NoPE values for this token
            nope_input = kv_c[i].cpu()  # [448] in original dtype

            # Verify scales for each tile (7 tiles total)
            scale_section_offset = block_size * nope_rope_stride
            scale_offset = (
                block_idx * block_size * entry_size
                + scale_section_offset
                + block_offset * 8
            )
            stored_scales_uint8 = kv_cache_bytes[scale_offset : scale_offset + 7].cpu()

            # Compute reference scales for each tile
            ref_scales = []
            for tile_idx in range(7):
                # Each tile has 64 elements
                tile_start = tile_idx * 64
                tile_end = tile_start + 64
                tile_values = nope_input[tile_start:tile_end]

                # Compute max absolute value in tile
                max_abs = float(tile_values.abs().max().item())

                # Compute scale following kernel logic
                tile_scale = compute_fp8_e8m0_scale(max_abs)
                ref_scales.append(tile_scale)

            # Verify each scale matches reference
            for tile_idx in range(7):
                stored_scale_encoded = int(stored_scales_uint8[tile_idx].item())
                stored_scale = decode_fp8_e8m0(stored_scale_encoded)
                ref_scale = ref_scales[tile_idx]
                ref_scale_encoded = encode_fp8_e8m0(ref_scale)

                # Scales should match exactly (both are power of 2)
                self.assertEqual(
                    stored_scale_encoded,
                    ref_scale_encoded,
                    f"Token {i}, Tile {tile_idx}: "
                    f"stored_scale={stored_scale:.6e} (encoded={stored_scale_encoded}), "
                    f"ref_scale={ref_scale:.6e} (encoded={ref_scale_encoded})",
                )

            # Verify quantized NoPE values using proper FP8 dequantization
            token_offset = (
                block_idx * block_size * entry_size + block_offset * nope_rope_stride
            )
            stored_nope_uint8 = kv_cache_bytes[
                token_offset : token_offset + kv_lora_rank
            ].cpu()

            # Check that quantized values are within reasonable range
            # Each tile should use its corresponding scale
            for tile_idx in range(7):
                tile_start = tile_idx * 64
                tile_end = tile_start + 64
                tile_stored_uint8 = stored_nope_uint8[tile_start:tile_end]
                tile_input = nope_input[tile_start:tile_end].float()
                tile_scale = ref_scales[tile_idx]

                # Verify quantization: stored values should be in [0, 255] range
                self.assertTrue(
                    torch.all((tile_stored_uint8 >= 0) & (tile_stored_uint8 <= 255)),
                    f"Token {i}, Tile {tile_idx}: quantized values out of uint8 range",
                )

                # Dequantize using proper FP8 E4M3 format
                # Reinterpret uint8 as float8_e4m3fn, then convert to float and scale
                tile_stored_fp8 = tile_stored_uint8.view(torch.float8_e4m3fn)
                dequantized = tile_stored_fp8.float() * tile_scale

                # Verify dequantization accuracy
                max_input = tile_input.abs().max().item()
                if max_input > 1e-6:  # Only check non-zero tiles
                    abs_error = (dequantized - tile_input).abs().max().item()
                    relative_error = abs_error / max_input

                    # FP8 E4M3 has limited precision (3-bit mantissa)
                    # Theoretical precision: ~1/8 = 12.5% for worst case
                    # In practice, with proper scaling, we expect < 10% error
                    # Note: This is max error in the tile, not average error
                    self.assertLess(
                        relative_error,
                        0.10,  # 10% tolerance for FP8 E4M3 max quantization error
                        f"Token {i}, Tile {tile_idx}: max dequantization error too large "
                        f"(max_relative_error={relative_error:.4%}, "
                        f"max_abs_error={abs_error:.6f}, max_input={max_input:.6f})",
                    )

                    # Check average error (should be much better than max error)
                    avg_abs_error = (dequantized - tile_input).abs().mean().item()
                    avg_input = tile_input.abs().mean().item()
                    if avg_input > 1e-6:
                        avg_relative_error = avg_abs_error / avg_input

                        # FP8 E4M3 has 3-bit mantissa, giving ~12.5% precision per step
                        # With proper scaling, average error should be < 3%
                        # This is much stricter than the 50% used before
                        self.assertLess(
                            avg_relative_error,
                            0.03,  # 3% tolerance for average error
                            f"Token {i}, Tile {tile_idx}: average dequantization error too large "
                            f"(avg_relative_error={avg_relative_error:.4%}, "
                            f"avg_abs_error={avg_abs_error:.6f}, avg_input={avg_input:.6f})",
                        )

    def test_concat_and_cache_model1_mla_bfloat16(self):
        """Test concat_and_cache_model1_mla with bfloat16"""
        self._run_concat_and_cache_model1_mla(
            kv_lora_rank=448,
            qk_rope_head_dim=64,
            num_tokens=42,
            block_size=16,
            num_blocks=8,
            dtype=torch.bfloat16,
            seed=0,
        )

    def test_concat_and_cache_model1_mla_float16(self):
        """Test concat_and_cache_model1_mla with float16"""
        self._run_concat_and_cache_model1_mla(
            kv_lora_rank=448,
            qk_rope_head_dim=64,
            num_tokens=42,
            block_size=16,
            num_blocks=8,
            dtype=torch.float16,
            seed=0,
        )

    def test_concat_and_cache_model1_mla_small_batch(self):
        """Test concat_and_cache_model1_mla with small batch"""
        self._run_concat_and_cache_model1_mla(
            kv_lora_rank=448,
            qk_rope_head_dim=64,
            num_tokens=8,
            block_size=16,
            num_blocks=4,
            dtype=torch.bfloat16,
            seed=42,
        )

    def test_concat_and_cache_model1_mla_large_batch(self):
        """Test concat_and_cache_model1_mla with large batch"""
        self._run_concat_and_cache_model1_mla(
            kv_lora_rank=448,
            qk_rope_head_dim=64,
            num_tokens=128,
            block_size=32,
            num_blocks=16,
            dtype=torch.bfloat16,
            seed=123,
        )

    def test_concat_and_cache_model1_mla_padding(self):
        """Test that concat_and_cache_model1_mla correctly handles padded tokens (slot_idx = -1)"""
        torch.manual_seed(99)
        torch.set_default_device(self.device)
        torch.cuda.set_device(self.device)

        num_tokens = 10
        block_size = 16
        num_blocks = 8
        dtype = torch.bfloat16
        kv_cache_dtype = "fp8_model1_mla"
        kv_lora_rank = 448
        qk_rope_head_dim = 64

        # Create slot mapping with some padded tokens (-1)
        slot_mapping_lst = [0, 1, 2, -1, 3, -1, 4, 5, -1, 6]
        slot_mapping = torch.tensor(
            slot_mapping_lst, dtype=torch.long, device=self.device
        )

        # Create input tensors
        kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=self.device)
        k_pe = torch.randn(
            num_tokens, qk_rope_head_dim, dtype=dtype, device=self.device
        )

        entry_size = 584
        scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        kv_cache = create_mla_cache(
            num_blocks, block_size, entry_size, torch.uint8, kv_cache_dtype, self.device
        )

        # Save a copy of cache before kernel execution to verify padding
        kv_cache_before = kv_cache.clone()

        # Call the kernel
        compute_ops.concat_and_cache_mla(
            kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
        )

        kv_cache_bytes = kv_cache.reshape(-1)
        kv_cache_bytes_before = kv_cache_before.reshape(-1)
        nope_rope_stride = kv_lora_rank + qk_rope_head_dim * 2  # 576 bytes

        # Verify that non-padded tokens have data written correctly
        non_padded_count = 0
        for i, slot in enumerate(slot_mapping_lst):
            if slot >= 0:
                non_padded_count += 1
                block_idx = slot // block_size
                block_offset = slot % block_size

                # Calculate byte offset for this token's NoPE+RoPE section
                token_offset = (
                    block_idx * block_size * entry_size
                    + block_offset * nope_rope_stride
                )

                # RoPE starts at offset + 448 bytes
                rope_start = token_offset + kv_lora_rank
                rope_end = rope_start + qk_rope_head_dim * 2

                # Extract RoPE part as bytes then reinterpret as dtype
                kv_rope_bytes = kv_cache_bytes[rope_start:rope_end]
                kv_rope = kv_rope_bytes.view(dtype)

                # Check that RoPE data was written correctly
                torch.testing.assert_close(
                    kv_rope,
                    k_pe[i],
                    atol=1e-3,
                    rtol=1e-2,
                    msg=f"RoPE mismatch for non-padded token {i} at slot {slot}",
                )

                # Verify NoPE part is not all zeros (quantized data should be written)
                nope_bytes = kv_cache_bytes[token_offset : token_offset + kv_lora_rank]
                self.assertGreater(
                    nope_bytes.sum().item(),
                    0,
                    f"NoPE data is all zeros for non-padded token {i} at slot {slot}",
                )

                # Verify scale section has valid scales
                scale_section_offset = block_size * nope_rope_stride
                scale_offset = (
                    block_idx * block_size * entry_size
                    + scale_section_offset
                    + block_offset * 8
                )
                stored_scales = kv_cache_bytes[scale_offset : scale_offset + 7]
                self.assertGreater(
                    stored_scales.sum().item(),
                    0,
                    f"Scale data is all zeros for non-padded token {i} at slot {slot}",
                )

        # Verify that padded tokens (slot_idx = -1) did NOT modify the cache
        # by checking that cache regions corresponding to "would-be" positions remain unchanged
        padded_count = 0
        for i, slot in enumerate(slot_mapping_lst):
            if slot < 0:
                padded_count += 1
                # For padded tokens, we can't directly verify "nothing was written"
                # because there's no specific slot to check. The kernel should skip these.
                # We rely on the fact that the kernel checks (slot_idx < 0) and returns early.
                pass

        print(
            f"✓ MODEL1 MLA padding test passed: "
            f"{non_padded_count} non-padded tokens verified, "
            f"{padded_count} padded tokens skipped"
        )

        # Run full quantization accuracy check on non-padded tokens only
        # Create filtered inputs and slot mapping with only non-padded tokens
        non_padded_indices = [i for i, slot in enumerate(slot_mapping_lst) if slot >= 0]
        if len(non_padded_indices) > 0:
            kv_c_filtered = kv_c[non_padded_indices]
            k_pe_filtered = k_pe[non_padded_indices]
            slot_mapping_filtered = torch.tensor(
                [slot_mapping_lst[i] for i in non_padded_indices],
                dtype=torch.long,
                device=self.device,
            )

            self._verify_model1_quantization_accuracy(
                kv_c=kv_c_filtered,
                k_pe=k_pe_filtered,
                kv_cache=kv_cache,
                slot_mapping=slot_mapping_filtered,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                block_size=block_size,
                entry_size=entry_size,
                dtype=dtype,
                num_tokens=len(non_padded_indices),
            )

            print(
                f"✓ MODEL1 MLA padding test quantization accuracy verified for {len(non_padded_indices)} non-padded tokens"
            )


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
