"""
Test suite for concat_and_cache_mla operation
Adapted from vLLM's test_cache.py
Using unittest framework
"""

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


class TestConcatAndCacheMLA(unittest.TestCase):
    """Test suite for concat_and_cache_mla operation"""

    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        cls.device = "cuda:0"
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.cuda.set_device(cls.device)

    def _run_concat_and_cache_mla(
        self,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        num_tokens: int,
        block_size: int,
        num_blocks: int,
        dtype: torch.dtype,
        seed: int,
        kv_cache_dtype: str,
    ) -> None:
        """
        Helper method to run concat_and_cache_mla test

        This test verifies that the MLA concat and cache operation correctly:
        1. Concatenates kv_c and k_pe tensors
        2. Stores them in the paged KV cache at the correct positions
        3. Optionally quantizes to FP8 format
        """
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
        entry_size = kv_lora_rank + qk_rope_head_dim

        # Create scale tensor
        scale = torch.tensor(0.1, dtype=torch.float32, device=self.device)

        # Create KV cache
        kv_cache = create_mla_cache(
            num_blocks, block_size, entry_size, dtype, kv_cache_dtype, self.device
        )

        # Create reference output
        ref_temp = torch.zeros(*kv_cache.shape, dtype=dtype, device=self.device)

        # Fill reference with expected values
        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            block_idx = slot // block_size
            block_offset = slot % block_size
            ref_temp[block_idx, block_offset, :kv_lora_rank] = kv_c[i]
            ref_temp[block_idx, block_offset, kv_lora_rank:] = k_pe[i]

        # Apply quantization to reference if needed
        if kv_cache_dtype == "fp8_e4m3":
            raise ValueError("FP8_E4M3 is not tested in this test suite")
        else:
            ref_kv_cache = ref_temp

        # Call the concat_and_cache_mla kernel
        compute_ops.concat_and_cache_mla(
            kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
        )

        # Compare results
        if kv_cache_dtype == "fp8_e4m3":
            # For FP8, just check that the kernel ran without error
            # Full numerical verification would require proper FP8 dequantization
            print(
                f"⚠ FP8 test completed (numerical verification skipped): "
                f"kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}, "
                f"num_tokens={num_tokens}, dtype={dtype}"
            )
        else:
            torch.testing.assert_close(kv_cache, ref_kv_cache)
            print(
                f"✓ Test passed: kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}, "
                f"num_tokens={num_tokens}, dtype={dtype}, kv_cache_dtype={kv_cache_dtype}"
            )

    def test_concat_and_cache_mla_bfloat16_auto(self):
        """Test concat_and_cache_mla with bfloat16 and auto cache dtype"""
        self._run_concat_and_cache_mla(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            num_tokens=42,
            block_size=16,
            num_blocks=8,
            dtype=torch.bfloat16,
            seed=0,
            kv_cache_dtype="auto",
        )

    def test_concat_and_cache_mla_float_auto(self):
        """Test concat_and_cache_mla with float and auto cache dtype"""
        self._run_concat_and_cache_mla(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            num_tokens=42,
            block_size=16,
            num_blocks=8,
            dtype=torch.float,
            seed=0,
            kv_cache_dtype="auto",
        )

    def test_concat_and_cache_mla_padding(self):
        """Test that concat_and_cache_mla correctly handles padded tokens (slot_idx = -1)"""
        torch.manual_seed(42)
        torch.set_default_device(self.device)
        torch.cuda.set_device(self.device)

        num_tokens = 10
        block_size = 16
        num_blocks = 8
        dtype = torch.bfloat16
        kv_cache_dtype = "auto"
        kv_lora_rank = 512
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
        entry_size = kv_lora_rank + qk_rope_head_dim

        scale = torch.tensor(0.1, dtype=torch.float32, device=self.device)
        kv_cache = create_mla_cache(
            num_blocks, block_size, entry_size, dtype, kv_cache_dtype, self.device
        )

        # Call the kernel
        compute_ops.concat_and_cache_mla(
            kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
        )

        # Verify that non-padded tokens are written correctly
        for i, slot in enumerate(slot_mapping_lst):
            if slot >= 0:
                block_idx = slot // block_size
                block_offset = slot % block_size

                # Check that data was written
                cache_kv_c = kv_cache[block_idx, block_offset, :kv_lora_rank]
                cache_k_pe = kv_cache[block_idx, block_offset, kv_lora_rank:]

                torch.testing.assert_close(cache_kv_c, kv_c[i], atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(cache_k_pe, k_pe[i], atol=1e-5, rtol=1e-5)

        print("✓ Padding test passed")

    def test_concat_and_cache_mla_zero_rope_dim(self):
        """Test concat_and_cache_mla with qk_rope_head_dim=64 nope_dim=448(model1 format)"""
        self._run_concat_and_cache_mla(
            kv_lora_rank=448,
            qk_rope_head_dim=64,
            num_tokens=42,
            block_size=16,
            num_blocks=8,
            dtype=torch.bfloat16,
            seed=0,
            kv_cache_dtype="auto",
        )


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
