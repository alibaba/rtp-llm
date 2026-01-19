import logging
import math
import unittest
from typing import List, Tuple

import torch
from attention_ref import compute_flashinfer_prefill_reference
from base_attention_test import compare_tensors, set_seed

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TestFlashInferPrefillPaddedWithMask(unittest.TestCase):
    """Test FlashInfer BatchPrefillWithPagedKVCacheWrapper with padded Q and custom mask"""

    def setUp(self):
        """Set up test fixtures"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        self.device = torch.device("cuda")
        set_seed(42)

    def _create_padded_qkv_and_mask(
        self,
        batch_size: int,
        real_seq_lens: List[int],
        num_heads: int,
        kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Create padded QKV tensors and custom mask

        Returns:
            q_padded: [batch, max_seq_len, num_heads, head_dim]
            k_padded: [batch, max_seq_len, kv_heads, head_dim]
            v_padded: [batch, max_seq_len, kv_heads, head_dim]
            custom_mask: [batch, max_seq_len, max_seq_len] boolean mask (True = masked)
            real_seq_lens: List of real sequence lengths
        """
        max_seq_len = max(real_seq_lens)

        # Create padded tensors
        q_padded = torch.randn(
            batch_size,
            max_seq_len,
            num_heads,
            head_dim,
            dtype=dtype,
            device=self.device,
        )
        k_padded = torch.randn(
            batch_size, max_seq_len, kv_heads, head_dim, dtype=dtype, device=self.device
        )
        v_padded = torch.randn(
            batch_size, max_seq_len, kv_heads, head_dim, dtype=dtype, device=self.device
        )

        # Create custom mask to mask out padding positions
        # True = masked (will be set to -inf in attention scores)
        custom_mask = torch.zeros(
            batch_size, max_seq_len, max_seq_len, dtype=torch.bool, device=self.device
        )

        for i in range(batch_size):
            real_len = real_seq_lens[i]
            # Mask out padding in Q dimension
            custom_mask[i, real_len:, :] = True
            # Mask out padding in KV dimension
            custom_mask[i, :, real_len:] = True
            # Apply causal mask for valid positions
            for q_pos in range(real_len):
                custom_mask[i, q_pos, q_pos + 1 : real_len] = True

        return q_padded, k_padded, v_padded, custom_mask, real_seq_lens

    def _convert_to_ragged(
        self,
        padded_tensor: torch.Tensor,
        real_seq_lens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert padded tensor to ragged format

        Args:
            padded_tensor: [batch, max_seq_len, ...]
            real_seq_lens: List of real sequence lengths

        Returns:
            ragged_tensor: [total_tokens, ...]
            cu_seqlens: [batch + 1] cumulative sequence lengths
        """
        batch_size = len(real_seq_lens)
        ragged_list = []

        for i in range(batch_size):
            real_len = real_seq_lens[i]
            ragged_list.append(padded_tensor[i, :real_len])

        ragged_tensor = torch.cat(ragged_list, dim=0)

        cu_seqlens = torch.tensor(
            [0] + [sum(real_seq_lens[: i + 1]) for i in range(batch_size)],
            dtype=torch.int32,
            device=self.device,
        )

        return ragged_tensor, cu_seqlens

    def _create_custom_mask_flattened(
        self,
        custom_mask: torch.Tensor,
        real_seq_lens: List[int],
    ) -> torch.Tensor:
        """
        Convert batched custom mask to flattened format for FlashInfer

        Args:
            custom_mask: [batch, max_seq_len, max_seq_len] boolean mask (True = masked)
            real_seq_lens: List of real sequence lengths

        Returns:
            mask_flat: [sum(seq_len * seq_len)] flattened mask
                      FlashInfer convention: False = masked, True = valid
        """
        batch_size = len(real_seq_lens)
        mask_list = []

        for i in range(batch_size):
            real_len = real_seq_lens[i]
            # Extract valid region [real_len, real_len]
            pytorch_mask = custom_mask[i, :real_len, :real_len]
            # FlashInfer convention is opposite: False = masked, True = valid
            # So we need to invert the PyTorch mask
            flashinfer_mask = ~pytorch_mask
            mask_list.append(flashinfer_mask.flatten())

        return torch.cat(mask_list, dim=0)

    def _compute_reference_with_mask(
        self,
        q_padded: torch.Tensor,
        k_padded: torch.Tensor,
        v_padded: torch.Tensor,
        mask: torch.Tensor,
        real_seq_lens: List[int],
    ) -> torch.Tensor:
        """
        Compute reference attention output with mask using PyTorch

        Args:
            q_padded: [batch, max_seq_len, num_heads, head_dim]
            k_padded: [batch, max_seq_len, kv_heads, head_dim]
            v_padded: [batch, max_seq_len, kv_heads, head_dim]
            mask: [batch, max_seq_len, max_seq_len] (True = masked)
            real_seq_lens: List of real sequence lengths

        Returns:
            output: [batch, max_seq_len, num_heads, head_dim]
        """
        batch_size, max_seq_len, num_heads, head_dim = q_padded.shape
        _, _, kv_heads, _ = k_padded.shape

        # Handle GQA: repeat KV heads if needed
        if num_heads != kv_heads:
            num_groups = num_heads // kv_heads
            k_expanded = k_padded.repeat_interleave(num_groups, dim=2)
            v_expanded = v_padded.repeat_interleave(num_groups, dim=2)
        else:
            k_expanded = k_padded
            v_expanded = v_padded

        # Reshape for batched matmul: [batch, num_heads, seq_len, head_dim]
        q = q_padded.transpose(1, 2)
        k = k_expanded.transpose(1, 2)
        v = v_expanded.transpose(1, 2)

        # Compute attention scores
        scale = 1.0 / math.sqrt(head_dim)
        scores = (
            torch.matmul(q, k.transpose(-2, -1)) * scale
        )  # [batch, heads, seq, seq]

        # Apply mask (expand mask for all heads)
        mask_expanded = mask.unsqueeze(1)  # [batch, 1, seq, seq]
        scores = scores.masked_fill(mask_expanded, float("-inf"))

        # Softmax and apply to values
        attn_weights = torch.softmax(scores, dim=-1)
        # Handle NaN from softmax of all -inf
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        output = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]
        output = output.transpose(1, 2)  # [batch, seq, heads, head_dim]

        return output

    def test_padded_q_with_custom_mask_single_batch(self):
        """Test with single batch, padded input and custom mask"""
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

        batch_size = 1
        real_seq_lens = [5]
        max_seq_len = 8  # Padded to 8
        num_heads = 8
        kv_heads = 8
        head_dim = 64
        page_size = 128

        # Create padded QKV and mask
        q_padded, k_padded, v_padded, custom_mask, _ = self._create_padded_qkv_and_mask(
            batch_size, real_seq_lens, num_heads, kv_heads, head_dim
        )

        # Compute reference output
        ref_output_padded = self._compute_reference_with_mask(
            q_padded, k_padded, v_padded, custom_mask, real_seq_lens
        )

        # Convert to ragged format for FlashInfer
        q_ragged, qo_indptr = self._convert_to_ragged(q_padded, real_seq_lens)
        k_ragged, kv_indptr = self._convert_to_ragged(k_padded, real_seq_lens)
        v_ragged, _ = self._convert_to_ragged(v_padded, real_seq_lens)

        # Create custom mask in flattened format
        custom_mask_flat = self._create_custom_mask_flattened(
            custom_mask, real_seq_lens
        )

        # Setup paged KV cache (simple: 1 page per request)
        total_pages = batch_size
        paged_kv_indptr = torch.arange(
            batch_size + 1, dtype=torch.int32, device=self.device
        )
        paged_kv_indices = torch.arange(
            batch_size, dtype=torch.int32, device=self.device
        )
        paged_kv_last_page_len = torch.tensor(
            real_seq_lens, dtype=torch.int32, device=self.device
        )

        # Prepare paged K/V cache in the correct format
        # FlashInfer expects: [num_pages, 2, page_size, num_heads, head_dim]
        # where dim=1 has [K, V] stacked
        paged_kv_cache = torch.zeros(
            total_pages,
            2,
            page_size,
            kv_heads,
            head_dim,
            dtype=q_ragged.dtype,
            device=self.device,
        )

        # Fill the cache with actual K/V data
        for i in range(batch_size):
            seq_len = real_seq_lens[i]
            start_idx = kv_indptr[i].item()
            end_idx = kv_indptr[i + 1].item()
            paged_kv_cache[i, 0, :seq_len] = k_ragged[start_idx:end_idx]  # K
            paged_kv_cache[i, 1, :seq_len] = v_ragged[start_idx:end_idx]  # V

        # Setup FlashInfer wrapper
        workspace_size = 128 * 1024 * 1024  # 128MB
        workspace_buffer = torch.empty(
            workspace_size, dtype=torch.uint8, device=self.device
        )

        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")

        # Plan with custom mask
        wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_heads,
            kv_heads,
            head_dim,
            page_size,
            causal=False,  # Use custom mask instead
            custom_mask=custom_mask_flat,
        )

        # Run attention
        output_ragged = wrapper.run(q_ragged, paged_kv_cache)

        # Convert output back to padded format for comparison
        output_padded = torch.zeros(
            batch_size,
            max_seq_len,
            num_heads,
            head_dim,
            dtype=output_ragged.dtype,
            device=self.device,
        )
        for i in range(batch_size):
            start_idx = qo_indptr[i].item()
            end_idx = qo_indptr[i + 1].item()
            seq_len = real_seq_lens[i]
            output_padded[i, :seq_len] = output_ragged[start_idx:end_idx]

        # Compare outputs (only valid regions)
        for i in range(batch_size):
            seq_len = real_seq_lens[i]
            actual = output_padded[i, :seq_len]
            expected = ref_output_padded[i, :seq_len]

            # compare_tensors raises AssertionError if mismatch
            compare_tensors(actual, expected, atol=1e-2, rtol=1e-2, name=f"Batch {i}")

    def test_padded_q_with_custom_mask_multi_batch(self):
        """Test with multiple batches of different lengths, padded input and custom mask"""
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

        batch_size = 3
        real_seq_lens = [5, 3, 7]
        max_seq_len = max(real_seq_lens)
        num_heads = 32
        kv_heads = 8  # GQA
        head_dim = 128
        page_size = 128

        # Create padded QKV and mask
        q_padded, k_padded, v_padded, custom_mask, _ = self._create_padded_qkv_and_mask(
            batch_size, real_seq_lens, num_heads, kv_heads, head_dim
        )

        # Compute reference output
        ref_output_padded = self._compute_reference_with_mask(
            q_padded, k_padded, v_padded, custom_mask, real_seq_lens
        )

        # Convert to ragged format for FlashInfer
        q_ragged, qo_indptr = self._convert_to_ragged(q_padded, real_seq_lens)
        k_ragged, kv_indptr = self._convert_to_ragged(k_padded, real_seq_lens)
        v_ragged, _ = self._convert_to_ragged(v_padded, real_seq_lens)

        # Create custom mask in flattened format
        custom_mask_flat = self._create_custom_mask_flattened(
            custom_mask, real_seq_lens
        )

        # Setup paged KV cache
        total_pages = batch_size
        paged_kv_indptr = torch.arange(
            batch_size + 1, dtype=torch.int32, device=self.device
        )
        paged_kv_indices = torch.arange(
            batch_size, dtype=torch.int32, device=self.device
        )
        paged_kv_last_page_len = torch.tensor(
            real_seq_lens, dtype=torch.int32, device=self.device
        )

        # Prepare paged K/V cache in the correct format
        # FlashInfer expects: [num_pages, 2, page_size, num_heads, head_dim]
        paged_kv_cache = torch.zeros(
            total_pages,
            2,
            page_size,
            kv_heads,
            head_dim,
            dtype=q_ragged.dtype,
            device=self.device,
        )

        # Fill the cache
        for i in range(batch_size):
            seq_len = real_seq_lens[i]
            start_idx = kv_indptr[i].item()
            end_idx = kv_indptr[i + 1].item()
            paged_kv_cache[i, 0, :seq_len] = k_ragged[start_idx:end_idx]  # K
            paged_kv_cache[i, 1, :seq_len] = v_ragged[start_idx:end_idx]  # V

        # Setup FlashInfer wrapper
        workspace_size = 128 * 1024 * 1024
        workspace_buffer = torch.empty(
            workspace_size, dtype=torch.uint8, device=self.device
        )

        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")

        # Plan with custom mask
        wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_heads,
            kv_heads,
            head_dim,
            page_size,
            causal=False,
            custom_mask=custom_mask_flat,
        )

        # Run attention
        output_ragged = wrapper.run(q_ragged, paged_kv_cache)

        # Convert output back to padded format
        output_padded = torch.zeros(
            batch_size,
            max_seq_len,
            num_heads,
            head_dim,
            dtype=output_ragged.dtype,
            device=self.device,
        )
        for i in range(batch_size):
            start_idx = qo_indptr[i].item()
            end_idx = qo_indptr[i + 1].item()
            seq_len = real_seq_lens[i]
            output_padded[i, :seq_len] = output_ragged[start_idx:end_idx]

        # Compare outputs
        for i in range(batch_size):
            seq_len = real_seq_lens[i]
            actual = output_padded[i, :seq_len]
            expected = ref_output_padded[i, :seq_len]

            # compare_tensors raises AssertionError if mismatch
            compare_tensors(
                actual,
                expected,
                atol=1e-2,
                rtol=1e-2,
                name=f"Batch {i} (len={seq_len})",
            )


if __name__ == "__main__":
    unittest.main()
