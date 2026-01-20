# -*- coding: utf-8 -*-
import logging
import math
import unittest
from typing import List, Tuple

import torch
from base_attention_test import compare_tensors, set_seed

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPrefillPagedAttnOp,
)
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import (
    ParamsBase,
    PyAttentionInputs,
    PyPrefillCudaGaphCopyParams,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TestPyFlashinferPrefillPagedWithMask(unittest.TestCase):
    """Test PyFlashinferPrefillPagedAttnOp with paged KV cache and custom mask"""

    def setUp(self):
        """Set up test fixtures"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        self.device = torch.device("cuda")
        set_seed(42)

    def _create_config(
        self,
        head_num: int = 32,
        head_num_kv: int = 8,
        size_per_head: int = 128,
    ):
        """Helper to create a test AttentionConfigs"""
        attn_configs = AttentionConfigs()
        attn_configs.head_num = head_num
        attn_configs.kv_head_num = head_num_kv
        attn_configs.size_per_head = size_per_head
        attn_configs.use_mla = False
        attn_configs.dtype = torch.float16

        return attn_configs

    def _create_attn_inputs(
        self,
        seq_lens: List[int],
        max_seq_len: int,
    ):
        """
        Helper to create PyAttentionInputs for padded Q format

        Args:
            seq_lens: List of real sequence lengths
            max_seq_len: Maximum sequence length (Q is padded to this)

        Returns:
            PyAttentionInputs configured for padded Q format with CUDA graph params
        """
        batch_size = len(seq_lens)

        # Create PyPrefillCudaGaphCopyParams to indicate CUDA graph padded mode
        prefill_params = PyPrefillCudaGaphCopyParams()
        prefill_params.max_seq_len = max_seq_len
        prefill_params.max_batch_size = batch_size

        # Create PyAttentionInputs for padded Q
        attn_inputs = PyAttentionInputs()
        attn_inputs.input_lengths = torch.tensor(
            seq_lens, dtype=torch.int32, device="cpu"
        )

        # cu_seqlens for padded Q: [0, max_seq_len, 2*max_seq_len, ...]
        attn_inputs.cu_seqlens = torch.tensor(
            [i * max_seq_len for i in range(batch_size + 1)],
            dtype=torch.int32,
            device=self.device,
        )

        # Set prefill_cuda_graph_copy_params to trigger custom mask generation
        attn_inputs.prefill_cuda_graph_copy_params = prefill_params

        return attn_inputs

    def _convert_to_ragged(
        self,
        padded_tensor: torch.Tensor,
        real_seq_lens: List[int],
    ) -> torch.Tensor:
        """
        Convert padded tensor to ragged format
        Args:
            padded_tensor: [batch, max_seq_len, ...]
            real_seq_lens: List of real sequence lengths
        Returns:
            ragged_tensor: [total_tokens, ...]
        """
        batch_size = len(real_seq_lens)
        ragged_list: List[torch.Tensor] = []

        for i in range(batch_size):
            real_len = real_seq_lens[i]
            ragged_list.append(padded_tensor[i, :real_len])

        ragged_tensor = torch.cat(ragged_list, dim=0)
        return ragged_tensor

    def _create_paged_kv_cache(
        self,
        k_ragged: torch.Tensor,
        v_ragged: torch.Tensor,
        real_seq_lens: List[int],
        page_size: int,
        kv_heads: int,
        head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create paged KV cache from ragged K/V tensors

        Args:
            k_ragged: [total_tokens, kv_heads, head_dim]
            v_ragged: [total_tokens, kv_heads, head_dim]
            real_seq_lens: List of real sequence lengths
            page_size: Page size
            kv_heads: Number of KV heads
            head_dim: Head dimension

        Returns:
            paged_kv_cache: [total_pages, 2, page_size, kv_heads, head_dim]
            paged_kv_indptr: [batch_size + 1]
            paged_kv_indices: [total_pages]
            paged_kv_last_page_len: [batch_size]
        """
        batch_size = len(real_seq_lens)

        # Calculate total pages needed
        total_pages = sum(
            (seq_len + page_size - 1) // page_size for seq_len in real_seq_lens
        )

        # Create paged KV cache
        paged_kv_cache = torch.zeros(
            total_pages,
            2,  # K=0, V=1
            page_size,
            kv_heads,
            head_dim,
            dtype=k_ragged.dtype,
            device=self.device,
        )

        # Create index structures
        paged_kv_indptr: List[int] = [0]
        paged_kv_indices: List[int] = []
        paged_kv_last_page_len: List[int] = []

        current_page_id = 0
        k_offset = 0
        v_offset = 0

        for i in range(batch_size):
            seq_len = real_seq_lens[i]
            num_pages = (seq_len + page_size - 1) // page_size

            # Fill pages for this sequence
            for page_idx in range(num_pages):
                page_id = current_page_id + page_idx
                paged_kv_indices.append(page_id)

                # Calculate how many tokens go into this page
                tokens_in_page = min(page_size, seq_len - page_idx * page_size)

                # Fill K and V for this page
                k_start = k_offset + page_idx * page_size
                k_end = k_start + tokens_in_page
                paged_kv_cache[page_id, 0, :tokens_in_page] = k_ragged[k_start:k_end]

                v_start = v_offset + page_idx * page_size
                v_end = v_start + tokens_in_page
                paged_kv_cache[page_id, 1, :tokens_in_page] = v_ragged[v_start:v_end]

            current_page_id += num_pages
            k_offset += seq_len
            v_offset += seq_len

            # Record metadata
            paged_kv_indptr.append(current_page_id)
            last_page_len = seq_len % page_size
            if last_page_len == 0:
                last_page_len = page_size
            paged_kv_last_page_len.append(last_page_len)

        return (
            paged_kv_cache,
            torch.tensor(paged_kv_indptr, dtype=torch.int32, device=self.device),
            torch.tensor(paged_kv_indices, dtype=torch.int32, device=self.device),
            torch.tensor(paged_kv_last_page_len, dtype=torch.int32, device=self.device),
        )

    def _create_custom_mask_flattened(
        self,
        real_seq_lens: List[int],
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Create custom mask in flattened format for FlashInfer (ragged Q and KV)

        Args:
            real_seq_lens: List of real sequence lengths
            causal: Whether to apply causal mask

        Returns:
            mask_flat: [sum(seq_len * seq_len)] flattened mask
                      FlashInfer convention: True = valid, False = masked
        """
        batch_size = len(real_seq_lens)
        mask_list: List[torch.Tensor] = []

        for i in range(batch_size):
            seq_len = real_seq_lens[i]

            if causal:
                # Create causal mask: lower triangular
                # FlashInfer: True = valid, False = masked
                mask = torch.tril(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device)
                )
            else:
                # All-to-all mask
                mask = torch.ones(
                    seq_len, seq_len, dtype=torch.bool, device=self.device
                )

            mask_list.append(mask.flatten())

        return torch.cat(mask_list, dim=0)

    def _create_custom_mask_for_padded_q(
        self,
        real_seq_lens: List[int],
        max_seq_len: int,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Create custom mask for PADDED Q and RAGGED KV scenario

        Args:
            real_seq_lens: List of real sequence lengths
            max_seq_len: Maximum sequence length (Q is padded to this)
            causal: Whether to apply causal mask

        Returns:
            mask_flat: [sum(max_seq_len * real_seq_len)] flattened mask
                      FlashInfer convention: True = valid, False = masked
        """
        batch_size = len(real_seq_lens)
        mask_list: List[torch.Tensor] = []

        for i in range(batch_size):
            real_len = real_seq_lens[i]

            # Create mask: [max_seq_len, real_len]
            # Q is padded to max_seq_len, KV has real_len tokens
            mask = torch.zeros(
                max_seq_len, real_len, dtype=torch.bool, device=self.device
            )

            # Valid region: [0:real_len, 0:real_len]
            if causal:
                # Causal mask for valid tokens
                for j in range(real_len):
                    mask[j, : j + 1] = True
            else:
                # All-to-all for valid tokens
                mask[:real_len, :real_len] = True

            # Padding tokens (beyond real_len) remain False (masked out)

            mask_list.append(mask.flatten())

        return torch.cat(mask_list, dim=0)

    def _compute_reference_output(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        real_seq_lens: List[int],
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Compute reference attention output using PyTorch

        Args:
            q: [total_tokens, num_heads, head_dim]
            k: [total_tokens, kv_heads, head_dim]
            v: [total_tokens, kv_heads, head_dim]
            real_seq_lens: List of real sequence lengths
            causal: Whether to use causal mask

        Returns:
            output: [total_tokens, num_heads, head_dim]
        """
        batch_size = len(real_seq_lens)
        num_heads = q.shape[1]
        kv_heads = k.shape[1]
        head_dim = q.shape[2]

        # Handle GQA: repeat KV heads if needed
        if num_heads != kv_heads:
            num_groups = num_heads // kv_heads
            k = k.repeat_interleave(num_groups, dim=1)
            v = v.repeat_interleave(num_groups, dim=1)

        # Process each sequence separately
        outputs: List[torch.Tensor] = []
        offset = 0

        for i in range(batch_size):
            seq_len = real_seq_lens[i]

            # Extract sequence
            q_seq = q[offset : offset + seq_len]  # [seq_len, heads, dim]
            k_seq = k[offset : offset + seq_len]
            v_seq = v[offset : offset + seq_len]
            offset += seq_len

            # Compute attention: [heads, seq_len, dim]
            q_seq = q_seq.transpose(0, 1)
            k_seq = k_seq.transpose(0, 1)
            v_seq = v_seq.transpose(0, 1)

            # Scores: [heads, seq_len, seq_len]
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(q_seq, k_seq.transpose(-2, -1)) * scale

            # Apply mask
            if causal:
                mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
                scores = scores.masked_fill(mask.unsqueeze(0) == 0, float("-inf"))

            # Softmax
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

            # Apply to values: [heads, seq_len, dim]
            output_seq = torch.matmul(attn_weights, v_seq)
            output_seq = output_seq.transpose(0, 1)  # [seq_len, heads, dim]

            outputs.append(output_seq)

        return torch.cat(outputs, dim=0)

    def test_paged_with_custom_mask_single_batch(self):
        """Test with single batch, paged KV cache and custom mask (padded Q + ragged KV)"""
        real_seq_lens = [64]
        num_heads = 8
        kv_heads = 8
        head_dim = 64
        page_size = 16
        max_seq_len = max(real_seq_lens)
        batch_size = len(real_seq_lens)

        # Create config and attention op
        attn_configs = self._create_config(num_heads, kv_heads, head_dim)
        attn_op = PyFlashinferPrefillPagedAttnOp(attn_configs, page_size=page_size)

        # Create PADDED Q (typical for CUDA graph / batched inference)
        # Shape: [batch_size * max_seq_len, num_heads, head_dim]
        total_padded_tokens = batch_size * max_seq_len
        q_padded = torch.randn(
            total_padded_tokens,
            num_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )

        # Create RAGGED K, V (no padding, saves memory)
        # Shape: [total_real_tokens, kv_heads, head_dim]
        total_real_tokens = sum(real_seq_lens)
        k_ragged = torch.randn(
            total_real_tokens,
            kv_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )
        v_ragged = torch.randn(
            total_real_tokens,
            kv_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )

        # Create paged KV cache from ragged K, V
        paged_kv_cache, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = (
            self._create_paged_kv_cache(
                k_ragged, v_ragged, real_seq_lens, page_size, kv_heads, head_dim
            )
        )

        # Create attention inputs (padded Q format)
        # Custom mask will be automatically generated in prepare() based on input_lengths
        attn_inputs = self._create_attn_inputs(real_seq_lens, max_seq_len)

        # Prepare (custom mask is automatically generated based on attn_inputs)
        attn_op.prepare(
            attn_inputs,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        )

        # Forward with padded Q
        output = attn_op.forward(q_padded, paged_kv_cache, ParamsBase())

        # For reference computation, extract valid Q tokens from padded Q
        q_ragged_list: List[torch.Tensor] = []
        for i in range(batch_size):
            start_idx = i * max_seq_len
            real_len = real_seq_lens[i]
            q_ragged_list.append(q_padded[start_idx : start_idx + real_len])
        q_ragged_for_ref = torch.cat(q_ragged_list, dim=0)

        # Compute reference with ragged Q, K, V
        ref_output = self._compute_reference_output(
            q_ragged_for_ref, k_ragged, v_ragged, real_seq_lens, causal=True
        )

        # Extract valid output from padded output
        output_ragged_list: List[torch.Tensor] = []
        for i in range(batch_size):
            start_idx = i * max_seq_len
            real_len = real_seq_lens[i]
            output_ragged_list.append(output[start_idx : start_idx + real_len])
        output_ragged = torch.cat(output_ragged_list, dim=0)

        # Compare
        compare_tensors(
            output_ragged, ref_output, atol=1e-2, rtol=1e-2, name="Single batch"
        )

    def test_paged_with_custom_mask_multi_batch(self):
        """Test with multiple batches of different lengths, paged KV cache and custom mask (padded Q + ragged KV)"""
        real_seq_lens = [32, 48, 64]
        num_heads = 32
        kv_heads = 8  # GQA
        head_dim = 128
        page_size = 16
        max_seq_len = max(real_seq_lens)
        batch_size = len(real_seq_lens)

        # Create config and attention op
        attn_configs = self._create_config(num_heads, kv_heads, head_dim)
        attn_op = PyFlashinferPrefillPagedAttnOp(attn_configs, page_size=page_size)

        # Create PADDED Q
        total_padded_tokens = batch_size * max_seq_len
        q_padded = torch.randn(
            total_padded_tokens,
            num_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )

        # Create RAGGED K, V
        total_real_tokens = sum(real_seq_lens)
        k_ragged = torch.randn(
            total_real_tokens,
            kv_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )
        v_ragged = torch.randn(
            total_real_tokens,
            kv_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )

        # Create paged KV cache from ragged K, V
        paged_kv_cache, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = (
            self._create_paged_kv_cache(
                k_ragged, v_ragged, real_seq_lens, page_size, kv_heads, head_dim
            )
        )

        # Create attention inputs (padded Q format)
        # Custom mask will be automatically generated in prepare() based on input_lengths
        attn_inputs = self._create_attn_inputs(real_seq_lens, max_seq_len)

        # Prepare (custom mask is automatically generated based on attn_inputs)
        attn_op.prepare(
            attn_inputs,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        )

        # Forward with padded Q
        output = attn_op.forward(q_padded, paged_kv_cache, ParamsBase())

        # For reference computation, extract valid Q tokens from padded Q
        q_ragged_list: List[torch.Tensor] = []
        for i in range(batch_size):
            start_idx = i * max_seq_len
            real_len = real_seq_lens[i]
            q_ragged_list.append(q_padded[start_idx : start_idx + real_len])
        q_ragged_for_ref = torch.cat(q_ragged_list, dim=0)

        # Compute reference with ragged Q, K, V
        ref_output = self._compute_reference_output(
            q_ragged_for_ref, k_ragged, v_ragged, real_seq_lens, causal=True
        )

        # Extract valid output from padded output
        output_ragged_list: List[torch.Tensor] = []
        for i in range(batch_size):
            start_idx = i * max_seq_len
            real_len = real_seq_lens[i]
            output_ragged_list.append(output[start_idx : start_idx + real_len])
        output_ragged = torch.cat(output_ragged_list, dim=0)

        # Compare
        compare_tensors(
            output_ragged, ref_output, atol=1e-2, rtol=1e-2, name="Multi batch"
        )

    def test_paged_varied_sequence_lengths(self):
        """Test with highly varied sequence lengths (padded Q + ragged KV)"""
        real_seq_lens = [16, 32, 64, 128, 256]
        num_heads = 32
        kv_heads = 8
        head_dim = 128
        page_size = 32
        max_seq_len = max(real_seq_lens)
        batch_size = len(real_seq_lens)

        # Create config and attention op
        attn_configs = self._create_config(num_heads, kv_heads, head_dim)
        attn_op = PyFlashinferPrefillPagedAttnOp(attn_configs, page_size=page_size)

        # Create PADDED Q
        total_padded_tokens = batch_size * max_seq_len
        q_padded = torch.randn(
            total_padded_tokens,
            num_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )

        # Create RAGGED K, V
        total_real_tokens = sum(real_seq_lens)
        k_ragged = torch.randn(
            total_real_tokens,
            kv_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )
        v_ragged = torch.randn(
            total_real_tokens,
            kv_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )

        # Create paged KV cache from ragged K, V
        paged_kv_cache, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = (
            self._create_paged_kv_cache(
                k_ragged, v_ragged, real_seq_lens, page_size, kv_heads, head_dim
            )
        )

        # Create attention inputs (padded Q format)
        # Custom mask will be automatically generated in prepare() based on input_lengths
        attn_inputs = self._create_attn_inputs(real_seq_lens, max_seq_len)

        # Prepare (custom mask is automatically generated based on attn_inputs)
        attn_op.prepare(
            attn_inputs,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        )

        # Forward with padded Q
        output = attn_op.forward(q_padded, paged_kv_cache, ParamsBase())

        # For reference computation, extract valid Q tokens from padded Q
        q_ragged_list: List[torch.Tensor] = []
        for i in range(batch_size):
            start_idx = i * max_seq_len
            real_len = real_seq_lens[i]
            q_ragged_list.append(q_padded[start_idx : start_idx + real_len])
        q_ragged_for_ref = torch.cat(q_ragged_list, dim=0)

        # Compute reference with ragged Q, K, V
        ref_output = self._compute_reference_output(
            q_ragged_for_ref, k_ragged, v_ragged, real_seq_lens, causal=True
        )

        # Extract valid output from padded output
        output_ragged_list: List[torch.Tensor] = []
        for i in range(batch_size):
            start_idx = i * max_seq_len
            real_len = real_seq_lens[i]
            output_ragged_list.append(output[start_idx : start_idx + real_len])
        output_ragged = torch.cat(output_ragged_list, dim=0)

        # Compare
        compare_tensors(
            output_ragged, ref_output, atol=1e-2, rtol=1e-2, name="Varied lengths"
        )


if __name__ == "__main__":
    unittest.main()
