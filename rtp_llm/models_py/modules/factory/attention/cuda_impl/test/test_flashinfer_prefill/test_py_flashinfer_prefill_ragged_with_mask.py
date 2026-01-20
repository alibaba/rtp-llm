import logging
import unittest
from typing import List

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPrefillAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.attention_ref import (
    compute_flashinfer_prefill_reference,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    compare_tensors,
    set_seed,
)
from rtp_llm.ops import AttentionConfigs

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TestPyFlashinferPrefillWithMask(unittest.TestCase):
    """Test PyFlashinferPrefillAttnOp with custom mask (padded Q + ragged K/V)"""

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
        Helper to create PyAttentionInputs for CUDA graph padded mode

        Args:
            seq_lens: List of real sequence lengths
            max_seq_len: Maximum sequence length (padding target)

        Returns:
            PyAttentionInputs configured for CUDA graph padded mode
        """
        from rtp_llm.ops.compute_ops import (
            PyAttentionInputs,
            PyPrefillCudaGaphCopyParams,
        )

        batch_size = len(seq_lens)

        # Create prefill_cuda_graph_copy_params to indicate CUDA graph padded mode
        prefill_params = PyPrefillCudaGaphCopyParams()
        prefill_params.max_seq_len = max_seq_len
        prefill_params.max_batch_size = batch_size

        # Create PyAttentionInputs
        attn_inputs = PyAttentionInputs()
        attn_inputs.prefill_cuda_graph_copy_params = prefill_params
        attn_inputs.input_lengths = torch.tensor(
            seq_lens, dtype=torch.int32, device="cpu"
        )
        attn_inputs.cu_seqlens = torch.tensor(
            [i * max_seq_len for i in range(batch_size + 1)],
            dtype=torch.int32,
            device=self.device,
        )

        return attn_inputs

    def _create_padded_q_ragged_kv_and_mask(
        self,
        batch_size: int,
        seq_lens: List[int],
        num_heads: int,
        kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Create PADDED Q tensors and RAGGED K/V tensors with custom mask

        This tests the realistic scenario where:
        1. Q is padded to max_seq_len (common in batched inference)
        2. K, V are ragged (not padded, saves memory)
        3. Custom mask masks out Q's padding tokens

        Returns:
            q: [total_padded_tokens, num_heads, head_dim] - PADDED
            k: [total_real_tokens, kv_heads, head_dim] - RAGGED (no padding)
            v: [total_real_tokens, kv_heads, head_dim] - RAGGED (no padding)
            custom_mask: Flattened mask for FlashInfer (masks padding + causal)
            qo_indptr: [batch_size + 1] cumulative PADDED lengths for Q
            kv_indptr: [batch_size + 1] cumulative REAL lengths for K/V
        """
        max_seq_len = max(seq_lens)
        total_padded_tokens = batch_size * max_seq_len
        total_real_tokens = sum(seq_lens)

        # Create PADDED Q tensors (batch_size, max_seq_len, heads, head_dim)
        q_padded = torch.randn(
            batch_size,
            max_seq_len,
            num_heads,
            head_dim,
            dtype=dtype,
            device=self.device,
        )
        # Flatten to (total_padded_tokens, heads, head_dim)
        q = q_padded.reshape(total_padded_tokens, num_heads, head_dim)

        # Create RAGGED K, V tensors (no padding!)
        k = torch.randn(
            total_real_tokens, kv_heads, head_dim, dtype=dtype, device=self.device
        )
        v = torch.randn(
            total_real_tokens, kv_heads, head_dim, dtype=dtype, device=self.device
        )

        # Create qo_indptr for PADDED Q (each sequence occupies max_seq_len slots)
        qo_indptr = torch.tensor(
            [i * max_seq_len for i in range(batch_size + 1)],
            dtype=torch.int32,
            device=self.device,
        )

        # Create kv_indptr for RAGGED K/V (each sequence uses only real length)
        kv_indptr = torch.tensor(
            [0] + [sum(seq_lens[: i + 1]) for i in range(batch_size)],
            dtype=torch.int32,
            device=self.device,
        )

        # Create custom mask that:
        # 1. Q tokens can only attend to valid (non-padding) K/V tokens
        # 2. Applies causal masking within valid tokens
        mask_list = []
        for seq_len in seq_lens:
            # Mask shape: [max_seq_len (Q), seq_len (K/V)]
            # Q is padded to max_seq_len, K/V only has seq_len real tokens
            mask = torch.zeros(
                max_seq_len, seq_len, dtype=torch.bool, device=self.device
            )

            # Set valid region with causal mask
            # For Q token i (if i < seq_len), can attend to K/V tokens [0:i+1]
            for i in range(seq_len):
                mask[i, : i + 1] = True  # Q token i can attend to K/V tokens 0...i

            # Q tokens beyond seq_len (padding) cannot attend to anything (all False)

            # Flatten and add to list
            mask_list.append(mask.flatten())

        custom_mask = torch.cat(mask_list, dim=0)

        # Also save K/V in padded format for reference computation
        k_padded = torch.zeros(
            batch_size, max_seq_len, kv_heads, head_dim, dtype=dtype, device=self.device
        )
        v_padded = torch.zeros(
            batch_size, max_seq_len, kv_heads, head_dim, dtype=dtype, device=self.device
        )

        offset = 0
        for i, seq_len in enumerate(seq_lens):
            k_padded[i, :seq_len] = k[offset : offset + seq_len]
            v_padded[i, :seq_len] = v[offset : offset + seq_len]
            offset += seq_len

        return (
            q,
            k,
            v,
            custom_mask,
            qo_indptr,
            kv_indptr,
            q_padded,
            k_padded,
            v_padded,
            max_seq_len,
        )

    def _compute_reference_with_padded_mask(
        self,
        q_padded: torch.Tensor,
        k_padded: torch.Tensor,
        v_padded: torch.Tensor,
        seq_lens: List[int],
        max_seq_len: int,
    ) -> torch.Tensor:
        """
        Compute reference output for padded input with causal masking
        Only computes on valid (non-padding) regions

        Args:
            q_padded: [batch_size, max_seq_len, num_heads, head_dim]
            k_padded: [batch_size, max_seq_len, kv_heads, head_dim]
            v_padded: [batch_size, max_seq_len, kv_heads, head_dim]
            seq_lens: List of real sequence lengths
            max_seq_len: Maximum sequence length (padding length)

        Returns:
            output: [batch_size * max_seq_len, num_heads, head_dim] (flattened, with padding)
        """
        batch_size = len(seq_lens)
        num_heads = q_padded.shape[2]
        head_dim = q_padded.shape[3]

        # Create output tensor (padded)
        output_padded = torch.zeros_like(q_padded)

        for i in range(batch_size):
            seq_len = seq_lens[i]

            # Extract valid region (no padding)
            q_i = q_padded[i, :seq_len]  # [seq_len, num_heads, head_dim]
            k_i = k_padded[i, :seq_len]  # [seq_len, kv_heads, head_dim]
            v_i = v_padded[i, :seq_len]  # [seq_len, kv_heads, head_dim]

            # Create cu_seqlens for single sequence
            cu_seqlens = torch.tensor(
                [0, seq_len], dtype=torch.int32, device=self.device
            )

            # Compute attention on valid region with causal masking
            output_i = compute_flashinfer_prefill_reference(
                q_i,  # [seq_len, num_heads, head_dim]
                k_i,  # [seq_len, kv_heads, head_dim]
                v_i,  # [seq_len, kv_heads, head_dim]
                cu_seqlens,  # [2]
                causal=True,
            )

            # Place in output (padding region remains 0)
            output_padded[i, :seq_len] = output_i

        # Flatten to match FlashInfer output format
        return output_padded.reshape(batch_size * max_seq_len, num_heads, head_dim)

    def test_padded_q_with_mask_single_batch(self):
        """Test PADDED Q with automatic custom mask generation - single sequence"""
        logging.info(
            "\n=== Testing PADDED Q with auto mask via PyFlashinferPrefillAttnOp (single batch) ==="
        )

        batch_size = 1
        seq_lens = [64]  # Real length is 64, but will pad to 128 for testing
        max_seq_len = 128  # Pad to this length
        num_heads = 32
        kv_heads = 8
        head_dim = 128

        # Create PADDED Q (simulating real-world batched inference)
        q_padded = torch.randn(
            batch_size,
            max_seq_len,
            num_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )
        k_padded = torch.randn(
            batch_size,
            max_seq_len,
            kv_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )
        v_padded = torch.randn(
            batch_size,
            max_seq_len,
            kv_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )

        # Flatten for FlashInfer
        q = q_padded.reshape(batch_size * max_seq_len, num_heads, head_dim)
        k = k_padded.reshape(batch_size * max_seq_len, kv_heads, head_dim)
        v = v_padded.reshape(batch_size * max_seq_len, kv_heads, head_dim)

        logging.info(f"Real seq_len: {seq_lens[0]}, Q padded to: {max_seq_len}")
        logging.info(f"Q shape: {q.shape} (PADDED)")

        # Create PyAttentionInputs for CUDA graph padded mode
        attn_inputs = self._create_attn_inputs(seq_lens, max_seq_len)

        # Create PyFlashinferPrefillAttnOp
        attn_configs = self._create_config(
            head_num=num_heads,
            head_num_kv=kv_heads,
            size_per_head=head_dim,
        )
        attn_op = PyFlashinferPrefillAttnOp(attn_configs, backend="fa2")

        # Prepare - will automatically build custom mask from is_s_padded and input_lengths
        attn_op.prepare(attn_inputs)

        # Run attention
        output = attn_op.prefill_wrapper.run(q, k, v)

        # Compute reference output (only on valid regions)
        ref_output = self._compute_reference_with_padded_mask(
            q_padded, k_padded, v_padded, seq_lens, max_seq_len
        )

        # Compare outputs
        compare_tensors(
            output, ref_output, atol=1e-2, rtol=1e-2, name="Padded Q with auto mask"
        )

        logging.info("✓ Single batch test passed (auto mask from is_s_padded)")

    def test_padded_q_with_mask_multi_batch(self):
        """Test PADDED Q with automatic custom mask generation - multiple sequences"""
        logging.info(
            "\n=== Testing PADDED Q with auto mask via PyFlashinferPrefillAttnOp (multi-batch) ==="
        )

        batch_size = 4
        seq_lens = [32, 64, 128, 96]  # Different REAL lengths
        max_seq_len = max(seq_lens)
        num_heads = 32
        kv_heads = 8
        head_dim = 128

        logging.info(f"Real sequence lengths: {seq_lens}, Q will pad to {max_seq_len}")
        logging.info(f"Padding amounts: {[max_seq_len - sl for sl in seq_lens]}")

        # Create PADDED Q, K, V
        q_padded = torch.randn(
            batch_size,
            max_seq_len,
            num_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )
        k_padded = torch.randn(
            batch_size,
            max_seq_len,
            kv_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )
        v_padded = torch.randn(
            batch_size,
            max_seq_len,
            kv_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )

        # Flatten
        q = q_padded.reshape(batch_size * max_seq_len, num_heads, head_dim)
        k = k_padded.reshape(batch_size * max_seq_len, kv_heads, head_dim)
        v = v_padded.reshape(batch_size * max_seq_len, kv_heads, head_dim)

        logging.info(f"Q shape: {q.shape} (PADDED)")

        # Create PyAttentionInputs for CUDA graph padded mode
        attn_inputs = self._create_attn_inputs(seq_lens, max_seq_len)

        # Create PyFlashinferPrefillAttnOp
        attn_configs = self._create_config(
            head_num=num_heads,
            head_num_kv=kv_heads,
            size_per_head=head_dim,
        )
        attn_op = PyFlashinferPrefillAttnOp(attn_configs, backend="fa2")

        # Prepare - auto build custom mask
        attn_op.prepare(attn_inputs)

        # Run attention
        output = attn_op.prefill_wrapper.run(q, k, v)

        # Compute reference
        ref_output = self._compute_reference_with_padded_mask(
            q_padded, k_padded, v_padded, seq_lens, max_seq_len
        )

        # Compare
        compare_tensors(
            output,
            ref_output,
            atol=1e-2,
            rtol=1e-2,
            name="Multi-batch Q padded with auto mask",
        )

        logging.info("✓ Multi-batch test passed (auto mask from is_s_padded)")

    def test_padded_q_with_mask_varied_lengths(self):
        """Test PADDED Q with automatic custom mask - highly varied sequence lengths"""
        logging.info(
            "\n=== Testing PADDED Q with auto mask via PyFlashinferPrefillAttnOp (varied lengths) ==="
        )

        batch_size = 5
        seq_lens = [16, 48, 128, 64, 200]  # Highly variable real lengths
        max_seq_len = max(seq_lens)
        num_heads = 32
        kv_heads = 8
        head_dim = 128

        logging.info(f"Real sequence lengths: {seq_lens}, Q pads to {max_seq_len}")
        logging.info(
            f"Q padding waste: {[f'{(max_seq_len-sl)/max_seq_len:.1%}' for sl in seq_lens]}"
        )

        # Create PADDED Q, K, V
        q_padded = torch.randn(
            batch_size,
            max_seq_len,
            num_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )
        k_padded = torch.randn(
            batch_size,
            max_seq_len,
            kv_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )
        v_padded = torch.randn(
            batch_size,
            max_seq_len,
            kv_heads,
            head_dim,
            dtype=torch.float16,
            device=self.device,
        )

        # Flatten
        q = q_padded.reshape(batch_size * max_seq_len, num_heads, head_dim)
        k = k_padded.reshape(batch_size * max_seq_len, kv_heads, head_dim)
        v = v_padded.reshape(batch_size * max_seq_len, kv_heads, head_dim)

        # Create PyAttentionInputs for CUDA graph padded mode
        attn_inputs = self._create_attn_inputs(seq_lens, max_seq_len)

        # Create PyFlashinferPrefillAttnOp
        attn_configs = self._create_config(
            head_num=num_heads,
            head_num_kv=kv_heads,
            size_per_head=head_dim,
        )
        attn_op = PyFlashinferPrefillAttnOp(attn_configs, backend="fa2")

        # Prepare - auto build custom mask
        attn_op.prepare(attn_inputs)

        # Run attention
        output = attn_op.prefill_wrapper.run(q, k, v)

        # Compute reference
        ref_output = self._compute_reference_with_padded_mask(
            q_padded, k_padded, v_padded, seq_lens, max_seq_len
        )

        # Compare
        compare_tensors(
            output,
            ref_output,
            atol=1e-2,
            rtol=1e-2,
            name="Varied lengths Q padded with auto mask",
        )

        logging.info(
            "✓ Varied lengths test passed (auto mask handles large padding correctly)"
        )


if __name__ == "__main__":
    unittest.main()
