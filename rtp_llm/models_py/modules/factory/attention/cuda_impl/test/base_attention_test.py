import logging
import math
import unittest
from typing import List, NamedTuple

import torch

from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compare_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    name: str = "tensor",
):
    """Compare two tensors and assert they are close"""
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        diff = torch.abs(a - b)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        logging.error(f"{name} comparison failed!")
        logging.error(f"  Max diff: {max_diff}")
        logging.error(f"  Mean diff: {mean_diff}")
        logging.error(f"  rtol: {rtol}, atol: {atol}")
        raise AssertionError(
            f"{name} mismatch: max_diff={max_diff}, mean_diff={mean_diff}"
        )
    else:
        logging.info(f"{name} comparison passed ✓")


class TestConfig(NamedTuple):
    """Configuration container for attention tests"""

    attn_configs: AttentionConfigs
    parallelism_config: ParallelismConfig
    head_num: int
    head_num_kv: int
    size_per_head: int
    seq_size_per_block: int
    tp_size: int


class BaseAttentionTest(unittest.TestCase):
    """Base test class for attention decode operations with common helper functions"""

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
        seq_size_per_block: int = 64,
        tp_size: int = 1,
        data_type: str = "fp16",
    ) -> TestConfig:
        """Helper to create a test config"""
        attn_configs = AttentionConfigs()
        attn_configs.head_num = head_num
        attn_configs.kv_head_num = head_num_kv
        attn_configs.size_per_head = size_per_head
        attn_configs.tokens_per_block = seq_size_per_block
        attn_configs.use_mla = False
        # Set dtype based on data_type parameter
        dtype_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }
        attn_configs.dtype = dtype_map.get(data_type, torch.float16)

        parallelism_config = ParallelismConfig()
        parallelism_config.tp_size = tp_size

        return TestConfig(
            attn_configs=attn_configs,
            parallelism_config=parallelism_config,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            tp_size=tp_size,
        )

    def _create_attention_inputs_base(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        seq_size_per_block: int,
    ) -> PyAttentionInputs:
        """Helper to create PyAttentionInputs for decode

        This is the base implementation that can be customized by subclasses.

        Note: sequence_lengths here represent the current KV cache length (including current token).
        """
        attn_inputs = PyAttentionInputs()

        # Decode mode
        attn_inputs.is_prefill = False
        attn_inputs.sequence_lengths = (
            torch.tensor(sequence_lengths, dtype=torch.int32, device="cpu") - 1
        ).pin_memory()

        # Input lengths for decode are all 1 (generating one token per sequence)
        attn_inputs.input_lengths = torch.ones(
            batch_size, dtype=torch.int32, device="cpu"
        )

        # Use empty tensor for prefix_lengths to trigger decode branch
        attn_inputs.prefix_lengths = torch.empty(0, dtype=torch.int32, device="cpu")

        # Create KV cache block IDs
        max_blocks = max(
            [math.ceil(seq_len / seq_size_per_block) for seq_len in sequence_lengths]
        )
        kv_cache_block_id = torch.zeros(
            [batch_size, max_blocks], dtype=torch.int32, device="cpu"
        )

        # Fill block IDs sequentially for each batch
        block_offset = 0
        for i, seq_len in enumerate(sequence_lengths):
            num_blocks = math.ceil(seq_len / seq_size_per_block)
            kv_cache_block_id[i, :num_blocks] = torch.arange(
                block_offset, block_offset + num_blocks, dtype=torch.int32
            )
            block_offset += num_blocks

        attn_inputs.kv_cache_block_id_host = kv_cache_block_id
        attn_inputs.kv_cache_block_id_device = kv_cache_block_id.to(self.device)

        # Create cu_seqlens for decode (just counting tokens)
        attn_inputs.cu_seqlens = torch.arange(
            0, batch_size + 1, dtype=torch.int32, device=self.device
        )

        return attn_inputs

    def _create_kv_cache(
        self,
        total_blocks: int,
        seq_size_per_block: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ):
        """Helper to create KV cache

        Note: For HND layout, kv_cache_base should be a 5D tensor:
        [total_blocks, 2, num_kv_heads, seq_size_per_block, head_dim]
        where dimension 1 index 0 is K cache and index 1 is V cache.
        """
        kv_cache = KVCache()

        # Create combined KV cache with shape [total_blocks, 2, num_kv_heads, seq_size_per_block, head_dim]
        # where dim=1, index=0 is K and index=1 is V
        kv_cache_combined = torch.randn(
            total_blocks,
            2,  # K and V
            num_kv_heads,
            seq_size_per_block,
            head_dim,
            dtype=dtype,
            device=self.device,
        )

        kv_cache.kv_cache_base = kv_cache_combined

        # Extract separate K and V for reference computation
        k_cache = kv_cache_combined[
            :, 0, :, :, :
        ]  # [total_blocks, num_kv_heads, seq_size_per_block, head_dim]
        v_cache = kv_cache_combined[
            :, 1, :, :, :
        ]  # [total_blocks, num_kv_heads, seq_size_per_block, head_dim]

        return kv_cache, k_cache, v_cache

    def _create_query_tensor(
        self,
        batch_size: int,
        head_num: int,
        size_per_head: int,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Helper to create query tensor"""
        return torch.randn(
            batch_size,
            head_num,
            size_per_head,
            dtype=dtype,
            device=self.device,
        )

    def _calculate_total_blocks(
        self,
        sequence_lengths: List[int],
        seq_size_per_block: int,
    ) -> int:
        """Helper to calculate total number of blocks needed"""
        return sum(
            [math.ceil(seq_len / seq_size_per_block) for seq_len in sequence_lengths]
        )

    def _generate_block_id_list(
        self,
        attn_inputs: PyAttentionInputs,
        sequence_lengths: List[int],
        seq_size_per_block: int,
    ) -> List[List[int]]:
        """Generate block ID list from attention inputs for reference computation"""
        block_id_list = []
        for i, seq_len in enumerate(sequence_lengths):
            num_blocks = math.ceil(seq_len / seq_size_per_block)
            block_ids = attn_inputs.kv_cache_block_id_host[i, :num_blocks].tolist()
            block_id_list.append(block_ids)
        return block_id_list
