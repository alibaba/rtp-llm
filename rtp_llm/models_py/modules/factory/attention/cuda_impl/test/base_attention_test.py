import logging
import math
import unittest
from typing import List, NamedTuple, Optional, Sequence

import torch

from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, get_typemeta
from rtp_llm.test.utils.numeric_util import assert_close_with_mismatch_tolerance

logging.basicConfig(level=logging.INFO, format="%(message)s")

FP8_CACHE_DTYPES = (torch.float8_e4m3fn,)


def make_fp8_unit_scale(
    total_pages: int,
    num_kv_heads: int,
    page_size: int,
    device: torch.device,
) -> torch.Tensor:
    """All-ones cache scale, matching the kv_scale == 1.0 FP8 contract."""
    return torch.ones(
        total_pages,
        2 * num_kv_heads * page_size,
        dtype=torch.float32,
        device=device,
    )


def fill_paged_kv_cache(
    k: Sequence[torch.Tensor],
    v: Sequence[torch.Tensor],
    fill_lengths: Sequence[int],
    block_table: torch.Tensor,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    cache_dtype: torch.dtype,
    device: torch.device,
    total_pages: Optional[int] = None,
) -> LayerKVCache:
    """Scatter per-batch K/V into a paged HND KV cache following ``block_table``.

    Args:
        k: Per-batch tensors [length, num_kv_heads, head_dim]; only the first
            ``fill_lengths[i]`` tokens of each are written.
        v: Value counterpart of ``k``.
        fill_lengths: Tokens to write per batch. Use the prefix length when the
            op under test appends the new tokens itself, or the full sequence
            length to populate the whole cache.
        block_table: [batch, max_pages] page ids, as carried by
            ``kv_cache_kernel_block_id``.
        cache_dtype: Cache element dtype; FP8 dtypes also get a unit scale.
        total_pages: Cache capacity in pages. Defaults to the pages required by
            ``fill_lengths``; pass explicitly when the block table indexes into
            a larger or sparsely used page pool.

    Returns:
        LayerKVCache whose kv_cache_base is
        [total_pages, 2, num_kv_heads, page_size, head_dim], index 0 of dim 1
        being K and index 1 being V.
    """
    if total_pages is None:
        total_pages = sum(math.ceil(length / page_size) for length in fill_lengths)

    paged_kv_cache = torch.zeros(
        total_pages,
        2,
        num_kv_heads,
        page_size,
        head_dim,
        dtype=cache_dtype,
        device=device,
    )
    for batch_idx, fill_len in enumerate(fill_lengths):
        for page_offset in range(math.ceil(fill_len / page_size)):
            page_id = int(block_table[batch_idx, page_offset].item())
            start = page_offset * page_size
            end = min(start + page_size, fill_len)
            # [num_tokens, H, D] -> [H, num_tokens, D]
            paged_kv_cache[page_id, 0, :, : end - start, :] = k[batch_idx][
                start:end
            ].transpose(0, 1)
            paged_kv_cache[page_id, 1, :, : end - start, :] = v[batch_idx][
                start:end
            ].transpose(0, 1)

    kv_cache = LayerKVCache()
    kv_cache.kv_cache_base = paged_kv_cache
    if cache_dtype in FP8_CACHE_DTYPES:
        kv_cache.kv_scale_base = make_fp8_unit_scale(
            total_pages, num_kv_heads, page_size, device
        )
    return kv_cache


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

    kv_cache_dtype = KvCacheDataType.BASE
    rtol = 1e-2
    atol = 1e-2
    max_mismatch_rate = 0.0

    @staticmethod
    def cache_dtype(attn_configs: AttentionConfigs) -> torch.dtype:
        return (
            torch.float8_e4m3fn
            if attn_configs.kv_cache_dtype == KvCacheDataType.FP8
            else attn_configs.dtype
        )

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
        attn_configs.kernel_tokens_per_block = seq_size_per_block
        attn_configs.use_mla = False

        # Set dtype based on data_type parameter
        dtype_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }
        attn_configs.dtype = dtype_map.get(data_type, torch.float16)
        attn_configs.kv_cache_dtype = self.kv_cache_dtype

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

    def _assert_output_close(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        name: str,
        *,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> None:
        compare_rtol = self.rtol if rtol is None else rtol
        compare_atol = self.atol if atol is None else atol
        if self.max_mismatch_rate > 0:
            assert_close_with_mismatch_tolerance(
                actual,
                expected,
                rtol=compare_rtol,
                atol=compare_atol,
                max_mismatched_elements=math.ceil(
                    self.max_mismatch_rate * expected.numel()
                ),
            )
        else:
            compare_tensors(
                actual,
                expected,
                rtol=compare_rtol,
                atol=compare_atol,
                name=name,
            )

    def _create_kv_cache_block_ids(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        seq_size_per_block: int,
    ) -> torch.Tensor:
        """Helper to create KV cache block IDs

        Args:
            batch_size: Number of sequences in the batch
            sequence_lengths: List of sequence lengths for each batch item
            seq_size_per_block: Number of tokens per block (page size)

        Returns:
            Tensor of shape [batch_size, max_blocks] with block IDs
        """
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

        return kv_cache_block_id

    def _create_attention_inputs_base(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        seq_size_per_block: int,
        dtype: torch.dtype = torch.float16,
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

        # Create KV cache block IDs using the extracted helper
        kv_cache_block_id = self._create_kv_cache_block_ids(
            batch_size, sequence_lengths, seq_size_per_block
        )
        attn_inputs.kv_cache_block_id = kv_cache_block_id
        attn_inputs.kv_cache_block_id_device = kv_cache_block_id.to(self.device)
        attn_inputs.kv_cache_kernel_block_id = kv_cache_block_id
        attn_inputs.kv_cache_kernel_block_id_device = kv_cache_block_id.to(self.device)

        # Create cu_seqlens for decode (just counting tokens)
        attn_inputs.cu_seqlens_device = torch.arange(
            0, batch_size + 1, dtype=torch.int32, device=self.device
        )

        # Set dtype using get_typemeta
        attn_inputs.dtype = get_typemeta(torch.zeros([1], dtype=dtype))

        return attn_inputs

    def _create_prefill_attention_inputs(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        seq_size_per_block: int,
        dtype: torch.dtype = torch.float16,
        with_kv_cache_block_ids: bool = True,
    ) -> PyAttentionInputs:
        """Helper to create PyAttentionInputs for prefill mode

        Args:
            batch_size: Number of sequences in the batch
            sequence_lengths: List of sequence lengths for each batch item
            seq_size_per_block: Number of tokens per block (page size)
            dtype: Data type for attention computation (default: torch.float16)
            with_kv_cache_block_ids: Whether to populate paged KV cache block IDs

        Returns:
            PyAttentionInputs configured for prefill mode
        """
        attn_inputs = PyAttentionInputs()

        # Prefill mode
        attn_inputs.is_prefill = True

        # input_lengths is the length of each sequence in the batch
        attn_inputs.input_lengths = torch.tensor(
            sequence_lengths, dtype=torch.int32, device="cpu"
        ).pin_memory()

        # sequence_lengths for prefill is same as input_lengths
        attn_inputs.sequence_lengths = torch.tensor(
            sequence_lengths, dtype=torch.int32, device="cpu"
        ).pin_memory()

        # prefix_lengths is all zeros for pure prefill (no prefix caching)
        attn_inputs.prefix_lengths = torch.zeros(
            batch_size, dtype=torch.int32, device="cpu"
        )

        if with_kv_cache_block_ids:
            kv_cache_block_id = self._create_kv_cache_block_ids(
                batch_size, sequence_lengths, seq_size_per_block
            )
            attn_inputs.kv_cache_block_id = kv_cache_block_id
            attn_inputs.kv_cache_block_id_device = kv_cache_block_id.to(self.device)
            attn_inputs.kv_cache_kernel_block_id = kv_cache_block_id
            attn_inputs.kv_cache_kernel_block_id_device = kv_cache_block_id.to(
                self.device
            )

        # Create cu_seqlens (cumulative sequence lengths) for ragged tensor
        cu_seqlens = [0]
        for seq_len in sequence_lengths:
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
        attn_inputs.cu_seqlens_device = torch.tensor(
            cu_seqlens, dtype=torch.int32, device=self.device
        )

        # Set dtype using get_typemeta
        attn_inputs.dtype = get_typemeta(torch.zeros([1], dtype=dtype))

        return attn_inputs

    def _create_chunked_prefill_attention_inputs(
        self,
        batch_size: int,
        prefix_lengths: List[int],
        input_lengths: List[int],
        seq_size_per_block: int,
        dtype: torch.dtype = torch.float16,
    ) -> PyAttentionInputs:
        """Create PyAttentionInputs for chunked prefill: new Q tokens on top of
        an existing KV prefix; cu_seqlens accumulates input lengths only."""
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.is_cuda_graph = False
        attn_inputs.input_lengths = torch.tensor(
            input_lengths, dtype=torch.int32, device="cpu"
        ).pin_memory()
        attn_inputs.prefix_lengths = torch.tensor(
            prefix_lengths, dtype=torch.int32, device="cpu"
        ).pin_memory()
        sequence_lengths = [p + i for p, i in zip(prefix_lengths, input_lengths)]
        attn_inputs.sequence_lengths = torch.tensor(
            sequence_lengths, dtype=torch.int32, device="cpu"
        ).pin_memory()

        kv_cache_block_id = self._create_kv_cache_block_ids(
            batch_size, sequence_lengths, seq_size_per_block
        )
        attn_inputs.kv_cache_block_id = kv_cache_block_id
        attn_inputs.kv_cache_block_id_device = kv_cache_block_id.to(self.device)
        attn_inputs.kv_cache_kernel_block_id = kv_cache_block_id
        attn_inputs.kv_cache_kernel_block_id_device = kv_cache_block_id.to(self.device)

        cu_seqlens = [0]
        for input_len in input_lengths:
            cu_seqlens.append(cu_seqlens[-1] + input_len)
        attn_inputs.cu_seqlens = torch.tensor(
            cu_seqlens, dtype=torch.int32, device="cpu"
        ).pin_memory()
        attn_inputs.cu_seqlens_device = attn_inputs.cu_seqlens.to(
            self.device, non_blocking=True
        )
        attn_inputs.dtype = get_typemeta(torch.zeros([1], dtype=dtype))
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
        FP8 caches also include the scale buffer required by fused cache writes.
        """
        kv_cache = LayerKVCache()

        # Create combined KV cache with shape [total_blocks, 2, num_kv_heads, seq_size_per_block, head_dim]
        # where dim=1, index=0 is K and index=1 is V
        is_fp8 = dtype in FP8_CACHE_DTYPES
        kv_cache_combined = torch.randn(
            total_blocks,
            2,  # K and V
            num_kv_heads,
            seq_size_per_block,
            head_dim,
            dtype=torch.bfloat16 if is_fp8 else dtype,
            device=self.device,
        )
        if is_fp8:
            kv_cache_combined = kv_cache_combined.to(dtype)

        kv_cache.kv_cache_base = kv_cache_combined
        if is_fp8:
            kv_cache.kv_scale_base = make_fp8_unit_scale(
                total_blocks, num_kv_heads, seq_size_per_block, self.device
            )

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
            block_ids = attn_inputs.kv_cache_block_id[i, :num_blocks].tolist()
            block_id_list.append(block_ids)
        return block_id_list
