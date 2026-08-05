import logging
import math
import unittest
from typing import List, NamedTuple, Optional

import torch

from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig, RopeStyle
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, get_typemeta

logging.basicConfig(level=logging.INFO, format="%(message)s")


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
        rope_style: Optional[RopeStyle] = None,
        kv_cache_dtype: Optional[KvCacheDataType] = None,
        need_rope_kv_cache: Optional[bool] = None,
        is_causal: Optional[bool] = None,
        max_seq_len: Optional[int] = None,
    ) -> TestConfig:
        """Helper to create a test config

        Optional arguments are only applied when supplied, so callers that omit
        them keep the plain attention config.
        """
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

        if max_seq_len is not None:
            attn_configs.max_seq_len = max_seq_len
        if rope_style is not None:
            attn_configs.rope_config.style = rope_style
            attn_configs.rope_config.dim = (
                size_per_head if rope_style != RopeStyle.No else 0
            )
            attn_configs.rope_config.base = 10000
            attn_configs.rope_config.max_pos = max_seq_len or 2048
        if kv_cache_dtype is not None:
            attn_configs.kv_cache_dtype = kv_cache_dtype
        if need_rope_kv_cache is not None:
            attn_configs.need_rope_kv_cache = need_rope_kv_cache
        if is_causal is not None:
            attn_configs.is_causal = is_causal

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
        input_lengths: List[int],
        seq_size_per_block: int,
        dtype: torch.dtype = torch.float16,
        with_kv_cache_block_ids: bool = True,
        prefix_lengths: Optional[List[int]] = None,
        empty_prefix: bool = False,
        block_ids: Optional[torch.Tensor] = None,
        is_cuda_graph: bool = False,
        graph_step: Optional[int] = None,
    ) -> PyAttentionInputs:
        """Helper to create PyAttentionInputs for prefill mode

        Args:
            batch_size: Number of sequences in the batch
            input_lengths: Number of new tokens per batch item
            seq_size_per_block: Number of tokens per block (page size)
            dtype: Data type for attention computation (default: torch.float16)
            with_kv_cache_block_ids: Whether to populate paged KV cache block IDs
            prefix_lengths: Reused prefix length per batch item (default: zeros)
            empty_prefix: Emit no prefix vector at all, as encoder-only requests do
            block_ids: Explicit block table, instead of a sequential one
            is_cuda_graph: Mark the inputs as belonging to a CUDA graph batch
            graph_step: Capture-time padded row stride. Defaults to the local
                maximum for eager fixtures; replay fixtures should pass the
                selected graph's immutable capture step explicitly.

        Returns:
            PyAttentionInputs configured for prefill mode
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(input_lengths) != batch_size:
            raise ValueError(
                "input_lengths must contain one entry per batch item "
                f"(batch_size={batch_size}, len={len(input_lengths)})"
            )
        if any(length < 0 for length in input_lengths):
            raise ValueError(f"input_lengths must be non-negative, got {input_lengths}")
        if not any(length > 0 for length in input_lengths):
            raise ValueError(
                f"input_lengths must contain an active request, got {input_lengths}"
            )
        if not is_cuda_graph and any(length == 0 for length in input_lengths):
            raise ValueError(
                "zero input_lengths are reserved for padded CUDA graph slots, "
                f"got {input_lengths}"
            )
        if empty_prefix and prefix_lengths is not None:
            raise ValueError(
                "empty_prefix=True cannot be combined with explicit prefix_lengths"
            )
        prefix_lens = (
            list(prefix_lengths) if prefix_lengths is not None else [0] * batch_size
        )
        if len(prefix_lens) != batch_size:
            raise ValueError(
                "prefix_lengths must contain one entry per batch item "
                f"(batch_size={batch_size}, len={len(prefix_lens)})"
            )
        if any(length < 0 for length in prefix_lens):
            raise ValueError(f"prefix_lengths must be non-negative, got {prefix_lens}")
        if block_ids is not None and not with_kv_cache_block_ids:
            raise ValueError("block_ids require with_kv_cache_block_ids=True")

        attn_inputs = PyAttentionInputs()

        # Prefill mode
        attn_inputs.is_prefill = True
        attn_inputs.is_cuda_graph = is_cuda_graph

        kv_lengths = [
            input_len + prefix_len
            for input_len, prefix_len in zip(input_lengths, prefix_lens, strict=True)
        ]

        # input_lengths is the number of new tokens of each sequence in the batch
        attn_inputs.input_lengths = torch.tensor(
            input_lengths, dtype=torch.int32, device="cpu"
        ).pin_memory()

        # sequence_lengths covers the prefix as well as the new tokens
        attn_inputs.sequence_lengths = torch.tensor(
            kv_lengths, dtype=torch.int32, device="cpu"
        ).pin_memory()

        attn_inputs.prefix_lengths = (
            torch.empty(0, dtype=torch.int32, device="cpu").pin_memory()
            if empty_prefix
            else torch.tensor(prefix_lens, dtype=torch.int32, device="cpu").pin_memory()
        )

        if with_kv_cache_block_ids:
            kv_cache_block_id = (
                block_ids
                if block_ids is not None
                else self._create_kv_cache_block_ids(
                    batch_size, kv_lengths, seq_size_per_block
                )
            )
            if kv_cache_block_id.dtype != torch.int32:
                raise ValueError(
                    f"block_ids must use torch.int32, got {kv_cache_block_id.dtype}"
                )
            if kv_cache_block_id.dim() != 2 or kv_cache_block_id.size(0) != batch_size:
                raise ValueError(
                    "block_ids must have shape [batch_size, max_blocks], got "
                    f"{tuple(kv_cache_block_id.shape)}"
                )
            required_blocks = max(
                math.ceil(kv_len / seq_size_per_block) for kv_len in kv_lengths
            )
            if kv_cache_block_id.size(1) < required_blocks:
                raise ValueError(
                    "block_ids has insufficient columns "
                    f"(got={kv_cache_block_id.size(1)}, need={required_blocks})"
                )
            attn_inputs.kv_cache_block_id = kv_cache_block_id
            attn_inputs.kv_cache_block_id_device = kv_cache_block_id.to(self.device)
            attn_inputs.kv_cache_kernel_block_id = kv_cache_block_id
            attn_inputs.kv_cache_kernel_block_id_device = kv_cache_block_id.to(
                self.device
            )

        # Q and KV cumulative lengths stay independent: prepare paths refresh
        # them separately during CUDA graph replay.
        cu_seqlens = [0]
        cu_kv_seqlens = [0]
        position_ids = []
        for input_len, prefix_len in zip(input_lengths, prefix_lens, strict=True):
            cu_seqlens.append(cu_seqlens[-1] + input_len)
            cu_kv_seqlens.append(cu_kv_seqlens[-1] + input_len + prefix_len)
            position_ids.extend(range(prefix_len, prefix_len + input_len))
        attn_inputs.cu_seqlens = torch.tensor(
            cu_seqlens, dtype=torch.int32, device="cpu"
        ).pin_memory()
        attn_inputs.cu_seqlens_device = attn_inputs.cu_seqlens.to(self.device)
        attn_inputs.cu_kv_seqlens_device = torch.tensor(
            cu_kv_seqlens, dtype=torch.int32, device=self.device
        )
        attn_inputs.combo_position_ids = torch.tensor(
            position_ids, dtype=torch.int32, device=self.device
        )

        padding_stride = graph_step if graph_step is not None else max(input_lengths)
        if padding_stride < max(input_lengths):
            raise ValueError(
                "graph_step must cover every input length "
                f"(graph_step={padding_stride}, input_lengths={input_lengths})"
            )
        cumulative_padding = 0
        padding_offsets = []
        for input_len in input_lengths:
            padding_offsets.extend([cumulative_padding] * input_len)
            cumulative_padding += padding_stride - input_len
        attn_inputs.padding_offset = torch.tensor(
            padding_offsets, dtype=torch.int32, device=self.device
        )
        attn_inputs.total_tokens = cu_seqlens[-1]
        attn_inputs.context_total_kv_length = cu_kv_seqlens[-1]

        # Set dtype using get_typemeta
        attn_inputs.dtype = get_typemeta(torch.zeros([1], dtype=dtype))

        return attn_inputs

    def _create_kv_cache(
        self,
        total_blocks: int,
        seq_size_per_block: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        content: Optional[torch.Tensor] = None,
        fp8_scale_fill: float = 1.0,
    ):
        """Helper to create KV cache

        Note: For HND layout, kv_cache_base should be a 5D tensor:
        [total_blocks, 2, num_kv_heads, seq_size_per_block, head_dim]
        where dimension 1 index 0 is K cache and index 1 is V cache.
        FP8 caches also include the scale buffer required by fused cache writes.
        Tests that verify the precise write footprint can request a NaN fill;
        other attention tests receive the production-neutral all-ones default.
        Pass content to start from the bytes of an existing cache, so a
        reference and an implementation under test can share a starting state.
        """
        kv_cache = LayerKVCache()

        # Create combined KV cache with shape [total_blocks, 2, num_kv_heads, seq_size_per_block, head_dim]
        # where dim=1, index=0 is K and index=1 is V
        is_fp8 = dtype == torch.float8_e4m3fn
        expected_shape = (
            total_blocks,
            2,
            num_kv_heads,
            seq_size_per_block,
            head_dim,
        )
        if content is not None:
            if tuple(content.shape) != expected_shape:
                raise ValueError(
                    f"content shape must be {expected_shape}, got {tuple(content.shape)}"
                )
            if content.dtype != dtype:
                raise ValueError(f"content dtype must be {dtype}, got {content.dtype}")
            kv_cache_combined = content.clone()
        else:
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
            kv_cache.kv_scale_base = torch.full(
                (total_blocks, 2 * num_kv_heads * seq_size_per_block),
                fp8_scale_fill,
                dtype=torch.float32,
                device=self.device,
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
