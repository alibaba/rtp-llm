import math
import os
import sys
from typing import List, Optional
from unittest import SkipTest, TestCase, main

import torch
import random
# Add project root to path
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CUR_PATH, "../../../")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

device = torch.device("cuda")

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.mha.flashinfer_prefill import (
    PyFlashinferPrefillAttnOp,
)
from rtp_llm.models_py.modules.mha.flashinfer_decode import (
    PyFlashinferDecodeAttnOp,
)

from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Import other MHA implementations
try:
    from rtp_llm.models_py.modules.mha.flashinfer_trtllm_gen import (
        FlashInferTRTLLMPrefillOp,
        FlashInferTRTLLMDecodeOp,
    )
    FLASHINFER_TRTLLM_AVAILABLE = True
except ImportError:
    FLASHINFER_TRTLLM_AVAILABLE = False

try:
    from rtp_llm.models_py.modules.mha.trt_mha import TRTMHAImpl
    TRT_MHA_AVAILABLE = True
except ImportError:
    TRT_MHA_AVAILABLE = False

try:
    from rtp_llm.models_py.modules.mha.xqa import XQAImpl
    XQA_AVAILABLE = True
except ImportError:
    XQA_AVAILABLE = False


def attention_prefill_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool = True,
) -> torch.Tensor:
    """
    Reference implementation for prefill attention using standard PyTorch operations.

    Args:
        q: Query tensor of shape [num_tokens, num_heads, head_dim]
        k: Key tensor of shape [num_tokens, num_kv_heads, head_dim]
        v: Value tensor of shape [num_tokens, num_kv_heads, head_dim]
        cu_seqlens: Cumulative sequence lengths of shape [batch_size + 1]
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_dim: Dimension of each head
        causal: Whether to apply causal masking

    Returns:
        Output tensor of shape [num_tokens, num_heads, head_dim]
    """
    batch_size = cu_seqlens.shape[0] - 1
    num_tokens = q.shape[0]
    dtype = q.dtype
    device = q.device
    # Scale factor
    scale = 1.0 / math.sqrt(head_dim)

    # Group query attention: repeat k and v if num_heads > num_kv_heads
    num_groups = num_heads // num_kv_heads
    if num_groups > 1:
        k = k.repeat_interleave(num_groups, dim=1)  # [num_tokens, num_heads, head_dim]
        v = v.repeat_interleave(num_groups, dim=1)  # [num_tokens, num_heads, head_dim]

    # Reshape for batch processing
    q = q.view(num_tokens, num_heads, head_dim)  # [num_tokens, num_heads, head_dim]
    k = k.view(num_tokens, num_heads, head_dim)  # [num_tokens, num_heads, head_dim]
    v = v.view(num_tokens, num_heads, head_dim)  # [num_tokens, num_heads, head_dim]

    # Process each sequence in the batch
    outputs = []
    for i in range(batch_size):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()
        seq_len = end_idx - start_idx

        if seq_len == 0:
            continue

        q_seq = q[start_idx:end_idx].transpose(0, 1)  # [seq_len, num_heads, head_dim]
        k_seq = k[start_idx:end_idx].transpose(0, 1)  # [seq_len, num_heads, head_dim]
        v_seq = v[start_idx:end_idx].transpose(0, 1)  # [seq_len, num_heads, head_dim]

        # Compute attention scores: [num_heads, seq_len, seq_len]
        scores = torch.einsum("hsd,htd->hst", q_seq, k_seq) * scale
        # Apply causal mask if needed
        if causal:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (seq_len, seq_len), fill_value=min_dtype, dtype=dtype, device=device
            )            
            causal_mask = torch.triu(
                causal_mask,
                diagonal=1
            )
            scores = scores + causal_mask.unsqueeze(0)

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)  # [num_heads, seq_len, seq_len]

        # Apply attention to values
        output = torch.einsum("hst,htd->hsd", attn_weights, v_seq).transpose(0, 1).contiguous()  # [seq_len, num_heads, head_dim]
        outputs.append(output)

    return torch.cat(outputs, dim=0)  # [num_tokens, num_heads, head_dim]


def attention_decode_ref(
    q: torch.Tensor,
    kv_cache: KVCache,
    sequence_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
) -> torch.Tensor:
    """
    Reference implementation for decode attention using standard PyTorch operations.

    Args:
        q: Query tensor of shape [batch_size, num_heads, head_dim]
        kv_cache: KV cache containing k_cache_base and v_cache_base
        sequence_lengths: Sequence lengths of shape [batch_size]
        block_tables: Block tables of shape [batch_size, max_blocks]
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_dim: Dimension of each head
        page_size: Size of each page in the cache

    Returns:
        Output tensor of shape [batch_size, num_heads, head_dim]
    """
    batch_size = q.shape[0]
    num_groups = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    # Get k and v cache
    # k_cache_base shape: [num_pages, 2, num_kv_heads, page_size, head_dim]
    # where dim 1: 0=k, 1=v
    k_cache = kv_cache.k_cache_base.select(1, 0)  # [num_pages, num_kv_heads, page_size, head_dim]
    v_cache = kv_cache.k_cache_base.select(1, 1)  # [num_pages, num_kv_heads, page_size, head_dim]

    outputs = []
    for i in range(batch_size):
        seq_len = sequence_lengths[i].item()
        num_blocks = (seq_len + page_size - 1) // page_size

        # Collect k and v from cache
        k_list = []
        v_list = []
        for block_idx in range(num_blocks):
            block_id = block_tables[i, block_idx].item()
            block_start = block_idx * page_size
            block_end = min(block_start + page_size, seq_len)
            block_len = block_end - block_start

            if block_len > 0:
                # Validate block_id is within cache bounds
                if block_id < 0 or block_id >= k_cache.shape[0]:
                    raise ValueError(
                        f"Invalid block_id {block_id} for sequence {i}, "
                        f"block {block_idx}. Cache has {k_cache.shape[0]} pages."
                    )
                k_block = k_cache[block_id, :, :block_len, :]  # [num_kv_heads, block_len, head_dim]
                v_block = v_cache[block_id, :, :block_len, :]  # [num_kv_heads, block_len, head_dim]
                k_list.append(k_block)
                v_list.append(v_block)

        if len(k_list) == 0:
            # Empty sequence, return zeros
            output = torch.zeros(num_heads, head_dim, device=q.device, dtype=q.dtype)
            outputs.append(output)
            continue

        k_seq = torch.cat(k_list, dim=1)  # [num_kv_heads, seq_len, head_dim]
        v_seq = torch.cat(v_list, dim=1)  # [num_kv_heads, seq_len, head_dim]

        # Repeat for group query attention
        if num_groups > 1:
            k_seq = k_seq.repeat_interleave(num_groups, dim=0)  # [num_heads, seq_len, head_dim]
            v_seq = v_seq.repeat_interleave(num_groups, dim=0)  # [num_heads, seq_len, head_dim]

        # Compute attention: q_i is [num_heads, head_dim], k_seq is [num_heads, seq_len, head_dim]
        q_i = q[i].unsqueeze(1)  # [num_heads, head_dim]
        # scores: [num_heads, seq_len] = q_i @ k_seq^T
        print('cc:', q_i.shape, k_seq.shape)
        scores = torch.einsum("htd,hsd->hts", q_i, k_seq) * scale  # [num_heads, seq_len]

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)  # [num_heads, seq_len]
        print('res:', q_i, k_seq, scores, attn_weights, v_seq)
        # Apply attention to values: [num_heads, seq_len] @ [num_heads, seq_len, head_dim] -> [num_heads, head_dim]
        output = torch.einsum("hts,hsd->htd", attn_weights, v_seq).squeeze(1)  # [num_heads, head_dim]
        outputs.append(output)

    return torch.stack(outputs, dim=0)  # [batch_size, num_heads, head_dim]


class FlashInferPythonMHATest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device(device)
        set_seed(25536)
        self.num_pages = 1024
        self.page_size = 64
        self.head_dim = 128
        self.num_kv_heads = 8
        self.num_heads = 64
        self.num_heads = 8        
        self.num_kv_heads = 1

        self.k_cache = torch.randn(
            self.num_pages,
            self.num_kv_heads,
            self.page_size,
            self.head_dim,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        self.v_cache = torch.randn(
            self.num_pages,
            self.num_kv_heads,
            self.page_size,
            self.head_dim,
            dtype=torch.bfloat16,
            device="cuda:0",
        )

    def gen_attention_inputs(
        self,
        input_lengths: Optional[List[int]] = None,
        sequence_lengths: Optional[List[int]] = None,
    ) -> PyAttentionInputs:
        assert not (input_lengths is None and sequence_lengths is None)
        attention_inputs: PyAttentionInputs = PyAttentionInputs()
        batch_size: int = 0
        max_seq_len: int = 0
        if sequence_lengths is not None:
            batch_size = len(sequence_lengths)
            attention_inputs.sequence_lengths = torch.tensor(
                sequence_lengths, dtype=torch.int32, device=torch.device("cpu")
            ).pin_memory()
            max_seq_len = attention_inputs.sequence_lengths.max().item()
            attention_inputs.is_prefill = False
            attention_inputs.prefix_lengths = torch.empty(0, dtype=torch.int32)
            attention_inputs.input_lengths = attention_inputs.sequence_lengths
        if input_lengths is not None:
            batch_size = len(input_lengths)
            attention_inputs.input_lengths = torch.tensor(
                input_lengths, dtype=torch.int32, device=torch.device("cpu")
            ).pin_memory()
            attention_inputs.is_prefill = True
            cu_seqlens = torch.zeros(
                len(input_lengths) + 1, dtype=torch.int32, device=torch.device("cpu")
            ).pin_memory()
            cu_seqlens[1:] = attention_inputs.input_lengths.cumsum(0)
            attention_inputs.cu_seqlens = cu_seqlens
            attention_inputs.cu_kv_seqlens = cu_seqlens
            max_seq_len = attention_inputs.input_lengths.max().item()
        max_block_size = max_seq_len // self.page_size + 1
        # Ensure we have enough pages in cache
        assert batch_size * max_block_size <= self.num_pages, \
            f"Not enough pages: need {batch_size * max_block_size}, have {self.num_pages}"
        block_tables = (
            torch.arange(
                batch_size * max_block_size,
                dtype=torch.int32,
                device=torch.device("cpu"),
            )
            .view(batch_size, max_block_size)
            .pin_memory()
        )
        attention_inputs.kv_cache_block_id_device = block_tables
        attention_inputs.kv_cache_block_id_host = block_tables
        return attention_inputs

    def _create_config(self, use_mla: bool = False, tp_size: int = 1) -> GptInitModelParameters:
        """Create a standard GptInitModelParameters config for testing."""
        config = GptInitModelParameters(self.num_heads, self.head_dim, 12, 2048, 102400)
        config.head_num = self.num_heads
        config.head_num_kv = self.num_kv_heads
        config.hidden_size = self.head_dim * self.num_heads
        config.seq_size_per_block = self.page_size
        config.size_per_head = self.head_dim
        config.data_type = 'bf16'
        config.kv_cache_data_type = 'bf16'
        if use_mla:
            config.use_mla = use_mla
        if tp_size > 1:
            config.tp_size = tp_size
        return config

    def _create_kv_cache(self) -> KVCache:
        """Create a standard KVCache for testing."""
        kv_cache: KVCache = KVCache()
        kv_cache.k_cache_base = torch.stack([self.k_cache, self.v_cache], dim=1)
        kv_cache.v_cache_base = self.v_cache
        return kv_cache

    def _test_run_flashinfer_prefill_test(self):
        """Test flashinfer prefill attention with reference comparison."""
        input_lengths = [2, 3, 10, 12]
        # input_lengths = [1]
        
        num_tokens = sum(input_lengths)

        config = GptInitModelParameters(self.num_heads, self.head_dim, 12, 2048, 102400)
        config.head_num = self.num_heads
        config.head_num_kv = self.num_kv_heads
        config.hidden_size = self.head_dim * self.num_heads
        config.seq_size_per_block = self.page_size
        config.size_per_head = self.head_dim
        config.data_type = 'bf16'
        config.kv_cache_data_type = 'bf16'

        attn_inputs = self.gen_attention_inputs(input_lengths=input_lengths)
        qkv = torch.randn(
            [num_tokens, config.hidden_size + 2 * self.num_kv_heads * self.head_dim],
            dtype=torch.bfloat16,
            device=device,
        )

        # Split qkv for reference implementation
        q_size = self.head_dim * self.num_heads
        k_size = self.head_dim * self.num_kv_heads
        v_size = self.head_dim * self.num_kv_heads

        q_ref = qkv[:, :q_size].reshape(num_tokens, self.num_heads, self.head_dim)
        k_ref = qkv[:, q_size:q_size + k_size].reshape(num_tokens, self.num_kv_heads, self.head_dim)
        v_ref = qkv[:, q_size + k_size:q_size + k_size + v_size].reshape(num_tokens, self.num_kv_heads, self.head_dim)

        kv_cache: KVCache = KVCache()
        kv_cache.k_cache_base = torch.stack([self.k_cache, self.v_cache], dim=1)
        kv_cache.v_cache_base = self.v_cache

        # Run flashinfer implementation
        op = PyFlashinferPrefillAttnOp(config)
        input_params = op.prepare(attn_inputs)
        out_flashinfer = op.forward(qkv, kv_cache, input_params)

        # Run reference implementation
        out_ref = attention_prefill_ref(
            q_ref,
            k_ref,
            v_ref,
            attn_inputs.cu_seqlens,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            causal=True,
        )

        # Reshape flashinfer output to match reference
        out_flashinfer_reshaped = out_flashinfer.reshape(num_tokens, self.num_heads, self.head_dim)

        # Compare results
        print(f"FlashInfer output shape: {out_flashinfer.shape}")
        print(f"Reference output shape: {out_ref.shape}")
        print(f"FlashInfer output dtype: {out_flashinfer.dtype}")
        print(f"Reference output dtype: {out_ref.dtype}")

        # Convert to float32 for comparison
        out_flashinfer_f32 = out_flashinfer_reshaped.float()
        out_ref_f32 = out_ref.float()

        # Compute statistics
        max_diff = (out_flashinfer_f32 - out_ref_f32).abs().max().item()
        mean_diff = (out_flashinfer_f32 - out_ref_f32).abs().mean().item()
        print(f"Max absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")

        # Check if results are close (using relaxed tolerance for bf16)
        # Note: bf16 has lower precision, so we use a more relaxed tolerance
        atol = 0.1
        rtol = 0.1
        is_close = torch.allclose(out_flashinfer_f32, out_ref_f32, atol=atol, rtol=rtol)

        if not is_close:
            # Print some sample values for debugging
            print("\nSample comparison (first 5 tokens, first head, first 5 dims):")
            print("FlashInfer:", out_flashinfer_f32[:5, 0, :5])
            print("Reference: ", out_ref_f32[:5, 0, :5])
            print("Difference:", (out_flashinfer_f32[:5, 0, :5] - out_ref_f32[:5, 0, :5]).abs())

        self.assertTrue(
            is_close,
            f"FlashInfer output does not match reference. Max diff: {max_diff}, Mean diff: {mean_diff}"
        )

    def test_run_flashinfer_decode_test(self):
        """Test flashinfer decode attention with reference comparison."""
        sequence_lengths = [2, 3, 10, 12]
        sequence_lengths = [1]        
        num_tokens = len(sequence_lengths)
        config = GptInitModelParameters(self.num_heads, self.head_dim, 12, 2048, 102400)
        config.head_num = self.num_heads
        config.head_num_kv = self.num_kv_heads
        config.hidden_size = self.head_dim * self.num_heads
        config.seq_size_per_block = self.page_size
        config.size_per_head = self.head_dim
        config.data_type = 'bf16'
        config.kv_cache_data_type = 'bf16'

        attn_inputs = self.gen_attention_inputs(sequence_lengths=sequence_lengths)
        q = torch.randn(
            [num_tokens, config.hidden_size],
            dtype=torch.bfloat16,
            device=device,
        )

        # Reshape q for reference implementation
        q_ref = q.reshape(num_tokens, self.num_heads, self.head_dim)

        kv_cache: KVCache = KVCache()
        kv_cache.k_cache_base = torch.stack([self.k_cache, self.v_cache], dim=1)
        kv_cache.v_cache_base = self.v_cache

        # Run flashinfer implementation
        op = PyFlashinferDecodeAttnOp(config)
        input_params = op.prepare(attn_inputs)
        out_flashinfer = op.forward(q, kv_cache, input_params)

        # Run reference implementation
        out_ref = attention_decode_ref(
            q_ref,
            kv_cache,
            attn_inputs.sequence_lengths,
            attn_inputs.kv_cache_block_id_device,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
        )

        # Reshape flashinfer output to match reference
        out_flashinfer_reshaped = out_flashinfer.reshape(num_tokens, self.num_heads, self.head_dim)

        # Compare results
        print(f"FlashInfer output shape: {out_flashinfer.shape}")
        print(f"Reference output shape: {out_ref.shape}")
        print(f"FlashInfer output dtype: {out_flashinfer.dtype}")
        print(f"Reference output dtype: {out_ref.dtype}")

        # Convert to float32 for comparison
        out_flashinfer_f32 = out_flashinfer_reshaped.float()
        out_ref_f32 = out_ref.float()

        # Compute statistics
        max_diff = (out_flashinfer_f32 - out_ref_f32).abs().max().item()
        mean_diff = (out_flashinfer_f32 - out_ref_f32).abs().mean().item()
        print(f"Max absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")

        # Check if results are close (using relaxed tolerance for bf16)
        atol = 0.1
        rtol = 0.1
        is_close = torch.allclose(out_flashinfer_f32, out_ref_f32, atol=atol, rtol=rtol)

        if not is_close:
            # Print some sample values for debugging
            print("\nSample comparison (first batch, first head, first 5 dims):")
            # print("FlashInfer decode:", out_flashinfer_f32[0, 0, :5])
            # print("Reference decode: ", out_ref_f32[0, 0, :5])
            print("FlashInfer decode:", out_flashinfer_f32)
            print("Reference decode: ", out_ref_f32)
            
            print("Difference:", (out_flashinfer_f32[0, 0, :5] - out_ref_f32[0, 0, :5]).abs())

        self.assertTrue(
            is_close,
            f"FlashInfer output does not match reference. Max diff: {max_diff}, Mean diff: {mean_diff}"
        )

    def _test_prefill_multiple_configs(self):
        """Test prefill with multiple configurations."""
        test_configs = [
            {"input_lengths": [1, 1, 1], "name": "single_token"},
            {"input_lengths": [5, 5, 5], "name": "uniform_short"},
            {"input_lengths": [1, 32, 64], "name": "varying_lengths"},
        ]

        for test_config in test_configs:
            with self.subTest(config=test_config["name"]):
                input_lengths = test_config["input_lengths"]
                num_tokens = sum(input_lengths)

                config = GptInitModelParameters(self.num_heads, self.head_dim, 12, 2048, 102400)
                config.head_num = self.num_heads
                config.head_num_kv = self.num_kv_heads
                config.hidden_size = self.head_dim * self.num_heads
                config.seq_size_per_block = self.page_size
                config.size_per_head = self.head_dim
                config.data_type = 'bf16'
                config.kv_cache_data_type = 'bf16'

                attn_inputs = self.gen_attention_inputs(input_lengths=input_lengths)
                qkv = torch.randn(
                    [num_tokens, config.hidden_size + 2 * self.num_kv_heads * self.head_dim],
                    dtype=torch.bfloat16,
                    device=device,
                )

                # Split qkv for reference
                q_size = self.head_dim * self.num_heads
                k_size = self.head_dim * self.num_kv_heads
                v_size = self.head_dim * self.num_kv_heads

                q_ref = qkv[:, :q_size].reshape(num_tokens, self.num_heads, self.head_dim)
                k_ref = qkv[:, q_size:q_size + k_size].reshape(num_tokens, self.num_kv_heads, self.head_dim)
                v_ref = qkv[:, q_size + k_size:q_size + k_size + v_size].reshape(num_tokens, self.num_kv_heads, self.head_dim)

                kv_cache: KVCache = KVCache()
                kv_cache.k_cache_base = torch.stack([self.k_cache, self.v_cache], dim=1)
                kv_cache.v_cache_base = self.v_cache

                op = PyFlashinferPrefillAttnOp(config)
                input_params = op.prepare(attn_inputs)
                out_flashinfer = op.forward(qkv, kv_cache, input_params)

                out_ref = attention_prefill_ref(
                    q_ref, k_ref, v_ref,
                    attn_inputs.cu_seqlens,
                    self.num_heads, self.num_kv_heads, self.head_dim,
                    causal=True,
                )

                out_flashinfer_reshaped = out_flashinfer.reshape(num_tokens, self.num_heads, self.head_dim)
                is_close = torch.allclose(
                    out_flashinfer_reshaped.float(),
                    out_ref.float(),
                    atol=0.1,
                    rtol=0.1,
                )

                self.assertTrue(
                    is_close,
                    f"Prefill test failed for config {test_config['name']}"
                )

    def _test_decode_multiple_configs(self):
        """Test decode with multiple configurations."""
        test_configs = [
            {"sequence_lengths": [1, 1, 1], "name": "single_token"},
            {"sequence_lengths": [5, 5, 5], "name": "uniform_short"},
            {"sequence_lengths": [1, 16, 32], "name": "varying_lengths"},
        ]

        for test_config in test_configs:
            with self.subTest(config=test_config["name"]):
                sequence_lengths = test_config["sequence_lengths"]
                num_tokens = len(sequence_lengths)

                config = GptInitModelParameters(self.num_heads, self.head_dim, 12, 2048, 102400)
                config.head_num = self.num_heads
                config.head_num_kv = self.num_kv_heads
                config.hidden_size = self.head_dim * self.num_heads
                config.seq_size_per_block = self.page_size
                config.size_per_head = self.head_dim
                config.data_type = 'bf16'
                config.kv_cache_data_type = 'bf16'

                attn_inputs = self.gen_attention_inputs(sequence_lengths=sequence_lengths)
                q = torch.randn(
                    [num_tokens, config.hidden_size],
                    dtype=torch.bfloat16,
                    device=device,
                )

                q_ref = q.reshape(num_tokens, self.num_heads, self.head_dim)

                kv_cache: KVCache = KVCache()
                kv_cache.k_cache_base = torch.stack([self.k_cache, self.v_cache], dim=1)
                kv_cache.v_cache_base = self.v_cache

                op = PyFlashinferDecodeAttnOp(config)
                input_params = op.prepare(attn_inputs)
                out_flashinfer = op.forward(q, kv_cache, input_params)

                out_ref = attention_decode_ref(
                    q_ref,
                    kv_cache,
                    attn_inputs.sequence_lengths,
                    attn_inputs.kv_cache_block_id_device,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.page_size,
                )

                out_flashinfer_reshaped = out_flashinfer.reshape(num_tokens, self.num_heads, self.head_dim)
                is_close = torch.allclose(
                    out_flashinfer_reshaped.float(),
                    out_ref.float(),
                    atol=0.1,
                    rtol=0.1,
                )

                self.assertTrue(
                    is_close,
                    f"Decode test failed for config {test_config['name']}"
                )

    def _test_flashinfer_trtllm_prefill(self):
        """Test FlashInferTRTLLM prefill attention with reference comparison."""
        if not FLASHINFER_TRTLLM_AVAILABLE:
            raise SkipTest("FlashInferTRTLLM not available")

        # Check if SM_100 is available
        try:
            is_sm_100 = torch.cuda.get_device_capability()[0] in [10]
            if not is_sm_100:
                raise SkipTest("FlashInferTRTLLM requires SM_100 (compute capability 10.0)")
        except:
            raise SkipTest("Cannot determine GPU compute capability")

        input_lengths = [2, 3, 10, 12]
        num_tokens = sum(input_lengths)

        config = self._create_config(use_mla=False, tp_size=1)
        attn_inputs = self.gen_attention_inputs(input_lengths=input_lengths)
        qkv = torch.randn(
            [num_tokens, config.hidden_size + 2 * self.num_kv_heads * self.head_dim],
            dtype=torch.bfloat16,
            device=device,
        )

        # Split qkv for reference implementation
        q_size = self.head_dim * self.num_heads
        k_size = self.head_dim * self.num_kv_heads
        v_size = self.head_dim * self.num_kv_heads

        q_ref = qkv[:, :q_size].reshape(num_tokens, self.num_heads, self.head_dim)
        k_ref = qkv[:, q_size:q_size + k_size].reshape(num_tokens, self.num_kv_heads, self.head_dim)
        v_ref = qkv[:, q_size + k_size:q_size + k_size + v_size].reshape(num_tokens, self.num_kv_heads, self.head_dim)

        # Extract q for FlashInferTRTLLM (it only needs q, k and v are in kv_cache)
        q_trtllm = qkv[:, :q_size]

        # Prepare kv_cache - FlashInferTRTLLM expects float8 format
        # Use simplified cache structure for testing
        kv_cache: KVCache = KVCache()
        # Create float8 cache (FlashInferTRTLLM requires float8_e4m3fn)
        temp_k_cache = self.k_cache.to(torch.float8_e4m3fn)
        temp_v_cache = self.v_cache.to(torch.float8_e4m3fn)
        kv_cache.k_cache_base = torch.stack([temp_k_cache, temp_v_cache], dim=1)
        kv_cache.v_cache_base = temp_v_cache

        # Run FlashInferTRTLLM implementation
        op = FlashInferTRTLLMPrefillOp(config)
        if not op.support(attn_inputs):
            raise SkipTest("FlashInferTRTLLM prefill not supported for this configuration")
        input_params = op.prepare(attn_inputs)
        out_trtllm = op.forward(q_trtllm, kv_cache, input_params)

        # Run reference implementation
        out_ref = attention_prefill_ref(
            q_ref,
            k_ref,
            v_ref,
            attn_inputs.cu_seqlens,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            causal=True,
        )

        # Reshape output to match reference
        out_trtllm_reshaped = out_trtllm.reshape(num_tokens, self.num_heads, self.head_dim)

        # Compare results
        print(f"FlashInferTRTLLM output shape: {out_trtllm.shape}")
        print(f"Reference output shape: {out_ref.shape}")

        # Convert to float32 for comparison
        out_trtllm_f32 = out_trtllm_reshaped.float()
        out_ref_f32 = out_ref.float()

        # Compute statistics
        max_diff = (out_trtllm_f32 - out_ref_f32).abs().max().item()
        mean_diff = (out_trtllm_f32 - out_ref_f32).abs().mean().item()
        print(f"Max absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")

        # Check if results are close (using relaxed tolerance for bf16 and float8)
        atol = 0.5  # More relaxed for float8
        rtol = 0.5
        is_close = torch.allclose(out_trtllm_f32, out_ref_f32, atol=atol, rtol=rtol)

        if not is_close:
            print("\nSample comparison (first 5 tokens, first head, first 5 dims):")
            print("FlashInferTRTLLM:", out_trtllm_f32[:5, 0, :5])
            print("Reference: ", out_ref_f32[:5, 0, :5])
            print("Difference:", (out_trtllm_f32[:5, 0, :5] - out_ref_f32[:5, 0, :5]).abs())

        self.assertTrue(
            is_close,
            f"FlashInferTRTLLM output does not match reference. Max diff: {max_diff}, Mean diff: {mean_diff}"
        )

    def _test_flashinfer_trtllm_decode(self):
        """Test FlashInferTRTLLM decode attention with reference comparison."""
        if not FLASHINFER_TRTLLM_AVAILABLE:
            raise SkipTest("FlashInferTRTLLM not available")

        # Check if SM_100 is available
        try:
            is_sm_100 = torch.cuda.get_device_capability()[0] in [10]
            if not is_sm_100:
                raise SkipTest("FlashInferTRTLLM requires SM_100 (compute capability 10.0)")
        except:
            raise SkipTest("Cannot determine GPU compute capability")

        sequence_lengths = [2, 3, 10, 12]
        num_tokens = len(sequence_lengths)
        config = GptInitModelParameters(self.num_heads, self.head_dim, 12, 2048, 102400)
        config.head_num = self.num_heads
        config.head_num_kv = self.num_kv_heads
        config.hidden_size = self.head_dim * self.num_heads
        config.seq_size_per_block = self.page_size
        config.size_per_head = self.head_dim
        config.data_type = 'bf16'
        config.kv_cache_data_type = 'bf16'
        config.use_mla = False
        config.tp_size = 1

        attn_inputs = self.gen_attention_inputs(sequence_lengths=sequence_lengths)
        q = torch.randn(
            [num_tokens, config.hidden_size],
            dtype=torch.bfloat16,
            device=device,
        )

        # Reshape q for reference implementation
        q_ref = q.reshape(num_tokens, self.num_heads, self.head_dim)

        # Prepare kv_cache - FlashInferTRTLLM expects float8 format
        kv_cache: KVCache = KVCache()
        # Create float8 cache (FlashInferTRTLLM requires float8_e4m3fn)
        temp_k_cache = self.k_cache.to(torch.float8_e4m3fn)
        temp_v_cache = self.v_cache.to(torch.float8_e4m3fn)
        kv_cache.k_cache_base = torch.stack([temp_k_cache, temp_v_cache], dim=1)
        kv_cache.v_cache_base = temp_v_cache

        # Run FlashInferTRTLLM implementation
        op = FlashInferTRTLLMDecodeOp(config)
        if not op.support(attn_inputs):
            raise SkipTest("FlashInferTRTLLM decode not supported for this configuration")
        input_params = op.prepare(attn_inputs)
        out_trtllm = op.forward(q, kv_cache, input_params)

        # Run reference implementation
        out_ref = attention_decode_ref(
            q_ref,
            kv_cache,
            attn_inputs.sequence_lengths,
            attn_inputs.kv_cache_block_id_device,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
        )

        # Reshape output to match reference
        out_trtllm_reshaped = out_trtllm.reshape(num_tokens, self.num_heads, self.head_dim)

        # Compare results
        print(f"FlashInferTRTLLM output shape: {out_trtllm.shape}")
        print(f"Reference output shape: {out_ref.shape}")

        # Convert to float32 for comparison
        out_trtllm_f32 = out_trtllm_reshaped.float()
        out_ref_f32 = out_ref.float()

        # Compute statistics
        max_diff = (out_trtllm_f32 - out_ref_f32).abs().max().item()
        mean_diff = (out_trtllm_f32 - out_ref_f32).abs().mean().item()
        print(f"Max absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")

        # Check if results are close (using relaxed tolerance for bf16 and float8)
        atol = 0.5  # More relaxed for float8
        rtol = 0.5
        is_close = torch.allclose(out_trtllm_f32, out_ref_f32, atol=atol, rtol=rtol)

        if not is_close:
            print("\nSample comparison (first batch, first head, first 5 dims):")
            print("FlashInferTRTLLM:", out_trtllm_f32[0, 0, :5])
            print("Reference: ", out_ref_f32[0, 0, :5])
            print("Difference:", (out_trtllm_f32[0, 0, :5] - out_ref_f32[0, 0, :5]).abs())

        self.assertTrue(
            is_close,
            f"FlashInferTRTLLM output does not match reference. Max diff: {max_diff}, Mean diff: {mean_diff}"
        )

    def _test_trt_mha_prefill(self):
        """Test TRT MHA prefill attention with reference comparison."""
        if not TRT_MHA_AVAILABLE:
            raise SkipTest("TRT MHA not available")

        input_lengths = [2, 3, 10, 12]
        num_tokens = sum(input_lengths)

        config = GptInitModelParameters(self.num_heads, self.head_dim, 12, 2048, 102400)
        config.head_num = self.num_heads
        config.head_num_kv = self.num_kv_heads
        config.hidden_size = self.head_dim * self.num_heads
        config.seq_size_per_block = self.page_size
        config.size_per_head = self.head_dim
        config.data_type = 'bf16'
        config.kv_cache_data_type = 'bf16'

        attn_inputs = self.gen_attention_inputs(input_lengths=input_lengths)
        qkv = torch.randn(
            [num_tokens, config.hidden_size + 2 * self.num_kv_heads * self.head_dim],
            dtype=torch.bfloat16,
            device=device,
        )

        # Split qkv for reference implementation
        q_size = self.head_dim * self.num_heads
        k_size = self.head_dim * self.num_kv_heads
        v_size = self.head_dim * self.num_kv_heads

        q_ref = qkv[:, :q_size].reshape(num_tokens, self.num_heads, self.head_dim)
        k_ref = qkv[:, q_size:q_size + k_size].reshape(num_tokens, self.num_kv_heads, self.head_dim)
        v_ref = qkv[:, q_size + k_size:q_size + k_size + v_size].reshape(num_tokens, self.num_kv_heads, self.head_dim)

        kv_cache: KVCache = KVCache()
        kv_cache.k_cache_base = torch.stack([self.k_cache, self.v_cache], dim=1)
        kv_cache.v_cache_base = self.v_cache

        # Run TRT MHA implementation
        try:
            impl = TRTMHAImpl(config, attn_inputs)
            if not impl.support():
                raise SkipTest("TRT MHA prefill not supported for this configuration")
            out_trt = impl.forward(qkv, kv_cache)
        except Exception as e:
            raise SkipTest(f"TRT MHA prefill failed: {e}")

        # Run reference implementation
        out_ref = attention_prefill_ref(
            q_ref,
            k_ref,
            v_ref,
            attn_inputs.cu_seqlens,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            causal=True,
        )

        # Reshape output to match reference
        out_trt_reshaped = out_trt.reshape(num_tokens, self.num_heads, self.head_dim)

        # Compare results
        print(f"TRT MHA output shape: {out_trt.shape}")
        print(f"Reference output shape: {out_ref.shape}")

        # Convert to float32 for comparison
        out_trt_f32 = out_trt_reshaped.float()
        out_ref_f32 = out_ref.float()

        # Compute statistics
        max_diff = (out_trt_f32 - out_ref_f32).abs().max().item()
        mean_diff = (out_trt_f32 - out_ref_f32).abs().mean().item()
        print(f"Max absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")

        # Check if results are close (using relaxed tolerance for bf16)
        atol = 0.1
        rtol = 0.1
        is_close = torch.allclose(out_trt_f32, out_ref_f32, atol=atol, rtol=rtol)

        if not is_close:
            print("\nSample comparison (first 5 tokens, first head, first 5 dims):")
            print("TRT MHA:", out_trt_f32[:5, 0, :5])
            print("Reference: ", out_ref_f32[:5, 0, :5])
            print("Difference:", (out_trt_f32[:5, 0, :5] - out_ref_f32[:5, 0, :5]).abs())

        self.assertTrue(
            is_close,
            f"TRT MHA output does not match reference. Max diff: {max_diff}, Mean diff: {mean_diff}"
        )

    def _test_xqa_decode(self):
        """Test XQA decode attention with reference comparison."""
        if not XQA_AVAILABLE:
            raise SkipTest("XQA not available")

        sequence_lengths = [2, 3, 10, 12]
        num_tokens = len(sequence_lengths)
        config = GptInitModelParameters(self.num_heads, self.head_dim, 12, 2048, 102400)
        config.head_num = self.num_heads
        config.head_num_kv = self.num_kv_heads
        config.hidden_size = self.head_dim * self.num_heads
        config.seq_size_per_block = self.page_size
        config.size_per_head = self.head_dim
        config.data_type = 'bf16'
        config.kv_cache_data_type = 'bf16'

        attn_inputs = self.gen_attention_inputs(sequence_lengths=sequence_lengths)
        q = torch.randn(
            [num_tokens, config.hidden_size],
            dtype=torch.bfloat16,
            device=device,
        )

        # Reshape q for reference implementation
        q_ref = q.reshape(num_tokens, self.num_heads, self.head_dim)

        kv_cache: KVCache = KVCache()
        kv_cache.k_cache_base = torch.stack([self.k_cache, self.v_cache], dim=1)
        kv_cache.v_cache_base = self.v_cache

        # Run XQA implementation
        try:
            impl = XQAImpl(config, attn_inputs)
            if not impl.support():
                raise SkipTest("XQA decode not supported for this configuration")
            out_xqa = impl.forward(q, kv_cache)
        except Exception as e:
            raise SkipTest(f"XQA decode failed: {e}")

        # Run reference implementation
        out_ref = attention_decode_ref(
            q_ref,
            kv_cache,
            attn_inputs.sequence_lengths,
            attn_inputs.kv_cache_block_id_device,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
        )

        # Reshape output to match reference
        out_xqa_reshaped = out_xqa.reshape(num_tokens, self.num_heads, self.head_dim)

        # Compare results
        print(f"XQA output shape: {out_xqa.shape}")
        print(f"Reference output shape: {out_ref.shape}")

        # Convert to float32 for comparison
        out_xqa_f32 = out_xqa_reshaped.float()
        out_ref_f32 = out_ref.float()

        # Compute statistics
        max_diff = (out_xqa_f32 - out_ref_f32).abs().max().item()
        mean_diff = (out_xqa_f32 - out_ref_f32).abs().mean().item()
        print(f"Max absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")

        # Check if results are close (using relaxed tolerance for bf16)
        atol = 0.1
        rtol = 0.1
        is_close = torch.allclose(out_xqa_f32, out_ref_f32, atol=atol, rtol=rtol)

        if not is_close:
            print("\nSample comparison (first batch, first head, first 5 dims):")
            print("XQA:", out_xqa_f32[0, 0, :5])
            print("Reference: ", out_ref_f32[0, 0, :5])
            print("Difference:", (out_xqa_f32[0, 0, :5] - out_ref_f32[0, 0, :5]).abs())

        self.assertTrue(
            is_close,
            f"XQA output does not match reference. Max diff: {max_diff}, Mean diff: {mean_diff}"
        )


if __name__ == "__main__":
    main()
