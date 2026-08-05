"""Unit tests for MhaRotaryEmbeddingOp"""

import math
import unittest
from typing import Tuple

import torch
from flashinfer import get_batch_indices_positions, get_seq_lens

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    MhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
    KVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPagedPrefillImpl,
    PyFlashinferPrefillAttnOp,
    PyFlashinferPrefillImpl,
    PyFlashinferPrefillPagedAttnOp,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, RopeStyle
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCachePrefillOpQOut,
    LayerKVCache,
    PyAttentionInputs,
    get_typemeta,
    init_exec_ctx,
)


class RopeParams:
    """Simple RoPE parameters container for testing"""

    def __init__(
        self,
        batch_indice_d: torch.Tensor,
        positions_d: torch.Tensor,
        page_indice_d: torch.Tensor,
        decode_page_indptr_d: torch.Tensor,
        paged_kv_last_page_len_d: torch.Tensor,
    ):
        self.batch_indice_d = batch_indice_d
        self.positions_d = positions_d
        self.page_indice_d = page_indice_d
        self.decode_page_indptr_d = decode_page_indptr_d
        self.paged_kv_last_page_len_d = paged_kv_last_page_len_d


def create_cos_sin_cache(
    head_dim: int, max_seq_len: int = 2048, base: float = 10000.0, device: str = "cuda"
) -> torch.Tensor:
    """Create cos_sin_cache for RoPE using standard implementation.

    Args:
        head_dim: Head dimension (RoPE dimension)
        max_seq_len: Maximum sequence length
        base: RoPE base frequency
        device: Device to create the cache on

    Returns:
        cos_sin_cache: [max_seq_len, head_dim] tensor with first half cos, second half sin
    """
    # Create inverse frequencies
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )

    # Create position IDs
    t = torch.arange(max_seq_len, dtype=torch.float32)

    # Compute frequencies
    freqs = torch.outer(t, inv_freq)  # [max_seq_len, head_dim/2]

    # Compute cos and sin
    cos = freqs.cos()  # [max_seq_len, head_dim/2]
    sin = freqs.sin()  # [max_seq_len, head_dim/2]

    # Concatenate cos and sin (non-interleaved format)
    # Format: [cos[0], cos[1], ..., cos[dim/2-1], sin[0], sin[1], ..., sin[dim/2-1]]
    # This matches what apply_rope_reference and flashinfer expect
    cos_sin_cache = torch.cat([cos, sin], dim=-1)  # [max_seq_len, head_dim]

    return cos_sin_cache.contiguous().to(device).to(torch.float32)


def create_test_attn_config(
    head_num: int = 8,
    kv_head_num: int = 4,
    size_per_head: int = 64,
    tokens_per_block: int = 16,
    max_seq_len: int = 2048,
    dtype: torch.dtype = torch.float16,
) -> AttentionConfigs:
    """Create AttentionConfigs for testing.

    Args:
        head_num: Number of query heads
        kv_head_num: Number of key-value heads
        size_per_head: Dimension of each head
        tokens_per_block: Tokens per KV cache block
        max_seq_len: Maximum sequence length
        dtype: Data type for attention

    Returns:
        AttentionConfigs object for testing
    """
    config = AttentionConfigs()
    config.head_num = head_num
    config.kv_head_num = kv_head_num
    config.size_per_head = size_per_head
    config.tokens_per_block = tokens_per_block
    config.kernel_tokens_per_block = tokens_per_block
    config.rope_config.style = RopeStyle.Base
    config.rope_config.dim = size_per_head
    config.rope_config.base = 10000
    config.rope_config.max_pos = max_seq_len
    config.max_seq_len = max_seq_len
    config.dtype = dtype
    config.use_mla = False
    return config


def apply_rope_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation of RoPE (non-interleave style).

    Args:
        q: Query [num_tokens, num_heads, head_dim]
        k: Key [num_tokens, num_kv_heads, head_dim]
        cos_sin_cache: [max_seq_len, head_dim] with first half cos, second half sin
        positions: Position IDs [num_tokens]

    Returns:
        q_rope, k_rope: Rotated query and key tensors
    """
    head_dim = q.size(2)
    half_dim = head_dim // 2

    # Extract cos and sin for the given positions
    cos_sin = cos_sin_cache[positions]  # [num_tokens, head_dim]
    cos_half = cos_sin[:, :half_dim]  # [num_tokens, half_dim]
    sin_half = cos_sin[:, half_dim:]  # [num_tokens, half_dim]

    # Expand cos and sin to full head_dim by repeating for both halves
    # For non-interleave style: [cos, cos] and [sin, sin]
    cos = torch.cat([cos_half, cos_half], dim=-1)  # [num_tokens, head_dim]
    sin = torch.cat([sin_half, sin_half], dim=-1)  # [num_tokens, head_dim]

    # Apply RoPE (non-interleave style)
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Split and rotate for RoPE"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    # Apply to Q: q_rope = q * cos + rotate_half(q) * sin
    q_rope = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))

    # Apply to K: k_rope = k * cos + rotate_half(k) * sin
    k_rope = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))

    return q_rope, k_rope


class TestMhaRotaryEmbeddingOp(unittest.TestCase):
    """Test suite for MhaRotaryEmbeddingOp"""

    def setUp(self):
        """Set up test fixtures"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available - this test requires CUDA")

        self.device = torch.device("cuda")
        torch.manual_seed(42)

        # Initialize device for C++ operators (needed for FusedRopeKVCachePrefillOpQOut)
        try:
            py_env_configs = PyEnvConfigs()
            py_env_configs.runtime_config.fifo_scheduler_config.max_context_batch_size = (
                64
            )

            engine_config = EngineConfig.create(py_env_configs)
            model_config = ModelConfig()
            model_config.max_seq_len = 2048

            pc = engine_config.parallelism_config
            init_exec_ctx(
                device_id=pc.world_rank % pc.local_world_size,
                trace_memory=engine_config.profiling_debug_logging_config.trace_memory,
                enable_comm_overlap=engine_config.device_resource_config.enable_comm_overlap,
                mla_ops_type=int(model_config.mla_ops_type),
            )
            self.device_initialized = True
        except Exception as e:
            # If device initialization fails, some tests may still work
            print(f"Warning: Failed to initialize device: {e}")
            self.device_initialized = False

    def test_fused_rope_vs_mha_rope(self):
        """Compare FusedRopeKVCachePrefillOpQOut (C++) vs MhaRotaryEmbeddingOp (Python)"""
        if not self.device_initialized:
            self.skipTest(
                "Device not initialized - required for FusedRopeKVCachePrefillOpQOut"
            )

        print("\n" + "=" * 80)
        print("Testing: FusedRopeKVCachePrefillOpQOut vs MhaRotaryEmbeddingOp")
        print("=" * 80)

        # Test parameters
        num_tokens = 128
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        token_per_block = 16
        batch_size = 1

        # Create attention config
        attn_config = create_test_attn_config(
            head_num=num_heads,
            kv_head_num=num_kv_heads,
            size_per_head=head_dim,
            tokens_per_block=token_per_block,
        )

        # Create shared cos_sin_cache
        cos_sin_cache = create_cos_sin_cache(
            head_dim=head_dim, max_seq_len=2048, device="cuda"
        )

        # Create input QKV
        qkv_shape = (num_tokens, (num_heads + 2 * num_kv_heads) * head_dim)
        qkv = torch.randn(qkv_shape, dtype=torch.float16, device=self.device)

        # Create KV cache
        num_pages = math.ceil(num_tokens / token_per_block)
        kv_cache_base = torch.zeros(
            num_pages,
            2,
            num_kv_heads,
            token_per_block,
            head_dim,
            device=self.device,
            dtype=torch.float16,
        )

        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = kv_cache_base

        # Create position IDs (crucial: both implementations should use the same positions)
        # For prefill from scratch, positions should be [0, 1, 2, ..., num_tokens-1]
        positions = torch.arange(num_tokens, dtype=torch.int32, device=self.device)

        # ========== Reference Implementation (Pure PyTorch) ==========
        print("\n[1] Testing Reference Implementation (Pure PyTorch)")

        # Extract Q, K from qkv for reference computation
        q_size = num_heads * head_dim
        k_size = num_kv_heads * head_dim

        q_orig = qkv[:, :q_size].reshape(num_tokens, num_heads, head_dim).clone()
        k_orig = (
            qkv[:, q_size : q_size + k_size]
            .reshape(num_tokens, num_kv_heads, head_dim)
            .clone()
        )

        # Apply reference RoPE (only need Q for comparison)
        # Convert to FP32 for reference computation, then back to FP16 for comparison
        q_ref, _ = apply_rope_reference(
            q_orig.float(), k_orig.float(), cos_sin_cache.float(), positions
        )
        q_ref = q_ref.half()  # Convert back to FP16 to match other implementations

        print(f"  Reference Q shape: {q_ref.shape}")
        print(f"  Reference Q dtype: {q_ref.dtype}")
        print(
            f"  Reference Q range: [{q_ref.min().item():.4f}, {q_ref.max().item():.4f}]"
        )

        # ========== Prepare Common Parameters for C++ and Python ==========
        # These parameters are shared between C++ and Python implementations
        kv_len = [num_tokens]
        num_pages_per_req = torch.tensor(
            [math.ceil(len / token_per_block) for len in kv_len],
            dtype=torch.int32,
            device=self.device,
        )
        kv_append_length = torch.tensor(kv_len, dtype=torch.int32, device=self.device)
        kv_append_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=self.device),
                torch.cumsum(kv_append_length, dim=0),
            ]
        )
        kv_page_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=self.device),
                torch.cumsum(num_pages_per_req, dim=0),
            ]
        )
        kv_page_indices = torch.arange(
            int(sum(num_pages_per_req)), dtype=torch.int32, device=self.device
        )
        kv_last_page_len = torch.tensor(
            [
                (
                    len % token_per_block
                    if len % token_per_block != 0
                    else token_per_block
                )
                for len in kv_len
            ],
            dtype=torch.int32,
            device=self.device,
        )

        # ========== C++ Implementation (FusedRopeKVCachePrefillOpQOut) ==========
        print("\n[2] Testing FusedRopeKVCachePrefillOpQOut (C++)")

        # Create FusedRopeKVCachePrefillOpQOut
        fused_rope_op = FusedRopeKVCachePrefillOpQOut(attn_config)

        # Create PyAttentionInputs for C++ implementation
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor(
            [num_tokens], dtype=torch.int32, device=self.device
        )
        attn_inputs.prefix_lengths = torch.zeros(
            batch_size, dtype=torch.int32, device=self.device
        )
        attn_inputs.sequence_lengths = torch.tensor(
            [num_tokens], dtype=torch.int32, device=self.device
        )
        # Set dtype from qkv tensor
        attn_inputs.dtype = get_typemeta(qkv)

        # Create cu_seqlens: cumulative sequence lengths [0, num_tokens]
        attn_inputs.cu_seqlens_device = torch.tensor(
            [0, num_tokens], dtype=torch.int32, device=self.device
        )
        attn_inputs.cu_kv_seqlens_device = attn_inputs.cu_seqlens_device.clone()

        # Set KV cache block IDs (shape: [batch_size, max_blocks_per_seq])
        # For batch_size=1, reshape kv_page_indices from [num_pages] to [1, num_pages]
        kv_cache_block_id = kv_page_indices.unsqueeze(0).cpu()  # [1, num_pages]
        attn_inputs.kv_cache_block_id = kv_cache_block_id
        attn_inputs.kv_cache_block_id_device = kv_cache_block_id.to(self.device)
        attn_inputs.kv_cache_kernel_block_id = kv_cache_block_id
        attn_inputs.kv_cache_kernel_block_id_device = kv_cache_block_id.to(self.device)

        # Prepare params
        trt_params = fused_rope_op.prepare(attn_inputs)

        # Reset KV cache for fair comparison
        kv_cache_cpp = LayerKVCache()
        kv_cache_cpp.kv_cache_base = torch.zeros_like(kv_cache_base)

        # Run C++ implementation
        q_cpp = fused_rope_op.forward(
            qkv.clone(),
            kv_cache_cpp,
            trt_params,
        )

        print(f"  C++ Q shape: {q_cpp.shape}")
        print(f"  C++ Q dtype: {q_cpp.dtype}")

        # ========== Python Implementation (MhaRotaryEmbeddingOp) ==========
        print("\n[3] Testing MhaRotaryEmbeddingOp (Python)")

        # Create MhaRotaryEmbeddingOp
        mha_rope_op = MhaRotaryEmbeddingOp(attn_config)

        # Create rope_params for Python implementation (using shared parameters)
        batch_indices, _positions_rope = get_batch_indices_positions(
            kv_append_indptr,
            get_seq_lens(kv_page_indptr, kv_last_page_len, token_per_block),
            num_tokens,
        )

        rope_params = RopeParams(
            batch_indice_d=batch_indices,
            positions_d=positions,  # Use same positions as C++
            page_indice_d=kv_page_indices,
            decode_page_indptr_d=kv_page_indptr,
            paged_kv_last_page_len_d=kv_last_page_len,
        )

        # Set params for MhaRotaryEmbeddingOp
        mha_rope_op.set_params(rope_params)

        # Reset KV cache for fair comparison
        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = torch.zeros_like(kv_cache_base)

        # Create KV cache write op
        kv_cache_write_op = KVCacheWriteOp(
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            token_per_block=token_per_block,
        )

        # Set params for KV cache write op
        kv_cache_write_op.set_params(rope_params)

        # Run Python implementation
        q_python, k_python, v_python = mha_rope_op.forward(qkv.clone())

        # Write KV to cache
        kv_cache_write_op.forward(k_python, v_python, kv_cache)

        print(f"  Python Q shape: {q_python.shape}")
        print(f"  Python Q dtype: {q_python.dtype}")

        # ========== Compare Results ==========
        print("\n[4] Comparing Results")
        print("-" * 80)

        # Compare Python vs Reference
        print("\n  [4.1] Python vs Reference")
        q_python_reshaped = q_python.reshape(num_tokens, num_heads, head_dim)
        q_diff_ref = (q_python_reshaped - q_ref).abs()
        q_diff_ref_max = q_diff_ref.max().item()
        q_diff_ref_mean = q_diff_ref.mean().item()
        q_diff_ref_relative = (
            (q_diff_ref_max / q_ref.abs().max().item())
            if q_ref.abs().max().item() > 0
            else 0
        )

        print(f"    Q Max Absolute Difference: {q_diff_ref_max:.6e}")
        print(f"    Q Mean Absolute Difference: {q_diff_ref_mean:.6e}")
        print(f"    Q Relative Difference: {q_diff_ref_relative:.6%}")

        # Compare C++ vs Reference
        print("\n  [4.2] C++ vs Reference")
        q_cpp_reshaped = q_cpp.reshape(num_tokens, num_heads, head_dim)
        q_cpp_diff_ref = (q_cpp_reshaped - q_ref).abs()
        q_cpp_diff_ref_max = q_cpp_diff_ref.max().item()
        q_cpp_diff_ref_mean = q_cpp_diff_ref.mean().item()
        q_cpp_diff_ref_relative = (
            (q_cpp_diff_ref_max / q_ref.abs().max().item())
            if q_ref.abs().max().item() > 0
            else 0
        )

        print(f"    Q Max Absolute Difference: {q_cpp_diff_ref_max:.6e}")
        print(f"    Q Mean Absolute Difference: {q_cpp_diff_ref_mean:.6e}")
        print(f"    Q Relative Difference: {q_cpp_diff_ref_relative:.6%}")

        # Compare Python vs C++
        print("\n  [4.3] Python vs C++")
        q_diff = (q_python - q_cpp).abs()
        q_diff_max = q_diff.max().item()
        q_diff_mean = q_diff.mean().item()
        q_diff_relative = (
            (q_diff_max / q_python.abs().max().item())
            if q_python.abs().max().item() > 0
            else 0
        )

        print(f"    Q Max Absolute Difference: {q_diff_max:.6e}")
        print(f"    Q Mean Absolute Difference: {q_diff_mean:.6e}")
        print(f"    Q Relative Difference: {q_diff_relative:.6%}")
        print(
            f"    Q Python range: [{q_python.min().item():.4f}, {q_python.max().item():.4f}]"
        )
        print(f"    Q C++ range: [{q_cpp.min().item():.4f}, {q_cpp.max().item():.4f}]")

        # Check for NaN
        print("\n  [4.4] NaN Check")
        has_nan_ref = torch.isnan(q_ref).any().item()
        has_nan_python = torch.isnan(q_python).any().item()
        has_nan_cpp = torch.isnan(q_cpp).any().item()
        print(f"    Q Reference has NaN: {has_nan_ref}")
        print(f"    Q Python has NaN: {has_nan_python}")
        print(f"    Q C++ has NaN: {has_nan_cpp}")

        # Compare KV caches
        print("\n  [4.5] KV Cache Comparison")
        k_python = kv_cache.kv_cache_base[:, 0, :, :, :]
        k_cpp = kv_cache_cpp.kv_cache_base[:, 0, :, :, :]
        k_diff_max = (k_python - k_cpp).abs().max().item()

        v_python = kv_cache.kv_cache_base[:, 1, :, :, :]
        v_cpp = kv_cache_cpp.kv_cache_base[:, 1, :, :, :]
        v_diff_max = (v_python - v_cpp).abs().max().item()

        print(f"    K Cache Max Difference (Python vs C++): {k_diff_max:.6e}")
        print(f"    V Cache Max Difference (Python vs C++): {v_diff_max:.6e}")

        # Assertions
        print("\n[5] Validation")
        print("-" * 80)

        self.assertFalse(has_nan_ref, "Reference implementation produced NaN")
        self.assertFalse(has_nan_python, "Python implementation produced NaN")
        self.assertFalse(has_nan_cpp, "C++ implementation produced NaN")

        # Allow for reasonable floating point differences (FP16 precision ~1e-3)
        tolerance_rtol = 1e-2  # 1% relative tolerance
        tolerance_atol = 1e-2  # absolute tolerance for FP16

        # Check Python vs Reference
        q_python_vs_ref_match = torch.allclose(
            q_python_reshaped, q_ref, rtol=tolerance_rtol, atol=tolerance_atol
        )

        # Check C++ vs Reference
        q_cpp_vs_ref_match = torch.allclose(
            q_cpp_reshaped, q_ref, rtol=tolerance_rtol, atol=tolerance_atol
        )

        # Check Python vs C++
        q_python_vs_cpp_match = torch.allclose(
            q_python, q_cpp, rtol=tolerance_rtol, atol=tolerance_atol
        )

        print("\n  Results:")
        if q_python_vs_ref_match:
            print(f"  ✓ Python vs Reference MATCH (max_diff={q_diff_ref_max:.6e})")
        else:
            print(
                f"  ✗ Python vs Reference DO NOT MATCH (max_diff={q_diff_ref_max:.6e}, relative={q_diff_ref_relative:.6%})"
            )

        if q_cpp_vs_ref_match:
            print(f"  ✓ C++ vs Reference MATCH (max_diff={q_cpp_diff_ref_max:.6e})")
        else:
            print(
                f"  ✗ C++ vs Reference DO NOT MATCH (max_diff={q_cpp_diff_ref_max:.6e}, relative={q_cpp_diff_ref_relative:.6%})"
            )

        if q_python_vs_cpp_match:
            print(f"  ✓ Python vs C++ MATCH (max_diff={q_diff_max:.6e})")
        else:
            print(
                f"  ✗ Python vs C++ DO NOT MATCH (max_diff={q_diff_max:.6e}, relative={q_diff_relative:.6%})"
            )

            # Print sample of differences for debugging
            print(f"\n  Sample differences (first 5 tokens, first head):")
            for i in range(min(5, num_tokens)):
                sample_diff = q_diff[i, 0, :5].cpu().tolist()
                print(f"    Token {i}: {sample_diff}")

        # Assertions
        self.assertTrue(
            q_python_vs_ref_match,
            f"Python Q should match reference: max_diff={q_diff_ref_max:.6e}, relative={q_diff_ref_relative:.6%}",
        )
        self.assertTrue(
            q_cpp_vs_ref_match,
            f"C++ Q should match reference: max_diff={q_cpp_diff_ref_max:.6e}, relative={q_cpp_diff_ref_relative:.6%}",
        )
        self.assertTrue(
            q_python_vs_cpp_match,
            f"Python and C++ Q should match: max_diff={q_diff_max:.6e}, relative={q_diff_relative:.6%}",
        )

        print(
            "\n✓ All implementations (Reference, Python, C++) produce consistent results"
        )
        print("=" * 80)

    def _write_fp8_reference_cache(
        self,
        cache: LayerKVCache,
        key: torch.Tensor,
        value: torch.Tensor,
        input_lengths: list[int],
        prefix_lengths: list[int],
        block_ids: torch.Tensor,
        page_size: int,
    ) -> None:
        """Write RoPE output with the fused op's saturating E4M3 contract."""
        token_index = 0
        for batch_index, input_length in enumerate(input_lengths):
            for request_token_index in range(input_length):
                cache_position = prefix_lengths[batch_index] + request_token_index
                page_index = int(
                    block_ids[batch_index, cache_position // page_size].item()
                )
                page_offset = cache_position % page_size
                cache.kv_cache_base[page_index, 0, :, page_offset, :] = (
                    key[token_index].clamp(-448, 448).to(torch.float8_e4m3fn)
                )
                cache.kv_cache_base[page_index, 1, :, page_offset, :] = (
                    value[token_index].clamp(-448, 448).to(torch.float8_e4m3fn)
                )
                token_index += 1

    def test_fused_paged_prefill_fp8_accuracy(self):
        """Compare fused paged prefill with an explicit FP8 cache reference."""
        if not self.device_initialized:
            self.skipTest("device initialization is required")

        batch_size = 4
        seq_len = 512
        sequence_lengths = [seq_len] * batch_size
        total_tokens = sum(sequence_lengths)
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        page_size = 64
        hidden_size = (num_heads + 2 * num_kv_heads) * head_dim

        config = create_test_attn_config(
            head_num=num_heads,
            kv_head_num=num_kv_heads,
            size_per_head=head_dim,
            tokens_per_block=page_size,
            dtype=torch.bfloat16,
        )
        config.need_rope_kv_cache = True
        config.kv_cache_dtype = KvCacheDataType.FP8
        config.is_causal = True

        input_lengths = torch.tensor(sequence_lengths, dtype=torch.int32)
        cu_seqlens = torch.arange(
            0,
            total_tokens + 1,
            seq_len,
            dtype=torch.int32,
            device=self.device,
        )
        pages_per_request = math.ceil(seq_len / page_size)
        block_ids = torch.arange(
            batch_size * pages_per_request, dtype=torch.int32
        ).reshape(batch_size, pages_per_request)
        positions = torch.arange(seq_len, dtype=torch.int32, device=self.device).repeat(
            batch_size
        )

        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = input_lengths.pin_memory()
        attn_inputs.prefix_lengths = torch.zeros(
            batch_size, dtype=torch.int32
        ).pin_memory()
        attn_inputs.sequence_lengths = input_lengths.pin_memory()
        attn_inputs.kv_cache_block_id = block_ids
        attn_inputs.kv_cache_block_id_device = block_ids.to(self.device)
        attn_inputs.kv_cache_kernel_block_id = block_ids
        attn_inputs.kv_cache_kernel_block_id_device = block_ids.to(self.device)
        attn_inputs.cu_seqlens_device = cu_seqlens
        attn_inputs.cu_kv_seqlens_device = cu_seqlens
        attn_inputs.combo_position_ids = positions
        attn_inputs.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))

        qkv_source = (
            torch.randn(
                total_tokens,
                hidden_size,
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.5
        )
        num_pages = batch_size * pages_per_request

        def make_cache() -> LayerKVCache:
            cache = LayerKVCache()
            cache.kv_cache_base = torch.zeros(
                num_pages,
                2,
                num_kv_heads,
                page_size,
                head_dim,
                dtype=torch.float8_e4m3fn,
                device=self.device,
            )
            cache.kv_scale_base = torch.ones(
                num_pages,
                2 * num_kv_heads * page_size,
                dtype=torch.float32,
                device=self.device,
            )
            return cache

        # Reference pipeline: Python RoPE plus explicit saturating FP8 writes.
        split_attn = PyFlashinferPrefillPagedAttnOp(config, attn_inputs)
        split_params = split_attn.prepare(attn_inputs)
        split_rope = MhaRotaryEmbeddingOp(config)
        split_rope.set_params(split_params)
        split_cache = make_cache()

        # Proposed fused pipeline wired through the production implementation.
        fused_impl = PyFlashinferPagedPrefillImpl(config, attn_inputs)
        fused_cache = make_cache()

        split_input = qkv_source.clone()
        split_q, split_k, split_v = split_rope.forward(split_input)
        self._write_fp8_reference_cache(
            split_cache,
            split_k,
            split_v,
            sequence_lengths,
            [0] * batch_size,
            block_ids,
            page_size,
        )
        split_output = split_attn.forward(split_q, split_cache)

        fused_output = fused_impl.forward(qkv_source.clone(), fused_cache)

        torch.testing.assert_close(
            fused_cache.kv_cache_base.float(),
            split_cache.kv_cache_base.float(),
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            fused_output.float(), split_output.float(), rtol=2e-2, atol=2e-2
        )

    def test_fused_ragged_prefill_fp8_accuracy(self):
        """Compare fused ragged prefill with an explicit FP8 cache reference."""
        if not self.device_initialized:
            self.skipTest("device initialization is required")

        batch_size = 4
        seq_len = 512
        total_tokens = batch_size * seq_len
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128
        page_size = 64
        hidden_size = (num_heads + 2 * num_kv_heads) * head_dim

        config = create_test_attn_config(
            head_num=num_heads,
            kv_head_num=num_kv_heads,
            size_per_head=head_dim,
            tokens_per_block=page_size,
            dtype=torch.bfloat16,
        )
        config.need_rope_kv_cache = True
        config.kv_cache_dtype = KvCacheDataType.FP8
        config.is_causal = True

        lengths = torch.full((batch_size,), seq_len, dtype=torch.int32)
        cu_seqlens = torch.arange(
            0,
            total_tokens + 1,
            seq_len,
            dtype=torch.int32,
            device=self.device,
        )
        pages_per_request = math.ceil(seq_len / page_size)
        block_ids = torch.arange(
            batch_size * pages_per_request, dtype=torch.int32
        ).reshape(batch_size, pages_per_request)

        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = lengths.pin_memory()
        attn_inputs.prefix_lengths = torch.zeros(
            batch_size, dtype=torch.int32
        ).pin_memory()
        attn_inputs.sequence_lengths = lengths.pin_memory()
        attn_inputs.kv_cache_block_id = block_ids
        attn_inputs.kv_cache_block_id_device = block_ids.to(self.device)
        attn_inputs.kv_cache_kernel_block_id = block_ids
        attn_inputs.kv_cache_kernel_block_id_device = block_ids.to(self.device)
        attn_inputs.cu_seqlens_device = cu_seqlens
        attn_inputs.cu_kv_seqlens_device = cu_seqlens
        attn_inputs.combo_position_ids = torch.arange(
            seq_len, dtype=torch.int32, device=self.device
        ).repeat(batch_size)
        attn_inputs.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))

        qkv_source = (
            torch.randn(
                total_tokens,
                hidden_size,
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.5
        )
        num_pages = batch_size * pages_per_request

        def make_cache() -> LayerKVCache:
            cache = LayerKVCache()
            cache.kv_cache_base = torch.zeros(
                num_pages,
                2,
                num_kv_heads,
                page_size,
                head_dim,
                dtype=torch.float8_e4m3fn,
                device=self.device,
            )
            cache.kv_scale_base = torch.ones(
                num_pages,
                2 * num_kv_heads * page_size,
                dtype=torch.float32,
                device=self.device,
            )
            return cache

        split_attn = PyFlashinferPrefillAttnOp(config)
        split_params = split_attn.prepare(attn_inputs)
        split_rope = MhaRotaryEmbeddingOp(config)
        split_rope.set_params(split_params)
        split_cache = make_cache()

        fused_impl = PyFlashinferPrefillImpl(config, attn_inputs)
        fused_cache = make_cache()

        split_input = qkv_source.clone()
        split_q, split_k, split_v = split_rope.forward(split_input)
        self._write_fp8_reference_cache(
            split_cache,
            split_k,
            split_v,
            [seq_len] * batch_size,
            [0] * batch_size,
            block_ids,
            page_size,
        )
        split_qkv = torch.cat(
            [
                split_q.flatten(1),
                split_k.flatten(1),
                split_v.flatten(1),
            ],
            dim=-1,
        )
        split_output = split_attn.forward(split_qkv, split_cache)
        fused_output = fused_impl.forward(qkv_source.clone(), fused_cache)

        torch.testing.assert_close(
            fused_cache.kv_cache_base.float(),
            split_cache.kv_cache_base.float(),
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            fused_output.float(), split_output.float(), rtol=2e-2, atol=2e-2
        )

    def test_fused_paged_prefill_prefix_and_reused_blocks(self):
        """Fused writes must preserve prefix offsets and reordered block IDs."""
        if not self.device_initialized:
            self.skipTest("device initialization is required")

        input_lengths = [5, 3]
        prefix_lengths = [8, 0]
        sequence_lengths = [13, 3]
        total_tokens = sum(input_lengths)
        num_heads = 8
        num_kv_heads = 2
        head_dim = 128
        page_size = 8
        num_pages = 6

        config = create_test_attn_config(
            head_num=num_heads,
            kv_head_num=num_kv_heads,
            size_per_head=head_dim,
            tokens_per_block=page_size,
            dtype=torch.bfloat16,
        )
        config.need_rope_kv_cache = True
        config.kv_cache_dtype = KvCacheDataType.FP8
        config.is_causal = True

        block_ids = torch.tensor([[5, 4], [2, 3]], dtype=torch.int32)
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.input_lengths = torch.tensor(
            input_lengths, dtype=torch.int32
        ).pin_memory()
        attn_inputs.prefix_lengths = torch.tensor(
            prefix_lengths, dtype=torch.int32
        ).pin_memory()
        attn_inputs.sequence_lengths = torch.tensor(
            sequence_lengths, dtype=torch.int32
        ).pin_memory()
        attn_inputs.kv_cache_block_id = block_ids
        attn_inputs.kv_cache_block_id_device = block_ids.to(self.device)
        attn_inputs.kv_cache_kernel_block_id = block_ids
        attn_inputs.kv_cache_kernel_block_id_device = block_ids.to(self.device)
        attn_inputs.cu_seqlens_device = torch.tensor(
            [0, 5, 8], dtype=torch.int32, device=self.device
        )
        attn_inputs.cu_kv_seqlens_device = attn_inputs.cu_seqlens_device
        attn_inputs.combo_position_ids = torch.tensor(
            [8, 9, 10, 11, 12, 0, 1, 2],
            dtype=torch.int32,
            device=self.device,
        )
        attn_inputs.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))

        qkv = torch.randn(
            total_tokens,
            (num_heads + 2 * num_kv_heads) * head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        value_start = (num_heads + num_kv_heads) * head_dim
        qkv[0::2, value_start:] = 1000
        qkv[1::2, value_start:] = -1000
        cache_seed = torch.randn(
            num_pages,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        ).to(torch.float8_e4m3fn)

        def make_cache() -> LayerKVCache:
            cache = LayerKVCache()
            cache.kv_cache_base = cache_seed.clone()
            cache.kv_scale_base = torch.ones(
                num_pages,
                2 * num_kv_heads * page_size,
                dtype=torch.float32,
                device=self.device,
            )
            return cache

        split_attn = PyFlashinferPrefillPagedAttnOp(config, attn_inputs)
        split_params = split_attn.prepare(attn_inputs)
        split_rope = MhaRotaryEmbeddingOp(config)
        split_rope.set_params(split_params)
        split_cache = make_cache()
        query, key, value = split_rope.forward(qkv.clone())
        self._write_fp8_reference_cache(
            split_cache,
            key,
            value,
            input_lengths,
            prefix_lengths,
            block_ids,
            page_size,
        )
        split_output = split_attn.forward(query, split_cache)

        fused_cache = make_cache()
        fused_impl = PyFlashinferPagedPrefillImpl(config, attn_inputs)
        fused_output = fused_impl.forward(qkv.clone(), fused_cache)

        torch.testing.assert_close(
            fused_cache.kv_cache_base.float(),
            split_cache.kv_cache_base.float(),
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            fused_output.float(), split_output.float(), rtol=2e-2, atol=2e-2
        )

    def test_fused_paged_prefill_fp8_cuda_graph_replay(self):
        """Capture and replay fused preprocessing with a changed block table."""
        if not self.device_initialized:
            self.skipTest("device initialization is required")

        batch_size = 2
        seq_len = 16
        page_size = 16
        num_heads = 8
        num_kv_heads = 2
        head_dim = 128
        total_tokens = batch_size * seq_len

        config = create_test_attn_config(
            head_num=num_heads,
            kv_head_num=num_kv_heads,
            size_per_head=head_dim,
            tokens_per_block=page_size,
            dtype=torch.bfloat16,
        )
        config.need_rope_kv_cache = True
        config.kv_cache_dtype = KvCacheDataType.FP8
        config.is_causal = True

        def make_inputs(block_ids: torch.Tensor, is_graph: bool) -> PyAttentionInputs:
            inputs = PyAttentionInputs()
            inputs.is_prefill = True
            inputs.is_cuda_graph = is_graph
            lengths = torch.full((batch_size,), seq_len, dtype=torch.int32)
            inputs.input_lengths = lengths.pin_memory()
            inputs.prefix_lengths = torch.zeros(
                batch_size, dtype=torch.int32
            ).pin_memory()
            inputs.sequence_lengths = lengths.pin_memory()
            inputs.kv_cache_block_id = block_ids
            inputs.kv_cache_block_id_device = block_ids.to(self.device)
            inputs.kv_cache_kernel_block_id = block_ids
            inputs.kv_cache_kernel_block_id_device = block_ids.to(self.device)
            inputs.cu_seqlens_device = torch.tensor(
                [0, seq_len, total_tokens],
                dtype=torch.int32,
                device=self.device,
            )
            inputs.cu_kv_seqlens_device = inputs.cu_seqlens_device
            inputs.combo_position_ids = torch.arange(
                seq_len, dtype=torch.int32, device=self.device
            ).repeat(batch_size)
            inputs.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
            return inputs

        def make_cache() -> LayerKVCache:
            cache = LayerKVCache()
            cache.kv_cache_base = torch.zeros(
                batch_size,
                2,
                num_kv_heads,
                page_size,
                head_dim,
                dtype=torch.float8_e4m3fn,
                device=self.device,
            )
            cache.kv_scale_base = torch.ones(
                batch_size,
                2 * num_kv_heads * page_size,
                dtype=torch.float32,
                device=self.device,
            )
            return cache

        capture_blocks = torch.tensor([[0], [1]], dtype=torch.int32)
        replay_blocks = torch.tensor([[1], [0]], dtype=torch.int32)
        capture_inputs = make_inputs(capture_blocks, True)
        replay_inputs = make_inputs(replay_blocks, True)
        graph_impl = PyFlashinferPagedPrefillImpl(config, capture_inputs)
        graph_cache = make_cache()
        static_qkv = torch.randn(
            total_tokens,
            (num_heads + 2 * num_kv_heads) * head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                graph_impl.forward(static_qkv, graph_cache)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = graph_impl.forward(static_qkv, graph_cache)

        replay_qkv = torch.randn_like(static_qkv)
        graph_impl.prepare_cuda_graph(replay_inputs)
        static_qkv.copy_(replay_qkv)
        graph_cache.kv_cache_base.zero_()
        graph.replay()
        torch.cuda.synchronize()

        eager_inputs = make_inputs(replay_blocks, False)
        eager_impl = PyFlashinferPagedPrefillImpl(config, eager_inputs)
        eager_cache = make_cache()
        eager_output = eager_impl.forward(replay_qkv, eager_cache)

        torch.testing.assert_close(
            graph_cache.kv_cache_base.float(),
            eager_cache.kv_cache_base.float(),
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            graph_output.float(), eager_output.float(), rtol=2e-2, atol=2e-2
        )


if __name__ == "__main__":
    unittest.main()
