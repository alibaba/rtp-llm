import os
import unittest
import torch
from packaging import version
import flashinfer
from flashinfer.utils import get_compute_capability


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
}

GPU_DEVICE = "cuda:0"

# Check version requirements
def check_flashinfer_version():
    """Check if flashinfer version >= 0.5.2"""
    try:
        flashinfer_version = version.parse(flashinfer.__version__)
        return flashinfer_version >= version.parse("0.5.2")
    except:
        return False

def check_cuda_version():
    """Check if CUDA version >= 12.8"""
    try:
        cuda_version_str = torch.version.cuda
        if cuda_version_str:
            cuda_version = version.parse(cuda_version_str)
            return cuda_version >= version.parse("12.8")
        return False
    except:
        return False

# Global version check results
FLASHINFER_VERSION_OK = check_flashinfer_version()
CUDA_VERSION_OK = check_cuda_version()
VERSION_REQUIREMENTS_MET = FLASHINFER_VERSION_OK and CUDA_VERSION_OK

# Skip reason
SKIP_REASON = []
if not FLASHINFER_VERSION_OK:
    SKIP_REASON.append(f"flashinfer version {flashinfer.__version__} < 0.5.2")
if not CUDA_VERSION_OK:
    SKIP_REASON.append(f"CUDA version {torch.version.cuda} < 12.8")
SKIP_MESSAGE = "Requirements not met: " + ", ".join(SKIP_REASON) if SKIP_REASON else ""


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def generate_seq_lens_decode(batch_size, q_len_per_req, max_in_kv_len):
    """Generate consistent sequence lengths for all requests"""
    q_lens = torch.full((batch_size,), q_len_per_req, dtype=torch.int32)
    in_kv_lens = torch.full((batch_size,), max_in_kv_len, dtype=torch.int32)  # All same length
    seq_lens = q_lens + in_kv_lens  # Total KV length
    return q_lens, in_kv_lens, seq_lens


def generate_cumsum_lens(lens):
    return torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=GPU_DEVICE),
            torch.cumsum(lens.to(GPU_DEVICE), dim=0, dtype=torch.int32),
        ]
    )

def create_output(q, o_dtype):
    """Create output tensor for the given query and output dtype."""
    if o_dtype == "fp8":
        o_scale = torch.rand(1).item() * 0.5 + 0.5  # Scale range: 0.5 ~ 1.0
        out = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=q.device)
    else:
        o_scale = 1.0
        out = torch.empty(q.shape, dtype=DTYPE_MAP[o_dtype], device=q.device)

    return out, o_scale

def create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype):
    q = torch.randn(
        torch.sum(q_lens).item(),
        num_qo_heads,
        head_dim,
        dtype=torch.bfloat16 if q_dtype == "fp8" else DTYPE_MAP[q_dtype],
        device=GPU_DEVICE,
    )
    if q_dtype == "fp8":
        q, q_scale = to_float8(q)
        # Reference implementation have functional issue or low precision with fp8, use bfloat16 and fake-quantization instead.
        ref_q = q.bfloat16() * q_scale
    else:
        q_scale = 1.0
        ref_q = q

    return q, q_scale, ref_q


def create_kv_cache(
    batch_size,
    seq_lens,
    page_size,
    num_kv_heads,
    head_dim,
    kv_dtype,
    kv_layout="HND",
):
    """Create KV cache in HND layout for XQADecodeImpl"""
    max_seq_len = torch.max(seq_lens).item()
    num_pages_per_seq = (max_seq_len + page_size - 1) // page_size
    num_pages = num_pages_per_seq * batch_size
    # For fp8, create in bfloat16 first, then convert
    kv_dtype_torch = DTYPE_MAP["bf16"] if kv_dtype == "fp8" else DTYPE_MAP[kv_dtype]
    # HND layout: [num_pages, num_kv_heads, page_size, head_dim]
    k_cache = torch.randn(
        num_pages,
        num_kv_heads,
        page_size,
        head_dim,
        dtype=kv_dtype_torch,
        device=GPU_DEVICE,
    )
    v_cache = torch.randn(
        num_pages,
        num_kv_heads,
        page_size,
        head_dim,
        dtype=kv_dtype_torch,
        device=GPU_DEVICE,
    )
    k_scale = 1.0
    v_scale = 1.0
    if kv_dtype == "fp8":
        k_cache, k_scale = to_float8(k_cache / 4.0)
        v_cache, v_scale = to_float8(v_cache / 4.0)
        # Create reference KV cache in bfloat16 for reference implementation
        ref_kv_cache = torch.stack([
            k_cache.to(torch.bfloat16) * k_scale,
            v_cache.to(torch.bfloat16) * v_scale,
        ], dim=1)
    else:
        ref_kv_cache = None
    
    # Stack K and V: [num_pages, 2, num_kv_heads, page_size, head_dim]
    kv_cache = torch.stack([k_cache, v_cache], dim=1)
    
    return kv_cache, k_scale, v_scale, ref_kv_cache


def create_page_table(batch_size, seq_lens, page_size):
    """Create page table for KV cache"""
    seq_lens = seq_lens.to(GPU_DEVICE)
    page_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_pages_per_seq = torch.max(page_per_seq).item()
    
    # Generate sequential page IDs
    total_pages_needed = torch.sum(page_per_seq).item()
    all_page_ids = torch.arange(
        total_pages_needed, dtype=torch.int32, device=GPU_DEVICE
    )
    
    # Use cumsum to create page offsets for each sequence
    page_offsets = torch.cat(
        [
            torch.tensor([0], device=GPU_DEVICE, dtype=torch.int32),
            torch.cumsum(page_per_seq[:-1], dim=0, dtype=torch.int32),
        ]
    )
    
    # Create page tables using broadcasting
    page_idx_range = torch.arange(
        max_num_pages_per_seq, device=GPU_DEVICE, dtype=torch.int32
    ).unsqueeze(0)
    page_tables = page_offsets.unsqueeze(1) + page_idx_range
    
    return page_tables, all_page_ids, page_per_seq


def create_reference_output(
    q, kv_cache, ref_kv_cache, page_table, seq_lens, page_size, num_kv_heads, head_dim, q_len_per_req
):
    """Create reference output using flashinfer decode API"""
    batch_size = page_table.shape[0]
    num_qo_heads = q.shape[1]
    
    # Use ref_kv_cache for reference if available (for fp8), otherwise use kv_cache
    kv_for_ref = ref_kv_cache if ref_kv_cache is not None else kv_cache
    
    # Prepare workspace buffer
    workspace_buffer = torch.empty(
        256 * 1024 * 1024, dtype=torch.int8, device=GPU_DEVICE
    )
    
    # Calculate page_per_seq
    page_per_seq = (seq_lens + page_size - 1) // page_size
    
    # Create indices and last_page_len
    kv_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=GPU_DEVICE),
            torch.cumsum(page_per_seq, dim=0, dtype=torch.int32),
        ]
    )
    
    all_page_ids = page_table.reshape(-1)
    valid_page_count = kv_indptr[-1].item()
    all_page_ids = all_page_ids[:valid_page_count]
    
    last_page_len = seq_lens % page_size
    last_page_len = torch.where(last_page_len == 0, page_size, last_page_len)
    
    # Use flashinfer BatchDecodeWithPagedKVCacheWrapper
    if q_len_per_req == 1:
        wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, "HND", use_tensor_cores=True
        )
        wrapper.plan(
            indptr=kv_indptr,
            indices=all_page_ids,
            last_page_len=last_page_len.to(GPU_DEVICE),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            pos_encoding_mode="NONE",
            kv_data_type=kv_for_ref.dtype,
            q_data_type=q.dtype,
            window_left=-1,
        )
        # q shape: [total_tokens, num_heads, head_dim]
        output_ref = wrapper.run(q, kv_for_ref)
    else:
        # speculative decoding: use prefill wrapper
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "HND"
        )
        q_indptr = generate_cumsum_lens(torch.full((batch_size,), q_len_per_req, dtype=torch.int32))
        
        wrapper.plan(
            qo_indptr=q_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=all_page_ids,
            paged_kv_last_page_len=last_page_len.to(GPU_DEVICE),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=True,
            logits_soft_cap=0.0,
            q_data_type=q.dtype,
            kv_data_type=kv_for_ref.dtype,
        )
        output_ref = wrapper.run(q, kv_for_ref)
    
    return output_ref


class TestXQABatchDecode(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        import faulthandler
        import signal
        
        faulthandler.enable()
        signal.signal(signal.SIGSEGV, signal.SIG_DFL)
        signal.signal(signal.SIGABRT, signal.SIG_DFL)
        
        super().__init__(methodName)
        
        self.compute_capability = get_compute_capability(torch.device(device="cuda"))[0]
        self.xqa_supported = self.compute_capability in [9, 10, 12]
    
    @classmethod
    def setUpClass(cls):
        from rtp_llm.ops.compute_ops import init_device
        from rtp_llm.ops import (
            ParallelismConfig, ModelConfig, EPLBConfig, FMHAConfig,
            DeviceResourceConfig, MoeConfig, SpeculativeExecutionConfig,
            MiscellaneousConfig, ProfilingDebugLoggingConfig, HWKernelConfig,
            ConcurrencyConfig, FfnDisAggregateConfig, RuntimeConfig
        )
        model_config = ModelConfig()
        model_config.attn_config.head_num = 8 
        model_config.attn_config.kv_head_num = 1
        model_config.attn_config.size_per_head = 128
        model_config.attn_config.tokens_per_block = 64
        model_config.max_seq_len = 2048
        init_device(
            parallelism_config=ParallelismConfig(),
            model_config=model_config,
            eplb_config=EPLBConfig(),
            fmha_config=FMHAConfig(),
            device_resource_config=DeviceResourceConfig(),
            moe_config=MoeConfig(),
            sp_config=SpeculativeExecutionConfig(),
            misc_config=MiscellaneousConfig(),
            profiling_debug_logging_config=ProfilingDebugLoggingConfig(),
            hw_kernel_config=HWKernelConfig(),
            concurrency_config=ConcurrencyConfig(),
            ffn_disaggregate_config=FfnDisAggregateConfig(),
            runtime_config=RuntimeConfig(),
        )
        
    def _test_xqa_decode_impl(
        self,
        batch_size,
        q_len_per_req,
        page_size,
        num_kv_heads,
        head_grp_size,
        q_dtype,
        o_dtype,
        kv_dtype,
        max_in_kv_len,
        head_dim,
    ):
        """Test XQADecodeImpl forward pass"""
        
        if not VERSION_REQUIREMENTS_MET:
            self.skipTest(SKIP_MESSAGE)
        
        torch.manual_seed(0)
        num_qo_heads = num_kv_heads * head_grp_size
        
        q_lens, in_kv_lens, seq_lens = generate_seq_lens_decode(
            batch_size, q_len_per_req, max_in_kv_len
        )
        
        # Create query tensor [total_tokens, num_heads, head_dim]
        q, q_scale, ref_q = create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype)
        
        # Create KV cache in HND layout
        kv_cache_tensor, k_scale, v_scale, ref_kv_cache_tensor = create_kv_cache(
            batch_size, seq_lens, page_size, num_kv_heads, head_dim, kv_dtype, "HND"
        )
        page_table, all_page_ids, page_per_seq = create_page_table(
            batch_size, seq_lens, page_size
        )
        output_4d, o_scale = create_output(q, o_dtype)

        sm_scale = float(1.0 / (head_dim**0.5))
        from rtp_llm.ops.compute_ops import PyAttentionInputs, KVCache, get_typemeta
        from rtp_llm.ops import AttentionConfigs, KvCacheDataType
        
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = False
        attn_inputs.sequence_lengths = in_kv_lens
        attn_inputs.input_lengths = q_lens 
        attn_inputs.kv_cache_block_id_device = page_table 
        attn_inputs.dtype = get_typemeta(q) 
        attn_inputs.total_tokens = q.shape[0]
        attn_inputs.decode_cu_seqlens_d = generate_cumsum_lens(q_lens)
        attn_inputs.cu_seqlens = generate_cumsum_lens(q_lens).cpu()
        attn_inputs.cu_kv_seqlens = generate_cumsum_lens(seq_lens).cpu()
        attn_configs = AttentionConfigs()
        attn_configs.head_num = num_qo_heads
        attn_configs.kv_head_num = num_kv_heads
        attn_configs.size_per_head = head_dim
        attn_configs.tokens_per_block = page_size
        attn_configs.kv_cache_dtype = KvCacheDataType.FP8 if kv_dtype == "fp8" else KvCacheDataType.BASE
        attn_configs.dtype = q.dtype
        
        kv_cache = KVCache()
        kv_cache.kv_cache_base = kv_cache_tensor
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa import XQADecodeImpl
        from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
        
        original_init = FMHAImplBase.__init__
        
        def patched_init(self, fmha_impl, rope_kvcache_impl, attn_inputs, init_params=True):
            result = original_init(self, fmha_impl, rope_kvcache_impl, attn_inputs, init_params=False)
            self.rope_kvcache_impl = rope_kvcache_impl
            return result
        
        FMHAImplBase.__init__ = patched_init
        
        try:
            xqa_impl = XQADecodeImpl(attn_configs, attn_inputs)
            # Prepare fmha_params with scale parameters
            if kv_dtype == "fp8":
                xqa_impl.fmha_params = xqa_impl.fmha_impl.prepare(
                    attn_inputs, 
                    q_scale=q_scale * k_scale * sm_scale,
                    kv_scale=v_scale / o_scale,
                    o_scale=o_scale
                )
            else:
                xqa_impl.fmha_params = xqa_impl.fmha_impl.prepare(attn_inputs)
            xqa_impl.attn_inputs = attn_inputs
            class DummyRopeParams:
                pass
            xqa_impl.rope_params = DummyRopeParams()
        finally:
            FMHAImplBase.__init__ = original_init
     
        output_4d = xqa_impl.forward(q, kv_cache, need_rope_kv_cache=False).squeeze(1)
        output = output_4d.reshape(-1, output_4d.shape[2], output_4d.shape[3])
        output_ref = create_reference_output(
            q, kv_cache_tensor, ref_kv_cache_tensor, page_table, seq_lens.to(GPU_DEVICE), 
            page_size, num_kv_heads, head_dim, q_len_per_req
        )
        assert output.shape == output_ref.shape, \
            f"Output shape mismatch: XQA={output.shape}, ref={output_ref.shape}"
        
        max_diff = (output.float() - output_ref.float()).abs().max().item()
        mean_diff = (output.float() - output_ref.float()).abs().mean().item()
        
        self.assertTrue(
            torch.allclose(
                output.float(),
                output_ref.float(),
                rtol=1e-1 if kv_dtype == "fp8" else 1e-2,
                atol=2e-1 if kv_dtype == "fp8" else 1e-2,
            ),
            f"Output mismatch: max diff = {max_diff}, mean diff = {mean_diff}, "
            f"output shape = {output.shape}, ref shape = {output_ref.shape}"
        )

    @unittest.skipIf(not VERSION_REQUIREMENTS_MET, SKIP_MESSAGE)
    def test_xqa_decode_comprehensive(self):
        """Run comprehensive test cases for XQADecodeImpl"""
        
        test_cases = [
            # (batch_size, q_len_per_req, num_kv_heads, kv_dtype)
            (2, 1, 1, "fp8"), 
            (2, 5, 1, "fp8"), 
            (2, 5, 1, "bf16"),
            (2, 1, 1, "bf16"),
        ]
        
        for batch_size, q_len_per_req, num_kv_heads, kv_dtype in test_cases:
            with self.subTest(bs=batch_size, q_len=q_len_per_req, kv_dtype=kv_dtype):
                self._test_xqa_decode_impl(
                    batch_size=batch_size,
                    q_len_per_req=q_len_per_req,
                    page_size=64,
                    num_kv_heads=num_kv_heads,
                    head_grp_size=8,
                    q_dtype="bf16",
                    o_dtype="bf16",
                    kv_dtype=kv_dtype,
                    max_in_kv_len=10,
                    head_dim=128,
                )


if __name__ == "__main__":
    unittest.main()
