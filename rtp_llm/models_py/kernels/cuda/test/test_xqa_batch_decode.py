import faulthandler
import os
import signal
import unittest

import torch
from packaging import version

try:
    import flashinfer
    from flashinfer.utils import get_compute_capability

    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False

from rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa import XQADecodeImpl
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ModelConfig

# RTP-LLM imports
from rtp_llm.ops.compute_ops import (
    KVCache,
    LayerKVCache,
    PyAttentionInputs,
    get_typemeta,
    init_exec_ctx,
)

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
}

GPU_DEVICE = "cuda:0"


# Check version requirements
def check_flashinfer_version():
    """Check if flashinfer version >= 0.5.2"""
    if not _FLASHINFER_AVAILABLE:
        return False
    try:
        flashinfer_version = version.parse(flashinfer.__version__)
        return flashinfer_version >= version.parse("0.5.2")
    except Exception:
        return False


def check_cuda_version():
    """Check if CUDA version >= 12.8"""
    try:
        cuda_version_str = torch.version.cuda
        if cuda_version_str:
            cuda_version = version.parse(cuda_version_str)
            return cuda_version >= version.parse("12.8")
        return False
    except Exception:
        return False


# Global version check results
FLASHINFER_VERSION_OK = check_flashinfer_version()
CUDA_VERSION_OK = check_cuda_version()
VERSION_REQUIREMENTS_MET = FLASHINFER_VERSION_OK and CUDA_VERSION_OK

# Skip reason
SKIP_REASON = []
if not FLASHINFER_VERSION_OK:
    fi_ver = flashinfer.__version__ if _FLASHINFER_AVAILABLE else "not installed"
    SKIP_REASON.append(f"flashinfer version {fi_ver} < 0.5.2")
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
    in_kv_lens = torch.full(
        (batch_size,), max_in_kv_len, dtype=torch.int32
    )  # All same length
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
        ref_kv_cache = torch.stack(
            [
                k_cache.to(torch.bfloat16) * k_scale,
                v_cache.to(torch.bfloat16) * v_scale,
            ],
            dim=1,
        )
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
    q,
    kv_cache,
    ref_kv_cache,
    page_table,
    seq_lens,
    page_size,
    num_kv_heads,
    head_dim,
    q_len_per_req,
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
        q_indptr = generate_cumsum_lens(
            torch.full((batch_size,), q_len_per_req, dtype=torch.int32)
        )

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
        faulthandler.enable()
        signal.signal(signal.SIGSEGV, signal.SIG_DFL)
        signal.signal(signal.SIGABRT, signal.SIG_DFL)

        super().__init__(methodName)

        self.compute_capability = (
            get_compute_capability(torch.device(device="cuda"))[0]
            if _FLASHINFER_AVAILABLE
            else 0
        )
        self.xqa_supported = self.compute_capability in [9, 10, 12]

    @classmethod
    def setUpClass(cls):
        model_config = ModelConfig()
        model_config.attn_config.head_num = 8
        model_config.attn_config.kv_head_num = 1
        model_config.attn_config.size_per_head = 128
        model_config.attn_config.tokens_per_block = 64
        model_config.attn_config.kernel_tokens_per_block = 64
        model_config.max_seq_len = 2048

        init_exec_ctx(
            device_id=0,
            trace_memory=False,
            enable_comm_overlap=False,
            mla_ops_type=int(model_config.mla_ops_type),
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

        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = False
        attn_inputs.sequence_lengths = in_kv_lens
        attn_inputs.input_lengths = q_lens
        attn_inputs.kv_cache_block_id_device = page_table
        attn_inputs.kv_cache_kernel_block_id_device = page_table
        attn_inputs.kv_cache_kernel_block_id_host = page_table.cpu()
        attn_inputs.dtype = get_typemeta(q)
        attn_inputs.total_tokens = q.shape[0]
        attn_inputs.decode_cu_seqlens_d = generate_cumsum_lens(q_lens)
        attn_inputs.decode_cu_seqlens_host = generate_cumsum_lens(q_lens).cpu()
        attn_inputs.cu_seqlens = generate_cumsum_lens(q_lens).cpu()
        attn_inputs.cu_kv_seqlens = generate_cumsum_lens(seq_lens).cpu()

        attn_configs = AttentionConfigs()
        attn_configs.head_num = num_qo_heads
        attn_configs.kv_head_num = num_kv_heads
        attn_configs.size_per_head = head_dim
        attn_configs.tokens_per_block = page_size
        attn_configs.kernel_tokens_per_block = page_size
        attn_configs.kv_cache_dtype = (
            KvCacheDataType.FP8 if kv_dtype == "fp8" else KvCacheDataType.BASE
        )
        attn_configs.dtype = q.dtype

        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = kv_cache_tensor

        original_init = FMHAImplBase.__init__

        def patched_init(
            self, fmha_impl, rope_kvcache_impl, attn_inputs, init_params=True
        ):
            result = original_init(
                self, fmha_impl, rope_kvcache_impl, attn_inputs, init_params=False
            )
            self.rope_kvcache_impl = rope_kvcache_impl
            return result

        FMHAImplBase.__init__ = patched_init
        try:
            attn_configs.need_rope_kv_cache = False
            xqa_impl = XQADecodeImpl(attn_configs, attn_inputs)
            # Prepare fmha_params with scale parameters
            if kv_dtype == "fp8":
                xqa_impl.fmha_params = xqa_impl.fmha_impl.prepare(
                    attn_inputs,
                    q_scale=q_scale * k_scale * sm_scale,
                    kv_scale=v_scale / o_scale,
                    o_scale=o_scale,
                )
            else:
                xqa_impl.fmha_params = xqa_impl.fmha_impl.prepare(attn_inputs)
            xqa_impl.attn_inputs = attn_inputs

            class DummyRopeParams:
                pass

            xqa_impl.rope_params = DummyRopeParams()
        finally:
            FMHAImplBase.__init__ = original_init

        output_4d = xqa_impl.forward(q, kv_cache).squeeze(1)
        output = output_4d.reshape(-1, output_4d.shape[2], output_4d.shape[3])
        output_ref = create_reference_output(
            q,
            kv_cache_tensor,
            ref_kv_cache_tensor,
            page_table,
            seq_lens.to(GPU_DEVICE),
            page_size,
            num_kv_heads,
            head_dim,
            q_len_per_req,
        )
        assert (
            output.shape == output_ref.shape
        ), f"Output shape mismatch: XQA={output.shape}, ref={output_ref.shape}"

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
            f"output shape = {output.shape}, ref shape = {output_ref.shape}",
        )

    @unittest.skipIf(not VERSION_REQUIREMENTS_MET, SKIP_MESSAGE)
    def test_xqa_decode_cuda_graph_replay(self):
        """Test prepare_cuda_graph: same batch size, updated sequence_lengths and page_table"""
        torch.manual_seed(42)

        batch_size = 4
        page_size = 64
        num_kv_heads = 1
        head_grp_size = 8
        num_qo_heads = num_kv_heads * head_grp_size
        head_dim = 128
        q_len_per_req = 1
        capture_kv_len = 10
        kv_dtype = "bf16"

        q_lens_cap, in_kv_lens_cap, seq_lens_cap = generate_seq_lens_decode(
            batch_size, q_len_per_req, capture_kv_len
        )
        kv_cache_tensor, k_scale, v_scale, _ = create_kv_cache(
            batch_size, seq_lens_cap, page_size, num_kv_heads, head_dim, kv_dtype
        )
        page_table_cap, _, _ = create_page_table(batch_size, seq_lens_cap, page_size)

        cap_inputs = PyAttentionInputs()
        cap_inputs.is_prefill = False
        cap_inputs.sequence_lengths = in_kv_lens_cap
        cap_inputs.input_lengths = q_lens_cap
        cap_inputs.kv_cache_block_id_device = page_table_cap
        cap_inputs.kv_cache_kernel_block_id_device = page_table_cap
        cap_inputs.kv_cache_kernel_block_id_host = page_table_cap.cpu()
        cap_inputs.dtype = get_typemeta(
            torch.empty(1, dtype=DTYPE_MAP["bf16"], device=GPU_DEVICE)
        )
        cap_inputs.total_tokens = batch_size * q_len_per_req
        cap_inputs.decode_cu_seqlens_d = generate_cumsum_lens(q_lens_cap)
        cap_inputs.decode_cu_seqlens_host = generate_cumsum_lens(q_lens_cap).cpu()
        cap_inputs.cu_seqlens = generate_cumsum_lens(q_lens_cap).cpu()
        cap_inputs.cu_kv_seqlens = generate_cumsum_lens(seq_lens_cap).cpu()

        attn_configs = AttentionConfigs()
        attn_configs.head_num = num_qo_heads
        attn_configs.kv_head_num = num_kv_heads
        attn_configs.size_per_head = head_dim
        attn_configs.tokens_per_block = page_size
        attn_configs.kernel_tokens_per_block = page_size
        attn_configs.kv_cache_dtype = KvCacheDataType.BASE
        attn_configs.dtype = DTYPE_MAP["bf16"]

        original_init = FMHAImplBase.__init__

        def patched_init(
            self, fmha_impl, rope_kvcache_impl, attn_inputs, init_params=True
        ):
            original_init(
                self, fmha_impl, rope_kvcache_impl, attn_inputs, init_params=False
            )
            self.rope_kvcache_impl = rope_kvcache_impl

        FMHAImplBase.__init__ = patched_init
        try:
            attn_configs.need_rope_kv_cache = False
            xqa_impl = XQADecodeImpl(attn_configs, cap_inputs)
            xqa_impl.fmha_params = xqa_impl.fmha_impl.prepare(cap_inputs)
            xqa_impl.attn_inputs = cap_inputs
            xqa_impl.rope_params = xqa_impl.rope_kvcache_impl.prepare(cap_inputs)
        finally:
            FMHAImplBase.__init__ = original_init

        replay_kv_len = capture_kv_len + 5
        q_lens_rep, in_kv_lens_rep, seq_lens_rep = generate_seq_lens_decode(
            batch_size, q_len_per_req, replay_kv_len
        )
        kv_cache_tensor_rep, _, _, _ = create_kv_cache(
            batch_size, seq_lens_rep, page_size, num_kv_heads, head_dim, kv_dtype
        )
        page_table_rep, _, _ = create_page_table(batch_size, seq_lens_rep, page_size)

        rep_inputs = PyAttentionInputs()
        rep_inputs.is_prefill = False
        rep_inputs.sequence_lengths = in_kv_lens_rep
        rep_inputs.input_lengths = q_lens_rep
        rep_inputs.kv_cache_block_id_device = page_table_rep
        rep_inputs.kv_cache_kernel_block_id_device = page_table_rep
        rep_inputs.kv_cache_kernel_block_id_host = page_table_rep.cpu()
        rep_inputs.dtype = get_typemeta(
            torch.empty(1, dtype=DTYPE_MAP["bf16"], device=GPU_DEVICE)
        )
        rep_inputs.total_tokens = batch_size * q_len_per_req
        rep_inputs.decode_cu_seqlens_d = generate_cumsum_lens(q_lens_rep)
        rep_inputs.decode_cu_seqlens_host = generate_cumsum_lens(q_lens_rep).cpu()
        rep_inputs.cu_seqlens = generate_cumsum_lens(q_lens_rep).cpu()
        rep_inputs.cu_kv_seqlens = generate_cumsum_lens(seq_lens_rep).cpu()

        xqa_impl.prepare_cuda_graph(rep_inputs)

        self.assertEqual(xqa_impl.fmha_params.batch_size, batch_size)
        self.assertTrue(
            torch.equal(xqa_impl.fmha_params.seq_lens, in_kv_lens_rep),
            "sequence_lengths not updated after prepare_cuda_graph",
        )
        self.assertTrue(
            torch.equal(xqa_impl.fmha_params.page_table, page_table_rep),
            "page_table not updated after prepare_cuda_graph",
        )

        q_rep, _, _ = create_query_tensor(q_lens_rep, num_qo_heads, head_dim, "bf16")
        kv_cache_rep = LayerKVCache()
        kv_cache_rep.kv_cache_base = kv_cache_tensor_rep
        xqa_impl.attn_inputs = rep_inputs

        output = xqa_impl.forward(q_rep, kv_cache_rep).squeeze(1)
        output = output.reshape(-1, output.shape[2], output.shape[3])

        output_ref = create_reference_output(
            q_rep,
            kv_cache_tensor_rep,
            None,
            page_table_rep,
            seq_lens_rep.to(GPU_DEVICE),
            page_size,
            num_kv_heads,
            head_dim,
            q_len_per_req,
        )

        self.assertEqual(output.shape, output_ref.shape)
        max_diff = (output.float() - output_ref.float()).abs().max().item()
        self.assertTrue(
            torch.allclose(output.float(), output_ref.float(), rtol=1e-2, atol=1e-2),
            f"CUDA graph replay output mismatch: max diff = {max_diff}",
        )

    @unittest.skipIf(not VERSION_REQUIREMENTS_MET, SKIP_MESSAGE)
    def test_xqa_decode_cuda_graph_padding_replay(self):
        """Test prepare_cuda_graph: actual batch < captured graph batch (padding replay).

        Simulates what CudaGraphRunner does: capture at a larger batch size,
        replay with fewer active requests. Padding rows get zeroed block IDs
        and seq_lens=0, and output is sliced to the actual batch.
        """
        torch.manual_seed(42)

        capture_batch_size = 8
        page_size = 64
        num_kv_heads = 1
        head_grp_size = 8
        num_qo_heads = num_kv_heads * head_grp_size
        head_dim = 128
        q_len_per_req = 1
        capture_kv_len = 10
        kv_dtype = "bf16"

        # --- Capture phase: initialize with capture_batch_size ---
        q_lens_cap, in_kv_lens_cap, seq_lens_cap = generate_seq_lens_decode(
            capture_batch_size, q_len_per_req, capture_kv_len
        )
        _, _, _, _ = create_kv_cache(
            capture_batch_size,
            seq_lens_cap,
            page_size,
            num_kv_heads,
            head_dim,
            kv_dtype,
        )
        page_table_cap, _, _ = create_page_table(
            capture_batch_size, seq_lens_cap, page_size
        )

        cap_inputs = PyAttentionInputs()
        cap_inputs.is_prefill = False
        cap_inputs.sequence_lengths = in_kv_lens_cap
        cap_inputs.input_lengths = q_lens_cap
        cap_inputs.kv_cache_block_id_device = page_table_cap
        cap_inputs.kv_cache_kernel_block_id_device = page_table_cap
        cap_inputs.kv_cache_kernel_block_id_host = page_table_cap.cpu()
        cap_inputs.dtype = get_typemeta(
            torch.empty(1, dtype=DTYPE_MAP["bf16"], device=GPU_DEVICE)
        )
        cap_inputs.total_tokens = capture_batch_size * q_len_per_req
        cap_inputs.decode_cu_seqlens_d = generate_cumsum_lens(q_lens_cap)
        cap_inputs.decode_cu_seqlens_host = generate_cumsum_lens(q_lens_cap).cpu()
        cap_inputs.cu_seqlens = generate_cumsum_lens(q_lens_cap).cpu()
        cap_inputs.cu_kv_seqlens = generate_cumsum_lens(seq_lens_cap).cpu()

        attn_configs = AttentionConfigs()
        attn_configs.head_num = num_qo_heads
        attn_configs.kv_head_num = num_kv_heads
        attn_configs.size_per_head = head_dim
        attn_configs.tokens_per_block = page_size
        attn_configs.kernel_tokens_per_block = page_size
        attn_configs.kv_cache_dtype = KvCacheDataType.BASE
        attn_configs.dtype = DTYPE_MAP["bf16"]

        original_init = FMHAImplBase.__init__

        def patched_init(
            self, fmha_impl, rope_kvcache_impl, attn_inputs, init_params=True
        ):
            original_init(
                self, fmha_impl, rope_kvcache_impl, attn_inputs, init_params=False
            )
            self.rope_kvcache_impl = rope_kvcache_impl

        FMHAImplBase.__init__ = patched_init
        try:
            attn_configs.need_rope_kv_cache = False
            xqa_impl = XQADecodeImpl(attn_configs, cap_inputs)
            xqa_impl.fmha_params = xqa_impl.fmha_impl.prepare(cap_inputs)
            xqa_impl.attn_inputs = cap_inputs
            xqa_impl.rope_params = xqa_impl.rope_kvcache_impl.prepare(cap_inputs)
        finally:
            FMHAImplBase.__init__ = original_init

        # --- Replay phase: actual_batch < capture_batch ---
        for actual_batch_size in [1, 2, 5]:
            with self.subTest(
                capture_bs=capture_batch_size, actual_bs=actual_batch_size
            ):
                replay_kv_len = capture_kv_len + 5

                # Create real data for actual_batch_size
                q_lens_real, in_kv_lens_real, seq_lens_real = generate_seq_lens_decode(
                    actual_batch_size, q_len_per_req, replay_kv_len
                )
                kv_cache_tensor_rep, _, _, _ = create_kv_cache(
                    actual_batch_size,
                    seq_lens_real,
                    page_size,
                    num_kv_heads,
                    head_dim,
                    kv_dtype,
                )
                page_table_real, _, _ = create_page_table(
                    actual_batch_size, seq_lens_real, page_size
                )

                # Pad to capture_batch_size (simulating CudaGraphRunner.prepareInputs)
                pad_size = capture_batch_size - actual_batch_size
                padded_seq_lens = torch.cat(
                    [in_kv_lens_real, torch.zeros(pad_size, dtype=torch.int32)]
                )
                padded_q_lens = torch.cat(
                    [
                        q_lens_real,
                        torch.full((pad_size,), q_len_per_req, dtype=torch.int32),
                    ]
                )
                padded_page_table = torch.zeros(
                    capture_batch_size,
                    page_table_real.shape[1],
                    dtype=torch.int32,
                    device=GPU_DEVICE,
                )
                padded_page_table[:actual_batch_size] = page_table_real

                rep_inputs = PyAttentionInputs()
                rep_inputs.is_prefill = False
                rep_inputs.sequence_lengths = padded_seq_lens
                rep_inputs.input_lengths = padded_q_lens
                rep_inputs.kv_cache_block_id_device = padded_page_table
                rep_inputs.kv_cache_kernel_block_id_device = padded_page_table
                rep_inputs.kv_cache_kernel_block_id_host = padded_page_table.cpu()
                rep_inputs.dtype = get_typemeta(
                    torch.empty(1, dtype=DTYPE_MAP["bf16"], device=GPU_DEVICE)
                )
                rep_inputs.total_tokens = capture_batch_size * q_len_per_req
                rep_inputs.decode_cu_seqlens_d = generate_cumsum_lens(padded_q_lens)
                rep_inputs.decode_cu_seqlens_host = generate_cumsum_lens(
                    padded_q_lens
                ).cpu()
                rep_inputs.cu_seqlens = generate_cumsum_lens(padded_q_lens).cpu()
                padded_full_seq_lens = torch.cat(
                    [
                        seq_lens_real,
                        torch.full((pad_size,), q_len_per_req, dtype=torch.int32),
                    ]
                )
                rep_inputs.cu_kv_seqlens = generate_cumsum_lens(
                    padded_full_seq_lens
                ).cpu()

                xqa_impl.prepare_cuda_graph(rep_inputs)

                # batch_size stays at capture_batch_size (padded tensor size)
                self.assertEqual(xqa_impl.fmha_params.batch_size, capture_batch_size)
                # First actual_batch_size entries have correct seq_lens
                self.assertTrue(
                    torch.equal(
                        xqa_impl.fmha_params.seq_lens[:actual_batch_size],
                        in_kv_lens_real,
                    ),
                    "Active seq_lens not updated correctly",
                )
                # Padding entries have seq_lens=0
                self.assertTrue(
                    (xqa_impl.fmha_params.seq_lens[actual_batch_size:] == 0).all(),
                    "Padding seq_lens should be zero",
                )
                # Padding page_table rows are zeroed
                self.assertTrue(
                    (xqa_impl.fmha_params.page_table[actual_batch_size:] == 0).all(),
                    "Padding page_table rows should be zeroed",
                )

                # Run forward with padded query
                q_padded = torch.randn(
                    capture_batch_size * q_len_per_req,
                    num_qo_heads,
                    head_dim,
                    dtype=DTYPE_MAP["bf16"],
                    device=GPU_DEVICE,
                )
                q_real = q_padded[: actual_batch_size * q_len_per_req]
                kv_cache_rep = LayerKVCache()
                kv_cache_rep.kv_cache_base = kv_cache_tensor_rep
                xqa_impl.attn_inputs = rep_inputs

                output = xqa_impl.forward(q_padded, kv_cache_rep).squeeze(1)
                output = output.reshape(-1, output.shape[2], output.shape[3])
                # Slice to actual batch
                output_actual = output[:actual_batch_size]

                output_ref = create_reference_output(
                    q_real,
                    kv_cache_tensor_rep,
                    None,
                    page_table_real,
                    seq_lens_real.to(GPU_DEVICE),
                    page_size,
                    num_kv_heads,
                    head_dim,
                    q_len_per_req,
                )

                self.assertEqual(output_actual.shape, output_ref.shape)
                max_diff = (
                    (output_actual.float() - output_ref.float()).abs().max().item()
                )
                self.assertTrue(
                    torch.allclose(
                        output_actual.float(),
                        output_ref.float(),
                        rtol=1e-2,
                        atol=1e-2,
                    ),
                    f"Padding replay output mismatch (capture_bs={capture_batch_size}, "
                    f"actual_bs={actual_batch_size}): max diff = {max_diff}",
                )

    @unittest.skipIf(not VERSION_REQUIREMENTS_MET, SKIP_MESSAGE)
    def test_xqa_decode_real_cuda_graph_capture_replay(self):
        """Test real CUDA Graph capture/replay: verify that graph-replayed output
        matches non-graph output after updating sequence_lengths and page_table
        via prepare_cuda_graph().
        """
        torch.manual_seed(42)

        batch_size = 4
        page_size = 64
        num_kv_heads = 1
        head_grp_size = 8
        num_qo_heads = num_kv_heads * head_grp_size
        head_dim = 128
        q_len_per_req = 1
        capture_kv_len = 10
        kv_dtype = "bf16"

        # --- Capture phase setup ---
        q_lens_cap, in_kv_lens_cap, seq_lens_cap = generate_seq_lens_decode(
            batch_size, q_len_per_req, capture_kv_len
        )
        kv_cache_tensor_cap, _, _, _ = create_kv_cache(
            batch_size, seq_lens_cap, page_size, num_kv_heads, head_dim, kv_dtype
        )
        page_table_cap, _, _ = create_page_table(batch_size, seq_lens_cap, page_size)

        cap_inputs = PyAttentionInputs()
        cap_inputs.is_prefill = False
        cap_inputs.sequence_lengths = in_kv_lens_cap
        cap_inputs.input_lengths = q_lens_cap
        cap_inputs.kv_cache_block_id_device = page_table_cap
        cap_inputs.kv_cache_kernel_block_id_device = page_table_cap
        cap_inputs.kv_cache_kernel_block_id_host = page_table_cap.cpu()
        cap_inputs.dtype = get_typemeta(
            torch.empty(1, dtype=DTYPE_MAP["bf16"], device=GPU_DEVICE)
        )
        cap_inputs.total_tokens = batch_size * q_len_per_req
        cap_inputs.decode_cu_seqlens_d = generate_cumsum_lens(q_lens_cap)
        cap_inputs.decode_cu_seqlens_host = generate_cumsum_lens(q_lens_cap).cpu()
        cap_inputs.cu_seqlens = generate_cumsum_lens(q_lens_cap).cpu()
        cap_inputs.cu_kv_seqlens = generate_cumsum_lens(seq_lens_cap).cpu()

        attn_configs = AttentionConfigs()
        attn_configs.head_num = num_qo_heads
        attn_configs.kv_head_num = num_kv_heads
        attn_configs.size_per_head = head_dim
        attn_configs.tokens_per_block = page_size
        attn_configs.kernel_tokens_per_block = page_size
        attn_configs.kv_cache_dtype = KvCacheDataType.BASE
        attn_configs.dtype = DTYPE_MAP["bf16"]

        original_init = FMHAImplBase.__init__

        def patched_init(
            self, fmha_impl, rope_kvcache_impl, attn_inputs, init_params=True
        ):
            original_init(
                self, fmha_impl, rope_kvcache_impl, attn_inputs, init_params=False
            )
            self.rope_kvcache_impl = rope_kvcache_impl

        FMHAImplBase.__init__ = patched_init
        try:
            attn_configs.need_rope_kv_cache = False
            xqa_impl = XQADecodeImpl(attn_configs, cap_inputs)
            xqa_impl.fmha_params = xqa_impl.fmha_impl.prepare(cap_inputs)
            xqa_impl.attn_inputs = cap_inputs
            xqa_impl.rope_params = xqa_impl.rope_kvcache_impl.prepare(cap_inputs)
        finally:
            FMHAImplBase.__init__ = original_init

        # Pre-allocate fixed input/output buffers for graph capture
        q_cap, _, _ = create_query_tensor(q_lens_cap, num_qo_heads, head_dim, "bf16")
        kv_cache_cap = LayerKVCache()
        kv_cache_cap.kv_cache_base = kv_cache_tensor_cap

        # Warmup (required before CUDA graph capture)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _ = xqa_impl.forward(q_cap, kv_cache_cap)
        torch.cuda.current_stream().wait_stream(stream)

        # Capture CUDA graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_output = xqa_impl.forward(q_cap, kv_cache_cap)
        torch.cuda.current_stream().wait_stream(stream)

        # --- Replay phase: new KV lengths, new page table ---
        replay_kv_len = capture_kv_len + 5
        q_lens_rep, in_kv_lens_rep, seq_lens_rep = generate_seq_lens_decode(
            batch_size, q_len_per_req, replay_kv_len
        )
        kv_cache_tensor_rep, _, _, _ = create_kv_cache(
            batch_size, seq_lens_rep, page_size, num_kv_heads, head_dim, kv_dtype
        )
        page_table_rep, _, _ = create_page_table(batch_size, seq_lens_rep, page_size)

        rep_inputs = PyAttentionInputs()
        rep_inputs.is_prefill = False
        rep_inputs.sequence_lengths = in_kv_lens_rep
        rep_inputs.input_lengths = q_lens_rep
        rep_inputs.kv_cache_block_id_device = page_table_rep
        rep_inputs.kv_cache_kernel_block_id_device = page_table_rep
        rep_inputs.kv_cache_kernel_block_id_host = page_table_rep.cpu()
        rep_inputs.dtype = get_typemeta(
            torch.empty(1, dtype=DTYPE_MAP["bf16"], device=GPU_DEVICE)
        )
        rep_inputs.total_tokens = batch_size * q_len_per_req
        rep_inputs.decode_cu_seqlens_d = generate_cumsum_lens(q_lens_rep)
        rep_inputs.decode_cu_seqlens_host = generate_cumsum_lens(q_lens_rep).cpu()
        rep_inputs.cu_seqlens = generate_cumsum_lens(q_lens_rep).cpu()
        rep_inputs.cu_kv_seqlens = generate_cumsum_lens(seq_lens_rep).cpu()

        # Update parameters for replay (in-place update of captured tensors)
        xqa_impl.prepare_cuda_graph(rep_inputs)
        xqa_impl.attn_inputs = rep_inputs

        # Update captured input buffers in-place (CUDA graph binds GPU addresses at capture time)
        q_rep, _, _ = create_query_tensor(q_lens_rep, num_qo_heads, head_dim, "bf16")
        q_cap.copy_(q_rep)
        kv_cache_tensor_cap.copy_(kv_cache_tensor_rep)

        # Replay the captured graph
        graph.replay()
        torch.cuda.synchronize()

        replay_output = graph_output.clone().squeeze(1)
        replay_output = replay_output.reshape(
            -1, replay_output.shape[2], replay_output.shape[3]
        )

        # Reference: non-graph forward with replay inputs
        output_ref = create_reference_output(
            q_rep,
            kv_cache_tensor_rep,
            None,
            page_table_rep,
            seq_lens_rep.to(GPU_DEVICE),
            page_size,
            num_kv_heads,
            head_dim,
            q_len_per_req,
        )

        self.assertEqual(replay_output.shape, output_ref.shape)
        max_diff = (replay_output.float() - output_ref.float()).abs().max().item()
        self.assertTrue(
            torch.allclose(
                replay_output.float(), output_ref.float(), rtol=1e-2, atol=1e-2
            ),
            f"Real CUDA graph replay output mismatch: max diff = {max_diff}",
        )

    @unittest.skipIf(not VERSION_REQUIREMENTS_MET, SKIP_MESSAGE)
    def test_xqa_decode_real_cuda_graph_cross_page_replay(self):
        """Test CUDA Graph replay when page_table content genuinely changes between
        capture and replay (replay crosses a page boundary).

        Simulates production behavior: pre-allocate fixed-address tensors and
        update them in-place with copy_(), so the CUDA graph reads new data from
        the same data_ptr().
        """
        torch.manual_seed(42)

        batch_size = 4
        page_size = 64
        num_kv_heads = 1
        head_grp_size = 8
        num_qo_heads = num_kv_heads * head_grp_size
        head_dim = 128
        q_len_per_req = 1
        capture_kv_len = 10  # 1 page per seq
        replay_kv_len = 70  # 2 pages per seq — page_table content differs
        kv_dtype = "bf16"

        # Pre-allocate for the max case (replay) so buffers are large enough for both phases
        max_kv_len = replay_kv_len
        _, _, max_seq_lens = generate_seq_lens_decode(
            batch_size, q_len_per_req, max_kv_len
        )
        max_pages_per_seq = (
            torch.max(max_seq_lens).item() + page_size - 1
        ) // page_size
        total_max_pages = max_pages_per_seq * batch_size

        # Fixed-address kv_cache buffer (large enough for both phases)
        kv_cache_fixed = torch.randn(
            total_max_pages,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=DTYPE_MAP["bf16"],
            device=GPU_DEVICE,
        )
        # Fixed-address page_table buffer
        page_table_fixed = torch.zeros(
            batch_size,
            max_pages_per_seq,
            dtype=torch.int32,
            device=GPU_DEVICE,
        )

        # --- Capture phase ---
        q_lens_cap, in_kv_lens_cap, seq_lens_cap = generate_seq_lens_decode(
            batch_size, q_len_per_req, capture_kv_len
        )
        kv_cache_cap, _, _, _ = create_kv_cache(
            batch_size, seq_lens_cap, page_size, num_kv_heads, head_dim, kv_dtype
        )
        page_table_cap, _, _ = create_page_table(batch_size, seq_lens_cap, page_size)

        # Fill fixed buffers with capture-phase data
        cap_pages = kv_cache_cap.shape[0]
        kv_cache_fixed[:cap_pages].copy_(kv_cache_cap)
        page_table_fixed[:, : page_table_cap.shape[1]].copy_(page_table_cap)

        cap_inputs = PyAttentionInputs()
        cap_inputs.is_prefill = False
        cap_inputs.sequence_lengths = in_kv_lens_cap
        cap_inputs.input_lengths = q_lens_cap
        cap_inputs.kv_cache_block_id_device = page_table_fixed
        cap_inputs.kv_cache_kernel_block_id_device = page_table_fixed
        cap_inputs.kv_cache_kernel_block_id_host = page_table_fixed.cpu()
        cap_inputs.dtype = get_typemeta(
            torch.empty(1, dtype=DTYPE_MAP["bf16"], device=GPU_DEVICE)
        )
        cap_inputs.total_tokens = batch_size * q_len_per_req
        cap_inputs.decode_cu_seqlens_d = generate_cumsum_lens(q_lens_cap)
        cap_inputs.decode_cu_seqlens_host = generate_cumsum_lens(q_lens_cap).cpu()
        cap_inputs.cu_seqlens = generate_cumsum_lens(q_lens_cap).cpu()
        cap_inputs.cu_kv_seqlens = generate_cumsum_lens(seq_lens_cap).cpu()

        attn_configs = AttentionConfigs()
        attn_configs.head_num = num_qo_heads
        attn_configs.kv_head_num = num_kv_heads
        attn_configs.size_per_head = head_dim
        attn_configs.tokens_per_block = page_size
        attn_configs.kernel_tokens_per_block = page_size
        attn_configs.kv_cache_dtype = KvCacheDataType.BASE
        attn_configs.dtype = DTYPE_MAP["bf16"]

        original_init = FMHAImplBase.__init__

        def patched_init(
            self, fmha_impl, rope_kvcache_impl, attn_inputs, init_params=True
        ):
            original_init(
                self, fmha_impl, rope_kvcache_impl, attn_inputs, init_params=False
            )
            self.rope_kvcache_impl = rope_kvcache_impl

        FMHAImplBase.__init__ = patched_init
        try:
            attn_configs.need_rope_kv_cache = False
            xqa_impl = XQADecodeImpl(attn_configs, cap_inputs)
            xqa_impl.fmha_params = xqa_impl.fmha_impl.prepare(cap_inputs)
            xqa_impl.attn_inputs = cap_inputs
            xqa_impl.rope_params = xqa_impl.rope_kvcache_impl.prepare(cap_inputs)
        finally:
            FMHAImplBase.__init__ = original_init

        q_cap, _, _ = create_query_tensor(q_lens_cap, num_qo_heads, head_dim, "bf16")
        kv_cache_layer = LayerKVCache()
        kv_cache_layer.kv_cache_base = kv_cache_fixed

        # Warmup + capture
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _ = xqa_impl.forward(q_cap, kv_cache_layer)
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_output = xqa_impl.forward(q_cap, kv_cache_layer)
        torch.cuda.current_stream().wait_stream(stream)

        # --- Replay phase: cross-page, in-place update of fixed buffers ---
        q_lens_rep, in_kv_lens_rep, seq_lens_rep = generate_seq_lens_decode(
            batch_size, q_len_per_req, replay_kv_len
        )
        kv_cache_rep, _, _, _ = create_kv_cache(
            batch_size, seq_lens_rep, page_size, num_kv_heads, head_dim, kv_dtype
        )
        page_table_rep, _, _ = create_page_table(batch_size, seq_lens_rep, page_size)

        # In-place update fixed buffers (simulates C++ CudaGraphRunner D2D copy)
        rep_pages = kv_cache_rep.shape[0]
        kv_cache_fixed[:rep_pages].copy_(kv_cache_rep)
        page_table_fixed.zero_()
        page_table_fixed[:, : page_table_rep.shape[1]].copy_(page_table_rep)

        rep_inputs = PyAttentionInputs()
        rep_inputs.is_prefill = False
        rep_inputs.sequence_lengths = in_kv_lens_rep
        rep_inputs.input_lengths = q_lens_rep
        rep_inputs.kv_cache_block_id_device = page_table_fixed
        rep_inputs.kv_cache_kernel_block_id_device = page_table_fixed
        rep_inputs.kv_cache_kernel_block_id_host = page_table_fixed.cpu()
        rep_inputs.dtype = get_typemeta(
            torch.empty(1, dtype=DTYPE_MAP["bf16"], device=GPU_DEVICE)
        )
        rep_inputs.total_tokens = batch_size * q_len_per_req
        rep_inputs.decode_cu_seqlens_d = generate_cumsum_lens(q_lens_rep)
        rep_inputs.decode_cu_seqlens_host = generate_cumsum_lens(q_lens_rep).cpu()
        rep_inputs.cu_seqlens = generate_cumsum_lens(q_lens_rep).cpu()
        rep_inputs.cu_kv_seqlens = generate_cumsum_lens(seq_lens_rep).cpu()

        xqa_impl.prepare_cuda_graph(rep_inputs)
        xqa_impl.attn_inputs = rep_inputs

        q_rep, _, _ = create_query_tensor(q_lens_rep, num_qo_heads, head_dim, "bf16")
        q_cap.copy_(q_rep)

        graph.replay()
        torch.cuda.synchronize()

        replay_output = graph_output.clone().squeeze(1)
        replay_output = replay_output.reshape(
            -1, replay_output.shape[2], replay_output.shape[3]
        )

        # Reference: non-graph forward with replay data
        output_ref = create_reference_output(
            q_rep,
            kv_cache_rep,
            None,
            page_table_rep,
            seq_lens_rep.to(GPU_DEVICE),
            page_size,
            num_kv_heads,
            head_dim,
            q_len_per_req,
        )

        self.assertEqual(replay_output.shape, output_ref.shape)
        max_diff = (replay_output.float() - output_ref.float()).abs().max().item()
        self.assertTrue(
            torch.allclose(
                replay_output.float(), output_ref.float(), rtol=1e-2, atol=1e-2
            ),
            f"Cross-page CUDA graph replay output mismatch: max diff = {max_diff}",
        )

    @unittest.skipIf(not VERSION_REQUIREMENTS_MET, SKIP_MESSAGE)
    def test_xqa_decode_real_cuda_graph_padding_replay(self):
        """Test real CUDA Graph capture/replay with padding: capture at larger batch size,
        replay with smaller actual batch (padding rows with zeroed block IDs and seq_lens=0).
        Verifies that graph-replayed output after slicing matches the non-graph reference.

        This mirrors what CudaGraphRunner does when a smaller batch hits a larger graph key.
        """
        torch.manual_seed(42)

        capture_batch_size = 8
        page_size = 64
        num_kv_heads = 1
        head_grp_size = 8
        num_qo_heads = num_kv_heads * head_grp_size
        head_dim = 128
        q_len_per_req = 1
        capture_kv_len = 10
        kv_dtype = "bf16"

        # --- Capture phase: allocate buffers for capture_batch_size ---
        q_lens_cap, in_kv_lens_cap, seq_lens_cap = generate_seq_lens_decode(
            capture_batch_size, q_len_per_req, capture_kv_len
        )
        kv_cache_tensor_cap, _, _, _ = create_kv_cache(
            capture_batch_size,
            seq_lens_cap,
            page_size,
            num_kv_heads,
            head_dim,
            kv_dtype,
        )
        page_table_cap, _, _ = create_page_table(
            capture_batch_size, seq_lens_cap, page_size
        )

        cap_inputs = PyAttentionInputs()
        cap_inputs.is_prefill = False
        cap_inputs.sequence_lengths = in_kv_lens_cap
        cap_inputs.input_lengths = q_lens_cap
        cap_inputs.kv_cache_block_id_device = page_table_cap
        cap_inputs.kv_cache_kernel_block_id_device = page_table_cap
        cap_inputs.kv_cache_kernel_block_id_host = page_table_cap.cpu()
        cap_inputs.dtype = get_typemeta(
            torch.empty(1, dtype=DTYPE_MAP["bf16"], device=GPU_DEVICE)
        )
        cap_inputs.total_tokens = capture_batch_size * q_len_per_req
        cap_inputs.decode_cu_seqlens_d = generate_cumsum_lens(q_lens_cap)
        cap_inputs.decode_cu_seqlens_host = generate_cumsum_lens(q_lens_cap).cpu()
        cap_inputs.cu_seqlens = generate_cumsum_lens(q_lens_cap).cpu()
        cap_inputs.cu_kv_seqlens = generate_cumsum_lens(seq_lens_cap).cpu()

        attn_configs = AttentionConfigs()
        attn_configs.head_num = num_qo_heads
        attn_configs.kv_head_num = num_kv_heads
        attn_configs.size_per_head = head_dim
        attn_configs.tokens_per_block = page_size
        attn_configs.kernel_tokens_per_block = page_size
        attn_configs.kv_cache_dtype = KvCacheDataType.BASE
        attn_configs.dtype = DTYPE_MAP["bf16"]

        original_init = FMHAImplBase.__init__

        def patched_init(
            self, fmha_impl, rope_kvcache_impl, attn_inputs, init_params=True
        ):
            original_init(
                self, fmha_impl, rope_kvcache_impl, attn_inputs, init_params=False
            )
            self.rope_kvcache_impl = rope_kvcache_impl

        FMHAImplBase.__init__ = patched_init
        try:
            attn_configs.need_rope_kv_cache = False
            xqa_impl = XQADecodeImpl(attn_configs, cap_inputs)
            xqa_impl.fmha_params = xqa_impl.fmha_impl.prepare(cap_inputs)
            xqa_impl.attn_inputs = cap_inputs
            xqa_impl.rope_params = xqa_impl.rope_kvcache_impl.prepare(cap_inputs)
        finally:
            FMHAImplBase.__init__ = original_init

        # Pre-allocate fixed input/output buffers for graph capture
        q_cap, _, _ = create_query_tensor(q_lens_cap, num_qo_heads, head_dim, "bf16")
        kv_cache_cap = LayerKVCache()
        kv_cache_cap.kv_cache_base = kv_cache_tensor_cap

        # Warmup
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _ = xqa_impl.forward(q_cap, kv_cache_cap)
        torch.cuda.current_stream().wait_stream(stream)

        # Capture CUDA graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_output = xqa_impl.forward(q_cap, kv_cache_cap)
        torch.cuda.current_stream().wait_stream(stream)

        # --- Replay with padding: actual batch < capture batch ---
        for actual_batch_size in [1, 2, 5]:
            with self.subTest(
                capture_bs=capture_batch_size, actual_bs=actual_batch_size
            ):
                replay_kv_len = capture_kv_len + 5

                pad_size = capture_batch_size - actual_batch_size

                q_lens_real, in_kv_lens_real, seq_lens_real = generate_seq_lens_decode(
                    actual_batch_size, q_len_per_req, replay_kv_len
                )
                # Create kv_cache with capture_batch_size pages for graph compatibility
                # (CUDA graph binds tensor shapes). Only first actual_batch_size have
                # valid data; padding entries are masked by page_table zeros.
                padded_seq_lens = torch.cat(
                    [seq_lens_real, torch.zeros(pad_size, dtype=torch.int32)]
                )
                kv_cache_tensor_rep, _, _, _ = create_kv_cache(
                    capture_batch_size,
                    padded_seq_lens,
                    page_size,
                    num_kv_heads,
                    head_dim,
                    kv_dtype,
                )
                page_table_real, _, _ = create_page_table(
                    actual_batch_size, seq_lens_real, page_size
                )

                # Pad to capture_batch_size
                pad_size = capture_batch_size - actual_batch_size
                padded_seq_lens = torch.cat(
                    [in_kv_lens_real, torch.zeros(pad_size, dtype=torch.int32)]
                )
                padded_q_lens = torch.cat(
                    [
                        q_lens_real,
                        torch.full((pad_size,), q_len_per_req, dtype=torch.int32),
                    ]
                )
                padded_page_table = torch.zeros(
                    capture_batch_size,
                    page_table_real.shape[1],
                    dtype=torch.int32,
                    device=GPU_DEVICE,
                )
                padded_page_table[:actual_batch_size] = page_table_real

                rep_inputs = PyAttentionInputs()
                rep_inputs.is_prefill = False
                rep_inputs.sequence_lengths = padded_seq_lens
                rep_inputs.input_lengths = padded_q_lens
                rep_inputs.kv_cache_block_id_device = padded_page_table
                rep_inputs.kv_cache_kernel_block_id_device = padded_page_table
                rep_inputs.kv_cache_kernel_block_id_host = padded_page_table.cpu()
                rep_inputs.dtype = get_typemeta(
                    torch.empty(1, dtype=DTYPE_MAP["bf16"], device=GPU_DEVICE)
                )
                rep_inputs.total_tokens = capture_batch_size * q_len_per_req
                rep_inputs.decode_cu_seqlens_d = generate_cumsum_lens(padded_q_lens)
                rep_inputs.decode_cu_seqlens_host = generate_cumsum_lens(
                    padded_q_lens
                ).cpu()
                rep_inputs.cu_seqlens = generate_cumsum_lens(padded_q_lens).cpu()
                padded_full_seq_lens = torch.cat(
                    [
                        seq_lens_real,
                        torch.full((pad_size,), q_len_per_req, dtype=torch.int32),
                    ]
                )
                rep_inputs.cu_kv_seqlens = generate_cumsum_lens(
                    padded_full_seq_lens
                ).cpu()

                # In-place update captured parameters
                xqa_impl.prepare_cuda_graph(rep_inputs)
                xqa_impl.attn_inputs = rep_inputs

                # In-place update captured input buffers (graph binds GPU addresses)
                q_rep, _, _ = create_query_tensor(
                    padded_q_lens, num_qo_heads, head_dim, "bf16"
                )
                q_cap.copy_(q_rep)
                kv_cache_cap.kv_cache_base.copy_(kv_cache_tensor_rep)

                # Replay the graph
                graph.replay()
                torch.cuda.synchronize()

                replay_output = graph_output.clone().squeeze(1)
                replay_output = replay_output.reshape(
                    -1, replay_output.shape[2], replay_output.shape[3]
                )
                # Slice to actual batch size
                output_actual = replay_output[:actual_batch_size]

                # Reference: non-graph forward with actual batch
                q_real = q_rep[: actual_batch_size * q_len_per_req]
                output_ref = create_reference_output(
                    q_real,
                    kv_cache_tensor_rep,
                    None,
                    page_table_real,
                    seq_lens_real.to(GPU_DEVICE),
                    page_size,
                    num_kv_heads,
                    head_dim,
                    q_len_per_req,
                )

                self.assertEqual(output_actual.shape, output_ref.shape)
                max_diff = (
                    (output_actual.float() - output_ref.float()).abs().max().item()
                )
                self.assertTrue(
                    torch.allclose(
                        output_actual.float(),
                        output_ref.float(),
                        rtol=1e-2,
                        atol=1e-2,
                    ),
                    f"Padding graph replay output mismatch (capture_bs={capture_batch_size}, "
                    f"actual_bs={actual_batch_size}): max diff = {max_diff}",
                )

    @unittest.skipIf(not VERSION_REQUIREMENTS_MET, SKIP_MESSAGE)
    def test_xqa_decode_2d_packed_kv_cache(self):
        """Test XQADecodeImpl with 2D packed KV cache (hybrid cache layout).

        In hybrid attention models, the per-layer KV cache arrives as a raw 2D
        buffer [block_num, kv_block_stride_elems]. XQAWrapper.forward() must
        reshape it to 5D via common.reshape_paged_kv_cache() before use.
        This test constructs a 2D packed cache (with extra stride padding to
        simulate hybrid mode) and verifies output matches the 5D reference.
        """
        torch.manual_seed(42)

        batch_size = 2
        page_size = 64
        num_kv_heads = 1
        head_grp_size = 8
        num_qo_heads = num_kv_heads * head_grp_size
        head_dim = 128
        q_len_per_req = 1
        max_in_kv_len = 10
        kv_dtype = "bf16"

        q_lens, in_kv_lens, seq_lens = generate_seq_lens_decode(
            batch_size, q_len_per_req, max_in_kv_len
        )
        kv_cache_5d, k_scale, v_scale, _ = create_kv_cache(
            batch_size, seq_lens, page_size, num_kv_heads, head_dim, kv_dtype
        )
        page_table, _, _ = create_page_table(batch_size, seq_lens, page_size)

        # Reshape 5D [num_pages, 2, num_kv_heads, page_size, head_dim] → 2D packed
        num_pages = kv_cache_5d.shape[0]
        elems_per_block = 2 * num_kv_heads * page_size * head_dim
        # Add extra padding to simulate hybrid stride (e.g. linear attention uses more)
        hybrid_extra = 1024
        kv_cache_2d = torch.zeros(
            num_pages,
            elems_per_block + hybrid_extra,
            dtype=kv_cache_5d.dtype,
            device=kv_cache_5d.device,
        )
        kv_cache_2d[:, :elems_per_block] = kv_cache_5d.reshape(
            num_pages, elems_per_block
        )

        q, q_scale, ref_q = create_query_tensor(q_lens, num_qo_heads, head_dim, "bf16")

        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = False
        attn_inputs.sequence_lengths = in_kv_lens
        attn_inputs.input_lengths = q_lens
        attn_inputs.kv_cache_block_id_device = page_table
        attn_inputs.kv_cache_kernel_block_id_device = page_table
        attn_inputs.kv_cache_kernel_block_id_host = page_table.cpu()
        attn_inputs.dtype = get_typemeta(q)
        attn_inputs.total_tokens = q.shape[0]
        attn_inputs.decode_cu_seqlens_d = generate_cumsum_lens(q_lens)
        attn_inputs.decode_cu_seqlens_host = generate_cumsum_lens(q_lens).cpu()
        attn_inputs.cu_seqlens = generate_cumsum_lens(q_lens).cpu()
        attn_inputs.cu_kv_seqlens = generate_cumsum_lens(seq_lens).cpu()

        attn_configs = AttentionConfigs()
        attn_configs.head_num = num_qo_heads
        attn_configs.kv_head_num = num_kv_heads
        attn_configs.size_per_head = head_dim
        attn_configs.tokens_per_block = page_size
        attn_configs.kernel_tokens_per_block = page_size
        attn_configs.kv_cache_dtype = KvCacheDataType.BASE
        attn_configs.dtype = q.dtype

        original_init = FMHAImplBase.__init__

        def patched_init(
            self, fmha_impl, rope_kvcache_impl, attn_inputs, init_params=True
        ):
            original_init(
                self, fmha_impl, rope_kvcache_impl, attn_inputs, init_params=False
            )
            self.rope_kvcache_impl = rope_kvcache_impl

        FMHAImplBase.__init__ = patched_init
        try:
            attn_configs.need_rope_kv_cache = False
            xqa_impl = XQADecodeImpl(attn_configs, attn_inputs)
            xqa_impl.fmha_params = xqa_impl.fmha_impl.prepare(attn_inputs)
            xqa_impl.attn_inputs = attn_inputs

            class DummyRopeParams:
                pass

            xqa_impl.rope_params = DummyRopeParams()
        finally:
            FMHAImplBase.__init__ = original_init

        # Forward with 2D packed KV cache
        kv_cache_packed = LayerKVCache()
        kv_cache_packed.kv_cache_base = kv_cache_2d

        output_2d = xqa_impl.forward(q, kv_cache_packed).squeeze(1)
        output_2d = output_2d.reshape(-1, output_2d.shape[2], output_2d.shape[3])

        # Forward with original 5D KV cache for reference
        kv_cache_normal = LayerKVCache()
        kv_cache_normal.kv_cache_base = kv_cache_5d

        xqa_impl2 = XQADecodeImpl.__new__(XQADecodeImpl)
        FMHAImplBase.__init__ = patched_init
        try:
            attn_configs.need_rope_kv_cache = False
            xqa_impl2 = XQADecodeImpl(attn_configs, attn_inputs)
            xqa_impl2.fmha_params = xqa_impl2.fmha_impl.prepare(attn_inputs)
            xqa_impl2.attn_inputs = attn_inputs
            xqa_impl2.rope_params = DummyRopeParams()
        finally:
            FMHAImplBase.__init__ = original_init

        output_5d = xqa_impl2.forward(q, kv_cache_normal).squeeze(1)
        output_5d = output_5d.reshape(-1, output_5d.shape[2], output_5d.shape[3])

        self.assertEqual(output_2d.shape, output_5d.shape)
        max_diff = (output_2d.float() - output_5d.float()).abs().max().item()
        self.assertTrue(
            torch.allclose(output_2d.float(), output_5d.float(), rtol=1e-5, atol=1e-5),
            f"2D packed KV cache output differs from 5D: max diff = {max_diff}",
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
