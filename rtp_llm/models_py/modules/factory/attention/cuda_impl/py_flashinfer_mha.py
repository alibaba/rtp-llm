from pathlib import Path
from typing import Any, Optional

import torch
from flashinfer import decode as flashinfer_decode
from flashinfer.cascade import merge_state_in_place
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    MhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
    KVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
    check_attention_inputs,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.utils.arch import is_sm10x, is_sm90
from rtp_llm.ops import (
    AttentionConfigs,
    KvCacheDataType,
    ParallelismConfig,
    RopeConfig,
    RopeStyle,
    check_rope_cache,
    get_rope_cache_once,
)
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    LayerKVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
    rtp_llm_ops,
)
from rtp_llm.ops.fused_rope_kvcache_op import FusedRopeKVCachePrefillOpQKVOut

# Constants
DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB = 128

# FP8 KV cache uses a unit quantization scale: K/V are cast
# directly to float8_e4m3fn and FA3 FP8 kernels run with scale_q/k/v = 1.0.
FP8_UNIT_SCALE = 1.0
_g_fp8_unit_scale_tensors: dict[torch.device, torch.Tensor] = {}


def _get_fp8_unit_scale_tensor(device: torch.device) -> torch.Tensor:
    scale = _g_fp8_unit_scale_tensors.get(device)
    if scale is None:
        scale = torch.tensor([FP8_UNIT_SCALE], dtype=torch.float32, device=device)
        _g_fp8_unit_scale_tensors[device] = scale
    return scale


def quantize_to_fp8_if_needed(
    tensor: torch.Tensor, target_dtype: torch.dtype
) -> torch.Tensor:
    """Return a matching tensor or quantize it to the supported FP8 dtype."""
    if tensor.dtype == target_dtype:
        return tensor
    if target_dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"unsupported dtype conversion from {tensor.dtype} to {target_dtype}; "
            "only quantization to torch.float8_e4m3fn is supported"
        )

    # per_tensor_quant_fp8 requires contiguous input; packed-QKV split
    # views are strided, so materialize them first.
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    output = torch.empty_like(tensor, dtype=target_dtype)
    rtp_llm_ops.per_tensor_quant_fp8(
        tensor, output, _get_fp8_unit_scale_tensor(tensor.device), True
    )
    return output


# Global workspace buffer pool
_g_py_flashinfer_workspace_pool: dict[torch.device, list[torch.Tensor]] = {}
_g_py_flashinfer_pool_lock = __import__("threading").Lock()
_g_flashinfer_jit_header_lock = __import__("threading").Lock()
_g_dynamic_fp8_jit_modules: dict[str, tuple[Any, list[str]]] = {}


def get_py_flashinfer_workspace_buffer(
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Get a PyFlashInfer workspace buffer from the pool."""
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and resolved_device.index is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    with _g_py_flashinfer_pool_lock:
        device_pool = _g_py_flashinfer_workspace_pool.get(resolved_device)
        if device_pool:
            return device_pool.pop()
    return torch.zeros(
        DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB * 1024 * 1024,
        dtype=torch.uint8,
        device=resolved_device,
    )


def release_py_flashinfer_workspace_buffer(buffer: torch.Tensor) -> None:
    """Release a PyFlashInfer workspace buffer back to the pool."""
    with _g_py_flashinfer_pool_lock:
        _g_py_flashinfer_workspace_pool.setdefault(buffer.device, []).append(buffer)


def _host_i32(t):
    """Lift a possibly CUDA-resident tensor to host for the host fill path.

    The C++ fillParams reads raw host pointers; hybrid-attention models
    (qwen3-next) and the MTP device-state fast path can hand in CUDA-resident
    lengths/block tables even when input_lengths stays on the host.
    """
    return t.cpu() if t is not None and t.numel() > 0 and t.is_cuda else t


def _device_or(device_tensor, host_tensor):
    """Prefer a populated *_device mirror; fall back to the base field."""
    if device_tensor is not None and device_tensor.numel() > 0:
        return device_tensor
    return host_tensor


def _is_dynamic_fp8(attn_configs: AttentionConfigs) -> bool:
    return getattr(attn_configs, "fp8_kv_cache_mode", 0) == 2


def _use_fused_mrope_prefill(
    attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
) -> bool:
    rope_config = attn_configs.rope_config
    mrope_dims = (
        rope_config.mrope_dim1,
        rope_config.mrope_dim2,
        rope_config.mrope_dim3,
    )
    position_ids = getattr(attn_inputs, "combo_position_ids", None)
    input_lengths = getattr(attn_inputs, "input_lengths", None)
    valid_position_ids = (
        isinstance(position_ids, torch.Tensor)
        and position_ids.is_cuda
        and position_ids.dtype == torch.int32
        and position_ids.is_contiguous()
        and isinstance(input_lengths, torch.Tensor)
        and position_ids.numel() == 3 * int(input_lengths.sum().item())
    )
    return (
        getattr(attn_inputs, "is_prefill", False)
        and not getattr(attn_inputs, "is_cuda_graph", False)
        and getattr(attn_configs, "fp8_kv_cache_mode", 0) == 1
        and attn_configs.kv_cache_dtype == KvCacheDataType.FP8
        and attn_configs.need_rope_kv_cache
        and rope_config.style == RopeStyle.Mrope
        and rope_config.mrope_interleaved
        and rope_config.index_factor == 3
        and all(dim > 0 for dim in mrope_dims)
        and 2 * sum(mrope_dims) == rope_config.dim
        and valid_position_ids
    )


def _supports_prefill_rope(
    attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
) -> bool:
    if attn_configs.rope_config.style != RopeStyle.Mrope:
        return True
    return _use_fused_mrope_prefill(attn_configs, attn_inputs)


def _active_page_indices(
    page_indices: torch.Tensor, page_indptr: torch.Tensor
) -> torch.Tensor:
    if page_indptr is None or page_indptr.numel() == 0:
        raise ValueError("paged attention requires a non-empty page indptr")
    page_count = int(page_indptr[-1].item())
    if page_count < 0 or page_count > page_indices.numel():
        raise ValueError(
            f"active page count {page_count} exceeds page index capacity "
            f"{page_indices.numel()}"
        )
    return page_indices.narrow(0, 0, page_count)


def attn_kv_dtype(attn_configs: AttentionConfigs) -> torch.dtype:
    # Dynamic mode dequantizes into the model's base dtype before FlashInfer.
    if _is_dynamic_fp8(attn_configs):
        return attn_configs.dtype
    if attn_configs.kv_cache_dtype == KvCacheDataType.FP8:
        return torch.float8_e4m3fn
    return attn_configs.dtype


def supports_scalar_prefill_rope(
    attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
) -> bool:
    """Scalar RoPE also handles text MRoPE when every axis has scalar positions."""
    if attn_configs.rope_config.style != RopeStyle.Mrope:
        return True
    # Do not silently apply scalar RoPE to image/video axes or graph replays
    # whose position values can change after this host-side validation.
    if attn_inputs.is_cuda_graph:
        return False
    positions = attn_inputs.combo_position_ids
    if positions is None or positions.numel() == 0:
        return False
    lengths = _host_i32(attn_inputs.input_lengths).to(torch.int64)
    prefixes = _host_i32(attn_inputs.prefix_lengths).to(torch.int64)
    if lengths.numel() != prefixes.numel():
        return False
    count = int(lengths.sum())
    if positions.numel() != count * 3:
        return False
    starts = lengths.cumsum(0) - lengths
    scalar_positions = torch.arange(count) + torch.repeat_interleave(
        prefixes - starts, lengths
    )
    axes = positions.cpu().reshape(count, 3)
    return torch.equal(axes, scalar_positions[:, None].expand(-1, 3))


def attn_q_dtype(attn_configs: AttentionConfigs) -> torch.dtype:
    # Dynamic mode keeps Q in the base dtype to match the gathered cache.
    if _is_dynamic_fp8(attn_configs):
        return attn_configs.dtype
    # FA3 FP8 (Hopper wgmma) requires Q/KV in the same FP8 dtype and is
    # SM90-only with head_dim 64/128/256;
    # Otherwise keep Q in fp16/bf16 (FA2 KV-dequant path)
    # Q uses the same unit-scale FP8 contract as the KV cache on this path.
    if (
        attn_configs.kv_cache_dtype == KvCacheDataType.FP8
        and is_sm90()
        and attn_configs.size_per_head in (64, 128, 256)
    ):
        return torch.float8_e4m3fn
    return attn_configs.dtype


def _page_geometry(attn_configs: AttentionConfigs) -> tuple[int, int, int]:
    kernel_page_size = int(attn_configs.kernel_tokens_per_block)
    if not _is_dynamic_fp8(attn_configs):
        return kernel_page_size, kernel_page_size, 1
    physical_page_size = int(attn_configs.tokens_per_block)
    if (
        physical_page_size <= 0
        or kernel_page_size <= 0
        or physical_page_size % kernel_page_size != 0
    ):
        raise ValueError(
            "FP8 KV cache mode 2 requires tokens_per_block to be divisible by "
            "kernel_tokens_per_block, got "
            f"{physical_page_size} and {kernel_page_size}"
        )
    return (
        physical_page_size,
        kernel_page_size,
        physical_page_size // kernel_page_size,
    )


_DYNAMIC_FP8_DIRECT_SCALE_JIT_VERSION = "v3"
_DYNAMIC_FP8_DIRECT_SCALE_VARIANT_NAME = "RtpLlmDynamicFp8Attention<use_custom_mask>"
_DYNAMIC_FP8_DIRECT_SCALE_VARIANT_DECL = r"""
#include <flashinfer/attention/variants.cuh>

template <typename Params>
struct RtpLlmDynamicFp8DefaultParams {
  const Params& source;
  float sm_scale;
  float logits_soft_cap;
  int32_t window_left;
  uint8_t* maybe_custom_mask;

  __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return source.get_qo_len(batch_idx);
  }

  __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return source.get_kv_len(batch_idx);
  }
};

template <bool use_custom_mask>
struct RtpLlmDynamicFp8Attention
    : DefaultAttention<false, USE_SLIDING_WINDOW,
                       USE_LOGITS_SOFT_CAP, false> {
  using Base = DefaultAttention<false, USE_SLIDING_WINDOW,
                                USE_LOGITS_SOFT_CAP, false>;

  static constexpr bool use_per_token_kv_scale = true;

  template <typename Params>
  __device__ __forceinline__ RtpLlmDynamicFp8Attention(
      const Params& params, uint32_t batch_idx, uint8_t* smem_ptr)
      : Base(RtpLlmDynamicFp8DefaultParams<Params>{
                 params, math::rsqrt(float(HEAD_DIM_QK)), 0.0f,
                 params.window_left, nullptr},
             batch_idx, smem_ptr) {}

  template <typename Params>
  __device__ __forceinline__ float KVScale(
      const Params&, uint32_t, uint32_t, uint32_t, uint32_t) const {
    return 1.0f;
  }

  __device__ __forceinline__ float KVScale(
      const PagedParams& params, uint32_t batch_idx, uint32_t kv_idx,
      uint32_t kv_head_idx, uint32_t kv_selector) const {
    if (kv_idx >= this->kv_len) {
      return 1.0f;
    }

    uint32_t page_offset, token_offset;
    params.paged_kv.page_size.divmod(kv_idx, page_offset, token_offset);
    const auto kernel_page_id =
        params.paged_kv.indices[params.paged_kv.indptr[batch_idx] + page_offset];
    const uint64_t scale_offset =
        ((static_cast<uint64_t>(kernel_page_id) * 2 + kv_selector) *
             params.paged_kv.num_heads +
         kv_head_idx) *
            params.paged_kv.page_size +
        token_offset;
    return params.kv_scale[scale_offset];
  }
};
"""


def _dtype_uri_component(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.").replace(".", "_")


def _dynamic_fp8_decode_jit_args(
    q_dtype: torch.dtype, output_dtype: torch.dtype, head_dim: int
) -> tuple[Any, ...]:
    kv_dtype = torch.float8_e4m3fn
    index_dtype = torch.int32
    uri = (
        f"rtp_llm_dynamic_fp8_direct_scale_{_DYNAMIC_FP8_DIRECT_SCALE_JIT_VERSION}_"
        f"q_{_dtype_uri_component(q_dtype)}_"
        f"kv_{_dtype_uri_component(kv_dtype)}_"
        f"o_{_dtype_uri_component(output_dtype)}_"
        f"idx_{_dtype_uri_component(index_dtype)}_"
        f"hdq_{head_dim}_hdv_{head_dim}"
    )
    return (
        uri,
        q_dtype,
        kv_dtype,
        output_dtype,
        index_dtype,
        head_dim,
        head_dim,
        ["kv_scale"],
        ["float"],
        [],
        [],
        _DYNAMIC_FP8_DIRECT_SCALE_VARIANT_NAME,
        _DYNAMIC_FP8_DIRECT_SCALE_VARIANT_DECL,
    )


def _create_dynamic_fp8_decode_wrapper(
    workspace: torch.Tensor,
    q_dtype: torch.dtype,
    output_dtype: torch.dtype,
    head_dim: int,
) -> BatchDecodeWithPagedKVCacheWrapper:
    include_dir = (
        Path(__file__).resolve().parent / "flashinfer_direct_scale" / "include"
    )
    patched_header = include_dir / "flashinfer" / "attention" / "prefill.cuh"
    if not patched_header.is_file():
        raise RuntimeError(f"missing FlashInfer direct-scale header: {patched_header}")

    jit_args = _dynamic_fp8_decode_jit_args(q_dtype, output_dtype, head_dim)
    uri = jit_args[0]
    with _g_flashinfer_jit_header_lock:
        cached_module = _g_dynamic_fp8_jit_modules.get(uri)
        if cached_module is None:
            original_generator = flashinfer_decode.gen_customize_batch_prefill_module

            def generate_with_direct_scale_headers(*args, **kwargs):
                spec = original_generator(*args, **kwargs)
                spec.extra_include_dirs = [
                    include_dir,
                    *(spec.extra_include_dirs or []),
                ]
                return spec

            flashinfer_decode.gen_customize_batch_prefill_module = (
                generate_with_direct_scale_headers
            )
            try:
                jit_module = flashinfer_decode.get_batch_prefill_jit_module(
                    uri,
                    flashinfer_decode.gen_customize_batch_prefill_module(
                        "fa2", *jit_args
                    ).build_and_load(),
                )
            finally:
                flashinfer_decode.gen_customize_batch_prefill_module = (
                    original_generator
                )
            cached_module = (jit_module, list(jit_args[7]))
            _g_dynamic_fp8_jit_modules[uri] = cached_module

    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "HND",
        backend="fa2",
        use_tensor_cores=True,
    )
    # FlashInfer 0.6.9 has no constructor argument for a preloaded JIT module.
    wrapper._jit_module = cached_module[0]
    wrapper._jit_additional_tensor_names = list(cached_module[1])
    return wrapper


def _validate_dynamic_fp8_scale(
    kv_cache: LayerKVCache, num_kv_heads: int, kernel_page_size: int
) -> torch.Tensor:
    kv_payload = kv_cache.kv_cache_base
    kv_scale = getattr(kv_cache, "kv_scale_base", None)
    if kv_payload is None or not isinstance(kv_payload, torch.Tensor):
        raise ValueError("FP8 KV cache mode 2 requires a kv_cache_base tensor")
    if not kv_payload.is_cuda:
        raise ValueError("FP8 KV cache mode 2 requires kv_cache_base to be CUDA")
    if kv_payload.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "FP8 KV cache mode 2 requires kv_cache_base dtype torch.float8_e4m3fn"
        )
    if not kv_payload.is_contiguous():
        raise ValueError("FP8 KV cache mode 2 requires contiguous kv_cache_base")
    expected_payload_shape = (2, num_kv_heads, kernel_page_size)
    if kv_payload.dim() != 5 or tuple(kv_payload.shape[1:4]) != expected_payload_shape:
        raise ValueError(
            "FP8 KV cache mode 2 requires kv_cache_base shape "
            f"[num_kernel_pages, 2, {num_kv_heads}, {kernel_page_size}, head_dim], "
            f"got {tuple(kv_payload.shape)}"
        )
    if kv_scale is None or not isinstance(kv_scale, torch.Tensor):
        raise ValueError("FP8 KV cache mode 2 requires a kv_scale_base tensor")
    if not kv_scale.is_cuda:
        raise ValueError("FP8 KV cache mode 2 requires kv_scale_base to be CUDA")
    if kv_scale.dtype != torch.float32:
        raise ValueError(
            "FP8 KV cache mode 2 requires kv_scale_base dtype torch.float32"
        )
    if not kv_scale.is_contiguous():
        raise ValueError("FP8 KV cache mode 2 requires contiguous kv_scale_base")
    if kv_scale.device != kv_payload.device:
        raise ValueError(
            "FP8 KV cache mode 2 requires kv_scale_base and kv_cache_base "
            "on the same device"
        )
    expected_width = 2 * num_kv_heads * kernel_page_size
    if (
        kv_scale.dim() != 2
        or kv_scale.shape[0] != kv_payload.shape[0]
        or kv_scale.shape[1] != expected_width
    ):
        raise ValueError(
            "FP8 KV cache mode 2 requires kv_scale_base shape "
            f"[{kv_payload.shape[0]}, {expected_width}], got {tuple(kv_scale.shape)}"
        )
    return kv_scale


def _compact_page_indices(source_page_indices: torch.Tensor) -> torch.Tensor:
    """Create dense page ids on the same device used by FlashInfer plan()."""
    return torch.arange(
        source_page_indices.numel(),
        dtype=source_page_indices.dtype,
        device=source_page_indices.device,
    )


def _gather_dynamic_fp8_cache(
    kv_cache: LayerKVCache,
    source_page_indices: torch.Tensor,
    output_dtype: torch.dtype,
    num_kv_heads: int,
    head_dim: int,
    physical_page_size: int,
    kernel_page_size: int,
    subdivision: int,
) -> torch.Tensor:
    """Gather arbitrary persistent pages into a dense base-dtype cache."""
    kv_scales = getattr(kv_cache, "kv_scale_base", None)
    if kv_scales is None or kv_scales.numel() == 0:
        raise ValueError("FP8 KV cache mode 2 requires a non-empty kv_scale_base")
    output = torch.empty(
        (
            source_page_indices.numel(),
            2,
            num_kv_heads,
            kernel_page_size,
            head_dim,
        ),
        dtype=output_dtype,
        device=kv_cache.kv_cache_base.device,
    )
    rtp_llm_ops.gather_and_dequantize_fp8_kv_cache(
        kv_cache.kv_cache_base,
        kv_scales,
        source_page_indices,
        output,
        physical_page_size,
        kernel_page_size,
        subdivision,
    )
    return output


class PyFlashinferPrefillPagedAttnOp(object):
    """FlashInfer Prefill Attention Op with Paged KV Cache support"""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        backend: str = "auto",
    ) -> None:
        self.g_workspace_buffer = get_py_flashinfer_workspace_buffer()
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.size_per_head
        self.dynamic_fp8 = _is_dynamic_fp8(attn_configs)
        (
            self.physical_page_size,
            self.page_size,
            self.subdivision,
        ) = _page_geometry(attn_configs)
        self.dtype = attn_configs.dtype
        self.kv_dtype = attn_kv_dtype(attn_configs)
        self.q_dtype = attn_q_dtype(attn_configs)
        self.max_seq_len = attn_configs.max_seq_len
        self.is_causal = attn_configs.is_causal
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.enable_cuda_graph = attn_inputs.is_cuda_graph
        self.prefill_cuda_graph_copy_params = None
        # Pre-allocated buffers for CUDA graph copy path (avoid per-forward allocation)
        self._aligned_q_buf = None
        # reserve buffer for q cast
        self._aligned_q_cast_buf = None
        self._compact_out_buf = None
        self._active_source_page_indices = None
        # Use Paged KV Cache wrapper
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
            backend=backend,
        )

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_params(self, params: rtp_llm_ops.FlashInferMlaAttnParams):
        """Set the params object to be used by this op."""
        self.fmha_params = params

    def prepare(
        self,
        attn_inputs: PyAttentionInputs,
        forbid_realloc: bool = False,
    ) -> ParamsBase:
        """
        Prepare the prefill wrapper with paged KV cache parameters.

        forbid_realloc: True only when called from prepare_cuda_graph (replay); forbids buffer realloc.
        """
        check_attention_inputs(attn_inputs)
        block_id_host = attn_inputs.kv_cache_kernel_block_id
        if block_id_host is None or block_id_host.numel() == 0:
            block_id_host = attn_inputs.kv_cache_kernel_block_id_device
        # Keep the same fill path for capture and replay: the host fill sizes
        # buffers exactly while the device fill sizes for the worst case, so
        # switching paths between capture and replay forces a (forbidden)
        # reallocation during graph replay.
        if attn_inputs.input_lengths.is_cuda:
            self.fmha_params.fill_params_mha_device(
                _device_or(
                    attn_inputs.prefix_lengths_device, attn_inputs.prefix_lengths
                ),
                attn_inputs.sequence_lengths,
                _device_or(attn_inputs.input_lengths_device, attn_inputs.input_lengths),
                _device_or(
                    attn_inputs.kv_cache_kernel_block_id_device,
                    attn_inputs.kv_cache_kernel_block_id,
                ),
                self.page_size,
                forbid_realloc,
            )
        else:
            self.fmha_params.fill_params(
                _host_i32(attn_inputs.prefix_lengths),
                _host_i32(attn_inputs.sequence_lengths),
                _host_i32(attn_inputs.input_lengths),
                _host_i32(block_id_host),
                self.page_size,
                forbid_realloc,
            )
        # Store CUDA graph copy parameters
        # Define qo_indptr early for CUDA graph initialization
        if attn_inputs.prefill_cuda_graph_copy_params is not None:
            # For CUDA graph mode, create a buffer that will be filled later
            self.input_lengths = attn_inputs.input_lengths
            self.cu_seq_lens = attn_inputs.cu_seqlens_device
            qo_indptr = attn_inputs.cu_seqlens_device.clone()
        else:
            qo_indptr = attn_inputs.cu_seqlens_device[
                : attn_inputs.input_lengths.size(0) + 1
            ]

        if self.enable_cuda_graph and self.prefill_wrapper._qo_indptr_buf is None:
            self.prefill_wrapper._use_cuda_graph = True
            self.prefill_wrapper._qo_indptr_buf = qo_indptr
            self.prefill_wrapper._paged_kv_indptr_buf = (
                self.fmha_params.decode_page_indptr_d
            )
            self.prefill_wrapper._paged_kv_last_page_len_buf = (
                self.fmha_params.paged_kv_last_page_len_d
            )
            self.prefill_wrapper._paged_kv_indices_buf = self.fmha_params.page_indice_d
            self.prefill_wrapper._fixed_batch_size = (
                len(attn_inputs.cu_seqlens_device) - 1
            )
            if attn_inputs.prefill_cuda_graph_copy_params is not None:
                self.prefill_cuda_graph_copy_params = (
                    attn_inputs.prefill_cuda_graph_copy_params
                )
                # input_lengths and cu_seq_lens were already set above
                self.qo_indptr = qo_indptr
                # Fill with cumulative sequence: [0, max_seq_len, 2*max_seq_len, ...]
                self.qo_indptr.copy_(
                    torch.arange(
                        self.qo_indptr.size(0),
                        device=self.qo_indptr.device,
                        dtype=self.qo_indptr.dtype,
                    )
                    * self.prefill_cuda_graph_copy_params.max_seq_len
                )

        # Update buffers for subsequent calls if in CUDA graph mode
        if self.prefill_cuda_graph_copy_params is not None:
            assert attn_inputs.prefill_cuda_graph_copy_params is not None
            assert self.input_lengths is not None
            assert self.cu_seq_lens is not None
            self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size[0] = (
                attn_inputs.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size
            )
            self.input_lengths[: attn_inputs.input_lengths.size(0)] = (
                attn_inputs.input_lengths
            )
            self.cu_seq_lens[: attn_inputs.cu_seqlens_device.size(0)] = (
                attn_inputs.cu_seqlens_device
            )
            qo_indptr = self.qo_indptr

        plan_page_indices = self.fmha_params.page_indice_d
        if self.dynamic_fp8:
            self._active_source_page_indices = _active_page_indices(
                plan_page_indices, self.fmha_params.decode_page_indptr_d
            )
            plan_page_indices = _compact_page_indices(self._active_source_page_indices)

        self.prefill_wrapper.plan(
            qo_indptr,
            self.fmha_params.decode_page_indptr_d,
            plan_page_indices,
            self.fmha_params.paged_kv_last_page_len_d,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.page_size,
            causal=self.is_causal,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_dtype,
            o_data_type=self.dtype,
        )
        return self.fmha_params

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        return True

    def forward(
        self, q: torch.Tensor, kv_cache: Optional[LayerKVCache]
    ) -> torch.Tensor:
        """
        Forward pass with paged KV cache

        Args:
            q: Query tensor [total_tokens, num_heads, head_dim]
            kv_cache: Paged KV cache [num_pages, 2, page_size, kv_heads, head_dim]
            params: Parameters (not used currently)

        Returns:
            output: [total_tokens, num_heads, head_dim]
        """
        from rtp_llm.ops.compute_ops import (
            cuda_graph_copy_large2small,
            cuda_graph_copy_small2large,
        )

        assert kv_cache is not None, "kv_cache is required for paged attention"
        assert (
            q.dim() == 3
        ), f"Expected q to be 3D tensor [total_tokens, num_heads, head_dim], got {q.dim()}D"

        if self.dynamic_fp8:
            if self._active_source_page_indices is None:
                raise RuntimeError("paged prefill must be prepared before forward")
            paged_kv_cache = _gather_dynamic_fp8_cache(
                kv_cache,
                self._active_source_page_indices,
                self.dtype,
                self.local_kv_head_num,
                self.head_dim_qk,
                self.physical_page_size,
                self.page_size,
                self.subdivision,
            )
        else:
            paged_kv_cache = kv_cache.kv_cache_base
            if paged_kv_cache.dim() == 2:
                paged_kv_cache = common.reshape_paged_kv_cache(
                    paged_kv_cache,
                    self.local_kv_head_num,
                    self.page_size,
                    self.head_dim_qk,
                )
        # CUDA graph copy logic for prefill
        if self.prefill_cuda_graph_copy_params:
            assert (
                self.input_lengths is not None
            ), "input_lengths is required for CUDA graph copy"
            assert (
                self.cu_seq_lens is not None
            ), "cu_seq_lens is required for CUDA graph copy"

            # Reshape from 3D [token_num, head_num, head_size] to 2D [token_num, hidden_size]
            token_num, head_num, head_size = q.shape
            hidden_size = head_num * head_size

            # Pre-allocate buffers on first use (avoid per-forward GPU allocation)
            total_len = (
                self.prefill_cuda_graph_copy_params.max_seq_len
                * self.prefill_cuda_graph_copy_params.max_batch_size
            )
            if self._aligned_q_buf is None or self._aligned_q_buf.shape != (
                total_len,
                hidden_size,
            ):
                self._aligned_q_buf = torch.zeros(
                    (total_len, hidden_size), dtype=q.dtype, device=q.device
                )
            if self._compact_out_buf is None or self._compact_out_buf.shape != (
                token_num,
                hidden_size,
            ):
                self._compact_out_buf = torch.zeros(
                    (token_num, hidden_size), dtype=q.dtype, device=q.device
                )

            q_2d = q.view(token_num, hidden_size).contiguous()
            self._aligned_q_buf.zero_()

            # Copy small to large (compact -> aligned)
            cuda_graph_copy_small2large(
                q_2d,
                self._aligned_q_buf,
                self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size,
                self.prefill_cuda_graph_copy_params.max_batch_size,
                self.prefill_cuda_graph_copy_params.max_seq_len,
                self.input_lengths,
                hidden_size,
                self.cu_seq_lens,
            )

            # Reshape back to 3D for FlashInfer
            q_aligned = self._aligned_q_buf.view(total_len, head_num, head_size)
            if q_aligned.dtype != self.q_dtype:
                if (
                    self._aligned_q_cast_buf is None
                    or self._aligned_q_cast_buf.shape != q_aligned.shape
                    or self._aligned_q_cast_buf.dtype != self.q_dtype
                    or self._aligned_q_cast_buf.device != q_aligned.device
                ):
                    self._aligned_q_cast_buf = torch.empty(
                        q_aligned.shape,
                        dtype=self.q_dtype,
                        device=q_aligned.device,
                    )
                rtp_llm_ops.per_tensor_quant_fp8(
                    q_aligned,
                    self._aligned_q_cast_buf,
                    _get_fp8_unit_scale_tensor(q_aligned.device),
                    True,
                )
                q_aligned = self._aligned_q_cast_buf

            # Paged FP8 defaults to unit scales and the output dtype from plan().
            result = self.prefill_wrapper.run(q_aligned, paged_kv_cache)

            # Reshape result to 2D for copy back (ensure contiguous)
            result_2d = result.view(total_len, hidden_size).contiguous()
            self._compact_out_buf.zero_()

            # Copy large to small (aligned -> compact)
            cuda_graph_copy_large2small(
                result_2d,
                self._compact_out_buf,
                self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size,
                self.prefill_cuda_graph_copy_params.max_batch_size,
                self.prefill_cuda_graph_copy_params.max_seq_len,
                self.input_lengths,
                hidden_size,
                self.cu_seq_lens,
            )

            # Reshape back to 3D
            result = self._compact_out_buf.view(token_num, head_num, head_size)
        else:
            # Dynamic mode consumes base-dtype Q and a dequantized compact cache.
            # Legacy FP8 keeps the existing unit-scale cast path.
            q_input = (
                q if self.dynamic_fp8 else quantize_to_fp8_if_needed(q, self.q_dtype)
            )
            result = self.prefill_wrapper.run(q_input, paged_kv_cache)

        return result


class PyFlashinferPrefillAttnOp(object):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        backend: str = "auto",
        sm_scale: Optional[float] = None,
    ) -> None:
        self.g_workspace_buffer = get_py_flashinfer_workspace_buffer()
        # attn_configs.head_num and kv_head_num are already divided by tp_size in ModelConfig::getAttentionConfigs
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.page_size = attn_configs.kernel_tokens_per_block
        # TODO: maybe use v_head_dim
        self.head_dim_vo = attn_configs.size_per_head
        self.prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.g_workspace_buffer,
            backend=backend,
        )
        self.dynamic_fp8 = _is_dynamic_fp8(attn_configs)
        self.dtype = attn_configs.dtype
        self.q_dtype = attn_q_dtype(attn_configs)
        self.kv_dtype = attn_kv_dtype(attn_configs)
        self.is_causal = attn_configs.is_causal
        self.sm_scale = sm_scale
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_params(self, params: rtp_llm_ops.FlashInferMlaAttnParams):
        """Set the params object to be used by this op."""
        self.fmha_params = params

    def prepare(self, attn_inputs: PyAttentionInputs) -> ParamsBase:
        """
        Prepare the prefill wrapper

        Args:
            attn_inputs: Attention inputs containing sequence information
        """
        batch_size = attn_inputs.input_lengths.size(0)
        cu_seqlens = attn_inputs.cu_seqlens_device[: batch_size + 1]

        # Encoder-only models (BERT) have no paged kv cache; fill_params
        # pybind requires a Tensor, so substitute an empty int32 tensor.
        kv_block_id = attn_inputs.kv_cache_kernel_block_id
        if kv_block_id is None or kv_block_id.numel() == 0:
            kv_block_id = attn_inputs.kv_cache_kernel_block_id_device
        if kv_block_id is None:
            kv_block_id = torch.empty(0, dtype=torch.int32)

        self.fmha_params.fill_params(
            _host_i32(attn_inputs.prefix_lengths),
            _host_i32(attn_inputs.sequence_lengths),
            _host_i32(attn_inputs.input_lengths),
            _host_i32(kv_block_id),
            self.page_size,
        )

        self.prefill_wrapper.plan(
            cu_seqlens,
            cu_seqlens,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.head_dim_vo,
            causal=self.is_causal,
            sm_scale=self.sm_scale,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_dtype,
            o_data_type=self.dtype,
        )
        return self.fmha_params

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        return (
            attn_inputs.prefix_lengths.numel() <= 0
            or attn_inputs.prefix_lengths.sum().item() == 0
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        if not self.dynamic_fp8:
            q = quantize_to_fp8_if_needed(q, self.q_dtype)
            k = quantize_to_fp8_if_needed(k, self.kv_dtype)
            v = quantize_to_fp8_if_needed(v, self.kv_dtype)
        if q.dtype == torch.float8_e4m3fn:
            # FlashInfer's FA3 FP8 need scale_q, scale_k, scale_v and an output matching the planned dtype.
            out = torch.empty(
                q.shape[:-1] + v.shape[-1:], dtype=self.dtype, device=q.device
            )
            return self.prefill_wrapper.run(
                q,
                k,
                v,
                FP8_UNIT_SCALE,
                FP8_UNIT_SCALE,
                FP8_UNIT_SCALE,
                out=out,
            )
        return self.prefill_wrapper.run(q, k, v)


class PyFlashinferHybridPrefillAttnOp(object):
    """FlashInfer hybrid prefill op.

    It evaluates attention over the newest ragged KV segment and the existing
    prefix paged KV cache sequentially on the current CUDA stream, then merges
    the two attention states with FlashInfer's LSE merge.
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        backend: str = "auto",
    ) -> None:
        self.g_workspace_buffer = get_py_flashinfer_workspace_buffer()
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.size_per_head
        self.dynamic_fp8 = _is_dynamic_fp8(attn_configs)
        (
            self.physical_page_size,
            self.page_size,
            self.subdivision,
        ) = _page_geometry(attn_configs)
        self.dtype = attn_configs.dtype
        self.kv_dtype = attn_kv_dtype(attn_configs)
        self.q_dtype = attn_q_dtype(attn_configs)
        self.is_causal = attn_configs.is_causal
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        # The serial ragged/write/paged flow can share one workspace buffer.
        self.ragged_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.g_workspace_buffer,
            backend=backend,
        )
        self.prefix_paged_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
            backend=backend,
        )

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_params(self, params: rtp_llm_ops.FlashInferMlaAttnParams):
        """Set the params object to be used by this op."""
        self.fmha_params = params

    def prepare(
        self,
        attn_inputs: PyAttentionInputs,
        forbid_realloc: bool = False,
    ) -> ParamsBase:
        """Prepare ragged-new and paged-prefix FlashInfer wrappers."""
        block_table = attn_inputs.kv_cache_kernel_block_id
        if block_table is None or block_table.numel() == 0:
            block_table = attn_inputs.kv_cache_kernel_block_id_device
        assert (
            block_table is not None and block_table.numel() > 0
        ), "hybrid prefill requires a non-empty kv_cache_kernel_block_id"
        self.fmha_params.fill_params(
            _host_i32(attn_inputs.prefix_lengths),
            _host_i32(attn_inputs.sequence_lengths),
            _host_i32(attn_inputs.input_lengths),
            _host_i32(block_table),
            self.page_size,
            forbid_realloc,
        )

        batch_size = attn_inputs.input_lengths.size(0)
        qo_indptr = attn_inputs.cu_seqlens_device[: batch_size + 1]

        self.ragged_wrapper.plan(
            qo_indptr,
            qo_indptr,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.head_dim_vo,
            causal=self.is_causal,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_dtype,
            o_data_type=self.dtype,
        )

        # batch_reuse_info_vec_h columns are defined in FlashInferMlaParams.cc.
        prefix_len_col = 1
        page_start_col = 2
        reuse_info = self.fmha_params.batch_reuse_info_vec_h
        prefix_lengths = reuse_info[:, prefix_len_col]
        if (prefix_lengths <= 0).any().item():
            raise ValueError(
                "hybrid prefill requires a non-empty prefix cache per batch item"
            )

        prefix_paged_kv_indptr = torch.empty(
            batch_size + 1, dtype=torch.int32, device="cpu"
        )
        prefix_paged_kv_indptr[:-1].copy_(reuse_info[:, page_start_col])
        prefix_paged_kv_indptr[-1] = self.fmha_params.reuse_cache_page_indice_h.numel()
        prefix_paged_kv_last_page_len = (prefix_lengths - 1) % self.page_size + 1

        prefix_plan_page_indices = self.fmha_params.reuse_cache_page_indice_d
        if self.dynamic_fp8:
            prefix_plan_page_indices = _compact_page_indices(prefix_plan_page_indices)

        self.prefix_paged_wrapper.plan(
            qo_indptr,
            prefix_paged_kv_indptr,
            prefix_plan_page_indices,
            prefix_paged_kv_last_page_len,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.page_size,
            causal=False,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_dtype,
            o_data_type=self.dtype,
        )
        return self.fmha_params

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        block_table = attn_inputs.kv_cache_kernel_block_id
        if block_table is None or block_table.numel() == 0:
            block_table = attn_inputs.kv_cache_kernel_block_id_device
        prefix_lengths = attn_inputs.prefix_lengths
        return (
            block_table is not None
            and block_table.numel() > 0
            and prefix_lengths is not None
            and prefix_lengths.numel() > 0
            and prefix_lengths.min().item() > 0
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        kv_cache_write_op: Optional[KVCacheWriteOp] = None,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required for hybrid prefill"
        if not self.dynamic_fp8:
            q = quantize_to_fp8_if_needed(q, self.q_dtype)
            k = quantize_to_fp8_if_needed(k, self.kv_dtype)
            v = quantize_to_fp8_if_needed(v, self.kv_dtype)
        if q.dtype == torch.float8_e4m3fn:
            # Positional scale_q/scale_k/scale_v; see FP8_UNIT_SCALE.
            out = torch.empty(
                q.shape[:-1] + v.shape[-1:], dtype=self.dtype, device=q.device
            )
            new_out, new_lse = self.ragged_wrapper.run(
                q,
                k,
                v,
                FP8_UNIT_SCALE,
                FP8_UNIT_SCALE,
                FP8_UNIT_SCALE,
                out=out,
                return_lse=True,
            )
        else:
            new_out, new_lse = self.ragged_wrapper.run(q, k, v, return_lse=True)

        if kv_cache_write_op is not None:
            kv_cache_write_op.forward(k, v, kv_cache)

        if self.dynamic_fp8:
            paged_kv_cache = _gather_dynamic_fp8_cache(
                kv_cache,
                self.fmha_params.reuse_cache_page_indice_d,
                self.dtype,
                self.local_kv_head_num,
                self.head_dim_qk,
                self.physical_page_size,
                self.page_size,
                self.subdivision,
            )
        else:
            paged_kv_cache = kv_cache.kv_cache_base
            if paged_kv_cache.dim() == 2:
                paged_kv_cache = common.reshape_paged_kv_cache(
                    paged_kv_cache,
                    self.local_kv_head_num,
                    self.page_size,
                    self.head_dim_qk,
                )

        prefix_out, prefix_lse = self.prefix_paged_wrapper.run(
            q, paged_kv_cache, return_lse=True
        )
        merge_state_in_place(new_out, new_lse, prefix_out, prefix_lse)
        return new_out


class PyFlashinferPrefillImplBase(FMHAImplBase):
    """Base class for FlashInfer prefill implementations (Ragged and Paged)."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        """Initialize prefill implementation with common setup.

        Args:
            attn_configs: Attention configuration
            attn_inputs: Attention inputs
        """
        # Store configs and inputs
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.dynamic_fp8 = _is_dynamic_fp8(attn_configs)
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs

        self.fmha_impl = self._create_fmha_impl(attn_configs, attn_inputs)
        use_fused_mrope = _use_fused_mrope_prefill(attn_configs, attn_inputs)
        self.fused_mrope_impl = (
            FusedRopeKVCachePrefillOpQKVOut(attn_configs) if use_fused_mrope else None
        )
        self.fused_mrope_params = (
            self.fused_mrope_impl.prepare(attn_inputs)
            if self.fused_mrope_impl is not None
            else None
        )
        self.rope_impl = (
            None
            if self.fused_mrope_impl is not None
            else self._create_rope_impl(attn_configs)
        )
        physical_page_size, kernel_page_size, _ = _page_geometry(attn_configs)
        self.kv_cache_write_op = KVCacheWriteOp(
            num_kv_heads=attn_configs.kv_head_num,
            head_size=attn_configs.size_per_head,
            physical_page_size=physical_page_size,
            kernel_page_size=kernel_page_size,
            dynamic_mode=self.dynamic_fp8,
        )
        self.create_params(attn_inputs)
        self.fmha_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        self.fmha_impl.prepare(attn_inputs, forbid_realloc=True)

    def create_params(self, attn_inputs: PyAttentionInputs):
        """Create FlashInfer MLA attention parameters.

        Similar to MLA implementation, this creates and initializes the params
        that will be used for both FMHA and RoPE operations.
        """
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.rope_params = self.fmha_params
        # Pass the shared params to all ops
        self.fmha_impl.set_params(self.fmha_params)
        if self.rope_impl is not None:
            self.rope_impl.set_params(self.rope_params)
        # KV cache write always needs params (even without RoPE)
        self.kv_cache_write_op.set_params(self.rope_params)

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        """Create FMHA implementation. To be overridden by subclasses."""
        raise NotImplementedError("Subclass must implement _create_fmha_impl")

    def _create_rope_impl(self, attn_configs: AttentionConfigs) -> Any:
        """Create RoPE implementation. To be overridden by subclasses."""
        raise NotImplementedError("Subclass must implement _create_rope_impl")

    def _split_qkv(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split QKV tensor into query, key, value.

        Args:
            qkv: QKV tensor [total_tokens, (num_heads + 2*num_kv_heads) * head_dim]

        Returns:
            Tuple of (query, key, value) tensors
        """
        qkv = qkv.reshape(qkv.shape[0], -1)
        num_heads = self.attn_configs.head_num
        num_kv_heads = self.attn_configs.kv_head_num
        head_dim = self.attn_configs.size_per_head

        q, k, v = torch.split(
            qkv,
            [
                head_dim * num_heads,
                head_dim * num_kv_heads,
                head_dim * num_kv_heads,
            ],
            dim=-1,
        )

        query = q.reshape(q.shape[0], num_heads, head_dim)
        key = k.reshape(k.shape[0], num_kv_heads, head_dim)
        value = v.reshape(v.shape[0], num_kv_heads, head_dim)

        return query, key, value

    def _apply_rope_and_split(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.fused_mrope_impl is not None:
            rotated_qkv = self.fused_mrope_impl.forward(
                qkv, None, self.fused_mrope_params
            )
            return self._split_qkv(rotated_qkv)
        if self.need_rope_kv_cache and self.rope_impl is not None:
            return self.rope_impl.forward(qkv)
        return self._split_qkv(qkv)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Common forward implementation for all prefill implementations."""
        query, key, value = self._apply_rope_and_split(qkv)

        # Legacy FP8 shares one unit-scale cast between cache write and
        # attention. Dynamic mode must preserve base-dtype post-RoPE K/V for
        # per-row quantization and ragged attention.
        if not self.dynamic_fp8:
            kv_dtype = attn_kv_dtype(self.attn_configs)
            key = quantize_to_fp8_if_needed(key, kv_dtype)
            value = quantize_to_fp8_if_needed(value, kv_dtype)

        if self.need_rope_kv_cache or self.dynamic_fp8:
            self.kv_cache_write_op.forward(key, value, kv_cache)

        fmha_inputs = self._prepare_fmha_input(query, key, value)

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(*fmha_inputs, kv_cache)

    def _prepare_fmha_input(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Positional args for fmha_impl.forward; kv_cache is appended by forward().

        Default: only query (paged layout, KV is read from the cache).
        """
        return (query,)


class PyFlashinferPagedPrefillImpl(PyFlashinferPrefillImplBase):
    """FlashInfer prefill implementation with paged KV cache layout using MhaRotaryEmbeddingOp."""

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        """Create paged FMHA implementation."""
        return PyFlashinferPrefillPagedAttnOp(attn_configs, attn_inputs)

    def _create_rope_impl(self, attn_configs: AttentionConfigs) -> Any:
        """Create RoPE implementation for paged layout."""
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return MhaRotaryEmbeddingOp(attn_configs)

    def _prepare_fmha_input(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """For paged layout, only pass query (KV is already in cache)."""
        return (query,)

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """Check if paged prefill implementation is supported.

        Returns True if:
        1. Not running on SM10x datacenter Blackwell, where TRTLLMGen is preferred.
           SM12x consumer Blackwell keeps this FlashInfer paged fallback because
           TRTLLMGen/XQA do not have sm_120a support in this build.
        2. The underlying paged FMHA op supports the inputs
        3. MhaRotaryEmbeddingOp supports the inputs
        """
        return (
            not is_sm10x()
            and PyFlashinferPrefillPagedAttnOp.support(attn_inputs)
            and (
                supports_scalar_prefill_rope(attn_configs, attn_inputs)
                or _supports_prefill_rope(attn_configs, attn_inputs)
            )
        )

    def support_cuda_graph(self) -> bool:
        return not self.dynamic_fp8


class PyFlashinferHybridPrefillImpl(PyFlashinferPrefillImplBase):
    """FlashInfer hybrid prefill implementation.

    The current qkv chunk first attends to the new ragged KV, appends that KV to
    the cache, and then attends to the existing prefix through its prefix-only
    page table. The two attention states are merged via LSE.
    """

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        """Create hybrid FMHA implementation."""
        return PyFlashinferHybridPrefillAttnOp(attn_configs, attn_inputs)

    def _create_rope_impl(self, attn_configs: AttentionConfigs) -> Any:
        """Create RoPE implementation for hybrid layout."""
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return MhaRotaryEmbeddingOp(attn_configs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Run ragged attention, append KV, then run paged prefix attention."""
        query, key, value = self._apply_rope_and_split(qkv)

        if not self.dynamic_fp8:
            query = quantize_to_fp8_if_needed(query, attn_q_dtype(self.attn_configs))
            kv_dtype = attn_kv_dtype(self.attn_configs)
            key = quantize_to_fp8_if_needed(key, kv_dtype)
            value = quantize_to_fp8_if_needed(value, kv_dtype)

        # Write new K/V after ragged attention and before paged attention.
        kv_cache_write_op = (
            self.kv_cache_write_op
            if self.need_rope_kv_cache or self.dynamic_fp8
            else None
        )
        result = self.fmha_impl.forward(
            query,
            key,
            value,
            kv_cache,
            kv_cache_write_op,
        )
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return result

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """Check if hybrid prefill implementation is supported."""
        return (
            not attn_inputs.is_cuda_graph
            and not is_sm10x()
            and PyFlashinferHybridPrefillAttnOp.support(attn_inputs)
            and _supports_prefill_rope(attn_configs, attn_inputs)
        )

    def support_cuda_graph(self) -> bool:
        return False


class PyFlashinferPrefillImpl(PyFlashinferPrefillImplBase):
    """FlashInfer prefill implementation with ragged KV cache layout using MhaRotaryEmbeddingOp."""

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        """Create ragged FMHA implementation."""
        return PyFlashinferPrefillAttnOp(attn_configs)

    def _create_rope_impl(self, attn_configs: AttentionConfigs) -> Any:
        """Create RoPE implementation for ragged layout."""
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return MhaRotaryEmbeddingOp(attn_configs)

    def _prepare_fmha_input(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """For ragged layout, pass Q/K/V directly to the ragged wrapper."""
        return query, key, value

    def support_cuda_graph(self) -> bool:
        return False

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """Check if ragged prefill implementation is supported.

        Returns True if:
        1. The underlying ragged FMHA op supports the inputs
           (requires prefix_lengths to be empty or zero)
        2. MhaRotaryEmbeddingOp supports the inputs
        3. Mrope is not used

        Note: Unlike the paged variant, ragged prefill is kept enabled on
        Blackwell: TRT-LLM Gen prefill requires a paged kv cache and
        therefore does not cover BERT-style encoder-only inputs that lack
        one. Without this fallback, sm_120 has no usable prefill impl for
        such cases.
        """
        return PyFlashinferPrefillAttnOp.support(
            attn_inputs
        ) and _supports_prefill_rope(attn_configs, attn_inputs)


def determine_use_tensor_core_from_configs(attn_configs: AttentionConfigs) -> bool:
    """Determine whether to use tensor cores based on attention configs."""
    # Use tensor cores for larger head dimensions and when kv_head_num matches requirements
    return attn_configs.head_num // attn_configs.kv_head_num >= 4


class PyFlashinferDecodeAttnOp(object):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ) -> None:
        self.g_workspace_buffer = get_py_flashinfer_workspace_buffer()
        # attn_configs already has head_num and kv_head_num divided by tp_size
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.size_per_head
        self.dynamic_fp8 = _is_dynamic_fp8(attn_configs)
        (
            self.physical_page_size,
            self.seq_size_per_block,
            self.subdivision,
        ) = _page_geometry(attn_configs)
        self.dtype = attn_configs.dtype
        if self.dynamic_fp8:
            self.use_tensor_core = True
            self.q_dtype = self.dtype
            self.kv_dtype = torch.float8_e4m3fn
            self.decode_wrapper = _create_dynamic_fp8_decode_wrapper(
                self.g_workspace_buffer,
                self.q_dtype,
                self.dtype,
                self.head_dim_qk,
            )
        else:
            self.use_tensor_core = determine_use_tensor_core_from_configs(attn_configs)
            self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.g_workspace_buffer,
                "HND",
                use_tensor_cores=self.use_tensor_core,
            )
            self.kv_dtype = attn_kv_dtype(attn_configs)
            # CUDA-core decode dequantizes FP8 KV; tensor-core decode uses the
            # batch-prefill path and therefore shares attn_q_dtype().
            self.q_dtype = (
                attn_q_dtype(attn_configs) if self.use_tensor_core else self.dtype
            )
        self.enable_cuda_graph = attn_inputs.is_cuda_graph
        # Snapshot of the page indptr used by the last CUDA-core graph plan.
        # Dtype, head counts, and page size are fixed for this op's lifetime.
        self._cuda_core_plan_page_indptr_h: Optional[torch.Tensor] = None
        self._dynamic_fp8_cuda_graph_state: Optional[tuple] = None
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_params(self, params: rtp_llm_ops.FlashInferMlaAttnParams) -> None:
        """Install params before initial prepare and invalidate the plan snapshot."""
        if self.decode_wrapper._fixed_batch_size != 0:
            raise RuntimeError(
                "FlashInfer decode params cannot be replaced after CUDA graph buffers "
                "have been bound"
            )
        self.fmha_params = params
        self._cuda_core_plan_page_indptr_h = None

    def _tensor_core_cuda_graph_needs_replan(self) -> bool:
        # FlashInfer BatchDecode routes tensor-core decode through BatchPrefill.
        # Its plan derives kv_lens from both page indptr and last-page lengths,
        # so the pinned 0.2.5/0.6.9/0.6.15.post1 implementations must replan
        # every replay.
        return self.enable_cuda_graph and self.use_tensor_core

    def _uses_cuda_core_graph_plan_cache(self) -> bool:
        return self.enable_cuda_graph and not self.use_tensor_core

    def _cuda_core_cuda_graph_needs_replan(self) -> bool:
        if not self._uses_cuda_core_graph_plan_cache():
            return False
        current_page_indptr = self.fmha_params.decode_page_indptr_h
        return self._cuda_core_plan_page_indptr_h is None or not torch.equal(
            self._cuda_core_plan_page_indptr_h,
            current_page_indptr,
        )

    def _cuda_graph_replay_needs_replan(self) -> bool:
        return (
            self._tensor_core_cuda_graph_needs_replan()
            or self._cuda_core_cuda_graph_needs_replan()
        )

    def _plan_decode_wrapper(self, attn_inputs: PyAttentionInputs) -> None:
        use_cuda_core_graph_plan_cache = self._uses_cuda_core_graph_plan_cache()
        if self.dynamic_fp8 and not self.enable_cuda_graph:
            page_indptr = self.fmha_params.decode_page_indptr_h
            page_indice = self.fmha_params.page_indice_d
            last_page_len = self.fmha_params.paged_kv_last_page_len_h
            plan_kwargs = {"non_blocking": True}
        elif self.use_tensor_core:
            # Tensor-core decode plans from host mirrors in both eager and graph
            # modes; only replay decides whether another plan call is required.
            page_indptr = self.fmha_params.decode_page_indptr_h
            # Graph indices already alias the device buffer updated by
            # fill_params. Passing host indices makes FlashInfer's graph plan
            # perform a blocking H2D copy, serializing the previous replay.
            page_indice = (
                self.fmha_params.page_indice_d
                if self.enable_cuda_graph
                else self.fmha_params.page_indice_h
            )
            last_page_len = self.fmha_params.paged_kv_last_page_len_h
            plan_kwargs = {"non_blocking": True}
        elif use_cuda_core_graph_plan_cache:
            # FlashInfer 0.2.5/0.6.9/0.6.15.post1 CUDA-core BatchDecode derives
            # its work partition from page indptr. The last-page-length values
            # are read by the kernel at runtime; their tensor length still
            # matches the graph's fixed batch size. Keep both metadata tensors
            # on host to avoid a D2H copy. Indices stay on device and alias the
            # graph-bound buffer refreshed by fill_params() before each replay.
            # It is the same tensor as the wrapper's indices buffer, so the
            # plan-internal copy is a no-op; forbid_realloc=True guarantees
            # that its capture-time capacity remains valid during replay.
            page_indptr = self.fmha_params.decode_page_indptr_h
            page_indice = self.fmha_params.page_indice_d
            last_page_len = self.fmha_params.paged_kv_last_page_len_h
            plan_kwargs = {"non_blocking": True}
        else:
            page_indptr = self.fmha_params.decode_page_indptr_d
            page_indice = self.fmha_params.page_indice_d
            last_page_len = self.fmha_params.paged_kv_last_page_len_d
            plan_kwargs = {}

        if self.dynamic_fp8:
            # The custom scale reader uses these IDs to address both the FP8
            # payload and its scale row, so never remap them to dense IDs.
            page_indice = _active_page_indices(page_indice, page_indptr)

        self.decode_wrapper.plan(
            page_indptr,
            page_indice,
            last_page_len,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.seq_size_per_block,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_dtype,
            o_data_type=self.dtype,
            **plan_kwargs,
        )
        if use_cuda_core_graph_plan_cache:
            self._cuda_core_plan_page_indptr_h = (
                self.fmha_params.decode_page_indptr_h.clone()
            )

    def _validate_dynamic_fp8_cuda_graph_state(self) -> None:
        if not (self.dynamic_fp8 and self.enable_cuda_graph):
            return
        state = (
            tuple(self.decode_wrapper._plan_info),
            self.decode_wrapper._fixed_batch_size,
            self.decode_wrapper._float_workspace_buffer.data_ptr(),
            self.decode_wrapper._int_workspace_buffer.data_ptr(),
            self.decode_wrapper._qo_indptr_buf.data_ptr(),
            self.decode_wrapper._paged_kv_indptr_buf.data_ptr(),
            self.decode_wrapper._paged_kv_indices_buf.data_ptr(),
            self.decode_wrapper._paged_kv_last_page_len_buf.data_ptr(),
        )
        if self._dynamic_fp8_cuda_graph_state is None:
            self._dynamic_fp8_cuda_graph_state = state
        elif state != self._dynamic_fp8_cuda_graph_state:
            raise RuntimeError(
                "FP8 KV cache mode 2 CUDA graph plan or buffer addresses changed"
            )

    def prepare(
        self,
        attn_inputs: PyAttentionInputs,
        forbid_realloc: bool = False,
    ) -> ParamsBase:
        """
        Prepare the decode wrapper with paged KV cache parameters.

        forbid_realloc: True only when called from prepare_cuda_graph (replay); forbids buffer realloc.
        """
        # Graph planning uses HOST mirrors for tensor-core decode and for the
        # CUDA-core topology cache. The device fill leaves those mirrors at
        # their stale capacity sizes (MIN_CACHE_BATCH_SIZE), which corrupts
        # the plan's batch size, so both graph backends use the host fill.
        if self.dynamic_fp8 and not self.enable_cuda_graph:
            block_id_device = _device_or(
                attn_inputs.kv_cache_kernel_block_id_device,
                attn_inputs.kv_cache_kernel_block_id,
            )
            if block_id_device is not None and not block_id_device.is_cuda:
                block_id_device = block_id_device.cuda()
            sequence_lengths_plus_1_device = attn_inputs.sequence_lengths_plus_1_device
            if (
                sequence_lengths_plus_1_device is None
                or not sequence_lengths_plus_1_device.is_cuda
            ):
                sequence_lengths_plus_1_device = (
                    _host_i32(attn_inputs.sequence_lengths) + 1
                ).to(block_id_device.device, non_blocking=True)
            self.fmha_params.fill_decode_params_device(
                _host_i32(attn_inputs.sequence_lengths),
                sequence_lengths_plus_1_device,
                block_id_device,
                self.seq_size_per_block,
            )
        elif (
            attn_inputs.input_lengths.is_cuda
            and not self.use_tensor_core
            and not self.enable_cuda_graph
        ):
            self.fmha_params.fill_params_mha_device(
                _device_or(
                    attn_inputs.prefix_lengths_device, attn_inputs.prefix_lengths
                ),
                attn_inputs.sequence_lengths,
                _device_or(attn_inputs.input_lengths_device, attn_inputs.input_lengths),
                _device_or(
                    attn_inputs.kv_cache_kernel_block_id_device,
                    attn_inputs.kv_cache_kernel_block_id,
                ),
                self.seq_size_per_block,
                forbid_realloc=forbid_realloc,
            )
        else:
            block_id_host = attn_inputs.kv_cache_kernel_block_id
            if block_id_host is None or block_id_host.numel() == 0:
                block_id_host = attn_inputs.kv_cache_kernel_block_id_device
            self.fmha_params.fill_params(
                _host_i32(attn_inputs.prefix_lengths),
                _host_i32(attn_inputs.sequence_lengths),
                _host_i32(attn_inputs.input_lengths),
                _host_i32(block_id_host),
                self.seq_size_per_block,
                forbid_realloc=forbid_realloc,
            )

        if self.enable_cuda_graph and self.decode_wrapper._fixed_batch_size == 0:
            batch_size = attn_inputs.input_lengths.size(0)
            self.decode_wrapper._use_cuda_graph = True
            # Both decode backends read these buffers during run(); replay only
            # updates fmha_params in-place, so the wrapper must hold these views.
            self.decode_wrapper._paged_kv_indptr_buf = (
                self.fmha_params.decode_page_indptr_d
            )
            self.decode_wrapper._paged_kv_last_page_len_buf = (
                self.fmha_params.paged_kv_last_page_len_d
            )
            self.decode_wrapper._paged_kv_indices_buf = self.fmha_params.page_indice_d
            self.decode_wrapper._fixed_batch_size = batch_size
            if self.use_tensor_core:
                self.decode_wrapper._qo_indptr_buf = torch.arange(
                    batch_size + 1,
                    dtype=torch.int32,
                    device=self.g_workspace_buffer.device,
                )

        self._plan_decode_wrapper(attn_inputs)
        self._validate_dynamic_fp8_cuda_graph_state()
        return self.fmha_params

    def prepare_for_cuda_graph_replay(self, attn_inputs: PyAttentionInputs) -> None:
        """Refresh FlashInfer runtime buffers before replaying the captured graph."""
        if not attn_inputs.sequence_lengths.is_cuda:
            # Host pipeline: refresh the host mirrors and re-plan when the
            # selected backend's cached metadata no longer matches them.
            block_id_host = attn_inputs.kv_cache_kernel_block_id
            if block_id_host is None or block_id_host.numel() == 0:
                block_id_host = attn_inputs.kv_cache_kernel_block_id_device
            self.fmha_params.fill_params(
                _host_i32(attn_inputs.prefix_lengths),
                _host_i32(attn_inputs.sequence_lengths),
                _host_i32(attn_inputs.input_lengths),
                _host_i32(block_id_host),
                self.seq_size_per_block,
                forbid_realloc=True,
            )
            if self._cuda_graph_replay_needs_replan():
                self._plan_decode_wrapper(attn_inputs)
            self._validate_dynamic_fp8_cuda_graph_state()
            return

        # Device-metadata compatibility path inherited from the base
        # implementation. CudaGraphRunner routes graph replay through the
        # pinned host mirrors above.
        if self.dynamic_fp8:
            raise RuntimeError(
                "FP8 KV cache mode 2 CUDA graph replay requires host metadata mirrors"
            )
        seq_plus_1 = attn_inputs.sequence_lengths_plus_1_device
        if seq_plus_1 is None or not seq_plus_1.is_cuda:
            seq_plus_1 = (attn_inputs.sequence_lengths.to(torch.int32) + 1).cuda()
        block_id = _device_or(
            attn_inputs.kv_cache_kernel_block_id_device,
            attn_inputs.kv_cache_kernel_block_id,
        )
        if block_id is not None and not block_id.is_cuda:
            block_id = block_id.cuda()
        self.fmha_params.fill_decode_cuda_graph_params(
            seq_plus_1,
            block_id,
            self.seq_size_per_block,
        )

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def forward(
        self, q: torch.Tensor, kv_cache: Optional[LayerKVCache], params: ParamsBase
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required"
        q = q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk)
        if self.dynamic_fp8:
            kv_scale = _validate_dynamic_fp8_scale(
                kv_cache, self.local_kv_head_num, self.seq_size_per_block
            )
            return self.decode_wrapper.run(q, kv_cache.kv_cache_base, kv_scale)

        q = quantize_to_fp8_if_needed(q, self.q_dtype)
        paged_kv_cache = kv_cache.kv_cache_base
        if paged_kv_cache is not None and paged_kv_cache.dim() == 2:
            paged_kv_cache = common.reshape_paged_kv_cache(
                paged_kv_cache,
                self.local_kv_head_num,
                self.seq_size_per_block,
                self.head_dim_qk,
            )
        return self.decode_wrapper.run(q, paged_kv_cache)


class PyFlashinferDecodeImpl(FMHAImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.dynamic_fp8 = _is_dynamic_fp8(attn_configs)
        self.fmha_impl = PyFlashinferDecodeAttnOp(attn_configs, attn_inputs)
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs

        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.fmha_impl.set_params(self.fmha_params)
        self.fmha_impl.prepare(attn_inputs)

        if self.dynamic_fp8:
            _, self.kernel_page_size, _ = _page_geometry(attn_configs)
            self.dynamic_rope_config = attn_configs.rope_config
            if not self.need_rope_kv_cache:
                self.dynamic_rope_config = RopeConfig()
                self.dynamic_rope_config.style = RopeStyle.No
            self.dynamic_rope_cache = None
            if self.dynamic_rope_config.style in (RopeStyle.Base, RopeStyle.Yarn):
                rope_cache = get_rope_cache_once(
                    self.dynamic_rope_config,
                    attn_configs.max_seq_len,
                    is_cuda=True,
                    interleave=True,
                )
                if check_rope_cache(self.dynamic_rope_config, rope_cache):
                    self.dynamic_rope_cache = rope_cache.data
        else:
            self.rope_impl = FusedRopeKVCacheDecodeOp(attn_configs)
            self.rope_params = self.rope_impl.prepare(attn_inputs)
            self.kv_cache_write_op = None

        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        """Prepare FlashInfer/RoPE buffers and metadata for CUDA graph replay."""
        self.fmha_impl.prepare_for_cuda_graph_replay(attn_inputs)
        if self.dynamic_fp8:
            return
        if self.need_rope_kv_cache:
            # Update rope params for correct position encoding during replay.
            new_rope_params = self.rope_impl.prepare(
                attn_inputs, forbid_reallocation=True
            )
            common.copy_kv_cache_offset(
                self.rope_params.kv_cache_offset, new_rope_params.kv_cache_offset
            )
            self.rope_params.sequence_lengths = new_rope_params.sequence_lengths

    def support_cuda_graph(self) -> bool:
        return True

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return not attn_configs.use_mla

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if self.dynamic_fp8:
            if kv_cache is None:
                raise ValueError("FP8 KV cache mode 2 requires kv_cache")
            kv_scales = _validate_dynamic_fp8_scale(
                kv_cache,
                self.attn_configs.kv_head_num,
                self.kernel_page_size,
            )
            qkv = rtp_llm_ops.fused_rope_quantize_and_write_fp8_kv_cache(
                qkv,
                kv_cache.kv_cache_base,
                kv_scales,
                self.fmha_params.batch_indice_d,
                self.fmha_params.positions_d,
                self.fmha_params.decode_page_indptr_d,
                self.fmha_params.page_indice_d,
                self.attn_configs.head_num,
                self.attn_configs.kv_head_num,
                self.kernel_page_size,
                self.dynamic_rope_config,
                self.dynamic_rope_cache,
            )
        elif self.need_rope_kv_cache:
            qkv = self.rope_impl.forward(qkv, kv_cache, self.rope_params)

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(qkv, kv_cache, self.fmha_params)
