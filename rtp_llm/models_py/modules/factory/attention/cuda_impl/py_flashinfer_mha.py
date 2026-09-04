import math
from typing import Any, Optional

import torch
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
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig, RopeStyle
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    LayerKVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
    rtp_llm_ops,
)

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
_g_py_flashinfer_workspace_pool: list[torch.Tensor] = []
_g_py_flashinfer_pool_lock = __import__("threading").Lock()


def get_py_flashinfer_workspace_buffer(device: str = "cuda") -> torch.Tensor:
    """Get a PyFlashInfer workspace buffer from the pool.

    This function manages workspace buffers to support multiple concurrent instances.
    """
    with _g_py_flashinfer_pool_lock:
        if _g_py_flashinfer_workspace_pool:
            return _g_py_flashinfer_workspace_pool.pop()
    return torch.zeros(
        DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )


def release_py_flashinfer_workspace_buffer(buffer: torch.Tensor) -> None:
    """Release a PyFlashInfer workspace buffer back to the pool."""
    with _g_py_flashinfer_pool_lock:
        _g_py_flashinfer_workspace_pool.append(buffer)


def _host_i32(t):
    """Lift a possibly CUDA-resident tensor to host for the host fill path.

    The C++ fillParams reads raw host pointers; hybrid-attention models
    (qwen3-next) and the MTP device-state fast path can hand in CUDA-resident
    lengths/block tables even when input_lengths stays on the host.
    """
    return t.cpu() if t is not None and t.numel() > 0 and t.is_cuda else t


def _device_or(device_tensor, host_tensor):
    """Prefer the *_device mirror; fall back to the base field.

    Unit tests (and some callers) construct PyAttentionInputs with only the
    base fields populated (possibly already CUDA-resident), so the device
    mirror may be missing.
    """
    if device_tensor is not None and device_tensor.numel() >= 0:
        return device_tensor
    return host_tensor


def attn_kv_dtype(attn_configs: AttentionConfigs) -> torch.dtype:
    # Use one dtype source for both plan() and forward().
    if attn_configs.kv_cache_dtype == KvCacheDataType.FP8:
        return torch.float8_e4m3fn
    return attn_configs.dtype


def attn_q_dtype(attn_configs: AttentionConfigs) -> torch.dtype:
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


def attn_head_dim_vo(attn_configs: AttentionConfigs) -> int:
    """V/output head dimension; equals the QK one unless the model splits them."""
    return attn_configs.v_size_per_head or attn_configs.size_per_head


def window_left_from_sliding_window(sliding_window: int) -> int:
    """Translate a HuggingFace ``sliding_window`` into FlashInfer's ``window_left``.

    HF masks with ``kv_idx > q_idx - sliding_window``, i.e. a query attends to
    ``sliding_window`` keys including itself. FlashInfer masks with
    ``kv_idx + qo_len + window_left >= kv_len + qo_idx``, i.e. ``window_left + 1`` keys
    including itself. Passing ``sliding_window`` straight through widens the window by
    one key.
    """
    if sliding_window <= 0:
        return -1
    return sliding_window - 1


def resolve_paged_backend(backend: str, head_dim_qk: int, head_dim_vo: int) -> str:
    """Pick a paged-KV backend that can handle this head-dim pair.

    FlashInfer's SM90 (fa3) paged prefill only dispatches when
    ``HEAD_DIM_QK == HEAD_DIM_VO``; anything else falls through to
    ``return cudaErrorNotSupported`` (``hopper/prefill_sm90.cuh``,
    ``BatchPrefillWithPagedKVCacheDispatched``). The AOT module for
    ``head_dim_qk_192_head_dim_vo_128 ... sm90`` does get compiled, so the failure only
    surfaces at launch as "BatchPrefillWithPagedKVCacheSM90Run failed with error:
    operation not supported".

    The fa3 *ragged* path has no such restriction, which is why asymmetric head dims
    work there and this only bites once a layer moves to the paged wrapper. fa2's paged
    dispatch carries no head-dim equality check, so pin it.
    """
    if backend == "auto" and head_dim_qk != head_dim_vo:
        return "fa2"
    return backend


def resolve_ragged_backend(backend: str, head_dim_qk: int, head_dim_vo: int) -> str:
    """Keep asymmetric ragged MHA on the same FA2 path as paged/decode.

    MiMo V2.5 uses QK=192 and V=128 for both global and sliding-window layers.
    FlashInfer can dispatch that ragged shape through FA3, but MiMo's validated
    attention path is FA2. Pin only asymmetric MHA here so symmetric models retain
    the existing automatic backend selection.
    """
    if backend == "auto" and head_dim_qk != head_dim_vo:
        return "fa2"
    return backend


def normalize_sink_bias(bias: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Put a per-head attention sink bias into the dtype the correction needs.

    The checkpoint stores the bias in the model dtype (bf16 for MiMo V2.5) and the
    correction below runs in fp32; the upcast is exact, so this only fixes the type.
    """
    if bias is None:
        return None
    if bias.dtype != torch.float32:
        bias = bias.to(torch.float32)
    return bias.contiguous()


# ln 2, converting FlashInfer's base-2 LSE into a natural log.
_LN2 = math.log(2.0)


def apply_attention_sink(
    out: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor
) -> torch.Tensor:
    """Fold a per-head attention sink into an output computed without one.

    FlashInfer's wrappers accept a ``sinks=`` argument but only the ``trtllm-gen``
    branch of ``prefill.py``'s ``paged_run`` shim forwards it to the kernel; the fa2 and
    fa3 branches drop it silently, and their default ``DefaultAttention`` variant
    declares no sink tensor at all. Applying sinks natively would mean the JIT-only
    ``BatchAttentionWithAttentionSinkWrapper``, whose AOT coverage is (64, 64) only.
    Doing it here instead is exact and needs no extra kernel.

    A sink contributes one extra logit ``b_h`` to the softmax denominator and no value,
    so with ``D = sum_j exp(s_j)`` over the visible keys:

        out_sink = (sum_j exp(s_j) v_j) / (D + exp(b_h))
                 = out * D / (D + exp(b_h))
                 = out * sigmoid(log(D) - b_h)

    Both backends report ``lse = log2(D)`` -- fa2 writes ``log2(d) + m`` with ``m``
    already scaled into the log2 domain (``prefill.cuh``), fa3 writes
    ``row_max * sm_scale_log2 + log2(sum)`` (``hopper/attention_updater.cuh``) -- hence
    the ln2 factor.

    Args:
        out: attention output, ``[tokens, num_qo_heads, head_dim_vo]``
        lse: base-2 log-sum-exp, ``[tokens, num_qo_heads]``
        sink: per-head sink bias, ``[num_qo_heads]``, fp32
    """
    scale = torch.sigmoid(lse.float() * _LN2 - sink.view(1, -1))
    return out * scale.unsqueeze(-1).to(out.dtype)


def select_paged_kv_cache(
    kv_cache: LayerKVCache,
    head_dim_qk: int,
    head_dim_vo: int,
    local_kv_head_num: int,
    page_size: int,
) -> Any:
    """Present the layer's cache the way FlashInfer's ``run()`` wants it.

    With asymmetric K/V there is no single tensor holding both, so hand over the
    ``(k_cache, v_cache)`` pair the C++ layer narrowed out of the interleaved block;
    their strides are identical, which is what the paged kernels require. Otherwise
    return the merged 5D view.
    """
    if head_dim_vo != head_dim_qk:
        k_cache = kv_cache.k_cache
        if k_cache is not None and k_cache.numel() > 0:
            return (k_cache, kv_cache.v_cache)
        raise ValueError(
            "asymmetric K/V attention requires the split k_cache/v_cache views; "
            "got an empty k_cache, which means the cache group was built with a "
            "symmetric MHA spec"
        )
    paged_kv_cache = kv_cache.kv_cache_base
    if paged_kv_cache is not None and paged_kv_cache.dim() == 2:
        paged_kv_cache = common.reshape_paged_kv_cache(
            paged_kv_cache, local_kv_head_num, page_size, head_dim_qk
        )
    return paged_kv_cache


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
        self.head_dim_vo = attn_head_dim_vo(attn_configs)
        self.page_size = attn_configs.kernel_tokens_per_block
        self.dtype = attn_configs.dtype
        self.kv_dtype = attn_kv_dtype(attn_configs)
        self.q_dtype = attn_q_dtype(attn_configs)
        self.max_seq_len = attn_configs.max_seq_len
        self.is_causal = attn_configs.is_causal
        self.window_left = window_left_from_sliding_window(attn_configs.sliding_window)
        self.sink_bias: Optional[torch.Tensor] = None
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.enable_cuda_graph = attn_inputs.is_cuda_graph
        self.prefill_cuda_graph_copy_params = None
        # Pre-allocated buffers for CUDA graph copy path (avoid per-forward allocation)
        self._aligned_q_buf = None
        # reserve buffer for q cast
        self._aligned_q_cast_buf = None
        self._compact_out_buf = None
        # Use Paged KV Cache wrapper
        self.backend = resolve_paged_backend(
            backend, self.head_dim_qk, self.head_dim_vo
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
            backend=self.backend,
        )

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_params(self, params: rtp_llm_ops.FlashInferMlaAttnParams):
        """Set the params object to be used by this op."""
        self.fmha_params = params

    def set_sink_bias(self, bias: Optional[torch.Tensor]) -> None:
        """Set this layer's per-head attention sink bias, or None if it has none."""
        self.sink_bias = normalize_sink_bias(bias)

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
        # Windowed groups can carry NULL pages before their active suffix.  Use
        # the SWA-aware device planner for both host and device input mirrors so
        # those pages are redirected to the reserved block before FlashInfer
        # sees them.  The regular host planner intentionally preserves negative
        # page ids for full-attention groups.
        if attn_inputs.input_lengths.is_cuda or self.window_left >= 0:
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
                is_sliding_window=self.window_left >= 0,
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

        self.prefill_wrapper.plan(
            qo_indptr,
            self.fmha_params.decode_page_indptr_d,
            self.fmha_params.page_indice_d,
            self.fmha_params.paged_kv_last_page_len_d,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.page_size,
            causal=self.is_causal,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_dtype,
            o_data_type=self.dtype,
            head_dim_vo=self.head_dim_vo,
            window_left=self.window_left,
        )
        return self.fmha_params

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        return True

    def _run(self, q: torch.Tensor, paged_kv_cache: Any) -> torch.Tensor:
        """Run the wrapper, folding in the attention sink when this layer has one.

        ``sinks=`` is deliberately not passed to run(): fa2/fa3 accept and discard it,
        see apply_attention_sink().
        """
        if self.sink_bias is None:
            return self.prefill_wrapper.run(q, paged_kv_cache)
        out, lse = self.prefill_wrapper.run(q, paged_kv_cache, return_lse=True)
        return apply_attention_sink(out, lse, self.sink_bias)

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

        paged_kv_cache = select_paged_kv_cache(
            kv_cache,
            self.head_dim_qk,
            self.head_dim_vo,
            self.local_kv_head_num,
            self.page_size,
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
            # The output carries head_dim_vo, which may differ from Q's head dim.
            out_head_size = self.head_dim_vo
            out_hidden_size = head_num * out_head_size

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
                out_hidden_size,
            ):
                self._compact_out_buf = torch.zeros(
                    (token_num, out_hidden_size), dtype=q.dtype, device=q.device
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
            result = self._run(q_aligned, paged_kv_cache)

            # Reshape result to 2D for copy back (ensure contiguous)
            result_2d = result.view(total_len, out_hidden_size).contiguous()
            self._compact_out_buf.zero_()

            # Copy large to small (aligned -> compact)
            cuda_graph_copy_large2small(
                result_2d,
                self._compact_out_buf,
                self.prefill_cuda_graph_copy_params.cuda_graph_prefill_batch_size,
                self.prefill_cuda_graph_copy_params.max_batch_size,
                self.prefill_cuda_graph_copy_params.max_seq_len,
                self.input_lengths,
                out_hidden_size,
                self.cu_seq_lens,
            )

            # Reshape back to 3D
            result = self._compact_out_buf.view(token_num, head_num, out_head_size)
        else:
            # No CUDA graph copy, direct execution
            # Paged FP8 defaults to unit scales and the output dtype from plan().
            result = self._run(
                quantize_to_fp8_if_needed(q, self.q_dtype), paged_kv_cache
            )

        return result


class PyFlashinferPrefillAttnOp(object):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        backend: str = "auto",
    ) -> None:
        self.g_workspace_buffer = get_py_flashinfer_workspace_buffer()
        # attn_configs.head_num and kv_head_num are already divided by tp_size in ModelConfig::getAttentionConfigs
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.page_size = attn_configs.kernel_tokens_per_block
        self.head_dim_vo = attn_head_dim_vo(attn_configs)
        self.backend = resolve_ragged_backend(
            backend, self.head_dim_qk, self.head_dim_vo
        )
        self.prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.g_workspace_buffer,
            backend=self.backend,
        )
        self.dtype = attn_configs.dtype
        self.q_dtype = attn_q_dtype(attn_configs)
        self.kv_dtype = attn_kv_dtype(attn_configs)
        self.is_causal = attn_configs.is_causal
        self.window_left = window_left_from_sliding_window(attn_configs.sliding_window)
        self.sink_bias: Optional[torch.Tensor] = None
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_params(self, params: rtp_llm_ops.FlashInferMlaAttnParams):
        """Set the params object to be used by this op."""
        self.fmha_params = params

    def set_sink_bias(self, bias: Optional[torch.Tensor]) -> None:
        """Set this layer's per-head attention sink bias, or None if it has none.

        The ragged wrapper's ``run(..., return_lse=True)`` returns the same
        ``(out, lse)`` pair the paged one does, and apply_attention_sink() needs only
        those two plus the bias -- it does not care whether K/V came from a cache or a
        dense tensor. So sinks work here exactly as on the paged path.
        """
        self.sink_bias = normalize_sink_bias(bias)

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
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_dtype,
            o_data_type=self.dtype,
            window_left=self.window_left,
        )
        return self.fmha_params

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        return (
            attn_inputs.prefix_lengths.numel() <= 0
            or attn_inputs.prefix_lengths.sum().item() == 0
        )

    def _run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.sink_bias is None:
            return self.prefill_wrapper.run(q, k, v)
        out, lse = self.prefill_wrapper.run(q, k, v, return_lse=True)
        return apply_attention_sink(out, lse, self.sink_bias)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        q = quantize_to_fp8_if_needed(q, self.q_dtype)
        k = quantize_to_fp8_if_needed(k, self.kv_dtype)
        v = quantize_to_fp8_if_needed(v, self.kv_dtype)
        if q.dtype == torch.float8_e4m3fn:
            # FlashInfer's FA3 FP8 need scale_q, scale_k, scale_v and an output matching the planned dtype.
            out = torch.empty(
                q.shape[:-1] + v.shape[-1:], dtype=self.dtype, device=q.device
            )
            if self.sink_bias is None:
                return self.prefill_wrapper.run(
                    q,
                    k,
                    v,
                    FP8_UNIT_SCALE,
                    FP8_UNIT_SCALE,
                    FP8_UNIT_SCALE,
                    out=out,
                )
            fp8_out, fp8_lse = self.prefill_wrapper.run(
                q,
                k,
                v,
                FP8_UNIT_SCALE,
                FP8_UNIT_SCALE,
                FP8_UNIT_SCALE,
                out=out,
                return_lse=True,
            )
            return apply_attention_sink(fp8_out, fp8_lse, self.sink_bias)
        return self._run(q, k, v)


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
        self.page_size = attn_configs.kernel_tokens_per_block
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

        self.prefix_paged_wrapper.plan(
            qo_indptr,
            prefix_paged_kv_indptr,
            self.fmha_params.reuse_cache_page_indice_d,
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
        paged_kv_cache = kv_cache.kv_cache_base
        if paged_kv_cache.dim() == 2:
            paged_kv_cache = common.reshape_paged_kv_cache(
                paged_kv_cache, self.local_kv_head_num, self.page_size, self.head_dim_qk
            )

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

        # Paged FP8 defaults to unit scales and the output dtype from plan().
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
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs

        self.fmha_impl = self._create_fmha_impl(attn_configs, attn_inputs)
        self.rope_impl = self._create_rope_impl(attn_configs)
        # Create KV cache write op
        self.kv_cache_write_op = KVCacheWriteOp(
            num_kv_heads=attn_configs.kv_head_num,
            head_size=attn_configs.size_per_head,
            token_per_block=attn_configs.kernel_tokens_per_block,
        )
        self.create_params(attn_inputs)
        self.fmha_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        self.fmha_impl.prepare(attn_inputs, forbid_realloc=True)

    def set_sink_bias(self, bias: Optional[torch.Tensor]) -> None:
        """Set the current layer's per-head attention sink bias, or None if it has none.

        Presence of this method is how a model description probes whether an
        implementation can serve a layer with an attention sink.
        """
        self.fmha_impl.set_sink_bias(bias)

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
        head_dim_vo = attn_head_dim_vo(self.attn_configs)

        q, k, v = torch.split(
            qkv,
            [
                head_dim * num_heads,
                head_dim * num_kv_heads,
                head_dim_vo * num_kv_heads,
            ],
            dim=-1,
        )

        query = q.reshape(q.shape[0], num_heads, head_dim)
        key = k.reshape(k.shape[0], num_kv_heads, head_dim)
        value = v.reshape(v.shape[0], num_kv_heads, head_dim_vo)

        return query, key, value

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Common forward implementation for all prefill implementations."""
        if self.need_rope_kv_cache and self.rope_impl is not None:
            query, key, value = self.rope_impl.forward(qkv)
        else:
            query, key, value = self._split_qkv(qkv)

        # Cast K/V once so the KV cache write and the attention op share
        # the same tensors.
        kv_dtype = attn_kv_dtype(self.attn_configs)
        key = quantize_to_fp8_if_needed(key, kv_dtype)
        value = quantize_to_fp8_if_needed(value, kv_dtype)

        if self.need_rope_kv_cache:
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

        Windowed groups use the SWA-aware planner, which substitutes the reserved
        page for NULL block-table entries.  FlashInfer's ``window_left`` then masks
        those pages because they are outside the active window.  A fresh request
        still normally selects the ragged implementation first; this paged path is
        needed when a non-empty prefix makes ragged attention inapplicable.
        """
        return (
            not is_sm10x()
            and PyFlashinferPrefillPagedAttnOp.support(attn_inputs)
            and attn_configs.rope_config.style != RopeStyle.Mrope
        )

    def support_cuda_graph(self) -> bool:
        return True


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
        # Single-stream flow: RoPE -> ragged attention -> KV write -> paged attention.
        # Hybrid always needs the new K/V for its ragged half.
        if self.need_rope_kv_cache and self.rope_impl is not None:
            query, key, value = self.rope_impl.forward(qkv)
        else:
            query, key, value = self._split_qkv(qkv)

        query = quantize_to_fp8_if_needed(query, attn_q_dtype(self.attn_configs))
        kv_dtype = attn_kv_dtype(self.attn_configs)
        key = quantize_to_fp8_if_needed(key, kv_dtype)
        value = quantize_to_fp8_if_needed(value, kv_dtype)

        # Write new K/V after ragged attention and before paged attention.
        kv_cache_write_op = self.kv_cache_write_op if self.need_rope_kv_cache else None
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
        """Check if hybrid prefill implementation is supported.

        Excluded for asymmetric K/V and for windowed layers. Its paged half plans over
        the whole prefix and reads it from the cache, which is the same reason the paged
        variant declines a windowed layer: a windowed group only ever holds a window's
        worth of KV, so the prefix pages outside it were never written.
        """
        return (
            not attn_inputs.is_cuda_graph
            and not is_sm10x()
            and attn_configs.sliding_window <= 0
            and attn_head_dim_vo(attn_configs) == attn_configs.size_per_head
            and PyFlashinferHybridPrefillAttnOp.support(attn_inputs)
            and attn_configs.rope_config.style != RopeStyle.Mrope
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

        Windowed and sink layers land here rather than on the paged variant. Ragged
        reads K/V from the QKV tensor the layer just produced, so their prefill never
        touches the KV cache at all -- which is what lets a windowed group hold only a
        window's worth per request. ``window_left`` comes from plan() and the sink is
        folded in from the LSE, both of which the ragged wrapper supports.
        """
        return (
            PyFlashinferPrefillAttnOp.support(attn_inputs)
            and attn_configs.rope_config.style != RopeStyle.Mrope
        )


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
        self.head_dim_vo = attn_head_dim_vo(attn_configs)
        self.seq_size_per_block = attn_configs.kernel_tokens_per_block
        self.window_left = window_left_from_sliding_window(attn_configs.sliding_window)
        self.is_sliding_window = attn_configs.sliding_window > 0
        self.sink_bias: Optional[torch.Tensor] = None
        # BatchDecodeWithPagedKVCacheWrapper.plan() carries a single head_dim and no
        # head_dim_vo, so an asymmetric layer decodes through the paged *prefill*
        # wrapper instead -- decode is prefill with one query token per request. fa2,
        # not "auto": see resolve_paged_backend().
        self.use_prefill_for_decode = self.head_dim_qk != self.head_dim_vo
        if self.use_prefill_for_decode:
            self.use_tensor_core = True
            self.decode_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.g_workspace_buffer,
                "HND",
                backend="fa2",
            )
        else:
            self.use_tensor_core = determine_use_tensor_core_from_configs(attn_configs)
            self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.g_workspace_buffer,
                "HND",
                use_tensor_cores=self.use_tensor_core,
            )
        self.dtype = attn_configs.dtype
        self.kv_dtype = attn_kv_dtype(attn_configs)
        # CUDA-core decode dequantizes FP8 KV; tensor-core decode uses the
        # batch-prefill path and therefore shares attn_q_dtype().
        self.q_dtype = (
            attn_q_dtype(attn_configs) if self.use_tensor_core else self.dtype
        )
        self.enable_cuda_graph = attn_inputs.is_cuda_graph
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_params(self, params: rtp_llm_ops.FlashInferMlaAttnParams) -> None:
        """Set the params object to be used by this op."""
        self.fmha_params = params

    def set_sink_bias(self, bias: Optional[torch.Tensor]) -> None:
        """Set this layer's per-head attention sink bias, or None if it has none."""
        self.sink_bias = normalize_sink_bias(bias)

    def _requires_tensor_core_cuda_graph_replan(self) -> bool:
        # Tensor-core decode refreshes replay-time plan metadata. The paged-prefill
        # substitute plans from the device buffers instead, like CUDA-core decode.
        return self.use_tensor_core and not self.use_prefill_for_decode

    def _plan_decode_wrapper(self, attn_inputs: PyAttentionInputs) -> None:
        if self._requires_tensor_core_cuda_graph_replan():
            page_indptr = self.fmha_params.decode_page_indptr_h
            page_indice = self.fmha_params.page_indice_h
            last_page_len = self.fmha_params.paged_kv_last_page_len_h
            plan_kwargs = {"non_blocking": True}
        else:
            page_indptr = self.fmha_params.decode_page_indptr_d
            page_indice = self.fmha_params.page_indice_d
            last_page_len = self.fmha_params.paged_kv_last_page_len_d
            plan_kwargs = {}

        if self.use_prefill_for_decode:
            batch_size = attn_inputs.input_lengths.size(0)
            # One query token per request, so qo_indptr is just [0, 1, 2, ...].
            qo_indptr = torch.arange(
                batch_size + 1,
                dtype=torch.int32,
                device=self.g_workspace_buffer.device,
            )
            self.decode_wrapper.plan(
                qo_indptr,
                page_indptr,
                page_indice,
                last_page_len,
                self.local_head_num,
                self.local_kv_head_num,
                self.head_dim_qk,
                self.seq_size_per_block,
                # A single query token has nothing ahead of it to mask.
                causal=False,
                q_data_type=self.q_dtype,
                kv_data_type=self.kv_dtype,
                o_data_type=self.dtype,
                head_dim_vo=self.head_dim_vo,
                window_left=self.window_left,
                **plan_kwargs,
            )
            return

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
            window_left=self.window_left,
            **plan_kwargs,
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
        # Tensor-core decode plans from the HOST mirrors
        # (decode_page_indptr_h etc. in _plan_decode_wrapper); the device fill
        # only populates the device buffers and leaves the host mirrors at
        # their stale capacity sizes (MIN_CACHE_BATCH_SIZE), which corrupts
        # plan's batch size. Route tensor-core through the host fill.
        if self.use_prefill_for_decode or (
            attn_inputs.input_lengths.is_cuda and not self.use_tensor_core
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
                is_sliding_window=self.is_sliding_window,
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
        return self.fmha_params

    def prepare_for_cuda_graph_replay(self, attn_inputs: PyAttentionInputs) -> None:
        """Refresh FlashInfer runtime buffers before replaying the captured graph."""
        if not attn_inputs.sequence_lengths.is_cuda:
            # Host pipeline: refresh host metadata. Tensor-core decode must
            # re-plan from these mirrors because its plan uses host metadata.
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
            if self._requires_tensor_core_cuda_graph_replan():
                self._plan_decode_wrapper(attn_inputs)
            return

        # Device pipeline: update the device-resident buffers in place.
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
        q = quantize_to_fp8_if_needed(
            q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk),
            self.q_dtype,
        )
        paged_kv_cache = select_paged_kv_cache(
            kv_cache,
            self.head_dim_qk,
            self.head_dim_vo,
            self.local_kv_head_num,
            self.seq_size_per_block,
        )
        # Decode FP8 defaults to unit scales and the output dtype from plan().
        # ``sinks=`` is deliberately not passed to run(): fa2/fa3 accept and discard it,
        # so the sink is folded in from the LSE. See apply_attention_sink().
        if self.sink_bias is None:
            return self.decode_wrapper.run(q, paged_kv_cache)
        out, lse = self.decode_wrapper.run(q, paged_kv_cache, return_lse=True)
        return apply_attention_sink(out, lse, self.sink_bias)


class PyFlashinferDecodeImpl(FMHAImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = PyFlashinferDecodeAttnOp(attn_configs, attn_inputs)
        self.rope_impl = FusedRopeKVCacheDecodeOp(attn_configs)
        self.attn_configs = attn_configs
        # The fused C++ RoPE+KV-write kernel writes K and V through one head dim, so an
        # asymmetric layer splits the two steps: Python RoPE, then an explicit KV write.
        # Same arrangement the prefill path already uses.
        self.asymmetric_kv = (
            attn_head_dim_vo(attn_configs) != attn_configs.size_per_head
        )
        if self.asymmetric_kv:
            self.py_rope_impl = MhaRotaryEmbeddingOp(attn_configs)
            self.kv_cache_write_op = KVCacheWriteOp(
                num_kv_heads=attn_configs.kv_head_num,
                head_size=attn_configs.size_per_head,
                token_per_block=attn_configs.kernel_tokens_per_block,
            )

        # Store input info
        self.attn_inputs = attn_inputs

        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.fmha_impl.set_params(self.fmha_params)
        self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)
        if self.asymmetric_kv:
            # fmha_params carries positions_d / batch_indice_d and the page metadata
            # both of these need.
            self.py_rope_impl.set_params(self.fmha_params)
            self.kv_cache_write_op.set_params(self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        """Prepare FlashInfer/RoPE buffers and metadata for CUDA graph replay."""
        self.fmha_impl.prepare_for_cuda_graph_replay(attn_inputs)
        # Update rope params for correct position encoding during cuda graph replay
        new_rope_params = self.rope_impl.prepare(attn_inputs)
        common.copy_kv_cache_offset(
            self.rope_params.kv_cache_offset, new_rope_params.kv_cache_offset
        )

    def support_cuda_graph(self) -> bool:
        # The asymmetric path runs RoPE and the KV write from Python, which the capture
        # does not cover.
        return not self.asymmetric_kv

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return not attn_configs.use_mla

    def set_sink_bias(self, bias: Optional[torch.Tensor]) -> None:
        """Set the current layer's per-head attention sink bias, or None if it has none."""
        self.fmha_impl.set_sink_bias(bias)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            if self.asymmetric_kv:
                query, key, value = self.py_rope_impl.forward(qkv)
                self.kv_cache_write_op.forward(key, value, kv_cache)
                # Only Q goes on to attention; K/V now live in the cache.
                qkv = query
            else:
                qkv = self.rope_impl.forward(qkv, kv_cache, self.rope_params)

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(qkv, kv_cache, self.fmha_params)
