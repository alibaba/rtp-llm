import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional

import torch
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
    FMHAType,
    KvCacheDataType,
    ParallelismConfig,
    RopeStyle,
)
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOpQOut,
    LayerKVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
    get_scalar_type,
    rtp_llm_ops,
)

# Constants
DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB = 128
DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES = (
    DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB * 1024 * 1024
)
MAX_PY_FLASHINFER_WORKSPACE_SIZE_BYTES = 2 * 1024 * 1024 * 1024
MAX_PY_FLASHINFER_POOL_BUFFERS_PER_DEVICE = 4
MAX_PY_FLASHINFER_WORKSPACE_RETRIES = 8
logger = logging.getLogger(__name__)

# Global workspace buffer pool
_g_py_flashinfer_workspace_pool: list[torch.Tensor] = []
_g_py_flashinfer_cuda_graph_workspace_buffers: dict[
    tuple[str, Optional[int]], torch.Tensor
] = {}
_g_py_flashinfer_pool_lock = __import__("threading").Lock()
_MISSING = object()


def _round_up_power_of_2(value: int) -> int:
    return 1 << (max(value, 1) - 1).bit_length()


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _get_flashinfer_version() -> str:
    for package_name in ("flashinfer-python", "flashinfer_python", "flashinfer"):
        try:
            return version(package_name)
        except PackageNotFoundError:
            continue
    return "unknown"


def _flashinfer_private_attr_error(wrapper: Any, attr_name: str) -> RuntimeError:
    return RuntimeError(
        "Unsupported FlashInfer wrapper layout: "
        f"version={_get_flashinfer_version()}, wrapper={type(wrapper).__name__}, "
        f"missing private attribute {attr_name}. "
        "Update the PyFlashInfer compatibility layer before enabling this backend."
    )


def _get_flashinfer_private_attr(
    wrapper: Any,
    attr_name: str,
    default: Any = _MISSING,
) -> Any:
    if hasattr(wrapper, attr_name):
        return getattr(wrapper, attr_name)
    if default is not _MISSING:
        return default
    raise _flashinfer_private_attr_error(wrapper, attr_name)


def _set_flashinfer_private_attr(wrapper: Any, attr_name: str, value: Any) -> None:
    if not hasattr(wrapper, attr_name):
        raise _flashinfer_private_attr_error(wrapper, attr_name)
    setattr(wrapper, attr_name, value)


def _get_flashinfer_method(wrapper: Any, method_name: str) -> Any:
    method = getattr(wrapper, method_name, None)
    if callable(method):
        return method
    raise RuntimeError(
        "Unsupported FlashInfer wrapper layout: "
        f"version={_get_flashinfer_version()}, wrapper={type(wrapper).__name__}, "
        f"missing method {method_name}."
    )


def _validate_py_flashinfer_prefill_wrapper(wrapper: Any) -> None:
    required_attrs = (
        "_backend",
        "_fixed_batch_size",
        "_int_workspace_buffer",
        "_paged_kv_indices_buf",
        "_paged_kv_indptr_buf",
        "_paged_kv_last_page_len_buf",
        "_plan_info",
        "_qo_indptr_buf",
        "_use_cuda_graph",
    )
    for attr_name in required_attrs:
        _get_flashinfer_private_attr(wrapper, attr_name)
    _get_flashinfer_method(wrapper, "reset_workspace_buffer")


def _set_prefill_wrapper_cuda_graph_buffers(
    wrapper: Any,
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    fixed_batch_size: int,
) -> None:
    _set_flashinfer_private_attr(wrapper, "_use_cuda_graph", True)
    _set_flashinfer_private_attr(wrapper, "_qo_indptr_buf", qo_indptr)
    _set_flashinfer_private_attr(wrapper, "_paged_kv_indptr_buf", paged_kv_indptr)
    _set_flashinfer_private_attr(wrapper, "_paged_kv_indices_buf", paged_kv_indices)
    _set_flashinfer_private_attr(
        wrapper, "_paged_kv_last_page_len_buf", paged_kv_last_page_len
    )
    _set_flashinfer_private_attr(wrapper, "_fixed_batch_size", fixed_batch_size)


def _normalize_device(device: str | torch.device) -> torch.device:
    requested_device = torch.device(device)
    if (
        requested_device.type == "cuda"
        and requested_device.index is None
        and torch.cuda.is_available()
    ):
        return torch.device("cuda", torch.cuda.current_device())
    return requested_device


def _device_key(device: str | torch.device) -> tuple[str, Optional[int]]:
    normalized_device = _normalize_device(device)
    return normalized_device.type, normalized_device.index


def _device_matches(buffer: torch.Tensor, device: str | torch.device) -> bool:
    requested_device = _normalize_device(device)
    return (
        buffer.device.type == requested_device.type
        and buffer.device.index == requested_device.index
    )


def _workspace_allocation_size(min_size_bytes: int) -> int:
    workspace_size_bytes = _round_up_power_of_2(
        max(DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES, min_size_bytes)
    )
    if workspace_size_bytes > MAX_PY_FLASHINFER_WORKSPACE_SIZE_BYTES:
        raise RuntimeError(
            "PyFlashInfer workspace request exceeds the configured safety limit: "
            f"requested={min_size_bytes}, rounded={workspace_size_bytes}, "
            f"limit={MAX_PY_FLASHINFER_WORKSPACE_SIZE_BYTES}"
        )
    return workspace_size_bytes


def get_py_flashinfer_workspace_buffer(
    device: str | torch.device = "cuda",
    min_size_bytes: int = DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES,
) -> torch.Tensor:
    """Get a PyFlashInfer workspace buffer from the pool.

    This function manages workspace buffers to support multiple concurrent instances.
    """
    workspace_size_bytes = _workspace_allocation_size(min_size_bytes)
    with _g_py_flashinfer_pool_lock:
        best_idx = -1
        best_size = 0
        for idx, buffer in enumerate(_g_py_flashinfer_workspace_pool):
            buffer_size = buffer.numel() * buffer.element_size()
            if (
                _device_matches(buffer, device)
                and buffer_size >= workspace_size_bytes
                and (best_idx < 0 or buffer_size < best_size)
            ):
                best_idx = idx
                best_size = buffer_size
        if best_idx >= 0:
            return _g_py_flashinfer_workspace_pool.pop(best_idx)
    return torch.zeros(
        workspace_size_bytes,
        dtype=torch.uint8,
        device=device,
    )


def get_py_flashinfer_cuda_graph_workspace_buffer(
    device: str | torch.device = "cuda",
    min_size_bytes: int = DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES,
) -> torch.Tensor:
    """Return the workspace shared by paged prefill graphs on one device.

    Same-device graph replays must be serialized because their wrappers share
    this buffer. Growth replaces the registry entry for future wrappers;
    existing wrappers retain their old tensor and remain valid.
    """
    normalized_device = _normalize_device(device)
    workspace_size_bytes = _workspace_allocation_size(min_size_bytes)
    key = _device_key(normalized_device)
    with _g_py_flashinfer_pool_lock:
        buffer = _g_py_flashinfer_cuda_graph_workspace_buffers.get(key)
        if (
            buffer is not None
            and buffer.numel() * buffer.element_size() >= workspace_size_bytes
        ):
            return buffer

        buffer = torch.zeros(
            workspace_size_bytes,
            dtype=torch.uint8,
            device=normalized_device,
        )
        _g_py_flashinfer_cuda_graph_workspace_buffers[key] = buffer
        return buffer


def release_py_flashinfer_workspace_buffer(buffer: torch.Tensor) -> None:
    """Release a PyFlashInfer workspace buffer back to the pool."""
    with _g_py_flashinfer_pool_lock:
        _g_py_flashinfer_workspace_pool.append(buffer)
        matching_indices = [
            idx
            for idx, pooled_buffer in enumerate(_g_py_flashinfer_workspace_pool)
            if pooled_buffer.device == buffer.device
        ]
        if len(matching_indices) > MAX_PY_FLASHINFER_POOL_BUFFERS_PER_DEVICE:
            largest_idx = max(
                matching_indices,
                key=lambda idx: _g_py_flashinfer_workspace_pool[idx].numel()
                * _g_py_flashinfer_workspace_pool[idx].element_size(),
            )
            del _g_py_flashinfer_workspace_pool[largest_idx]


def _resolve_py_flashinfer_prefill_backend(
    configured_backend: str,
    current_backend: str,
    device: torch.device,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
) -> str:
    supported_backends = ("fa2", "fa3")
    if current_backend in ("fa2", "fa3"):
        return current_backend
    if configured_backend in supported_backends:
        return configured_backend
    if configured_backend != "auto":
        raise ValueError(
            "Unsupported PyFlashInfer prefill backend "
            f"{configured_backend!r}; expected one of auto, fa2, fa3"
        )
    try:
        from flashinfer.utils import PosEncodingMode, determine_attention_backend

        resolved_backend = determine_attention_backend(
            device,
            PosEncodingMode["NONE"].value,
            False,
            False,
            q_data_type,
            kv_data_type,
        )
        if resolved_backend not in supported_backends:
            raise ValueError(
                f"FlashInfer selected unsupported prefill backend {resolved_backend!r}"
            )
        return resolved_backend
    except (ImportError, AttributeError, RuntimeError, ValueError) as error:
        logger.warning(
            "Unable to resolve the PyFlashInfer prefill backend; falling back to "
            "fa2 (device=%s, q_dtype=%s, kv_dtype=%s): %s",
            device,
            q_data_type,
            kv_data_type,
            error,
        )
        return "fa2"


def _get_py_flashinfer_prefill_plan_workspace_size_bytes(
    plan_info: Any,
    num_qo_heads: int,
    head_dim_vo: int,
) -> int:
    def warn_and_use_current_workspace(reason: str) -> int:
        logger.warning(
            "Unable to inspect FlashInfer fa2 prefill plan_info; retaining the "
            "current workspace as the conservative CUDA Graph bound: %s",
            reason,
        )
        return 0

    if plan_info is None:
        return warn_and_use_current_workspace("plan_info is None")
    if torch.is_tensor(plan_info):
        plan_info_host = plan_info.detach().to("cpu").tolist()
    else:
        try:
            plan_info_host = list(plan_info)
        except TypeError:
            return warn_and_use_current_workspace(
                f"unexpected type {type(plan_info).__name__}"
            )
    if len(plan_info_host) != 15:
        return warn_and_use_current_workspace(
            f"unexpected size (expected=15, actual={len(plan_info_host)})"
        )

    padded_batch_size = int(plan_info_host[0])
    cta_tile_q = int(plan_info_host[3])
    v_offset = int(plan_info_host[10])
    s_offset = int(plan_info_host[11])
    split_kv = bool(plan_info_host[14])
    if not split_kv:
        return 0
    if padded_batch_size < 0 or cta_tile_q <= 0:
        return warn_and_use_current_workspace(
            "invalid values: "
            f"padded_batch_size={padded_batch_size}, cta_tile_q={cta_tile_q}"
        )

    tmp_v_bytes = _align_up(
        num_qo_heads * padded_batch_size * cta_tile_q * head_dim_vo * 4, 16
    )
    tmp_s_bytes = _align_up(num_qo_heads * padded_batch_size * cta_tile_q * 4, 16)
    return max(v_offset + tmp_v_bytes, s_offset + tmp_s_bytes)


def _is_flashinfer_workspace_plan_error(error: Exception) -> bool:
    message = str(error)
    return any(
        pattern in message
        for pattern in (
            "batch_prefill_tmp_s",
            "batch_prefill_tmp_v",
            "float workspace",
            "float_workspace",
        )
    )


class PyFlashinferPrefillPagedAttnOp(object):
    """FlashInfer Prefill Attention Op with Paged KV Cache support"""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        backend: str = "auto",
    ) -> None:
        self.enable_cuda_graph = attn_inputs.is_cuda_graph
        self.g_workspace_buffer = (
            get_py_flashinfer_cuda_graph_workspace_buffer()
            if self.enable_cuda_graph
            else get_py_flashinfer_workspace_buffer()
        )
        self._owns_workspace_buffer = not self.enable_cuda_graph
        self._cuda_graph_workspace_size_upper_bound_bytes = 0
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.size_per_head
        self.page_size = attn_configs.kernel_tokens_per_block
        self.datatype = attn_configs.dtype
        self.kv_cache_dtype = attn_configs.kv_cache_dtype
        if self.kv_cache_dtype == KvCacheDataType.FP8:
            self.kv_datatype = torch.float8_e4m3fn
        else:
            self.kv_datatype = self.datatype
        self.backend = _resolve_py_flashinfer_prefill_backend(
            backend,
            backend,
            self.g_workspace_buffer.device,
            self.datatype,
            self.kv_datatype,
        )
        logger.info(
            "Using PyFlashInfer paged prefill backend=%s (configured=%s, "
            "target_verify=%s, cuda_graph=%s)",
            self.backend,
            backend,
            attn_inputs.is_target_verify,
            self.enable_cuda_graph,
        )
        self.max_seq_len = attn_configs.max_seq_len
        self._plan_shape = (0, 0, 0)
        self.is_causal = attn_configs.is_causal
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.prefill_cuda_graph_copy_params = None
        # Pre-allocated buffers for CUDA graph copy path (avoid per-forward allocation)
        self._aligned_q_buf = None
        self._compact_out_buf = None
        # Use Paged KV Cache wrapper
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
            backend=self.backend,
        )
        _validate_py_flashinfer_prefill_wrapper(self.prefill_wrapper)

    def __del__(self):
        workspace_buffer = getattr(self, "g_workspace_buffer", None)
        if workspace_buffer is not None and getattr(
            self, "_owns_workspace_buffer", True
        ):
            release_py_flashinfer_workspace_buffer(workspace_buffer)

    def set_params(self, params: rtp_llm_ops.FlashInferMlaAttnParams):
        """Set the params object to be used by this op."""
        self.fmha_params = params

    def _workspace_size_bytes(self) -> int:
        return self.g_workspace_buffer.numel() * self.g_workspace_buffer.element_size()

    def _resize_workspace_buffer(self, required_bytes: int) -> None:
        old_workspace_buffer = self.g_workspace_buffer
        old_workspace_bytes = self._workspace_size_bytes()
        if required_bytes <= old_workspace_bytes:
            return
        self.g_workspace_buffer = (
            get_py_flashinfer_cuda_graph_workspace_buffer(
                old_workspace_buffer.device,
                required_bytes,
            )
            if self.enable_cuda_graph
            else get_py_flashinfer_workspace_buffer(
                old_workspace_buffer.device,
                required_bytes,
            )
        )
        batch_size, max_q_len, max_kv_len = self._plan_shape
        logger.info(
            "Resized PyFlashInfer paged prefill workspace: old=%d, "
            "requested=%d, allocated=%d, cuda_graph=%s, batch=%d, "
            "max_q_len=%d, max_kv_len=%d",
            old_workspace_bytes,
            required_bytes,
            self._workspace_size_bytes(),
            self.enable_cuda_graph,
            batch_size,
            max_q_len,
            max_kv_len,
        )
        reset_workspace_buffer = _get_flashinfer_method(
            self.prefill_wrapper, "reset_workspace_buffer"
        )
        reset_workspace_buffer(
            self.g_workspace_buffer,
            _get_flashinfer_private_attr(self.prefill_wrapper, "_int_workspace_buffer"),
        )
        if self._owns_workspace_buffer:
            release_py_flashinfer_workspace_buffer(old_workspace_buffer)

    def _check_cuda_graph_replay_workspace_size(self, forbid_realloc: bool) -> None:
        current_bytes = self._workspace_size_bytes()
        if not (forbid_realloc and self.enable_cuda_graph):
            return

        required_bytes = self._cuda_graph_workspace_size_upper_bound_bytes
        if required_bytes <= 0:
            raise RuntimeError(
                "PyFlashInfer CUDA graph workspace upper bound is not initialized"
            )
        if current_bytes < required_bytes:
            raise RuntimeError(
                "PyFlashInfer workspace is too small during CUDA graph replay: "
                f"current={current_bytes}, required={required_bytes}"
            )

    def _record_workspace_size_after_plan(self, forbid_realloc: bool) -> bool:
        if forbid_realloc and self.enable_cuda_graph:
            return False

        backend = _resolve_py_flashinfer_prefill_backend(
            self.backend,
            _get_flashinfer_private_attr(self.prefill_wrapper, "_backend"),
            self.g_workspace_buffer.device,
            self.datatype,
            self.kv_datatype,
        )
        current_bytes = self._workspace_size_bytes()
        if backend != "fa2":
            if self.enable_cuda_graph:
                self._cuda_graph_workspace_size_upper_bound_bytes = max(
                    self._cuda_graph_workspace_size_upper_bound_bytes,
                    current_bytes,
                )
            return False

        plan_info = _get_flashinfer_private_attr(self.prefill_wrapper, "_plan_info")
        actual_bytes = _get_py_flashinfer_prefill_plan_workspace_size_bytes(
            plan_info,
            self.local_head_num,
            self.head_dim_vo,
        )
        if actual_bytes > current_bytes:
            if forbid_realloc:
                raise RuntimeError(
                    "PyFlashInfer fa2 prefill plan exceeds workspace during replay: "
                    f"actual={actual_bytes}, current={current_bytes}"
                )
            self._resize_workspace_buffer(actual_bytes)
            return True

        if self.enable_cuda_graph:
            self._cuda_graph_workspace_size_upper_bound_bytes = max(
                self._cuda_graph_workspace_size_upper_bound_bytes,
                current_bytes,
            )
        return False

    def _plan_prefill_with_workspace_retry(
        self,
        forbid_realloc: bool,
        *args,
        **kwargs,
    ) -> None:
        last_workspace_error: Optional[Exception] = None
        for _ in range(MAX_PY_FLASHINFER_WORKSPACE_RETRIES):
            try:
                self.prefill_wrapper.plan(*args, **kwargs)
            except Exception as error:
                if forbid_realloc or not _is_flashinfer_workspace_plan_error(error):
                    raise
                last_workspace_error = error
                current_bytes = self._workspace_size_bytes()
                try:
                    next_bytes = _workspace_allocation_size(current_bytes * 2)
                except RuntimeError as limit_error:
                    batch_size, max_q_len, max_kv_len = self._plan_shape
                    raise RuntimeError(
                        "PyFlashInfer prefill workspace reached its safety limit: "
                        f"batch={batch_size}, max_q_len={max_q_len}, "
                        f"max_kv_len={max_kv_len}, current={current_bytes}, "
                        f"limit={MAX_PY_FLASHINFER_WORKSPACE_SIZE_BYTES}"
                    ) from limit_error
                logger.warning(
                    "Retrying PyFlashInfer prefill plan after workspace exhaustion: "
                    "current=%d, next=%d, error=%s",
                    current_bytes,
                    next_bytes,
                    error,
                )
                self._resize_workspace_buffer(next_bytes)
                continue

            if not self._record_workspace_size_after_plan(forbid_realloc):
                return

        raise RuntimeError(
            "PyFlashInfer prefill plan did not stabilize after workspace retries: "
            f"retries={MAX_PY_FLASHINFER_WORKSPACE_RETRIES}, "
            f"current={self._workspace_size_bytes()}, "
            f"limit={MAX_PY_FLASHINFER_WORKSPACE_SIZE_BYTES}"
        ) from last_workspace_error

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
        input_lengths = attn_inputs.input_lengths
        batch_size = input_lengths.size(0)
        max_q_len = int(input_lengths.max().item()) if batch_size else 0
        if attn_inputs.prefix_lengths.numel() == batch_size:
            max_kv_len = int(
                (attn_inputs.prefix_lengths + input_lengths).max().item()
            )
        elif attn_inputs.sequence_lengths.numel() > 0:
            max_kv_len = int(attn_inputs.sequence_lengths.max().item())
        else:
            max_kv_len = max_q_len
        self._plan_shape = (batch_size, max_q_len, max_kv_len)
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id,
            self.page_size,
            forbid_realloc,
        )
        # Store CUDA graph copy parameters
        # Define qo_indptr early for CUDA graph initialization
        if attn_inputs.prefill_cuda_graph_copy_params is not None:
            # FlashInfer and the compact/aligned copy kernels require stable
            # device pointers. Replay metadata may arrive in pinned host memory,
            # so copy it into fixed CUDA buffers allocated before graph capture.
            self.input_lengths = attn_inputs.input_lengths
            self.cu_seq_lens = torch.empty(
                attn_inputs.cu_seqlens_device.shape,
                dtype=attn_inputs.cu_seqlens_device.dtype,
                device=self.g_workspace_buffer.device,
            )
            self.cu_seq_lens.copy_(
                attn_inputs.cu_seqlens_device,
                non_blocking=attn_inputs.cu_seqlens_device.is_pinned(),
            )
            qo_indptr = torch.empty_like(self.cu_seq_lens)
        else:
            qo_indptr = attn_inputs.cu_seqlens_device[
                : attn_inputs.input_lengths.size(0) + 1
            ]

        if (
            self.enable_cuda_graph
            and _get_flashinfer_private_attr(self.prefill_wrapper, "_qo_indptr_buf")
            is None
        ):
            _set_prefill_wrapper_cuda_graph_buffers(
                self.prefill_wrapper,
                qo_indptr,
                self.fmha_params.decode_page_indptr_d,
                self.fmha_params.page_indice_d,
                self.fmha_params.paged_kv_last_page_len_d,
                len(attn_inputs.cu_seqlens_device) - 1,
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

        self._check_cuda_graph_replay_workspace_size(forbid_realloc)
        self._plan_prefill_with_workspace_retry(
            forbid_realloc,
            qo_indptr,
            self.fmha_params.decode_page_indptr_d,
            self.fmha_params.page_indice_d,
            self.fmha_params.paged_kv_last_page_len_d,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.page_size,
            causal=self.is_causal,
            q_data_type=self.datatype,
            kv_data_type=self.kv_datatype,
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

        paged_kv_cache = kv_cache.kv_cache_base
        if paged_kv_cache.dim() == 2:
            paged_kv_cache = common.reshape_paged_kv_cache(
                paged_kv_cache, self.local_kv_head_num, self.page_size, self.head_dim_qk
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
            # No CUDA graph copy, direct execution
            result = self.prefill_wrapper.run(q, paged_kv_cache)

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
        # TODO: maybe use v_head_dim
        self.head_dim_vo = attn_configs.size_per_head
        self.prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.g_workspace_buffer,
            backend=backend,
        )
        self.datatype = attn_configs.dtype
        self.is_causal = attn_configs.is_causal
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
        kv_block_id_host = attn_inputs.kv_cache_kernel_block_id
        if kv_block_id_host is None:
            kv_block_id_host = torch.empty(0, dtype=torch.int32)

        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            kv_block_id_host,
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
            q_data_type=get_scalar_type(attn_inputs.dtype),
        )
        return self.fmha_params

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        return (
            attn_inputs.prefix_lengths.numel() <= 0
            or attn_inputs.prefix_lengths.sum().item() == 0
        )

    ## 1. pure prefill attn: qkv contains q and k,v
    ## 2. paged attn: qkv is only q, and kv is in kv_cache
    def forward(
        self, qkv: torch.Tensor, kv_cache: Optional[LayerKVCache]
    ) -> torch.Tensor:
        qkv = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(
            qkv,
            [
                self.head_dim_qk * self.local_head_num,
                self.head_dim_qk * self.local_kv_head_num,
                self.head_dim_vo * self.local_kv_head_num,
            ],
            dim=-1,
        )
        q = q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk)
        k = k.reshape(k.shape[0], self.local_kv_head_num, self.head_dim_qk)
        v = v.reshape(v.shape[0], self.local_kv_head_num, self.head_dim_vo)
        return self.prefill_wrapper.run(q, k, v)


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

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Common forward implementation for all prefill implementations."""
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            if self.rope_impl is not None:
                # Apply RoPE and get Q, K, V
                query, key, value = self.rope_impl.forward(qkv)
            else:
                # No RoPE, just split QKV
                query, key, value = self._split_qkv(qkv)

            # Write KV to cache
            self.kv_cache_write_op.forward(key, value, kv_cache)

            # Pass query to FMHA (for paged) or reconstruct qkv (for ragged)
            qkv = self._prepare_fmha_input(query, key, value)

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(qkv, kv_cache)

    def _prepare_fmha_input(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Prepare input for FMHA. To be overridden by subclasses if needed."""
        # Default: just return query (for paged layout)
        return query


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
    ) -> torch.Tensor:
        """For paged layout, only return query (KV is already in cache)."""
        return query

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
            and attn_configs.rope_config.style != RopeStyle.Mrope
        )

    def support_cuda_graph(self) -> bool:
        return True


def _supports_py_flashinfer_fa2_target_verify(
    attn_configs: AttentionConfigs,
    attn_inputs: PyAttentionInputs,
) -> bool:
    page_size = attn_configs.kernel_tokens_per_block
    return (
        is_sm90()
        and attn_inputs.is_prefill
        and attn_inputs.is_target_verify
        and attn_configs.need_rope_kv_cache
        and attn_configs.dtype in {torch.float16, torch.bfloat16}
        and attn_configs.kv_cache_dtype
        in {KvCacheDataType.BASE, KvCacheDataType.FP8}
        and attn_configs.size_per_head in {64, 128, 256}
        and page_size > 0
        and page_size.bit_count() == 1
        and attn_configs.is_causal
        and attn_configs.head_num > 0
        and attn_configs.kv_head_num > 0
        and attn_configs.head_num % attn_configs.kv_head_num == 0
    )


class PyFlashinferFa2TargetVerifyImpl(PyFlashinferPagedPrefillImpl):
    """SM9x target verification using the explicitly selected FlashInfer FA2 backend."""

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        return PyFlashinferPrefillPagedAttnOp(
            attn_configs,
            attn_inputs,
            backend="fa2",
        )

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        return (
            _supports_py_flashinfer_fa2_target_verify(attn_configs, attn_inputs)
            and attn_configs.rope_config.style != RopeStyle.Mrope
        )


class PyFlashinferMropeTargetVerifyImpl(FMHAImplBase):
    """SM9x target-verify path using fused MRoPE with FlashInfer FA2."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.fmha_impl = PyFlashinferPrefillPagedAttnOp(
            attn_configs,
            attn_inputs,
            backend="fa2",
        )
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.fmha_impl.set_params(self.fmha_params)
        self.fmha_impl.prepare(attn_inputs)

        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQOut(attn_configs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.attn_inputs = attn_inputs
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ) -> bool:
        return _supports_py_flashinfer_fa2_target_verify(
            attn_configs,
            attn_inputs,
        ) and attn_configs.rope_config.style == RopeStyle.Mrope

    def support_cuda_graph(self) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        query = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        common.apply_write_cache_store(
            self.write_cache_store_impl,
            self.attn_inputs,
            kv_cache,
        )
        return self.fmha_impl.forward(query, kv_cache)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        self.fmha_impl.prepare(attn_inputs, forbid_realloc=True)
        new_kv_cache_offset = self.rope_kvcache_impl.prepare(
            attn_inputs
        ).kv_cache_offset
        if new_kv_cache_offset is not None:
            common.copy_kv_cache_offset(
                self.rope_params.kv_cache_offset,
                new_kv_cache_offset,
            )


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
    ) -> torch.Tensor:
        """For ragged layout, reconstruct full qkv tensor from q, k, v."""
        # query: [total_tokens, num_heads, head_dim]
        # key: [total_tokens, num_kv_heads, head_dim]
        # value: [total_tokens, num_kv_heads, head_dim]

        # Flatten to 2D and concatenate
        q_flat = query.reshape(
            query.shape[0], -1
        )  # [total_tokens, num_heads * head_dim]
        k_flat = key.reshape(
            key.shape[0], -1
        )  # [total_tokens, num_kv_heads * head_dim]
        v_flat = value.reshape(
            value.shape[0], -1
        )  # [total_tokens, num_kv_heads * head_dim]

        # Concatenate along feature dimension
        qkv = torch.cat(
            [q_flat, k_flat, v_flat], dim=-1
        )  # [total_tokens, (num_heads + 2*num_kv_heads) * head_dim]

        return qkv

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
        self.head_dim_vo = attn_configs.size_per_head
        self.seq_size_per_block = attn_configs.kernel_tokens_per_block
        self.use_tensor_core = determine_use_tensor_core_from_configs(attn_configs)
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.g_workspace_buffer,
            "HND",
            use_tensor_cores=self.use_tensor_core,
        )
        self.kv_cache_dtype = attn_configs.kv_cache_dtype
        self.enable_cuda_graph = attn_inputs.is_cuda_graph
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer)

    def set_params(self, params: rtp_llm_ops.FlashInferMlaAttnParams) -> None:
        """Set the params object to be used by this op."""
        self.fmha_params = params

    def _get_kv_data_type(self, attn_inputs: PyAttentionInputs) -> torch.dtype:
        if self.kv_cache_dtype == KvCacheDataType.FP8:
            return torch.float8_e4m3fn
        return get_scalar_type(attn_inputs.dtype)

    def _requires_fa2_cuda_graph_replan(self) -> bool:
        # FlashInfer BatchDecode routes tensor-core decode through fa2 BatchPrefill.
        # fa3 prefill paths do not need this replay-time plan refresh.
        return self.use_tensor_core

    def _plan_decode_wrapper(self, attn_inputs: PyAttentionInputs) -> None:
        if self._requires_fa2_cuda_graph_replan():
            page_indptr = self.fmha_params.decode_page_indptr_h
            page_indice = self.fmha_params.page_indice_h
            last_page_len = self.fmha_params.paged_kv_last_page_len_h
            plan_kwargs = {"non_blocking": True}
        else:
            page_indptr = self.fmha_params.decode_page_indptr_d
            page_indice = self.fmha_params.page_indice_d
            last_page_len = self.fmha_params.paged_kv_last_page_len_d
            plan_kwargs = {}

        self.decode_wrapper.plan(
            page_indptr,
            page_indice,
            last_page_len,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            self.seq_size_per_block,
            q_data_type=get_scalar_type(attn_inputs.dtype),
            kv_data_type=self._get_kv_data_type(attn_inputs),
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
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id,
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
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id,
            self.seq_size_per_block,
            forbid_realloc=True,
        )
        if self._requires_fa2_cuda_graph_replan():
            self._plan_decode_wrapper(attn_inputs)

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def forward(
        self, q: torch.Tensor, kv_cache: Optional[LayerKVCache], params: ParamsBase
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required"
        q = q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk)
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
        # Create implementations
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = PyFlashinferDecodeAttnOp(attn_configs, attn_inputs)
        self.rope_impl = FusedRopeKVCacheDecodeOp(attn_configs)
        self.attn_configs = attn_configs

        # Store input info
        self.attn_inputs = attn_inputs

        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.fmha_impl.set_params(self.fmha_params)
        self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        """Prepare FlashInfer/RoPE buffers and metadata for CUDA graph replay."""
        self.fmha_impl.prepare_for_cuda_graph_replay(attn_inputs)
        # Update rope params for correct position encoding during cuda graph replay
        new_rope_params = self.rope_impl.prepare(attn_inputs)
        common.copy_kv_cache_offset(
            self.rope_params.kv_cache_offset, new_rope_params.kv_cache_offset
        )

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
        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            qkv = self.rope_impl.forward(qkv, kv_cache, self.rope_params)

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FMHA forward
        return self.fmha_impl.forward(qkv, kv_cache, self.fmha_params)
