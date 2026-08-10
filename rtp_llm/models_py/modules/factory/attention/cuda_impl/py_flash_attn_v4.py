import logging
import math
import os
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Callable, NamedTuple, Optional

import torch
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.utils.arch import is_sm90
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCachePrefillOpQOut,
    LayerKVCache,
    PyAttentionInputs,
)

_FA4_TILE_M = 64
_FA4_TILE_N = 32
_FA4_NUM_THREADS = 256
_FA4_MAX_SPLITS = 128
logger = logging.getLogger(__name__)

_FA4_DEPENDENCY_SPECS = {
    "nvidia-cutlass-dsl": SpecifierSet(">=4.5.3,<4.6"),
    "apache-tvm-ffi": SpecifierSet(">=0.1.12,<0.2"),
    "quack-kernels": SpecifierSet(">=0.5.0,<0.6"),
    "torch-c-dlpack-ext": SpecifierSet(">=0.1.5,<0.2"),
}
_FA4_LOG_LEVEL_NAMES = {"off": 0, "host": 1, "kernel": 2, "max": 3}


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def get_fa4_target_verify_num_splits(
    *,
    sm_count: int,
    batch_size: int,
    query_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    max_kv_len: int,
    tile_m: int = _FA4_TILE_M,
    tile_n: int = _FA4_TILE_N,
) -> int:
    """Fill one SM wave with split-KV CTAs for a fixed CUDA Graph bucket."""
    if (
        min(
            sm_count,
            batch_size,
            query_len,
            num_q_heads,
            num_kv_heads,
            max_kv_len,
            tile_m,
            tile_n,
        )
        <= 0
    ):
        raise ValueError("FA4 split inputs must all be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    packed_q_rows = query_len * (num_q_heads // num_kv_heads)
    m_blocks_per_kv_head = _ceil_div(packed_q_rows, tile_m)
    total_m_blocks = batch_size * num_kv_heads * m_blocks_per_kv_head
    num_kv_tiles = _ceil_div(max_kv_len, tile_n)
    return max(1, min(num_kv_tiles, sm_count // total_m_blocks, _FA4_MAX_SPLITS))


def _check_fa4_dependencies() -> None:
    for package_name, specifier in _FA4_DEPENDENCY_SPECS.items():
        try:
            installed = version(package_name)
        except PackageNotFoundError as error:
            raise RuntimeError(
                f"FA4 target verify requires {package_name}{specifier}"
            ) from error
        try:
            installed_version = Version(installed)
        except InvalidVersion as error:
            raise RuntimeError(
                f"FA4 target verify found invalid {package_name} version {installed!r}"
            ) from error
        if installed_version not in specifier:
            raise RuntimeError(
                f"FA4 target verify requires {package_name}{specifier}, got {installed}"
            )


def _get_fa4_log_level() -> int:
    raw_level = os.environ.get("FA_LOG_LEVEL", "0")
    normalized_level = raw_level.strip().lower()
    if normalized_level in _FA4_LOG_LEVEL_NAMES:
        return _FA4_LOG_LEVEL_NAMES[normalized_level]
    try:
        level = int(normalized_level)
    except ValueError:
        logger.warning(
            "invalid FA_LOG_LEVEL=%r; disabling FA4 host and kernel logging",
            raw_level,
        )
        return 0
    if not 0 <= level <= 3:
        clamped_level = max(0, min(level, 3))
        logger.warning(
            "FA_LOG_LEVEL=%r is outside [0, 3]; using %d",
            raw_level,
            clamped_level,
        )
        return clamped_level
    return level


def _bind_fa4_log_imports(fa_logging, fa_log: Callable[[int, str], None]) -> None:
    previous_fa_log = getattr(fa_logging, "fa_log", None)
    package_names = set()
    package_name = getattr(fa_logging, "__package__", None)
    if package_name:
        package_names.add(package_name)
    module_name = getattr(fa_logging, "__name__", "")
    if "." in module_name:
        package_names.add(module_name.rsplit(".", 1)[0])
    for loaded_name, module in tuple(sys.modules.items()):
        if module is fa_logging and loaded_name.endswith(".fa_logging"):
            package_names.add(loaded_name.rsplit(".", 1)[0])

    fa_logging.fa_log = fa_log
    interface_names = {f"{name}.interface" for name in package_names}
    package_prefixes = tuple(f"{name}." for name in package_names)
    for loaded_name, module in tuple(sys.modules.items()):
        if module is None:
            continue
        is_interface = loaded_name in interface_names
        holds_previous_import = (
            previous_fa_log is not None
            and loaded_name.startswith(package_prefixes)
            and getattr(module, "fa_log", None) is previous_fa_log
        )
        if is_interface or holds_previous_import:
            module.fa_log = fa_log


def _configure_fa4_logging(fa_logging) -> Callable[[int, str], None]:
    """Route vendored host logs through RTP-LLM's logging configuration."""
    configured_level = _get_fa4_log_level()
    vendor_logger = logging.getLogger("flash_attn")
    fa_logging.set_fa_log_level(0)
    retained_handler_ids = {id(handler) for handler in vendor_logger.handlers}
    fa_logging.set_fa_log_level(configured_level)
    for handler in tuple(vendor_logger.handlers):
        if id(handler) not in retained_handler_ids:
            vendor_logger.removeHandler(handler)
    vendor_logger.setLevel(logging.NOTSET)
    vendor_logger.propagate = True

    def rtp_fa_log(level: int, message: str) -> None:
        if fa_logging.get_fa_log_level() >= level:
            log_method = vendor_logger.info if level <= 1 else vendor_logger.debug
            log_method(message)

    # Interface modules import fa_log during initialization, so install the
    # host-integrated implementation before importing the FA4 interface.
    _bind_fa4_log_imports(fa_logging, rtp_fa_log)
    return rtp_fa_log


@lru_cache(maxsize=1)
def _load_fa4_forward():
    _check_fa4_dependencies()
    try:
        from rtp_llm.third_party.vllm_flash_attention.cute import fa_logging

        rtp_fa_log = _configure_fa4_logging(fa_logging)
        from rtp_llm.third_party.vllm_flash_attention.cute import (
            interface as fa4_interface,
        )

        _bind_fa4_log_imports(fa_logging, rtp_fa_log)
        _flash_attn_fwd = fa4_interface._flash_attn_fwd
    except Exception as error:
        raise RuntimeError(
            f"failed to load vendored FA4 CuTe backend: {error}"
        ) from error
    logger.info("loaded vendored FA4 CuTe target-verify backend")
    return _flash_attn_fwd


@lru_cache(maxsize=1)
def _fa4_is_available() -> bool:
    try:
        _load_fa4_forward()
    except Exception as error:
        logger.error("FA4 target verify is unavailable; falling back: %s", error)
        return False
    return True


class FlashAttn4TargetVerifyParams(NamedTuple):
    batch_size: int
    query_len: int
    max_kv_len: int
    num_splits: int
    query_lengths: torch.Tensor
    kv_lengths: torch.Tensor
    cu_kv_seqlens: torch.Tensor
    page_table: torch.Tensor


class FlashAttn4TargetVerifyOp:
    """Shape-specialized SM90 FA4 paged attention for target verification."""

    def __init__(self, attn_configs: AttentionConfigs) -> None:
        self.attn_configs = attn_configs
        self.head_dim = attn_configs.size_per_head
        self.head_num = attn_configs.head_num
        self.kv_head_num = attn_configs.kv_head_num
        self.page_size = attn_configs.kernel_tokens_per_block
        self.softmax_scale = (
            attn_configs.softmax_extra_scale
            / attn_configs.q_scaling
            * self.head_dim**-0.5
        )
        self._forward = _load_fa4_forward()

    @staticmethod
    def _uniform_query_len(attn_inputs: PyAttentionInputs) -> int:
        input_lengths = attn_inputs.input_lengths
        if input_lengths is None or input_lengths.numel() == 0:
            return 0
        query_len = int(input_lengths[0].item())
        if query_len <= 0 or not bool((input_lengths == query_len).all().item()):
            return 0
        return query_len

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        if not (
            is_sm90()
            and attn_inputs.is_target_verify
            and attn_inputs.is_prefill
            and attn_configs.need_rope_kv_cache
            and attn_inputs.is_cuda_graph
            and attn_configs.dtype == torch.bfloat16
            and attn_configs.kv_cache_dtype == KvCacheDataType.BASE
            and attn_configs.size_per_head == 256
            and attn_configs.kernel_tokens_per_block == 64
            and attn_configs.is_causal
            and attn_configs.head_num > 0
            and attn_configs.kv_head_num > 0
            and attn_configs.head_num % attn_configs.kv_head_num == 0
        ):
            return False

        query_len = cls._uniform_query_len(attn_inputs)
        packed_q_rows = query_len * (attn_configs.head_num // attn_configs.kv_head_num)
        return (
            0 < packed_q_rows <= _FA4_TILE_M
            and attn_inputs.input_lengths_device is not None
            and attn_inputs.input_lengths_device.is_cuda
            and attn_inputs.cu_kv_seqlens_device is not None
            and attn_inputs.cu_kv_seqlens_device.is_cuda
            and attn_inputs.kv_cache_kernel_block_id_device is not None
            and attn_inputs.kv_cache_kernel_block_id_device.is_cuda
            and _fa4_is_available()
        )

    def prepare(self, attn_inputs: PyAttentionInputs) -> FlashAttn4TargetVerifyParams:
        query_len = self._uniform_query_len(attn_inputs)
        if query_len <= 0:
            raise ValueError(
                "FA4 target verify requires uniform positive query lengths"
            )

        # max_kv_len and num_splits are Python scalars compiled into the CUDA
        # Graph launch. CudaGraphRunner initializes capture prefix lengths from
        # the global KV limit and reuses the device metadata buffers on replay,
        # so this capture must represent an upper bound for every replay.
        cu_kv_seqlens = attn_inputs.cu_kv_seqlens_device
        kv_lengths = torch.empty_like(cu_kv_seqlens[:-1])
        torch.sub(cu_kv_seqlens[1:], cu_kv_seqlens[:-1], out=kv_lengths)
        batch_size = attn_inputs.input_lengths.size(0)
        max_kv_len = int(
            (attn_inputs.prefix_lengths + attn_inputs.input_lengths).max().item()
        )
        sm_count = torch.cuda.get_device_properties(
            attn_inputs.input_lengths_device.device
        ).multi_processor_count
        num_splits = get_fa4_target_verify_num_splits(
            sm_count=sm_count,
            batch_size=batch_size,
            query_len=query_len,
            num_q_heads=self.head_num,
            num_kv_heads=self.kv_head_num,
            max_kv_len=max_kv_len,
        )
        return FlashAttn4TargetVerifyParams(
            batch_size=batch_size,
            query_len=query_len,
            max_kv_len=max_kv_len,
            num_splits=num_splits,
            query_lengths=attn_inputs.input_lengths_device,
            kv_lengths=kv_lengths,
            cu_kv_seqlens=cu_kv_seqlens,
            page_table=attn_inputs.kv_cache_kernel_block_id_device,
        )

    def prepare_cuda_graph(
        self,
        params: FlashAttn4TargetVerifyParams,
        attn_inputs: PyAttentionInputs,
    ) -> None:
        replay_batch_size = attn_inputs.input_lengths.numel()
        metadata_batch_sizes = {
            "query_lengths": attn_inputs.input_lengths_device.numel(),
            "prefix_lengths": attn_inputs.prefix_lengths.numel(),
            "kv_lengths": params.kv_lengths.numel(),
            "cu_kv_seqlens": attn_inputs.cu_kv_seqlens_device.numel() - 1,
            "page_table": attn_inputs.kv_cache_kernel_block_id_device.size(0),
        }
        if replay_batch_size != params.batch_size or any(
            size != params.batch_size for size in metadata_batch_sizes.values()
        ):
            raise RuntimeError(
                "FA4 CUDA graph replay batch size does not match the capture "
                f"contract: capture batch_size={params.batch_size}, "
                f"replay batch_size={replay_batch_size}, "
                f"metadata batch sizes={metadata_batch_sizes}"
            )

        # CudaGraphRunner refreshes each host/device metadata pair before this
        # callback, so host lengths can enforce scalar graph bounds without a
        # device-to-host synchronization.
        query_lengths = attn_inputs.input_lengths
        valid_query_lengths = (query_lengths == params.query_len) | (query_lengths == 0)
        replay_max_query_len = int(query_lengths.max().item())
        if replay_max_query_len != params.query_len or not bool(
            valid_query_lengths.all().item()
        ):
            raise RuntimeError(
                "FA4 CUDA graph replay query length differs from capture: "
                f"capture query_len={params.query_len}, "
                f"replay query_lengths={query_lengths.tolist()}"
            )

        replay_kv_lengths = attn_inputs.prefix_lengths + query_lengths
        replay_min_kv_len = int(replay_kv_lengths.min().item())
        replay_max_kv_len = int(replay_kv_lengths.max().item())
        if replay_min_kv_len < 0:
            raise RuntimeError(
                "FA4 CUDA graph replay requires non-negative KV lengths, "
                f"got replay min_kv_len={replay_min_kv_len}"
            )
        if replay_max_kv_len > params.max_kv_len:
            raise RuntimeError(
                "FA4 CUDA graph replay KV length exceeds capture max_kv_len: "
                f"capture max_kv_len={params.max_kv_len}, "
                f"replay max_kv_len={replay_max_kv_len}"
            )

        page_table = attn_inputs.kv_cache_kernel_block_id_device
        if page_table.dim() != 2:
            raise RuntimeError(
                "FA4 CUDA graph replay requires a 2D page table, "
                f"got replay shape={tuple(page_table.shape)}"
            )
        page_table_kv_capacity = page_table.size(1) * self.page_size
        if replay_max_kv_len > page_table_kv_capacity:
            raise RuntimeError(
                "FA4 CUDA graph replay KV length exceeds page-table capacity: "
                f"capture max_kv_len={params.max_kv_len}, "
                f"replay max_kv_len={replay_max_kv_len}, "
                f"page_table_kv_capacity={page_table_kv_capacity}"
            )

        max_capture_splits = min(
            _ceil_div(params.max_kv_len, _FA4_TILE_N), _FA4_MAX_SPLITS
        )
        if not 1 <= params.num_splits <= max_capture_splits:
            raise RuntimeError(
                "FA4 CUDA graph capture num_splits exceeds its max_kv_len bound: "
                f"capture num_splits={params.num_splits}, "
                f"capture max_kv_len={params.max_kv_len}, "
                f"max num_splits={max_capture_splits}, "
                f"replay max_kv_len={replay_max_kv_len}"
            )

        fixed_buffers = (
            ("query_lengths", params.query_lengths, attn_inputs.input_lengths_device),
            (
                "cu_kv_seqlens",
                params.cu_kv_seqlens,
                attn_inputs.cu_kv_seqlens_device,
            ),
            ("page_table", params.page_table, page_table),
        )
        for name, capture_buffer, replay_buffer in fixed_buffers:
            if capture_buffer.data_ptr() != replay_buffer.data_ptr():
                raise RuntimeError(
                    "FA4 CUDA graph replay replaced a captured metadata buffer: "
                    f"buffer={name}, capture data_ptr={capture_buffer.data_ptr()}, "
                    f"replay data_ptr={replay_buffer.data_ptr()}"
                )

        torch.sub(
            params.cu_kv_seqlens[1:],
            params.cu_kv_seqlens[:-1],
            out=params.kv_lengths,
        )

    def _call_fa4(
        self,
        dense_query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        params: FlashAttn4TargetVerifyParams,
    ) -> torch.Tensor:
        return self._forward(
            dense_query,
            key_cache,
            value_cache,
            seqused_q=params.query_lengths,
            seqused_k=params.kv_lengths,
            max_seqlen_q=params.query_len,
            max_seqlen_k=params.max_kv_len,
            page_table=params.page_table,
            softmax_scale=self.softmax_scale,
            causal=True,
            tile_mn=(_FA4_TILE_M, _FA4_TILE_N),
            mma_pv_is_rs=True,
            intra_wg_overlap=True,
            num_threads=_FA4_NUM_THREADS,
            num_splits=params.num_splits,
            pack_gqa=True,
        )[0]

    def compile_probe(self, params: FlashAttn4TargetVerifyParams) -> None:
        """Compile and launch the captured shape while factory fallback is possible."""
        device = params.query_lengths.device
        dense_query = torch.empty(
            params.batch_size,
            params.query_len,
            self.head_num,
            self.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        combined_cache = torch.empty(
            1,
            2,
            self.kv_head_num,
            self.page_size,
            self.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        key_cache = combined_cache.select(1, 0).permute(0, 2, 1, 3)
        value_cache = combined_cache.select(1, 1).permute(0, 2, 1, 3)
        probe_params = params._replace(
            kv_lengths=torch.full_like(params.kv_lengths, self.page_size),
            page_table=torch.zeros_like(params.page_table),
        )
        self._call_fa4(dense_query, key_cache, value_cache, probe_params)
        torch.cuda.synchronize(device)

    def forward(
        self,
        query: torch.Tensor,
        kv_cache: LayerKVCache,
        params: FlashAttn4TargetVerifyParams,
    ) -> torch.Tensor:
        if kv_cache is None:
            raise ValueError("FA4 target verify requires a paged KV cache")
        paged_kv_cache = common.reshape_paged_kv_cache(
            kv_cache.kv_cache_base,
            self.kv_head_num,
            self.page_size,
            self.head_dim,
        )
        if query.dtype != paged_kv_cache.dtype:
            raise TypeError(
                "FA4 target verify requires matching BF16 Q/KV dtypes, "
                f"got Q={query.dtype}, KV={paged_kv_cache.dtype}"
            )

        original_shape = query.shape
        dense_query = query.reshape(
            params.batch_size,
            params.query_len,
            self.head_num,
            self.head_dim,
        )
        key_cache = paged_kv_cache.select(1, 0).permute(0, 2, 1, 3)
        value_cache = paged_kv_cache.select(1, 1).permute(0, 2, 1, 3)
        output = self._call_fa4(dense_query, key_cache, value_cache, params)
        return output.reshape(original_shape)


class FlashAttn4TargetVerifyImpl(FMHAImplBase):
    """Default SM90 CUDA Graph target-verify implementation."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.fmha_impl = FlashAttn4TargetVerifyOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQOut(attn_configs)
        self.attn_inputs = attn_inputs
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.fmha_impl.compile_probe(self.fmha_params)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return FlashAttn4TargetVerifyOp.support(attn_configs, attn_inputs)

    def support_cuda_graph(self) -> bool:
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if kv_cache is None:
            raise ValueError("FA4 target verify requires a paged KV cache")
        query = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(query, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        self.fmha_impl.prepare_cuda_graph(self.fmha_params, attn_inputs)
        new_kv_cache_offset = self.rope_kvcache_impl.prepare(
            attn_inputs
        ).kv_cache_offset
        if new_kv_cache_offset is not None:
            common.copy_kv_cache_offset(
                self.rope_params.kv_cache_offset,
                new_kv_cache_offset,
            )
