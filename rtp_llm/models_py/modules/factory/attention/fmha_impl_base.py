from abc import ABC, abstractmethod
from enum import Enum
from math import isfinite
from typing import Any, Dict, List, Optional

import torch

from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.ops import AttentionConfigs, FMHAConfig, FMHAType, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, ParamsBase, PyAttentionInputs


class PrefillCudaGraphCapability(str, Enum):
    """Full-prefill CUDA Graph capability of an attention implementation."""

    NEVER = "never"
    FULL_NO_PREFIX = "full_no_prefix"


class MlaImplBase(object):
    """Base class for MLA attention implementations."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        use_trt_fmha: bool = False,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        """Initialize MLA implementation base class.

        Args:
            attn_configs: Attention configuration
            attn_inputs: Attention input tensors
            weights: Model weights
            cos_sin_cache: Cosine and sine cache for RoPE
            fmha_config: FMHA configuration
            use_trt_fmha: Whether to use TensorRT FMHA
            quant_config: Quantization configuration
            max_seq_len: Maximum sequence length
            is_cuda_graph: Whether CUDA graph is enabled
        """
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.weights = weights
        self.cos_sin_cache = cos_sin_cache
        self.fmha_config = fmha_config
        self.use_trt_fmha = use_trt_fmha
        self.quant_config = quant_config
        self.max_seq_len = max_seq_len
        self.is_cuda_graph = is_cuda_graph
        self.fmha_params: Any = None
        self.cp_params: Optional[Any] = None
        self.parallelism_config = parallelism_config

    @staticmethod
    def is_sparse() -> bool:
        return False

    @staticmethod
    @abstractmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        return False

    def support_cuda_graph(self) -> bool:
        """Check if CUDA graph is supported."""
        return callable(getattr(self, "prepare_cuda_graph", None))

    def prefill_cuda_graph_capability(self) -> PrefillCudaGraphCapability:
        return PrefillCudaGraphCapability.NEVER

    def prepare(self, attn_inputs: PyAttentionInputs):
        """Prepare for attention computation."""
        pass

    @classmethod
    def support_parallelism_config(
        cls, parallelism_config: Optional[ParallelismConfig]
    ) -> bool:
        if parallelism_config is None:
            return True

        if not parallelism_config.prefill_cp_config.is_enabled():
            return True

        return cls.support_prefill_cp()

    @classmethod
    def support_prefill_cp(cls) -> bool:
        return False

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for attention computation."""
        raise NotImplementedError("forward method must be implemented by subclass")


class FMHAImplBase(ABC):
    """Flash Multi-Head Attention 实现的接口基类。
    该类定义了 FMHA 实现必须提供的接口方法。
    所有具体的实现类都应该继承此类并实现这些方法。
    """

    accepts_fmha_config = False

    @abstractmethod
    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """执行前向传播计算。

        Args:
            qkv: 输入的 QKV 张量
            kv_cache: 可选的 KV Cache，用于存储历史键值对
            layer_idx: 当前层索引，用于 headwise 等需要 per-layer 配置的实现

        Returns:
            计算后的注意力输出张量
        """
        raise NotImplementedError("Subclass must implement forward method")

    @staticmethod
    @abstractmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """检查当前实现是否支持给定的输入。

        Args:
            attn_configs: 注意力配置
            attn_inputs: 注意力输入参数

        Returns:
            bool: 如果支持则返回 True，否则返回 False
        """
        return False

    @classmethod
    def support_parallelism_config(
        cls, parallelism_config: Optional[ParallelismConfig]
    ) -> bool:
        """检查当前实现是否支持给定的并行配置。

        Args:
            parallelism_config: 并行配置，如果为 None 表示无特殊并行要求

        Returns:
            bool: 如果支持则返回 True，否则返回 False
        """
        # 如果没有并行配置，默认支持
        if parallelism_config is None:
            return True

        # 如果 prefill context parallel 未启用，默认支持
        if not parallelism_config.prefill_cp_config.is_enabled():
            return True

        # 如果 prefill CP 已启用，检查实现是否支持
        return cls.support_prefill_cp()

    def support_cuda_graph(self) -> bool:
        """检查是否支持 CUDA Graph 优化。


        Returns:
            bool: 如果支持 CUDA Graph 则返回 True，否则返回 False
            如果想支持cuda graph需要子类支持prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):这个函数
        """
        return callable(getattr(self, "prepare_cuda_graph", None))

    def prefill_cuda_graph_capability(self) -> PrefillCudaGraphCapability:
        """Return an explicit capability; having prepare_cuda_graph is insufficient."""
        return PrefillCudaGraphCapability.NEVER

    @classmethod
    def support_prefill_cp(cls) -> bool:
        return False

    # def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
    #     pass


class CacheFreePrefillGraphParams(ParamsBase):
    def check_recycle(self):
        return True


class CacheFreePrefillGraphImpl(FMHAImplBase):
    """Fixed-address ragged attention for cache-free embedding backbones.

    Planning runs outside capture. RoPE uses the engine's canonical cache and
    per-sequence positions; padding has its own ragged sequence. No pooling or
    application-specific segment semantics are imposed by this backend.
    """

    @staticmethod
    def unsupported_reason(configs) -> str:
        from importlib.util import find_spec

        from rtp_llm.ops import RopeStyle

        if configs.use_mla:
            return "cache-free embedding graph MLA is not implemented"
        if (
            getattr(configs, "is_sparse", False)
            or getattr(configs, "use_logn_attn", False)
            or getattr(configs, "q_scaling", 1.0) != 1.0
            or getattr(configs, "softmax_extra_scale", 1.0) != 1.0
        ):
            return "embedding graph sparse/logn/custom attention scaling requires an adapter"
        if configs.dtype not in (torch.float16, torch.bfloat16):
            return "embedding graph attention requires FP16 or BF16 activations"
        if configs.rope_config.style not in (RopeStyle.No, RopeStyle.Base):
            return "embedding graph supports no RoPE or base RoPE; other styles require an adapter"
        if configs.rope_config.style != RopeStyle.No and (
            configs.rope_config.dim <= 0
            or configs.rope_config.dim % 2
            or configs.rope_config.dim > configs.size_per_head
        ):
            return "invalid embedding graph rotary dimension"
        if configs.rope_config.offset != 0:
            return "embedding graph RoPE offsets require an adapter"
        if configs.rope_config.style == RopeStyle.Base and (
            not isfinite(configs.rope_config.scale) or configs.rope_config.scale < 1.0
        ):
            return "embedding graph base RoPE requires a finite scale >= 1"
        if (
            configs.head_num <= 0
            or configs.kv_head_num <= 0
            or configs.size_per_head <= 0
            or configs.head_num % configs.kv_head_num
        ):
            return "invalid embedding graph query/KV head counts"
        backend = "aiter" if torch.version.hip is not None else "flashinfer"
        if find_spec(backend) is None:
            return f"embedding graph attention backend {backend} is not installed"
        return ""

    def __init__(
        self,
        configs,
        attn_inputs,
        token_capacity,
        max_sequence_length=None,
        custom_mask=False,
        allocate_indptr=True,
    ):
        self.is_causal = configs.is_causal
        self.heads = configs.head_num
        self.kv_heads = configs.kv_head_num
        self.dim = configs.size_per_head
        self.dtype = configs.dtype
        self.token_capacity = token_capacity
        self.max_sequence_length = max_sequence_length
        self.request_capacity = attn_inputs.input_lengths.numel()
        self.padding_slots = (
            token_capacity + (max_sequence_length or token_capacity) - 1
        ) // (max_sequence_length or token_capacity)
        self.fmha_params = CacheFreePrefillGraphParams()
        self.cu_seqlens_host = None
        self.cu_seqlens_device = None
        if allocate_indptr:
            self.cu_seqlens_host = torch.empty(
                self.request_capacity + self.padding_slots + 1,
                dtype=torch.int32,
                pin_memory=True,
            )
            self.cu_seqlens_device = torch.empty_like(
                self.cu_seqlens_host, device="cuda"
            )
        self.rope_cache = None
        self.positions_host = None
        self.positions_device = None
        rope_config = getattr(configs, "rope_config", None)
        if rope_config is not None and int(rope_config.style) != 0:
            from rtp_llm.ops import get_rope_cache

            # Own the configuration and lifetime: the process singleton can
            # already contain another model's dimensions, base or shorter table.
            self.rope_cache = get_rope_cache(
                rope_config,
                max_sequence_length or token_capacity,
                interleave=False,
            )
            self.positions_host = torch.zeros(
                token_capacity, dtype=torch.int64, pin_memory=True
            )
            self.positions_device = torch.empty_like(self.positions_host, device="cuda")
        self.wrapper = None
        if torch.version.hip is None:
            from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

            # Each graph owns its planner workspace and indptr storage. Replanning
            # another bucket must not overwrite this graph's execution plan.
            self.workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
            self.kv_cu_seqlens_device = torch.empty_like(self.cu_seqlens_device)
            mask_buffers = {}
            if custom_mask:
                # Each sequence is packed separately and may round up one byte.
                if token_capacity * token_capacity > torch.iinfo(torch.int32).max:
                    raise ValueError("custom mask exceeds int32 mask-indptr capacity")
                mask_bytes = (
                    (token_capacity * token_capacity + 7) // 8
                    + self.request_capacity
                    + self.padding_slots
                )
                self.custom_mask_buffer = torch.empty(
                    mask_bytes, dtype=torch.uint8, device="cuda"
                )
                self.mask_indptr_buffer = torch.empty_like(self.cu_seqlens_device)
                mask_buffers = dict(
                    custom_mask_buf=self.custom_mask_buffer,
                    mask_indptr_buf=self.mask_indptr_buffer,
                )
            self.wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace,
                use_cuda_graph=True,
                qo_indptr_buf=self.cu_seqlens_device,
                kv_indptr_buf=self.kv_cu_seqlens_device,
                backend="fa2",
                **mask_buffers,
            )
        # Both synthetic requests and padding obey the per-sequence limit.
        self._prepare(attn_inputs, for_capture=True)

    def prepare_cuda_graph(self, attn_inputs):
        self._prepare(attn_inputs, for_capture=False)

    def _prepare(self, attn_inputs, for_capture):
        """Plan synthetic capture inputs or validate and plan a real replay batch."""
        lengths = attn_inputs.input_lengths
        if lengths.is_cuda:
            raise ValueError("embedding graph planning requires host input lengths")
        if lengths.numel() != self.request_capacity:
            raise ValueError("embedding graph request capacity changed")
        prefix = attn_inputs.prefix_lengths
        if prefix is not None and prefix.numel() and bool(prefix.any()):
            raise ValueError("embedding graph does not support prefix KV")
        if bool((lengths < 0).any()):
            raise ValueError("embedding graph sequence lengths must be nonnegative")
        if (
            not for_capture
            and self.max_sequence_length is not None
            and bool((lengths > self.max_sequence_length).any())
        ):
            raise ValueError("embedding graph sequence exceeds position-table capacity")
        real_tokens = int(lengths.sum())
        if real_tokens > self.token_capacity:
            raise ValueError("embedding graph token capacity exceeded")
        limit = self.max_sequence_length or self.token_capacity
        if for_capture:
            # Profile legal sequence lengths, rather than a bucket-sized
            # synthetic request that can exceed the model's position limit.
            lengths = torch.tensor(
                [
                    min(limit, max(0, self.token_capacity - i * limit))
                    for i in range(self.request_capacity)
                ],
                dtype=torch.int32,
            )
            real_tokens = int(lengths.sum())
        self._planned_lengths = lengths
        self.cu_seqlens_host[0] = 0
        torch.cumsum(
            lengths, 0, out=self.cu_seqlens_host[1 : self.request_capacity + 1]
        )
        for i in range(self.padding_slots):
            self.cu_seqlens_host[self.request_capacity + i + 1] = min(
                self.token_capacity, real_tokens + (i + 1) * limit
            )
        if self.positions_host is not None:
            positions = getattr(attn_inputs, "combo_position_ids", None)
            if positions is not None and positions.numel():
                raise ValueError(
                    "explicit embedding graph positions require a model adapter"
                )
            self.positions_host.zero_()
            offset = 0
            for length in lengths.tolist():
                positions = torch.arange(length, dtype=torch.int64)
                self.positions_host[offset : offset + length].copy_(positions)
                offset += length
            self.positions_device.copy_(self.positions_host)
        if self.wrapper is None:
            self.cu_seqlens_device.copy_(self.cu_seqlens_host)
        else:
            self.wrapper.plan(
                self.cu_seqlens_host,
                self.cu_seqlens_host,
                self.heads,
                self.kv_heads,
                self.dim,
                self.dim,
                causal=self.is_causal,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
                o_data_type=self.dtype,
                disable_split_kv=True,
                **self._mask_plan_args(attn_inputs, for_capture),
            )

    def _mask_plan_args(self, attn_inputs, for_capture):
        return {}

    def _apply_rotary(self, q, k):
        if self.rope_cache is None:
            return q, k
        # Canonical cache has non-interleaved [cos, sin] rows, matching the
        # existing ragged eager RoPE path. QK normalization remains in the layer.
        cache = self.rope_cache.index_select(0, self.positions_device).float()
        cos, sin = cache.chunk(2, dim=-1)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        rotary_dim = cache.size(-1)

        def rotate(tensor):
            left, right = tensor[..., :rotary_dim].float().chunk(2, dim=-1)
            rotated = torch.cat(
                (left * cos - right * sin, right * cos + left * sin), dim=-1
            )
            return torch.cat(
                (rotated.to(tensor.dtype), tensor[..., rotary_dim:]), dim=-1
            )

        return rotate(q), rotate(k)

    def forward(self, qkv, kv_cache, layer_idx=0):
        if kv_cache is not None:
            raise ValueError("embedding graph cannot use KV cache")
        q, k, v = torch.split(
            qkv.reshape(self.token_capacity, -1),
            [self.heads * self.dim, self.kv_heads * self.dim, self.kv_heads * self.dim],
            dim=-1,
        )
        q = q.reshape(self.token_capacity, self.heads, self.dim)
        k = k.reshape(self.token_capacity, self.kv_heads, self.dim)
        v = v.reshape(self.token_capacity, self.kv_heads, self.dim)
        q, k = self._apply_rotary(q, k)
        if self.wrapper is not None:
            output = self.wrapper.run(q, k, v)
        else:
            from aiter import flash_attn_varlen_func

            output = flash_attn_varlen_func(
                q,
                k,
                v,
                self.cu_seqlens_device,
                self.cu_seqlens_device,
                self.max_sequence_length or self.token_capacity,
                self.max_sequence_length or self.token_capacity,
                dropout_p=0.0,
                causal=self.is_causal,
            )
        return output.reshape(self.token_capacity, self.heads * self.dim)

    @staticmethod
    def support(configs, attn_inputs):
        return attn_inputs.is_prefill

    def support_cuda_graph(self):
        return True
