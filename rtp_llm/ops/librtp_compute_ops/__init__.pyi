from __future__ import annotations

import typing

import libth_transformer_config
import torch

from . import rtp_llm_ops

__all__: list[str] = [
    "BertEmbeddingInputs",
    "CacheGroupType",
    "LayerKVCache",
    "KVCache",
    "ParamsBase",
    "PyAttentionInputs",
    "PyCacheStoreInputs",
    "PyCaptureMetaData",
    "PyContextParallelParams",
    "PyEmbeddingInputs",
    "PyModelInitResources",
    "PyModelInputs",
    "PyModelOutputs",
    "PyMultimodalInputs",
    "PyPrefillCudaGaphCopyParams",
    "TypeMeta",
    "clear_comm_ops",
    "destroy_cpu_tp_broadcaster",
    "get_device_id",
    "preprocess_gemm_weight_by_key",
    "preprocess_weight_scale",
    "get_scalar_type",
    "get_typemeta",
    "init_cpu_tp_broadcaster",
    "init_exec_ctx",
    "register_comm_ops",
    "rtp_llm_ops",
]
class BertEmbeddingInputs:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        combo_position_ids: torch.Tensor = ...,
        position_encoding: torch.Tensor = ...,
        combo_tokens_type_ids: torch.Tensor = ...,
        token_type_embedding: torch.Tensor = ...,
        input_embedding_scalar: float = 1.0,
    ) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def combo_position_ids(self) -> torch.Tensor:
        """
        Combined position IDs tensor
        """

    @combo_position_ids.setter
    def combo_position_ids(self, arg0: torch.Tensor) -> None: ...
    @property
    def combo_tokens_type_ids(self) -> torch.Tensor:
        """
        Combined token type IDs tensor
        """

    @combo_tokens_type_ids.setter
    def combo_tokens_type_ids(self, arg0: torch.Tensor) -> None: ...
    @property
    def input_embedding_scalar(self) -> float:
        """
        Input embedding scalar value
        """

    @input_embedding_scalar.setter
    def input_embedding_scalar(self, arg0: float) -> None: ...
    @property
    def position_encoding(self) -> torch.Tensor:
        """
        Position encoding tensor
        """

    @position_encoding.setter
    def position_encoding(self, arg0: torch.Tensor) -> None: ...
    @property
    def token_type_embedding(self) -> torch.Tensor:
        """
        Token type embedding tensor
        """

    @token_type_embedding.setter
    def token_type_embedding(self, arg0: torch.Tensor) -> None: ...

class CacheGroupType:
    """
    Members:

      LINEAR

      FULL

      SWA
    """

    FULL: typing.ClassVar[CacheGroupType]
    LINEAR: typing.ClassVar[CacheGroupType]
    SWA: typing.ClassVar[CacheGroupType]
    __members__: typing.ClassVar[dict[str, CacheGroupType]]
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class LayerKVCache:
    """Per-layer KV cache view. Returned by KVCache.get_layer_cache()."""

    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        kv_cache_base: torch.Tensor,
        seq_size_per_block: int,
        layer_id: int = -1,
        group_id: int = -1,
        tag: str = "default",
        kv_scale_base: torch.Tensor | None = None,
    ) -> None: ...
    @property
    def kv_cache_base(self) -> torch.Tensor:
        """
        Key/value cache tensor (per-layer view)
        """

    @kv_cache_base.setter
    def kv_cache_base(self, arg0: torch.Tensor) -> None: ...
    @property
    def kv_scale_base(self) -> torch.Tensor:
        """
        Key/value cache scale tensor
        """

    @kv_scale_base.setter
    def kv_scale_base(self, arg0: torch.Tensor) -> None: ...
    @property
    def layer_id(self) -> int:
        """
        Global layer id
        """

    @property
    def group_id(self) -> int:
        """
        Cache group id (-1 = default)
        """

    @property
    def tag(self) -> str:
        """
        Cache group tag
        """

    @property
    def seq_size_per_block(self) -> int:
        """
        Sequence size per block
        """

class KVCache:
    """Read-only whole-model KV cache created by the C++ runtime."""

    @property
    def group_tags(self) -> list[str]: ...
    @property
    def layer_count(self) -> int: ...
    @typing.overload
    def get_layer_cache(self, arg0: int) -> LayerKVCache:
        """Return a per-layer LayerKVCache for the given global layer id."""
        ...
    @typing.overload
    def get_layer_cache(self, arg0: int, arg1: str) -> LayerKVCache:
        """Return a LayerKVCache for the given layer and tag."""
        ...
    def get_layer_cache_groups(self, arg0: int) -> list[LayerKVCache]:
        """Return all LayerKVCache objects for every group the layer owns."""
        ...
    def get_seq_size_per_block(self, arg0: str) -> int:
        """Return the physical sequence size per block for a cache tag."""
        ...
    def get_kernel_seq_size_per_block(self, arg0: str) -> int:
        """Return the kernel sequence size per block for a cache tag."""
        ...

class ParamsBase:
    def __init__(self) -> None: ...
    def fill_params(
        self,
        sequence_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
        kv_cache_block_id_host: torch.Tensor,
        batch_size: int,
        seq_size_per_block: int,
    ) -> None:
        """
        Fill parameters for CUDA graph execution
        """

class PyAttentionInputs:
    def __init__(self) -> None: ...
    cache_store_inputs: PyCacheStoreInputs | None
    combo_position_ids: torch.Tensor
    context_parallel_info: PyContextParallelParams | None
    context_total_kv_length: int
    cu_kv_seqlens_device: torch.Tensor
    cu_seqlens_device: torch.Tensor
    cu_seqlens: torch.Tensor
    decode_cu_seqlens_device: torch.Tensor
    decode_cu_seqlens: torch.Tensor
    dtype: TypeMeta
    input_lengths: torch.Tensor
    is_cuda_graph: bool
    is_prefill: bool
    is_s_padded: bool
    is_target_verify: bool
    padding_offset: torch.Tensor
    prefill_cuda_graph_copy_params: PyPrefillCudaGaphCopyParams | None
    prefix_lengths: torch.Tensor
    sequence_lengths: torch.Tensor
    sequence_lengths_plus_1_device: torch.Tensor
    total_tokens: int
    headwise_config: dict | None
    kv_cache_kernel_block_id: torch.Tensor
    kv_cache_kernel_block_id_device: torch.Tensor
    kv_cache_block_id: torch.Tensor
    kv_cache_block_id_device: torch.Tensor
    @property
    def input_lengths_device(self) -> torch.Tensor: ...
    @property
    def prefix_lengths_device(self) -> torch.Tensor: ...
    def __repr__(self) -> str: ...
    def __copy__(self) -> PyAttentionInputs: ...

class PyCacheStoreInputs:
    def __init__(self) -> None: ...

class PyCaptureMetaData:
    def __init__(self) -> None: ...

class PyContextParallelParams:
    prefill_actual_input_lengths_cpu: torch.Tensor
    prefill_cp_chunk_lengths: torch.Tensor
    prefill_cp_padding_lengths: torch.Tensor
    prefill_qkv_padding_mask: torch.Tensor
    prefill_qkv_restore_indice: torch.Tensor
    prefill_shuffle_indices: torch.Tensor
    def __init__(self) -> None: ...

class PyEmbeddingInputs:
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def combo_tokens_type_ids(self) -> torch.Tensor:
        """
        Combined token type IDs tensor
        """

    @combo_tokens_type_ids.setter
    def combo_tokens_type_ids(self, arg0: torch.Tensor) -> None: ...
    @property
    def text_tokens_mask(self) -> torch.Tensor:
        """
        Text tokens mask tensor
        """

    @text_tokens_mask.setter
    def text_tokens_mask(self, arg0: torch.Tensor) -> None: ...

class PyModelInitResources:
    def __init__(self) -> None: ...
    @property
    def kv_cache(self) -> KVCache | None:
        """
        Layered kv cache for all layers
        """
    @property
    def is_speculative(self) -> bool: ...
    @property
    def is_decode_role(self) -> bool: ...
    @property
    def max_context_batch_size(self) -> int: ...

class PyModelInputs:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(
        self,
        input_ids: torch.Tensor = ...,
        input_hiddens: torch.Tensor = ...,
        combo_position_ids: torch.Tensor = ...,
        embedding_inputs: PyEmbeddingInputs = ...,
        multimodal_inputs: PyMultimodalInputs = ...,
        attention_inputs: PyAttentionInputs | dict[str, PyAttentionInputs] = ...,
        bert_embedding_inputs: BertEmbeddingInputs = ...,
    ) -> None: ...
    @property
    def attention_inputs(self) -> PyAttentionInputs | dict[str, PyAttentionInputs]:
        """
        Attention inputs structure
        """

    @attention_inputs.setter
    def attention_inputs(self, arg0: PyAttentionInputs | dict[str, PyAttentionInputs]) -> None: ...
    @property
    def bert_embedding_inputs(self) -> BertEmbeddingInputs:
        """
        BERT embedding inputs structure
        """

    @bert_embedding_inputs.setter
    def bert_embedding_inputs(self, arg0: BertEmbeddingInputs) -> None: ...
    @property
    def combo_position_ids(self) -> torch.Tensor:
        """
        Combo position IDs tensor
        """

    @combo_position_ids.setter
    def combo_position_ids(self, arg0: torch.Tensor) -> None: ...
    @property
    def embedding_inputs(self) -> PyEmbeddingInputs:
        """
        Embedding inputs structure
        """

    @embedding_inputs.setter
    def embedding_inputs(self, arg0: PyEmbeddingInputs) -> None: ...
    @property
    def input_hiddens(self) -> torch.Tensor:
        """
        Input hidden states tensor
        """

    @input_hiddens.setter
    def input_hiddens(self, arg0: torch.Tensor) -> None: ...
    @property
    def input_ids(self) -> torch.Tensor:
        """
        Input token IDs tensor
        """

    @input_ids.setter
    def input_ids(self, arg0: torch.Tensor) -> None: ...
    @property
    def multimodal_inputs(self) -> PyMultimodalInputs:
        """
        Multimodal inputs structure
        """

    @multimodal_inputs.setter
    def multimodal_inputs(self, arg0: PyMultimodalInputs) -> None: ...

class PyModelOutputs:
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """

    @typing.overload
    def __init__(self, hidden_states: torch.Tensor) -> None:
        """
        Initialize with hidden states tensor
        """

    @property
    def hidden_states(self) -> torch.Tensor:
        """
        Hidden states output tensor
        """

    @hidden_states.setter
    def hidden_states(self, arg0: torch.Tensor) -> None: ...

class PyMultimodalInputs:
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def mm_deepstack_embeds(self) -> list[torch.Tensor]:
        """
        Multimodal deepstack embeds tensor
        """

    @mm_deepstack_embeds.setter
    def mm_deepstack_embeds(self, arg0: list[torch.Tensor]) -> None: ...
    @property
    def mm_features_locs(self) -> torch.Tensor:
        """
        Multimodal features locations tensor
        """
    @mm_features_locs.setter
    def mm_features_locs(self, arg0: torch.Tensor) -> None:
        ...
    @property
    def multimodal_features(self) -> list[torch.Tensor]:
        """
        Multimodal features tensor
        """

    @multimodal_features.setter
    def multimodal_features(self, arg0: list[torch.Tensor]) -> None: ...

class PyPrefillCudaGaphCopyParams:
    cuda_graph_prefill_batch_size: torch.Tensor
    max_batch_size: int
    max_seq_len: int
    def __init__(self) -> None: ...

class TypeMeta:
    def __init__(self) -> None: ...

def get_device_id() -> int: ...
def preprocess_gemm_weight_by_key(key: str, weight: torch.Tensor, user_arm_gemm_use_kai: bool) -> torch.Tensor: ...
def preprocess_weight_scale(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor: ...
def get_scalar_type(arg0: TypeMeta) -> torch.dtype:
    """
    Convert TypeMeta to scalar type
    """

def get_typemeta(arg0: torch.Tensor) -> TypeMeta:
    """
    Convert tensor dtype to TypeMeta
    """

def init_exec_ctx(
    device_id: int,
    trace_memory: bool,
    enable_comm_overlap: bool,
    mla_ops_type: int,
) -> None: ...

def init_cpu_tp_broadcaster(tp_rank: int, tp_size: int, base_path: str) -> None: ...

def destroy_cpu_tp_broadcaster() -> None: ...

def register_comm_ops(broadcast_fn: typing.Callable, allreduce_fn: typing.Callable, allgather_fn: typing.Callable) -> None: ...

def clear_comm_ops() -> None: ...
