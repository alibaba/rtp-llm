from __future__ import annotations
import libth_transformer_config
import torch
import typing
from . import rtp_llm_ops
__all__: list[str] = ['BertEmbeddingInputs', 'DeviceExporter', 'DeviceType', 'KVCache', 'MlaParams', 'ParamsBase', 'PyAttentionInputs', 'PyCacheStoreInputs', 'PyCaptureMetaData', 'PyModelInitResources', 'PyModelInputs', 'PyModelOutputs', 'TypeMeta', 'get_device', 'init_device', 'rtp_llm_ops']
class BertEmbeddingInputs:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, combo_position_ids: torch.Tensor = ..., position_encoding: torch.Tensor = ..., combo_tokens_type_ids: torch.Tensor = ..., token_type_embedding: torch.Tensor = ..., input_embedding_scalar: float = 1.0) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def combo_position_ids(self) -> torch.Tensor:
        """
        Combined position IDs tensor
        """
    @combo_position_ids.setter
    def combo_position_ids(self, arg0: torch.Tensor) -> None:
        ...
    @property
    def combo_tokens_type_ids(self) -> torch.Tensor:
        """
        Combined token type IDs tensor
        """
    @combo_tokens_type_ids.setter
    def combo_tokens_type_ids(self, arg0: torch.Tensor) -> None:
        ...
    @property
    def input_embedding_scalar(self) -> float:
        """
        Input embedding scalar value
        """
    @input_embedding_scalar.setter
    def input_embedding_scalar(self, arg0: float) -> None:
        ...
    @property
    def position_encoding(self) -> torch.Tensor:
        """
        Position encoding tensor
        """
    @position_encoding.setter
    def position_encoding(self, arg0: torch.Tensor) -> None:
        ...
    @property
    def token_type_embedding(self) -> torch.Tensor:
        """
        Token type embedding tensor
        """
    @token_type_embedding.setter
    def token_type_embedding(self, arg0: torch.Tensor) -> None:
        ...
class DeviceExporter:
    def get_device_id(self) -> int:
        ...
    def get_device_type(self) -> DeviceType:
        ...
    def pack_int8_tensor_to_packed_int4(self, weight: torch.Tensor) -> torch.Tensor:
        ...
    def preprocess_gemm_weight_by_key(self, key: str, weight: torch.Tensor, user_arm_gemm_use_kai: bool) -> torch.Tensor:
        ...
    def preprocess_weight_scale(self, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        ...
    def preprocess_weights_for_mixed_gemm(self, weight: torch.Tensor, quant_type: typing.Any, arch: str) -> torch.Tensor:
        ...
    def symmetric_quantize_last_axis_of_batched_matrix(self, weight: torch.Tensor, quant_type: typing.Any, arch: str) -> list[torch.Tensor]:
        ...
    def update_current_torch_stream(self) -> None:
        ...
class DeviceType:
    """
    Members:
    
      Cpu
    
      Cuda
    
      Yitian
    
      ArmCpu
    
      ROCm
    
      Ppu
    """
    ArmCpu: typing.ClassVar[DeviceType]  # value = <DeviceType.ArmCpu: 3>
    Cpu: typing.ClassVar[DeviceType]  # value = <DeviceType.Cpu: 0>
    Cuda: typing.ClassVar[DeviceType]  # value = <DeviceType.Cuda: 1>
    Ppu: typing.ClassVar[DeviceType]  # value = <DeviceType.Ppu: 5>
    ROCm: typing.ClassVar[DeviceType]  # value = <DeviceType.ROCm: 4>
    Yitian: typing.ClassVar[DeviceType]  # value = <DeviceType.Yitian: 2>
    __members__: typing.ClassVar[dict[str, DeviceType]]  # value = {'Cpu': <DeviceType.Cpu: 0>, 'Cuda': <DeviceType.Cuda: 1>, 'Yitian': <DeviceType.Yitian: 2>, 'ArmCpu': <DeviceType.ArmCpu: 3>, 'ROCm': <DeviceType.ROCm: 4>, 'Ppu': <DeviceType.Ppu: 5>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class KVCache:
    def __init__(self) -> None:
        ...
    def get_layer_cache(self, arg0: int) -> KVCache:
        ...
    @property
    def k_cache_base(self) -> torch.Tensor:
        """
        Key cache base tensor
        """
    @k_cache_base.setter
    def k_cache_base(self, arg0: torch.Tensor) -> None:
        ...
    @property
    def k_scale_base(self) -> torch.Tensor:
        """
        Key cache scale tensor
        """
    @property
    def layer_id(self) -> int:
        """
        kv cache layer id
        """
    @property
    def v_cache_base(self) -> torch.Tensor:
        """
        Value cache base tensor
        """
    @property
    def v_scale_base(self) -> torch.Tensor:
        """
        Value cache scale tensor
        """
class MlaParams:
    def __init__(self) -> None:
        ...
    @property
    def batch_indice(self) -> torch.Tensor:
        ...
    @property
    def kvlen(self) -> torch.Tensor:
        ...
    @property
    def page_indice(self) -> torch.Tensor:
        ...
    @property
    def page_indptr(self) -> torch.Tensor:
        ...
    @property
    def paged_kv_last_page_len(self) -> torch.Tensor:
        ...
    @property
    def positions(self) -> torch.Tensor:
        ...
    @property
    def qo_indptr(self) -> torch.Tensor:
        ...
class ParamsBase:
    def __init__(self) -> None:
        ...
    def fill_params(self, sequence_lengths: torch.Tensor, input_lengths: torch.Tensor, kv_cache_block_id_host: torch.Tensor, batch_size: int, seq_size_per_block: int) -> None:
        """
        Fill parameters for CUDA graph execution
        """
class PyAttentionInputs:
    input_lengths: torch.Tensor
    is_prefill: bool
    kv_cache_block_id_host: torch.Tensor
    prefix_lengths: torch.Tensor
    sequence_lengths: torch.Tensor
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def cache_store_inputs(self) -> PyCacheStoreInputs | None:
        ...
    @property
    def cu_seqlens(self) -> torch.Tensor:
        ...
    @property
    def dtype(self) -> TypeMeta:
        ...
    @property
    def kv_block_offset(self) -> int:
        ...
    @property
    def kv_cache_block_id_device(self) -> torch.Tensor:
        ...
    @property
    def padding_offset(self) -> torch.Tensor:
        ...
class PyCacheStoreInputs:
    def __init__(self) -> None:
        ...
class PyCaptureMetaData:
    def __init__(self) -> None:
        ...
class PyModelInitResources:
    def __init__(self) -> None:
        ...
    @property
    def kv_cache(self) -> KVCache | None:
        """
        kv cache
        """
class PyModelInputs:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, input_ids: torch.Tensor = ..., attention_inputs: PyAttentionInputs = ..., bert_embedding_inputs: BertEmbeddingInputs = ...) -> None:
        ...
    @property
    def attention_inputs(self) -> PyAttentionInputs:
        """
        Attention inputs structure
        """
    @attention_inputs.setter
    def attention_inputs(self, arg0: PyAttentionInputs) -> None:
        ...
    @property
    def bert_embedding_inputs(self) -> BertEmbeddingInputs:
        """
        BERT embedding inputs structure
        """
    @bert_embedding_inputs.setter
    def bert_embedding_inputs(self, arg0: BertEmbeddingInputs) -> None:
        ...
    @property
    def input_ids(self) -> torch.Tensor:
        """
        Input token IDs tensor
        """
    @input_ids.setter
    def input_ids(self, arg0: torch.Tensor) -> None:
        ...
class PyModelOutputs:
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor
        """
    @typing.overload
    def __init__(self, hidden_states: torch.Tensor, params_ptr: ParamsBase) -> None:
        """
        Initialize with hidden states tensor and params pointer
        """
    @typing.overload
    def __init__(self, hidden_states: torch.Tensor) -> None:
        """
        Initialize with hidden states tensor only (params_ptr defaults to nullptr)
        """
    @typing.overload
    def __init__(self, params_ptr: ParamsBase) -> None:
        """
        Initialize with params pointer only (hidden_states defaults to empty tensor)
        """
    @typing.overload
    def __init__(self, hidden_states: torch.Tensor, params_ptr: typing.Any) -> None:
        """
        Initialize with hidden states tensor and params pointer
        """
    @property
    def hidden_states(self) -> torch.Tensor:
        """
        Hidden states output tensor
        """
    @hidden_states.setter
    def hidden_states(self, arg0: torch.Tensor) -> None:
        ...
    @property
    def params_ptr(self) -> ParamsBase:
        """
        Parameters pointer
        """
    @params_ptr.setter
    def params_ptr(self, arg0: ParamsBase) -> None:
        ...
class TypeMeta:
    def __init__(self) -> None:
        ...
def get_device() -> DeviceExporter:
    ...
def init_device(params: libth_transformer_config.GptInitParameter) -> None:
    ...
