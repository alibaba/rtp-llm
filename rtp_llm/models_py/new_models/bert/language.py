import hashlib
import logging
from typing import Any, Dict, Iterator, Tuple

import torch
import torch.nn as nn

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


def _as_iter(weights: Any) -> Iterator[Tuple[str, torch.Tensor]]:
    if isinstance(weights, dict):
        return iter(weights.items())
    return iter(weights)


def _strip_known_prefix(name: str, model_prefix: str) -> str:
    for prefix in (model_prefix + ".", "bert.", "roberta."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _float_to_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.is_floating_point():
        return tensor.to(dtype)
    return tensor


def _set_param(module: nn.Module, attr: str, tensor: torch.Tensor, dtype: torch.dtype):
    value = _float_to_dtype(tensor.detach().contiguous(), dtype)
    current = getattr(module, attr, None)
    if isinstance(current, nn.Parameter):
        if current.shape != value.shape:
            raise ValueError(f"Shape mismatch for {attr}: {value.shape} vs {current.shape}")
        current.data.copy_(value)
    else:
        module.register_parameter(attr, nn.Parameter(value, requires_grad=False))


def _required_param(module: nn.Module, attr: str) -> nn.Parameter:
    value = getattr(module, attr, None)
    if not isinstance(value, nn.Parameter):
        raise KeyError(attr)
    return value


def _optional_param(module: nn.Module, attr: str):
    value = getattr(module, attr, None)
    return value if isinstance(value, nn.Parameter) else None


def _weight_data(param: nn.Parameter, dtype: torch.dtype) -> torch.Tensor:
    return _float_to_dtype(param.data, dtype)


_ALLOWED_DROPPED_SUFFIXES = (
    "position_ids",
    "token_type_ids",
)


def _is_allowed_dropped_weight(name: str) -> bool:
    return any(name == suffix or name.endswith(f".{suffix}") for suffix in _ALLOWED_DROPPED_SUFFIXES)


class _BertEmbeddingParams(nn.Module):
    def load_weight(self, name: str, tensor: torch.Tensor, dtype: torch.dtype) -> bool:
        mapping = {
            "word_embeddings.weight": "word_embeddings_weight",
            "position_embeddings.weight": "position_embeddings_weight",
            "token_type_embeddings.weight": "token_type_embeddings_weight",
            "LayerNorm.weight": "layernorm_weight",
            "LayerNorm.bias": "layernorm_bias",
            "LayerNorm.gamma": "layernorm_weight",
            "LayerNorm.beta": "layernorm_bias",
        }
        attr = mapping.get(name)
        if attr is None:
            return False
        _set_param(self, attr, tensor, dtype)
        return True


class _BertLayerParams(nn.Module):
    _MAPPING = {
        "attention.self.query.weight": "attention_query_weight",
        "attention.self.query.bias": "attention_query_bias",
        "attention.self.key.weight": "attention_key_weight",
        "attention.self.key.bias": "attention_key_bias",
        "attention.self.value.weight": "attention_value_weight",
        "attention.self.value.bias": "attention_value_bias",
        "attention.output.dense.weight": "attention_output_dense_weight",
        "attention.output.dense.bias": "attention_output_dense_bias",
        "attention.output.LayerNorm.weight": "attention_output_layernorm_weight",
        "attention.output.LayerNorm.bias": "attention_output_layernorm_bias",
        "attention.output.LayerNorm.gamma": "attention_output_layernorm_weight",
        "attention.output.LayerNorm.beta": "attention_output_layernorm_bias",
        "intermediate.dense.weight": "intermediate_dense_weight",
        "intermediate.dense.bias": "intermediate_dense_bias",
        "output.dense.weight": "output_dense_weight",
        "output.dense.bias": "output_dense_bias",
        "output.LayerNorm.weight": "output_layernorm_weight",
        "output.LayerNorm.bias": "output_layernorm_bias",
        "output.LayerNorm.gamma": "output_layernorm_weight",
        "output.LayerNorm.beta": "output_layernorm_bias",
    }

    def load_weight(self, name: str, tensor: torch.Tensor, dtype: torch.dtype) -> bool:
        attr = self._MAPPING.get(name)
        if attr is None:
            return False
        _set_param(self, attr, tensor, dtype)
        return True


class _BertCustomParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.names: Dict[str, str] = {}

    @staticmethod
    def _attr(name: str) -> str:
        return "custom_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]

    def load_weight(self, name: str, tensor: torch.Tensor, dtype: torch.dtype):
        attr = self._attr(name)
        self.names[attr] = name
        _set_param(self, attr, tensor, dtype)


class _BertNewLoaderBase(nn.Module):
    """New-loader wrapper for legacy BERT/Roberta PyModel.

    The new loader creates this object first, then calls load_weights(weights_iter).
    load_weights stores checkpoint tensors as parameters on this PyModel.  The
    legacy BertModel still needs W.* layout for kernels, so we rebuild a small
    ModelWeights view from the PyModel parameters after loading.
    """

    model_prefix = "bert"

    def __init__(self, model_config, load_config):
        super().__init__()
        self.config = model_config
        self.load_config = load_config
        self.compute_dtype = getattr(load_config, "compute_dtype", torch.float16)
        self.parallelism_config = getattr(load_config, "parallelism_config", None)
        if self.parallelism_config is None:
            self.parallelism_config = ParallelismConfig()
            self.parallelism_config.tp_size = getattr(load_config, "tp_size", 1)
            self.parallelism_config.tp_rank = getattr(load_config, "tp_rank", 0)
            self.parallelism_config.ep_size = getattr(load_config, "ep_size", 1)
            self.parallelism_config.ep_rank = getattr(load_config, "ep_rank", 0)
            self.parallelism_config.world_size = max(self.parallelism_config.tp_size, 1)
            self.parallelism_config.local_world_size = self.parallelism_config.world_size
        if int(getattr(self.parallelism_config, "tp_size", 1) or 1) != 1:
            raise NotImplementedError(
                f"{self.__class__.__name__} newloader does not support tensor parallel "
                "loading yet; tp_size > 1 would require rank-local BERT weight "
                "partitioning before rebuilding ModelWeights"
            )
        self.model = None
        self.weights = None
        self.embeddings = _BertEmbeddingParams()
        self.layers = nn.ModuleList(
            [_BertLayerParams() for _ in range(int(getattr(self.config, "num_layers")))]
        )
        self.custom_params = _BertCustomParams()

    def _embedding_param(self, attr: str) -> nn.Parameter:
        return _required_param(self.embeddings, attr)

    def _optional_embedding_param(self, attr: str):
        return _optional_param(self.embeddings, attr)

    def _layer_param(self, layer_idx: int, attr: str) -> nn.Parameter:
        return _required_param(self.layers[layer_idx], attr)

    def _create_model_weights(self) -> ModelWeights:
        num_layers = int(getattr(self.config, "num_layers"))
        weights = ModelWeights(num_layers, "cpu", self.compute_dtype)

        def put_global(w_name: str, attr: str, *, optional: bool = False):
            param = self._optional_embedding_param(attr) if optional else self._embedding_param(attr)
            if param is not None:
                weights.set_global_weight(w_name, _weight_data(param, self.compute_dtype).contiguous())

        put_global(W.embedding, "word_embeddings_weight")
        put_global(W.positional_embedding, "position_embeddings_weight")
        put_global(W.token_type_embedding, "token_type_embeddings_weight", optional=True)
        if W.token_type_embedding not in weights.global_weights:
            type_vocab_size = int(getattr(self.config, "type_vocab_size", 0) or 0)
            if type_vocab_size > 0:
                hidden_size = int(getattr(self.config, "hidden_size"))
                embedding_weight = weights.get_global_weight(W.embedding)
                weights.set_global_weight(
                    W.token_type_embedding,
                    embedding_weight.new_zeros((type_vocab_size, hidden_size)),
                )
        put_global(W.pre_decoder_ln_gamma, "layernorm_weight")
        put_global(W.pre_decoder_ln_beta, "layernorm_bias")

        for i in range(num_layers):
            q_w = _weight_data(self._layer_param(i, "attention_query_weight"), self.compute_dtype)
            k_w = _weight_data(self._layer_param(i, "attention_key_weight"), self.compute_dtype)
            v_w = _weight_data(self._layer_param(i, "attention_value_weight"), self.compute_dtype)
            q_b = _weight_data(self._layer_param(i, "attention_query_bias"), self.compute_dtype)
            k_b = _weight_data(self._layer_param(i, "attention_key_bias"), self.compute_dtype)
            v_b = _weight_data(self._layer_param(i, "attention_value_bias"), self.compute_dtype)

            weights.set_layer_weight(
                i,
                W.attn_qkv_w,
                torch.cat([q_w.t(), k_w.t(), v_w.t()], dim=1)
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.attn_qkv_b,
                torch.cat([q_b, k_b, v_b], dim=0).contiguous().to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.attn_o_w,
                _weight_data(self._layer_param(i, "attention_output_dense_weight"), self.compute_dtype)
                .t()
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.attn_o_b,
                _weight_data(self._layer_param(i, "attention_output_dense_bias"), self.compute_dtype)
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.post_ln_gamma,
                _weight_data(self._layer_param(i, "attention_output_layernorm_weight"), self.compute_dtype)
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.post_ln_beta,
                _weight_data(self._layer_param(i, "attention_output_layernorm_bias"), self.compute_dtype)
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.ffn_w3,
                _weight_data(self._layer_param(i, "intermediate_dense_weight"), self.compute_dtype)
                .t()
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.ffn_b3,
                _weight_data(self._layer_param(i, "intermediate_dense_bias"), self.compute_dtype)
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.ffn_w2,
                _weight_data(self._layer_param(i, "output_dense_weight"), self.compute_dtype)
                .t()
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.ffn_b2,
                _weight_data(self._layer_param(i, "output_dense_bias"), self.compute_dtype)
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.post_ffn_ln_gamma,
                _weight_data(self._layer_param(i, "output_layernorm_weight"), self.compute_dtype)
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.post_ffn_ln_beta,
                _weight_data(self._layer_param(i, "output_layernorm_bias"), self.compute_dtype)
                .contiguous()
                .to(self.compute_dtype),
            )

        return weights

    def _add_custom_weights(self, model_weights: ModelWeights):
        for attr, name in self.custom_params.names.items():
            param = _required_param(self.custom_params, attr)
            model_weights.set_global_weight(
                CustomAtomicWeight.prefix + name,
                _weight_data(param, self.compute_dtype).contiguous(),
            )

    def _build_inner_model(self):
        from rtp_llm.models_py.model_desc.bert import BertModel

        self.model = BertModel(
            self.config,
            self.parallelism_config,
            self.weights,
            max_generate_batch_size=int(getattr(self.config, "max_generate_batch_size", 0) or 0),
            quant_config=getattr(self.config, "quant_config", None),
            fmha_config=getattr(self.load_config, "fmha_config", None),
            py_hw_kernel_config=getattr(self.load_config, "hw_kernel_config", None),
            device_resource_config=getattr(self.load_config, "device_resource_config", None),
        )

    def load_weights(self, weights):
        loaded = 0
        custom_loaded = 0
        dropped = []
        self.custom_params.names.clear()
        for name, tensor in _as_iter(weights):
            stripped = _strip_known_prefix(name, self.model_prefix)
            if stripped.startswith("embeddings."):
                if self.embeddings.load_weight(stripped[len("embeddings.") :], tensor, self.compute_dtype):
                    loaded += 1
                else:
                    dropped.append(name)
                continue
            if stripped.startswith("encoder.layer."):
                parts = stripped.split(".", 3)
                if len(parts) == 4 and parts[2].isdigit():
                    layer_idx = int(parts[2])
                    if 0 <= layer_idx < len(self.layers) and self.layers[layer_idx].load_weight(
                        parts[3], tensor, self.compute_dtype
                    ):
                        loaded += 1
                    else:
                        dropped.append(name)
                else:
                    dropped.append(name)
                continue
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                custom_name = name if name.startswith("bert.pooler.") else stripped
                self.custom_params.load_weight(custom_name, tensor, self.compute_dtype)
                custom_loaded += 1
            else:
                dropped.append(name)
        unexpected = [name for name in dropped if not _is_allowed_dropped_weight(name)]
        if unexpected:
            sample = unexpected[:10]
            more = f" (+{len(unexpected) - len(sample)} more)" if len(unexpected) > len(sample) else ""
            raise RuntimeError(
                f"{self.__class__.__name__} dropped unexpected checkpoint tensors: {sample}{more}"
            )
        if dropped:
            logger.info(
                "%s ignored %d known non-persistent BERT tensor(s): %s",
                self.__class__.__name__,
                len(dropped),
                dropped[:10],
            )
        self._loaded_tensor_count = loaded
        self._custom_tensor_count = custom_loaded
        self._dropped_tensor_count = len(dropped)
        self.weights = None
        self.model = None
        logger.info(
            "%s newloader streamed BERT-style weights into PyModel params: tensors=%d custom_tensors=%d dropped=%d",
            self.__class__.__name__,
            loaded,
            custom_loaded,
            len(dropped),
        )

    def process_weights_after_loading(self):
        self.weights = self._create_model_weights()
        self._add_custom_weights(self.weights)
        self._build_inner_model()

    def _apply(self, fn):
        super()._apply(fn)
        if self.weights is not None:
            self.weights = self._create_model_weights()
            self._add_custom_weights(self.weights)
            if self.model is not None:
                self._build_inner_model()
        return self

    def initialize(self, init_resource):
        if self.model is None:
            raise RuntimeError("BERT newloader model is not loaded")
        return self.model.initialize(init_resource)

    def fill_params(self, *args, **kwargs):
        return self.model.fill_params(*args, **kwargs)

    def prepare_fmha_impl(self, *args, **kwargs):
        if self.model is None:
            raise RuntimeError("BERT newloader model is not loaded")
        return self.model.prepare_fmha_impl(*args, **kwargs)

    @staticmethod
    def _is_missing_tensor(tensor) -> bool:
        return tensor is None or (hasattr(tensor, "numel") and tensor.numel() == 0)

    def _fill_bert_embedding_inputs(self, inputs: PyModelInputs):
        bert_inputs = inputs.bert_embedding_inputs
        if self._is_missing_tensor(bert_inputs.position_encoding):
            bert_inputs.position_encoding = self.weights.get_global_weight(W.positional_embedding)
        if self._is_missing_tensor(bert_inputs.token_type_embedding):
            bert_inputs.token_type_embedding = self.weights.get_global_weight(W.token_type_embedding)
        return bert_inputs

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        if self.model is None:
            raise RuntimeError("BERT newloader model is not loaded")
        self._fill_bert_embedding_inputs(inputs)
        return self.model(inputs, fmha_impl)


class BertForEmbedding(_BertNewLoaderBase):
    model_prefix = "bert"


class RobertaForEmbedding(_BertNewLoaderBase):
    model_prefix = "roberta"
