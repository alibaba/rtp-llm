import hashlib
import logging
from numbers import Real
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import RtpModule, _mark_loaded
from rtp_llm.models_py.new_models.model_base import select_fmha_impl_for_layer
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)
CUSTOM_WEIGHT_PREFIX = "__custom__."


def _as_iter(weights: Any) -> Iterator[Tuple[str, torch.Tensor]]:
    return iter(weights.items()) if isinstance(weights, dict) else iter(weights)


def _strip_known_prefix(name: str, model_prefix: str) -> str:
    prefix = model_prefix + "."
    return name[len(prefix) :] if name.startswith(prefix) else name


def _positive_config_int(config: Any, name: str) -> int:
    value = getattr(config, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"model_config.{name} must be a positive integer, got {value!r}"
        )
    return value


def _optional_nonnegative_config_int(config: Any, name: str) -> int:
    value = getattr(config, name, 0)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"model_config.{name} must be an integer")
    if value < 0:
        raise ValueError(f"model_config.{name} cannot be negative")
    return value


def _required_parameter(module: nn.Module, name: str) -> nn.Parameter:
    value = module._parameters.get(name)
    if not isinstance(value, nn.Parameter):
        raise RuntimeError(
            f"Required BERT parameter {type(module).__name__}.{name} is missing"
        )
    return value


def _is_missing_tensor(tensor: Any) -> bool:
    return tensor is None or (isinstance(tensor, torch.Tensor) and tensor.numel() == 0)


_ALLOWED_DROPPED_SUFFIXES = (
    "position_ids",
    "token_type_ids",
)

_ALLOWED_DROPPED_PREFIXES = (
    "classifier.",
    "cls.",
    "lm_head.",
    "pooler.",
)


def _is_allowed_dropped_weight(name: str, model_prefix: str) -> bool:
    stripped = _strip_known_prefix(name, model_prefix)
    return any(
        name == suffix or name.endswith(f".{suffix}")
        for suffix in _ALLOWED_DROPPED_SUFFIXES
    ) or any(stripped.startswith(prefix) for prefix in _ALLOWED_DROPPED_PREFIXES)


class BertLayerNorm(RtpModule):
    """LayerNorm whose checkpoint parameters live in the NewLoader model tree."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        params_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if not isinstance(eps, Real) or isinstance(eps, bool) or float(eps) <= 0:
            raise ValueError(f"LayerNorm eps must be positive, got {eps!r}")
        self.hidden_size = hidden_size
        self.eps = float(eps)
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=params_dtype), requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.zeros(hidden_size, dtype=params_dtype), requires_grad=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim == 0 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"BERT LayerNorm expected last dimension {self.hidden_size}, "
                f"got {tuple(hidden_states.shape)}"
            )
        return F.layer_norm(
            hidden_states,
            (self.hidden_size,),
            self.weight,
            self.bias,
            self.eps,
        )


class BertEmbeddings(RtpModule):
    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        type_vocab_size: int,
        hidden_size: int,
        layernorm_eps: float,
        params_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, dtype=params_dtype)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_size, dtype=params_dtype
        )
        self.token_type_embeddings: Optional[nn.Embedding]
        if type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                type_vocab_size, hidden_size, dtype=params_dtype
            )
        else:
            self.token_type_embeddings = None
        self.layernorm = BertLayerNorm(
            hidden_size, layernorm_eps, params_dtype=params_dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        input_embedding_scalar: float,
    ) -> torch.Tensor:
        if _is_missing_tensor(position_ids):
            raise ValueError("BERT position ids must be provided by the engine")
        if input_ids.shape != position_ids.shape:
            raise ValueError(
                "BERT input/position id shape mismatch: "
                f"{tuple(input_ids.shape)} vs {tuple(position_ids.shape)}"
            )
        if isinstance(input_embedding_scalar, bool) or not isinstance(
            input_embedding_scalar, Real
        ):
            raise TypeError("BERT input_embedding_scalar must be a real number")

        hidden_states = self.word_embeddings(input_ids)
        hidden_states = hidden_states * float(input_embedding_scalar)
        hidden_states = hidden_states + self.position_embeddings(position_ids)
        if self.token_type_embeddings is not None:
            if _is_missing_tensor(token_type_ids):
                token_type_ids = torch.zeros_like(input_ids)
            if input_ids.shape != token_type_ids.shape:
                raise ValueError(
                    "BERT input/token-type id shape mismatch: "
                    f"{tuple(input_ids.shape)} vs {tuple(token_type_ids.shape)}"
                )
            hidden_states = hidden_states + self.token_type_embeddings(token_type_ids)
        return self.layernorm(hidden_states)


class BertSelfAttention(RtpModule):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        layer_idx: int,
        tp_size: int,
        tp_rank: int,
        quant_config: QuantizationConfig,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            bias=True,
            params_dtype=params_dtype,
        )
        self.o_proj = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            bias=True,
            params_dtype=params_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache],
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        attention = fmha_impl.forward(qkv, kv_cache, self.layer_idx)
        attention = attention.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attention)


class BertMLP(RtpModule):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int,
        tp_rank: int,
        quant_config: QuantizationConfig,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.intermediate = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.intermediate",
            bias=True,
            params_dtype=params_dtype,
        )
        self.output = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.output",
            bias=True,
            params_dtype=params_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output(F.gelu(self.intermediate(hidden_states)))


class BertDecoderLayer(RtpModule):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
        layer_idx: int,
        tp_size: int,
        tp_rank: int,
        quant_config: QuantizationConfig,
        params_dtype: torch.dtype,
        layernorm_eps: float,
        prefix: str,
    ) -> None:
        super().__init__()
        self.self_attn = BertSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            layer_idx=layer_idx,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
            prefix=f"{prefix}.self_attn",
        )
        self.attention_layernorm = BertLayerNorm(
            hidden_size, layernorm_eps, params_dtype
        )
        self.mlp = BertMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
            prefix=f"{prefix}.mlp",
        )
        self.output_layernorm = BertLayerNorm(hidden_size, layernorm_eps, params_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache],
    ) -> torch.Tensor:
        attention = self.self_attn(hidden_states, fmha_impl, kv_cache)
        hidden_states = self.attention_layernorm(attention + hidden_states)
        output = self.mlp(hidden_states)
        return self.output_layernorm(output + hidden_states)


class _BertCustomParams(RtpModule):
    def __init__(self) -> None:
        super().__init__()
        self.names: Dict[str, str] = {}

    @staticmethod
    def _attr(name: str) -> str:
        return "custom_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]

    def load_weight(
        self, runtime_name: str, tensor: torch.Tensor, dtype: torch.dtype
    ) -> None:
        attr = self._attr(runtime_name)
        previous = self.names.get(attr)
        if previous is not None:
            raise RuntimeError(
                f"Duplicate or colliding BERT custom weight {runtime_name!r}; "
                f"existing={previous!r}"
            )
        value = tensor.detach().contiguous()
        if value.is_floating_point() and value.dtype != dtype:
            value = value.to(dtype)
        self.register_parameter(attr, nn.Parameter(value, requires_grad=False))
        self.names[attr] = runtime_name
        _mark_loaded(self, attr)


def _validate_attention_config(config: Any) -> Tuple[int, int, int]:
    hidden_size = _positive_config_int(config, "hidden_size")
    attn_config = getattr(config, "attn_config", None)
    if attn_config is None:
        raise ValueError("model_config.attn_config is required")
    head_num = _positive_config_int(attn_config, "head_num")
    kv_head_num = _positive_config_int(attn_config, "kv_head_num")
    size_per_head = _positive_config_int(attn_config, "size_per_head")
    if hidden_size != head_num * size_per_head:
        raise ValueError(
            "BERT hidden_size must equal head_num * size_per_head; "
            f"got {hidden_size} != {head_num} * {size_per_head}"
        )
    if kv_head_num != head_num:
        raise ValueError(
            "BERT newloader requires kv_head_num == head_num; "
            f"got {kv_head_num} != {head_num}"
        )
    return hidden_size, head_num, size_per_head


_EMBEDDING_MAPPING = {
    "embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
    "embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
    "embeddings.token_type_embeddings.weight": "embeddings.token_type_embeddings.weight",
    "embeddings.LayerNorm.weight": "embeddings.layernorm.weight",
    "embeddings.LayerNorm.bias": "embeddings.layernorm.bias",
    "embeddings.LayerNorm.gamma": "embeddings.layernorm.weight",
    "embeddings.LayerNorm.beta": "embeddings.layernorm.bias",
}

_LAYER_MAPPING = {
    "attention.self.query.weight": "self_attn.qkv_proj.q_proj.weight",
    "attention.self.query.bias": "self_attn.qkv_proj.q_proj.bias",
    "attention.self.key.weight": "self_attn.qkv_proj.k_proj.weight",
    "attention.self.key.bias": "self_attn.qkv_proj.k_proj.bias",
    "attention.self.value.weight": "self_attn.qkv_proj.v_proj.weight",
    "attention.self.value.bias": "self_attn.qkv_proj.v_proj.bias",
    "attention.output.dense.weight": "self_attn.o_proj.weight",
    "attention.output.dense.bias": "self_attn.o_proj.bias",
    "attention.output.LayerNorm.weight": "attention_layernorm.weight",
    "attention.output.LayerNorm.bias": "attention_layernorm.bias",
    "attention.output.LayerNorm.gamma": "attention_layernorm.weight",
    "attention.output.LayerNorm.beta": "attention_layernorm.bias",
    "intermediate.dense.weight": "mlp.intermediate.weight",
    "intermediate.dense.bias": "mlp.intermediate.bias",
    "output.dense.weight": "mlp.output.weight",
    "output.dense.bias": "mlp.output.bias",
    "output.LayerNorm.weight": "output_layernorm.weight",
    "output.LayerNorm.bias": "output_layernorm.bias",
    "output.LayerNorm.gamma": "output_layernorm.weight",
    "output.LayerNorm.beta": "output_layernorm.bias",
}


class _BertNewLoaderBase(GptModelBase):
    """BERT encoder implemented directly on NewLoader-owned parameters."""

    model_prefix = "bert"
    supports_custom_weight_mappings = True

    def __init__(self, model_config: Any, load_config: Any) -> None:
        parallelism_config = getattr(load_config, "parallelism_config", None)
        if parallelism_config is None:
            parallelism_config = ParallelismConfig()
            parallelism_config.tp_size = load_config.tp_size
            parallelism_config.tp_rank = load_config.tp_rank
            parallelism_config.ep_size = load_config.ep_size
            parallelism_config.ep_rank = load_config.ep_rank
            parallelism_config.world_size = load_config.tp_size
            parallelism_config.local_world_size = load_config.tp_size
        super().__init__(
            config=model_config,
            parallelism_config=parallelism_config,
            weight=None,
            max_generate_batch_size=0,
            fmha_config=getattr(load_config, "fmha_config", None),
            py_hw_kernel_config=getattr(
                getattr(load_config, "quant_config", None),
                "hw_kernel_config",
                None,
            ),
            device_resource_config=getattr(load_config, "device_resource_config", None),
        )
        if load_config.ep_size != 1:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not use expert parallelism; "
                "ep_size must be 1"
            )
        if (load_config.attn_tp_size, load_config.attn_tp_rank) != (
            load_config.tp_size,
            load_config.tp_rank,
        ):
            raise ValueError("BERT newloader does not support context parallelism")
        if (load_config.ffn_tp_size, load_config.ffn_tp_rank) != (
            load_config.tp_size,
            load_config.tp_rank,
        ):
            raise ValueError(
                "BERT newloader does not support independent FFN parallelism"
            )

        self.compute_dtype = getattr(load_config, "compute_dtype", torch.float16)
        if not isinstance(self.compute_dtype, torch.dtype):
            raise TypeError("load_config.compute_dtype must be a torch.dtype")
        quant_config = getattr(load_config, "quant_config", None)
        if quant_config is None:
            quant_config = QuantizationConfig("none")
        runtime_quant_type = getattr(quant_config, "quant_type", "none")
        configured_quantization = getattr(model_config, "quantization", "") or ""
        if not isinstance(configured_quantization, str):
            raise TypeError("model_config.quantization must be a string")
        if (
            getattr(model_config, "quant_config", None) is not None
            or configured_quantization.strip().lower() not in ("", "none")
            or str(runtime_quant_type).strip().lower() not in ("", "none")
        ):
            raise NotImplementedError(
                f"{self.__class__.__name__} newloader supports unquantized "
                "checkpoints only"
            )

        hidden_size, num_heads, head_dim = _validate_attention_config(model_config)
        intermediate_size = _positive_config_int(model_config, "inter_size")
        num_layers = _positive_config_int(model_config, "num_layers")
        vocab_size = _positive_config_int(model_config, "vocab_size")
        max_seq_len = _positive_config_int(model_config, "max_seq_len")
        type_vocab_size = _optional_nonnegative_config_int(
            model_config, "type_vocab_size"
        )
        layernorm_eps = getattr(model_config, "layernorm_eps", None)
        if (
            isinstance(layernorm_eps, bool)
            or not isinstance(layernorm_eps, Real)
            or float(layernorm_eps) <= 0
        ):
            raise ValueError(
                "model_config.layernorm_eps must be a positive real number"
            )

        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            max_position_embeddings=max_seq_len,
            type_vocab_size=type_vocab_size,
            hidden_size=hidden_size,
            layernorm_eps=float(layernorm_eps),
            params_dtype=self.compute_dtype,
        )
        self.layers = nn.ModuleList(
            [
                BertDecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    layer_idx=layer_idx,
                    tp_size=load_config.tp_size,
                    tp_rank=load_config.tp_rank,
                    quant_config=quant_config,
                    params_dtype=self.compute_dtype,
                    layernorm_eps=float(layernorm_eps),
                    prefix=f"layers.{layer_idx}",
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.custom_params = _BertCustomParams()
        self._custom_checkpoint_to_runtime = {
            checkpoint_name: runtime_name
            for runtime_name, checkpoint_name in load_config.custom_weight_mappings
        }
        self._expected_custom_weight_names = frozenset(
            self._custom_checkpoint_to_runtime.values()
        )
        self._loaded_checkpoint_targets = set()

    def _resolve_custom_weight_name(self, name: str, stripped: str) -> Optional[str]:
        matches = []
        for candidate in dict.fromkeys((name, stripped)):
            runtime_name = self._custom_checkpoint_to_runtime.get(candidate)
            if runtime_name is not None:
                matches.append(runtime_name)
        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous custom BERT weight {name!r}: matches={matches}"
            )
        return matches[0] if matches else None

    def _map_checkpoint_name(self, stripped: str) -> Optional[str]:
        target = _EMBEDDING_MAPPING.get(stripped)
        if target is not None:
            return target
        prefix = "encoder.layer."
        if not stripped.startswith(prefix):
            return None
        remainder = stripped[len(prefix) :]
        if "." not in remainder:
            return None
        layer_text, suffix = remainder.split(".", 1)
        if not layer_text.isdigit():
            return None
        layer_idx = int(layer_text)
        if not 0 <= layer_idx < len(self.layers):
            return None
        mapped_suffix = _LAYER_MAPPING.get(suffix)
        if mapped_suffix is None:
            return None
        return f"layers.{layer_idx}.{mapped_suffix}"

    def _load_mapped_weight(
        self, target: str, tensor: torch.Tensor, checkpoint_name: str
    ) -> None:
        if target in self._loaded_checkpoint_targets:
            raise RuntimeError(
                f"Duplicate BERT checkpoint tensor for {target!r}: "
                f"latest source={checkpoint_name!r}"
            )
        if not self._dispatch(self, target, tensor):
            raise RuntimeError(
                f"BERT checkpoint mapping target {target!r} for "
                f"{checkpoint_name!r} is unavailable"
            )
        self._loaded_checkpoint_targets.add(target)

    def load_weights(self, weights: Any) -> None:
        if self._loaded_checkpoint_targets or self.custom_params.names:
            raise RuntimeError(
                f"{self.__class__.__name__}.load_weights() may only run once"
            )
        loaded = 0
        custom_loaded = 0
        dropped = []
        for name, tensor in _as_iter(weights):
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise TypeError("Weights must be (str, torch.Tensor) pairs")
            stripped = _strip_known_prefix(name, self.model_prefix)
            runtime_name = self._resolve_custom_weight_name(name, stripped)
            if runtime_name is not None:
                if not tensor.is_floating_point():
                    raise TypeError(
                        f"Custom BERT weight {name!r} must be floating point"
                    )
                self.custom_params.load_weight(runtime_name, tensor, self.compute_dtype)
                custom_loaded += 1
                continue

            target = self._map_checkpoint_name(stripped)
            if target is None:
                dropped.append(name)
                continue
            self._load_mapped_weight(target, tensor, name)
            loaded += 1

        token_type = self.embeddings.token_type_embeddings
        token_type_target = "embeddings.token_type_embeddings.weight"
        if (
            token_type is not None
            and token_type_target not in self._loaded_checkpoint_targets
        ):
            self._load_mapped_weight(
                token_type_target,
                torch.zeros_like(token_type.weight),
                "<BERT default token-type embedding>",
            )

        missing_custom = self._expected_custom_weight_names - set(
            self.custom_params.names.values()
        )
        if missing_custom:
            raise RuntimeError(
                f"{self.__class__.__name__} is missing required custom weights: "
                f"{sorted(missing_custom)}"
            )
        unexpected = [
            name
            for name in dropped
            if not _is_allowed_dropped_weight(name, self.model_prefix)
        ]
        if unexpected:
            sample = unexpected[:10]
            suffix = (
                f" (+{len(unexpected) - len(sample)} more)"
                if len(unexpected) > 10
                else ""
            )
            raise RuntimeError(
                f"{self.__class__.__name__} dropped unexpected checkpoint tensors: "
                f"{sample}{suffix}"
            )
        logger.info(
            "%s streamed checkpoint tensors into the NewLoader model tree: "
            "model_tensors=%d custom_tensors=%d known_auxiliary=%d",
            self.__class__.__name__,
            loaded,
            custom_loaded,
            len(dropped),
        )

    def runtime_weight_view(self) -> Dict[str, torch.Tensor]:
        result = {
            W.embedding: self.embeddings.word_embeddings.weight,
            W.positional_embedding: self.embeddings.position_embeddings.weight,
            W.pre_decoder_ln_gamma: self.embeddings.layernorm.weight,
            W.pre_decoder_ln_beta: self.embeddings.layernorm.bias,
        }
        if self.embeddings.token_type_embeddings is not None:
            result[W.token_type_embedding] = (
                self.embeddings.token_type_embeddings.weight
            )
        for attr, runtime_name in self.custom_params.names.items():
            result[runtime_name] = _required_parameter(self.custom_params, attr)
        return result

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        bert_inputs = inputs.bert_embedding_inputs
        hidden_states = self.embeddings(
            inputs.input_ids,
            bert_inputs.combo_position_ids,
            bert_inputs.combo_tokens_type_ids,
            bert_inputs.input_embedding_scalar,
        )
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        for layer_idx, layer in enumerate(self.layers):
            layer_fmha_impl = select_fmha_impl_for_layer(
                fmha_impl, self.kv_cache, layer_idx
            )
            hidden_states = layer(
                hidden_states,
                layer_fmha_impl,
                self.kv_cache.get_layer_cache(layer_idx) if self.kv_cache else None,
            )
        return PyModelOutputs(hidden_states)


class BertForEmbedding(_BertNewLoaderBase):
    model_prefix = "bert"


class RobertaForEmbedding(_BertNewLoaderBase):
    model_prefix = "roberta"
