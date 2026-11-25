"""The production BERT encoder owns its checkpoint parameters directly."""

import json
import re
import weakref
from pathlib import Path

import torch
from torch import nn

from rtp_llm.models_py.downstream_heads import (
    ReplicatedHandlerHeads,
    check_encoder_handler_alias,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.read_spec import ReadSpec, unwrap_checkpoint_tensor
from rtp_llm.models_py.weight_mapper import WeightsMapper

from .checkpoint_metadata import (
    checkpoint_input_vocab_size,
    publish_input_vocab,
    static_int8_input_major,
)
from .parameters import BertEmbeddings, BertLayer, PackedBertAttention


class _BertMapper(WeightsMapper):
    _layer = re.compile(r"^(encoder\.layer\.\d+)\.(.+)$")
    _children = {
        "attention.self.query": "qkv.q_proj",
        "attention.self.key": "qkv.k_proj",
        "attention.self.value": "qkv.v_proj",
        "attention.output.dense": "attn_output",
        "attention.output.LayerNorm": "attn_norm",
        "intermediate.dense": "intermediate",
        "output.dense": "output",
        "output.LayerNorm": "output_norm",
    }

    def __init__(self, prefix, static_int8=False):
        super().__init__()
        self.prefix = prefix + "."
        self.static_int8 = static_int8

    def map_name(self, name):
        name = name.removeprefix(self.prefix)
        if self.static_int8:
            # HF static W8A8 names the four calibrated activations with an
            # underscore. Q, K and V share one activation, not three scales.
            if name.endswith(".attention.self.qkv_input_scale"):
                return (
                    name.removesuffix(".attention.self.qkv_input_scale")
                    + ".qkv.input_scale"
                )
            if name.endswith("_input_scale"):
                name = name.removesuffix("_input_scale") + ".input_scale"
        stem, _, leaf = name.rpartition(".")
        if stem.endswith(".LayerNorm") and leaf in ("beta", "gamma"):
            leaf = {"beta": "bias", "gamma": "weight"}[leaf]
            name = stem + "." + leaf
        match = self._layer.match(stem)
        if match and match[2] in self._children:
            if self.static_int8:
                leaf = {"qweight": "weight", "scales": "weight_scale"}.get(leaf, leaf)
            return ".".join((match[1], self._children[match[2]], leaf))
        return name


class _OptionalPooler(RtpModule):
    """BertModel has a pooler; BertForMaskedLM may legitimately omit it."""

    def __init__(self, h, dtype):
        super().__init__()
        self.width = h
        self.register_buffer("_anchor", torch.empty(0, dtype=dtype), persistent=False)
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def load_weights(self, weights):
        for name, value in weights.items():
            source, _ = unwrap_checkpoint_tensor(value)
            key = name.removeprefix("dense.")
            shape = {"weight": (self.width, self.width), "bias": (self.width,)}.get(key)
            if shape is None or source.shape != shape or not source.is_floating_point():
                raise ValueError(
                    f"Invalid BERT pooler tensor {name} {tuple(source.shape)}"
                )
            if self._parameters[key] is not None:
                raise ValueError(f"Duplicate BERT pooler tensor {name}")
            self._parameters[key] = nn.Parameter(
                source.detach().to(self._anchor, copy=True), requires_grad=False
            )
            self._mark_weight_loaded(key)

    def validate_weights_loaded(self, loaded_tensor_ids=None):
        if (self.weight is None) != (self.bias is None):
            raise ValueError("Incomplete BERT pooler projection")
        super().validate_weights_loaded(loaded_tensor_ids)


class BertEncoderOwner(GptModelBase, RtpModule):
    checkpoint_prefix = "bert"
    _embedding_exports = {
        "embedding": "embeddings.word_embeddings.weight",
        "position_encoding.weight": "embeddings.position_embeddings.weight",
        "token_type_embedding.weight": "embeddings.token_type_embeddings.weight",
        "pre_decoder_layernorm.gamma": "embeddings.LayerNorm.weight",
        "pre_decoder_layernorm.bias": "embeddings.LayerNorm.bias",
    }
    supports_custom_weight_mappings = True
    supports_ft_handler_heads = True

    @classmethod
    def from_ft_checkpoint_header(cls, config, load, checkpoint):
        rows, width = checkpoint.read_rank_slice(
            "global.embedding", checkpoint.topology.tp_rank
        )
        if rows <= 0 or width * load.attn_tp_size != config.hidden_size:
            raise ValueError("Invalid rank-local BERT FT embedding dimensions")
        explicit = getattr(config, "input_vocab_size", 0)
        if explicit and explicit != rows:
            raise ValueError("BERT FT embedding disagrees with explicit input_vocab_size")
        return cls(config, load, ft_input_rows=rows)

    resident_after_post_load = True

    def resident_constructor_state(self):
        return {
            "input_rows": self.checkpoint_input_vocab_size,
            "input_major": self._checkpoint_input_major,
            "has_pooler": self.pooler is not None and self.pooler.weight is not None,
            "handlers": self.handler_heads.resident_shapes(),
        }

    def finish_resident_checkpoint_restore(self):
        from rtp_llm.models_py.resident_checkpoint import restore_final_linears

        restore_final_linears(self)

    def __init__(self, config, load, *, ft_input_rows=None):
        super().__init__(
            config,
            load.parallelism_config,
            None,
            max_generate_batch_size=load.max_generate_batch_size,
            fmha_config=load.fmha_config,
            py_hw_kernel_config=load.py_hw_kernel_config,
        )
        static_int8 = (
            getattr(load.quant_config, "quant_type", None) == "pertensor_quant"
        )
        self.WEIGHTS_MAPPER = _BertMapper(self.checkpoint_prefix, static_int8)
        hf = json.loads((Path(config.ckpt_path) / "config.json").read_text())
        if hf.get("position_embedding_type", "absolute") != "absolute":
            raise ValueError("The BERT serving encoder expects absolute positions")
        position_rows = int(hf.get("max_position_embeddings", 512))
        # EmbeddingExecutor generates IDs in [0, max_seq_len), including
        # RoBERTa's padding offset. Keep all learned rows for shorter caps.
        if config.max_seq_len > position_rows:
            raise ValueError(
                f"BERT max_seq_len={config.max_seq_len} exceeds the "
                f"{position_rows}-row position embedding table"
            )
        self.hf_depth = int(hf["num_hidden_layers"])
        owned = tuple(self.layer_ids())
        self.first = 0 in owned
        self.last = config.num_layers - 1 in owned
        self.owned = frozenset(owned)
        resident = getattr(load, "resident_construction", None)
        if resident is not None and ft_input_rows is not None and ft_input_rows != resident["input_rows"]:
            raise ValueError("Resident BERT input vocabulary disagrees with FT header")
        self.checkpoint_input_vocab_size = (resident["input_rows"] if resident is not None else ft_input_rows) or checkpoint_input_vocab_size(
            config, self.checkpoint_prefix, tp_rank=load.tp_rank, tp_size=load.tp_size
        )
        self.embeddings = (
            BertEmbeddings(
                self.checkpoint_input_vocab_size,
                position_rows,
                int(hf.get("type_vocab_size", 2)),
                config.hidden_size,
                config.layernorm_eps,
                load.attn_tp_size,
                load.attn_tp_rank,
                load.compute_dtype,
            )
            if self.first
            else None
        )
        self.encoder = nn.Module()
        input_major = resident["input_major"] if resident is not None else (
            static_int8
            and ft_input_rows is None
            and static_int8_input_major(
                config,
                self.checkpoint_prefix,
                owned,
                tp_rank=load.tp_rank,
                tp_size=load.tp_size,
            )
        )
        self._checkpoint_input_major = input_major
        self.encoder.layer = nn.ModuleDict(
            {str(i): BertLayer(config, load, i, input_major=input_major) for i in owned}
        )
        if static_int8:
            for i in owned:
                current = self.encoder.layer[str(i)]
                current.attn_norm._scale_consumer = weakref.ref(current.intermediate)
                if i == 0:
                    self.embeddings.LayerNorm._scale_consumer = weakref.ref(current.qkv)
                elif str(i - 1) in self.encoder.layer:
                    self.encoder.layer[str(i - 1)].output_norm._scale_consumer = (
                        weakref.ref(current.qkv)
                    )
        self.pooler = (
            _OptionalPooler(config.hidden_size, load.compute_dtype)
            if self.last
            else None
        )
        self.local_heads = config.attn_config.head_num // load.attn_tp_size
        self.head_dim = config.attn_config.size_per_head
        mappings = tuple(load.custom_weight_mappings)
        for role, source, *_ in mappings:
            check_encoder_handler_alias(
                role, source, self.WEIGHTS_MAPPER.map_name(source)
            )
        self._ft_handler_roles = tuple(mapping[0] for mapping in mappings)
        self.all_handler_sources = {
            self.WEIGHTS_MAPPER.map_name(mapping[1]) for mapping in mappings
        }
        mappings = mappings if self.last else ()
        args = (mappings, self.WEIGHTS_MAPPER, config.hidden_size, load.compute_dtype)
        self.handler_heads = (
            ReplicatedHandlerHeads(*args, projection_sources=self.all_handler_sources)
            if mappings
            else ReplicatedHandlerHeads(*args)
        )

        if resident is not None:
            self.handler_heads.reserve_resident_parameters(resident["handlers"])
            if resident["has_pooler"]:
                if self.pooler is None:
                    raise ValueError("Resident pooler belongs to another PP stage")
                width = self.pooler.width
                for name, shape in (("weight", (width, width)), ("bias", (width,))):
                    self.pooler.register_parameter(
                        name, nn.Parameter(self.pooler._anchor.new_empty(shape), requires_grad=False)
                    )

    def finalize_runtime_input_vocab(self, config, tokenizer):
        publish_input_vocab(config, tokenizer, self.checkpoint_input_vocab_size)

    def load_ft_checkpoint(self, checkpoint):
        from .ft_checkpoint import load_bert_ft_checkpoint

        load_bert_ft_checkpoint(self, checkpoint)

    def checkpoint_weight_name_filter(self):
        def keep(source):
            name = self.WEIGHTS_MAPPER.map_name(source)
            if name in self.all_handler_sources:
                return self.last
            if name in ("embeddings.position_ids", "embeddings.token_type_ids"):
                return False  # HF version-dependent, nonpersistent integer buffers.
            if name.startswith("embeddings."):
                return self.first
            if name.startswith("pooler."):
                return self.last
            if name.startswith(("cls.", "lm_head.", "classifier.")):
                # Training-only MLM/NSP/classification heads not requested by this task.
                return False
            m = re.match(r"^encoder\.layer\.(\d+)\.", name)
            return m is None or int(m[1]) >= self.hf_depth or int(m[1]) in self.owned

        return keep

    def resolve_checkpoint_load_plan(self, name, shape):
        if name in self.handler_heads.sources:
            return None, ReadSpec.full(name, tuple(shape))
        return super().resolve_checkpoint_load_plan(name, shape)

    def load_weights(self, weights):
        def owned():
            stream = weights.items() if isinstance(weights, dict) else weights
            for name, value in self.WEIGHTS_MAPPER.apply(stream):
                if name in self.handler_heads.sources:
                    self.handler_heads.capture(name, value)
                else:
                    yield name, value

        return super().load_weights(owned())

    def validate_weights_loaded(self, loaded_tensor_ids=None):
        self.handler_heads.validate()
        super().validate_weights_loaded(loaded_tensor_ids)

    def runtime_weight_view(self):
        exports = self.handler_heads.exports()
        if self.embeddings is not None:
            exports.update((alias, self.get_parameter(path)) for alias, path in self._embedding_exports.items())
        return exports

    def online_checkpoint_name(self, name):
        return self._embedding_exports.get(name, name)

    def is_nonlocal_online_weight(self, name):
        # Requests are broadcast to every pipeline stage. Skip only known
        # embedding leaves on non-owners; unknown names must still fail.
        return not self.first and self.WEIGHTS_MAPPER.map_name(name) in self._embedding_exports.values()

    def _input_embeddings(self, inputs):
        """First-stage embedding boundary for specialized encoder modalities."""
        return self.embeddings(inputs.input_ids, inputs.bert_embedding_inputs)

    def forward_hidden(self, inputs, fmha_impl=None):
        hidden = (
            self._input_embeddings(inputs)
            if self.first
            else inputs.pp_intermediates["hidden_states"]
        )
        if fmha_impl is None:
            if hidden.device.type == "cpu":
                attn = inputs.attention_inputs
                lengths = getattr(attn, "input_lengths", None)
                if lengths is None or not len(lengths):
                    lengths = attn.input_lengths_host
                fmha_impl = PackedBertAttention(
                    lengths, self.local_heads, self.head_dim
                )
            else:
                fmha_impl = self.prepare_fmha_impl(inputs)
        from rtp_llm.models_py.new_models.model_base import select_fmha_impl_for_layer

        for i, layer in enumerate(self.encoder.layer.values()):
            impl = select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i)
            hidden = layer(
                hidden,
                impl,
                self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        return hidden

    def forward(self, inputs, fmha_impl=None):
        hidden = self.forward_hidden(inputs, fmha_impl)
        from rtp_llm.ops.compute_ops import PyModelOutputs

        result = PyModelOutputs(hidden)
        if not self.last:
            result.pp_intermediates = {"hidden_states": hidden}
        return result
