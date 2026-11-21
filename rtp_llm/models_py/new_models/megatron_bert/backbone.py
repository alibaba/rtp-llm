"""Megatron-BERT checkpoint owners and packed, bidirectional encoder.

Megatron puts both layer norms *before* their sublayers and applies a
separate final encoder norm. Ordinary BERT weights/forward are not aliases.
"""

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.downstream_heads import unpack_handler_mapping
from rtp_llm.models_py.module_base import RtpModule, _mark_loaded
from rtp_llm.models_py.read_spec import make_sharded_read_spec, unwrap_checkpoint_tensor


def _table(count, width, dtype):
    return nn.Embedding.from_pretrained(torch.empty(count, width, dtype=dtype))


class _Projection(RtpModule):
    """A bias-bearing HF linear. Only the producing rank's interval is read."""

    def __init__(self, inputs, outputs, dim, tp, rank, dtype):
        super().__init__()
        self.dim, self.tp, self.rank = dim, tp, rank
        self.full_shape = (outputs, inputs)
        local = list(self.full_shape)
        if local[dim] % tp:
            raise ValueError(
                f"Megatron-BERT projection {local} cannot split over TP={tp}"
            )
        local[dim] //= tp
        self.weight = nn.Parameter(torch.empty(local, dtype=dtype), requires_grad=False)
        self.bias = nn.Parameter(
            torch.empty(local[0], dtype=dtype), requires_grad=False
        )

    def checkpoint_read_spec(self, name, shape):
        if name not in ("weight", "bias"):
            return None
        full_shape = self.full_shape if name == "weight" else self.full_shape[:1]
        if tuple(shape) != full_shape:
            raise ValueError(
                f"Bad Megatron-BERT {name} shape: {tuple(shape)} != {full_shape}"
            )
        if self.tp > 1 and (name == "weight" or self.dim == 0):
            return make_sharded_read_spec(name, shape, self.dim, self.tp, self.rank)
        return None

    def forward(self, hidden):
        if self.dim == 0 or self.tp == 1:
            return F.linear(hidden, self.weight, self.bias)
        from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

        return all_reduce(F.linear(hidden, self.weight), group=Group.TP) + self.bias


class _Layer(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        h, f = config.hidden_size, config.inter_size
        dtype = load.compute_dtype
        args = (load.tp_size, load.tp_rank, dtype)
        self.attn_norm = nn.LayerNorm(h, eps=config.layernorm_eps, dtype=dtype)
        self.ffn_norm = nn.LayerNorm(h, eps=config.layernorm_eps, dtype=dtype)
        self.query = _Projection(h, h, 0, *args)
        self.key = _Projection(h, h, 0, *args)
        self.value = _Projection(h, h, 0, *args)
        self.out = _Projection(h, h, 1, *args)
        self.intermediate = _Projection(h, f, 0, *args)
        self.output = _Projection(f, h, 1, *args)

    def forward(self, hidden, fmha, cache, index):
        normalized = self.attn_norm(hidden)
        qkv = torch.cat(
            (self.query(normalized), self.key(normalized), self.value(normalized)), -1
        )
        attention = fmha.forward(qkv, cache, index).reshape(*hidden.shape[:-1], -1)
        hidden = hidden + self.out(attention)
        return hidden + self.output(F.gelu(self.intermediate(self.ffn_norm(hidden))))


class _Names:
    layer_parts = {
        "attention.ln": "attn_norm",
        "ln": "ffn_norm",
        "attention.self.query": "query",
        "attention.self.key": "key",
        "attention.self.value": "value",
        "attention.output.dense": "out",
        "intermediate.dense": "intermediate",
        "output.dense": "output",
    }

    def __init__(self, custom):
        self.custom = custom

    def map_name(self, source):
        name = source.removeprefix("bert.")
        if source in self.custom:
            return "custom_params." + self.custom[source]
        if name in self.custom:
            return "custom_params." + self.custom[name]
        if name.startswith("encoder.layer."):
            _, _, index, remainder = name.split(".", 3)
            stem, field = remainder.rsplit(".", 1)
            field = {"gamma": "weight", "beta": "bias"}.get(field, field)
            if stem in self.layer_parts and index.isdecimal():
                return f"layers.{index}.{self.layer_parts[stem]}.{field}"
        if name.startswith("encoder.ln."):
            field = name.rsplit(".", 1)[-1]
            return "final_norm." + {"gamma": "weight", "beta": "bias"}.get(field, field)
        return name


class MegatronParameters:
    """Shared initialization/forward for runtime and native-independent tests."""

    supports_custom_weight_mappings = True

    def build_encoder(self, config, load):
        h, heads, tp = config.hidden_size, config.attn_config.head_num, load.tp_size
        if heads % tp or h != heads * config.attn_config.size_per_head:
            raise ValueError(
                "Megatron-BERT needs complete attention heads on each TP rank"
            )
        if config.attn_config.kv_head_num != heads or config.attn_config.is_causal:
            raise ValueError("Megatron-BERT requires non-causal multi-head attention")
        if (load.attn_tp_size, load.ffn_tp_size) != (tp, tp):
            raise ValueError(
                "Megatron-BERT requires a shared TP group for attention and FFN"
            )
        if (load.attn_tp_rank, load.ffn_tp_rank) != (load.tp_rank, load.tp_rank):
            raise ValueError("Megatron-BERT attention and FFN ranks must match TP rank")
        if getattr(load.parallelism_config, "pp_size", 1) != 1:
            raise ValueError(
                "Megatron-BERT PP stage ownership has not been implemented"
            )
        if getattr(load.quant_config, "quant_type", "none") not in (
            "none",
            "",
        ) or getattr(config, "quantization", "") not in ("none", "", None):
            raise ValueError("Megatron-BERT expected an unquantized checkpoint")
        dtype = load.compute_dtype
        self.compute_dtype = dtype
        self.embeddings = nn.Module()
        self.embeddings.word_embeddings = _table(config.vocab_size, h, dtype)
        self.embeddings.position_embeddings = _table(config.max_seq_len, h, dtype)
        self.embeddings.token_type_embeddings = (
            _table(config.type_vocab_size, h, dtype) if config.type_vocab_size else None
        )
        self.layers = nn.ModuleList(
            [_Layer(config, load) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(h, eps=config.layernorm_eps, dtype=dtype)
        self.custom_params = nn.ParameterDict()
        mappings = tuple(map(unpack_handler_mapping, load.custom_weight_mappings))
        self._custom = {f"p{i}": runtime for i, (runtime, _, _) in enumerate(mappings)}
        self._custom_dtype = {
            f"p{i}": dt or dtype for i, (_, _, dt) in enumerate(mappings)
        }
        self.WEIGHTS_MAPPER = _Names(
            {source: f"p{i}" for i, (_, source, _) in enumerate(mappings)}
        )

    def checkpoint_weight_name_filter(self):
        def used(source):
            mapped = self.WEIGHTS_MAPPER.map_name(source)
            if mapped.startswith("custom_params."):
                return True
            return not (
                mapped.startswith(("pooler.", "classifier.", "cls."))
                or mapped in ("embeddings.position_ids", "embeddings.token_type_ids")
            )

        return used

    def load_weights(self, weights):
        seen = set()
        for source, tensor in weights.items() if isinstance(weights, dict) else weights:
            name = self.WEIGHTS_MAPPER.map_name(source)
            if name in seen:
                raise ValueError(f"Duplicate checkpoint entry for {name}")
            seen.add(name)
            if name.startswith("custom_params."):
                key = name.split(".", 1)[1]
                tensor, _ = unwrap_checkpoint_tensor(tensor)
                if not tensor.is_floating_point() or key not in self._custom:
                    raise ValueError(f"Invalid Megatron-BERT custom weight {source}")
                # The FST staging tensor does not survive the iterator.
                value = tensor.detach().to(self._custom_dtype[key], copy=True)
                self.custom_params[key] = nn.Parameter(value, requires_grad=False)
                _mark_loaded(self.custom_params, key)
                continue
            if name == "embeddings.position_embeddings.weight":
                tensor, _ = unwrap_checkpoint_tensor(tensor)
                if tensor.ndim == 2:
                    tensor = tensor[
                        : self.embeddings.position_embeddings.num_embeddings
                    ]
            if not self._dispatch(self, name, tensor):
                raise ValueError(
                    f"Unexpected Megatron-BERT checkpoint tensor {source!r}"
                )
        missing = self._custom.keys() - self.custom_params.keys()
        if missing:
            raise ValueError(
                f"Missing Megatron-BERT custom weights: {[self._custom[k] for k in sorted(missing)]}"
            )

    def encode_packed(self, ids, pos, types, scalar, fmha_for, cache_for):
        if pos is None or ids.shape != pos.shape:
            raise ValueError("Megatron-BERT requires a position ID per packed token")
        hidden = self.embeddings.word_embeddings(ids) * scalar
        if self.embeddings.token_type_embeddings is not None:
            if types is None or not types.numel():
                types = torch.zeros_like(ids)
            if types.shape != ids.shape:
                raise ValueError(
                    "Megatron-BERT token-type IDs must align with packed tokens"
                )
            hidden = hidden + self.embeddings.token_type_embeddings(types)
        hidden = hidden + self.embeddings.position_embeddings(pos)
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, fmha_for(i), cache_for(i), i)
        return self.final_norm(hidden)

    def custom_weight_view(self):
        return {name: self.custom_params[key] for key, name in self._custom.items()}
