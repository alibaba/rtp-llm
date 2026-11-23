"""Direct parameter ownership for declared classifier and reranker heads."""

import torch
from torch import nn

from rtp_llm.models_py.module_base import _mark_loaded
from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor


def handler_weight_mappings(names):
    """Turn declared checkpoint names into the embedding engine's global export roles."""
    if not isinstance(names, tuple) or not all(
        isinstance(name, str) and name for name in names
    ):
        raise TypeError("Handler checkpoint names must be a tuple of nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate handler checkpoint name")
    return tuple(("__custom__." + name, name) for name in names)


def unpack_handler_mapping(mapping):
    if not isinstance(mapping, tuple) or len(mapping) not in (2, 3):
        raise TypeError("Expected (runtime_name, checkpoint_name[, dtype])")
    role, source = mapping[:2]
    dtype = mapping[2] if len(mapping) == 3 else None
    if dtype is not None and dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise TypeError(f"Unsupported replicated handler weight dtype {dtype!r}")
    return role, source, dtype


def needs_generation_projection(config):
    from rtp_llm.ops import TaskType

    default = TaskType.LANGUAGE_MODEL
    task = config.get("task_type", default) if isinstance(config, dict) else getattr(config, "task_type", default)
    return task == default


class DecoderHandlerOwner:
    """Attach exactly the declared handler inputs to a score-model owner."""

    def _init_decoder_handlers(self, load_config, hidden_size):
        self.supports_custom_weight_mappings = True
        self.supports_handler_projection_layers = True
        self._handler_checkpoint_names = frozenset(
            item[1] for item in getattr(load_config, "custom_weight_mappings", ())
        )
        self.handler_heads = ReplicatedHandlerHeads(
            getattr(load_config, "custom_weight_mappings", ()),
            self.WEIGHTS_MAPPER, hidden_size, load_config.compute_dtype,
            projection_layers=getattr(load_config, "custom_projection_layers", ()),
            protected_roots={"embed_tokens", "lm_head", "norm", "layers", "handler_heads"},
        )

    def _capture_handler_inputs(self, items):
        heads = getattr(self, "handler_heads", None)
        if heads is None or not heads.sources:
            yield from items
            return
        for name, value in items:
            mapped = self.WEIGHTS_MAPPER.map_name(name)
            if mapped in heads.sources:
                value = heads.capture(mapped, value)
                if mapped not in heads.vocabulary_sources:
                    continue
                if mapped == "lm_head.weight" and self.lm_head is None:
                    continue
                if mapped == "embed_tokens.weight" and self.embed_tokens is None and (
                    self.lm_head is None or not self.tie_word_embeddings
                ):
                    continue
            yield name, value

    def resolve_checkpoint_load_plan(self, name, shape):
        heads = getattr(self, "handler_heads", None)
        if heads is not None and name in heads.sources:
            return None, None  # The handler consumes the complete, unprocessed source.
        return super().resolve_checkpoint_load_plan(name, shape)

    def validate_weights_loaded(self, loaded_tensor_ids=None):
        heads = getattr(self, "handler_heads", None)
        if heads is not None:
            heads.validate()
        super().validate_weights_loaded(loaded_tensor_ids)


class ReplicatedHandlerHeads(nn.Module):
    """Handler projections consume full, unprocessed weights on each serving rank."""

    # Unlike the classifier head, these are shared *checkpoint inputs* with the
    # language model: its TP/PP slice, pruning, or logit scaling is not valid
    # as the handler's full-vocabulary output projection.
    vocabulary_sources = frozenset(("embed_tokens.weight", "lm_head.weight"))
    allowed = vocabulary_sources | {"classifier.weight", "classifier.bias"}

    def __init__(
        self,
        mappings,
        mapper,
        hidden_size,
        dtype,
        *,
        projection_sources=(),
        projection_layers=(),
        protected_roots=(),
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.sources = {}
        self.dtypes = {}
        projections = set(projection_sources)
        self.chain = []
        chain_weights, chain_biases = set(), set()
        declared = {m[1] for m in mappings}
        if not isinstance(projection_layers, tuple):
            raise TypeError("Handler projection layers must be an ordered tuple")
        for pair in projection_layers:
            if not isinstance(pair, tuple) or len(pair) != 2 or pair[0] is None:
                raise ValueError(
                    "Expected (checkpoint weight, optional bias) per projection"
                )
            mapped = []
            for source, seen in zip(pair, (chain_weights, chain_biases)):
                if source is None:
                    mapped.append(None)
                    continue
                if not isinstance(source, str) or source not in declared:
                    raise ValueError(f"Undeclared handler projection source {source!r}")
                name = mapper.map_name(source)
                if (
                    name in chain_weights | chain_biases
                    or name.split(".", 1)[0] in protected_roots
                ):
                    raise ValueError(f"Ambiguous handler projection source {source}")
                seen.add(name)
                mapped.append(name)
            self.chain.append(tuple(mapped))
        self.chain_weights, self.chain_biases = chain_weights, chain_biases
        if not all(
            isinstance(n, str) and n.endswith((".weight", ".bias")) for n in projections
        ):
            raise ValueError("Handler projections must name a weight or bias")
        for mapping in mappings:
            target, source, declared_dtype = unpack_handler_mapping(mapping)
            name = mapper.map_name(source)
            if (
                name not in self.allowed
                and name not in projections
                and name not in chain_weights
                and name not in chain_biases
            ) or name in self.sources:
                raise ValueError(f"Unsupported or ambiguous handler source {source}")
            self.sources[name] = (str(len(self.sources)), target)
            self.dtypes[name] = dtype if declared_dtype is None else declared_dtype
        self.params = nn.ParameterDict()
        self.register_buffer("_anchor", torch.empty(0, dtype=dtype), persistent=False)

    def capture(self, name, value):
        """Return original tensor for optional dispatch into the language owner."""
        key, _ = self.sources[name]
        tensor, spec = unwrap_checkpoint_tensor(value)
        if key in self.params:
            raise ValueError(f"Duplicate downstream checkpoint tensor {name}")
        if spec is not None and (
            spec.slices is not None
            or spec.row_indices is not None
            or tuple(tensor.shape) != spec.source_shape
        ):
            raise ValueError(f"Downstream checkpoint tensor {name} must be replicated")
        is_bias = name in self.chain_biases or (
            name not in self.chain_weights and name.endswith(".bias")
        )
        if (
            not tensor.is_floating_point()
            or tensor.ndim != (1 if is_bias else 2)
            or tensor.shape[0] < 1
            or (
                not is_bias
                and (
                    tensor.shape[1] < 1
                    or (
                        name not in self.chain_weights
                        and tensor.shape[1] != self.hidden_size
                    )
                )
            )
        ):
            raise ValueError(f"Invalid downstream tensor {name}: {tuple(tensor.shape)}")
        # FST frees its backing storage; the handler owns its registered parameter.
        self.params[key] = nn.Parameter(
            tensor.detach().to(
                device=self._anchor.device, dtype=self.dtypes[name], copy=True
            ),
            requires_grad=False,
        )
        _mark_loaded(self.params, key)
        return tensor

    def ft_bindings(self, checkpoint):
        """The FT writer stores replicated handler tensors under their runtime role."""
        from rtp_llm.models_py.ft_checkpoint import RuntimeTensorBinding

        bindings = []
        for source, (slot, role) in self.sources.items():
            key = "global." + role
            if key not in checkpoint.entries:
                raise ValueError(f"Missing FT handler tensor: {key}")
            value = checkpoint.read(key)
            if value.dtype != self.dtypes[source]:
                raise ValueError(f"FT handler dtype mismatch for {key}: {value.dtype} vs {self.dtypes[source]}")
            self.capture(source, value)
            bindings.append(RuntimeTensorBinding((key,), f"handler_heads.params.{slot}", lambda values: values[0]))
        self.validate()
        return bindings

    def validate(self):
        missing = [
            name for name, (key, _) in self.sources.items() if key not in self.params
        ]
        if missing:
            raise RuntimeError(f"Missing downstream checkpoint tensors: {missing}")
        width = self.hidden_size
        for weight, bias in self.chain:
            tensor = self.params[self.sources[weight][0]]
            if tensor.shape[1] != width:
                raise ValueError(
                    f"Invalid handler projection {weight}: expected input width {width}, got {tensor.shape[1]}"
                )
            width = tensor.shape[0]
            if bias is not None and self.params[self.sources[bias][0]].shape != (
                width,
            ):
                raise ValueError(
                    f"Invalid handler projection bias {bias}: expected {width} outputs"
                )
        for name, (b_key, _) in self.sources.items():
            if (
                name.endswith(".bias")
                and name not in self.chain_biases
                and name not in self.chain_weights
            ):
                weight = name[:-4] + "weight"
                if weight in self.sources:
                    w_key, _ = self.sources[weight]
                    if self.params[w_key].shape[0] != self.params[b_key].shape[0]:
                        raise ValueError(
                            f"Classifier weight and bias label counts disagree for {name}"
                        )

    def exports(self):
        self.validate()
        return {target: self.params[key] for key, target in self.sources.values()}
