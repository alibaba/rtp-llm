"""Module-owned import of Meta Llama's raw consolidated tensors."""

import torch

from rtp_llm.models_py.ft_checkpoint import RuntimeTensorBinding
from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod
from rtp_llm.models_py.read_spec import LoadedCheckpointTensor


def load_meta_checkpoint(model, checkpoint):
    params = checkpoint.params
    first = model.layers[0].self_attn
    for name, expected in (
        ("dim", model.embed_tokens.embedding_dim),
        ("n_layers", len(model.layers)),
        ("n_heads", first.num_heads),
    ):
        if type(params.get(name)) is not int or params[name] != expected:
            raise ValueError(
                f"Meta params.json {name} does not match model: {params.get(name)!r} vs {expected}"
            )
    if params.get("n_kv_heads", params["n_heads"]) != first.num_kv_heads:
        raise ValueError("Meta params.json n_kv_heads does not match model")
    if params.get("norm_eps") != model.norm.eps:
        raise ValueError("Meta params.json norm_eps does not match model")
    for layer in model.layers:
        for linear in (
            layer.self_attn.qkv_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_up_proj,
            layer.mlp.down_proj,
        ):
            if not isinstance(linear.quant_method, UnquantizedLinearMethod):
                raise ValueError(
                    "Meta consolidated import requires an unquantized model adapter"
                )

    def local(name, dim, leaf, leaf_name="weight", wrapped=False):
        shape = checkpoint.shape(name, dim)
        resolver = getattr(leaf, "checkpoint_read_spec", None)
        spec = resolver(leaf_name, shape) if resolver else None
        rows = getattr(spec, "row_indices", None)
        tensor = (
            checkpoint.read_rows(name, dim, rows)
            if rows is not None
            else checkpoint.read_slice(name, dim, spec.slices if spec else None)
        )
        if not tensor.is_floating_point():
            raise ValueError(
                "Unquantized Meta checkpoint weights must be floating point"
            )
        # Meta consolidated files are raw model weights, not FT runtime snapshots:
        # checkpoint dtype is converted to the requested parameter compute dtype.
        tensor = tensor.to(dtype=leaf.weight.dtype)
        return LoadedCheckpointTensor(tensor, spec) if wrapped and spec else tensor

    def permute(value, head_dim):
        if head_dim % 2 or value.ndim != 2 or value.shape[0] % head_dim:
            raise ValueError(
                "Meta rotary Q/K layout must contain complete even-sized heads"
            )
        heads = value.shape[0] // head_dim
        return (
            value.reshape(heads, head_dim // 2, 2, value.shape[1])
            .transpose(1, 2)
            .reshape(value.shape)
        )

    bindings = []

    def add(names, target, reader):
        bindings.append(
            RuntimeTensorBinding(
                tuple(names), target, lambda values: values[0], lambda: (reader(),)
            )
        )

    def vocabulary(name, target, dim, leaf):
        def prepare(values):
            # prepare_weight_updates runs the actual leaf loader on isolated
            # storage. Embedding owns suppression of its live tied-head weakref.
            updates = leaf.prepare_weight_updates({"weight": values[0]})
            if len(updates) != 1 or updates[0].name != "weight":
                raise ValueError("Meta vocabulary conversion must produce one weight")
            return updates[0].source

        bindings.append(
            RuntimeTensorBinding(
                (name,),
                target,
                prepare,
                lambda: (local(name, dim, leaf, wrapped=True),),
                lambda values: leaf.load_weights({"weight": values[0]}),
            )
        )

    vocabulary("tok_embeddings.weight", "embed_tokens.weight", 1, model.embed_tokens)
    vocabulary("output.weight", "lm_head.weight", 0, model.lm_head)
    add(("norm.weight",), "norm.weight", lambda: local("norm.weight", None, model.norm))
    for index, layer in enumerate(model.layers):
        src = f"layers.{index}."
        dst = f"layers.{index}."
        for name, target, leaf, dim in (
            (
                "attention_norm.weight",
                "input_layernorm.weight",
                layer.input_layernorm,
                None,
            ),
            (
                "ffn_norm.weight",
                "post_attention_layernorm.weight",
                layer.post_attention_layernorm,
                None,
            ),
            (
                "attention.wo.weight",
                "self_attn.o_proj.weight",
                layer.self_attn.o_proj,
                1,
            ),
            ("feed_forward.w2.weight", "mlp.down_proj.weight", layer.mlp.down_proj, 1),
        ):
            full = src + name
            add(
                (full,),
                dst + target,
                lambda full=full, dim=dim, leaf=leaf: local(full, dim, leaf),
            )
        qkv_names = tuple(src + f"attention.w{part}.weight" for part in "qkv")

        def qkv(names=qkv_names, leaf=layer.self_attn.qkv_proj):
            values = [
                local(name, 0, leaf, f"{part}_proj.weight")
                for name, part in zip(names, "qkv")
            ]
            values[:2] = [permute(value, leaf.head_dim) for value in values[:2]]
            return torch.cat(values, dim=0)

        add(qkv_names, dst + "self_attn.qkv_proj.weight", qkv)
        mlp_names = (src + "feed_forward.w1.weight", src + "feed_forward.w3.weight")

        def mlp(names=mlp_names, leaf=layer.mlp.gate_up_proj):
            return torch.cat(
                (
                    local(names[0], 0, leaf, "gate_proj.weight"),
                    local(names[1], 0, leaf, "up_proj.weight"),
                ),
                dim=0,
            )

        add(mlp_names, dst + "mlp.gate_up_proj.weight", mlp)
    checkpoint.load_parameters(model, tuple(bindings))
