"""Qwen4 media text rows: a replicated CPU parameter, never a TP GPU table."""

import torch
from safetensors import safe_open

from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.registry import register_model
from rtp_llm.models_py.weight_mapper import (
    WeightsMapper,
    discover_ckpt_files,
    get_all_weights,
    safetensor_index_entries,
)

_EMBEDDING_NAMES = (
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
)


def _embedding_metadata(model_path):
    files = discover_ckpt_files(model_path)
    entries = safetensor_index_entries(model_path)
    shapes = {}
    if entries is not None:
        table = dict(entries)
        for name in _EMBEDDING_NAMES:
            if name in table:
                with safe_open(table[name], framework="pt", device="cpu") as source:
                    return name, source.get_slice(name).get_shape()
    elif files and all(path.endswith(".safetensors") for path in files):
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as source:
                for name in _EMBEDDING_NAMES:
                    if name in source.keys():
                        if name in shapes:
                            raise ValueError(
                                f"duplicate Qwen4 text embedding tensor {name}"
                            )
                        shapes[name] = source.get_slice(name).get_shape()
    else:
        # PyTorch checkpoints have no shape-only header. Inspect using the
        # existing safe reader; payload is released before the actual load.
        for name, tensor in get_all_weights(
            files, device="cpu", name_filter=lambda key: key in _EMBEDDING_NAMES
        ):
            if name in shapes:
                raise ValueError(f"duplicate Qwen4 text embedding tensor {name}")
            shapes[name] = list(tensor.shape)
    for name in _EMBEDDING_NAMES:
        if name in shapes:
            return name, shapes[name]
    raise ValueError("Qwen4 checkpoint is missing its text embedding weight")


@register_model("qwen4_media_text_embedding")
class Qwen4MediaTextEmbedding(RtpModule):
    fastsafetensors_load_isolated = True

    def __init__(self, model_config, load_config):
        super().__init__()
        if torch.device(load_config.device).type != "cpu":
            raise ValueError("Qwen4 media text embedding must remain on CPU")
        if load_config.tp_size != 1 or load_config.tp_rank != 0:
            raise ValueError(
                "Qwen4 media text embedding is a replicated full vocabulary"
            )
        self.checkpoint_name = model_config["checkpoint_name"]
        shape = model_config["embedding_shape"]
        if len(shape) != 2 or any(
            not isinstance(size, int) or size <= 0 for size in shape
        ):
            raise ValueError(
                "Qwen4 text embedding must have shape [vocabulary, hidden]"
            )
        self.text_embedding = torch.nn.Embedding(
            *shape, dtype=load_config.compute_dtype
        )
        self.WEIGHTS_MAPPER = WeightsMapper(
            exact_mapping={self.checkpoint_name: "text_embedding.weight"}
        )

    def checkpoint_weight_name_filter(self):
        return lambda name: name == self.checkpoint_name

    def load_weights(self, weights):
        iterator = weights.items() if isinstance(weights, dict) else weights
        super().load_weights(self.WEIGHTS_MAPPER.apply(iterator))


def load_qwen4_media_text_embedding(
    model_path, compute_dtype, load_method=NewLoaderLoadMethod.SCRATCH
):
    name, shape = _embedding_metadata(model_path)
    model = NewModelLoader(
        model_config=dict(
            model_type="qwen4_media_text_embedding",
            checkpoint_name=name,
            embedding_shape=shape,
        ),
        load_config=NewLoaderConfig(
            device="cpu",
            compute_dtype=compute_dtype,
            load_method=getattr(load_method, "value", load_method),
        ),
        model_path=model_path,
    ).load()
    # Unlike visual weights, main loads this auxiliary table directly into the
    # compute dtype, without staging through FP16. Keep that distinction.
    return model.text_embedding
