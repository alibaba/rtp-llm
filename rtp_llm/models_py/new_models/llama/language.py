"""Hugging Face Llama dense model for the streaming NewLoader.

Llama and Qwen2 use the same pre-norm SwiGLU decoder layout.  The checkpoint
tree and tensor-parallel sharding rules are also identical; the architectural
difference relevant to these modules is that standard Llama attention has no
Q/K/V projection bias.  Reusing the shared implementation keeps checkpoint
dispatch, completeness validation, TP ownership, quantization, and tied
embedding behavior on one tested path.

Only the standard Hugging Face ``LlamaForCausalLM`` layout is registered here.
Legacy formats such as Meta ``params.json``, Baichuan ``W_pack``, and InternLM2
``wqkv`` are intentionally not routed to this class; unsupported configurations
or checkpoint keys fail before inference instead of being silently interpreted
with the wrong layout.
"""

import json
import os
from typing import Any

from rtp_llm.models_py.new_models.qwen2.language import Qwen2ForCausalLM


def _checkpoint_config(model_config: Any) -> dict[str, Any]:
    """Return the checkpoint config when a production checkpoint is attached."""
    ckpt_path = (
        model_config.get("ckpt_path", "")
        if isinstance(model_config, dict)
        else getattr(model_config, "ckpt_path", "")
    )
    if ckpt_path is None:
        ckpt_path = ""
    if not isinstance(ckpt_path, str):
        raise TypeError(f"model_config.ckpt_path must be a string, got {ckpt_path!r}")
    if not ckpt_path:
        return model_config if isinstance(model_config, dict) else {}

    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            "Llama newloader requires checkpoint config.json to validate the "
            "decoder architecture"
        )
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"{config_path} must contain a JSON object")
    return config


def _validate_llama_config(model_config: Any) -> None:
    config = _checkpoint_config(model_config)

    model_type = config.get("model_type")
    if model_type is not None and model_type != "llama":
        raise ValueError(
            f"Llama newloader requires model_type='llama', got {model_type!r}"
        )
    architectures = config.get("architectures")
    if architectures is not None:
        if not isinstance(architectures, list) or not all(
            isinstance(name, str) for name in architectures
        ):
            raise TypeError("architectures must be a list of strings")
        if "LlamaForCausalLM" not in architectures:
            raise ValueError(
                "Llama newloader supports only the Hugging Face "
                "LlamaForCausalLM checkpoint layout"
            )

    def config_value(name: str, default: Any) -> Any:
        if config:
            return config.get(name, default)
        return getattr(model_config, name, default)

    for field_name, description in (
        ("attention_bias", "Attention projection bias"),
        ("mlp_bias", "MLP projection bias"),
    ):
        enabled = config_value(field_name, False)
        if not isinstance(enabled, bool):
            raise TypeError(f"{field_name} must be a bool")
        if enabled:
            raise ValueError(
                f"{description} is not supported by the Llama newloader path"
            )

    hidden_act = config_value("hidden_act", "silu")
    if hidden_act != "silu":
        raise ValueError(
            f"Llama newloader requires hidden_act='silu', got {hidden_act!r}"
        )


class LlamaForCausalLM(Qwen2ForCausalLM):
    """Standard bias-free Hugging Face Llama causal language model."""

    QKV_BIAS = False

    def __init__(self, model_config: Any, load_config: Any):
        _validate_llama_config(model_config)
        super().__init__(model_config, load_config)
