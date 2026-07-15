from typing import Mapping, Protocol

from rtp_llm.config.model_config import ModelConfig, apply_layer_num_override


class ModelConfigPostBuilder(Protocol):
    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None: ...


def apply_layer_override_and_post_build(
    model_config: ModelConfig,
    model_cls: type[ModelConfigPostBuilder],
    env_params: Mapping[str, str],
) -> ModelConfig:
    """Apply converter-only layer overrides before deriving cache descriptors."""
    num_layers = int(env_params.get("HACK_LAYER_NUM", str(model_config.num_layers)))
    if num_layers != model_config.num_layers:
        apply_layer_num_override(model_config, num_layers)
    model_cls._post_build_model_config(model_config)
    return model_config
