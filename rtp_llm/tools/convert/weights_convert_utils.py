from typing import Any, Mapping


def apply_layer_override_and_post_build(
    model_config: Any, model_cls: Any, env_params: Mapping[str, str]
) -> Any:
    """Apply converter-only layer overrides before deriving cache descriptors."""
    model_config.num_layers = int(
        env_params.get("HACK_LAYER_NUM", str(model_config.num_layers))
    )
    model_cls._post_build_model_config(model_config)
    return model_config
