"""Registry for model configuration creators.

This module provides a registration mechanism for model configuration creator
functions, allowing configuration creation to be decoupled from model classes.
"""

import logging
from typing import Callable, Dict, Optional, Union

from rtp_llm.config.model_config import ModelConfig

logger = logging.getLogger(__name__)

_config_creators: Dict[str, Callable[[str], ModelConfig]] = {}


def _normalize_model_type(model_type: str) -> str:
    """Normalize model type by converting to lowercase and removing underscores.

    Args:
        model_type: The model type string (e.g., "qwen_2", "QWen_V2")

    Returns:
        Normalized model type (e.g., "qwen2", "qwenv2")
    """
    return model_type.lower().replace("_", "")


def register_config_creator(
    model_type: Union[str, Callable[[str], ModelConfig]],
    creator_func: Optional[Callable[[str], ModelConfig]] = None,
) -> Union[Callable[[str], ModelConfig], Callable]:
    """Register a configuration creator function for a model type.

    This function can be used in two ways:
    1. As a decorator: @register_config_creator("model_type")
    2. As a function: register_config_creator("model_type", creator_func)

    The function automatically registers both the original model_type and its normalized
    version (lowercase without underscores) to handle naming variations.

    Args:
        model_type: The model type string (e.g., "bert", "roberta", "tbstars2_5")
                   OR the creator function (when used as decorator without parentheses)
        creator_func: A function that takes ckpt_path (str) and returns ModelConfig
                     (None when used as decorator)

    Returns:
        When used as decorator, returns the decorated function.
        When used as function, returns the creator_func.

    Raises:
        ValueError: If model_type is already registered with a different function
    """
    global _config_creators

    def _register_model_type(
        actual_model_type: str, func: Callable[[str], ModelConfig]
    ) -> None:
        """Register a model type and its normalized version."""
        # Register original model_type
        if (
            actual_model_type in _config_creators
            and _config_creators[actual_model_type] != func
        ):
            raise ValueError(
                f"Model type '{actual_model_type}' is already registered with a different "
                f"creator function. Existing: {_config_creators[actual_model_type]}, "
                f"New: {func}"
            )
        _config_creators[actual_model_type] = func
        logger.debug(f"Registered config creator for model type: {actual_model_type}")

        # Also register normalized version (lowercase without underscores)
        normalized_type = _normalize_model_type(actual_model_type)
        if normalized_type != actual_model_type:
            if (
                normalized_type in _config_creators
                and _config_creators[normalized_type] != func
            ):
                # Only warn if it's a different function, allow same function
                if _config_creators[normalized_type] != func:
                    logger.warning(
                        f"Normalized model type '{normalized_type}' is already registered with "
                        f"a different function, skipping duplicate registration."
                    )
                return
            _config_creators[normalized_type] = func
            logger.debug(
                f"Registered config creator for normalized model type: {normalized_type}"
            )

    # Decorator usage: @register_config_creator("model_type")
    if creator_func is None:
        # Check if model_type is a string (decorator with argument) or a function (decorator without argument)
        if isinstance(model_type, str):
            # Decorator with argument: @register_config_creator("model_type")
            actual_model_type = model_type

            def decorator(
                func: Callable[[str], ModelConfig]
            ) -> Callable[[str], ModelConfig]:
                _register_model_type(actual_model_type, func)
                return func

            return decorator
        else:
            # Decorator without argument: @register_config_creator
            # model_type is actually the function
            func = model_type  # type: ignore
            # Try to get model_type from function name or raise error
            raise ValueError(
                "register_config_creator used as decorator without model_type. "
                "Use @register_config_creator('model_type') instead."
            )

    # Function usage: register_config_creator("model_type", creator_func)
    if not isinstance(model_type, str):
        raise TypeError(f"model_type must be a string, got {type(model_type)}")
    _register_model_type(model_type, creator_func)
    return creator_func


def get_config_creator(
    model_type: str,
) -> Optional[Callable[[str], ModelConfig]]:
    """Get the configuration creator function for a model type.

    This function first tries exact match, then tries normalized version
    (lowercase without underscores), and finally tries common variations
    to handle naming inconsistencies.

    Args:
        model_type: The model type string

    Returns:
        The configuration creator function if registered, None otherwise
    """
    # Try exact match first
    creator = _config_creators.get(model_type)
    if creator:
        return creator

    # Try normalized version (lowercase without underscores)
    normalized_type = _normalize_model_type(model_type)
    if normalized_type != model_type:
        creator = _config_creators.get(normalized_type)
        if creator:
            return creator

    # Try common variations for cases like "qwen_2" vs "qwen_v2"
    # Generate variations by replacing common patterns
    variations = []
    if "_2" in model_type.lower():
        # "qwen_2" -> try "qwen_v2", "qwen2"
        variations.append(model_type.lower().replace("_2", "_v2"))
        variations.append(model_type.lower().replace("_2", "2"))
    if "_v2" in model_type.lower():
        # "qwen_v2" -> try "qwen_2", "qwen2"
        variations.append(model_type.lower().replace("_v2", "_2"))
        variations.append(model_type.lower().replace("_v2", "v2"))
    if "2_moe" in model_type.lower():
        # "qwen2_moe" -> try "qwen_2_moe"
        variations.append(model_type.lower().replace("2_moe", "_2_moe"))
    if "_2_moe" in model_type.lower():
        # "qwen_2_moe" -> try "qwen2_moe"
        variations.append(model_type.lower().replace("_2_moe", "2_moe"))

    for variation in variations:
        if variation != model_type and variation != normalized_type:
            creator = _config_creators.get(variation)
            if creator:
                return creator
            # Also try normalized version of variation
            normalized_variation = _normalize_model_type(variation)
            if normalized_variation != variation:
                creator = _config_creators.get(normalized_variation)
                if creator:
                    return creator

    return None


def list_registered_creators() -> list[str]:
    """List all registered model types.

    Returns:
        A list of registered model type strings
    """
    return list(_config_creators.keys())
