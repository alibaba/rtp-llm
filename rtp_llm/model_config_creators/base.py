"""Base utilities for model configuration creators.

This module provides common utility functions used by configuration creators,
such as reading config.json files.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_config_json(ckpt_path: str) -> Optional[Dict[str, Any]]:
    """Read and parse config.json from checkpoint path.

    Args:
        ckpt_path: Path to the model checkpoint directory

    Returns:
        Parsed config.json as a dictionary, or None if file doesn't exist

    Raises:
        FileNotFoundError: If config.json doesn't exist and is required
        json.JSONDecodeError: If config.json is invalid JSON
    """
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.exists(config_path):
        logger.warning(f"config.json not found at {config_path}")
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as reader:
            content = reader.read()
            config_json = json.loads(content)
            return config_json
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse config.json at {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to read config.json at {config_path}: {e}")
        raise


def require_config_json(ckpt_path: str) -> Dict[str, Any]:
    """Read and parse config.json, raising an error if not found.

    Args:
        ckpt_path: Path to the model checkpoint directory

    Returns:
        Parsed config.json as a dictionary

    Raises:
        FileNotFoundError: If config.json doesn't exist
        json.JSONDecodeError: If config.json is invalid JSON
    """
    config_json = get_config_json(ckpt_path)
    if config_json is None:
        raise FileNotFoundError(
            f"config.json not found at {os.path.join(ckpt_path, 'config.json')}"
        )
    return config_json
