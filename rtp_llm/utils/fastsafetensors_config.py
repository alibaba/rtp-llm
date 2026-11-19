"""Bridge registered loader options to the upstream environment-only API."""

import json
import os
from typing import Optional

_JSON_ENV = "FASTSAFETENSORS_CONFIG_JSON"
_PATH_ENV = "FASTSAFETENSORS_CONFIG"
_DEFAULT_JSON = json.dumps(
    {
        "max_broadcast_bucket_bytes": 64 * 1024 * 1024,
        "max_broadcast_tensor_bytes": 32 * 1024 * 1024,
    }
)
_injected_default = False


def configure_fastsafetensors(config_json: Optional[str], config_path: Optional[str]):
    """Publish parsed CLI/env options before workers spawn; preserve empty values."""
    global _injected_default
    if config_json is not None:
        os.environ[_JSON_ENV] = config_json
        _injected_default = False
    elif config_path is not None and _injected_default:
        if os.environ.get(_JSON_ENV) == _DEFAULT_JSON:
            del os.environ[_JSON_ENV]
        _injected_default = False
    if config_path is not None:
        os.environ[_PATH_ENV] = config_path


def clear_injected_fastsafetensors_default():
    """Do not parse our fallback as a user option on repeated server setup."""
    global _injected_default
    if _injected_default and os.environ.get(_JSON_ENV) == _DEFAULT_JSON:
        del os.environ[_JSON_ENV]
    _injected_default = False


def ensure_fastsafetensors_configured():
    """Also support direct CkptDatabase users that never run the server parser.

    The upstream AutoLoader has no config argument. This adapter is the sole
    compatibility boundary for its registered environment options. Preserve its
    JSON > path > working-directory file precedence and inherited worker config.
    """
    global _injected_default
    if (
        _JSON_ENV not in os.environ
        and _PATH_ENV not in os.environ
        and not os.path.isfile("fastsafetensors.json")
    ):
        os.environ[_JSON_ENV] = _DEFAULT_JSON
        _injected_default = True
