import functools
import glob
import json
import logging
import os
from typing import Dict, Optional, Tuple

import torch

# Global map to cache all config files
# Key: (E, N, K, device_name)
# Value: config dictionary
_CUTLASS_GROUPGEMM_CONFIG_MAP: Dict[Tuple[int, int, int, str], Dict] = {}


def _load_all_configs():
    """Load all cutlass groupgemm config files into the global map."""
    if _CUTLASS_GROUPGEMM_CONFIG_MAP:
        # Already loaded
        return

    op_name = "cutlass_groupgemm"

    # Load open source config directory
    opensource_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), op_name)
    if os.path.exists(opensource_dir):
        pattern = os.path.join(opensource_dir, "E=*-N=*-K=*-device_name=*.json")
        for config_file in glob.glob(pattern):
            filename = os.path.basename(config_file)
            try:
                # Parse filename: E={E}-N={N}-K={K}-device_name={device_name}.json
                parts = filename.replace(".json", "").split("-")
                E = int(parts[0].split("=")[1])
                N = int(parts[1].split("=")[1])
                K = int(parts[2].split("=")[1])
                device_name = parts[3].split("=")[1]

                # Load config
                with open(config_file) as f:
                    config_data = json.load(f)
                    # Convert string keys to int
                    config_data = {int(key): val for key, val in config_data.items()}

                # Store in global map
                key = (E, N, K, device_name)
                _CUTLASS_GROUPGEMM_CONFIG_MAP[key] = config_data
                logging.debug(f"Loaded config from {config_file}")
            except Exception as e:
                logging.warning(f"Failed to load config from {config_file}: {e}")

    logging.info(
        f"Loaded {len(_CUTLASS_GROUPGEMM_CONFIG_MAP)} cutlass groupgemm configurations"
    )
    try:
        import internal_source.rtp_llm.utils.register_cutlass_configs
    except:
        logging.info("internal_source not found")


def register_cutlass_groupgemm_config(
    E: int, N: int, K: int, device_name: str, config_data: Dict
):
    """Register a cutlass groupgemm configuration to the global map.

    This function allows external modules (e.g., internal_source) to register
    their configurations without the open source code needing to know about them.

    Args:
        E: Number of experts
        N: Output dimension
        K: Input dimension
        device_name: Device name (e.g., "NVIDIA_H20", "NVIDIA_L20X")
        config_data: Configuration dictionary mapping batch sizes to tile configurations
    """
    key = (E, N, K, device_name)
    _CUTLASS_GROUPGEMM_CONFIG_MAP[key] = config_data
    logging.debug(f"Registered config for E={E}, N={N}, K={K}, device={device_name}")


@functools.lru_cache
def get_cutlass_groupgemm_best_config(E: int, N: int, K: int) -> Optional[Dict]:
    """Get the best cutlass groupgemm configuration for given parameters.

    Args:
        E: Number of experts
        N: Output dimension
        K: Input dimension

    Returns:
        Configuration dictionary mapping batch sizes to tile configurations,
        or None if no configuration is found.
    """
    # Load all configs if not already loaded
    _load_all_configs()

    device_name = torch.cuda.get_device_name().replace("-", "_").replace(" ", "_")
    key = (E, N, K, device_name)

    if key in _CUTLASS_GROUPGEMM_CONFIG_MAP:
        logging.info(
            f"Using configuration for E={E}, N={N}, K={K}, device={device_name} for cutlass fp8 groupgemm."
        )
        return _CUTLASS_GROUPGEMM_CONFIG_MAP[key]
    else:
        logging.info(
            f"Config not found for E={E}, N={N}, K={K}, device={device_name}, performance might be sub-optimal."
        )
        return None
