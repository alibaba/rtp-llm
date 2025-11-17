import functools
import json
import logging
import os

import torch


@functools.lru_cache
def get_cutlass_groupgemm_best_config(E: int, N: int, K: int):
    # NVIDIA H20/NVIDIA L20X
    device_name = torch.cuda.get_device_name().replace("-", "_").replace(" ", "_")
    op_name = "cutlass_groupgemm"
    json_file_name = f"E={E},N={N},K={K},device_name={device_name}.json"

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), op_name, json_file_name
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logging.info(
                "Using default configuration from %s for cutlascs fp8 groupgemm.",
                config_file_path,
            )
            tuned_config = json.load(f)
            config_data = {int(key): val for key, val in tuned_config.items()}
            return config_data
    else:
        logging.info(
            "Config file not found at %s, performance might be sub-optimal.",
            config_file_path,
        )
        return None
