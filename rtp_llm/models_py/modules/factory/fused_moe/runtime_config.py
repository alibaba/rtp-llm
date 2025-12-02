from typing import Dict, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters


class RuntimeConfig:
    def __init__(
        self, config: GptInitModelParameters, disable_low_latency: bool = False
    ):
        self.config = config
        self.disable_low_latency = disable_low_latency

    @property
    def use_deepep_low_latency(self) -> bool:
        if self.disable_low_latency:
            return False
        return (
            self.config.moe_config.use_deepep_low_latency
            if self.config.moe_config
            else False
        )

    @property
    def model_config(self) -> GptInitModelParameters:
        return self.config
