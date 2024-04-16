from typing import Any, Dict
import torch
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters

class RtpLLMOp:
    def __init__(self, config: GptInitModelParameters, is_sp: bool, layer_weights: Any, global_weights: Any):
        super().__init__()
        assert layer_weights
        assert global_weights
        self.ft_op = torch.classes.FasterTransformer.RtpLLMOp()
        self.ft_op.init(config, layer_weights, global_weights)

    def stop(self):
        self.ft_op.stop()

    def update_lora(self, lora_infos: Dict[str, str]):
        raise Exception("not support yet")
