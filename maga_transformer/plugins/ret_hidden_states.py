from typing import Any
import torch

class CustomPlugin(object):
    def modify_response_plugin(self, response: str, hidden_states: torch.Tensor, **kwargs: Any):
        return hidden_states