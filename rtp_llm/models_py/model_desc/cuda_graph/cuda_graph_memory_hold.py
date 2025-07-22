import torch


class CaptureMemoryHold:
    norm_output: torch.Tensor
    decode_layer_hidden_states: torch.Tensor

    def __init__(
        self, norm_output: torch.Tensor, decode_layer_hidden_states: torch.Tensor
    ):
        self.norm_output = norm_output
        self.decode_layer_hidden_states = decode_layer_hidden_states
