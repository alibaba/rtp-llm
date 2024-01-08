import torch
import torch.nn as nn
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class MedusaHead(nn.Module):
    def __init__(self, config: GptInitModelParameters):
        super().__init__()
        assert config.medusa_config is not None
        self.hidden_size = config.size_per_head * config.head_num
        self.vocab_size = config.vocab_size
        self.medusa_num_layers = config.medusa_config.medusa_num_layers
        self.medusa_num_heads = config.medusa_config.medusa_num_heads
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * self.medusa_num_layers),
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(self.medusa_num_heads)
            ]
        )

    def forward(self, x: torch.Tensor):
        return torch.stack([module(x) for module in self.medusa_head], dim=0)