import torch

from rtp_llm.config.model_config import ModelConfig


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.prefix_projection = config.prefix_projection_
        hidden_size = config.head_num_ * config.size_per_head_
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_size, config.layer_num_ * hidden_size * 2),
            )
        else:
            self.embedding = torch.nn.Embedding(
                config.pre_seq_len,
                config.layer_num_ * config.size_per_head_ * config.head_num_kv_ * 2,
            )

    # input shape: [batch_size, pre_seq_len]
    # output shape: [batch_size, layer_num * 2, head_num, pre_seq_len, size_per_head]
    def forward(self, prefix: torch.Tensor):
        batch_size = prefix.size(0)
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        past_key_values = past_key_values.view(
            batch_size,
            self.config.pre_seq_len,
            self.config.layer_num_ * 2,
            self.config.head_num_kv_,
            self.config.size_per_head_,
        )
        past_key_values = past_key_values.permute(0, 2, 3, 1, 4).contiguous()
        return past_key_values
