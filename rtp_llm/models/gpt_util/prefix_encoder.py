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
        self.prefix_projection = config.prefix_projection
        hidden_size = config.attn_config.head_num * config.attn_config.size_per_head
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_size, config.num_layers * hidden_size * 2),
            )
        else:
            self.embedding = torch.nn.Embedding(
                config.pre_seq_len,
                config.num_layers * config.attn_config.size_per_head * config.attn_config.kv_head_num * 2,
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
            self.config.num_layers * 2,
            self.config.attn_config.kv_head_num,
            self.config.attn_config.size_per_head,
        )
        past_key_values = past_key_values.permute(0, 2, 3, 1, 4).contiguous()
        return past_key_values
