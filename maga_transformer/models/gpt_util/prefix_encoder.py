import torch
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters

class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config: GptInitModelParameters):
        super().__init__()
        self.config = config
        self.prefix_projection = config.prefix_projection
        hidden_size = config.head_num * config.size_per_head
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_size, config.layer_num * hidden_size * 2)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.layer_num * config.size_per_head * config.head_num_kv * 2)

    # input shape: [batch_size, pre_seq_len]
    # output shape: [batch_size, layer_num * 2, head_num, pre_seq_len, size_per_head]
    def forward(self, prefix: torch.Tensor):
        batch_size = prefix.size(0)
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        past_key_values = past_key_values.view(batch_size, self.config.pre_seq_len, self.config.layer_num * 2, 
                                               self.config.head_num_kv, self.config.size_per_head)
        past_key_values = past_key_values.permute(0, 2, 3, 1, 4).contiguous()
        return past_key_values