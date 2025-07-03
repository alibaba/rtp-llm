class BaseAddBiasResLayerNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.beta = beta
        self.variance_epsilon = eps
    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

class AddBiasResLayerNormROCmTorch(BaseAddBiasResLayerNorm):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)
    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor, bias: torch.Tensor):
        output = F.layer_norm(
            input=hidden_states,
            normalized_shape=(hidden_states.shape[-1],),
            weight=self.weight.data,
            bias=bias,
            eps=self.variance_epsilon,
        )
        return output

class AddBiasResLayerNorm(BaseAddBiasResLayerNorm):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        bias: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if hidden_states.shape[0] > 32 and hidden_states.shape[1] <= 768:
            return layer_norm(hidden_states, self.weight.data, bias, self.variance_epsilon, x_bias = None)
        else:
            output = torch.empty_like(hidden_states)
            torch.ops.rtp_llm_ops.layernorm(output, hidden_states, self.weight.data, self.beta, self.variance_epsilon, 0)
            return hidden_states
