import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, p=-1.0, eps=1e-8, use_bias=False):
        """
            Root Mean Square Layer Normalization
        :param dim: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param use_bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = dim
        self.p = p
        self.use_bias = use_bias

        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.register_parameter("weight", self.weight)

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(dim))
            self.register_parameter("bias", self.bias)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.use_bias:
            return self.weight * x_normed + self.bias

        return self.weight * x_normed

