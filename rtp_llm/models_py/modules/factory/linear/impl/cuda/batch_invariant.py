import torch

from rtp_llm.models_py.determinism.batch_invariant import batch_invariant_linear

from .f16_linear import CudaF16Linear


class BatchInvariantLinear(CudaF16Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return batch_invariant_linear(input, self.weight, self.bias)
