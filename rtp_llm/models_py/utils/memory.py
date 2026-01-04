import torch


def dispose_tensor(x: torch.Tensor):
    """Dispose a tensor by setting it to an empty tensor."""
    x.set_(torch.empty((0,), device=x.device, dtype=x.dtype))
