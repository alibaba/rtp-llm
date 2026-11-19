import torch


def dispose_tensor(x: torch.Tensor):
    """Dispose a tensor by setting it to an empty tensor."""
    x.set_(torch.empty((0,), device=x.device, dtype=x.dtype))


def tensor_spans_overlap(a, b):
    """Conservative byte-span overlap, including holes in strided views."""
    if a.device != b.device or not a.numel() or not b.numel():
        return False

    def span(tensor):
        start = tensor.data_ptr()
        last = sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
        return start, start + (last + 1) * tensor.element_size()

    a0, a1 = span(a)
    b0, b1 = span(b)
    return a0 < b1 and b0 < a1
