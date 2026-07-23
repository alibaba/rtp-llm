import torch.library as torch_library


def ensure_torch_library_wrap_triton() -> None:
    """Provide the identity fallback expected by older supported Torch releases."""
    try:
        torch_library.wrap_triton
    except AttributeError:

        def wrap_triton(fn):
            return fn

        torch_library.wrap_triton = wrap_triton
