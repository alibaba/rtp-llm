"""Vocabulary-parallel draft softmax with invocation-local scratch storage."""

import torch


def dspark_softmax(logits: torch.Tensor) -> torch.Tensor:
    if not (
        logits.is_cuda
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and logits.is_contiguous()
        and 1 < logits.shape[0] <= 32
        and 32768 <= logits.shape[1] <= 262144
    ):
        return torch.softmax(logits, -1)
    from flashinfer.sampling import get_sampling_module

    # FlashInfer's public convenience wrapper shares a process-wide workspace.
    # Keep scratch private so concurrent streams and graph instances cannot race.
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=logits.device)
    return get_sampling_module().softmax(workspace, logits, None, 1.0, False)
