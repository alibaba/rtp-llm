"""Optional forward-owned CPU originals for CP planning, not kernel inputs."""

import torch


def cp_planning_host_mirrors(cp_info):
    chunks = getattr(cp_info, "prefill_cp_chunk_lengths_cpu", None)
    shuffle = getattr(cp_info, "prefill_shuffle_indices_cpu", None)
    if chunks is None and shuffle is None:
        return None
    # Undefined pybind tensors may arrive as None; a partial mirror is invalid.
    for name, host, device in (
        ("chunk lengths", chunks, cp_info.prefill_cp_chunk_lengths),
        ("shuffle indices", shuffle, cp_info.prefill_shuffle_indices),
    ):
        if (
            not isinstance(host, torch.Tensor)
            or host.device.type != "cpu"
            or host.dtype not in (torch.int32, torch.int64)
            or host.ndim != 1
            or not host.is_contiguous()
            or host.numel() != device.numel()
        ):
            raise ValueError(f"invalid CP host {name} mirror")
    return chunks, shuffle
