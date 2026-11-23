"""Views into one layer of the executor's graph-stable EPLB counters."""

import torch


class ExpertRouteStats:
    def __init__(self, logical, gpu_loads, *, num_experts, ep_size, device, live_rows=None):
        for label, tensor, size in (
            ("logical stats", logical, num_experts),
            ("GPU loads", gpu_loads, ep_size),
        ):
            if tensor.shape != (size,) or tensor.dtype != torch.int32:
                raise ValueError(
                    f"EPLB {label} must be int32 {(size,)}, got "
                    f"{tuple(tensor.shape)} {tensor.dtype}"
                )
            if tensor.device != device:
                raise ValueError(
                    "EPLB stats and MoE weights must be on the same device"
                )
        self.logical = logical
        self.gpu_loads = gpu_loads
        if live_rows is not None and (
            live_rows.shape != ()
            or live_rows.dtype != torch.int32
            or live_rows.device != device
        ):
            raise ValueError("EPLB live_rows must be a device-local int32 scalar")
        self.live_rows = live_rows

    def record(self, logical_ids, physical_ids, physical_count):
        logical = logical_ids.reshape(-1).to(torch.int64)
        physical = physical_ids.reshape(-1).to(torch.int64)
        num_experts = self.logical.numel()
        valid = (logical >= 0) & (logical < num_experts)
        if self.live_rows is not None:
            # A padded row can select a perfectly valid expert (including 0).
            # Use the replay-time row bound, independently of the chosen IDs.
            by_token = valid.reshape(-1, logical_ids.shape[-1])
            rows = torch.arange(by_token.shape[0], device=logical.device)
            valid = (by_token & (rows < self.live_rows)[:, None]).reshape(-1)
        valid = valid.to(torch.int32)
        self.logical.scatter_add_(0, logical.clamp(0, num_experts - 1), valid)
        per_rank = physical_count // self.gpu_loads.numel()
        owners = physical.clamp(0, physical_count - 1) // per_rank
        self.gpu_loads.scatter_add_(0, owners, valid)
