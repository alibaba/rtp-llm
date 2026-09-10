"""Storage for ABI 1 of the optional K3 MegaMoE native observer."""

import torch


class K3TraceBuffers:
    def __init__(self, num_ranks, capacity, topk, width, device):
        if min(num_ranks, capacity, topk, width) <= 0 or width % 128:
            raise ValueError("Positive dimensions and width divisible by 128 required")
        shape = (num_ranks, capacity, topk)
        slots = num_ranks * capacity * topk
        nbytes = slots * (17 * width + width // 32 + 4)
        self.buffer = torch.empty(nbytes, dtype=torch.uint8, device=device)
        self.tensors = {}
        offset = 0
        for name, tail, dtype in (
            ("fc1_accumulator", (2, width), torch.float32),
            ("fc1_rounded", (2, width), torch.bfloat16),
            ("activation_weighted", (width,), torch.float32),
            ("activation_fp8_bytes", (width,), torch.uint8),
            ("activation_scales_ue8m0", (width // 32,), torch.uint8),
            ("expert_ids", (), torch.int32),
        ):
            elements = slots
            for dim in tail:
                elements *= dim
            size = elements * torch.empty((), dtype=dtype).element_size()
            self.tensors[name] = (
                self.buffer[offset : offset + size].view(dtype).reshape(shape + tail)
            )
            offset += size
        assert offset == nbytes

    def reset(self):
        """Enqueue on the same stream that will launch the observed kernel."""
        self.tensors["expert_ids"].fill_(-1)

    def snapshot(self):
        """Clone after the kernel; erase stale bytes in unwritten routes."""
        valid = self.tensors["expert_ids"] >= 0
        result = {"valid": valid.clone()}
        for name, value in self.tensors.items():
            if name == "expert_ids":
                result[name] = value.clone()
            else:
                mask = valid.reshape(valid.shape + (1,) * (value.ndim - valid.ndim))
                result[name] = torch.where(mask, value, 0)
        return result
