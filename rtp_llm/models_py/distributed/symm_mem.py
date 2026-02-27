# Adapted from https://github.com/vllm-project/vllm/blob/bf214ca22625e311a2c4c0dfbf7af19128f4919c/vllm/distributed/device_communicators/symm_mem.py
import logging
import math
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

MiB = 1024 * 1024

TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES = {
    9: {
        2: 64 * MiB,  # 64 MB
        4: 64 * MiB,  # 64 MB
        6: 128 * MiB,  # 128 MB
        8: 128 * MiB,  # 128 MB
    },
    10: {
        2: 64 * MiB,  # 64 MB
        4: 64 * MiB,  # 64 MB
        6: 128 * MiB,  # 128 MB
        8: 128 * MiB,  # 128 MB
    },
}

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    torch_symm_mem_available = False
    if torch.cuda.is_available() and torch.version.cuda:
        torch_symm_mem_available = True
except ImportError:
    torch_symm_mem_available = False


class TorchSymmMemCommunicator:
    """
    Thin wrapper around torch-symmetric-memory collectives.

    This communicator:
      - Validates device capability and world size.
      - Allocates a shared symmetric buffer.
      - Chooses between 'multimem' and 'two-shot' all-reduce kernels.
      - Exposes a fast-path all_reduce() compatible with bfloat16 inputs.

    If any prerequisite is not met, the instance remains disabled and will
    decline to perform symmetric-memory all-reduce.
    """

    # Mapping: compute capability major -> supported world sizes for multimem
    # If the current (cc_major, world_size) is not listed, we fall back
    # to the two-shot path.
    _WORLD_SIZES_MULTIMEM = {
        9: [4, 6, 8],
        10: [6, 8],
    }

    def __init__(self, group: ProcessGroup, device: Union[int, str, torch.device]):
        """
        Args:
            group: Torch process group used for rendezvous and naming.
            device: Target CUDA device (index, 'cuda:X', or torch.device).
        """

        self.disabled = True

        if not torch_symm_mem_available:
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        torch.cuda.set_device(device)
        self.dtype = torch.bfloat16
        self.device = device
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.device_capability = torch.cuda.get_device_capability(device)[0]
        if self.device_capability < 9:
            logging.warning(
                "TorchSymmMemCommunicator: Device capability %s not supported, "
                "communicator is not available.",
                self.device_capability,
            )
            return
        if (
            self.world_size
            not in TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES[self.device_capability]
        ):
            logging.warning(
                "TorchSymmMemCommunicator: World size %d not supported, "
                "communicator is not available.",
                self.world_size,
            )
            return
        self.max_size = TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES[self.device_capability][
            self.world_size
        ]
        self.buffer = torch_symm_mem.empty(
            self.max_size // self.dtype.itemsize,
            device=self.device,
            dtype=self.dtype,
        )
        # Try ProcessGroup object first, fallback to group_name if needed
        handle = torch_symm_mem.rendezvous(self.buffer, group=self.group.group_name)
        if handle.multicast_ptr == 0:
            logging.warning(
                "TorchSymmMemCommunicator: torch symmetric memory "
                "multicast operations are not supported."
            )
            self.buffer = None
            self.disabled = True
            return
        self.disabled = False

    def should_torch_symm_mem_allreduce(self, inp: torch.Tensor):
        """
        Fast-path eligibility check for a given tensor.

        Conditions:
          - Communicator must be enabled.
          - dtype must be bfloat16 (matches kernel + buffer dtype).
          - Total byte size must be 4-byte aligned (hardware requirement).
          - Payload must be smaller than the symmetric-memory max size.

        Returns:
            True if the symmetric-memory path can handle this tensor.
        """
        if self.disabled:
            return False
        if inp.dtype != self.dtype:
            return False
        inp_size = inp.numel() * inp.element_size()
        # enforce 4-byte alignment
        if inp_size % 4 != 0:
            return False
        return inp_size < self.max_size

    def all_reduce(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Perform an in-place sum all-reduce via torch symmetric memory.

        Args:
            inp: Input tensor on the target CUDA device (bfloat16).
            out: Optional output tensor; if omitted, a new tensor is allocated.

        Returns:
            The reduced tensor (same shape as inp), or None if disabled.

        Implementation details:
            - Stages 'inp' into the symmetric buffer.
            - Selects 'multimem' or 'two_shot' kernel based on topology.
            - Writes the result into 'out' and returns it.
        """
        if out is None:
            out = torch.empty_like(inp)
        self.buffer[: inp.numel()].copy_(inp.view(-1))
        if self.world_size in self._WORLD_SIZES_MULTIMEM[self.device_capability]:
            torch.ops.symm_mem.multimem_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        out.copy_(self.buffer[: inp.numel()].view(out.shape))
        return out

    # adapter from torch/distributed/_symmetric_memory/__init__.py
    def should_torch_symm_mem_allgather(self, shard: torch.Tensor) -> bool:
        """
        Fast-path eligibility check for all_gather.

        Aligns with torch.distributed._symmetric_memory constraints for
        multimem_all_gather_out:
          - Communicator must be enabled (implies multicast support).
          - dtype must be bfloat16.
          - Shard must be contiguous (op requirement).
          - Shard byte size must be 4-byte aligned (hardware requirement).
          - Gather is along dim 0 only; leading_dims * world_size <= 2048
            (empirical heuristic from PyTorch fused_all_gather_matmul).
          - Total gathered size (shard * world_size) must fit in the buffer.
        """
        if self.disabled or shard.dtype != self.dtype or not shard.is_contiguous():
            return False
        shard_bytes = shard.numel() * shard.element_size()
        if shard_bytes % 4 != 0:
            return False
        leading_numel = math.prod(shard.shape[:-1]) if shard.dim() >= 2 else 1
        if leading_numel * self.world_size > 2048:
            return False
        return shard_bytes * self.world_size < self.max_size

    def all_gather(
        self, shard: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Gather shards from all ranks into a single concatenated tensor.

        Each rank contributes its local 'shard'; the result on every rank is
        the concatenation [shard_rank0, shard_rank1, ..., shard_rank_{N-1}].

        Args:
            shard: Local input shard (bfloat16, any shape).
            out:   Optional pre-allocated output tensor of shape
                   (world_size * shard.numel(),); allocated if omitted.

        Returns:
            Gathered tensor of shape (world_size, *shard.shape), or None if
            disabled.

        Implementation details:
            - Uses multimem_all_gather_out which requires multicast support
              (already validated during __init__).
            - Output is staged through the symmetric buffer and then copied
              to a regular tensor.
        """
        shard_numel = shard.numel()
        total_numel = shard_numel * self.world_size
        if out is None:
            out = torch.empty(
                (self.world_size, *shard.shape), dtype=self.dtype, device=self.device
            )
        buf_out = self.buffer[:total_numel]
        torch.ops.symm_mem.multimem_all_gather_out(
            shard.view(-1), self.group.group_name, buf_out
        )
        out.copy_(buf_out.view(self.world_size, *shard.shape))
        return out


# Use lazy initialization instead of module-level initialization
_symm_mem_comm: Optional[TorchSymmMemCommunicator] = None


def init_symm_mem_communicator(
    tp_group: ProcessGroup,
) -> Optional[TorchSymmMemCommunicator]:
    """Initialize TorchSymmMemCommunicator for TP group."""
    global _symm_mem_comm
    try:
        symm_mem_comm = TorchSymmMemCommunicator(tp_group, torch.cuda.current_device())
        if symm_mem_comm.disabled:
            logging.warning(
                f"TorchSymmMemCommunicator is disabled, skipping initialization"
            )
            return None
        _symm_mem_comm = symm_mem_comm
        return symm_mem_comm
    except Exception as e:
        # If initialization fails, fall back to regular all_reduce
        logging.warning(f"Failed to initialize TorchSymmMemCommunicator: {e}")
        return None


def get_symm_mem_communicator() -> Optional[TorchSymmMemCommunicator]:
    """Get or initialize TorchSymmMemCommunicator (lazy initialization)."""
    global _symm_mem_comm
    return _symm_mem_comm
