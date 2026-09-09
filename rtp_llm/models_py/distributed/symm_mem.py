# Adapted from https://github.com/vllm-project/vllm/blob/bf214ca22625e311a2c4c0dfbf7af19128f4919c/vllm/distributed/device_communicators/symm_mem.py
import logging
import math
from datetime import timedelta
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


# Bounded on purpose. The process groups built by init_distributed_environment
# carry a 100-year timeout so that idle workers park instead of crashing, which
# means anything that waits on a peer there can never fail loudly. The symmetric
# memory handshake below waits on the rendezvous store instead, with a real
# deadline, so a genuinely stuck peer reports itself rather than hanging forever.
_AGREE_TIMEOUT_S = 120

# Bumped once per agreement so a second initialize() on the same group cannot
# read the previous round's keys. Every rank runs the same init sequence, so the
# counter advances in lockstep and the keys stay aligned across ranks.
_agree_epoch = 0


def _agree_across_group(group: ProcessGroup, local_ok: bool, tag: str) -> bool:
    """AND-reduce `local_ok` across every rank of `group`.

    Symmetric-memory setup is collective: torch_symm_mem.rendezvous() only
    returns once *every* rank of the group has called it. So the decision to use
    symmetric memory has to be unanimous and has to be made before anybody
    rendezvouses -- a rank that quietly opts out after its peers are already
    inside the rendezvous leaves them blocked for the group timeout (100 years).

    The vote goes through the rendezvous store rather than an all_reduce: the
    thing we are usually voting on is a CUDA/driver failure, and issuing another
    device collective to decide whether the device is usable just reintroduces
    the same failure inside the recovery path.
    """
    global _agree_epoch
    _agree_epoch += 1
    epoch = _agree_epoch

    store = dist.distributed_c10d._get_default_store()
    group_size = dist.get_world_size(group)
    group_rank = dist.get_rank(group)
    group_name = getattr(group, "group_name", None) or "default"

    prefix = f"rtp_llm/symm_mem/{tag}/{group_name}/{epoch}/"
    store.set(prefix + str(group_rank), b"1" if local_ok else b"0")
    keys = [prefix + str(r) for r in range(group_size)]
    store.wait(keys, timedelta(seconds=_AGREE_TIMEOUT_S))

    votes = [store.get(k) == b"1" for k in keys]
    if not all(votes):
        logging.warning(
            "TorchSymmMemCommunicator: disabling symmetric memory for %s on all "
            "%d rank(s) of group %s; rank(s) %s could not initialize it.",
            tag,
            group_size,
            group_name,
            [r for r, ok in enumerate(votes) if not ok],
        )
        return False
    return True


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
        self.buffer = None

        if not torch_symm_mem_available:
            # Build-level property, identical on every rank of the job, so
            # returning here cannot split the group.
            return

        if isinstance(device, int):
            from rtp_llm.device.device_type import get_device_type, DeviceType
            _dn = "npu" if get_device_type() == DeviceType.Ascend else ("hip" if get_device_type() == DeviceType.ROCm else "cuda")
            device = torch.device(f"{_dn}:{device}")
        elif isinstance(device, str):
            device = torch.device(device)

        # _create_process_groups calls this once per TP group for *every* world
        # rank, so with dp_size > 1 a non-member of `group` lands here. It has
        # nothing to allocate and no vote to cast, and it never enters the
        # rendezvous, so returning early cannot split the group.
        non_member = getattr(dist.GroupMember, "NON_GROUP_MEMBER", None)
        if group is None or group is non_member:
            return
        if dist.get_rank(group) < 0:
            return

        torch.cuda.set_device(device)
        self.dtype = torch.bfloat16
        self.device = device
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.device_capability = torch.cuda.get_device_capability(device)[0]

        # Everything below is a *local* verdict. Collect it instead of returning,
        # so the group can agree on one answer before anyone rendezvouses.
        local_ok = True
        reason = ""

        if self.device_capability < 9:
            local_ok = False
            reason = f"device capability {self.device_capability} not supported"
        elif (
            self.world_size
            not in TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES[self.device_capability]
        ):
            local_ok = False
            reason = f"world size {self.world_size} not supported"

        if local_ok:
            self.max_size = TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES[
                self.device_capability
            ][self.world_size]
            try:
                self.buffer = torch_symm_mem.empty(
                    self.max_size // self.dtype.itemsize,
                    device=self.device,
                    dtype=self.dtype,
                )
            except Exception as e:
                # Multicast allocation is the flaky step under memory pressure
                # ("CUDA driver error: unknown error"). Record it and let the
                # vote below take the whole group down the fallback path; do NOT
                # raise, or the peers stay parked in the rendezvous.
                local_ok = False
                reason = f"symmetric buffer allocation failed: {e}"
                self.buffer = None

        if reason:
            logging.warning(
                "TorchSymmMemCommunicator: %s, voting to disable for this group.",
                reason,
            )

        # Barrier #1: unanimous go/no-go *before* the collective rendezvous.
        if not _agree_across_group(self.group, local_ok, "alloc"):
            self.buffer = None
            self.disabled = True
            return

        handle = torch_symm_mem.rendezvous(self.buffer, group=self.group.group_name)

        # The post-rendezvous checks are local verdicts too. If they split the
        # group, some ranks would call multimem_all_reduce_ while others fall
        # back to NCCL, which diverges at request time instead of at startup.
        local_ok = True
        if handle.multicast_ptr == 0:
            local_ok = False
            reason = "torch symmetric memory multicast operations are not supported"
        elif not hasattr(torch.ops.symm_mem, "multimem_all_gather_out"):
            local_ok = False
            reason = (
                "torch.ops.symm_mem.multimem_all_gather_out is not available in "
                "this PyTorch build"
            )
        if not local_ok:
            logging.warning(
                "TorchSymmMemCommunicator: %s, voting to disable for this group.",
                reason,
            )

        # Barrier #2: unanimous go/no-go before anyone takes the fast path.
        if not _agree_across_group(self.group, local_ok, "rendezvous"):
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
    """Initialize TorchSymmMemCommunicator for TP group.

    Not wrapped in a catch-all any more. The constructor already absorbs the one
    recoverable failure (the local multicast allocation) and votes on it, so all
    ranks agree to fall back together. Anything still escaping means the group
    agreement or the collective rendezvous itself failed, and there is no safe
    rank-local fallback for that: returning None on a single rank is precisely
    what used to leave its peers parked in torch_symm_mem.rendezvous() until the
    100-year group timeout, surfacing two minutes later as a misleading
    "CpuTpBroadcaster connect(...) failed after 2400 attempts". Let it raise so
    startup fails fast, on every rank, with the real reason.
    """
    global _symm_mem_comm
    symm_mem_comm = TorchSymmMemCommunicator(tp_group, torch.cuda.current_device())
    if symm_mem_comm.disabled:
        logging.warning("TorchSymmMemCommunicator is disabled, skipping initialization")
        return None
    _symm_mem_comm = symm_mem_comm
    return symm_mem_comm


def get_symm_mem_communicator() -> Optional[TorchSymmMemCommunicator]:
    """Get or initialize TorchSymmMemCommunicator (lazy initialization)."""
    global _symm_mem_comm
    return _symm_mem_comm
