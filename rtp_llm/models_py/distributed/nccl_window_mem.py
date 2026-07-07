import logging
import os
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

MiB = 1024 * 1024

ENABLE_ENV = "DSV4_CP_NCCL_WINDOW_ENABLE"
_BF16_MAX_TOTAL_BYTES = 64 * MiB
_FP32_MAX_TOTAL_BYTES = 64 * MiB

_ops = None
_comms: Dict[Tuple[int, int, bool], "NcclWindowCommunicator"] = {}
_comm_ok_cache: Dict[Tuple[int, int, bool], bool] = {}
_prepared_cache: Dict[Tuple[int, int, bool, str], bool] = {}
_eligibility_cache: Dict[
    Tuple[int, torch.dtype, Tuple[int, ...], bool, bool], bool
] = {}
_disabled_by_error = False
_warned_ops_unavailable = False


def window_allgather_enabled() -> bool:
    value = os.environ.get(ENABLE_ENV, "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def _load_ops():
    global _ops
    if _ops is not None:
        return _ops
    try:
        import librtp_compute_ops
    except ImportError:
        import rtp_llm.ops

        rtp_llm.ops.ensure_compute_ops_loaded()
        import librtp_compute_ops
    if not getattr(librtp_compute_ops, "nccl_window_supported", lambda: False)():
        raise RuntimeError("prebuilt librtp_compute_ops does not support NCCL window")
    _ops = librtp_compute_ops
    return _ops


def _max_total_bytes(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return _BF16_MAX_TOTAL_BYTES
    if dtype == torch.float32:
        return _FP32_MAX_TOTAL_BYTES
    return 0


def _zero_cta_for_dtype(dtype: torch.dtype) -> bool:
    # Experiments showed that large FP32 window collectives are not worth using;
    # only <=64MiB total FP32 messages enter this path, where ZERO CTA keeps the
    # window communicator lightweight. BF16 uses the default CTA policy.
    return dtype == torch.float32


def should_use_window_allgather(shard: torch.Tensor, world_size: int) -> bool:
    if not window_allgather_enabled():
        return False
    return _tensor_policy_ok(shard, world_size)


def _tensor_policy_ok(shard: torch.Tensor, world_size: int) -> bool:
    if not shard.is_cuda or not shard.is_contiguous() or shard.dim() != 2:
        return False
    max_total_bytes = _max_total_bytes(shard.dtype)
    if max_total_bytes <= 0:
        return False
    total_bytes = shard.numel() * shard.element_size() * int(world_size)
    return 0 < total_bytes <= max_total_bytes


def _group_src_rank(group: ProcessGroup) -> int:
    try:
        return dist.get_global_rank(group, 0)
    except Exception:
        return 0


def _broadcast_unique_id(group: ProcessGroup, device: torch.device) -> torch.Tensor:
    ops = _load_ops()
    group_rank = dist.get_rank(group)
    if group_rank == 0:
        uid_cpu = ops.get_nccl_window_unique_id()
    else:
        uid_cpu = torch.empty(128, dtype=torch.uint8, device="cpu")
    uid_cuda = torch.empty_like(uid_cpu, device=device)
    if group_rank == 0:
        uid_cuda.copy_(uid_cpu.to(device=device, non_blocking=False))
    dist.broadcast(uid_cuda, src=_group_src_rank(group), group=group)
    return uid_cuda.cpu().contiguous()


def _all_ranks_true(group: ProcessGroup, device: torch.device, value: bool) -> bool:
    flag = torch.tensor([1 if value else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(int(flag.item()))


def _ops_available_on_all_ranks(group: ProcessGroup, device: torch.device) -> bool:
    global _disabled_by_error, _warned_ops_unavailable
    local_ok = True
    error = None
    try:
        _load_ops()
    except Exception as exc:
        local_ok = False
        error = exc

    if not _all_ranks_true(group, device, local_ok):
        _disabled_by_error = True
        if error is not None and not _warned_ops_unavailable:
            logging.warning("NCCL window all_gather disabled: %s", error)
            _warned_ops_unavailable = True
        return False
    return True


class NcclWindowCommunicator:
    def __init__(self, group: ProcessGroup, device: torch.device, *, zero_cta: bool):
        self.group = group
        self.device = device
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.zero_cta = zero_cta
        uid = _broadcast_unique_id(group, device)
        ops = _load_ops()
        self._comm = ops.NcclWindowAllGather(
            uid,
            self.rank,
            self.world_size,
            device.index if device.index is not None else torch.cuda.current_device(),
            bool(zero_cta),
        )

    def all_gather_direct_tensor(
        self, shard: torch.Tensor, *, key: str
    ) -> Optional[torch.Tensor]:
        return self._comm.all_gather(shard, self._full_key(shard.dtype, key))

    def prepare_all_gather(self, shard: torch.Tensor, *, key: str) -> None:
        self._comm.prepare_all_gather(shard, self._full_key(shard.dtype, key))

    def _full_key(self, dtype: torch.dtype, key: str) -> str:
        policy = "zero" if self.zero_cta else "default"
        return f"{policy}:{key}:{str(dtype)}"


def _communicator_key(
    group: ProcessGroup, dtype: torch.dtype, device: torch.device
) -> Tuple[int, int, bool]:
    zero_cta = _zero_cta_for_dtype(dtype)
    return (id(group), device.index if device.index is not None else 0, zero_cta)


def get_nccl_window_communicator(
    group: ProcessGroup, dtype: torch.dtype, device: Optional[torch.device] = None
) -> Optional[NcclWindowCommunicator]:
    global _disabled_by_error
    if _disabled_by_error:
        return None
    if not window_allgather_enabled():
        return None
    if not torch.cuda.is_available() or not dist.is_initialized():
        return None
    if _max_total_bytes(dtype) <= 0:
        return None
    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    zero_cta = _zero_cta_for_dtype(dtype)
    key = _communicator_key(group, dtype, device)
    comm = _comms.get(key)
    if comm is not None:
        return comm
    if not _ops_available_on_all_ranks(group, device):
        return None
    comm = NcclWindowCommunicator(group, device, zero_cta=zero_cta)
    _comms[key] = comm
    return comm


def try_window_allgather(
    shard: torch.Tensor, group: ProcessGroup, *, key: str
) -> Optional[torch.Tensor]:
    if not shard.is_cuda or not dist.is_initialized():
        return None

    world_size = dist.get_world_size(group)
    device = shard.device
    local_enabled = window_allgather_enabled() and not _disabled_by_error
    eligibility_key = (
        world_size,
        shard.dtype,
        tuple(int(dim) for dim in shard.size()),
        shard.is_contiguous(),
        local_enabled,
    )
    global_ok = _eligibility_cache.get(eligibility_key)
    if global_ok is None:
        local_ok = local_enabled and _tensor_policy_ok(shard, world_size)
        global_ok = _all_ranks_true(group, device, local_ok)
        _eligibility_cache[eligibility_key] = global_ok
    if not global_ok:
        return None

    comm_key = _communicator_key(group, shard.dtype, device)
    comm = None
    comm_error = None
    try:
        comm = get_nccl_window_communicator(group, shard.dtype, device=device)
    except Exception as exc:
        comm_error = exc
    if not _comm_ok_cache.get(comm_key, False):
        if not _all_ranks_true(group, device, comm is not None):
            _disable_after_error(comm_error)
            return None
        _comm_ok_cache[comm_key] = True
    assert comm is not None

    prepared_key = (*comm_key, comm._full_key(shard.dtype, key))
    if not _prepared_cache.get(prepared_key, False):
        prepare_error = None
        try:
            comm.prepare_all_gather(shard, key=key)
        except Exception as exc:
            prepare_error = exc
        if not _all_ranks_true(group, device, prepare_error is None):
            _disable_after_error(prepare_error)
            return None
        _prepared_cache[prepared_key] = True
    return comm.all_gather_direct_tensor(shard, key=key)


def _disable_after_error(error: Optional[Exception]) -> None:
    global _disabled_by_error, _warned_ops_unavailable
    _disabled_by_error = True
    if error is not None and not _warned_ops_unavailable:
        logging.warning("NCCL window all_gather disabled: %s", error)
        _warned_ops_unavailable = True


def destroy_nccl_window_communicators() -> None:
    global _comms, _comm_ok_cache, _prepared_cache, _eligibility_cache
    global _disabled_by_error, _warned_ops_unavailable
    _comms.clear()
    _comm_ok_cache.clear()
    _prepared_cache.clear()
    _eligibility_cache.clear()
    _disabled_by_error = False
    _warned_ops_unavailable = False
