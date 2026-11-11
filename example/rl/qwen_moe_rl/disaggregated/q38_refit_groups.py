"""The independent Qwen3.8 split-refit rendezvous and NCCL backends."""

from dataclasses import dataclass
from datetime import timedelta
from ipaddress import IPv4Address
import re
import uuid

from rtp_llm.utils.qwen38_expert_shards import ROLLOUT_EP, TRAIN_EP


REFIT_WORLD = TRAIN_EP + ROLLOUT_EP


def normalize_uuid(value):
    return str(uuid.UUID(value.lower().removeprefix("gpu-")))


@dataclass(frozen=True)
class RefitGroupPlan:
    session: str
    master: str
    port: int
    gpu_uuids: tuple
    timeout_seconds: int = 360

    def __post_init__(self):
        address = IPv4Address(self.master)
        if address.is_unspecified or address.is_multicast or address.is_loopback:
            raise ValueError("Refit rendezvous must use the assigned training address")
        if (not isinstance(self.session, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,96}", self.session)
                or type(self.port) is not int or not 1024 <= self.port <= 65535
                or type(self.timeout_seconds) is not int or not 30 <= self.timeout_seconds <= 1200):
            raise ValueError("Invalid refit session, port or timeout")
        if type(self.gpu_uuids) is not tuple or len(self.gpu_uuids) != REFIT_WORLD:
            raise ValueError("Refit plan must bind all 96 rank UUIDs")
        ids = tuple(normalize_uuid(value) for value in self.gpu_uuids)
        if len(set(ids)) != REFIT_WORLD:
            raise ValueError("Refit plan contains duplicate GPU ownership")
        object.__setattr__(self, "gpu_uuids", ids)

    @staticmethod
    def dense_rank(refit_rank):
        if refit_rank == 0:
            return 0
        return refit_rank - TRAIN_EP + 1 if refit_rank >= TRAIN_EP else None


class OwnedRefitGroups:
    """Two NCCL backends scoped to the session and owned by the assigned rank."""

    def __init__(self, plan, rank):
        import torch
        import torch.distributed as dist

        if type(rank) is not int or not 0 <= rank < REFIT_WORLD:
            raise ValueError("Invalid split-refit world rank")
        initialized = dist.is_initialized()
        if rank < TRAIN_EP:
            if not initialized or dist.get_rank() != rank or dist.get_world_size() != TRAIN_EP:
                raise ValueError("Refit training rank differs from the live Megatron group")
        elif initialized:
            raise ValueError("The rollout relay must own an isolated worker process")
        actual = normalize_uuid(str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid))
        if actual != plan.gpu_uuids[rank]:
            raise ValueError("Refit rank is bound to the wrong CUDA device")

        timeout = timedelta(seconds=plan.timeout_seconds)
        self.store = dist.TCPStore(plan.master, plan.port, REFIT_WORLD, rank == 0,
                                   timeout=timeout, wait_for_workers=False)
        scope = dist.PrefixStore("q38-refit-" + plan.session, self.store)
        if scope.add(f"claim/{rank}", 1) != 1:
            raise RuntimeError("Refit rank already claimed this rendezvous")
        scope.set(f"ready/{rank}", actual)
        keys = [f"ready/{index}" for index in range(REFIT_WORLD)]
        scope.wait(keys)
        if tuple(value.decode() for value in scope.multi_get(keys)) != plan.gpu_uuids:
            raise RuntimeError("Refit rendezvous ownership differs from the plan")

        self.rank = rank
        self.experts = dist.ProcessGroupNCCL(dist.PrefixStore("experts", scope), rank, REFIT_WORLD, timeout)
        dense_rank = plan.dense_rank(rank)
        self.dense = None if dense_rank is None else dist.ProcessGroupNCCL(
            dist.PrefixStore("dense", scope), dense_rank, 1 + ROLLOUT_EP, timeout
        )

    def close(self):
        """Collective healthy teardown after the coordinator drains all leases."""
        if self.experts is None:
            raise RuntimeError("Refit groups already closed")
        if self.dense is not None:
            self.dense.shutdown()
            self.dense = None
        self.experts.shutdown()
        self.experts = None
        self.store = None
