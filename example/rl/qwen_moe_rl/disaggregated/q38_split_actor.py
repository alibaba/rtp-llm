"""Bind one training or refit worker to its assigned physical GPU."""

import os
from pathlib import Path
import socket

import ray
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker

from q38_refit_groups import normalize_uuid
from q38_refit_stream import RolloutRefitWorker, TrainRefitWorker


TRAIN_METHODS = frozenset({
    "begin_step", "expert_prepare", "expert_send", "expert_ack", "expert_finish",
    "dense_prepare", "dense_send", "dense_ack", "dense_finish", "finish_step", "close",
})
ROLLOUT_METHODS = frozenset({
    "expert_prepare", "expert_receive", "expert_ack", "dense_prepare",
    "dense_receive", "dense_ack", "close",
})


def _identity(role):
    resources = ray.get_runtime_context().get_accelerator_ids()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if (len(resources.get("GPU", ())) != 1 or len(visible) != 1 or not visible[0]
            or torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0):
        raise RuntimeError(f"{role} requires one Ray-assigned CUDA device")
    return {
        "role": role,
        "ip": ray.util.get_node_ip_address(),
        "node_id": ray.get_runtime_context().get_node_id(),
        "actor_id": ray.get_runtime_context().get_actor_id(),
        "pid": os.getpid(),
        "start_ticks": int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
        "uuid": normalize_uuid(str(torch.cuda.get_device_properties(0).uuid)),
    }


def _refit(worker, methods, method, args):
    if worker is None or method not in methods:
        raise RuntimeError(f"unbound or unsupported split refit operation {method}")
    return getattr(worker, method)(*args)


class SplitActorWorker(ActorRolloutRefWorker):
    def __init__(self, config, role):
        if role != "actor":
            raise ValueError("split training worker requires the actor-only role")
        super().__init__(config=config, role=role)
        self._q38_refit = None
        self._q38_placement = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def q38_placement(self):
        row = dict(_identity("train"), rank=int(os.environ["RANK"]))
        if socket.gethostbyname(socket.gethostname()) != row["ip"]:
            raise RuntimeError("checkpoint relay and Ray disagree on the worker's container IP")
        return row

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def q38_configure(self, placement):
        if self._q38_placement is not None or placement != self.q38_placement():
            raise RuntimeError("training actor placement changed or was configured twice")
        self._q38_placement = placement
        from q38_megatron_loader import install_for_verl as install_loader
        from q38_megatron_optimizer import install_for_verl as install_optimizer
        install_loader()
        install_optimizer()
        return placement

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._q38_placement is None:
            raise RuntimeError("configure training ownership and model hooks before load")
        return super().init_model()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def q38_rank(self):
        rank = dist.get_rank()
        actual = (dist.get_world_size(), parallel_state.get_tensor_model_parallel_world_size(),
                  parallel_state.get_expert_model_parallel_world_size(),
                  parallel_state.get_expert_tensor_parallel_world_size())
        if (rank != self._q38_placement["rank"] or actual != (64, 8, 64, 1)
                or parallel_state.get_expert_model_parallel_rank() != rank
                or parallel_state.get_tensor_model_parallel_rank() != rank % 8
                or parallel_state.get_data_parallel_rank() != rank // 8):
            raise RuntimeError(f"unexpected trainer topology at rank {rank}: {actual}")
        return dict(self._q38_placement, rank=rank)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def q38_bind_refit(self, plan, source_shapes):
        from megatron.bridge.models.conversion.utils import unwrap_model
        from q38_megatron_loader import iter_dense_weights, iter_local_expert_weights

        if self._q38_refit is not None:
            raise RuntimeError("training refit worker already bound")
        chunks = unwrap_model(self.actor.engine.module)
        if len(chunks) != 1 or not hasattr(chunks[0], "_q38_import_bindings"):
            raise RuntimeError("expected one local PP1 model chunk")
        model = chunks[0]
        self._q38_refit = TrainRefitWorker(
            plan, self.q38_rank()["rank"], source_shapes,
            lambda: iter_local_expert_weights(model),
            lambda: iter_dense_weights(model),
        )
        return self._q38_refit.rank

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def q38_refit(self, method, *args):
        return _refit(self._q38_refit, TRAIN_METHODS, method, args)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def update_weights(self, *args, **kwargs):
        raise RuntimeError("Qwen3.8 split refit must use the native transaction coordinator")


class SplitRolloutRelay:
    def __init__(self):
        self._q38_refit = None

    def q38_placement(self):
        return _identity("rollout")

    def q38_prepare_ipc(self):
        from rtp_llm.utils.cuda_ipc import CompileHelper

        extension = CompileHelper.CUDA_EXTENSION
        if not callable(extension.export_tensor_ipc_v2):
            raise RuntimeError("the relay Torch ABI lacks the CUDA IPC v2 exporter")
        return extension.__file__

    def q38_bind_refit(self, plan, rank, incarnation, source_shapes):
        if self._q38_refit is not None:
            raise RuntimeError("rollout refit worker already bound")
        self._q38_refit = RolloutRefitWorker(plan, rank, incarnation, source_shapes)
        return rank

    def q38_refit(self, method, *args):
        return _refit(self._q38_refit, ROLLOUT_METHODS, method, args)
