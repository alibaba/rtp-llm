"""Colocated Trainer-to-RTP transactions with one native TP/EP replica per rank group."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import ray
import torch
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.replica import RolloutReplica
from rtp_native_server import (
    _audit_root,
    _base_port,
    _new_control,
    _sleep,
    _source,
    _wake,
    _write_json,
)
from rtp_topology import (
    ReplicaTopology,
    checkpoint_source_dtypes,
    required_source_names,
    training_rank_layout,
)


class RTPReplica(RolloutReplica):
    async def launch_servers(self):
        if (
            self.config.data_parallel_size != 1
            or self.config.pipeline_model_parallel_size != 1
        ):
            raise NotImplementedError(
                "use separate TP replicas for rollout DP; internal DP/PP are not implemented"
            )
        tp = int(self.config.tensor_model_parallel_size)
        if self.world_size != tp or len(self.workers) != tp:
            raise ValueError("one complete RTP TP group is required")
        infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda current: (
                        ray.get_runtime_context().get_node_id(),
                        ray.util.get_node_ip_address(),
                        list(ray.get_runtime_context().get_accelerator_ids()["GPU"]),
                    )
                )
                for worker in self.workers
            ]
        )
        if any(len(info[2]) != 1 for info in infos):
            raise RuntimeError(
                "Every colocated training worker must own exactly one physical GPU"
            )
        topology = ReplicaTopology(
            replica_rank=self.replica_rank,
            tp_size=tp,
            gpu_ids=tuple(str(info[2][0]) for info in infos),
            rank_hosts=tuple(info[1] for info in infos),
            start_port=_base_port(self.replica_rank),
            rank_port_stride=int(os.environ.get("RLLM_RTP_RANK_PORT_STRIDE", "9")),
            ep_size=int(os.environ.get("RTP_EP_SIZE", str(tp))),
        )
        node_ids = list(dict.fromkeys(info[0] for info in infos))
        local_world = int(os.environ.get("RLLM_RTP_GPUS_PER_NODE", str(tp // len(node_ids))))
        if (
            len(node_ids) != self.nnodes
            or len(node_ids) != len(topology.node_groups)
            or any(len(node["ranks"]) != local_world for node in topology.node_groups)
        ):
            raise RuntimeError(
                f"RTP requires one complete {local_world}-GPU group per assigned Ray node"
            )
        _write_json(
            _audit_root() / f"rtp_topology_{self.replica_rank}.json",
            topology.to_dict(),
        )
        from rtp_multinode_server import launch_multinode_server

        self._server_handle, self._native_owners = launch_multinode_server(
            self.model_config.path, topology.to_dict(), node_ids
        )
        self.servers = [self._server_handle]
        try:
            identity = await self._server_handle.prepare.remote()
            _write_json(_audit_root() / f"rtp_cluster_{self.replica_rank}.json", identity)
            await self._server_handle.launch_server.remote()
        except BaseException:
            await self._server_handle.shutdown.remote()
            raise
        self._server_address = (
            f"http://{topology.node_groups[0]['host']}:{topology.start_port}"
        )


class RTPServerAdapter(BaseRollout):
    def __init__(self, config, model_config, device_mesh, replica_rank=-1):
        super().__init__(config, model_config, device_mesh)
        if self.config.checkpoint_engine.backend != "naive":
            raise NotImplementedError("colocated RTP requires the naive checkpoint engine")
        if not self.config.free_cache_engine:
            raise ValueError(
                "RTP requires free_cache_engine=true to pause inference around training"
            )
        self.distributed = torch.distributed.is_initialized()
        self.global_rank = (
            torch.distributed.get_rank()
            if self.distributed
            else int(os.environ.get("RANK", "0"))
        )
        world = torch.distributed.get_world_size() if self.distributed else 1
        layout = training_rank_layout(
            self.global_rank,
            world,
            int(self.config.tensor_model_parallel_size),
            int(self.config.data_parallel_size),
            int(self.config.pipeline_model_parallel_size),
        )
        self.tp_rank = layout["tp_rank"]
        self.tp_size = layout["tp_size"]
        self.replica_rank = layout["replica_rank"]
        self.group_ranks = layout["group_ranks"]
        self.replica_count = layout["replica_count"]
        if replica_rank not in (-1, self.replica_rank):
            raise ValueError("explicit replica rank differs from the training layout")
        self.tp_group = (
            self.device_mesh["infer_tp"].get_group() if self.distributed else None
        )
        if self.distributed and self.device_mesh.get_local_rank("infer_tp") != self.tp_rank:
            raise ValueError("verl mesh and RTP TP rank order disagree")
        self.is_leader = self.tp_rank == 0
        checkpoint = Path(self.model_config.path)
        self.checkpoint_path = checkpoint
        self.hf_all_sources = set(
            json.loads((checkpoint / "model.safetensors.index.json").read_text())[
                "weight_map"
            ]
        )
        cfg = json.loads((checkpoint / "config.json").read_text())
        self.tied_embeddings = bool(
            cfg.get(
                "tie_word_embeddings",
                cfg.get("text_config", {}).get("tie_word_embeddings", False),
            )
        )
        self.hf_sources = None
        self.hf_source_dtypes = None
        self._last_sync_step = None
        self._last_sync_version = None
        self._unacknowledged_ipc = []
        _write_json(
            _audit_root() / f"rtp_worker_layout_rank{self.global_rank}.json",
            dict(
                layout,
                global_rank=self.global_rank,
                leader=self.is_leader,
                physical_cuda_mask=os.environ.get("CUDA_VISIBLE_DEVICES"),
            ),
        )

    def _broadcast(self, value):
        if not self.distributed:
            return value
        values = [value if self.is_leader else None]
        torch.distributed.broadcast_object_list(
            values, src=self.group_ranks[0], group=self.tp_group
        )
        return values[0]

    def _gather(self, value):
        if not self.distributed:
            return [value]
        values = [None] * self.tp_size
        torch.distributed.all_gather_object(values, value, group=self.tp_group)
        return values

    def _any_global_failure(self, error):
        flag = torch.tensor(
            int(error is not None),
            dtype=torch.int32,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        if self.distributed:
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        return bool(flag.item())

    async def _terminate_owned_replica(self):
        outcome = None
        if self.is_leader:
            try:
                server = ray.get_actor(f"rtp_server_{self.replica_rank}_0")
                receipt = await server.shutdown.remote()
                if not receipt.get("process_tree_terminated"):
                    raise RuntimeError(
                        "native process tree termination was not acknowledged"
                    )
                outcome = {"ok": True, "receipt": receipt}
            except BaseException as exc:
                outcome = {"ok": False, "error": repr(exc)}
        outcome = self._broadcast(outcome)
        if not outcome["ok"]:
            raise RuntimeError(
                "unacknowledged native reader; IPC storage retained: "
                + outcome["error"]
            )
        self._unacknowledged_ipc = []

    async def _group_lifecycle(self, action):
        outcome = None
        if self.is_leader:
            try:
                await (
                    _wake(self.replica_rank)
                    if action == "wake"
                    else _sleep(self.replica_rank)
                )
                outcome = {"ok": True}
            except BaseException as exc:
                outcome = {"ok": False, "error": repr(exc)}
        outcome = self._broadcast(outcome)
        if not outcome["ok"]:
            raise RuntimeError("RTP group lifecycle failed: " + outcome["error"])

    async def release(self):
        outcome = None
        if self.is_leader:
            try:
                server = ray.get_actor(f"rtp_server_{self.replica_rank}_0")
                receipt = await server.pause_for_training.remote()
                if not receipt.get("paused"):
                    raise RuntimeError("Every native TP rank must pause before training")
                outcome = {"error": None, "receipt": receipt}
            except BaseException as error:
                outcome = {"error": repr(error)}
        outcome = self._broadcast(outcome)
        if self._any_global_failure(outcome["error"]):
            await self._terminate_owned_replica()
            raise RuntimeError(
                "Native pause before training failed: "
                + str(outcome["error"] or "another replica failed")
            )

    async def resume(self, tags):
        if "kv_cache" not in tags:
            return
        error = None
        probe = None
        try:
            await self._group_lifecycle("wake")
            if self.is_leader and os.environ.get("RLLM_RTP_POST_SYNC_PROBE", "0") == "1":
                server = ray.get_actor(f"rtp_server_{self.replica_rank}_0")
                probe = await server.probe_synced_policy.remote(
                    self._last_sync_step, self._last_sync_version
                )
        except BaseException as exc:
            error = repr(exc)
        status = self._broadcast({"error": error, "probe": probe})
        error = status["error"]
        if self._any_global_failure(error):
            await self._terminate_owned_replica()
            raise RuntimeError(
                "RTP resume or post-update probe failed: "
                + str(error or "another replica failed")
            )

    def _required_sources(self):
        names = required_source_names(
            _audit_root() / f"rtp_weight_manifest_{self.replica_rank}", self.tp_size
        )
        if names - self.hf_all_sources:
            raise RuntimeError("native loader requires keys outside the checkpoint")
        return names

    def _canonical_target_dtype(self, name: str) -> torch.dtype:
        dt = (self.hf_source_dtypes or {}).get(name, "BF16")
        if dt == "F32":
            return torch.float32
        if dt == "BF16":
            return torch.bfloat16
        raise RuntimeError(f"unsupported canonical checkpoint dtype {dt!r} for {name}")

    def _local_ipc_binding(self, global_step):
        from rtp_multinode_server import canonical_uuid

        native = json.loads(
            (
                _audit_root()
                / f"rtp_weight_manifest_{self.replica_rank}"
                / f"rank-{self.tp_rank:05d}.json"
            ).read_text()
        )
        identity = canonical_uuid(
            torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
        )
        node_ip = ray.util.get_node_ip_address()
        if (
            native["owner_ip"] != node_ip
            or canonical_uuid(native["gpu_uuid"]) != identity
            or int(native["tp_rank"]) != self.tp_rank
            or int(native["tp_size"]) != self.tp_size
        ):
            raise RuntimeError(
                "CUDA IPC source and native receiver are not colocated on the same physical GPU"
            )
        proof = dict(
            passed=True,
            replica_rank=self.replica_rank,
            tp_rank=self.tp_rank,
            global_rank=self.global_rank,
            owner_ip=node_ip,
            owner_hostname=socket.gethostname(),
            gpu_uuid=identity,
            native_pid=int(native["pid"]),
            native_boot=int(native["boot_id"]),
            native_incarnation=native["incarnation"],
            global_step=int(global_step),
            checked_unix=time.time(),
        )
        _write_json(
            _audit_root()
            / f"rtp_ipc_binding_replica{self.replica_rank}_tprank{self.tp_rank}_step{global_step}.json",
            proof,
        )
        return proof

    @torch.no_grad()
    async def update_weights(self, weights, global_steps=None, **kwargs):
        if kwargs.get("peft_config") is not None:
            raise NotImplementedError("RTP updates full model parameters")
        _source()
        from rtp_llm.utils.cuda_ipc import CudaIpcV2

        full_sha = os.environ.get("RLLM_RTP_FULL_SHA256", "0") == "1"
        control = None
        failure = None
        before = None
        tensors = []
        source_digest = hashlib.sha256()
        restart = None
        if self.is_leader:
            try:
                server = ray.get_actor(f"rtp_server_{self.replica_rank}_0")
                restart = await server.ensure_native_running.remote(
                    int(global_steps or 0), self._last_sync_version
                )
                restart["error"] = None
            except BaseException as error:
                restart = {"error": repr(error)}
        restart = self._broadcast(restart)
        if restart["error"] is not None:
            failure = "Native readiness before refit failed: " + restart["error"]
        self._unacknowledged_ipc = []
        binding = None
        try:
            if failure is not None:
                raise RuntimeError(failure)
            self.hf_sources = self._required_sources()
            if self.hf_source_dtypes is None:
                self.hf_source_dtypes = checkpoint_source_dtypes(
                    self.checkpoint_path, self.hf_sources
                )
            binding = self._local_ipc_binding(int(global_steps or 0))
        except BaseException as exc:
            failure = repr(exc)
        manifest_checks = self._gather({"error": failure, "binding": binding})
        if any(x["error"] is not None for x in manifest_checks):
            failure = str([x["error"] for x in manifest_checks if x["error"] is not None])
        setup = None
        if self.is_leader:
            try:
                if failure is not None:
                    raise RuntimeError(failure)
                control = await _new_control(self.replica_rank)
                before = (await control.status())[0]
                if before.phase == "running":
                    await control.pause()
                await control.begin_update(
                    global_step=int(global_steps or 0),
                    max_buffer_bytes=int(
                        os.environ.get(
                            "RLLM_RTP_UPDATE_BUFFER_BYTES", str(16 * 1024**3)
                        )
                    ),
                    expected_source_names=sorted(self.hf_sources),
                )
                setup = {"error": None, "before_version": before.model_version}
            except BaseException as exc:
                setup = {"error": repr(exc)}
        setup = self._broadcast(setup)
        if setup["error"] is not None:
            failure = setup["error"]

        ack_bytes = 0
        # Always exhaust the collective weight iterator even on native transaction failure.
        for name, tensor in weights:
            if failure is not None:
                continue
            source_names = self._gather(name)
            if len(set(source_names)) != 1:
                failure = "trainer export name order differs across TP ranks: " + str(
                    source_names
                )
                continue
            if name in self.hf_all_sources and name not in self.hf_sources:
                continue
            item = {"name": name, "error": None}
            value = None
            source_bytes = 0
            try:
                if name not in self.hf_sources:
                    raise RuntimeError(
                        "trainer exported key outside native source manifest: " + name
                    )
                dtype = self._canonical_target_dtype(name)
                if dtype == torch.float32 and tensor.dtype != torch.float32:
                    raise RuntimeError(
                        "FP32 checkpoint source was downcast by the training exporter: "
                        + name
                    )
                local_device = torch.device("cuda", torch.cuda.current_device())
                if tensor.device.type != "cpu" and tensor.device != local_device:
                    raise RuntimeError(
                        f"Export for {name} is on {tensor.device}, expected CPU or {local_device}"
                    )
                value = tensor.detach().to(device=local_device, dtype=dtype).contiguous()
                self._unacknowledged_ipc = [value]
                source_bytes = value.numel() * value.element_size()
                if full_sha:
                    payload = memoryview(value.cpu().view(torch.uint8).numpy()).cast("B")
                    checksum = hashlib.sha256(payload).hexdigest()
                    source_digest.update(
                        name.encode()
                        + b"\0"
                        + str(tuple(value.shape)).encode()
                        + b"\0"
                        + str(value.dtype).encode()
                        + b"\0"
                    )
                    source_digest.update(payload)
                    del payload
                else:
                    checksum = ""
                    source_digest.update(
                        f"{name}:{tuple(value.shape)}:{value.dtype}\n".encode()
                    )
                # Producer device fence inside export_tensor before handing handle to RPC.
                metadata = CudaIpcV2.export_tensor(value)
                item.update(
                    shape=list(value.shape),
                    dtype=str(value.dtype),
                    sha256=checksum,
                    metadata=metadata,
                )
            except BaseException as exc:
                item["error"] = repr(exc)
            rank_items = self._gather(item)
            outcome = None
            if self.is_leader:
                try:
                    errors = [x["error"] for x in rank_items if x["error"] is not None]
                    if errors:
                        raise RuntimeError(str(errors))
                    check_keys = (
                        ["name", "shape", "dtype", "sha256"]
                        if full_sha
                        else ["name", "shape", "dtype"]
                    )
                    for peer in rank_items:
                        if any(peer[k] != rank_items[0][k] for k in check_keys):
                            raise RuntimeError(
                                "TP source tensor metadata differs across training ranks"
                            )
                    await control.push_weight(
                        name, [x["metadata"] for x in rank_items]
                    )
                    outcome = {"error": None}
                except BaseException as exc:
                    outcome = {"error": repr(exc)}
            outcome = self._broadcast(outcome)
            if outcome["error"] is not None:
                failure = outcome["error"]
                continue
            self._unacknowledged_ipc = []
            value = None
            tensors.append(
                {k: item[k] for k in ["name", "shape", "dtype", "sha256"]}
            )
            ack_bytes += source_bytes

        finish = None
        if self.is_leader:
            try:
                if failure is not None:
                    raise RuntimeError(failure)
                if {x["name"] for x in tensors} != self.hf_sources:
                    raise RuntimeError("incomplete native HF source coverage")
                replies = await control.commit_update()
                committed = replies[0]
                if (
                    committed.model_version == setup["before_version"]
                    or committed.global_step != int(global_steps or 0)
                ):
                    raise RuntimeError(
                        "RTP did not publish the expected new policy version"
                    )
                if (
                    committed.received_weights != len(tensors)
                    or committed.applied_weights <= 0
                ):
                    raise RuntimeError(
                        "RTP did not acknowledge complete internal parameter application"
                    )
                finish = {
                    "error": None,
                    "global_step": committed.global_step,
                    "model_version": committed.model_version,
                    "native_boot": restart["native_boot"],
                    "native_incarnation": restart["native_incarnation"],
                    "previous_committed_version": self._last_sync_version,
                    "restarted_since_previous_commit": restart["restarted"],
                    "received_weights": committed.received_weights,
                    "applied_weights": committed.applied_weights,
                    "rank_status": [
                        {
                            "tp_rank": x.tp_rank,
                            "tp_size": x.tp_size,
                            "model_version": x.model_version,
                            "global_step": x.global_step,
                            "phase": x.phase,
                        }
                        for x in replies
                    ],
                }
            except BaseException as exc:
                finish = {"error": repr(exc)}
            finally:
                if control is not None:
                    await control.close()
        finish = self._broadcast(finish)
        if finish["error"] is not None:
            failure = finish["error"]
        if self._any_global_failure(failure):
            await self._terminate_owned_replica()
            raise RuntimeError(
                "RTP distributed weight synchronization failed: "
                + str(failure or "another replica failed")
            )
        self._last_sync_step = finish["global_step"]
        self._last_sync_version = finish["model_version"]
        record = dict(
            engine="rtp-llm",
            replica_rank=self.replica_rank,
            tp_size=self.tp_size,
            source_training_ranks=self.group_ranks,
            global_step=finish["global_step"],
            model_version=finish["model_version"],
            before_model_version=setup["before_version"],
            native_boot=finish["native_boot"],
            native_incarnation=finish["native_incarnation"],
            native_received_weights=finish["received_weights"],
            native_applied_weights=finish["applied_weights"],
            full_source_sha256=source_digest.hexdigest(),
            full_sha256_enabled=full_sha,
            hf_tensor_count=len(tensors),
            acked_bytes=ack_bytes,
            rank_status=finish["rank_status"],
            ipc_bindings=[item["binding"] for item in manifest_checks],
            transport="native_rl_cuda_ipc_v2_per_tp_rank",
            kv_pool_retained_while_paused=True,
            finished_unix=time.time(),
        )
        if self.is_leader:
            _write_json(
                _audit_root()
                / f"rtp_sync_rank{self.replica_rank}_step{int(global_steps or 0)}.json",
                record,
            )
            print(json.dumps(record), flush=True)
        self._unacknowledged_ipc = []
