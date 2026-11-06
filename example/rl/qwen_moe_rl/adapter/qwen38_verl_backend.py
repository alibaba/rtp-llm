"""One Qwen3.8 EP72 rollout world colocated with 72 FSDP2 actor ranks.

The native attention TP group has eight ranks. Control, source leases and
teardown always cover the complete 72-rank expert world.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

import ray
import torch
import torch.distributed as dist
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.replica import RolloutReplica

from rtp_native_server import _audit_root, _base_port, _source, _write_json
from qwen38_refit_contract import (export_descriptor, validate_collective_stage,
                                  exact_native_ack, validate_combined_verl_mesh)

WORLD = 72
CONTROL_TOPOLOGY = dict(world_size=72, tp_size=8, dp_size=9, ep_size=72)


def _require_profile(config):
    sizes = (int(config.tensor_model_parallel_size), int(config.data_parallel_size),
             int(config.pipeline_model_parallel_size))
    if os.environ.get("RTP_LLM_RL_WORLD72") != "1" or sizes != (8, 9, 1):
        raise ValueError("Qwen3.8 rollout requires the explicit single TP8/DP9/EP72 world")
    if config.checkpoint_engine.backend != "naive" or not config.free_cache_engine:
        raise ValueError("The single-world source lease requires naive synchronization and free_cache_engine")


def _topology():
    from qwen38_world_server import World72Topology
    return World72Topology.from_dict(json.loads((_audit_root() / "rtp_topology_0.json").read_text()))


async def _control():
    _source()
    from rtp_llm.RL.replica_rpc_client import ReplicaRpcClient
    result = ReplicaRpcClient(_topology().control_addresses,
                              timeout=float(os.environ.get("RLLM_RTP_CONTROL_TIMEOUT", "600")),
                              topology=CONTROL_TOPOLOGY)
    try:
        await result.status()
    except BaseException:
        await result.close()
        raise
    return result


def _hash_source(value, source_digest, name, chunk_bytes=32 * 1024**2):
    """Hash canonical bytes through a bounded host window, never a 32GiB copy."""
    if not value.is_contiguous() or value.device.type != "cuda":
        raise ValueError("Canonical refit must supply a contiguous leased CUDA tensor")
    digest = hashlib.sha256()
    source_digest.update(name.encode() + b"\0" + str(tuple(value.shape)).encode()
                         + b"\0" + str(value.dtype).encode() + b"\0")
    flat = value.view(torch.uint8).reshape(-1)
    total = flat.numel()
    for begin in range(0, total, chunk_bytes):
        host = flat[begin:begin + chunk_bytes].cpu()
        payload = memoryview(host.numpy()).cast("B")
        digest.update(payload)
        source_digest.update(payload)
        del payload, host
    return digest.hexdigest(), total


def _confirm_worker_native_cleanup(worker, proof, expected_rank):
    """No collective: confirm real native teardown in this colocated actor."""
    from qwen38_owned_cleanup import validate_shutdown
    from qwen38_world_contract import canonical_uuid
    validate_shutdown(proof)
    if (not dist.is_initialized() or dist.get_world_size()!=WORLD or dist.get_rank()!=expected_rank
            or not isinstance(getattr(worker,'worker_dict',None),dict)):
        raise RuntimeError('World72 cleanup handoff reached the wrong initialized colocated GPU worker')
    matches=[]
    for child in worker.worker_dict.values():
        actor=getattr(child,'actor',None);engine=getattr(actor,'engine',None)
        if (getattr(child,'_is_actor',False) and hasattr(engine,'_q38_phases')
                and isinstance(getattr(child,'rollout',None),World72ServerAdapter)):
            matches.append((engine,child.rollout))
    if len(matches)!=1:
        raise RuntimeError('The colocated worker must own exactly one BF16 Qwen3.8 actor engine and world72 adapter')
    engine,adapter=matches[0]
    if engine._q38_phases.phase!='rollout' or not callable(adapter._cleanup_callback):
        raise RuntimeError('Native teardown must hand back a bound offloaded rollout phase, never an unbound/train phase')
    adapter._retained_source=None
    engine.q38_confirm_native_cleanup(proof)
    adapter._cleanup_callback=None
    if engine._q38_exporter is not None:
        raise RuntimeError('The actor still retains its canonical source exporter after native cleanup')
    torch.cuda.ipc_collect();torch.cuda.empty_cache()
    source_release=engine._q38_last_refit_source_release
    if source_release.get('all_source_views_released') is not True or source_release.get('backing_released') is not True:
        raise RuntimeError('Actor cleanup did not actually release all CUDA source views/backing')
    record=dict(world_rank=expected_rank,world_size=WORLD,phase=engine._q38_phases.phase,
        gpu_uuid=canonical_uuid(str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)),
        native_boot=proof['native_boot'],native_incarnation=proof['native_incarnation'],
        native_contexts_released=engine._q38_native_contexts_released,
        training_admission_safe=proof.get('training_admission_safe') is True,
        source_release=source_release,adapter_callback_cleared=True,finished_unix=time.time())
    _write_json(_audit_root()/f'rtp_worker_cleanup_rank{expected_rank}_boot{proof["native_boot"]}.json',record)
    return record


class World72Replica(RolloutReplica):
    async def launch_servers(self):
        _require_profile(self.config)
        if self.replica_rank != 0 or self.world_size != WORLD or len(self.workers) != WORLD:
            raise ValueError("Exactly one complete 72-GPU rollout replica is required")
        infos = await asyncio.gather(*[
            worker.__ray_call__.remote(lambda _current: (
                ray.get_runtime_context().get_node_id(), ray.util.get_node_ip_address(),
                list(ray.get_runtime_context().get_accelerator_ids()["GPU"])))
            for worker in self.workers
        ])
        if any(len(info[2]) != 1 for info in infos):
            raise RuntimeError("Each FSDP worker must own exactly one allocated GPU")
        from qwen38_world_server import World72Topology, launch_world72_server
        topology = World72Topology(
            gpu_ids=tuple(str(info[2][0]) for info in infos),
            rank_hosts=tuple(info[1] for info in infos), start_port=_base_port(0))
        node_ids = list(dict.fromkeys(info[0] for info in infos))
        if len(node_ids) != 18 or self.nnodes != 18:
            raise ValueError("All 18 four-GPU Ray nodes must belong to this one world")
        _write_json(_audit_root() / "rtp_topology_0.json", topology.to_dict())
        self._server_handle, self._native_owners = launch_world72_server(
            self.model_config.path, topology.to_dict(), node_ids)
        self.servers = [self._server_handle]
        try:
            identity = await self._server_handle.prepare.remote()
            self._world72_bound_topology=World72Topology.from_dict(identity['topology'])
            _write_json(_audit_root() / "rtp_cluster_0.json", identity)
            # This establishes the stable CPU gateway only. Native allocation
            # must wait for the actor's explicit partial-master offload.
            await self._server_handle.launch_server.remote()
        except BaseException:
            await self._server_handle.shutdown.remote()
            raise
        self._server_address = f"http://{topology.node_groups[0]['host']}:{topology.start_port}"

    async def sleep(self):
        proof=await self._server_handle.sleep.remote()
        if proof.get('native_never_started') is True:
            if proof.get('native_boot')!=-1 or proof.get('checked_native_ranks')!=[]:
                raise RuntimeError('Only the initial CPU-prepared/no-native world may skip the GPU cleanup handoff')
            return proof
        from qwen38_owned_cleanup import validate_shutdown
        from qwen38_world_contract import canonical_uuid
        validate_shutdown(proof)
        outcomes=await asyncio.gather(*[worker.__ray_call__.remote(_confirm_worker_native_cleanup,proof,rank)
            for rank,worker in enumerate(self.workers)],return_exceptions=True)
        errors=[dict(rank=rank,error=repr(row)) for rank,row in enumerate(outcomes) if isinstance(row,BaseException)]
        if errors:raise RuntimeError('Not all72 actor engines confirmed the true native cleanup: '+str(errors))
        if ([row['world_rank'] for row in outcomes]!=list(range(WORLD))
                or any(row['gpu_uuid']!=canonical_uuid(self._world72_bound_topology.gpu_uuids[rank])
                       or row['phase']!='rollout' or not row['adapter_callback_cleared']
                       for rank,row in enumerate(outcomes))):
            raise RuntimeError('Actor cleanup ACKs differ from the allocated72 physical GPU ranks')
        result=dict(proof,actor_cleanup_handoff_acks=outcomes,actor_cleanup_handoff_world_size=WORLD)
        _write_json(_audit_root()/f'rtp_sleep_worker_handoff_boot{proof["native_boot"]}.json',result)
        if (proof.get('training_admission_safe') is not True or proof.get('all_world_pause_acknowledged') is not True
                or any(row['native_contexts_released'] is not True for row in outcomes)):
            raise RuntimeError('Native cleanup-only proof released actor source backing but cannot resume training')
        return result


class World72ServerAdapter(BaseRollout):
    def __init__(self, config, model_config, device_mesh, replica_rank=-1):
        super().__init__(config, model_config, device_mesh)
        _require_profile(config)
        if not dist.is_initialized() or dist.get_world_size() != WORLD or replica_rank not in (-1, 0):
            raise ValueError("The adapter must run on the complete 72-rank actor process group")
        self.rank = dist.get_rank()
        combined_mesh = validate_combined_verl_mesh(device_mesh, self.rank)
        self.is_leader = self.rank == 0
        self.replica_rank = 0
        self._last_sync_step = None
        self._last_sync_version = None
        self._cleanup_callback = None
        self._retained_source = None
        from qwen38_training_contract import load_metadata
        _, _, policy, _mtp = load_metadata(model_config.path)
        self.required_sources = set(policy)
        _write_json(_audit_root() / f"q38-worker-{self.rank:05d}.json", dict(
            world_rank=self.rank, world_size=WORLD, tp_rank=self.rank % 8, tp_size=8,
            dp_rank=self.rank // 8, dp_size=9, ep_rank=self.rank, ep_size=72,
            logical_model_replicas=1, actual_verl_rollout_mesh=combined_mesh))

    def _broadcast(self, value):
        values = [value if self.is_leader else None]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def _gather(self, value):
        values = [None] * WORLD
        dist.all_gather_object(values, value)
        return values

    async def _leader_call(self, method, *args):
        result = None
        if self.is_leader:
            try:
                server = ray.get_actor("rtp_server_0_0")
                result = {"ok": True, "value": await getattr(server, method).remote(*args)}
            except BaseException as error:
                result = {"ok": False, "error": repr(error)}
        result = self._broadcast(result)
        if not result["ok"]:
            raise RuntimeError(f"Native world {method} failed: {result['error']}")
        return result["value"]

    async def release(self):
        proof = await self._leader_call("suspend_for_training")
        if self._cleanup_callback is None:
            raise RuntimeError("No actor phase callback was bound to this native rollout")
        self._retained_source = None
        self._cleanup_callback(proof)
        self._cleanup_callback = None
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        if proof.get("training_admission_safe") is not True or proof.get("all_world_pause_acknowledged") is not True:
            raise RuntimeError("Native processes exited, but the complete-world training transition failed")

    async def resume(self, tags):
        if "kv_cache" in tags:
            await self._leader_call("wake_and_probe", self._last_sync_step, self._last_sync_version)

    def _binding(self):
        import socket
        topology = _topology()
        manifest = json.loads((_audit_root() / "rtp_weight_manifest_0" /
                               f"rank-{self.rank:05d}.json").read_text())
        gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid).lower().removeprefix("gpu-")
        native_uuid = str(manifest["gpu_uuid"]).lower().removeprefix("gpu-")
        if (manifest["world_rank"] != self.rank or manifest["tp_rank"] != self.rank % 8
                or manifest["tp_size"] != 8 or manifest["dp_size"] != 9 or manifest["ep_size"] != 72
                or manifest["owner_ip"] != ray.util.get_node_ip_address()
                or topology.rank_hosts[self.rank] != ray.util.get_node_ip_address()
                or native_uuid != gpu_uuid or set(manifest["names"]) != self.required_sources):
            raise RuntimeError("Native rank, source coverage or CUDA IPC physical GPU binding differs")
        return dict(world_rank=self.rank, native_pid=manifest["pid"], gpu_uuid=gpu_uuid,
                    owner_hostname=socket.gethostname(), owner_ip=manifest["owner_ip"],
                    native_boot=manifest["boot_id"], native_incarnation=manifest["incarnation"])

    @torch.no_grad()
    async def update_weights(self, weights, global_steps=None, **kwargs):
        # A single rank must never leave before its peers reach the same next
        # collective. Validate both coverage and iteration order on all 72.
        preflight = dict(error=None)
        try:
            step = int(global_steps or 0)
            preflight.update(export_descriptor(weights, self.required_sources,
                peft_config=kwargs.get("peft_config"), step=step))
            source_order = tuple(weights.order)
        except BaseException as error:
            preflight["error"] = repr(error)
        validate_collective_stage(self._gather(preflight), "canonical exporter preflight",
                                  identity_fields=("source_count", "order_sha256", "global_step"))
        self._cleanup_callback = weights.confirm_native_cleanup
        control = None
        failure = None
        tensors = []
        source_digest = hashlib.sha256()
        try:
            restart = await self._leader_call("ensure_native_running", step, self._last_sync_version)
            binding = None
            try:
                binding = self._binding()
            except BaseException as error:
                failure = repr(error)
            checks = self._gather(dict(error=failure, binding=binding))
            if any(row["error"] for row in checks):
                raise RuntimeError("Native world binding failed: " + str([r["error"] for r in checks if r["error"]]))
            setup = None
            if self.is_leader:
                try:
                    control = await _control()
                    before = (await control.status())[0]
                    if before.phase == "running":
                        await control.pause()
                    await control.begin_update(global_step=step, max_buffer_bytes=34 * 1024**3)
                    setup = dict(error=None, before_version=before.model_version)
                except BaseException as error:
                    setup = dict(error=repr(error))
            setup = self._broadcast(setup)
            if setup["error"]:
                raise RuntimeError(setup["error"])
            _source()
            from rtp_llm.utils.cuda_ipc import CudaIpcV2
            iterator_status = dict(error=None)
            try:
                iterator = iter(weights)
            except BaseException as error:
                iterator_status["error"] = repr(error)
            validate_collective_stage(self._gather(iterator_status), "canonical iterator creation")
            for expected_name in source_order:
                next_status = dict(error=None, name=None)
                try:
                    name, tensor = next(iterator)
                    next_status["name"] = name
                    if name != expected_name:
                        raise ValueError("Canonical iterator differs from its collectively accepted source order")
                except BaseException as error:
                    next_status["error"] = repr(error)
                validate_collective_stage(self._gather(next_status), "canonical source next",
                                          identity_fields=("name",))
                item = dict(name=name, error=None)
                self._retained_source = tensor
                try:
                    expected_dtype = torch.float32 if name.endswith((".linear_attn.A_log", ".linear_attn.dt_bias")) else torch.bfloat16
                    if tensor.dtype != expected_dtype or tensor.device != torch.device("cuda", torch.cuda.current_device()):
                        raise ValueError("Canonical source dtype or device differs; the bridge must not clone its lease")
                    checksum, source_bytes = _hash_source(tensor, source_digest, name)
                    item.update(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=checksum,
                                metadata=CudaIpcV2.export_tensor(tensor), source_bytes=source_bytes)
                except BaseException as error:
                    item["error"] = repr(error)
                records = self._gather(item)
                outcome = None
                if self.is_leader:
                    try:
                        errors = [row["error"] for row in records if row["error"]]
                        if errors:
                            raise RuntimeError(str(errors))
                        fields = ("name", "shape", "dtype", "sha256", "source_bytes")
                        if any(any(row[key] != records[0][key] for key in fields) for row in records):
                            raise RuntimeError("Canonical source identity differs across the 72 training ranks")
                        replies = await control.push_weight(name, [row["metadata"] for row in records])
                        ranks = exact_native_ack([reply.world_rank for reply in replies])
                        outcome = dict(error=None, ranks=ranks)
                    except BaseException as error:
                        outcome = dict(error=repr(error))
                outcome = self._broadcast(outcome)
                if outcome["error"]:
                    # Keep the backing until all 72 readers have exited. Do not
                    # ask the iterator for another collective after a failed ACK.
                    del tensor
                    raise RuntimeError(outcome["error"])
                ack_status = dict(error=None, name=name)
                try:
                    exact_native_ack(outcome["ranks"])
                    weights.acknowledge(name, successful_native_ranks=outcome["ranks"])
                except BaseException as error:
                    ack_status["error"] = repr(error)
                validate_collective_stage(self._gather(ack_status), "canonical source lease ACK",
                                          identity_fields=("name",))
                tensors.append({key: item[key] for key in ("name", "shape", "dtype", "sha256", "source_bytes")})
                self._retained_source = None
                del tensor
                if self.is_leader and (len(tensors) % 16 == 0 or len(tensors) == len(self.required_sources)):
                    print("Q38_REFIT " + json.dumps(dict(acked=len(tensors), required=len(self.required_sources),
                                                         last_source=name, global_step=step)), flush=True)
            finish = None
            if self.is_leader:
                try:
                    if {row["name"] for row in tensors} != self.required_sources:
                        raise RuntimeError("The native transaction omitted a main-policy parameter")
                    replies = await control.commit_update()
                    committed = replies[0]
                    finish = dict(error=None, model_version=committed.model_version,
                                  global_step=committed.global_step,
                                  received_weights=committed.received_weights,
                                  ranks=[reply.world_rank for reply in replies])
                except BaseException as error:
                    finish = dict(error=repr(error))
            finish = self._broadcast(finish)
            if finish["error"]:
                raise RuntimeError(finish["error"])
            if (finish["global_step"] != step or finish["received_weights"] != len(self.required_sources)
                    or finish["model_version"] == setup["before_version"] or set(finish["ranks"]) != set(range(WORLD))):
                raise RuntimeError("Native commit did not publish the complete target policy")
            close_status = dict(error=None)
            try:
                release_receipt = weights.close()
            except BaseException as error:
                close_status["error"] = repr(error)
            validate_collective_stage(self._gather(close_status), "canonical export close")
            self._last_sync_step = step
            self._last_sync_version = finish["model_version"]
            if self.is_leader:
                _write_json(_audit_root() / f"rtp_sync_rank0_step{step}.json", dict(
                    engine="rtp-llm", logical_model_replicas=1, **CONTROL_TOPOLOGY,
                    before_model_version=setup["before_version"], **finish,
                    full_source_sha256=source_digest.hexdigest(), tensors=tensors,
                    native_boot=restart["native_boot"], native_incarnation=restart["native_incarnation"],
                    ipc_bindings=[row["binding"] for row in checks], source_release=release_receipt,
                    finished_unix=time.time()))
        except BaseException:
            # The exporter may retain a failed borrowed source until native
            # teardown is proven. Drop this consumer view before invoking it.
            self._retained_source = None
            try:
                del tensor
            except UnboundLocalError:
                pass
            proof = await self._leader_call("suspend_for_training")
            self._cleanup_callback(proof)
            self._cleanup_callback = None
            raise
        finally:
            if control is not None:
                await control.close()
