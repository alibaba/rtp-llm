"""Own one native RTP TP/EP world across nodes, preserving node-local CUDA IPC."""

from __future__ import annotations

import asyncio
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid

import psutil
import ray

from rtp_native_server import (
    RTPServer,
    _audit_root,
    _check_native_http_health,
    _deployment_root,
    _new_control,
    _source,
    _write_json,
)
from rtp_topology import ReplicaTopology


def canonical_uuid(value):
    return str(uuid.UUID(str(value).removeprefix("GPU-")))


def _native_process_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in (
        "RANK",
        "WORLD_RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "DISTRIBUTE_CONFIG_FILE",
    ):
        environment.pop(key, None)
    # The CP312 trainer limits pinned-block rounding during CPU Adam. Its
    # CP310 serving subprocess runs an older Torch that rejects this option.
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        if "pinned_max_round_threshold_mb" in environment.get(key, ""):
            parts = environment[key].split(",")
            value = ",".join(p for p in parts if p.partition(":")[0].strip() != "pinned_max_round_threshold_mb")
            if value:
                environment[key] = value
            else:
                del environment[key]
    return environment


def gpu_inventory():
    devices = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return dict(
        hostname=socket.gethostname(),
        devices=[
            dict(index=int(a), uuid=b.strip(), total_mib=int(c), used_mib=int(d))
            for a, b, c, d in csv.reader(devices.splitlines())
        ],
        processes=[
            dict(pid=int(a), uuid=b.strip(), used_mib=c.strip())
            for a, b, c in csv.reader(processes.splitlines())
            if a.strip().isdigit()
        ],
    )


class NativeNodeOwner(RTPServer):
    """One local process tree, holding a contiguous subset of native TP ranks."""

    def __init__(self, replica_rank, model_path, topology, node_index, enable_return_routed_experts=False, vmm_sleep_level=0):
        self.topology = ReplicaTopology.from_dict(topology)
        self.node_index = int(node_index)
        self.node = self.topology.node_groups[self.node_index]
        self.main_audit_root = _audit_root().resolve()
        self.boot = -1
        super().__init__(
            replica_rank,
            model_path,
            ",".join(self.node["gpu_ids"]),
            self.topology.tp_size,
        )
        self.enable_return_routed_experts = enable_return_routed_experts
        self.vmm_sleep_level = vmm_sleep_level
        self.owner_identity = dict(
            pid=os.getpid(),
            create_time=psutil.Process().create_time(),
            owner_hostname=socket.gethostname(),
            owner_ip=self.node["host"],
            node_index=self.node_index,
            audit_dir=str(self.main_audit_root),
            root=str(_deployment_root()),
            ray_actor=dict(
                name=f"rtp_native_owner_{replica_rank}_{self.node_index}",
                namespace=ray.get_runtime_context().namespace,
            ),
        )

    async def identity(self):
        return self.owner_identity

    async def start(self, boot, incarnation):
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("Native owner still holds a running incarnation")
        if self.process is not None and self._cleanup_receipt is None:
            self._terminate()
        self.boot = int(boot)
        self.poisoned = False
        self._cleanup_receipt = None
        self._owned_processes = {}
        self.process = self.process_identity = self.log = None
        own_root = (
            self.main_audit_root / f"boots/boot-{self.boot:03d}/node-{self.node_index:02d}"
        )
        own_root.mkdir(parents=True, exist_ok=True)
        os.environ["RLLM_RTP_ACCEPTANCE_DIR"] = str(own_root)
        _write_json(
            own_root / f"rtp_topology_{self.replica_rank}.json",
            self.topology.to_dict(),
        )
        for port in range(
            self.topology.start_port,
            self.topology.start_port
            + len(self.node["ranks"]) * self.topology.rank_port_stride
            + 12,
        ):
            with socket.socket() as probe:
                probe.bind((self.node["host"], port))
        environment = _native_process_environment()
        environment.update(self.topology.node_environment(self.node_index, incarnation))
        environment.update(
            RTP_MODEL_PATH=self.model_path,
            RTP_WEIGHT_MANIFEST_DIR=str(
                own_root / f"rtp_weight_manifest_{self.replica_rank}"
            ),
        )
        self.log = (own_root / f"rtp_replica_{self.replica_rank}.log").open("wb")
        command = ["/bin/bash", os.environ["RTP_SERVER_LAUNCHER"]]
        if self.enable_return_routed_experts:
            command += ["--enable_return_routed_experts", "1"]
        if self.vmm_sleep_level:
            command += ["--enable_rl_vmm_sleep", "1"]
        self.process = subprocess.Popen(
            command,
            env=environment,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        leader = psutil.Process(self.process.pid)
        self.process_identity = dict(
            pid=leader.pid,
            create_time=leader.create_time(),
            pgid=os.getpgid(leader.pid),
            replica_rank=self.replica_rank,
            boot_id=self.boot,
            incarnation=incarnation,
            node_index=self.node_index,
            owner_hostname=socket.gethostname(),
            owner_ip=self.node["host"],
            owned_tp_ranks=self.node["ranks"],
            tp_size=self.tp_size,
            root=str(_deployment_root()),
            audit_dir=str(own_root),
            started=time.time(),
            ray_actor=dict(
                name=f"rtp_native_owner_{self.replica_rank}_{self.node_index}",
                namespace=ray.get_runtime_context().namespace,
            ),
        )
        self._owned_processes[leader.pid] = leader.create_time()
        _write_json(
            own_root / f"rtp_process_{self.replica_rank}.json", self.process_identity
        )
        deadline = time.monotonic() + int(
            os.environ.get("RLLM_RTP_STARTUP_TIMEOUT", "5400")
        )
        last_error = None
        try:
            while time.monotonic() < deadline:
                self._capture_process_tree()
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Native node {self.node_index} exited {self.process.returncode}"
                    )
                try:
                    control = await _new_control(self.replica_rank)
                    try:
                        statuses = await control.status()
                    finally:
                        await control.close()
                    health = await asyncio.gather(
                        *[
                            asyncio.to_thread(
                                _check_native_http_health,
                                self.node["host"],
                                self.topology.start_port
                                + local * self.topology.rank_port_stride
                                + 5,
                                rank,
                                2.0,
                            )
                            for local, rank in enumerate(self.node["ranks"])
                        ]
                    )
                    manifests = []
                    for rank in self.node["ranks"]:
                        path = (
                            own_root
                            / f"rtp_weight_manifest_{self.replica_rank}/rank-{rank:05d}.json"
                        )
                        record = json.loads(path.read_text())
                        if (
                            record["tp_rank"] != rank
                            or record["tp_size"] != self.tp_size
                        ):
                            raise RuntimeError(
                                "Native manifest global rank/TP world mismatch"
                            )
                        process = psutil.Process(int(record["pid"]))
                        if self._owned_processes.get(
                            process.pid
                        ) != process.create_time() or not self._running(process):
                            raise RuntimeError(
                                "Native rank is not a live owned process"
                            )
                        record.update(
                            owner_hostname=socket.gethostname(),
                            owner_ip=self.node["host"],
                            owner_node_index=self.node_index,
                            boot_id=self.boot,
                            incarnation=incarnation,
                        )
                        manifests.append(record)
                    inventory = gpu_inventory()
                    by_index = {
                        str(row["index"]): canonical_uuid(row["uuid"])
                        for row in inventory["devices"]
                    }
                    expected = [
                        by_index.get(
                            gpu, canonical_uuid(gpu) if gpu.startswith("GPU-") else ""
                        )
                        for gpu in self.node["gpu_ids"]
                    ]
                    if [canonical_uuid(row["gpu_uuid"]) for row in manifests] != expected:
                        raise RuntimeError(
                            "Native rank UUIDs differ from this node GPU assignments"
                        )
                    proof = dict(
                        passed=True,
                        node_index=self.node_index,
                        boot_id=self.boot,
                        incarnation=incarnation,
                        hostname=socket.gethostname(),
                        node_ip=self.node["host"],
                        local_ranks=self.node["ranks"],
                        rank_manifests=manifests,
                        process_identity=self.process_identity,
                        http_health=health,
                        inventory=inventory,
                        statuses=[
                            dict(
                                tp_rank=r.tp_rank,
                                tp_size=r.tp_size,
                                model_version=r.model_version,
                                global_step=r.global_step,
                                phase=r.phase,
                            )
                            for r in statuses
                        ],
                        checked_unix=time.time(),
                    )
                    _write_json(
                        own_root / f"rtp_startup_readiness_{self.replica_rank}.json",
                        proof,
                    )
                    return proof
                except Exception as error:
                    last_error = repr(error)
                    await asyncio.sleep(1)
            raise TimeoutError("Native owner startup failed: " + str(last_error))
        except BaseException:
            await asyncio.to_thread(self._terminate)
            raise

    async def publish(self, manifests, topology, boot_proof):
        directory = self.main_audit_root / f"rtp_weight_manifest_{self.replica_rank}"
        directory.mkdir(parents=True, exist_ok=True)
        for record in manifests:
            _write_json(directory / f'rank-{record["tp_rank"]:05d}.json', record)
        _write_json(
            self.main_audit_root / f"rtp_topology_{self.replica_rank}.json", topology
        )
        _write_json(
            self.main_audit_root
            / f'rtp_boot_{self.replica_rank}_step{boot_proof["target_global_step"]}.json',
            boot_proof,
        )
        return dict(
            published=True,
            node_index=self.node_index,
            manifest_count=len(manifests),
        )

    async def stop_for_training(self):
        if self.process is not None:
            await asyncio.to_thread(self._terminate)
            if self.log is not None:
                self.log.close()
        result = dict(
            node_index=self.node_index,
            owner_hostname=socket.gethostname(),
            owner_ip=self.node["host"],
            boot_id=self.boot,
            **(
                self._cleanup_receipt
                or dict(process_tree_terminated=True, owned_pids=[])
            ),
        )
        result.update(finished_at_unix=time.time())
        return result


class MultiNodeRTPServer(RTPServer):
    """Stable RPC/gateway actor; native process ownership lives on each host."""

    def __init__(self, replica_rank, model_path, topology, owners, enable_return_routed_experts=False, vmm_sleep_level=0):
        self.topology = ReplicaTopology.from_dict(topology)
        self.owners = owners
        self.boot = -1
        self.incarnation = None
        self._started = False
        self._owners_active = False
        self._generation_ready = False
        self._committed_version = None
        self._lifecycle_lock = asyncio.Lock()
        self._last_suspend = None
        self._owner_identities = []
        super().__init__(replica_rank, model_path, "", self.topology.tp_size)
        self.enable_return_routed_experts = enable_return_routed_experts
        self.vmm_sleep_level = vmm_sleep_level
        self.controller_identity = dict(
            pid=os.getpid(),
            create_time=psutil.Process().create_time(),
            root=str(_deployment_root()),
            audit_dir=str(_audit_root()),
            replica_rank=replica_rank,
            owner_hostname=socket.gethostname(),
            ray_actor=dict(
                name=f"rtp_server_{replica_rank}_0",
                namespace=ray.get_runtime_context().namespace,
            ),
            node_owner_count=len(owners),
            topology=self.topology.to_dict(),
            node_owners=[],
        )
        _write_json(
            _audit_root() / f"rtp_cluster_{replica_rank}.json",
            self.controller_identity,
        )

    def _native_available(self):
        return self._started and self._generation_ready

    def _terminate(self):
        return None

    async def launch_server(self):
        await self.ensure_native_running(0, None)
        self._generation_ready = True
        return self.topology.addresses[0]

    async def _prepare_owners(self):
        if not self._owner_identities:
            identities = await asyncio.gather(
                *[owner.identity.remote() for owner in self.owners]
            )
            self._owner_identities = identities
            self.controller_identity["node_owners"] = identities
            _write_json(
                _audit_root() / f"rtp_cluster_{self.replica_rank}.json",
                self.controller_identity,
            )
        return self.controller_identity

    async def prepare(self):
        async with self._lifecycle_lock:
            return await self._prepare_owners()

    async def ensure_native_running(self, global_step, previous_committed_version):
        async with self._lifecycle_lock:
            if self.poisoned:
                raise RuntimeError("Cluster controller was permanently shut down")
            if self._started:
                return dict(
                    native_boot=self.boot,
                    native_incarnation=self.incarnation,
                    restarted=False,
                )
            if not self._owner_identities:
                await self._prepare_owners()
            self.boot += 1
            self.incarnation = uuid.uuid4().hex
            self._generation_ready = False
            self._owners_active = True
            results = await asyncio.gather(
                *[owner.start.remote(self.boot, self.incarnation) for owner in self.owners],
                return_exceptions=True,
            )
            if any(isinstance(value, BaseException) for value in results):
                await asyncio.gather(
                    *[owner.stop_for_training.remote() for owner in self.owners],
                    return_exceptions=True,
                )
                raise RuntimeError("Native startup failed: " + str(results))
            try:
                return await self._finish_start(
                    results, global_step, previous_committed_version
                )
            except BaseException:
                if self.client is not None:
                    await self.client.close()
                    self.client = None
                await asyncio.gather(
                    *[owner.stop_for_training.remote() for owner in self.owners],
                    return_exceptions=True,
                )
                self._started = self._generation_ready = False
                raise

    async def _finish_start(self, results, global_step, previous_committed_version):
        manifests = sorted(
            [row for proof in results for row in proof["rank_manifests"]],
            key=lambda row: row["tp_rank"],
        )
        if [row["tp_rank"] for row in manifests] != list(range(self.tp_size)) or len(
            {canonical_uuid(row["gpu_uuid"]) for row in manifests}
        ) != self.tp_size:
            raise RuntimeError("Native cluster is not a complete physical TP world")
        proof = dict(
            passed=True,
            replica_rank=self.replica_rank,
            native_boot=self.boot,
            native_incarnation=self.incarnation,
            target_global_step=int(global_step),
            previous_committed_version=previous_committed_version,
            nodes=results,
            manifest_count=len(manifests),
            started_unix=min(row["process_identity"]["started"] for row in results),
            ready_unix=time.time(),
        )
        await asyncio.gather(
            *[
                owner.publish.remote(manifests, self.topology.to_dict(), proof)
                for owner in self.owners
            ]
        )
        _write_json(
            _audit_root() / f"rtp_boot_{self.replica_rank}_step{global_step}.json",
            proof,
        )
        _write_json(
            _audit_root() / f"rtp_topology_{self.replica_rank}.json",
            self.topology.to_dict(),
        )
        _source()
        from transformers import AutoConfig, AutoTokenizer
        from rtp_llm.RL.token_rpc_client import TokenRpcClient

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True
        )
        config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
        gen_path = Path(self.model_path) / "generation_config.json"
        generation = json.loads(gen_path.read_text()) if gen_path.exists() else {}
        eos = generation.get("eos_token_id", self.tokenizer.eos_token_id)
        eos = eos if isinstance(eos, list) else ([] if eos is None else [eos])
        text_cfg = getattr(config, "text_config", config)
        self.client = TokenRpcClient(
            self.topology.addresses[0],
            routed_expert_shape=(text_cfg.num_hidden_layers, text_cfg.num_experts_per_tok)
            if self.enable_return_routed_experts else None,
            vocab_size=getattr(config, "text_config", config).vocab_size,
            eos_token_ids=eos,
            timeout=float(os.environ.get("RLLM_RTP_TOKEN_TIMEOUT", "600")),
        )
        self._started = True
        return dict(
            native_boot=self.boot,
            native_incarnation=self.incarnation,
            restarted=self.boot > 0,
        )

    async def probe_synced_policy(self, expected_step, expected_version):
        result = await super().probe_synced_policy(expected_step, expected_version)
        self._committed_version = expected_version
        self._generation_ready = True
        return result

    async def resume_for_rollout(self, tags, expected_step, expected_version):
        async with self._lifecycle_lock:
            control = await _new_control(self.replica_rank)
            try:
                statuses = await control.status()
                if self.vmm_sleep_level and statuses[0].phase == "paused":
                    statuses = await control.wake_up(tags=tags)
                if "kv_cache" not in tags:
                    return
                for status in statuses:
                    if (
                        not status.weights_ready
                        or not status.kv_cache_ready
                        or (expected_version is not None and status.model_version != expected_version)
                        or (expected_step is not None and status.global_step != expected_step)
                    ):
                        raise RuntimeError("RTP did not acknowledge the refitted rollout policy")
                if statuses[0].phase == "paused":
                    statuses = await control.resume()
                if any(status.phase != "running" for status in statuses):
                    raise RuntimeError("RTP did not resume generation")
                self._committed_version = statuses[0].model_version
                self._generation_ready = True
            finally:
                await control.close()

    async def pause_for_training(self):
        async with self._lifecycle_lock:
            if not self._started:
                raise RuntimeError("Native world must be live before a training pause")
            self._generation_ready = False
            control = await _new_control(self.replica_rank)
            try:
                statuses = await control.status()
                if self.vmm_sleep_level:
                    statuses = await control.sleep(level=self.vmm_sleep_level)
                elif statuses[0].phase == "running":
                    statuses = await control.pause()
            finally:
                await control.close()
            if len(statuses) != self.tp_size or any(
                value.phase != "paused" for value in statuses
            ):
                raise RuntimeError(
                    "Training pause was not acknowledged by every native rank"
                )
            proof = dict(
                paused=True,
                replica_rank=self.replica_rank,
                native_boot=self.boot,
                native_incarnation=self.incarnation,
                previous_committed_version=self._committed_version,
                kv_pool_retained=statuses[0].kv_cache_ready,
                released_vmm_bytes_by_rank=[s.released_vmm_bytes for s in statuses],
                statuses=[
                    dict(
                        tp_rank=value.tp_rank,
                        tp_size=value.tp_size,
                        model_version=value.model_version,
                        global_step=value.global_step,
                        phase=value.phase,
                    )
                    for value in statuses
                ],
                finished_at_unix=time.time(),
            )
            _write_json(
                _audit_root()
                / f"rtp_training_pause_{self.replica_rank}_version{self._committed_version}.json",
                proof,
            )
            return proof

    async def suspend_for_training(self):
        async with self._lifecycle_lock:
            if not self._owners_active:
                return self._last_suspend or dict(
                    process_tree_terminated=True, nodes=[]
                )
            self._generation_ready = False
            if self.client is not None:
                await self.client.close()
                self.client = None
            receipts = await asyncio.gather(
                *[owner.stop_for_training.remote() for owner in self.owners],
                return_exceptions=True,
            )
            if any(isinstance(row, BaseException) for row in receipts):
                raise RuntimeError(
                    "Native owner cleanup failed: " + str(receipts)
                )
            result = dict(
                process_tree_terminated=True,
                replica_rank=self.replica_rank,
                native_boot=self.boot,
                native_incarnation=self.incarnation,
                nodes=receipts,
                finished_at_unix=time.time(),
            )
            self._started = self._owners_active = self._generation_ready = False
            self._last_suspend = result
            return result

    async def wake_up(self, **kwargs):
        await self.resume_for_rollout(kwargs.get("tags", ["kv_cache"]), None, None)

    async def sleep(self):
        return await self.pause_for_training()

    async def shutdown(self):
        result = await self.suspend_for_training()
        self.poisoned = True
        _write_json(
            _audit_root() / f"rtp_cluster_shutdown_{self.replica_rank}.json", result
        )
        return result


def launch_multinode_server(model_path, topology, node_ids, *, enable_return_routed_experts=False, vmm_sleep_level=0):
    """Create the stable controller and local owners on an existing Ray cluster."""
    if vmm_sleep_level not in (0, 2):
        raise ValueError("RTP supports resident pause (0) and VMM sleep (2)")
    topology = ReplicaTopology.from_dict(topology)
    if not ray.is_initialized() or len(node_ids) != len(topology.node_groups):
        raise ValueError("A live Ray placement for every native owner is required")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Native owners must be placed on distinct Ray nodes")
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    owner_class = ray.remote(num_cpus=1, num_gpus=0)(NativeNodeOwner)
    owners = []
    runtime = {"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}}
    for index, node_id in enumerate(node_ids):
        owners.append(
            owner_class.options(
                name=f"rtp_native_owner_{topology.replica_rank}_{index}",
                max_concurrency=16,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id, soft=False
                ),
                runtime_env=runtime,
            ).remote(topology.replica_rank, model_path, topology.to_dict(), index, enable_return_routed_experts, vmm_sleep_level)
        )
    controller_class = ray.remote(num_cpus=1, num_gpus=0)(MultiNodeRTPServer)
    controller = controller_class.options(
        name=f"rtp_server_{topology.replica_rank}_0",
        max_concurrency=128,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_ids[0], soft=False),
        runtime_env=runtime,
    ).remote(topology.replica_rank, model_path, topology.to_dict(), owners, enable_return_routed_experts, vmm_sleep_level)
    return controller, owners
