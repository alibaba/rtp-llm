"""Integrate the pinned on-policy verl backend with the separate RTP world."""

from dataclasses import dataclass
from ipaddress import IPv4Address
import json
from pathlib import Path
import re

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy
from verl.utils.ray_utils import auto_await

from q38_driver_recipe import GPUS_PER_NODE, ROLLOUT_NODES, TRAIN_NODES
from q38_refit_groups import RefitGroupPlan, normalize_uuid
from q38_native32_topology import Native32Topology
from q38_split_actor import SplitRolloutRelay
from q38_split_native import NativeCoordinator, NativeServerManager, _write_json


@dataclass(frozen=True)
class Node:
    ip: str
    uuids: tuple
    node_id: str

    def __post_init__(self):
        ip = IPv4Address(self.ip)
        if not ip.is_private or ip.is_unspecified or ip.is_multicast or ip.is_loopback:
            raise ValueError("claim must use the authorized container IPs")
        if len(self.uuids) != GPUS_PER_NODE:
            raise ValueError("one claimed container must have exactly four GPU UUIDs")
        if not re.fullmatch(r"[a-f0-9]{56}", self.node_id):
            raise ValueError("one current Ray NodeID is required per container")
        object.__setattr__(self, "uuids", tuple(normalize_uuid(x) for x in self.uuids))


@dataclass(frozen=True)
class Deployment:
    run_id: str
    ray_session: str
    native_incarnation: str
    refit_port: int
    native_start_port: int
    output: str
    metadata: str
    manifest: str
    train: tuple
    rollout: tuple

    @classmethod
    def from_json(cls, path):
        args = json.loads(Path(path).read_bytes())
        for role in ("train", "rollout"):
            args[role] = tuple(Node(**item) for item in args[role])
        return cls(**args)

    def __post_init__(self):
        if (len(self.train), len(self.rollout)) != (TRAIN_NODES, ROLLOUT_NODES):
            raise ValueError("expected the exact 16 train and eight rollout containers")
        nodes = self.train + self.rollout
        if (len({node.ip for node in nodes}) != 24 or len({node.node_id for node in nodes}) != 24
                or len({u for node in nodes for u in node.uuids}) != 96):
            raise ValueError("GPU and container identities must be disjoint")
        if not re.fullmatch(r"q38-[a-z0-9-]{12,79}", self.run_id):
            raise ValueError("fresh scoped split training run ID required")
        if not re.fullmatch(r"q38-[a-z0-9-]{6,85}", self.native_incarnation):
            raise ValueError("fresh scoped native incarnation required")
        if not re.fullmatch(r"session_[0-9_-]+", self.ray_session):
            raise ValueError("exact active Ray session required")
        if type(self.refit_port) is not int or not 1024 <= self.refit_port <= 65535:
            raise ValueError("dedicated refit TCPStore port required")
        if not all(Path(path).is_absolute() for path in (self.output, self.metadata, self.manifest)):
            raise ValueError("deployment paths must be absolute")
        if Path(self.output).name != self.run_id:
            raise ValueError("evidence directory must be scoped to the run ID")

    def topology(self):
        return Native32Topology(
            tuple(node.ip for node in self.rollout),
            tuple(node.uuids for node in self.rollout),
            self.native_start_port,
        )


def ray_claim(deploy):
    if not ray.is_initialized() or ray.get_runtime_context().get_session_name() != deploy.ray_session:
        raise RuntimeError("driver is outside the claimed Ray session")
    if ray.get_runtime_context().namespace != deploy.run_id:
        raise RuntimeError("driver is outside the claimed training namespace")
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    by_ip = {n["NodeManagerAddress"]: n for n in nodes}
    if len(nodes) != 24 or set(by_ip) != {n.ip for n in deploy.train + deploy.rollout}:
        raise RuntimeError("live Ray membership does not match the authorized containers")
    for role, own in (("train", deploy.train), ("rollout", deploy.rollout)):
        for node in own:
            values = by_ip[node.ip]["Resources"]
            if (by_ip[node.ip]["NodeID"] != node.node_id
                    or values.get("GPU") != 4 or values.get("q38_" + role) != 4
                    or values.get("q38_" + ("train" if role == "rollout" else "rollout"), 0)
                    or values.get("node:" + node.ip, 0) < 0.0004
                    or values.get("q38_head", 0) != (1 if node == deploy.train[0] else 0)):
                raise RuntimeError(f"unexpected Ray resources on {role} container {node.ip}")
    return by_ip


def _groups(deploy, nodes, role):
    groups = []
    for index, node in enumerate(nodes):
        bundle = {"CPU": 1, "GPU": 1, "node:" + node.ip: 0.0001, "q38_" + role: 1}
        groups.append(placement_group(
            [dict(bundle) for _ in range(GPUS_PER_NODE)], strategy="STRICT_PACK",
            name=f"{deploy.run_id}-{role}-{index:02d}", lifetime="detached",
        ))
    ray.get([group.ready() for group in groups], timeout=180)
    return groups


def _check_workers(records, nodes, role, ray_nodes):
    expected = {(node.ip, gpu) for node in nodes for gpu in node.uuids}
    actual = {(row["ip"], row["uuid"]) for row in records}
    if (len(records) != len(expected) or expected != actual
            or len({row["actor_id"] for row in records}) != len(records)
            or any(row["role"] != role or row["node_id"] != ray_nodes[row["ip"]]["NodeID"] for row in records)):
        raise RuntimeError(f"Ray worker assignment is outside the claimed {role} GPU allocation")


class SplitCheckpointManager:
    def __init__(self, coordinator):
        self.coordinator = coordinator

    @auto_await
    async def update_weights(self, global_steps=None):
        return await self.coordinator.refit.remote(global_steps)

    @auto_await
    async def sleep_replicas(self):
        await self.coordinator.sleep.remote()


def initialize_backend(backend, deploy):
    from rllm.trainer.verl import verl_backend as vm
    from rtp_llm.models.qwen3_next.qwen38_bootstrap_contract import load_bootstrap_contract

    config = backend.config
    if backend.is_separated or backend.use_reference_policy or backend.ref_in_actor:
        raise ValueError("split Qwen3.8 requires synchronous GRPO with one actor and no reference policy")
    ray_nodes = ray_claim(deploy)
    head = deploy.train[0].ip
    if ray.util.get_node_ip_address() != head:
        raise RuntimeError("verl task runner must run on the claimed training head")
    backend.resource_pool_manager.create_resource_pool()
    pool = backend.resource_pool_manager.get_resource_pool(vm.Role.ActorRollout)
    if pool.store != [4] * TRAIN_NODES or pool.max_colocate_count != 1 or not pool.use_gpu:
        raise RuntimeError("verl did not provide one actor slot per training GPU")
    pool.pgs = _groups(deploy, deploy.train, "train")
    rollout_groups = _groups(deploy, deploy.rollout, "rollout")
    output = Path(deploy.output)
    _write_json(output / "placement-groups.json", {
        "train": [group.id.hex() for group in pool.pgs],
        "rollout": [group.id.hex() for group in rollout_groups],
    })

    worker = vm.RayClassWithInitArgs(
        cls=backend.role_worker_mapping[vm.Role.ActorRollout], config=config.actor_rollout_ref, role="actor",
    )
    backend.actor_rollout_wg = backend.ray_worker_group_cls(
        resource_pool=pool, ray_cls_with_init=worker, name_prefix=deploy.run_id + "-", detached=True,
        **vm.build_wg_kwargs(config, backend.device_name),
    )
    trainers = list(backend.actor_rollout_wg.workers)
    relay_cls = ray.remote(num_cpus=1, num_gpus=1, max_restarts=0)(SplitRolloutRelay)
    relays = [
        relay_cls.options(
            name=f"{deploy.run_id}-relay-{4*node+local:02d}", lifetime="detached",
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=group, placement_group_bundle_index=local,
                placement_group_capture_child_tasks=False,
            ),
        ).remote()
        for node, group in enumerate(rollout_groups) for local in range(GPUS_PER_NODE)
    ]
    training_rows = ray.get([worker.q38_placement.remote() for worker in trainers], timeout=180)
    rollout_rows = ray.get([relay.q38_placement.remote() for relay in relays], timeout=180)
    _check_workers(training_rows, deploy.train, "train", ray_nodes)
    _check_workers(rollout_rows, deploy.rollout, "rollout", ray_nodes)
    expected_ips = [ip for ip in sorted(node.ip for node in deploy.train) for _ in range(GPUS_PER_NODE)]
    if [row["ip"] for row in training_rows] != expected_ips or [row["rank"] for row in training_rows] != list(range(64)):
        raise RuntimeError("Megatron worker rank-to-IP order differs from the checkpoint relay allowlist")
    _write_json(output / "worker-ownership.json", {"train": training_rows, "rollout": rollout_rows})
    ray.get([actor.q38_configure.remote(row) for actor, row in zip(trainers, training_rows, strict=True)], timeout=180)

    topology = deploy.topology()
    source_shapes = dict(load_bootstrap_contract(deploy.metadata, deploy.manifest).source_shapes)
    coordinator = ray.remote(num_cpus=1, num_gpus=0, max_restarts=0, max_concurrency=96)(NativeCoordinator).options(
        name=deploy.run_id + "-native-control", lifetime="detached",
        scheduling_strategy=NodeAffinitySchedulingStrategy(ray_nodes[head]["NodeID"], soft=False),
    ).remote(topology, deploy.native_incarnation, deploy.metadata, deploy.manifest, str(output / "native"))
    backend._q38_coordinator = coordinator
    _write_json(output / "coordinator-ownership.json", ray.get(coordinator.ownership.remote(), timeout=180))
    ipc_loads = [relay.q38_prepare_ipc.remote() for relay in relays]
    backend.actor_rollout_wg.init_model()
    ipc_paths = ray.get(ipc_loads, timeout=240)
    _write_json(output / "relay-ipc-abi.json", [dict(row, extension=path)
                for row, path in zip(rollout_rows, ipc_paths, strict=True)])
    ranked = ray.get([actor.q38_rank.remote() for actor in trainers], timeout=180)
    actual = sorted(zip(ranked, trainers, strict=True), key=lambda pair: pair[0]["rank"])
    if [row["rank"] for row, _ in actual] != list(range(64)):
        raise RuntimeError("Megatron world did not bind exactly ranks zero through 63")
    rank0 = actual[0][0]["ip"]
    train_uuids = tuple(row["uuid"] for row, _ in actual)
    trainers = [actor for _, actor in actual]
    rollout_uuids = tuple(gpu for node in deploy.rollout for gpu in node.uuids)
    by_uuid = {row["uuid"]: relay for row, relay in zip(rollout_rows, relays, strict=True)}
    relays = [by_uuid[gpu] for gpu in rollout_uuids]
    plan = RefitGroupPlan(deploy.run_id, rank0, deploy.refit_port, train_uuids + rollout_uuids)
    _write_json(output / "refit-ownership.json", {
        "master": rank0, "port": plan.port, "train": [row for row, _ in actual],
        "rollout": [dict(next(row for row in rollout_rows if row["uuid"] == gpu), rank=rank)
                    for rank, gpu in enumerate(rollout_uuids)],
    })
    pending = [actor.q38_bind_refit.remote(plan, source_shapes) for actor in trainers]
    pending.extend(relay.q38_bind_refit.remote(plan, rank, deploy.native_incarnation, source_shapes)
                   for rank, relay in enumerate(relays))
    if ray.get(pending, timeout=plan.timeout_seconds + 45) != [*range(64), *range(32)]:
        raise RuntimeError("the 96 workers did not bind the expected refit ranks")
    if ray.get(coordinator.bind.remote(trainers, relays, source_shapes, rollout_uuids), timeout=630) != 1590:
        raise RuntimeError("native controller failed to verify all 1590 source coordinates")

    backend.llm_server_manager = NativeServerManager(coordinator)
    backend.async_rollout_manager = vm.AgentLoopManager.create(
        config=config, llm_client=backend.llm_server_manager.get_client(),
    )
    backend.checkpoint_manager = SplitCheckpointManager(coordinator)
    backend.checkpoint_manager.sleep_replicas()
