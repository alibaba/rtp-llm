"""Retire a drained Native32 transport and bind a fresh consumer to resident trainers."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from q38_split_recovery import RECEIPTS, SHA256, _original_groups, _receipt


@dataclass(frozen=True)
class Original:
    groups: dict
    workers: dict
    refit: dict
    coordinator: dict
    digests: dict
    source_sha256: str


def _original(deploy, spec):
    attempt = spec["attempt_id"]
    hashes = spec["receipt_sha256"]
    if (not re.fullmatch(r"q38-[a-z0-9-]{12,79}", attempt) or attempt == deploy.run_id
            or set(hashes) != set(RECEIPTS) or not all(SHA256.fullmatch(value) for value in hashes.values())):
        raise ValueError("Expected a fresh replacement scope and exact original receipts")
    root = Path(deploy.output)
    native = root / "native"
    if root.is_symlink() or native.is_symlink() or root.name != deploy.run_id:
        raise ValueError("Expected the original output directory")
    later = (root / "train-step1.json", root / "train-step2.json", root / "DRIVER-TRAINING-COMPLETE.json",
             native / "refit-step1.json", native / "refit-step2.json", native / "NATIVE-TRAINING-COMPLETE.json")
    if any(path.exists() for path in later):
        raise ValueError("The original scope contains later training or policy evidence")
    groups, workers, coordinator, refit, first = [_receipt(root, name, hashes[name]) for name in RECEIPTS]
    digests = {name: tuple(rows) for name, rows in first["source_digests"].items()}
    raw = json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()
    checksum = hashlib.sha256(raw).hexdigest()
    if (first["model_version"] != deploy.native_incarnation + ":1" or first["global_step"] != 0
            or first["received_sources"] != 1590 or first["changed_sources"] != 1590
            or first["native_ranks"] != list(range(32)) or first["logical_policy_instances"] != 1
            or first["rank_source_sha256"] != checksum or spec["initial_rank_source_sha256"] != checksum
            or len(digests) != 1590
            or any(len(rows) != 32 or any(not SHA256.fullmatch(value) for value in rows)
                   for rows in digests.values())):
        raise ValueError("The original policy-zero source ledger changed or is incomplete")
    if (coordinator["native_incarnation"] != deploy.native_incarnation
            or refit["port"] != deploy.refit_port or refit["master"] != min(node.ip for node in deploy.train)
            or [row["rank"] for row in refit["train"]] != list(range(64))
            or [row["rank"] for row in refit["rollout"]] != list(range(32))):
        raise ValueError("Original native incarnation or refit placement changed")
    return Original(groups, workers, refit, coordinator, digests, checksum)


def _train_names(run_id):
    return [f"{run_id}-SplitActorWorker_{rank // 4}:{rank % 4}" for rank in range(64)]


def _actor_id(handle):
    return handle._actor_id.hex()


def _claim(deploy, original):
    import ray
    from q38_split_backend import ray_claim

    nodes = ray_claim(deploy)
    if ray.util.get_node_ip_address() != deploy.train[0].ip:
        raise ValueError("Only the original task head may operate the replacement")
    groups = _original_groups(deploy, original.groups)
    names = _train_names(deploy.run_id)
    trainers = [ray.get_actor(name, namespace=deploy.run_id) for name in names]
    if [_actor_id(handle) for handle in trainers] != [row["actor_id"] for row in original.refit["train"]]:
        raise ValueError("Retained training handles differ from the 64 loaded actors")
    return nodes, groups, names, trainers


def _worker_callback():
    # This callable is serialized by value to original, unmodified worker code.
    def inspect_worker(actor, expected, rank, old_incarnation, attempt, action):
        is_train = rank < 64
        actual = actor.q38_rank() if is_train else actor.q38_placement()
        if actual != expected:
            raise RuntimeError("Resident worker no longer has its original process/rank/GPU")
        part_count = 0
        if is_train:
            from bf16_streaming_optimizer_coords_candidate import StreamingBf16AdamWCoordinates

            optimizer = actor.actor.engine.optimizer
            parts = [part.optimizer for part in optimizer.chained_optimizers if not part.is_stub_optimizer]
            if not parts or any(type(part) is not StreamingBf16AdamWCoordinates or part.poisoned or part.state
                                for part in parts):
                raise RuntimeError("Resident optimizer already has Adam state or a poisoned step")
            part_count = len(parts)
        refit = actor._q38_refit
        if action == "train_detached":
            if not is_train or refit is not None or getattr(actor, "_q38_retired_refit", None) != attempt:
                raise RuntimeError("Training actor did not release the original transport")
            return {"rank": rank, "actor_id": actual["actor_id"], "optimizer_parts": part_count, "stage": action}
        if refit is None or refit.groups.rank != rank:
            raise RuntimeError("Original worker refit rank changed")
        groups = refit.groups
        if is_train:
            leases = [refit.expert] + ([refit.dense] if rank == 0 else [])
            if (refit.rank != rank or refit.epoch is not None or refit.last_step != 0
                    or refit.expert_iter is not None or refit.dense_iter is not None
                    or refit.expert.fragment is not None or (rank == 0 and refit.dense.spec is not None)):
                raise RuntimeError("Training refit has an open or mismatched original epoch")
        else:
            leases = [refit.expert.slot, refit.dense.slot]
            if (refit.expert.pair is not None or any(slot.spec is not None or slot.target_rank != rank - 64
                    or slot.native_incarnation != old_incarnation for slot in leases)):
                raise RuntimeError("Original relay retains a source or a different consumer identity")
        if any(slot.state != "idle" or slot.buffer is not None for slot in leases):
            raise RuntimeError("An original IPC/source lease is still held")
        if action == "open":
            if (groups.store is None or groups.experts is None
                    or (groups.dense is not None) != (rank == 0 or rank >= 64)):
                raise RuntimeError("Original dedicated groups have an incomplete open state")
        elif action in ("closed", "detach"):
            if any(group is not None for group in (groups.experts, groups.dense, groups.store)):
                raise RuntimeError("Original dedicated groups have not all shut down")
            if action == "detach":
                if not is_train or getattr(actor, "_q38_retired_refit", None) is not None:
                    raise RuntimeError("Only one matching retirement may clear the trainer refit object")
                actor._q38_refit = None
                actor._q38_retired_refit = attempt
        else:
            raise ValueError("Unknown resident worker inspection phase")
        return {"rank": rank, "actor_id": actual["actor_id"], "optimizer_parts": part_count, "stage": action}

    return inspect_worker


def _worker_proofs(handles, rows, ranks, old, attempt, action):
    import ray

    callback = _worker_callback()
    proof = ray.get([handle.__ray_call__.remote(callback, row, rank, old.native_incarnation, attempt, action)
                     for handle, row, rank in zip(handles, rows, ranks, strict=True)], timeout=90)
    if [(row["rank"], row["actor_id"], row["stage"]) for row in proof] != [
            (rank, evidence["actor_id"], action) for rank, evidence in zip(ranks, rows, strict=True)
    ]:
        raise RuntimeError("Resident worker proofs do not match the exact requested phase")
    return proof


def _retire_callback():
    def retire(actor, ownership, train_ids, relay_ids, source_sha, attempt, namespace):
        import asyncio
        import ray
        from ray._raylet import PythonFunctionDescriptor

        core = ray._private.worker.global_worker.core_worker
        descriptor = PythonFunctionDescriptor("q38_split_native", "__init__", "NativeCoordinator")
        loop, _ = core.get_event_loop(descriptor, None)

        async def seal():
            async with actor.lifecycle:
                stream, policy = actor.stream, actor.policy
                version = ownership["native_incarnation"] + ":1"
                if (await actor.ownership() != ownership or ray.get_runtime_context().namespace != namespace
                        or stream is None or stream.closed or stream.active or stream.poisoned
                        or stream.pending or stream.held is not None or policy.held_lease is not None
                        or actor.step != 0 or policy._step != 0 or stream.last_step != 0
                        or actor.version != version or policy._version != version or policy.control.version != version
                        or policy.control.update_id or len(actor.receipts) != 1
                        or actor.receipts[0]["rank_source_sha256"] != source_sha
                        or [h._actor_id.hex() for h in stream.trainers] != train_ids
                        or [h._actor_id.hex() for h in stream.receivers] != relay_ids
                        or getattr(actor, "_q38_replacement", None) is not None):
                    raise RuntimeError("Original coordinator is not a drained committed policy zero")
                async with actor.condition:
                    actor.accepting = False
                await asyncio.wait_for(asyncio.gather(*(client.close() for client in policy.clients)), 8)
                async with actor.condition:
                    await asyncio.wait_for(actor.condition.wait_for(lambda: actor.active == 0), 8)
                    if policy._active != 0:
                        raise RuntimeError("Original generation is still active")
                await asyncio.wait_for(policy.control.close(), 4)
                actor.poisoned = True
                stream.closed = True
                actor._q38_replacement = attempt
                return {"actor_id": ownership["actor_id"], "version": version, "source_sha256": source_sha,
                        "generation_counts": dict(actor.generations), "retired": True}

        future = asyncio.run_coroutine_threadsafe(seal(), loop)
        try:
            return future.result(timeout=24)
        except TimeoutError:
            future.cancel()
            raise

    return retire


def _old_stopped(deploy, stopped):
    actual=stopped["driver_actual_wait"]
    proven_absence=(actual is None and
                    stopped.get("driver_exit_observation")=="exact_pidfd_signal_and_kernel_absence")
    if (stopped["run_id"] != deploy.run_id or stopped["native_incarnation"] != deploy.native_incarnation
            or (type(actual) is not int and not proven_absence) or stopped["driver_identities_released"] is not True
            or stopped["native_ranks_released"] != list(range(32))
            or len(stopped["native_actual_waits"]) != 8
            or any(type(code) is not int for code in stopped["native_actual_waits"])):
        raise ValueError("Exact old driver and all 32 native consumer exit receipts are required")


def retire(deploy, spec, stopped):
    """Close old refit groups, keep trainers, and request exact old relay/coordinator termination."""
    import ray

    _old_stopped(deploy, stopped)
    original = _original(deploy, spec)
    _, _, _, trainers = _claim(deploy, original)
    attempt = spec["attempt_id"]
    raw_relays = [ray.get_actor(f"{deploy.run_id}-relay-{rank:02d}", namespace=deploy.run_id) for rank in range(32)]
    by_id = {_actor_id(handle): handle for handle in raw_relays}
    coordinator = ray.get_actor(f"{deploy.run_id}-native-control", namespace=deploy.run_id)
    if _actor_id(coordinator) != original.coordinator["actor_id"]:
        raise RuntimeError("Old coordinator name no longer identifies the original process")
    train_rows = original.refit["train"]
    relay_rows = [{key: value for key, value in row.items() if key != "rank"} for row in original.refit["rollout"]]
    relays = [by_id[row["actor_id"]] for row in relay_rows]
    handles, rows, ranks = trainers + relays, train_rows + relay_rows, list(range(96))
    _worker_proofs(handles, rows, ranks, deploy, attempt, "open")
    sealed = ray.get(coordinator.__ray_call__.remote(
        _retire_callback(), original.coordinator, [_actor_id(h) for h in trainers],
        [_actor_id(h) for h in relays], original.source_sha256, attempt, deploy.run_id,
    ), timeout=35)
    if sealed["retired"] is not True:
        raise RuntimeError("The original coordinator did not seal generation")
    # Same per-worker operations as SplitRefitStream.close, with no native RPC.
    ray.get([handle.q38_refit.remote("close") for handle in handles], timeout=210)
    _worker_proofs(handles, rows, ranks, deploy, attempt, "closed")
    detached = _worker_proofs(trainers, train_rows, list(range(64)), deploy, attempt, "detach")
    for handle in relays:
        ray.kill(handle, no_restart=True)
    ray.kill(coordinator, no_restart=True)
    return {"attempt_id": attempt, "run_id": deploy.run_id, "native_incarnation": deploy.native_incarnation,
            "source_sha256": original.source_sha256, "train": detached, "old_relays": relay_rows,
            "old_coordinator": original.coordinator, "old_coordinator_seal": sealed, "closed_ranks": ranks}


def _guard_factory():
    # The replacement coordinator can deserialize this without a new import path.
    def guard(stream, digests):
        class Guarded:
            async def leases(self, step, control):
                async for lease in stream.leases(step, control):
                    if step == 0 and tuple(lease.sha256_by_rank) != digests.get(lease.name):
                        raise RuntimeError(f"Replacement policy-zero source changed: {lease.name}")
                    yield lease

            async def close(self):
                await stream.close()

        return Guarded()

    return guard


def _coordinator_class():
    from q38_split_native import NativeCoordinator

    guard = _guard_factory()

    class ReplacementCoordinator(NativeCoordinator):
        def __init__(self, *args, original_digests):
            super().__init__(*args)
            self.original_digests = original_digests

        async def bind(self, trainers, relays, source_shapes, rollout_uuids):
            count = await super().bind(trainers, relays, source_shapes, rollout_uuids)
            if set(self.original_digests) != set(self.policy.sources):
                raise RuntimeError("Replacement source ledger differs from native policy coverage")
            self.stream = guard(self.stream, self.original_digests)
            return count

    return ReplacementCoordinator


def attach(backend, old, replacement, spec, retired, readiness):
    """Attach the original trainers and new relays/coordinator to a paused fresh Native32."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy
    from rllm.trainer.verl import verl_backend as vm
    from rtp_llm.models.qwen3_next.qwen38_bootstrap_contract import load_bootstrap_contract
    from verl.single_controller.ray import RayResourcePool

    from q38_refit_groups import RefitGroupPlan
    from q38_split_actor import SplitActorWorker, SplitRolloutRelay
    from q38_split_backend import SplitCheckpointManager, _check_workers
    from q38_split_native import NativeServerManager, _write_json

    original = _original(old, spec)
    attempt = spec["attempt_id"]
    output = Path(replacement.output)
    old_native_ports = set(range(old.native_start_port - 11, old.native_start_port + 48))
    new_native_ports = set(range(replacement.native_start_port - 11, replacement.native_start_port + 48))
    if (replacement.run_id != old.run_id or replacement.ray_session != old.ray_session
            or replacement.train != old.train or replacement.rollout != old.rollout
            or replacement.metadata != old.metadata or replacement.manifest != old.manifest
            or replacement.native_incarnation == old.native_incarnation
            or old_native_ports & new_native_ports or replacement.refit_port == old.refit_port
            or output == Path(old.output) or output.is_symlink() or not output.is_dir()):
        raise ValueError("Replacement must preserve resident train scope with fresh native, refit port and output")
    if (retired["attempt_id"] != attempt or retired["run_id"] != old.run_id
            or retired["native_incarnation"] != old.native_incarnation
            or retired["source_sha256"] != original.source_sha256 or retired["closed_ranks"] != list(range(96))
            or [row["actor_id"] for row in retired["train"]] != [row["actor_id"] for row in original.refit["train"]]
            or [row["actor_id"] for row in retired["old_relays"]] != [row["actor_id"] for row in original.refit["rollout"]]
            or retired["old_coordinator"] != original.coordinator):
        raise ValueError("Old refit retirement differs from the exact retained actor scope")
    ready_rows = readiness["ranks"]
    if (readiness["old_relay_contexts_released"] is not True or readiness["run"] != replacement.native_incarnation
            or readiness["all_paused"] is not True or [row["world_rank"] for row in ready_rows] != list(range(32))
            or any((row["world_size"], row["tp_size"], row["dp_size"], row["ep_size"]) != (32, 8, 4, 32)
                   or row["tp_rank"] != rank % 8 or row["dp_rank"] != rank // 8 or row["ep_rank"] != rank
                   or row["model_version"] != replacement.native_incarnation + ":0" or row["phase"] != "paused"
                   or not row["quiescent"] or row["received_weights"] or row["applied_weights"]
                   for rank, row in enumerate(ready_rows))):
        raise ValueError("Fresh native scope must prove exact relay release and all 32 paused bootstrap ranks")
    manager = backend.resource_pool_manager
    if (backend.is_separated or backend.use_reference_policy or backend.ref_in_actor
            or manager.resource_pool_spec != {"q38-training": [4] * 16}
            or manager.max_colocate_count != 1 or manager.mapping.get(vm.Role.ActorRollout) != "q38-training"):
        raise RuntimeError("Original actor-only 64-GPU backend changed")
    limit = spec.get("native_max_active_requests")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 8):
        raise ValueError("Native request bound must fit one DP replica")
    nodes, groups, names, trainers = _claim(old, original)
    training_rows = original.refit["train"]
    _worker_proofs(trainers, training_rows, list(range(64)), old, attempt, "train_detached")
    for name in [f"{old.run_id}-relay-{rank:02d}" for rank in range(32)] + [f"{old.run_id}-native-control"]:
        try:
            ray.get_actor(name, namespace=old.run_id)
        except ValueError:
            continue
        raise RuntimeError("Original relay or coordinator is still registered")

    relay_class = ray.remote(num_cpus=1, num_gpus=1, max_restarts=0)(SplitRolloutRelay)
    relays = [relay_class.options(
        name=f"{old.run_id}-{attempt}-relay-{4*part+local:02d}", lifetime="detached",
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=group, placement_group_bundle_index=local, placement_group_capture_child_tasks=False,
        ),
    ).remote() for part, group in enumerate(groups["rollout"]) for local in range(4)]
    rows = ray.get([relay.q38_placement.remote() for relay in relays], timeout=150)
    _check_workers(rows, old.rollout, "rollout", nodes)
    by_uuid = {row["uuid"]: (relay, row) for relay, row in zip(relays, rows, strict=True)}
    uuids = tuple(gpu for node in old.rollout for gpu in node.uuids)
    relays = [by_uuid[gpu][0] for gpu in uuids]
    relay_rows = [dict(by_uuid[gpu][1], rank=rank) for rank, gpu in enumerate(uuids)]
    _write_json(output / "replacement-relays.json", relay_rows)
    ray.get([relay.q38_prepare_ipc.remote() for relay in relays], timeout=150)

    plan = RefitGroupPlan(attempt, original.refit["master"], replacement.refit_port,
                          tuple(row["uuid"] for row in training_rows) + uuids)
    contract = load_bootstrap_contract(replacement.metadata, replacement.manifest)
    shapes = dict(contract.source_shapes)
    pending = [actor.q38_bind_refit.remote(plan, shapes) for actor in trainers]
    pending += [relay.q38_bind_refit.remote(plan, rank, replacement.native_incarnation, shapes)
                for rank, relay in enumerate(relays)]
    if ray.get(pending, timeout=plan.timeout_seconds + 45) != [*range(64), *range(32)]:
        raise RuntimeError("Replacement refit groups did not bind every exact rank")

    options = {"name": f"{old.run_id}-{attempt}-native-control", "lifetime": "detached",
               "scheduling_strategy": NodeAffinitySchedulingStrategy(nodes[old.train[0].ip]["NodeID"], soft=False)}
    if limit is not None:
        options["runtime_env"] = {"env_vars": {"RLLM_RTP_MAX_ACTIVE_REQUESTS": str(limit)}}
    coordinator = ray.remote(num_cpus=1, num_gpus=0, max_restarts=0, max_concurrency=96)(
        _coordinator_class()
    ).options(**options).remote(replacement.topology(), replacement.native_incarnation, replacement.metadata,
                                replacement.manifest, str(output / "native"), original_digests=original.digests)
    ownership = ray.get(coordinator.ownership.remote(), timeout=120)
    _write_json(output / "replacement-coordinator.json", ownership)
    if ray.get(coordinator.bind.remote(trainers, relays, shapes, uuids), timeout=630) != 1590:
        raise RuntimeError("Fresh coordinator did not confirm the complete native bootstrap")

    pool = RayResourcePool(process_on_nodes=[4] * 16, use_gpu=True, max_colocate_count=1,
                           name_prefix="q38-training")
    pool.pgs = groups["train"]
    manager.resource_pool_dict["q38-training"] = pool
    config = backend.config
    worker = vm.RayClassWithInitArgs(
        cls=ray.remote(max_restarts=0)(SplitActorWorker), config=config.actor_rollout_ref, role="actor",
    )
    backend.actor_rollout_wg = backend.ray_worker_group_cls.from_detached(
        name_prefix=old.run_id + "-", worker_names=names, worker_handles=trainers, ray_cls_with_init=worker,
        **vm.build_wg_kwargs(config, backend.device_name),
    )
    backend.actor_rollout_wg.resource_pool = pool
    backend._q38_coordinator = coordinator
    backend.llm_server_manager = NativeServerManager(coordinator)
    backend.async_rollout_manager = vm.AgentLoopManager.create(
        config=config, llm_client=backend.llm_server_manager.get_client(),
    )
    backend.checkpoint_manager = SplitCheckpointManager(coordinator)
    return {"old_source_sha256": original.source_sha256, "coordinator": ownership,
            "new_incarnation": replacement.native_incarnation, "refit_port": plan.port,
            "train_actor_ids": [_actor_id(h) for h in trainers], "relay_actor_ids": [_actor_id(h) for h in relays]}
