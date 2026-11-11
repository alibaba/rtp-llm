"""Reattach one driver to its original, committed Qwen3.8 policy and actors."""

import hashlib
import json
from pathlib import Path
import re


RECEIPTS = (
    "placement-groups.json", "worker-ownership.json", "coordinator-ownership.json",
    "refit-ownership.json", "native/refit-step0.json",
)
SHA256 = re.compile(r"[a-f0-9]{64}\Z")


def _receipt(root, name, expected):
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing original recovery receipt {name}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Original recovery receipt changed: {name}")
    return json.loads(data)


def _load_original(deploy, spec):
    if not re.fullmatch(r"q38-recover-[a-f0-9]{12}", spec["attempt_id"]):
        raise ValueError("Expected one fresh scoped recovery attempt ID")
    hashes = spec["receipt_sha256"]
    source_sha = spec["initial_rank_source_sha256"]
    if (set(hashes) != set(RECEIPTS) or not SHA256.fullmatch(source_sha)
            or not all(SHA256.fullmatch(value) for value in hashes.values())):
        raise ValueError("Recovery must lock each original ownership and refit receipt")
    output = Path(deploy.output)
    native = output / "native"
    if output.is_symlink() or native.is_symlink() or output.name != deploy.run_id:
        raise ValueError("Recovery must use the original task output and native journal")
    later = (output / "train-step1.json", output / "train-step2.json", output / "DRIVER-TRAINING-COMPLETE.json",
             native / "refit-step1.json", native / "refit-step2.json", native / "NATIVE-TRAINING-COMPLETE.json")
    if (any(path.exists() for path in later) or any(native.glob("rtp_rpc_requests_replica*.jsonl"))
            or any((output / "episodes").glob("enriched_*.json"))):
        raise ValueError("The original scope already contains generation or later policy evidence")
    groups, workers, coordinator, refit, initial = [
        _receipt(output, name, hashes[name]) for name in RECEIPTS
    ]
    version = f"{deploy.native_incarnation}:1"
    digests = initial["source_digests"]
    raw = json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()
    if ((initial["model_version"], initial["global_step"], initial["received_sources"],
         initial["logical_policy_instances"], initial["changed_sources"]) != (version, 0, 1590, 1, 1590)
            or initial["native_world"] != 32 or initial["native_ranks"] != list(range(32))
            or initial["rank_source_sha256"] != source_sha or hashlib.sha256(raw).hexdigest() != source_sha
            or len(digests) != 1590
            or any(len(rows) != 32 or any(not SHA256.fullmatch(value) for value in rows)
                   for rows in digests.values())):
        raise ValueError("Original initial all-rank source commit is incomplete or changed")
    if coordinator["native_incarnation"] != deploy.native_incarnation or coordinator["ip"] != deploy.train[0].ip:
        raise ValueError("Original coordinator is outside the scoped native run or Ray head")
    if refit["port"] != deploy.refit_port or refit["master"] != min(node.ip for node in deploy.train):
        raise ValueError("Original refit rendezvous differs from the scoped training rank zero")
    expected = {
        "attempt_id": spec["attempt_id"], "run_id": deploy.run_id, "version": version,
        "ownership": coordinator, "native_dir": str(native), "initial_refit_sha256": hashes[RECEIPTS[-1]],
        "initial_rank_source_sha256": source_sha,
        "train_actor_ids": [row["actor_id"] for row in refit["train"]],
        "rollout_actor_ids": [row["actor_id"] for row in refit["rollout"]],
    }
    if (len(expected["train_actor_ids"]) != 64 or len(expected["rollout_actor_ids"]) != 32
            or len(set(expected["train_actor_ids"] + expected["rollout_actor_ids"])) != 96):
        raise ValueError("Original refit does not bind 96 distinct actors")
    return groups, workers, refit, expected


def _coordinator_callback():
    # A nested callable ships its own code to the already-running coordinator.
    def callback(actor, expected, reopen):
        import asyncio
        import hashlib
        from pathlib import Path

        import ray
        from ray._raylet import PythonFunctionDescriptor

        core = ray._private.worker.global_worker.core_worker
        descriptor = PythonFunctionDescriptor("q38_split_native", "__init__", "NativeCoordinator")
        loop, _ = core.get_event_loop(descriptor, None)

        async def inspect():
            async with actor.lifecycle, actor.condition:
                ownership = await actor.ownership()
                if ownership != expected["ownership"] or actor.evidence != Path(expected["native_dir"]):
                    raise RuntimeError("Coordinator no longer has its original process or journal")
                if ray.get_runtime_context().namespace != expected["run_id"]:
                    raise RuntimeError("Coordinator moved outside the original Ray namespace")
                policy, stream = actor.policy, actor.stream
                if (stream is None or actor.poisoned or policy.poisoned or stream.poisoned
                        or stream.closed or stream.active or stream.pending or stream.held is not None
                        or policy.held_lease is not None or stream.last_step != 0
                        or actor.step != 0 or policy._step != 0
                        or actor.version != expected["version"] or policy._version != expected["version"]
                        or actor.active != 0 or policy._active != 0 or policy._next_dp != 0
                        or actor.next_request_id != 1 or actor.generations):
                    raise RuntimeError("Coordinator has an incomplete lease, generation, or different policy")
                if ([handle._actor_id.hex() for handle in stream.trainers] != expected["train_actor_ids"]
                        or [handle._actor_id.hex() for handle in stream.receivers] != expected["rollout_actor_ids"]):
                    raise RuntimeError("Resident refit stream does not retain its original 96 actors")
                state = policy.control._observed_state
                if (policy.control._failed or policy.control.update_id or policy.control._pending_global_step is not None
                        or policy.control.version != expected["version"] or state is None
                        or state["phase"] != "running" or state["quiescent"] or state["update_id"]
                        or state["model_version"] != expected["version"] or state["global_step"] != 0
                        or state["received_weights"] != 1590 or state["applied_weights"] <= 0
                        or not state["kv_cache_ready"] or state["kv_cache_bytes"] <= 0):
                    raise RuntimeError("Cached actual native commit and resume state differs from the initial policy")
                source_sha = expected["initial_rank_source_sha256"]
                if (len(actor.receipts) != 1 or actor.changed != [1590]
                        or actor.receipts[0]["rank_source_sha256"] != source_sha
                        or set(actor.receipts[0]["source_digests"]) != policy.sources
                        or hashlib.sha256((actor.evidence / "refit-step0.json").read_bytes()).hexdigest()
                           != expected["initial_refit_sha256"]
                        or any(actor.evidence.glob("rtp_rpc_requests_replica*.jsonl"))):
                    raise RuntimeError("Resident policy-zero evidence or generation journal changed")
                attempt = getattr(actor, "_q38_recovery_attempt", None)
                if attempt is not None and attempt != expected["attempt_id"]:
                    raise RuntimeError("The existing coordinator belongs to a different recovery attempt")
                before = actor.accepting
                if reopen:
                    actor._q38_recovery_attempt = expected["attempt_id"]
                    actor.accepting = True
                return {"owner": ownership, "attempt_id": expected["attempt_id"], "model_version": actor.version,
                        "global_step": actor.step, "rank_source_sha256": source_sha, "received_sources": 1590,
                        "next_request_id": actor.next_request_id, "accepting_before": before,
                        "accepting": actor.accepting, "rearmed": reopen}

        future = asyncio.run_coroutine_threadsafe(inspect(), loop)
        try:
            return future.result(timeout=18)
        except TimeoutError:
            future.cancel()
            raise

    return callback


def _invoke(coordinator, expected, reopen):
    import ray

    if coordinator._actor_id.hex() != expected["ownership"]["actor_id"]:
        raise ValueError("The coordinator handle differs from the original actor ID")
    return ray.get(coordinator.__ray_call__.remote(_coordinator_callback(), expected, reopen), timeout=30)


def verify(coordinator, expected):
    """Verify the retained coordinator using cached actual all-rank evidence."""
    return _invoke(coordinator, expected, False)


def rearm(coordinator, expected):
    """Reopen in-memory rollout admission after the replacement gateway is healthy."""
    return _invoke(coordinator, expected, True)


def _original_groups(deploy, records):
    from ray.util.placement_group import get_placement_group, placement_group_table

    result = {}
    for role in ("train", "rollout"):
        nodes = getattr(deploy, role)
        groups = [get_placement_group(f"{deploy.run_id}-{role}-{part:02d}") for part in range(len(nodes))]
        if [group.id.hex() for group in groups] != records[role]:
            raise ValueError(f"Retained {role} placement group IDs differ from the original run")
        for node, group in zip(nodes, groups, strict=True):
            table = placement_group_table(group)
            if (group.bundle_count != 4 or table["state"] != "CREATED"
                    or [table["bundles_to_node_id"][i] for i in range(4)] != [node.node_id] * 4):
                raise ValueError(f"Retained {role} placement group no longer has its assigned node")
        result[role] = groups
    return result


def attach(backend, deploy, spec):
    """Construct the verl facade around existing workers and return a recovery expectation."""
    import ray
    from rllm.trainer.verl import verl_backend as vm
    from verl.single_controller.ray import RayResourcePool

    from q38_split_actor import SplitActorWorker
    from q38_split_backend import SplitCheckpointManager, _check_workers, ray_claim
    from q38_split_native import NativeServerManager

    if backend.is_separated or backend.use_reference_policy or backend.ref_in_actor:
        raise ValueError("Recovery expects the original actor-only synchronous GRPO backend")
    group_records, worker_records, refit, expected = _load_original(deploy, spec)
    nodes = ray_claim(deploy)
    if ray.util.get_node_ip_address() != deploy.train[0].ip:
        raise RuntimeError("The recovery driver is outside the original training head")
    groups = _original_groups(deploy, group_records)
    run_id = deploy.run_id
    train_names = [f"{run_id}-SplitActorWorker_{rank // 4}:{rank % 4}" for rank in range(64)]
    relay_names = [f"{run_id}-relay-{rank:02d}" for rank in range(32)]
    trainers = [ray.get_actor(name, namespace=run_id) for name in train_names]
    relays = [ray.get_actor(name, namespace=run_id) for name in relay_names]
    coordinator = ray.get_actor(f"{run_id}-native-control", namespace=run_id)
    rows = ray.get([actor.q38_rank.remote() for actor in trainers] +
                   [relay.q38_placement.remote() for relay in relays], timeout=70)
    training, rollout = rows[:64], rows[64:]
    _check_workers(training, deploy.train, "train", nodes)
    _check_workers(rollout, deploy.rollout, "rollout", nodes)
    by_uuid = {row["uuid"]: row for row in rollout}
    ordered_relays = [dict(by_uuid[gpu], rank=rank)
                      for rank, gpu in enumerate(gpu for node in deploy.rollout for gpu in node.uuids)]
    if (training != refit["train"] or training != worker_records["train"]
            or rollout != worker_records["rollout"] or ordered_relays != refit["rollout"]
            or [handle._actor_id.hex() for handle in trainers + relays] != [row["actor_id"] for row in rows]):
        raise RuntimeError("Live training or relay identities differ from the original load and refit")
    verify(coordinator, expected)

    manager = backend.resource_pool_manager
    if (manager.resource_pool_spec != {"q38-training": [4] * 16} or manager.max_colocate_count != 1
            or manager.mapping.get(vm.Role.ActorRollout) != "q38-training"):
        raise RuntimeError("The original 64-GPU training resource specification changed")
    pool = RayResourcePool(process_on_nodes=[4] * 16, use_gpu=True, max_colocate_count=1,
                           name_prefix="q38-training")
    pool.pgs = groups["train"]
    manager.resource_pool_dict["q38-training"] = pool
    config = backend.config
    worker = vm.RayClassWithInitArgs(
        cls=ray.remote(max_restarts=0)(SplitActorWorker), config=config.actor_rollout_ref, role="actor",
    )
    backend.actor_rollout_wg = backend.ray_worker_group_cls.from_detached(
        name_prefix=run_id + "-", worker_names=train_names, worker_handles=trainers, ray_cls_with_init=worker,
        **vm.build_wg_kwargs(config, backend.device_name),
    )
    backend.actor_rollout_wg.resource_pool = pool
    backend._q38_coordinator = coordinator
    backend.llm_server_manager = NativeServerManager(coordinator)
    backend.async_rollout_manager = vm.AgentLoopManager.create(
        config=config, llm_client=backend.llm_server_manager.get_client(),
    )
    backend.checkpoint_manager = SplitCheckpointManager(coordinator)
    return expected
