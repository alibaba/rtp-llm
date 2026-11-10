"""Opt-in rLLM on-policy initializer for physically separate rollout GPUs.

Candidate only: requires a standalone RTP replica and non-naive checkpoint
engine. The pinned rLLM/verl runtime and actual cluster remain to be qualified.
No connections, GPU allocation, or framework imports occur at module import.
"""
from __future__ import annotations

import inspect
import json
import numbers
import os


def _worker_gpu_identity(_worker):
    import ray
    import torch

    assigned = list(ray.get_runtime_context().get_accelerator_ids().get('GPU', []))
    if len(assigned) != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError('each worker must own exactly one visible GPU')
    uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    return {'node_id': str(ray.get_runtime_context().get_node_id()),
            'node_ip': str(ray.util.get_node_ip_address()),
            'gpu_id': str(assigned[0]), 'gpu_uuid': uuid}


def _collect_gpu_identities(workers):
    import ray
    return ray.get([worker.__ray_call__.remote(_worker_gpu_identity) for worker in workers])


def audit_gpu_pools(trainer_workers, replicas, *, allowed_node_ips, expected_train,
                    expected_rollout, max_gpus=32, max_nodes=8,
                    require_separate_nodes=True, collect=_collect_gpu_identities):
    """Reject duplicate physical devices before allowing weight transfer."""
    rollout_workers = [worker for replica in replicas for worker in replica.workers]
    if not trainer_workers or not rollout_workers:
        raise RuntimeError('training and rollout workers must both exist')
    if len(trainer_workers) != expected_train or len(rollout_workers) != expected_rollout:
        raise RuntimeError('worker allocation differs from the explicit pool plan')
    if not allowed_node_ips:
        raise RuntimeError('explicitly authorized node IP addresses are required')
    training = collect(trainer_workers)
    rollout = collect(rollout_workers)
    if len(training) != len(trainer_workers) or len(rollout) != len(rollout_workers):
        raise RuntimeError('GPU identity coverage is incomplete')
    identities = training + rollout
    if len(identities) > max_gpus:
        raise RuntimeError('GPU allocation exceeds the authorized task limit')
    for item in identities:
        if set(item) != {'node_id', 'node_ip', 'gpu_id', 'gpu_uuid'} or not all(item.values()):
            raise RuntimeError('GPU identity is incomplete')
        if item['node_ip'] not in allowed_node_ips:
            raise RuntimeError('worker is on a node outside the authorized task allocation')
    uuids = [item['gpu_uuid'].lower() for item in identities]
    if len(set(uuids)) != len(uuids):
        raise RuntimeError('training and rollout allocation is overlapping or duplicated')
    if len({item['node_id'] for item in identities}) > max_nodes:
        raise RuntimeError('GPU allocation exceeds the authorized node limit')
    if require_separate_nodes and ({item['node_id'] for item in training} &
                                   {item['node_id'] for item in rollout}):
        raise RuntimeError('training and rollout must use distinct nodes for this cross-node test')
    return {'training': training, 'rollout': rollout, 'max_gpus': max_gpus, 'max_nodes': max_nodes}


def guard_weight_versions(manager):
    """Poison a manager after an uncertain sync; never publish a retry silently."""
    if getattr(manager, '_q36_version_guard', False):
        return
    original = manager.update_weights
    state = {'last_committed_step': None, 'poisoned': False, 'active': False}

    async def guarded(global_steps=None):
        if state['poisoned']:
            raise RuntimeError('weight-sync state is uncertain; restart the owned rollout world')
        if state['active']:
            raise RuntimeError('overlapping weight-sync transactions are forbidden')
        if (not isinstance(global_steps, numbers.Integral) or isinstance(global_steps, bool)
                or global_steps < 0):
            raise ValueError('a nonnegative integer training step is required')
        if state['last_committed_step'] is not None and global_steps <= state['last_committed_step']:
            raise ValueError('weight-sync steps must advance monotonically')
        state['active'] = True
        try:
            result = original(global_steps=global_steps)
            if inspect.isawaitable(result):
                result = await result
            state['last_committed_step'] = int(global_steps)
            return result
        except BaseException:
            state['poisoned'] = True
            try:
                stopped = manager.abort_replicas()
                if inspect.isawaitable(stopped):
                    await stopped
            except BaseException as cleanup_error:
                print('Q36_DISAGG_ABORT_FAILED ' + repr(cleanup_error), flush=True)
            raise
        finally:
            state['active'] = False

    manager.update_weights = guarded
    manager._q36_version_guard = True
    manager._q36_version_state = state


def install(backend_module):
    """Install only in the explicitly selected driver process.

    Keep ``is_separated`` false so rLLM retains its on-policy loop and per-step
    weight update hook. Use the existing actor pool, with actor-only worker
    role, and ask verl's normal server manager for standalone resources.
    """
    backend_class = backend_module.VerlBackend
    original = backend_class._init_colocated_workers
    if getattr(original, "_q36_onpolicy_standalone", False):
        return original

    def initialize(self):
        config = self.config
        if self.is_separated:
            raise ValueError("on-policy standalone requires rLLM async_training.enable=false")
        if self.use_reference_policy or self.ref_in_actor:
            raise ValueError("candidate supports the no-reference, non-LoRA policy only")
        checkpoint = config.actor_rollout_ref.rollout.checkpoint_engine
        if checkpoint.backend == "naive":
            raise ValueError("standalone rollout requires a qualified non-naive checkpoint engine")
        role = backend_module.Role
        pool_role = next((r for r in (role.Actor, role.ActorRollout, role.ActorRolloutRef)
                          if r in self.role_worker_mapping), None)
        if pool_role is None:
            raise ValueError("actor worker class and actor resource pool are required")

        self.resource_pool_manager.create_resource_pool()
        pool = self.resource_pool_manager.get_resource_pool(pool_role)
        # Resource-pool mapping may use ActorRollout; worker behavior must not.
        worker = backend_module.RayClassWithInitArgs(
            cls=self.role_worker_mapping[pool_role],
            config=config.actor_rollout_ref,
            role=str(role.Actor),
        )
        workers = {str(pool_role): worker}
        colocated = backend_module.create_colocated_worker_cls(class_dict=workers)
        group = self.ray_worker_group_cls(
            resource_pool=pool,
            ray_cls_with_init=colocated,
            **backend_module.build_wg_kwargs(config, self.device_name),
        )
        groups = group.spawn(prefix_set=workers.keys())
        self.actor_rollout_wg = groups[str(pool_role)]
        if os.environ.get('Q36_STRICT_4PLUS4_CLAIM') == '1':
            from q36_4plus4_claim_loader import load_guard
            guard = load_guard()
            guard.validate_role_ownership(_collect_gpu_identities(self.actor_rollout_wg.workers),
                                          'training', guard.load_claim())
        self.actor_rollout_wg.init_model()

        self.llm_server_manager = backend_module.LLMServerManager.create(
            config=config, worker_group=None, rollout_resource_pool=None,
        )
        replicas = self.llm_server_manager.get_replicas()
        allowed_ips = {ip.strip() for ip in os.environ.get('Q36_DISAGG_ALLOWED_NODE_IPS', '').split(',') if ip.strip()}
        ownership = audit_gpu_pools(
            self.actor_rollout_wg.workers, replicas,
            allowed_node_ips=allowed_ips,
            expected_train=int(os.environ['Q36_DISAGG_TRAIN_GPUS']),
            expected_rollout=int(os.environ['Q36_DISAGG_ROLLOUT_GPUS']),
            max_gpus=int(os.environ.get('Q36_DISAGG_MAX_GPUS', '32')),
            max_nodes=int(os.environ.get('Q36_DISAGG_MAX_NODES', '8')),
        )
        if os.environ.get('Q36_STRICT_4PLUS4_CLAIM') == '1':
            from q36_4plus4_claim_loader import load_guard
            guard = load_guard()
            guard.validate_pool_ownership(ownership, guard.load_claim())
        print('Q36_DISAGG_GPU_OWNERSHIP ' + json.dumps(ownership, sort_keys=True), flush=True)
        self.async_rollout_manager = backend_module.AgentLoopManager.create(
            config=config, llm_client=self.llm_server_manager.get_client(),
        )
        ckpt_cfg = backend_module.omega_conf_to_dataclass(checkpoint)
        self.checkpoint_manager = backend_module.CheckpointEngineManager(
            config=ckpt_cfg, trainer=self.actor_rollout_wg,
            replicas=replicas,
        )
        guard_weight_versions(self.checkpoint_manager)
        self.checkpoint_manager.sleep_replicas()

    initialize._q36_onpolicy_standalone = True
    initialize._q36_original = original
    backend_class._init_colocated_workers = initialize
    return initialize
