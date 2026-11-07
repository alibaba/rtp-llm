"""Resume the first update from authenticated original policy-zero episodes.

Opt-in recovery only. No model output, reward, tool execution or DataProto is
invented. Complementary original rLLM logger and enriched-token artifacts are
joined, then passed through the unchanged trajectory/GRPO/verl conversion.
The first native sync is omitted only after all fresh FSDP workers prove their
newly loaded official BF16 shards and a current controller proves no native
incarnation or canonical source exporter has ever existed in this run.
"""
from __future__ import annotations
import asyncio,collections,functools,hashlib,json,os,struct,time
from pathlib import Path

SCHEMA='q38-policy0-original-episode-replay-v1'
FIELDS=('name','canonical_name','checkpoint_file','local_numel','local_bytes','source_begin',
        'source_end','source_reads','maximum_source_slice_bytes','sample_sha256','source_coordinates_verified')


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def shard_layout_digest(receipt):
    return digest_json([{key:row[key] for key in FIELDS} for row in receipt['source_slices']])


def checked_json(row):
    path=Path(row['path']);payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=row['sha256']:
        raise RuntimeError('An immutable original replay artifact differs: '+str(path))
    return json.loads(payload)


def load_manifest():
    path=Path(os.environ['Q38_INITIAL_POLICY_REPLAY_MANIFEST']);payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=os.environ['Q38_INITIAL_POLICY_REPLAY_MANIFEST_SHA256']:
        raise RuntimeError('The explicit immutable initial-policy replay manifest differs')
    value=json.loads(payload)
    if (value.get('schema')!=SCHEMA or value.get('global_step')!=0 or value.get('world_size')!=72
            or value.get('initial_full_source_sha256')!='1303c36b3d678821c7e9393c10f5a0754fc081143d7f9c7949ad2ed1b45b744f'
            or len(value.get('episodes',[]))!=72 or len(value.get('previous_shards',[]))!=72
            or {x['rank'] for x in value['previous_shards']}!=set(range(72))):
        raise RuntimeError('Replay is restricted to the complete unchanged official initial policy and 72 original trajectories')
    return value


def restore_episode(logger,enriched):
    """Join original serialized values; do not synthesize missing token data."""
    from rllm.types import Episode,Trajectory,Step
    from rllm.workflows.workflow import TerminationReason
    if (logger['episode_id']!=enriched['episode_id'] or logger['task']!=enriched['task']
            or logger['is_correct']!=enriched['is_correct']
            or len(logger['trajectories'])!=len(enriched['trajectories'])):
        raise RuntimeError('Original logger/enriched episode identities differ')
    trajectories=[]
    for logged,native in zip(logger['trajectories'],enriched['trajectories']):
        if logged['name']!=native['name'] or len(logged['steps'])!=len(native['steps']):
            raise RuntimeError('Original trajectory names or step counts differ')
        steps=[]
        for view,serialized in zip(logged['steps'],native['steps']):
            for key in ('observation','thought','action','reward','done','model_response','chat_completions'):
                if view[key]!=serialized[key]:raise RuntimeError('Original delivered step artifact differs: '+key)
            step=Step.from_dict(serialized)
            model=step.model_output
            if (model is None or not model.prompt_ids or not model.completion_ids
                    or len(model.logprobs)!=len(model.completion_ids)
                    or step.prompt_ids!=model.prompt_ids or step.response_ids!=model.completion_ids
                    or struct.pack(f'<{len(step.logprobs)}f',*step.logprobs)!=struct.pack(f'<{len(model.logprobs)}f',*model.logprobs)
                    or step.weight_version!=0):
                raise RuntimeError('Original initial-policy native IDs/FP32 log-probability payload is incomplete')
            steps.append(step)
        trajectories.append(Trajectory(uid=logged['uid'],name=logged['name'],task=None,steps=steps,
                                       reward=logged['reward']))
    episode=Episode(id=logger['episode_id'],task=logger['task'],session_id=logger['session_id'],
        termination_reason=TerminationReason(logger['termination_reason']),is_correct=logger['is_correct'],
        trajectories=trajectories,artifacts=enriched['artifacts'],metrics=logger['metrics'],metadata=logger['metadata'])
    # The original VerlBackend._execute_tasks_async performs this assignment
    # after AgentFlowEngine's logger/enriched artifacts are written.
    source=episode.task.get('data_source')
    if source is not None:episode.info['data_source']=source
    return episode


def read_original_episode_payloads(manifest):
    return [(checked_json(row['logger']),checked_json(row['enriched'])) for row in manifest['episodes']]


def load_episodes(manifest,payloads=None):
    if payloads is None:payloads=read_original_episode_payloads(manifest)
    episodes=[restore_episode(logger,enriched) for logger,enriched in payloads]
    if (len({e.id for e in episodes})!=72 or sum(e.is_correct for e in episodes)!=15
            or collections.Counter(e.task['source_row_index'] for e in episodes)!=collections.Counter({int(row):4 for row in manifest['source_rows']})):
        raise RuntimeError('The original 18x4 episodes or unmodified 57/15 rewards differ')
    return episodes


def verify_owned_canonical_sources(manifest,rank,*,maximum_read_bytes=128*1024**2):
    """CPU-stream an assigned subset, matching every original canonical SHA.

    No full canonical tensor/FP32 policy is materialized. The authoritative
    platform-mounted checkpoint is the same fixed read-only revision used by
    the exact-copy loader; this complements its current shard byte checks.
    """
    import math,torch
    from safetensors import safe_open
    checkpoint=Path('/mnt/fuse/Qwen/Qwen3.8-2.4T-A95B')
    index=json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    reference=checked_json(manifest['canonical_reference'])
    if (reference['global_step']!=0 or reference['received_weights']!=1590
            or reference['ranks']!=list(range(72)) or reference.get('error') is not None
            or reference['full_source_sha256']!=manifest['initial_full_source_sha256']
            or reference['model_version']!=manifest['original_native_model_version']):
        raise RuntimeError('Replay canonical reference is not the complete real initial native transaction')
    rows=reference['tensors'];names=[row['name'] for row in rows]
    if len(names)!=1590 or names!=sorted(set(names)):
        raise RuntimeError('The actual previous canonical source set/order is incomplete')
    return [dict(index=source_index,**hash_canonical_source(checkpoint,index,rows[source_index],maximum_read_bytes=maximum_read_bytes))
            for source_index in range(rank,len(rows),72)]


def hash_canonical_source(checkpoint,weight_map,row,*,maximum_read_bytes=128*1024**2):
    import math,torch
    from safetensors import safe_open
    name=row['name'];filename=weight_map[name]
    expected_dtype='torch.float32' if name.endswith(('.linear_attn.A_log','.linear_attn.dt_bias')) else 'torch.bfloat16'
    if row['dtype']!=expected_dtype or Path(filename).name!=filename:
        raise RuntimeError('Replay source dtype or checkpoint path differs')
    digest=hashlib.sha256();read_count=peak=bytes_hashed=0
    with safe_open(str(checkpoint/filename),framework='pt',device='cpu') as stream:
        source=stream.get_slice(name)
        if list(source.get_shape())!=row['shape']:raise RuntimeError('Replay source shape differs')
        row_bytes=math.prod(row['shape'][1:])*2;rows_per_read=maximum_read_bytes//row_bytes
        if rows_per_read<1:raise RuntimeError('One official source row escapes the bounded read window')
        for begin in range(0,row['shape'][0],rows_per_read):
            end=min(row['shape'][0],begin+rows_per_read)
            part=source[(slice(begin,end),)+(slice(None),)*(len(row['shape'])-1)]
            if part.dtype!=torch.bfloat16 or part.numel()*2>maximum_read_bytes:
                raise RuntimeError('Only original bounded BF16 source slices are allowed')
            canonical=part.float() if expected_dtype=='torch.float32' else part
            payload=memoryview(canonical.contiguous().view(torch.uint8).numpy()).cast('B')
            digest.update(payload);bytes_hashed+=len(payload);read_count+=1
            peak=max(peak,part.numel()*part.element_size()+canonical.numel()*canonical.element_size()*(canonical is not part))
            del payload,canonical,part
    actual=digest.hexdigest()
    if actual!=row['sha256'] or bytes_hashed!=row['source_bytes']:
        raise RuntimeError('Complete mounted canonical bytes differ from previous actual native COMMIT: '+name)
    return dict(name=name,sha256=actual,source_bytes=bytes_hashed,read_count=read_count,maximum_host_window_bytes=peak)


def restore_original_training_postconditions(child):
    """Restore the original naive-sync epilogue after authenticated sync0 replay.

    The initial native transaction is legitimately absent, but its caller also
    establishes process-local training allocator state. Keep the original IPC
    environment and all future real sync calls unchanged; only reproduce the
    original cleanup/base-sync/allocator epilogue on the actual actor child.
    """
    import importlib,inspect,torch
    expected={
        'verl.workers.engine_workers':'721f866442475a08768854b58176a4dcd99de4564809e3d067dced7cba1122db',
        'verl.utils.device':'52c5dd8dc3e4afc3e0d67756d3d4995282246b98b767be080c28448353c1ad11',
        'verl.utils.memory_utils':'2c356ff43c37d18f29dd0713cb60cb080209fff2e964f993b8d27f176669c111'}
    modules={name:importlib.import_module(name) for name in expected}
    for name,module in modules.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=expected[name]:
            raise RuntimeError('Original naive-sync training epilogue source differs: '+name)
    workers=modules['verl.workers.engine_workers']
    original=inspect.unwrap(workers.ActorRolloutRefWorker.update_weights)
    actual=inspect.unwrap(type(child).update_weights)
    if (not isinstance(child,workers.ActorRolloutRefWorker) or actual is not original
            or Path(inspect.getsourcefile(original)).resolve()!=Path(workers.__file__).resolve()
            or workers.set_expandable_segments is not modules['verl.utils.device'].set_expandable_segments
            or workers.aggressive_empty_cache is not modules['verl.utils.memory_utils'].aggressive_empty_cache
            or not torch.cuda.is_initialized() or torch.cuda.memory.get_allocator_backend()!='native'):
        raise RuntimeError('Replay epilogue is restricted to the original CUDA actor and native allocator')
    before=torch._C._accelerator_getAllocatorSettings()
    previous_base=child.base_sync_done
    # Same order as the pinned original update_weights epilogue. There is no
    # native cache to resume or source to offload in this fresh virgin state.
    workers.aggressive_empty_cache(force_sync=True)
    child.base_sync_done=True
    workers.set_expandable_segments(True)
    after=torch._C._accelerator_getAllocatorSettings()
    if 'expandable_segments:True' not in after.replace(' ','') or child.base_sync_done is not True:
        raise RuntimeError('Original training allocator postcondition was not established')
    return dict(schema='q38-original-sync-training-postconditions-v1',
        actual_actor_child_class=type(child).__module__+'.'+type(child).__name__,
        source_sha256=expected,original_sync_epilogue_replayed=True,
        force_synchronized_aggressive_empty_cache_completed=True,
        base_sync_done_before=previous_base,base_sync_done_after=child.base_sync_done,
        active_allocator_settings_before=before,active_allocator_settings_after=after,
        active_training_expandable_segments=True,allocator_backend='native',
        IPC_allocator_environment=os.environ.get('PYTORCH_ALLOC_CONF'),
        native_update_weights_method_unmodified=True,
        no_native_resume_or_weight_transaction_performed=True)


def _fresh_worker_proof(worker,manifest,expected_rank,expected_uuid,no_native):
    import torch
    import torch.distributed as dist
    from qwen38_bf16_optimizer import local
    from qwen38_world_contract import canonical_uuid
    matches=[]
    for child in worker.worker_dict.values():
        engine=getattr(getattr(child,'actor',None),'engine',None)
        if getattr(child,'_is_actor',False) and hasattr(engine,'_q38_phases'):matches.append((engine,child.rollout,child))
    if (len(matches)!=1 or not dist.is_initialized() or dist.get_world_size()!=72 or dist.get_rank()!=expected_rank):
        raise RuntimeError('Replay source verification reached the wrong current FSDP actor')
    engine,adapter,actor_child=matches[0]
    _require_initial_no_native(no_native)
    if (engine._q38_phases.phase!='train' or engine._q38_exporter is not None
            or engine._q38_gradient_store.slots or engine.optimizer.audit()['steps']
            or engine.optimizer.audit()['moments']['cpu_moment_bytes']!=0
            or getattr(adapter,'_cleanup_callback',None) is not None
            or getattr(adapter,'_retained_source',None) is not None
            or any(p.grad is not None or local(p).dtype!=torch.bfloat16 for p in engine.module.parameters())):
        raise RuntimeError('Only fresh, unupdated, fully BF16 training masters with no native exporter may replay policy zero')
    path=Path(os.environ['Q38_TRAINING_AUDIT_DIR'])/f'checkpoint-load-rank{expected_rank}.json'
    receipt=json.loads(path.read_text())
    old=next(x for x in manifest['previous_shards'] if x['rank']==expected_rank)
    layout=shard_layout_digest(receipt)
    if (receipt['rank']!=expected_rank or receipt['world_size']!=72
            or receipt['global_policy_numel']!=2419804697984 or receipt['source_keys']!=1590
            or receipt['runtime_parameters']!=7324 or receipt['all_master_dtype']!='bfloat16'
            or layout!=old['source_layout_sha256']
            or any(row.get('full_destination_BF16_bytes_verified') is not True
                   or len(row.get('full_slice_bytes_sha256',[]))!=row['source_reads'] for row in receipt['source_slices'])):
        raise RuntimeError('Current fully byte-verified official BF16 shard differs from the original policy-zero source coordinates')
    actual_uuid=canonical_uuid(str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid))
    if actual_uuid!=canonical_uuid(expected_uuid):raise RuntimeError('The fresh replay actor uses a different allocated GPU')
    canonical_verified=verify_owned_canonical_sources(manifest,expected_rank)
    training_postconditions=restore_original_training_postconditions(actor_child)
    proof=dict(schema='q38-fresh-initial-policy-replay-worker-v1',rank=expected_rank,world_size=72,
        pid=os.getpid(),time_ns=time.time_ns(),current_run_root=os.environ['Q38_RUN_ROOT'],gpu_uuid=actual_uuid,
        previous_run_root=manifest['source_run_root'],previous_checkpoint_load_sha256=old['sha256'],
        current_checkpoint_load_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_layout_sha256=layout,local_numel=receipt['loaded_local_numel'],global_policy_numel=receipt['global_policy_numel'],
        current_full_source_copy_verified=True,
        bounded_slice_digest=digest_json([row['full_slice_bytes_sha256'] for row in receipt['source_slices']]),
        source_policy_full_sha256=manifest['initial_full_source_sha256'],
        current_assigned_canonical_sources=canonical_verified,native_never_started_proof=no_native,
        native_source_export_never_created=True,current_optimizer_steps=[],current_CPU_moment_bytes=0,
        current_FP32_gradient_bytes=0,old_native_version=manifest['original_native_model_version'],
        old_evidence_is_not_current_native_cleanup=True,
        original_training_postconditions=training_postconditions)
    engine._q38_initial_policy_replay_proof=proof
    output=Path(os.environ['Q38_TRAINING_AUDIT_DIR'])/f'initial-policy-replay-rank{expected_rank}.json'
    with output.open('x') as stream:json.dump(proof,stream);stream.write('\n')
    return proof


def _require_initial_no_native(proof):
    if (proof.get('native_never_started') is not True or proof.get('native_boot')!=-1
            or proof.get('native_incarnation') is not None or proof.get('world_size')!=72
            or proof.get('checked_native_ranks')!=[] or proof.get('remaining_owned_gpu_contexts')!=[]
            or proof.get('training_admission_safe') is not False):
        raise RuntimeError('A real current virgin controller proof is required; historical teardown cannot authorize replay')


def install():
    if os.environ.get('Q38_INITIAL_POLICY_REPLAY')!='1':return False
    from rllm.trainer.verl.verl_backend import VerlBackend,interleave_tasks
    original_start=VerlBackend.on_train_start;original_generate=VerlBackend.generate_episodes
    if getattr(original_start,'_q38_initial_policy_replay',False):return True

    @functools.wraps(original_start)
    async def start(self,state):
        manifest=load_manifest();manager=self.checkpoint_manager;old_sync=manager.update_weights
        calls=[]
        async def initial_only(global_steps=None):
            if global_steps!=0 or calls:raise RuntimeError('Only the original policy-zero sync may be recovered from original rollouts')
            replicas=self.llm_server_manager.get_replicas()
            if len(replicas)!=1 or replicas[0].world_size!=72 or len(replicas[0].workers)!=72:
                raise RuntimeError('Replay requires the current single 72-worker replica')
            replica=replicas[0];no_native=await replica._server_handle.sleep.remote();_require_initial_no_native(no_native)
            topology=replica._world72_bound_topology
            outcomes=await asyncio.gather(*[worker.__ray_call__.remote(_fresh_worker_proof,manifest,rank,
                topology.gpu_uuids[rank],no_native) for rank,worker in enumerate(replica.workers)],return_exceptions=True)
            errors=[{'rank':rank,'error':repr(row)} for rank,row in enumerate(outcomes) if isinstance(row,BaseException)]
            if errors:raise RuntimeError('Fresh complete-world replay-source validation failed: '+str(errors))
            if [r['rank'] for r in outcomes]!=list(range(72)) or len({r['gpu_uuid'] for r in outcomes})!=72:
                raise RuntimeError('Replay proof does not cover all actual 72 allocated GPU actors once')
            verified=[row for proof in outcomes for row in proof['current_assigned_canonical_sources']]
            if len(verified)!=1590 or sorted(row['index'] for row in verified)!=list(range(1590)):
                raise RuntimeError('Current CPU source-byte validation did not cover every canonical source exactly once')
            self._q38_replay_initial_world_proofs=outcomes;self._q38_replay_manifest=manifest;calls.append(0)
            root=Path(os.environ['Q38_RUN_ROOT']);root.mkdir(parents=True,exist_ok=True)
            with (root/'initial-policy-replay-world72.json').open('x') as stream:
                json.dump(dict(schema=SCHEMA,manifest_sha256=os.environ['Q38_INITIAL_POLICY_REPLAY_MANIFEST_SHA256'],
                    first_native_sync_skipped_at_step=0,current_run_root=str(root),source_run_root=manifest['source_run_root'],
                    current_fresh_worker_proofs=outcomes,historical_native_pids_used_as_current_proof=False,
                    policy1_and_policy2_native_sync_unmodified=True),stream);stream.write('\n')
        manager.update_weights=initial_only
        try:result=await original_start(self,state)
        finally:manager.update_weights=old_sync
        if calls!=[0]:raise RuntimeError('Original startup did not pass through exactly the expected initial-sync boundary')
        return result

    @functools.wraps(original_generate)
    async def generate(self,batch,agent_workflow_engine,is_validation=False,**kwargs):
        if is_validation or self.global_steps!=1:
            return await original_generate(self,batch,agent_workflow_engine,is_validation=is_validation,**kwargs)
        if getattr(self,'_q38_replay_consumed',False) or not getattr(self,'_q38_replay_initial_world_proofs',None):
            raise RuntimeError('Policy-zero replay must occur once after current all72-source verification')
        manifest=self._q38_replay_manifest
        # Original artifacts are node-local. Fetch their unchanged scalar
        # payloads from the actual source node instead of assuming the new
        # CPU TaskRunner was scheduled on that node or sharing local disk.
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        nodes=[node for node in ray.nodes() if node['Alive'] and node['NodeManagerAddress']==manifest['source_container_ip']]
        if len(nodes)!=1:raise RuntimeError('The original episode source node is not uniquely alive')
        reader=ray.remote(num_cpus=1)(read_original_episode_payloads)
        payloads=await reader.options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[0]['NodeID'],soft=False)).remote(manifest)
        episodes=load_episodes(manifest,payloads)
        tasks,_=interleave_tasks(batch,4)
        fields=('question','ground_truth','data_source','source_row_index','original_question')
        key=lambda task:digest_json({name:task.get(name) for name in fields})
        if collections.Counter(map(key,tasks))!=collections.Counter(key(e.task) for e in episodes):
            raise RuntimeError('The new scheduled 18x4 batch differs from the original recovered episode tasks')
        self._q38_replay_consumed=True
        root=Path(os.environ['Q38_RUN_ROOT'])/'replay';root.mkdir(parents=True,exist_ok=True)
        with (root/'original-episodes-used.json').open('x') as stream:
            json.dump(dict(schema=SCHEMA,source_run_root=manifest['source_run_root'],source_policy_step=0,
                original_native_version=manifest['original_native_model_version'],episode_ids=[e.id for e in episodes],
                correct_episodes=sum(e.is_correct for e in episodes),new_generation_or_tool_execution_performed=False,
                unchanged_original_training_converter_will_run=True,original_ids_logprob_bits_rewards_preserved=True,
                time_ns=time.time_ns()),stream);stream.write('\n')
        return episodes

    start._q38_initial_policy_replay=True
    VerlBackend.on_train_start=start;VerlBackend.generate_episodes=generate
    return True
