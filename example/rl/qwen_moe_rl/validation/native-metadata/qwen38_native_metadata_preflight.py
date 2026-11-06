"""Actual72 native metadata-only boot/status/owned teardown; never training.

Run only when root explicitly releases the preflight start gate. This uses the
production world72 factory and native launcher without creating HF workers.
"""
from __future__ import annotations
import asyncio,hashlib,json,os
from pathlib import Path
import signal,socket,sys,time,traceback


def require(condition,message):
    if not condition:raise RuntimeError(message)


def write(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.writing-{os.getpid()}')
    temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def owner_snapshot(owner):
    """Runs in the already-owned CPU actor; queries NVML, never CUDA APIs."""
    import json
    from pathlib import Path
    from qwen38_world_server import gpu_inventory
    with owner._process_lock:
        owner._capture_process_tree()
        owned=dict(owner._owned_processes);process_identity=owner.process_identity
    root=owner.main_audit_root/f'world-boots/boot-{owner.boot:03d}/node-{owner.node_index:02d}'
    files={}
    for name in ('local-startup.json','actual-nccl.json','rtp_process_0.json','rtp_shutdown_0.json'):
        path=root/name
        if path.exists():files[name]=json.loads(path.read_text())
    mappings=[];handles=[];inspection_errors=[]
    for row in owner._local_manifests:
        pid=row['pid'];base=Path('/proc')/str(pid)
        try:
            mappings.extend(dict(pid=pid,line=line) for line in (base/'maps').read_text().splitlines()
                if '.safetensors' in line and owner.model_path in line)
            for fd in (base/'fd').iterdir():
                try:target=str(fd.resolve(strict=True))
                except (OSError,RuntimeError):continue
                if target.startswith(owner.model_path+'/') and '.safetensors' in target:
                    handles.append(dict(pid=pid,fd=fd.name,path=target))
        except OSError as error:inspection_errors.append(dict(pid=pid,error=repr(error)))
    return dict(node_index=owner.node_index,owner_identity=owner.owner_identity,
        process_identity=process_identity,owned_process_identities=[dict(pid=int(pid),create_time=created) for pid,created in owned.items()],
        rank_manifests=list(owner._local_manifests),files=files,inventory=gpu_inventory(),
        current_safetensors_mappings=mappings,current_safetensors_file_handles=handles,
        process_inspection_errors=inspection_errors,finished_unix=time.time())


def controller_snapshot(controller):
    from pathlib import Path
    path=controller.main_audit_root/'rtp_boot_0_step0.json'
    return dict(identity=controller.controller_identity,started=controller._started,
        generation_admission_open=controller._generation_ready,committed_step=controller._committed_step,
        committed_version=controller._committed_version,inflight=controller._inflight,
        native_boot=controller.boot,native_incarnation=controller.incarnation,
        boot_proof=json.loads(path.read_text()) if path.exists() else None,
        current_source_transactions=[str(p) for p in controller.main_audit_root.glob('rtp_sync_rank0_step*.json')])


async def await_ref(ref):return await ref


async def run():
    require(os.environ.get('Q38_NATIVE_METADATA_BOOT_AUTHORIZED')=='1',
        'Metadata72 launch is only allowed after root releases all GPU small tests and deploys every dependency')
    inputs_path=Path(os.environ['Q38_NATIVE_METADATA_INPUT_FILE']);raw=inputs_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==os.environ['Q38_NATIVE_METADATA_INPUT_SHA256'],'Frozen preflight input SHA differs')
    inputs=json.loads(raw);namespace=os.environ['Q38_NATIVE_METADATA_NAMESPACE']
    require(namespace.startswith('q38meta-') and namespace not in ('default','auto'), 'Use a fresh dedicated metadata-only Ray namespace')
    root=Path(os.environ['Q38_RUN_ROOT']).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'metadata-preflight.claim').open('x') as handle:handle.write(namespace+'\n')
    require(os.environ['RLLM_RTP_ACCEPTANCE_DIR']==str(root/'rtp'),'Recipe did not bind the preflight run audit root')
    require(os.environ.get('RTP_MODEL_PATH')==inputs['model_path'],'Metadata preflight must use the existing official mounted checkpoint')
    profile={'RTP_LLM_RL_MODE':'1','RTP_LLM_RL_WORLD72':'1','RTP_LLM_RL_START_PAUSED':'1','RTP_LLM_RL_METADATA_BOOTSTRAP':'1',
        'RTP_FIXED_PHYSICAL_EP_LAYOUT':'balanced_v1','RTP_RL_BORROW_SINGLETON_HF_SOURCES':'1',
        'RTP_NCCL_EP_REFERENCE_MAX_PADDED_TOKENS':'73728','PYTORCH_ALLOC_CONF':'backend:native,expandable_segments:False'}
    require(all(os.environ.get(name)==value for name,value in profile.items()) and 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ,
        'The real recipe/native bootstrap and native allocator profile must be explicit before Ray starts')
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['RAY_USAGE_STATS_ENABLED']='0'
    import psutil,ray,torch
    from ray.util.placement_group import placement_group,remove_placement_group
    from qwen38_world_contract import World72Topology,canonical_uuid,validate_world_manifests
    from qwen38_world_server import launch_world72_server,CONTROL_TOPOLOGY,_status_rows
    from qwen38_owned_cleanup import validate_identity,validate_shutdown
    from qwen38_training_contract import load_metadata
    from rtp_llm.RL.replica_rpc_client import ReplicaRpcClient
    require(not torch.cuda.is_initialized(),'The CPU metadata driver must not create a CUDA context')
    _,_,sources,_=load_metadata(inputs['model_path'])
    current=psutil.Process()
    identity=dict(pid=os.getpid(),create_time=current.create_time(),pgid=os.getpgid(os.getpid()),host=socket.gethostname(),
        source_path=str(Path(__file__).resolve()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        namespace=namespace,address=inputs['address'],input_sha256=hashlib.sha256(raw).hexdigest(),started_unix=time.time())
    write(root/'metadata-driver-identity.json',identity);write(root/'metadata-inputs.json',inputs)
    stop=asyncio.Event();loop=asyncio.get_running_loop()
    def requested(signal_number):
        write(root/'metadata-stop-requested.json',dict(signal=int(signal_number),time=time.time()))
        stop.set()
    for number in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(number,requested,number)
    runtime_env={'env_vars':{key:value for key,value in os.environ.items() if key.startswith(('RTP_','RLLM_','Q38_','NCCL_'))
        or key in ('PYTHONPATH','LD_PRELOAD','LD_LIBRARY_PATH','OMP_NUM_THREADS','PYTHONDONTWRITEBYTECODE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTORCH_ALLOC_CONF')}}
    runtime_env['env_vars']['RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES']='1'
    controller=None;owners=[];controller_identity=None;pg=None;cleanup=None;bootstrap=None;error=None;cleanup_errors=[];actors_killed=[]
    stage='connecting_existing_owned_Ray';heartbeat_task=None;boot_task=None
    async def heartbeat():
        while True:
            write(root/'metadata-heartbeat.json',dict(stage=stage,stop_requested=stop.is_set(),time=time.time(),**identity))
            await asyncio.sleep(5)
    ray.init(address=inputs['address'],namespace=namespace,runtime_env=runtime_env,logging_level='ERROR')
    try:
        heartbeat_task=asyncio.create_task(heartbeat())
        live=[node for node in ray.nodes() if node['Alive']];by_ip={node['NodeManagerAddress']:node for node in live}
        require(len(live)==18 and set(by_ip)=={node['container_ip'] for node in inputs['nodes']},'Ray is not the exact current18-container allocation')
        require(all(node['Resources'].get('GPU')==4 for node in live),'Every current Ray node must advertise exactly its4 allocated GPUs')
        require(len(ray.util.list_named_actors(all_namespaces=False))==0,'The metadata namespace is not fresh')
        stage='reserving_only_GPU_resources_no_HF_workers'
        pg=placement_group([{'GPU':4,f'node:{node["container_ip"]}':0.001} for node in inputs['nodes']],strategy='PACK',name=namespace+'-GPU-lease')
        await asyncio.wait_for(await_ref(pg.ready()),90)
        node_ids=[by_ip[node['container_ip']]['NodeID'] for node in inputs['nodes']]
        topology=World72Topology(gpu_ids=tuple('GPU-'+value for value in inputs['rank_gpu_uuids']),
            gpu_uuids=tuple(inputs['rank_gpu_uuids']),rank_hosts=tuple(node['container_ip'] for node in inputs['nodes'] for _ in range(4)),
            start_port=inputs['start_port'])
        os.environ['RLLM_RTP_START_PORT']=str(topology.start_port)
        stage='preparing_actual18_CPU_owners'
        controller,owners=launch_world72_server(inputs['model_path'],topology.to_dict(),node_ids)
        controller_identity=await controller.prepare.remote()
        validate_identity(controller_identity,root/'rtp',Path(os.environ['RLLM_RTP_DEPLOY_ROOT']).resolve(),namespace)
        inventories=[row['inventory'] for row in controller_identity['node_owners']]
        require(all(not row['processes'] for row in inventories),'A GPU still has a pre-existing compute process; native preflight did not start')
        write(root/'metadata-cpu-prepared.json',dict(controller=controller_identity,topology=topology.to_dict(),
            preexisting_GPU_processes=[],placement_group_id=pg.id.hex(),fake_HF_workers_created=0))
        if stop.is_set():raise InterruptedError('Root stopped preflight before any native GPU allocation')
        stage='native72_START_PAUSED_metadata_target_allocation'
        boot_task=asyncio.create_task(await_ref(controller.ensure_native_running.remote(0,None)))
        stopped=asyncio.create_task(stop.wait())
        done,_=await asyncio.wait((boot_task,stopped),return_when=asyncio.FIRST_COMPLETED)
        if stopped in done and stop.is_set():raise InterruptedError('Root requested metadata preflight stop during native startup')
        stopped.cancel();restart=await boot_task
        if stop.is_set():raise InterruptedError('Root requested preflight stop immediately after native startup')
        stage='checking_actual72_proto3_status_and_descriptors'
        client=ReplicaRpcClient(topology.control_addresses,timeout=120,topology=CONTROL_TOPOLOGY)
        try:statuses=await client.status()
        finally:await client.close()
        require(len(statuses)==72 and all(row.phase=='paused' and row.quiescent and not row.update_id
            and row.received_weights==0 and row.applied_weights==0 and row.protocol_version==3 for row in statuses),
            'Every real native rank must be paused/quiescent with zero committed/pushed source weights')
        require(all(row.model_version.startswith(restart['native_incarnation']+':') for row in statuses),'Native status belongs to another incarnation')
        owner_views=await asyncio.gather(*[owner.__ray_call__.remote(owner_snapshot) for owner in owners])
        observed=await controller.__ray_call__.remote(controller_snapshot)
        require(observed['started'] is True and observed['generation_admission_open'] is False
            and observed['committed_step'] is None and observed['committed_version'] is None
            and observed['inflight']==0 and not observed['current_source_transactions'],
            'A source transaction, committed policy or generation admission appeared during metadata-only preflight')
        manifests=validate_world_manifests([row for view in owner_views for row in view['rank_manifests']],
            topology,sources,boot=restart['native_boot'],incarnation=restart['native_incarnation'])
        expected_native=os.environ['RTP_RL_NATIVE_LIBRARY_SHA256'];expected_nccl=os.environ['RLLM_Q38_NCCL_LIBRARY_SHA256']
        for view in owner_views:
            require(not view['process_inspection_errors'] and not view['current_safetensors_mappings']
                and not view['current_safetensors_file_handles'],'A native process has unexpected weight-file mappings/handles or cannot be inspected')
            owned={row['pid'] for row in view['owned_process_identities']}
            require(all(row['pid'] in owned for row in view['inventory']['processes']),'An unrelated GPU compute process appeared during preflight')
            rank_pids={row['pid'] for row in view['rank_manifests']}
            require(rank_pids.issubset({row['pid'] for row in view['inventory']['processes']}),'A native manifest rank does not own an actual GPU context')
            actual=view['files'].get('actual-nccl.json',{}).get('actual_mappings',[])
            require(len(actual)==4 and all(row['library_sha256']==expected_nccl for row in actual),'Actual mapped native NCCL differs from the qualified binding')
        require(all(row['bootstrap_native_library']['native_library_sha256']==expected_native for row in manifests),
            'The actual72 loaded native libraries differ from the accepted fresh artifact')
        bootstrap=dict(schema='q38-actual72-native-metadata-only-bootstrap-v1',restart=restart,statuses=_status_rows(statuses),
            manifests=manifests,nodes=owner_views,controller=observed,physical_GPU_UUIDs=[canonical_uuid(row['gpu_uuid']) for row in manifests],
            full_model_HF_workers=0,weight_update_commands=[],generation_or_training_performed=False,
            current_weight_payload_mappings_or_handles=[],historical_kernel_byte_IO_tracing_performed=False,
            weight_access_scope='Pinned metadata-only allocation contract; no native source transaction; current process mappings/FDs independently inspected',
            finished_unix=time.time())
        write(root/'metadata-bootstrap.json',bootstrap)
        stage='verified_owned18_native_teardown'
    except BaseException as exception:
        error=repr(exception);write(root/'metadata-failure.json',dict(stage=stage,error=error,traceback=traceback.format_exc()))
    finally:
        if controller is not None:
            try:
                # This identity RPC deliberately does not wait behind startup.
                controller_identity=await controller.owned_identity.remote()
                validate_identity(controller_identity,root/'rtp',Path(os.environ['RLLM_RTP_DEPLOY_ROOT']).resolve(),namespace)
                if bootstrap is not None and not stop.is_set():
                    cleanup=await asyncio.wait_for(await_ref(controller.suspend_for_training.remote()),240)
                    # Permanent identity-bound stop also blocks any queued new
                    # start. It must return the same actual completed receipt.
                    final=await asyncio.wait_for(await_ref(controller.shutdown_if_owned.remote(controller_identity['pid'],
                        controller_identity['create_time'],str(root/'rtp'))),240)
                    require(final==cleanup,'Permanent stop did not retain the actual metadata-only cleanup proof')
                else:
                    cleanup=await asyncio.wait_for(await_ref(controller.shutdown_if_owned.remote(controller_identity['pid'],
                        controller_identity['create_time'],str(root/'rtp'))),240)
                validate_shutdown(cleanup)
                require(cleanup.get('training_admission_safe') is False,
                    'Metadata-only cleanup must never claim training-safe without a committed/probed policy')
                if bootstrap is not None:
                    require(len(cleanup.get('nodes',[]))==18 and cleanup.get('checked_native_ranks')==list(range(72)),
                        'Real metadata72 cleanup must cover all18 native leader trees and72 physical ranks')
                    require(all(type(row.get('leader_actual_returncode')) is int for row in cleanup['nodes']),
                        'A native leader has no actual Popen exit code')
                inventories=await controller.node_inventories.remote()
                require(len(inventories)==18 and all(not row['processes'] for row in inventories),
                    'Not every current GPU compute context has actually disappeared after preflight cleanup')
                write(root/'metadata-cleanup.json',dict(schema='q38-actual72-metadata-only-owned-cleanup-v1',
                    controller_proof=cleanup,actual_post_cleanup_inventories=inventories,
                    cleanup_only_not_training_safe=True,all72_compute_contexts_absent=True,finished_unix=time.time()))
            except BaseException as exception:
                cleanup_errors.append(repr(exception));write(root/'metadata-cleanup-failure.json',
                    dict(error=repr(exception),controller_identity=controller_identity,partial_proof=cleanup,traceback=traceback.format_exc()))
            if not cleanup_errors:
                for actor in [controller,*owners]:
                    try:ray.kill(actor,no_restart=True);actors_killed.append(str(actor))
                    except BaseException as exception:cleanup_errors.append('owned CPU actor release: '+repr(exception))
        if pg is not None and not cleanup_errors:remove_placement_group(pg)
        if heartbeat_task is not None:heartbeat_task.cancel()
        result=dict(schema='q38-actual72-native-metadata-preflight-result-v1',passed=bootstrap is not None and error is None and not cleanup_errors,
            namespace=namespace,driver_identity=identity,metadata_bootstrap_completed=bootstrap is not None,
            actual72_native_GPU_UUIDs_verified=bootstrap is not None,descriptor_sources_per_rank=1590 if bootstrap is not None else None,
            actual18_native_leader_exits=([row.get('leader_actual_returncode') for row in cleanup.get('nodes',[])] if cleanup else []),
            all_owned_native_contexts_released=cleanup is not None and not cleanup_errors,
            training_admission_safe=False,complete_policy_commits=0,HF_workers_created=0,
            weight_refit_generation_training_performed=False,full_model_training_accepted=False,
            stop_requested=stop.is_set(),error=error,cleanup_errors=cleanup_errors,owned_CPU_actor_release_requests=len(actors_killed),
            external_Ray_cluster_left_running=True,finished_unix=time.time())
        write(root/'metadata.RESULT.json',result)
        print('METADATA72_PREFLIGHT_RESULT '+json.dumps(result),flush=True)
        ray.shutdown()
    return 0 if result['passed'] else 2


if __name__=='__main__':
    raise SystemExit(asyncio.run(run()))
