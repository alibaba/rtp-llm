"""FSDP-to-RTP transactions with one native TP replica per training rank group."""
from __future__ import annotations
import asyncio
import contextlib
import hashlib
import json
import os
from pathlib import Path
import time
import ray
import torch
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.replica import RolloutReplica
from rtp_native_server import RTPServer, _audit_root, _base_port, _new_control, _sleep, _wake, _write_json, _source
from rtp_topology import ReplicaTopology, preserve_fp32_source, required_source_names, training_rank_layout


class RTPReplica(RolloutReplica):
    async def launch_servers(self):
        if self.config.data_parallel_size != 1 or self.config.pipeline_model_parallel_size != 1:
            raise NotImplementedError("use separate TP replicas for rollout DP; internal DP/PP are not implemented")
        tp=int(self.config.tensor_model_parallel_size)
        if self.world_size!=tp or len(self.workers)!=tp:
            raise ValueError("one complete RTP TP group is required")
        infos=await asyncio.gather(*[worker.__ray_call__.remote(lambda current:(
            ray.get_runtime_context().get_node_id(), ray.util.get_node_ip_address(),
            list(ray.get_runtime_context().get_accelerator_ids()["GPU"]))) for worker in self.workers])
        if any(len(info[2])!=1 for info in infos):
            raise RuntimeError("Every colocated FSDP worker must own exactly one physical GPU")
        topology=ReplicaTopology(replica_rank=self.replica_rank,tp_size=tp,
            gpu_ids=tuple(str(info[2][0]) for info in infos),rank_hosts=tuple(info[1] for info in infos),
            start_port=_base_port(self.replica_rank),
            rank_port_stride=int(os.environ.get("RLLM_RTP_RANK_PORT_STRIDE","9")),
            ep_size=int(os.environ.get("RTP_EP_SIZE","1")))
        node_ids=list(dict.fromkeys(info[0] for info in infos))
        local_world=int(os.environ.get('RLLM_RTP_GPUS_PER_NODE','4'))
        if (len(node_ids)!=self.nnodes or len(node_ids)!=len(topology.node_groups)
                or any(len(node['ranks'])!=local_world for node in topology.node_groups)):
            raise RuntimeError(f"RTP requires one complete {local_world}-GPU group per assigned Ray node")
        _write_json(_audit_root()/f"rtp_topology_{self.replica_rank}.json",topology.to_dict())
        from rtp_multinode_server import launch_multinode_server
        self._server_handle,self._native_owners=launch_multinode_server(
            self.model_config.path,topology.to_dict(),node_ids)
        self.servers=[self._server_handle]
        try:
            identity=await self._server_handle.prepare.remote()
            _write_json(_audit_root()/f'rtp_cluster_{self.replica_rank}.json',identity)
            await self._server_handle.launch_server.remote()
        except BaseException:
            await self._server_handle.shutdown.remote()
            raise
        self._server_address=f"http://{topology.node_groups[0]['host']}:{topology.start_port}"


class RTPServerAdapter(BaseRollout):
    def __init__(self,config,model_config,device_mesh,replica_rank=-1):
        super().__init__(config,model_config,device_mesh)
        if self.config.checkpoint_engine.backend!='naive':
            raise NotImplementedError("colocated RTP requires the naive checkpoint engine")
        if not self.config.free_cache_engine:
            raise ValueError("RTP requires free_cache_engine=true to pause inference around training")
        self.distributed=torch.distributed.is_initialized()
        self.global_rank=torch.distributed.get_rank() if self.distributed else int(os.environ.get('RANK','0'))
        world=torch.distributed.get_world_size() if self.distributed else 1
        layout=training_rank_layout(self.global_rank,world,int(self.config.tensor_model_parallel_size),
            int(self.config.data_parallel_size),int(self.config.pipeline_model_parallel_size))
        self.tp_rank=layout['tp_rank'];self.tp_size=layout['tp_size'];self.replica_rank=layout['replica_rank']
        self.group_ranks=layout['group_ranks'];self.replica_count=layout['replica_count']
        if replica_rank not in (-1,self.replica_rank):raise ValueError('explicit replica rank differs from the FSDP layout')
        self.tp_group=self.device_mesh['infer_tp'].get_group() if self.distributed else None
        if self.distributed and self.device_mesh.get_local_rank('infer_tp')!=self.tp_rank:
            raise ValueError('verl mesh and RTP TP rank order disagree')
        self.is_leader=self.tp_rank==0
        checkpoint=Path(self.model_config.path)
        self.hf_all_sources=set(json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map'])
        cfg=json.loads((checkpoint/'config.json').read_text())
        self.tied_embeddings=bool(cfg.get('tie_word_embeddings',cfg.get('text_config',{}).get('tie_word_embeddings',False)))
        self.hf_sources=None;self._last_sync_step=None;self._last_sync_version=None;self._unacknowledged_ipc=[]
        _write_json(_audit_root()/f'rtp_worker_layout_rank{self.global_rank}.json',dict(layout,
            global_rank=self.global_rank,leader=self.is_leader,physical_cuda_mask=os.environ.get('CUDA_VISIBLE_DEVICES')))

    def _broadcast(self,value):
        if not self.distributed:return value
        values=[value if self.is_leader else None]
        torch.distributed.broadcast_object_list(values,src=self.group_ranks[0],group=self.tp_group)
        return values[0]

    def _gather(self,value):
        if not self.distributed:return [value]
        values=[None]*self.tp_size
        torch.distributed.all_gather_object(values,value,group=self.tp_group)
        return values

    def _any_global_failure(self,error):
        flag=torch.tensor(int(error is not None),dtype=torch.int32,device=torch.device('cuda',torch.cuda.current_device()))
        if self.distributed:torch.distributed.all_reduce(flag,op=torch.distributed.ReduceOp.MAX)
        return bool(flag.item())

    async def _terminate_owned_replica(self):
        outcome=None
        if self.is_leader:
            try:
                server=ray.get_actor(f'rtp_server_{self.replica_rank}_0');receipt=await server.shutdown.remote()
                if not receipt.get('process_tree_terminated'):raise RuntimeError('native process tree termination was not acknowledged')
                outcome={'ok':True,'receipt':receipt}
            except BaseException as exc:outcome={'ok':False,'error':repr(exc)}
        outcome=self._broadcast(outcome)
        if not outcome['ok']:
            raise RuntimeError('unacknowledged native reader; IPC storage retained: '+outcome['error'])
        self._unacknowledged_ipc=[]

    async def _group_lifecycle(self,action):
        outcome=None
        if self.is_leader:
            try:
                await (_wake(self.replica_rank) if action=='wake' else _sleep(self.replica_rank));outcome={'ok':True}
            except BaseException as exc:outcome={'ok':False,'error':repr(exc)}
        outcome=self._broadcast(outcome)
        if not outcome['ok']:raise RuntimeError('RTP group lifecycle failed: '+outcome['error'])

    async def release(self):
        outcome=None
        if self.is_leader:
            try:
                server=ray.get_actor(f'rtp_server_{self.replica_rank}_0')
                receipt=await server.pause_for_training.remote()
                if not receipt.get('paused'):
                    raise RuntimeError('Every native TP rank must pause before training')
                outcome={'error':None,'receipt':receipt}
            except BaseException as error:
                outcome={'error':repr(error)}
        outcome=self._broadcast(outcome)
        if self._any_global_failure(outcome['error']):
            await self._terminate_owned_replica()
            raise RuntimeError('Native pause before training failed: '+str(outcome['error'] or 'another replica failed'))

    async def resume(self,tags):
        if 'kv_cache' not in tags:return
        error=None;probe=None
        try:
            await self._group_lifecycle('wake')
            if self.is_leader and os.environ.get('RLLM_RTP_POST_SYNC_PROBE','1')!='0':
                server=ray.get_actor(f'rtp_server_{self.replica_rank}_0')
                probe=await server.probe_synced_policy.remote(self._last_sync_step,self._last_sync_version)
        except BaseException as exc:error=repr(exc)
        status=self._broadcast({'error':error,'probe':probe});error=status['error'];probe=status['probe']
        if self._any_global_failure(error):
            await self._terminate_owned_replica()
            raise RuntimeError('RTP resume or post-update probe failed: '+str(error or 'another replica failed'))
        if probe is not None:
            signature=dict(replica_rank=self.replica_rank,tp_rank=self.tp_rank,tp_size=self.tp_size,
                global_step=probe['global_step'],source_sha256=probe['full_source_sha256'],
                model_version=probe['model_version'],raw_log_probs=probe['raw_log_probs'])
            signatures=[None]*(torch.distributed.get_world_size() if self.distributed else 1)
            if self.distributed:torch.distributed.all_gather_object(signatures,signature)
            else:signatures=[signature]
            if len({x['source_sha256'] for x in signatures})!=1 or len({x['global_step'] for x in signatures})!=1:
                await self._terminate_owned_replica();raise RuntimeError('RTP replicas received different source policies')
            reference=signatures[0]['raw_log_probs']
            if any(len(x['raw_log_probs'])!=len(reference) for x in signatures):
                await self._terminate_owned_replica();raise RuntimeError('RTP score probe shapes differ across replicas')
            error_max=max(abs(a-b) for x in signatures for a,b in zip(reference,x['raw_log_probs']))
            if error_max>float(os.environ.get('RLLM_RTP_REPLICA_SCORE_TOLERANCE','0.05')):
                await self._terminate_owned_replica();raise RuntimeError('RTP score probes differ across replicas')
            distribution=dict(passed=True,independent_rollout_replicas=self.replica_count,training_ranks=len(signatures),
                probe_broadcast_within_tp_group=True,cross_tp_compute_parity_validated=False,
                max_score_error_across_replicas=error_max,signatures=signatures)
            if self.is_leader:
                _write_json(_audit_root()/f"rtp_policy_distribution_replica{self.replica_rank}_step{probe['global_step']}.json",distribution)
            if self.global_rank==0:
                _write_json(_audit_root()/f"rtp_policy_distribution_step{probe['global_step']}.json",distribution)

    def _required_sources(self):
        names=required_source_names(_audit_root()/f'rtp_weight_manifest_{self.replica_rank}',self.tp_size)
        if names-self.hf_all_sources:raise RuntimeError('native loader requires keys outside the checkpoint')
        return names

    def _local_ipc_binding(self,global_step):
        from rtp_multinode_server import canonical_uuid
        import socket
        native=json.loads((_audit_root()/f'rtp_weight_manifest_{self.replica_rank}'/
                           f'rank-{self.tp_rank:05d}.json').read_text())
        identity=canonical_uuid(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        node_ip=ray.util.get_node_ip_address()
        if (native['owner_ip']!=node_ip or canonical_uuid(native['gpu_uuid'])!=identity
                or int(native['tp_rank'])!=self.tp_rank or int(native['tp_size'])!=self.tp_size):
            raise RuntimeError('CUDA IPC source and native receiver are not colocated on the same physical GPU')
        proof=dict(passed=True,replica_rank=self.replica_rank,tp_rank=self.tp_rank,global_rank=self.global_rank,
            owner_ip=node_ip,owner_hostname=socket.gethostname(),gpu_uuid=identity,
            native_pid=int(native['pid']),native_boot=int(native['boot_id']),
            native_incarnation=native['incarnation'],global_step=int(global_step),checked_unix=time.time())
        _write_json(_audit_root()/f'rtp_ipc_binding_replica{self.replica_rank}_tprank{self.tp_rank}_step{global_step}.json',proof)
        return proof

    @torch.no_grad()
    async def update_weights(self,weights,global_steps=None,**kwargs):
        # Private opt-in: own the reusable bounded staging workspace for the
        # full transaction, including failure/iterator unwind. This wrapper is
        # intentionally outside the source/ACK handling below.
        digest_workspaces=[]
        try:
            return await self._update_weights_digest_impl(weights,global_steps,digest_workspaces,**kwargs)
        finally:
            import sys
            in_flight_error=sys.exc_info()[0] is not None
            for workspace in digest_workspaces:
                try:workspace.close()
                except BaseException:
                    if not in_flight_error:raise

    async def _update_weights_digest_impl(self,weights,global_steps,digest_workspaces,**kwargs):
        if kwargs.get('peft_config') is not None:raise NotImplementedError('RTP updates full model parameters')
        profile=None
        if os.environ.get('Q36_RTP_PROFILE_REFIT','0')=='1':
            from q36_source_binding import verify_source
            verify_source('q36_refit_timing')
            from q36_refit_timing import RefitTimings
            profile=RefitTimings(True)
        stage=lambda name: profile.stage(name) if profile is not None else contextlib.nullcontext()
        _source()
        from rtp_llm.utils.cuda_ipc import CudaIpcV2
        control=None;failure=None;before=None;tensors=[];aliases={};source_digest=hashlib.sha256()
        restart=None
        if self.is_leader:
            try:
                server=ray.get_actor(f'rtp_server_{self.replica_rank}_0')
                restart=await server.ensure_native_running.remote(int(global_steps or 0),self._last_sync_version)
                restart['error']=None
            except BaseException as error:
                restart={'error':repr(error)}
        restart=self._broadcast(restart)
        if restart['error'] is not None:
            failure='Native readiness before refit failed: '+restart['error']
        self._unacknowledged_ipc=[]
        binding=None
        try:
            if failure is not None:
                raise RuntimeError(failure)
            self.hf_sources=self._required_sources()
            binding=self._local_ipc_binding(int(global_steps or 0))
        except BaseException as exc:failure=repr(exc)
        manifest_checks=self._gather({'error':failure,'binding':binding})
        if any(x['error'] is not None for x in manifest_checks):
            failure=str([x['error'] for x in manifest_checks if x['error'] is not None])
        setup=None
        if self.is_leader:
            try:
                if failure is not None:raise RuntimeError(failure)
                with stage('rpc_setup_status_pause_begin'):
                    control=await _new_control(self.replica_rank);before=(await control.status())[0]
                    if before.phase=='running':await control.pause()
                    await control.begin_update(global_step=int(global_steps or 0),
                        max_buffer_bytes=int(os.environ.get('RLLM_RTP_UPDATE_BUFFER_BYTES',str(16*1024**3))))
                setup={'error':None,'before_version':before.model_version}
            except BaseException as exc:setup={'error':repr(exc)}
        setup=self._broadcast(setup)
        if setup['error'] is not None:failure=setup['error']
        ack_bytes = 0
        last_progress = None
        # Exhaust this FSDP collective iterator even after a native transaction failure.
        for name,tensor in (profile.iterate(weights) if profile is not None else weights):
            if failure is not None:continue
            with stage('tp_name_gather'):
                source_names=self._gather(name)
            if len(set(source_names))!=1:
                failure='trainer export name order differs across TP ranks: '+str(source_names)
                continue
            if name in self.hf_all_sources and name not in self.hf_sources:continue
            item={'name':name,'error':None};value=None
            try:
                if name not in self.hf_sources:
                    if name=='lm_head.weight' and self.tied_embeddings:
                        aliases[name]=hashlib.sha256(tensor.detach().to(torch.bfloat16).cpu().view(torch.uint8).numpy().tobytes()).hexdigest();continue
                    raise RuntimeError('trainer exported key outside native source manifest: '+name)
                dtype=torch.float32 if preserve_fp32_source(name) else torch.bfloat16
                if preserve_fp32_source(name) and tensor.dtype!=torch.float32:
                    raise RuntimeError('FP32 GDN source was downcast by the training exporter: '+name)
                local_device = torch.device('cuda', torch.cuda.current_device())
                if tensor.device.type != 'cpu' and tensor.device != local_device:
                    raise RuntimeError(f'Export for {name} is on {tensor.device}, expected CPU or {local_device}')
                # An offloaded FSDP2 export is staged on this TP worker's own GPU
                # for this tensor only. Native readers retain the buffer until ACK.
                with stage('device_stage_cast_contiguous'):
                    value=tensor.detach().to(device=local_device, dtype=dtype).contiguous()
                self._unacknowledged_ipc=[value]
                # Keep the packed expert in one host buffer for both SHA
                # passes. Updating with prefix and bytes separately preserves the digest.
                if os.environ.get('Q36_REFIT_BOUNDED_DIGEST','0')=='1':
                    from q36_source_binding import verify_source
                    verify_source('q36_refit_bounded_digest_candidate')
                    from q36_refit_bounded_digest_candidate import CudaDigestWorkspace
                    if not digest_workspaces:
                        digest_workspaces.append(CudaDigestWorkspace(local_device))
                    with stage('bounded_device_to_host_and_sha256'):
                        checksum,source_bytes=digest_workspaces[0].hash(value,name,source_digest)
                else:
                    with stage('device_to_host_for_hash'):
                        payload=memoryview(value.cpu().view(torch.uint8).numpy()).cast('B')
                    with stage('host_sha256'):
                        checksum=hashlib.sha256(payload).hexdigest()
                with stage('ipc_export_and_device_fence'):
                    metadata=CudaIpcV2.export_tensor(value)
                item.update(shape=list(value.shape),dtype=str(value.dtype),sha256=checksum,metadata=metadata)
                if os.environ.get('Q36_REFIT_BOUNDED_DIGEST','0')!='1':
                    with stage('host_sha256'):
                        source_digest.update(name.encode()+b'\0'+str(tuple(value.shape)).encode()+b'\0'+str(value.dtype).encode()+b'\0')
                        source_digest.update(payload)
                    source_bytes=payload.nbytes
                    del payload
            except BaseException as exc:item['error']=repr(exc)
            with stage('tp_metadata_gather'):
                rank_items=self._gather(item)
            outcome=None
            if self.is_leader:
                try:
                    errors=[x['error'] for x in rank_items if x['error'] is not None]
                    if errors:raise RuntimeError(str(errors))
                    for peer in rank_items:
                        if any(peer[k]!=rank_items[0][k] for k in ['name','shape','dtype','sha256']):
                            raise RuntimeError('TP source tensor identity differs across training ranks')
                    with stage('rpc_push_through_native_ack'):
                        await control.push_weight(name,[x['metadata'] for x in rank_items])
                    outcome={'error':None}
                except BaseException as exc:outcome={'error':repr(exc)}
            with stage('tp_ack_broadcast'):
                outcome=self._broadcast(outcome)
            if outcome['error'] is not None:failure=outcome['error'];continue
            # Native ACK synchronizes all reads and owns its pending group copies.
            self._unacknowledged_ipc=[];value=None
            tensors.append({k:item[k] for k in ['name','shape','dtype','sha256']})
            ack_bytes += source_bytes
            if profile is not None:profile.record_bytes(source_bytes)
            now = time.monotonic()
            if self.is_leader and (last_progress is None or now-last_progress >= 30 or len(tensors) == len(self.hf_sources)):
                progress = dict(step=int(global_steps or 0), acked_sources=len(tensors),
                    required_sources=len(self.hf_sources), acked_bytes=ack_bytes,
                    last_source=name, updated_unix=time.time())
                print('RTP_REFIT_PROGRESS '+json.dumps(progress), flush=True)
                try:
                    _write_json(_audit_root()/f'rtp_refit_progress_{self.replica_rank}_step{int(global_steps or 0)}.json',progress)
                except OSError as error:
                    print(f'RTP refit progress file unavailable: {error}', flush=True)
                last_progress = now
        finish=None
        if self.is_leader:
            try:
                if failure is not None:raise RuntimeError(failure)
                if {x['name'] for x in tensors}!=self.hf_sources:raise RuntimeError('incomplete native HF source coverage')
                if aliases:
                    embedding=next(x for x in tensors if x['name'].endswith('embed_tokens.weight'))
                    if aliases['lm_head.weight']!=embedding['sha256']:raise RuntimeError('tied head and embedding differ')
                with stage('rpc_commit_through_native_ack'):
                    replies=await control.commit_update()
                committed=replies[0]
                if committed.model_version==setup['before_version'] or committed.global_step!=int(global_steps or 0):
                    raise RuntimeError('RTP did not publish the expected new policy version')
                if committed.received_weights!=len(tensors) or committed.applied_weights<=0:
                    raise RuntimeError('RTP did not acknowledge complete internal parameter application')
                finish={'error':None,'global_step':committed.global_step,'model_version':committed.model_version,
                    'native_boot':restart['native_boot'],'native_incarnation':restart['native_incarnation'],
                    'previous_committed_version':self._last_sync_version,
                    'restarted_since_previous_commit':restart['restarted'],
                    'received_weights':committed.received_weights,'applied_weights':committed.applied_weights,
                    'rank_status':[{'tp_rank':x.tp_rank,'tp_size':x.tp_size,'model_version':x.model_version,
                                    'global_step':x.global_step,'phase':x.phase} for x in replies]}
            except BaseException as exc:finish={'error':repr(exc)}
            finally:
                if control is not None:await control.close()
        finish=self._broadcast(finish)
        if finish['error'] is not None:failure=finish['error']
        if self._any_global_failure(failure):
            await self._terminate_owned_replica()
            raise RuntimeError('RTP distributed weight synchronization failed: '+str(failure or 'another replica failed'))
        self._last_sync_step=finish['global_step'];self._last_sync_version=finish['model_version']
        record=dict(engine='rtp-llm',replica_rank=self.replica_rank,tp_size=self.tp_size,source_training_ranks=self.group_ranks,
            global_step=finish['global_step'],model_version=finish['model_version'],before_model_version=setup['before_version'],
            native_boot=finish['native_boot'],native_incarnation=finish['native_incarnation'],
            previous_committed_version=finish['previous_committed_version'],
            restarted_since_previous_commit=finish['restarted_since_previous_commit'],
            native_received_weights=finish['received_weights'],native_applied_weights=finish['applied_weights'],
            full_source_sha256=source_digest.hexdigest(),hf_tensor_count=len(tensors),tensors=tensors,
            fp32_sources=[x['name'] for x in tensors if x['dtype']=='torch.float32'],rank_status=finish['rank_status'],
            ipc_bindings=[item['binding'] for item in manifest_checks],
            transport='native_rl_cuda_ipc_v2_per_tp_rank',source_release='per-tensor all-TP ACK',
            kv_pool_retained_while_paused=os.environ.get('RLLM_RTP_KEEP_KV_CACHE','1')=='1',
            kv_pool_retained_across_training=os.environ.get('RLLM_RTP_KEEP_KV_CACHE','1')=='1',
            native_lifecycle='resident_native_pause_training_refit_resume',finished_unix=time.time())
        if self.is_leader:
            _write_json(_audit_root()/f'rtp_sync_rank{self.replica_rank}_step{int(global_steps or 0)}.json',record)
            print(json.dumps({k:v for k,v in record.items() if k!='tensors'}),flush=True)
        if profile is not None:
            timing=profile.snapshot()
            timing.update(global_rank=self.global_rank,tp_rank=self.tp_rank,replica_rank=self.replica_rank,
                          global_step=int(global_steps or 0),leader=self.is_leader,
                          status='committed',native_applied_weights=finish['applied_weights'])
            try:
                _write_json(_audit_root()/f'rtp_refit_timing_rank{self.global_rank}_step{int(global_steps or 0)}.json',timing)
            except OSError as error:
                print(f'RTP refit timing file unavailable: {error}', flush=True)
            print('Q36_RTP_REFIT_TIMING '+json.dumps(timing,sort_keys=True),flush=True)
        self._unacknowledged_ipc=[]
