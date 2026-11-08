"""Explicit opt-in verl integration for the pinned 72-rank BF16 policy/state path.

Import through actor_rollout_ref.model.external_lib. Nothing patches installed
files. All hooks are restricted to the exact Qwen3.8 config and 72-rank mesh.
The RTP bridge must acknowledge each source lease and actual native teardown.
"""
from __future__ import annotations

import functools
import inspect
import json
import math
import os
from pathlib import Path
import socket
import time

from qwen38_bf16_streamed_checkpoint import (is_qwen38, meta_checkpoint_model,
                                      load_checkpoint_shards,restore_initialized_buffers_after_engine_cast)
from qwen38_training_contract import POLICY_NUMEL, WORLD_SIZE

_INSTALLED=False


def _is_target(engine):return is_qwen38(engine.model_config.hf_config)


def _optimizer_step_after_global_clip(engine,norm,reduce_changed):
    """Opt-in acceptance gate; leave valid optimizer arithmetic untouched."""
    required=os.environ.get('RLLM_REQUIRE_REAL_UPDATE')=='1'
    engine._q38_last_real_update_gate=dict(enabled=required,global_gradient_norm=float(norm),
                                          global_changed_weight_elements=None)
    if required and (not math.isfinite(float(norm)) or norm<=0):
        raise RuntimeError('RLLM_REQUIRE_REAL_UPDATE refuses nonpositive/nonfinite global gradients before Adam allocation/scan')
    engine.optimizer.step()
    if required:
        changed=int(reduce_changed(int(engine.optimizer.changed_weight_elements)))
        engine._q38_last_real_update_gate['global_changed_weight_elements']=changed
        if changed<=0:
            raise RuntimeError('RLLM_REQUIRE_REAL_UPDATE refuses a successful step with no globally changed BF16 weights')
        engine._q38_last_real_update_gate['changed_count_world_size']=WORLD_SIZE


def _audit(engine, boundary):
    import torch.distributed as dist
    directory=Path(os.environ['Q38_TRAINING_AUDIT_DIR']);directory.mkdir(parents=True,exist_ok=True)
    record={'schema':'qwen38-BF16-state-engine-boundary-v1','boundary':boundary,'rank':dist.get_rank(),
            'world_size':dist.get_world_size(),'time_ns':time.time_ns(),'pid':os.getpid(),
            'phase':engine._q38_phases.phase,'optimizer':engine.optimizer.audit(),
            'phase_transition_records':engine._q38_phases.transition_records,
            'real_update_gate':getattr(engine,'_q38_last_real_update_gate',None),
            'native_contexts_released':getattr(engine,'_q38_native_contexts_released',False),
            'last_native_cleanup_proof':getattr(engine,'_q38_last_native_cleanup_proof',None),
            'last_refit_source_release':getattr(engine,'_q38_last_refit_source_release',None),
            'initial_policy_replay_proof':getattr(engine,'_q38_initial_policy_replay_proof',None)}
    with (directory/f'{boundary}-rank{dist.get_rank()}-{record["time_ns"]}.json').open('x') as stream:
        json.dump(record,stream);stream.write('\n')


def install():
    global _INSTALLED
    if _INSTALLED:return
    if os.environ.get('Q38_ENABLE_BF16_STATE_RUNTIME')!='1':
        raise RuntimeError('Explicit Q38_ENABLE_BF16_STATE_RUNTIME=1 is required')
    import torch
    import torch.distributed as dist
    # The current base image's cuDNN SDPA plugin has a sublibrary mismatch.
    # Keep native Torch SDPA Flash/math dispatch, with cuDNN SDPA excluded.
    torch.backends.cuda.enable_cudnn_sdp(False)
    import verl.utils.model as model_utils
    import verl.models.transformers.monkey_patch as monkey_patch
    import verl.utils.fsdp_utils as fsdp_utils
    import verl.workers.engine.fsdp.transformer_impl as engine_module
    from qwen38_chunked_experts import ExpertChunk,shard_expert_chunks_before_decoder_wrap
    from qwen38_vocab_tiles import VocabTile,shard_vocab_tiles_before_root_wrap
    from qwen38_bf16_optimizer import GradientStore,CpuBf16Moments,Bf16CpuAdamW,local
    from qwen38_bf16_master_phases import MasterPhaseManager,disable_prefetch_and_bound_reduction
    from qwen38_canonical_export import CanonicalExport
    from qwen38_policy_probe import audit_policy_before_rollout
    Engine=engine_module.FSDPEngine
    from verl.workers.engine.base import BaseEngineCtx
    original_context_switch=BaseEngineCtx._context_switch
    if list(inspect.signature(engine_module.fsdp2_load_full_state_dict).parameters)!=[
            'model','full_state','device_mesh','cpu_offload']:
        raise RuntimeError('Review the actual verl checkpoint-streaming hook signature')
    original_factory=model_utils.get_hf_auto_model_class
    original_patch=monkey_patch.apply_monkey_patch
    original_wrap=engine_module.apply_fsdp2
    original_load=engine_module.fsdp2_load_full_state_dict
    original_optimizer=Engine._build_optimizer
    original_build_module=Engine._build_module
    original_step=Engine.optimizer_step
    original_to=Engine.to
    original_export=Engine.get_per_tensor_param

    def factory(hf_config):
        base=original_factory(hf_config)
        if not is_qwen38(hf_config):return base
        class Qwen38MetaAuto:
            @classmethod
            def from_pretrained(cls,*args,**kwargs):
                # The real verl engine passes config= explicitly. Consume it
                # before forwarding the selected config positionally, while
                # retaining the original exact-instance guard.
                options=dict(kwargs)
                if options.pop('config',hf_config) is not hf_config:
                    raise ValueError('The selected Qwen3.8 config instance must be preserved')
                return meta_checkpoint_model(base,hf_config,*args,**options)
        return Qwen38MetaAuto

    def patch(model,ulysses_sp_size=1,use_remove_padding=True,use_fused_kernels=False,**kwargs):
        if not is_qwen38(model.config):
            return original_patch(model,ulysses_sp_size=ulysses_sp_size,use_remove_padding=use_remove_padding,
                                  use_fused_kernels=use_fused_kernels,**kwargs)
        if ulysses_sp_size!=1 or use_remove_padding or use_fused_kernels:
            raise ValueError('Initial Qwen3.8 validation uses real HF padded SP1 unfused forward')
        if not hasattr(model,'_qwen38_expert_sources'):
            raise RuntimeError('The exact meta-only expert subdivision was not applied')

    def wrap(module,fsdp_kwargs,engine_config):
        if not is_qwen38(module.config):return original_wrap(module,fsdp_kwargs,engine_config)
        if (engine_config.strategy!='fsdp2' or engine_config.fsdp_size!=WORLD_SIZE
                or fsdp_kwargs.get('offload_policy') is not None):
            raise ValueError('Full CPU master offload and a partial-world mesh violate this resource profile')
        mp=fsdp_kwargs.get('mp_policy')
        if (mp is None or mp.param_dtype!=torch.bfloat16 or mp.reduce_dtype!=torch.float32
                or fsdp_kwargs.get('reshard_after_forward') is not True):
            raise ValueError('The fixed profile uses BF16 compute, FP32 gradient reduction and per-unit resharding')
        shard_expert_chunks_before_decoder_wrap(module,fsdp_kwargs)
        shard_vocab_tiles_before_root_wrap(module,fsdp_kwargs)
        result=original_wrap(module,fsdp_kwargs,engine_config)
        disable_prefetch_and_bound_reduction(module)
        return result

    def load(model,*args,**kwargs):
        if is_qwen38(model.config):return load_checkpoint_shards(model,*args,**kwargs)
        return original_load(model,*args,**kwargs)

    def build_module(self):
        model=original_build_module(self)
        if _is_target(self):
            restore_initialized_buffers_after_engine_cast(model)
        return model

    def context_switch(context,device):
        active=context.engine
        if (not context.disable_auto_offload and str(device)!='cpu' and _is_target(active)
                and hasattr(active,'_q38_phases') and active._q38_phases.phase=='rollout'):
            # Generic offload flags are deliberately false: a generic engine
            # must never move the whole policy or optimizer. The custom phase
            # still needs explicit restoration before HF eval/train. Engine.to
            # independently requires the real complete-world cleanup proof.
            active.to(device=device,model=True,optimizer=False,grad=False)
        return original_context_switch(context,device)

    def optimizer(self,module):
        if not _is_target(self):return original_optimizer(self,module)
        if not getattr(module,'_qwen38_streaming_verified',False):
            raise RuntimeError('Real source-coordinate and FSDP72 coverage validation must precede Adam allocation')
        if (self.scaler is not None or getattr(self,'_qat_enabled',False)
                or self.model_config.lora_rank>0 or self.engine_config.forward_only):
            raise ValueError('This is the full-policy BF16-storage/compute GRPO actor path')
        if self.engine_config.model_dtype!='bf16':raise ValueError('BF16 master storage must remain explicit')
        self._is_offload_param=False;self._is_offload_optimizer=False
        gradients=GradientStore(retained_bytes=int(os.environ.get('Q38_GPU_GRADIENT_CACHE_BYTES',80*1024**3)))
        params=list(module.parameters())
        for name,p in module.named_parameters():p._qwen36_parameter_name=name
        gradients.register_hooks(params)
        rank=dist.get_rank();local_rank=int(os.environ.get('LOCAL_RANK',rank%4))
        # Pure CPU BF16 states; no pathname, mapped file, or disk fallback.
        hot_node=int(os.environ.get('Q38_CPU_MOMENT_BYTES_PER_NODE',512*1024**3))
        hot_budget=hot_node//4+(local_rank<hot_node%4)
        moments=CpuBf16Moments(cpu_budget_bytes=hot_budget)
        plan=moments.plan(module.named_parameters())
        owned_numel=sum(local(p).numel() for p in params if p.requires_grad)
        largest_gradient=max((local(p).numel()*4 for p in params if p.requires_grad),default=0)
        master_budget=int(os.environ.get('Q38_GPU_REFIT_MASTER_CACHE_BYTES',24*1024**3))
        largest_offload=max((sum(local(p).numel()*local(p).element_size() for p in unit.parameters())
            for unit in module.modules() if isinstance(unit,(ExpertChunk,VocabTile))),default=0)
        local_plan={'rank':rank,'hostname':socket.gethostname(),**plan,
                    'local_policy_numel':owned_numel,'gpu_master_train_bytes':owned_numel*2,
                    'gpu_FP32_gradient_cache_bytes':gradients.limit,
                    'cpu_gradient_spill_upper_bytes':max(0,owned_numel*4-gradients.limit+largest_gradient),
                    'gpu_rollout_master_budget_bytes':master_budget,
                    'cpu_rollout_master_upper_bytes':max(0,owned_numel*2-max(0,master_budget-largest_offload)),
                    'largest_local_gradient_bytes':largest_gradient,
                    'largest_local_offload_unit_bytes':largest_offload,
                    'cpu_reserved_bytes_per_node':int(os.environ.get('Q38_CPU_RESERVE_BYTES_PER_NODE',96*1024**3))}
        all_plans=[None for _ in range(WORLD_SIZE)]
        dist.all_gather_object(all_plans,local_plan)
        peers=[row for row in all_plans if row['hostname']==local_plan['hostname']]
        if len(peers)!=4:raise RuntimeError('Actual node placement must contain exactly four training ranks')
        reserve=max(row['cpu_reserved_bytes_per_node'] for row in peers)
        train_cpu=sum(row['planned_cpu_moment_bytes']+row['cpu_gradient_spill_upper_bytes'] for row in peers)+reserve
        rollout_cpu=sum(row['planned_cpu_moment_bytes']+row['cpu_rollout_master_upper_bytes'] for row in peers)+reserve
        cpu_limit=int(os.environ.get('Q38_VERIFIED_CPU_LIMIT_BYTES_PER_NODE',800*1024**3))
        if max(train_cpu,rollout_cpu)>cpu_limit:
            raise MemoryError('The actual four-rank BF16 CPU state/FP32 gradient plan exceeds the verified RAM limit')
        plan_path=Path(os.environ['Q38_TRAINING_AUDIT_DIR'])/f'optimizer-placement-plan-rank{rank}.json'
        plan_path.parent.mkdir(parents=True,exist_ok=True)
        with plan_path.open('x') as stream:
            json.dump({'rank_plan':local_plan,'node_plans':peers,
                'node_train_CPU_bytes_with_reserve':train_cpu,'node_rollout_CPU_bytes_with_reserve':rollout_cpu,
                'node_verified_CPU_limit_bytes':cpu_limit,'disk_offload_forbidden':True},stream,indent=2)
        config=self.optimizer_config
        if config.optimizer!='Bf16CpuAdamW' or config.optimizer_impl!='qwen38_bf16_training_runtime':
            raise ValueError('The audited BF16 CPU-state optimizer must be explicitly selected in the run configuration')
        options={'lr':config.lr,'betas':tuple(config.betas),'eps':float(getattr(config,'eps',1e-8)),
                 'weight_decay':config.weight_decay,'chunk_bytes':8*1024**2,'rounding_seed':int(os.environ.get('Q38_BF16_ROUNDING_SEED',20261106))}
        overrides=dict(config.override_optimizer_config or {})
        if set(overrides)-{'lr','betas','eps','weight_decay','maximize'}:
            raise ValueError('Review unsupported optimizer overrides instead of silently dropping them')
        options.update(overrides)
        active=Bf16CpuAdamW(params,gradients=gradients,moments=moments,**options)
        self._q38_gradient_store=gradients
        self._q38_owned_numel=sum(local(p).numel() for p in params)
        if os.environ.get('Q38_CPU_FIRST_ADAM_ADMISSION_REQUIRED')=='1':
            # All ranks create this CPU-only group in the same constructor
            # sequence; it must be separately qualified on the current 72 GPUs.
            from datetime import timedelta
            self._q38_cpu_admission_group=dist.new_group(backend='gloo',timeout=timedelta(seconds=120))
            self._q38_planned_moment_bytes=plan['planned_cpu_moment_bytes']
            self._q38_first_adam_admitted=False
        self._q38_phases=MasterPhaseManager(module,gradients,active,
            leaf_predicate=lambda item:isinstance(item,(ExpertChunk,VocabTile)),
            retained_bytes=int(os.environ.get('Q38_GPU_REFIT_MASTER_CACHE_BYTES',24*1024**3)))
        self._q38_native_contexts_released=False;self._q38_exporter=None
        module.register_forward_pre_hook(lambda _module,_inputs:self._q38_phases.require_training())
        return active

    def step(self):
        if not _is_target(self):return original_step(self)
        self._q38_phases.require_training()
        if self.scaler is not None:raise RuntimeError('A loss scaler changes the reviewed BF16 gradient path')
        norm=self._q38_gradient_store.clip_grad_norm(self.optimizer_config.clip_grad,
            expected_local_numel=self._q38_owned_numel,expected_global_numel=POLICY_NUMEL,
            collective_device=torch.device('cuda',torch.cuda.current_device()))
        if (os.environ.get('Q38_CPU_FIRST_ADAM_ADMISSION_REQUIRED')=='1'
                and not self._q38_first_adam_admitted):
            from distributed_admission import first_adam_admission
            first_adam_admission(self,dist,self._q38_cpu_admission_group,world=WORLD_SIZE)
            self._q38_first_adam_admitted=True
        def reduce_changed(count):
            if not dist.is_initialized() or dist.get_world_size()!=WORLD_SIZE:
                raise RuntimeError('The real-update changed-weight count must cover all72 actor ranks')
            changed=torch.tensor(count,device=torch.device('cuda',torch.cuda.current_device()),dtype=torch.int64)
            dist.all_reduce(changed,op=dist.ReduceOp.SUM)
            return changed.item()
        _optimizer_step_after_global_clip(self,norm,reduce_changed)
        _audit(self,'optimizer-step')
        # This is the committed optimizer boundary; the fixed initial profile
        # has one global minibatch per update. Retain the audit, then release
        # every CPU/GPU reduced gradient before native inference can start.
        self.optimizer.zero_grad()
        return norm

    def to(self,device,model=True,optimizer=True,grad=True):
        if not _is_target(self) or not hasattr(self,'_q38_phases'):
            return original_to(self,device,model=model,optimizer=optimizer,grad=grad)
        if str(device)=='cpu':
            if model and self._q38_phases.phase=='train':
                if self._q38_gradient_store.slots:
                    raise RuntimeError('Explicit optimizer_zero_grad is required before the native rollout phase')
                audit_policy_before_rollout(self)
                self._q38_phases.prepare_rollout();self._q38_native_contexts_released=False
                _audit(self,'partial-master-offload')
        elif torch.device(device).type=='cuda':
            if self._q38_phases.phase=='rollout':
                self._q38_phases.restore_training(native_contexts_released=self._q38_native_contexts_released,
                    refit_sources_released=self._q38_exporter is None)
                _audit(self,'partial-master-restore')
        else:raise ValueError('Only the exact allocated CUDA device or CPU phase is supported')
        # CPU BF16 moments remain on CPU across generic engine.to.

    def export(self,*args,**kwargs):
        if not _is_target(self):return original_export(self,*args,**kwargs)
        if self._q38_phases.phase=='train':
            audit_policy_before_rollout(self)
            self._q38_phases.prepare_rollout();self._q38_native_contexts_released=False
        if self._q38_exporter is not None:
            raise RuntimeError('Close the preceding native source session before exporting another policy version')
        self._q38_exporter=CanonicalExport(self.module)
        # The world adapter retains only this bound method across release; it
        # carries no Tensor/source view and therefore cannot pin the backing.
        self._q38_exporter.confirm_native_cleanup=self.q38_confirm_native_cleanup
        return self._q38_exporter,None

    def confirm_cleanup(self,proof):
        if not _is_target(self):raise ValueError('Qwen3.8 cleanup proof applies only to its dedicated actor')
        if (not isinstance(proof,dict) or proof.get('world_size')!=WORLD_SIZE
                or proof.get('all_owned_native_processes_exited') is not True
                or proof.get('remaining_owned_gpu_contexts')!=[]
                or len(proof.get('checked_native_ranks',[]))!=WORLD_SIZE
                or set(proof.get('checked_native_ranks',[]))!=set(range(WORLD_SIZE))):
            raise RuntimeError('Actual all-rank native process and GPU-context teardown evidence is required')
        if self._q38_exporter is not None:
            self._q38_last_refit_source_release=self._q38_exporter.close(all_owned_native_contexts_released=True)
            self._q38_exporter=None
        # Cleanup-only evidence frees borrowed source memory but must not grant
        # training admission after a failed refit or missing all-world pause.
        self._q38_native_contexts_released=(proof.get('training_admission_safe') is True
            and proof.get('all_world_pause_acknowledged') is True)
        self._q38_last_native_cleanup_proof=proof
        _audit(self,'native-cleanup-confirmed')

    model_utils.get_hf_auto_model_class=factory
    monkey_patch.apply_monkey_patch=patch;engine_module.apply_monkey_patch=patch
    engine_module.apply_fsdp2=wrap;engine_module.fsdp2_load_full_state_dict=load
    if fsdp_utils.fsdp2_load_full_state_dict is original_load:fsdp_utils.fsdp2_load_full_state_dict=load
    Engine._build_module=build_module
    BaseEngineCtx._context_switch=context_switch
    Engine._build_optimizer=optimizer;Engine.optimizer_step=step;Engine.to=to
    Engine.get_per_tensor_param=export;Engine.q38_confirm_native_cleanup=confirm_cleanup
    # Actual verl's naive worker calls resume(weights) BEFORE it asks the
    # model engine for exported parameters. Offload masters before that resume
    # can allocate the native policy; otherwise 125GiB+native cannot fit.
    import verl.workers.engine_workers as workers
    original_worker_update=workers.ActorRolloutRefWorker.update_weights
    @functools.wraps(original_worker_update)
    async def update_before_native_resume(self,global_steps=None,mode='auto'):
        actor=getattr(self,'actor',None);active=getattr(actor,'engine',None)
        if active is not None and _is_target(active):
            effective=mode if mode!='auto' else self.config.rollout.checkpoint_engine.backend
            if effective!='naive':
                raise ValueError('The single-instance leased source path requires direct colocated naive transfer')
            if active._q38_phases.phase=='train':
                audit_policy_before_rollout(active)
                active._q38_phases.prepare_rollout();active._q38_native_contexts_released=False
                _audit(active,'partial-master-offload-before-native-resume')
        return await original_worker_update(self,global_steps=global_steps,mode=mode)
    update_before_native_resume._q38_offloads_before_native_resume=True
    workers.ActorRolloutRefWorker.update_weights=update_before_native_resume

    # The direct naive worker already passes the object unchanged. Preserve
    # the same contract for callers of ColocatedCheckpointEngine.receive_weights
    # whose original `yield from` would erase ACK/cleanup methods.
    from verl.checkpoint_engine.base import ColocatedCheckpointEngine
    original_receive=ColocatedCheckpointEngine.receive_weights
    @functools.wraps(original_receive)
    def receive(self,global_steps=None):
        weights=getattr(self,'weights',None)
        if isinstance(weights,CanonicalExport):
            self.weights=None
            return weights
        return original_receive(self,global_steps=global_steps)
    receive._q38_preserves_export_lease=True
    ColocatedCheckpointEngine.receive_weights=receive
    from qwen38_policy_audit import install_policy_loss_audit
    install_policy_loss_audit()
    from qwen38_memory_boundaries import install as install_memory_boundaries
    install_memory_boundaries()
    _INSTALLED=True


if os.environ.get('Q38_ENABLE_BF16_STATE_RUNTIME')=='1':install()
