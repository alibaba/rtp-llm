"""Finalize the real policy-2 native probe before rLLM shuts down its backend.

The public worker setup function imports only standard-library/passive bridge
code. Torch and Ray import naturally after Ray has assigned device masks.
"""
from __future__ import annotations
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

_BACKEND_PATCHED=False


def validate_final_policy_proof(proof, *, expected_step=2):
    if not isinstance(proof,dict) or proof.get('final_native_GPU_reprobe') is not True:
        raise RuntimeError('A fresh final native GPU re-probe RPC is required')
    if (proof.get('global_step')!=expected_step or proof.get('logical_model_replicas')!=1
            or (proof.get('world_size'),proof.get('tp_size'),proof.get('dp_size'),proof.get('ep_size'))!=(72,8,9,72)
            or not proof.get('model_version') or not proof.get('native_incarnation')):
        raise RuntimeError('Final probe must serve the precise single-world policy-2 version')
    statuses=proof.get('statuses',[])
    if (len(statuses)!=72 or {r.get('world_rank') for r in statuses}!=set(range(72))
            or any(r.get('global_step')!=expected_step or r.get('model_version')!=proof['model_version']
                   or r.get('phase')!='running' or r.get('received_weights')!=1590
                   or r.get('applied_weights',0)<=0 or r.get('update_id') for r in statuses)):
        raise RuntimeError('Every real native rank must ACK and serve all 1590 final sources')
    digest=proof.get('full_source_sha256','')
    if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise RuntimeError('Final probe lacks the actual complete canonical source digest')
    target=proof.get('response_ids',[]);hf=proof.get('hf_raw_log_probs',[]);probes=proof.get('probes',[])
    if (not proof.get('prompt_ids') or not target or len(hf)!=len(target)
            or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in hf)
            or len(probes)!=9 or {r.get('dp_rank') for r in probes}!=set(range(9))):
        raise RuntimeError('Final probe must include the identical actual HF and all nine DP target scores')
    for row in probes:
        values=row.get('raw_log_probs',[])
        if (row.get('generation_world_rank')!=row['dp_rank']*8 or len(values)!=len(hf)
                or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values)
                or row.get('raw_fp32_bits_hex')!=struct.pack(f'<{len(values)}f',*values).hex()):
            raise RuntimeError('Final DP probe must preserve raw FP32 target-logprob bits')
    differences=[[abs(a-b) for a,b in zip(hf,row['raw_log_probs'])] for row in probes]
    maximum=max(value for row in differences for value in row)
    mean=sum(value for row in differences for value in row)/(len(probes)*len(hf))
    if proof.get('max_HF_error')!=maximum:
        raise RuntimeError('Final native/HF maximum error report differs from the actual raw scores')
    dp_error=max(abs(a-b) for row in probes for a,b in zip(probes[0]['raw_log_probs'],row['raw_log_probs']))
    if dp_error>.05 or ('max_dp_error' in proof and proof['max_dp_error']!=dp_error):
        raise RuntimeError('Final native attention-DP copies disagree beyond the unchanged 0.05 gate')
    report_only=os.environ.get('Q38_HF_RTP_REPORT_ONLY')=='1'
    if report_only:
        if (proof.get('HF_comparison_mode')!='report_only' or proof.get('HF_error_used_for_acceptance') is not False
                or proof.get('hf_raw_fp32_bits_hex')!=struct.pack(f'<{len(hf)}f',*hf).hex()
                or not isinstance(proof.get('mean_HF_absolute_error'),(int,float))
                or not math.isclose(proof['mean_HF_absolute_error'],mean,rel_tol=1e-12,abs_tol=1e-12)):
            raise RuntimeError('The explicit HF report-only contract lacks the actual raw-bit/mean-error report')
        for row,errors in zip(probes,differences):
            if (row.get('absolute_HF_errors')!=errors
                    or not isinstance(row.get('mean_HF_absolute_error'),(int,float))
                    or not math.isclose(row['mean_HF_absolute_error'],sum(errors)/len(errors),rel_tol=1e-12,abs_tol=1e-12)):
                raise RuntimeError('A final DP per-token HF error report differs from the actual raw scores')
    else:
        if proof.get('HF_comparison_mode')=='report_only' or proof.get('HF_error_used_for_acceptance') is False:
            raise RuntimeError('The final probe comparison mode differs from the explicit task profile')
        if maximum>.05:
            raise RuntimeError('Final policy-2 native/HF scores differ beyond the default strict 0.05 gate')
    return proof


def _target(backend):
    return (os.environ.get('Q38_ENABLE_BF16_STATE_RUNTIME')=='1'
        and os.environ.get('RTP_LLM_RL_WORLD72')=='1'
        and backend.config.actor_rollout_ref.rollout.name=='rtp')


def require_final_policy_completion(backend):
    record=getattr(backend,'_q38_final_policy_proof',None)
    if not record or record.get('actual_batch_sync_steps')!=[1,2]:
        raise RuntimeError('Training ended without real final policy-2 native verification')
    return validate_final_policy_proof(record['proof'])


def validate_single_global_minibatch(backend, batch, trainer_state):
    """Check the original backend's dynamic ROWS metadata before GPU dispatch.

    The untouched backend normalizes by 72 rollouts while expanding the one
    minibatch to contain all real multi-turn rows and neutral DP padding.
    """
    actor=backend.config.actor_rollout_ref.actor
    metadata=batch.meta_info
    rows=len(batch)
    mini=int(metadata.get('rllm_actor_mini_batch_size',0))
    denominator=int(metadata.get('rllm_actor_global_batch_size',0))
    updates=int(metadata.get('rllm_actor_num_updates',0))
    real_rows=int(metadata.get('rllm_actor_rows_pre_pad',0))
    dp=int(backend._get_dp_size(backend.actor_rollout_wg,'actor'))
    if (not backend.config.trainer.balance_batch or int(actor.ppo_epochs)!=1
            or int(backend.config.data.train_batch_size)!=18
            or int(actor.ppo_mini_batch_size)!=18
            or int(backend.config.actor_rollout_ref.rollout.n)!=4
            or dp!=72 or rows<=0 or rows%dp or mini!=rows
            or updates!=1 or denominator!=72 or not (0<real_rows<=rows)):
        raise RuntimeError('Q38 requires exactly one original dynamically padded global minibatch per iteration')
    loss_map=backend.algorithm_config.loss_fn_map if backend.algorithm_config is not None else {}
    roles=batch.non_tensor_batch.get('group_roles')
    actual_losses={loss_map.get(str(role),'vanilla') for role in roles.tolist()} if roles is not None else set()
    if len(actual_losses)>1:
        raise RuntimeError('Q38 single-update acceptance refuses multiple sequential per-role loss updates')
    record=dict(schema='q38-real-single-global-minibatch-before-dispatch-v1',
        trainer_global_step=int(trainer_state.global_step), rows_pre_pad=real_rows, rows_post_pad=rows,
        global_minibatch_rows=mini, rollout_loss_denominator=denominator, data_parallel_size=dp,
        rows_per_rank=rows//dp, ppo_epochs=1, predicted_optimizer_steps_this_iteration=1,
        actual_distinct_role_losses=sorted(actual_losses), original_backend_metadata_preserved=True,
        optimizer_step_executed_by_this_observer=False, time_ns=time.time_ns(),pid=os.getpid())
    root=Path(os.environ['Q38_RUN_ROOT']);root.mkdir(parents=True,exist_ok=True)
    path=root/f'global-minibatch-step{record["trainer_global_step"]}-pid{os.getpid()}-{record["time_ns"]}.json'
    with path.open('x') as stream:json.dump(record,stream);stream.write('\n')
    return record


def install_backend_hooks():
    global _BACKEND_PATCHED
    if _BACKEND_PATCHED:return
    from rllm.trainer.verl.verl_backend import VerlBackend
    from rllm.trainer.unified_trainer import UnifiedTrainer
    from qwen38_length_guards import install as install_length_guards
    install_length_guards()
    old_batch_end=VerlBackend.on_batch_end
    old_train_end=VerlBackend.on_train_end
    old_update=VerlBackend._update_actor_with_loss_routing
    old_fit=UnifiedTrainer.fit_async

    @functools.wraps(old_update)
    def checked_update(self,batch,trainer_state):
        if _target(self) and os.environ.get('RLLM_REQUIRE_REAL_UPDATE')=='1':
            validate_single_global_minibatch(self,batch,trainer_state)
        return old_update(self,batch,trainer_state)

    @functools.wraps(old_batch_end)
    async def batch_end(self,state):
        # The real restored rLLM already runs checkpoint_manager.update_weights
        # here, including the final batch. Do not replay a second 4.8TB refit.
        result=await old_batch_end(self,state)
        if _target(self):
            completed=getattr(self,'_q38_completed_native_sync_steps',[])
            completed.append(int(state.global_step));self._q38_completed_native_sync_steps=completed
        return result

    @functools.wraps(old_train_end)
    async def train_end(self,state):
        if _target(self):
            if int(state.global_step)!=2 or getattr(self,'_q38_completed_native_sync_steps',[])!=[1,2]:
                raise RuntimeError('Exactly two actual batch updates and completed native syncs must precede final verification')
            replicas=self.llm_server_manager.get_replicas()
            if (len(replicas)!=1 or replicas[0].replica_rank!=0 or replicas[0].world_size!=72
                    or len(replicas[0].workers)!=72):
                raise RuntimeError('The final policy must belong to the single actual 72-worker replica')
            # This RPC executes native target scoring on all nine attention-DP
            # groups. It binds the already committed 1590-source update-2
            # transaction and actual HF2 reference, before backend shutdown.
            proof=await replicas[0]._server_handle.verify_final_policy.remote(2)
            validate_final_policy_proof(proof)
            record=dict(schema='q38-final-policy-2-real-native-verification-v1',
                proof=proof,actual_batch_sync_steps=list(self._q38_completed_native_sync_steps),
                third_optimizer_update_performed=False,time_ns=time.time_ns(),pid=os.getpid())
            self._q38_final_policy_proof=record
            directory=Path(os.environ['Q38_RUN_ROOT']);directory.mkdir(parents=True,exist_ok=True)
            path=directory/f'final-policy2-verified-trainer-{os.getpid()}.json'
            with path.open('x') as stream:json.dump(record,stream);stream.write('\n')
        return await old_train_end(self,state)

    @functools.wraps(old_fit)
    async def checked_fit(self,*args,**kwargs):
        result=await old_fit(self,*args,**kwargs)
        if _target(self.backend):
            # Original rLLM logs and swallows on_train_end exceptions in its
            # finally block. Make a missing/failed actual RPC fail the run.
            require_final_policy_completion(self.backend)
        return result

    batch_end._q38_final_policy_completion=True
    train_end._q38_final_policy_completion=True
    checked_fit._q38_final_policy_completion=True
    VerlBackend.on_batch_end=batch_end;VerlBackend.on_train_end=train_end
    VerlBackend._update_actor_with_loss_routing=checked_update
    UnifiedTrainer.fit_async=checked_fit
    _BACKEND_PATCHED=True
    from qwen38_initial_policy_replay import install as install_initial_policy_replay
    install_initial_policy_replay()


def _forward_worker_setup():
    from rllm.trainer.verl import ray_runtime_env
    previous=ray_runtime_env.get_ppo_ray_runtime_env
    if getattr(previous,'_q38_final_policy_setup',False):return
    @functools.wraps(previous)
    def runtime_env():
        result=previous()
        # rLLM forwards CUDA_* by default. A CPU coordinator legitimately has
        # CUDA_VISIBLE_DEVICES='', which must never become a job/actor mask.
        # Keep the real process mask unchanged; Ray owns worker assignment.
        result.setdefault('env_vars',{}).pop('CUDA_VISIBLE_DEVICES',None)
        result['env_vars'].pop('RLLM_EXCLUDE',None)
        if result.get('worker_process_setup_hook') not in (None,'install_rtp_bridge.worker_setup',
                                                         'qwen38_final_policy.worker_setup'):
            raise RuntimeError('An unreviewed worker hook cannot be overwritten')
        result['worker_process_setup_hook']='qwen38_final_policy.worker_setup'
        return result
    runtime_env._q38_final_policy_setup=True
    ray_runtime_env.get_ppo_ray_runtime_env=runtime_env


def worker_setup():
    """Passive hook: no Torch, Ray or CUDA initialization before GPU masking."""
    from qwen38_python310_compat import install_python310_tomllib
    install_python310_tomllib()
    # A Raylet may have inherited the deprecated alias even when the current
    # driver has unset it. Remove it before either bridge or Torch imports.
    os.environ.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    # This documented control is process-local and is deliberately not
    # forwarded by rLLM. Reapply it in every passive setup hook so original
    # function aliases used inside the CPU TaskRunner also exclude the mask.
    excluded=[name.strip() for name in os.environ.get('RLLM_EXCLUDE','').split(',') if name.strip()]
    if 'CUDA_VISIBLE_DEVICES' not in excluded:excluded.append('CUDA_VISIBLE_DEVICES')
    os.environ['RLLM_EXCLUDE']=','.join(excluded)
    if (os.environ.get('RTP_LLM_RL_WORLD72')=='1'
            and os.environ.get('PYTORCH_ALLOC_CONF')!='backend:native,expandable_segments:False'):
        raise RuntimeError('World72 CUDA IPC requires explicit PYTORCH_ALLOC_CONF=backend:native,expandable_segments:False')
    import install_rtp_bridge as bridge
    if not getattr(bridge._install_driver,'_q38_final_policy_driver',False):
        original=bridge._install_driver
        @functools.wraps(original)
        def complete_driver():
            original()
            install_backend_hooks()
            _forward_worker_setup()
        complete_driver._q38_final_policy_driver=True
        bridge._install_driver=complete_driver
    bridge.worker_setup()


def install():
    worker_setup()
    import install_rtp_bridge as bridge
    bridge.install()
