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
    maximum=max(abs(a-b) for row in probes for a,b in zip(hf,row['raw_log_probs']))
    if maximum>.05 or proof.get('max_HF_error')!=maximum:
        raise RuntimeError('Final policy-2 native/HF scores differ beyond the actual 0.05 gate')
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


def install_backend_hooks():
    global _BACKEND_PATCHED
    if _BACKEND_PATCHED:return
    from rllm.trainer.verl.verl_backend import VerlBackend
    from rllm.trainer.unified_trainer import UnifiedTrainer
    old_batch_end=VerlBackend.on_batch_end
    old_train_end=VerlBackend.on_train_end
    old_fit=UnifiedTrainer.fit_async

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
    UnifiedTrainer.fit_async=checked_fit
    _BACKEND_PATCHED=True


def _forward_worker_setup():
    from rllm.trainer.verl import ray_runtime_env
    previous=ray_runtime_env.get_ppo_ray_runtime_env
    if getattr(previous,'_q38_final_policy_setup',False):return
    @functools.wraps(previous)
    def runtime_env():
        result=previous()
        if result.get('worker_process_setup_hook') not in (None,'install_rtp_bridge.worker_setup',
                                                         'qwen38_final_policy.worker_setup'):
            raise RuntimeError('An unreviewed worker hook cannot be overwritten')
        result['worker_process_setup_hook']='qwen38_final_policy.worker_setup'
        return result
    runtime_env._q38_final_policy_setup=True
    ray_runtime_env.get_ppo_ray_runtime_env=runtime_env


def worker_setup():
    """Passive hook: no Torch, Ray or CUDA initialization before GPU masking."""
    # A Raylet may have inherited the deprecated alias even when the current
    # driver has unset it. Remove it before either bridge or Torch imports.
    os.environ.pop('PYTORCH_CUDA_ALLOC_CONF',None)
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
