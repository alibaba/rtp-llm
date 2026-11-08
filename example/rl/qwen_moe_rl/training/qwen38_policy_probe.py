"""Read-only teacher-forced HF scores before the same policy is sent to RTP."""
import json
import math
import os
from pathlib import Path
import struct
import time

import torch
import torch.distributed as dist


@torch.no_grad()
def teacher_forced_target_scores(module, prompt_ids, target_ids, device, *, allow_invalid=False):
    if not prompt_ids or not target_ids or len(prompt_ids)+len(target_ids)>128:
        raise ValueError('The short fixed HF numerical probe must contain prompt and target')
    ids=torch.tensor([prompt_ids+target_ids],device=device,dtype=torch.int64)
    previous=module.training
    try:
        module.eval()
        result=module(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
        # Only content/shape errors after a successful model forward are
        # optional. In particular, CUDA RuntimeError and forward/collective
        # failures are never caught by this block.
        try:
            logits=result.logits[0,len(prompt_ids)-1:len(prompt_ids)+len(target_ids)-1].float()
            probabilities=logits.log_softmax(dim=-1)
            target=torch.tensor(target_ids,device=device,dtype=torch.int64)
            scores=probabilities.gather(-1,target[:,None]).squeeze(-1).cpu().tolist()
        except (AttributeError,TypeError,IndexError,ValueError):
            if allow_invalid:return None
            raise
        if not allow_invalid and (len(scores)!=len(target_ids) or not all(math.isfinite(v) for v in scores)):
            raise FloatingPointError('Actual HF probe produced nonfinite or incomplete target scores')
        return scores
    finally:module.train(previous)


def audit_policy_before_rollout(engine):
    if engine._q38_phases.phase!='train':
        raise RuntimeError('HF policy probe requires resident train-phase BF16 master shards')
    steps=engine.optimizer.audit()['steps']
    if len(steps)>1:raise RuntimeError('Persistent Adam states disagree about the policy version')
    step=steps[0] if steps else 0
    if engine._q38_gradient_store.slots or any(p.grad is not None for p in engine.module.parameters()):
        raise RuntimeError('Clear committed gradients before the read-only policy probe')
    mode=os.environ.get('Q38_OPTIONAL_NUMERICAL_PROBES','1')
    cached=getattr(engine,'_q38_hf_probe_step',None)==step
    modes=[None for _ in range(dist.get_world_size())]
    # All participants take the same branch before any teacher-forced FSDP
    # forward. Collective failures remain fatal; nothing is swallowed here.
    dist.all_gather_object(modes,dict(rank=dist.get_rank(),step=step,mode=mode,cached=cached))
    if ({row.get('rank') for row in modes}!=set(range(len(modes))) or
            any(row.get('step')!=step or row.get('mode')!=mode or row.get('cached') is not cached for row in modes)
            or mode not in ('0','1','2')):
        raise RuntimeError('Optional probe mode/version/cache differs across the actual training world')
    if cached:return
    if mode=='0':
        # This diagnostic is not executed. RL generation logprobs, update
        # math, finite gradients and committed policy refits remain mandatory.
        engine._q38_hf_probe_step=step
        return
    if mode=='2':
        return _audit_optional_hf_scores(engine,step,steps)
    if not hasattr(engine,'_q38_hf_probe_tokenizer'):
        from transformers import AutoTokenizer
        engine._q38_hf_probe_tokenizer=AutoTokenizer.from_pretrained(
            '/mnt/fuse/Qwen/Qwen3.8-2.4T-A95B',local_files_only=True,trust_remote_code=False)
    tokenizer=engine._q38_hf_probe_tokenizer
    prompt=tokenizer.encode('The value of 37 * 43 is',add_special_tokens=False)
    target=tokenizer.encode(' 1591.',add_special_tokens=False)
    scores=teacher_forced_target_scores(engine.module,prompt,target,
                                       torch.device('cuda',torch.cuda.current_device()))
    observed=[None for _ in range(dist.get_world_size())]
    dist.all_gather_object(observed,dict(rank=dist.get_rank(),step=step,prompt_ids=prompt,
        response_ids=target,raw_log_probs=scores))
    if any(r['step']!=step or r['prompt_ids']!=prompt or r['response_ids']!=target for r in observed):
        raise RuntimeError('Actual all-rank HF probes used different policy versions or token IDs')
    if any(len(r['raw_log_probs'])!=len(target) or not all(math.isfinite(x) for x in r['raw_log_probs']) for r in observed):
        raise FloatingPointError('Actual all-rank HF probes contain nonfinite or incomplete scores')
    maximum=max(abs(a-b) for row in observed for a,b in zip(scores,row['raw_log_probs']))
    report_only=os.environ.get('Q38_LOGPROB_REPORT_ONLY')=='1'
    if not report_only and maximum>.05:raise RuntimeError('Actual all-rank HF probe differs beyond the numerical gate')
    record=dict(schema='q38-real-HF-policy-probe-v1',rank=dist.get_rank(),world_size=dist.get_world_size(),
        global_step=step,prompt_ids=prompt,response_ids=target,raw_log_probs=scores,
        raw_fp32_bits_hex=struct.pack(f'<{len(scores)}f',*scores).hex(),
        max_HF_rank_error=maximum,master_dtype='bfloat16',compute_dtype='bfloat16',
        score_dtype='float32',all_rank_HF_probe_agreement=maximum<=.05,
        HF_rank_comparison_mode='report_only' if report_only else 'absolute_error_gate',
        HF_rank_error_used_for_acceptance=not report_only,
        optimizer_steps=steps,time_ns=time.time_ns(),native_RTP_numerical_match_proven_here=False)
    directory=Path(os.environ['Q38_TRAINING_AUDIT_DIR']);directory.mkdir(parents=True,exist_ok=True)
    (directory/f'hf-probe-step{step}-rank{dist.get_rank()}.json').write_text(json.dumps(record)+'\n')
    engine._q38_hf_probe_step=step


def _audit_optional_hf_scores(engine,step,steps):
    """Try diagnostic scores only after all ranks agree inputs are usable.

    Only tokenizer/input/report problems are optional. FSDP forward, CUDA and
    distributed communication exceptions are deliberately never caught.
    """
    rank=dist.get_rank();world=dist.get_world_size();local_error=None;prompt=target=None
    try:
        if not hasattr(engine,'_q38_hf_probe_tokenizer'):
            from transformers import AutoTokenizer
            engine._q38_hf_probe_tokenizer=AutoTokenizer.from_pretrained(
                '/mnt/fuse/Qwen/Qwen3.8-2.4T-A95B',local_files_only=True,trust_remote_code=False)
        tokenizer=engine._q38_hf_probe_tokenizer
        prompt=tokenizer.encode('The value of 37 * 43 is',add_special_tokens=False)
        target=tokenizer.encode(' 1591.',add_special_tokens=False)
        if (not isinstance(prompt,list) or not isinstance(target,list) or not prompt or not target
                or len(prompt)+len(target)>128 or any(type(x) is not int for x in prompt+target)):
            raise ValueError('Diagnostic tokenizer output is invalid')
    except (OSError,ValueError,TypeError,ImportError) as exc:local_error=type(exc).__name__
    ready=[None]*world
    dist.all_gather_object(ready,dict(rank=rank,step=step,error=local_error,prompt=prompt,target=target))
    if {row.get('rank') for row in ready}!=set(range(world)) or any(row.get('step')!=step for row in ready):
        raise RuntimeError('Diagnostic readiness lost real rank or policy identity')
    if (any(row.get('error') is not None for row in ready)
            or any((row.get('prompt'),row.get('target'))!=(prompt,target) for row in ready)):
        engine._q38_hf_probe_step=step
        return
    # Do not catch actual model execution or collective failures here.
    scores=teacher_forced_target_scores(engine.module,prompt,target,
        torch.device('cuda',torch.cuda.current_device()),allow_invalid=True)
    observed=[None]*world
    dist.all_gather_object(observed,dict(rank=rank,step=step,prompt_ids=prompt,response_ids=target,raw_log_probs=scores))
    if {row.get('rank') for row in observed}!=set(range(world)) or any(row.get('step')!=step for row in observed):
        raise RuntimeError('Diagnostic scores lost real rank or policy identity')
    def valid(row):
        value=row.get('raw_log_probs')
        return (row.get('prompt_ids')==prompt and row.get('response_ids')==target
            and isinstance(value,list) and len(value)==len(target)
            and all(type(x) in (int,float) and math.isfinite(x) for x in value))
    all_valid=all(valid(row) for row in observed);own_valid=valid(observed[rank])
    maximum=max(abs(a-b) for row in observed for a,b in zip(scores,row['raw_log_probs'])) if all_valid else None
    record=dict(schema='q38-real-HF-policy-probe-v1',rank=rank,world_size=world,global_step=step,
        prompt_ids=prompt,response_ids=target,raw_log_probs=scores if own_valid else None,
        raw_fp32_bits_hex=struct.pack(f'<{len(scores)}f',*scores).hex() if own_valid else None,
        max_HF_rank_error=maximum,master_dtype='bfloat16',compute_dtype='bfloat16',score_dtype='float32',
        all_rank_HF_probe_agreement=(maximum<=.05) if maximum is not None else None,
        HF_rank_comparison_mode='report_only',HF_rank_error_used_for_acceptance=False,
        score_available=own_valid,all_rank_scores_available=all_valid,optimizer_steps=steps,time_ns=time.time_ns(),
        native_RTP_numerical_match_proven_here=False)
    try:
        directory=Path(os.environ['Q38_TRAINING_AUDIT_DIR']);directory.mkdir(parents=True,exist_ok=True)
        (directory/f'hf-probe-step{step}-rank{rank}.json').write_text(json.dumps(record,allow_nan=False)+'\n')
    except OSError:pass  # A missing diagnostic file cannot invalidate training.
    engine._q38_hf_probe_step=step
