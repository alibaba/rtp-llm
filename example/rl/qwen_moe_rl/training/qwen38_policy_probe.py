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
def teacher_forced_target_scores(module, prompt_ids, target_ids, device):
    if not prompt_ids or not target_ids or len(prompt_ids)+len(target_ids)>128:
        raise ValueError('The short fixed HF numerical probe must contain prompt and target')
    ids=torch.tensor([prompt_ids+target_ids],device=device,dtype=torch.int64)
    previous=module.training
    try:
        module.eval()
        result=module(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
        logits=result.logits[0,len(prompt_ids)-1:len(prompt_ids)+len(target_ids)-1].float()
        probabilities=logits.log_softmax(dim=-1)
        target=torch.tensor(target_ids,device=device,dtype=torch.int64)
        scores=probabilities.gather(-1,target[:,None]).squeeze(-1).cpu().tolist()
        if len(scores)!=len(target_ids) or not all(math.isfinite(v) for v in scores):
            raise FloatingPointError('Actual HF probe produced nonfinite or incomplete target scores')
        return scores
    finally:module.train(previous)


def audit_policy_before_rollout(engine):
    if engine._q38_phases.phase!='train':
        raise RuntimeError('HF policy probe requires resident train-phase BF16 master shards')
    steps=engine.optimizer.audit()['steps']
    if len(steps)>1:raise RuntimeError('Persistent Adam states disagree about the policy version')
    step=steps[0] if steps else 0
    if getattr(engine,'_q38_hf_probe_step',None)==step:return
    if engine._q38_gradient_store.slots or any(p.grad is not None for p in engine.module.parameters()):
        raise RuntimeError('Clear committed gradients before the read-only policy probe')
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
