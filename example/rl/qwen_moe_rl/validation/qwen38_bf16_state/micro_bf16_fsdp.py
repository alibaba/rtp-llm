"""Actual four-GPU FP32 reduction before BF16 .grad cast, and BF16 Adam."""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import socket

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch import nn
from torch.nn import functional as F

from qwen38_bf16_optimizer import (Bf16CpuAdamW, CpuBf16Moments, GradientStore, local,
                                 counter_random16, rounding_key, stochastic_bf16)
from qwen38_bf16_master_phases import MasterPhaseManager, disable_prefetch_and_bound_reduction
from qwen38_canonical_export import CanonicalExport
from test_bf16_numerics import numpy_adam


def collective_check(boundary, error):
    """Reconcile local numerical checks before the next tensor collective."""
    rows=[None for _ in range(dist.get_world_size())]
    dist.all_gather_object(rows,dict(rank=dist.get_rank(),boundary=boundary,error=error))
    failures=[r for r in rows if r['error'] is not None or r['boundary']!=boundary]
    if failures:raise RuntimeError(f'All-rank micro qualification failed: {failures}')


def replay_gpu_arithmetic(prior, grad, *, step, device):
    """Diagnostic only: the exact existing GPU FP32 operations before SR."""
    weight,mean,variance=[torch.tensor(v,device=device,dtype=torch.float32) for v in prior]
    gradient=torch.tensor(grad,device=device,dtype=torch.float32)
    weight.mul_(1.0)
    mean.lerp_(gradient,1-.9)
    variance.mul_(.999).addcmul_(gradient,gradient,value=1-.999)
    denominator=variance.sqrt().div_(math_sqrt(1-.999**step)).add_(1e-8)
    weight.addcdiv_(mean,denominator,value=-1e-4/(1-.9**step))
    return weight,mean,variance


class Model(nn.Module):
    def __init__(self, dimension=16):
        super().__init__()
        self.first=nn.Linear(dimension,dimension*2,bias=False,dtype=torch.bfloat16)
        self.second=nn.Linear(dimension*2,dimension,bias=False,dtype=torch.bfloat16)
    def forward(self,x):return self.second(F.silu(self.first(x)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--job-id',required=True);args=parser.parse_args()
    rank=int(os.environ['RANK']);lr=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(lr);device=torch.device('cuda',lr)
    dist.init_process_group('nccl',timeout=timedelta(seconds=90),device_id=device)
    world=dist.get_world_size()
    if world not in (4,72):raise ValueError('Only owner-authorized four- or 72-GPU probe')
    dimension=72 if world==72 else 16
    torch.manual_seed(755)
    model=Model(dimension).to(device);reference=copy.deepcopy(model)
    mesh=init_device_mesh('cuda',(world,));policy=MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,reduce_dtype=torch.float32,cast_forward_inputs=True)
    for layer in (model.first,model.second):fully_shard(layer,mesh=mesh,mp_policy=policy,reshard_after_forward=True)
    fully_shard(model,mesh=mesh,mp_policy=policy,reshard_after_forward=True)
    disable_prefetch_and_bound_reduction(model)
    params=list(model.parameters());named=list(model.named_parameters())
    for name,p in named:p._qwen36_parameter_name=name
    gradients=GradientStore(retained_bytes=local(params[0]).numel()*4)
    gradients.register_hooks(params)
    moments=CpuBf16Moments(cpu_budget_bytes=sum(local(p).numel()*4 for p in params))
    optimizer=Bf16CpuAdamW(params,gradients=gradients,moments=moments,
                         lr=1e-4,chunk_bytes=512,rounding_seed=57)
    phases=MasterPhaseManager(model,gradients,optimizer,leaf_predicate=lambda m:isinstance(m,nn.Linear),
                             retained_bytes=local(params[0]).numel()*2)
    model._qwen38_expert_sources={}
    reference_parameters=dict(reference.named_parameters());independent={}
    for name,p in named:
        a=local(p).detach().float().cpu().numpy().copy()
        independent[name]=(a,np.zeros_like(a),np.zeros_like(a))
    records=[]
    for step in range(1,3):
        expected={name:torch.zeros_like(p,dtype=torch.float32) for name,p in reference_parameters.items()}
        optimizer.zero_grad();reference.zero_grad(set_to_none=True)
        for micro in range(2):
            generator=torch.Generator().manual_seed(511+step*31+micro*3+rank)
            x=torch.randn(3,dimension,generator=generator).to(device,dtype=torch.bfloat16)
            y=torch.randn(3,dimension,generator=generator).to(device)
            ref_loss=F.mse_loss(reference(x).float(),y)/2;ref_loss.backward()
            loss=F.mse_loss(model(x).float(),y)/2;loss.backward()
            for name,p in reference_parameters.items():
                expected[name].add_(p.grad.float());p.grad=None
        torch.cuda.synchronize(device)
        max_gradient_error=0.0;lost_if_bf16=0;shadow_error=None
        expected_owned={}
        for name,p in named:
            whole=expected[name];dist.all_reduce(whole);whole.div_(world)
            expected_local=whole.chunk(world,dim=0)[rank].contiguous()
            actual=gradients.slots[p].gradient
            err=float((actual.to(device)-expected_local).abs().max())
            max_gradient_error=max(max_gradient_error,err)
            lost_if_bf16+=int((expected_local!=expected_local.to(torch.bfloat16).float()).sum())
            # The 72-way NCCL AVG may pre-scale inputs while this independent
            # reference sums BF16 contributions then divides. Preserve the
            # exact observed FP32 input for the separate optimizer-equation
            # reference after the reduction error has been checked.
            expected_owned[name]=actual.detach().cpu().numpy().copy()
            if p.grad is not None:shadow_error='Transient BF16 parameter.grad was not cleared'
        gradient_tolerance=0.0 if world==4 else 1e-8
        error=(str(('Actual FP32 reduction capture mismatch',max_gradient_error,lost_if_bf16))
               if max_gradient_error>gradient_tolerance or lost_if_bf16<1 else shadow_error)
        collective_check(f'step{step}-independent-FP32-gradient',error)
        expected_norm=math_sqrt(sum(float(np.square(v.astype(np.float64)).sum()) for v in expected_owned.values()))
        scalar=torch.tensor(expected_norm**2,device=device,dtype=torch.float64);dist.all_reduce(scalar)
        expected_norm=math_sqrt(float(scalar))
        norm=gradients.clip_grad_norm(.01,expected_local_numel=sum(local(p).numel() for p in params),
            expected_global_numel=sum(p.numel() for p in reference.parameters()),collective_device=device)
        collective_check(f'step{step}-global-clipping',
            'Global FP32 clipping norm differs' if abs(norm-expected_norm)>1e-12 else None)
        clipped=gradients.audit()
        scale=min(1,.01/(norm+1e-6));optimizer_error=None;previous_actual={}
        for name,p in named:
            weight=local(p).detach().cpu().float().numpy().copy()
            if step==1:previous_actual[name]=(weight,np.zeros_like(weight),np.zeros_like(weight))
            else:
                handle=optimizer.state[p]['moments']
                previous_actual[name]=(weight,handle['first'].float().numpy().copy(),
                                       handle['second'].float().numpy().copy())
        try:optimizer.step()
        except Exception as exc:optimizer_error=f'{type(exc).__name__}: {exc}'
        collective_check(f'step{step}-optimizer',optimizer_error)
        max_weight_error=0.0;state_mismatches=0;max_BF16_ulp=0;compared=0;examples=[]
        mismatch_by_role=dict(weight=0,exp_avg=0,exp_avg_sq=0)
        compared_by_role=dict(weight=0,exp_avg=0,exp_avg_sq=0)
        replay_mismatches=0;unexplained_threshold_mismatches=0;maximum_raw_ulp=0
        for name,p in named:
            # Reference receives its own independently averaged/reduced gradient,
            # then applies the same public global clip scalar.
            g=np.multiply(expected_owned[name],np.float32(scale),dtype=np.float32)
            offset=rank*local(p).numel()
            old=previous_actual[name]
            # NumPy reference uses global counters for the same named parameter.
            from test_bf16_numerics import reference_round
            import qwen38_bf16_optimizer as opt
            from unittest.mock import patch
            original_round=reference_round
            def round_with_offset(v,key,unused=0):return original_round(v,key,offset)
            with patch('test_bf16_numerics.reference_round',round_with_offset):
                independent[name],cpu_raw=numpy_adam(*old,g,step=step,lr=1e-4,betas=(.9,.999),
                                            eps=1e-8,wd=0,name=name,seed=57,return_unrounded=True)
            gpu_raw=replay_gpu_arithmetic(old,g,step=step,device=device)
            handle=optimizer.state[p]['moments']
            arrays=(local(p).detach().cpu().float().numpy(),handle['first'].float().numpy(),
                    handle['second'].float().numpy())
            mismatches=[int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)))
                        for a,b in zip(arrays,independent[name])]
            state_mismatches+=sum(mismatches)
            for role,a,b,raw_cpu,raw_gpu in zip(('weight','exp_avg','exp_avg_sq'),arrays,
                                               independent[name],cpu_raw,gpu_raw):
                actual_bits=a.view(np.uint32).reshape(-1)>>16
                reference_bits=b.view(np.uint32).reshape(-1)>>16
                delta=np.abs(actual_bits.astype(np.int64)-reference_bits.astype(np.int64))
                max_BF16_ulp=max(max_BF16_ulp,int(delta.max()));compared+=a.size
                mismatch_by_role[role]+=int(np.count_nonzero(delta));compared_by_role[role]+=a.size
                role_key=rounding_key(name,step,role,57)
                replay=stochastic_bf16(raw_gpu.contiguous(),key=role_key,offset=offset).float().cpu().numpy()
                replay_mismatches+=int(np.count_nonzero(a.view(np.uint32)!=replay.view(np.uint32)))
                gpu_bits=raw_gpu.cpu().numpy().view(np.uint32).reshape(-1).astype(np.int64)
                cpu_bits=raw_cpu.view(np.uint32).reshape(-1).astype(np.int64)
                random=counter_random16(a.size,key=role_key,offset=offset,device='cpu').numpy()
                for index in np.nonzero(delta)[0][:4]:
                    low=min(int(gpu_bits[index]),int(cpu_bits[index]))
                    high=max(int(gpu_bits[index]),int(cpu_bits[index]))
                    rng=int(random[index]);bias=65535-rng
                    threshold=((low+bias)//65536+1)*65536-bias
                    raw_ulp=high-low;maximum_raw_ulp=max(maximum_raw_ulp,raw_ulp)
                    # The sole allowed mismatch is a <=8-FP32-ULP FMA
                    # perturbation crossing this exact counter SR threshold.
                    explained=(raw_ulp<=8 and delta[index]==1 and low<threshold<=high
                        and (int(gpu_bits[index])+bias)//65536==int(actual_bits[index])
                        and (int(cpu_bits[index])+bias)//65536==int(reference_bits[index]))
                    unexplained_threshold_mismatches+=int(not explained)
                    examples.append(dict(parameter=name,role=role,global_coordinate=offset+int(index),
                        actual=float(a.reshape(-1)[index]),reference=float(b.reshape(-1)[index]),
                        actual_BF16_bits=int(actual_bits[index]),reference_BF16_bits=int(reference_bits[index]),
                        BF16_ULP_difference=int(delta[index]),GPU_pre_SR_FP32_bits=int(gpu_bits[index]),
                        CPU_pre_SR_FP32_bits=int(cpu_bits[index]),pre_SR_FP32_ULP_difference=raw_ulp,
                        RNG16=rng,GPU_low16=int(gpu_bits[index])&65535,CPU_low16=int(cpu_bits[index])&65535,
                        exact_SR_threshold_FP32_bits=threshold,FMA_threshold_crossing_explained=bool(explained)))
            max_weight_error=max(max_weight_error,float(np.max(np.abs(arrays[0]-independent[name][0]))))
        # CPU NumPy and GPU lerp/addcmul associate FP32 FMA differently; a
        # threshold crossing may select the adjacent BF16 result. Production
        # GPU arithmetic is unchanged. Bound and publish every difference.
        limit=0 if world==4 else max(1,compared//10000)
        all_statistics=[None for _ in range(world)]
        dist.all_gather_object(all_statistics,dict(mismatches=mismatch_by_role,compared=compared_by_role))
        global_mismatch_by_role={role:sum(row['mismatches'][role] for row in all_statistics)
                                 for role in mismatch_by_role}
        global_compared_by_role={role:sum(row['compared'][role] for row in all_statistics)
                                for role in compared_by_role}
        global_mismatch=sum(global_mismatch_by_role.values())
        global_compared=sum(global_compared_by_role.values())
        global_limit=0 if world==4 else max(1,global_compared//10000)
        bad=(state_mismatches>limit or max_BF16_ulp>1 or replay_mismatches
             or unexplained_threshold_mismatches or global_mismatch>global_limit)
        collective_check(f'step{step}-independent-BF16-Adam',
            str(dict(mismatches=state_mismatches,maximum_ULP=max_BF16_ulp,
                     compared=compared,examples=examples)) if bad else None)
        audit=optimizer.audit();optimizer.zero_grad()
        before={name:local(p).detach().cpu().clone() for name,p in named}
        phases.prepare_rollout()
        exporter=CanonicalExport(model,expected_world_size=world,collective_device=device)
        for name,value in exporter:
            reference_parameters[name].data.copy_(value)
            dist.barrier();exporter.acknowledge(name,successful_native_ranks=range(world))
        del value
        released=exporter.close()
        phases.restore_training(native_contexts_released=True,refit_sources_released=exporter.closed)
        collective_check(f'step{step}-offload-restore',
            'BF16 master bits changed across CPU offload round trip'
            if any(not torch.equal(before[name],local(p).cpu()) for name,p in named) else None)
        records.append(dict(step=step,FP32_gradient_max_abs_error=max_gradient_error,
            independent_reduction_reference_tolerance=gradient_tolerance,
            elements_lost_if_BF16_cast=lost_if_bf16,global_norm=norm,clip_scale=scale,
            independent_BF16_Adam_bit_mismatches=state_mismatches,gradient_store=clipped,
            independent_BF16_Adam_max_ULP=max_BF16_ulp,
            independent_BF16_Adam_compared_elements=compared,
            independent_BF16_Adam_difference_examples=examples,
            independent_BF16_Adam_allowed_mismatches=limit,
            independent_BF16_Adam_mismatches_by_state=mismatch_by_role,
            independent_BF16_Adam_mismatch_fraction_by_state={role:mismatch_by_role[role]/compared_by_role[role]
                for role in mismatch_by_role},
            global_independent_BF16_Adam_mismatches_by_state=global_mismatch_by_role,
            global_independent_BF16_Adam_mismatch_fraction_by_state={role:
                global_mismatch_by_role[role]/global_compared_by_role[role] for role in global_mismatch_by_role},
            global_independent_BF16_Adam_mismatch_fraction=global_mismatch/global_compared,
            global_independent_BF16_Adam_allowed_mismatches=global_limit,
            exact_GPU_FP32_replay_bit_mismatches=replay_mismatches,
            maximum_FP32_ULP_at_SR_crossing=maximum_raw_ulp,
            unexplained_SR_threshold_mismatches=unexplained_threshold_mismatches,
            per_step_reference_uses_actual_persistent_previous_state=True,
            optimizer=audit,refit_release= released,actual_native_RTP_consumer=False))
    result=dict(schema='q38-BF16-state-FP32-gradient-FSDP-world-v2',passed=True,rank=rank,world_size=world,
        pid=os.getpid(),hostname=socket.gethostname(),job_id=args.job_id,
        gpu_uuid=str(torch.cuda.get_device_properties(device).uuid),updates=records,
        phase_transitions=phases.transition_records,official_model_weights_opened=False,
        full_Qwen38_RL_validated=False)
    out=Path(args.output)/args.job_id/f'rank{rank}';out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('BF16_FSDP_WORLD_PASS',rank,json.dumps({'passed':True,'world':world,'updates':len(records)}),flush=True)
    gradients.remove_hooks();dist.barrier();dist.destroy_process_group()


def math_sqrt(value):return float(value)**.5


if __name__=='__main__':main()
