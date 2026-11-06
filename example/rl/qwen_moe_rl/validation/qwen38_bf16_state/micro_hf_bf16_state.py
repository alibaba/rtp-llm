"""Real tiny Qwen3.5-MoE HF forward/backward through the new BF16 FSDP path.

The separate unmodified HF model is a tiny reference only. No official-sized
parameter or native RTP process is created by this probe.
"""
from __future__ import annotations
import argparse,copy,json,os,socket,time
from pathlib import Path
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard,MixedPrecisionPolicy
from torch.utils.checkpoint import set_checkpoint_early_stop
from transformers import AutoConfig,AutoModelForCausalLM
from qwen38_chunked_experts import ChunkedExperts,ExpertChunk
from qwen38_vocab_tiles import TiledEmbedding,TiledLMHead,VocabTile
from qwen38_bf16_optimizer import GradientStore,CpuBf16Moments,Bf16CpuAdamW,local,global_parameter_offset
from qwen38_bf16_master_phases import MasterPhaseManager,disable_prefetch_and_bound_reduction
from qwen38_canonical_export import CanonicalExport
from hf_qwen38_meta_and_cpu import tiny_config_dict
from qwen38_policy_probe import teacher_forced_target_scores


def install_tiny_storage(model):
    """Test-only small materialized copies; production helpers require meta."""
    sources={}
    candidates=[(name,module) for name,module in model.named_modules()
                if type(module).__name__=='Qwen3_5MoeExperts']
    if len(candidates)!=4:raise RuntimeError('Wrong synthetic tiny architecture')
    for prefix,old in candidates:
        parent,attribute=prefix.rsplit('.',1);replacement=ChunkedExperts(old,chunk_experts=2,require_meta=False)
        setattr(model.get_submodule(parent),attribute,replacement)
        for index,chunk in enumerate(replacement.chunks):
            for field in ['gate_up_proj','down_proj']:
                sources[f'{prefix}.chunks.{index}.{field}']={'canonical_name':prefix+'.'+field,
                    'expert_begin':chunk.expert_begin,'expert_end':chunk.expert_end,
                    'runtime_shape':list(getattr(chunk,field).shape),'source_shape':list(getattr(old,field).shape)}
    for prefix,old,kind in [('model.embed_tokens',model.model.embed_tokens,TiledEmbedding),
                           ('lm_head',model.lm_head,TiledLMHead)]:
        replacement=kind(old,rows=72,require_meta=False)
        parent,attribute=prefix.rsplit('.',1) if '.' in prefix else ('',prefix)
        setattr(model.get_submodule(parent),attribute,replacement)
        for index,tile in enumerate(replacement.tiles):
            sources[f'{prefix}.tiles.{index}.weight']={'canonical_name':prefix+'.weight',
                'vocab_row_begin':tile.row_begin,'vocab_row_end':tile.row_end,
                'runtime_shape':list(tile.weight.shape),'source_shape':list(old.weight.shape)}
    model._qwen38_expert_sources=sources


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--job-id',required=True)
    parser.add_argument('--metadata',required=True);args=parser.parse_args()
    started=time.time();rank=int(os.environ['RANK']);lr=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(lr);device=torch.device('cuda',lr)
    from datetime import timedelta
    dist.init_process_group('nccl',timeout=timedelta(seconds=150),device_id=device)
    if dist.get_world_size()!=4:raise ValueError('This actual HF tiny probe is authorized for node2 four GPUs')
    torch.manual_seed(719);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.enable_cudnn_sdp(False)
    original=json.loads((Path(args.metadata)/'config.json').read_text());small=tiny_config_dict(original)
    config=AutoConfig.for_model('qwen3_5_moe_text',**{key:value for key,value in small.items() if key!='model_type'})
    config._attn_implementation='sdpa';config._experts_implementation='eager'
    source=AutoModelForCausalLM.from_config(config,dtype=torch.bfloat16)
    reference=copy.deepcopy(source).to(device)
    for parameter in reference.parameters():
        parameter.requires_grad_(False);parameter.data=parameter.data.to(torch.bfloat16)
    # Keep the initialized nonpersistent rotary buffers FP32, as in the actor.
    reference.eval();install_tiny_storage(source);model=source.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'early_stop':False})
    mesh=init_device_mesh('cuda',(4,));mp=MixedPrecisionPolicy(param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,cast_forward_inputs=True)
    leaves=[module for module in model.modules() if isinstance(module,(ExpertChunk,VocabTile))]
    for module in leaves:fully_shard(module,mesh=mesh,mp_policy=mp,reshard_after_forward=True)
    for module in model.model.layers:fully_shard(module,mesh=mesh,mp_policy=mp,reshard_after_forward=True)
    fully_shard(model,mesh=mesh,mp_policy=mp,reshard_after_forward=True)
    fsdp_groups=disable_prefetch_and_bound_reduction(model)
    for name,p in model.named_parameters():p._qwen36_parameter_name=name
    params=list(model.parameters());owned=sum(local(p).numel() for p in params)
    whole=sum(p.numel() for p in reference.parameters());target=Path(args.output)/args.job_id/f'rank{rank}'
    target.mkdir(parents=True,exist_ok=False)
    gradients=GradientStore(retained_bytes=64*1024);gradients.register_hooks(params)
    moments=CpuBf16Moments(cpu_budget_bytes=owned*4)
    optimizer=Bf16CpuAdamW(params,gradients=gradients,moments=moments,lr=1e-4,chunk_bytes=4096)
    import numpy as np
    from test_bf16_numerics import numpy_adam,reference_round
    from unittest.mock import patch
    independent={}
    for name,p in model.named_parameters():
        a=local(p).detach().float().cpu().numpy().copy()
        independent[name]=(a,np.zeros_like(a),np.zeros_like(a))
    offload_ids={id(p) for module in leaves for p in module.parameters()}
    dense=sum(local(p).numel()*2 for p in params if id(p) not in offload_ids)
    phases=MasterPhaseManager(model,gradients,optimizer,
        leaf_predicate=lambda module:isinstance(module,(ExpertChunk,VocabTile)),retained_bytes=dense+32*1024)
    reference_parameters=dict(reference.named_parameters());updates=[]
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(2):
        phases.require_training();optimizer.zero_grad();model.train()
        prompt_ids=[4,7,19,31];target_ids=[9,12,5]
        hf_scores=teacher_forced_target_scores(model,prompt_ids,target_ids,device)
        reference_scores=teacher_forced_target_scores(reference,prompt_ids,target_ids,device)
        hf_probe_error=max(abs(a-b) for a,b in zip(hf_scores,reference_scores))
        if hf_probe_error>.05:raise RuntimeError('Real HF target-score probe differs from independent reference')
        generator=torch.Generator().manual_seed(1700+rank*31+step*241)
        ids=torch.randint(4,289,(2,8),generator=generator).to(device);mask=torch.ones_like(ids)
        with torch.no_grad():expected=reference(input_ids=ids,attention_mask=mask,use_cache=False).logits
        # Every rank must recompute every checkpointed chunk in the same order,
        # even when its local expert routing hits a different subset.
        with set_checkpoint_early_stop(False):
            result=model(input_ids=ids,attention_mask=mask,labels=ids,use_cache=False)
        error=float((result.logits.detach().float()-expected.float()).abs().max())
        if error>0.05 or not torch.isfinite(result.loss) or not result.loss.requires_grad:
            raise RuntimeError(f'Actual HF BF16 logits/loss check failed: {error}')
        result.loss.backward();torch.cuda.synchronize(device)
        norm=gradients.clip_grad_norm(0.01,expected_local_numel=owned,
            expected_global_numel=whole,collective_device=device)
        if norm<=0 or not gradients.cpu_spilled_bytes or not gradients.retained_bytes:
            raise RuntimeError('The actual BF16 HF backward did not execute both FP32 reduced-gradient tiers')
        expected_grad={name:gradients.slots[p].gradient.detach().cpu().numpy().copy()
                       for name,p in model.named_parameters()}
        gradient_audit=gradients.audit();optimizer.step()
        weight_error=0.0;moment_error=0.0;bit_mismatches=0;max_BF16_ulp=0;compared=0
        for name,actual in model.named_parameters():
            if not local(actual).numel():continue
            offset=global_parameter_offset(actual)
            def round_offset(value,key,unused=0):return reference_round(value,key,offset)
            with patch('test_bf16_numerics.reference_round',round_offset):
                independent[name]=numpy_adam(*independent[name],expected_grad[name],step=step+1,
                    lr=1e-4,betas=(.9,.999),eps=1e-8,wd=0,name=name,seed=20261106)
            handle=optimizer.state[actual]['moments']
            arrays=(local(actual).detach().cpu().float().numpy(),handle['first'].float().numpy(),
                    handle['second'].float().numpy())
            bit_mismatches+=sum(int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)))
                               for a,b in zip(arrays,independent[name]))
            compared+=sum(a.size for a in arrays)
            max_BF16_ulp=max(max_BF16_ulp,*[int(np.max(np.abs(
                (a.view(np.uint32).astype(np.int64)>>16)-(b.view(np.uint32).astype(np.int64)>>16))))
                for a,b in zip(arrays,independent[name])])
            weight_error=max(weight_error,float(np.max(np.abs(arrays[0]-independent[name][0]))))
            moment_error=max(moment_error,*[float(np.max(np.abs(arrays[i]-independent[name][i]))) for i in (1,2)])
        # The independent CPU equations and GPU lerp/addcmul use different
        # FP32 FMA association. A rare SR threshold crossing can select the
        # adjacent BF16 neighbor; retain/report this difference explicitly.
        if max_BF16_ulp>1 or bit_mismatches>max(1,compared//10000):
            raise RuntimeError(f'Actual HF BF16 optimizer reference bound failed: {bit_mismatches,max_BF16_ulp,compared}')
        record={'step':step+1,'loss':float(result.loss.detach()),'loss_nonzero':result.loss.item()!=0,
            'HF_target_score_probe_max_abs_error':hf_probe_error,'HF_unmodified_reference_logits_max_abs_error':error,'weight_max_abs_error':weight_error,
            'moment_max_abs_error':moment_error,'independent_BF16_state_bit_mismatches':bit_mismatches,'maximum_BF16_ULP_difference':max_BF16_ulp,'compared_state_elements':compared,'independent_reference_tolerance':'at most one BF16 ULP on at most 0.01 percent of states, due to FP32 FMA before SR','gradient_store':gradient_audit,'norm':norm}
        optimizer.zero_grad();phases.prepare_rollout()
        exporter=CanonicalExport(model,expected_world_size=4,collective_device=device);keys=[]
        for name,value in exporter:
            with torch.no_grad():reference_parameters[name].copy_(value)
            checked=torch.ones((),device=device,dtype=torch.int64);dist.all_reduce(checked)
            if checked.item()!=4:raise RuntimeError('Missing tiny HF reference consumer ACK')
            exporter.acknowledge(name,successful_native_ranks=range(4));keys.append(name)
        del value;released=exporter.close()
        if released['cuda_allocated_bytes_before_release']-released['cuda_allocated_bytes_after_release']<exporter.maximum_backing_bytes:
            raise RuntimeError('Tiny HF refit backing was not released by the actual allocator')
        if set(keys)!=set(reference_parameters):raise RuntimeError('Actual HF canonical refit omitted a parameter')
        phases.restore_training(native_contexts_released=True,refit_sources_released=exporter.closed)
        record['canonical_export']={'source_keys':len(keys),'all_actual_HF_keys_covered':True,
                                     'release_receipt':released,'native_RTP_consumer':False}
        updates.append(record);dist.barrier()
    report={'schema':'qwen38-actual-tiny-HF-BF16-storage-state-FSDP4-v1','passed':True,'job_id':args.job_id,
        'rank':rank,'world_size':4,'hostname':socket.gethostname(),'pid':os.getpid(),
        'gpu_uuid':str(torch.cuda.get_device_properties(device).uuid),'torch_version':torch.__version__,
        'compute_dtype':'bfloat16','master_dtype':'bfloat16','gradient_dtype':'float32','persistent_moment_dtype':'bfloat16',
        'checkpointing':'HF nonreentrant; early_stop=False for rank-consistent chunk collectives',
        'tiny_config':small,'real_local_numel':owned,'global_parameter_numel':whole,
        'updates':updates,'phase_transitions':phases.transition_records,'fsdp_modules':fsdp_groups,
        'optimizer':optimizer.audit(),'maximum_allocated_gpu_bytes':torch.cuda.max_memory_allocated(device),
        'maximum_reserved_gpu_bytes':torch.cuda.max_memory_reserved(device),
        'official_weight_files_opened':False,'full_Qwen38_RL_validated':False,'native_RTP_started':False,
        'started_unix':started,'finished_unix':time.time()}
    with (target/'result.json').open('x') as stream:json.dump(report,stream,indent=2)
    print('ACTUAL_TINY_HF_BF16_STATE_FSDP_PASS',rank,json.dumps({'updates':len(updates),'weight_max_abs_error':max(r['weight_max_abs_error'] for r in updates),'moment_max_abs_error':max(r['moment_max_abs_error'] for r in updates)}),flush=True)
    gradients.remove_hooks();dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
