"""Compare real FSDP unsharded chunk lifetimes, old whole-layer vs bounded AC.

Synthetic HF weights only; reads the official configuration, never weights.
"""
import argparse, copy, gc, json, os, socket, time
from pathlib import Path
import torch
from torch.distributed import init_process_group,destroy_process_group
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard,MixedPrecisionPolicy
from transformers import AutoConfig,AutoModelForCausalLM
from qwen38_chunked_experts import ChunkedExperts,ExpertChunk
from qwen38_bf16_optimizer import GradientStore,local
from qwen38_bf16_master_phases import disable_prefetch_and_bound_reduction
from qwen38_bounded_checkpoint import install_bounded_checkpoints
from hf_qwen38_meta_and_cpu import tiny_config_dict


def convert(model):
    for layer in model.model.layers:
        layer.mlp.experts=ChunkedExperts(layer.mlp.experts,chunk_experts=2,require_meta=False)


class LifetimeAudit:
    def __init__(self,model):
        self.chunks=[(name,module) for name,module in model.named_modules() if isinstance(module,ExpertChunk)]
        self.events=[];self.maximum_live_chunks=0;self.maximum_live_bytes=0
        self.maximum_live_chunks_per_layer=0;self.hooks=[]
        for name,chunk in self.chunks:
            self.hooks.append(chunk.register_forward_pre_hook(self._hook(name,'pre')))
            self.hooks.append(chunk.register_forward_hook(self._hook(name,'post')))

    def snapshot(self,name,boundary):
        if torch._C._current_graph_task_id()<0:return
        live=[];real_bytes=0;by_layer={};pending=0
        for key,chunk in self.chunks:
            state=chunk._get_fsdp_state()
            for group in state._fsdp_param_groups:
                if group._all_gather_result is not None:pending+=1
                if not group.is_unsharded:continue
                # Inspect actual local CUDA backing storages, not global
                # DTensor numel or a theoretical allocation formula.
                seen=set();size=0
                for p in group.fsdp_params:
                    value=p.unsharded_param
                    store=value.untyped_storage();pointer=store.data_ptr()
                    if pointer not in seen:seen.add(pointer);size+=store.nbytes()
                if size:
                    live.append(key);real_bytes+=size
                    layer=key.split('.mlp.experts.')[0];by_layer[layer]=by_layer.get(layer,0)+1
        self.maximum_live_chunks=max(self.maximum_live_chunks,len(live))
        self.maximum_live_bytes=max(self.maximum_live_bytes,real_bytes)
        self.maximum_live_chunks_per_layer=max(self.maximum_live_chunks_per_layer,max(by_layer.values(),default=0))
        self.events.append(dict(name=name,boundary=boundary,live_count=len(live),live_actual_cuda_storage_bytes=real_bytes,
                               pending_all_gathers=pending,live_names=live))

    def _hook(self,name,boundary):
        def observe(*unused):self.snapshot(name,boundary)
        return observe

    def result(self):
        return dict(chunk_modules=len(self.chunks),maximum_live_chunks_during_backward=self.maximum_live_chunks,
            maximum_live_chunks_one_layer_during_backward=self.maximum_live_chunks_per_layer,
            maximum_actual_unsharded_chunk_cuda_bytes=self.maximum_live_bytes,observed_events=len(self.events),
            peak_event=max(self.events,key=lambda r:r['live_actual_cuda_storage_bytes']) if self.events else None)

    def close(self):
        for h in self.hooks:h.remove()


def run(mode,source,mesh,ids,device):
    model=copy.deepcopy(source).to(device)
    if mode=='bounded':install_bounded_checkpoints(model,expected_layers=4)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'early_stop':False})
    mp=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,cast_forward_inputs=True)
    for module in model.modules():
        if isinstance(module,ExpertChunk):fully_shard(module,mesh=mesh,mp_policy=mp,reshard_after_forward=True)
    for layer in model.model.layers:fully_shard(layer,mesh=mesh,mp_policy=mp,reshard_after_forward=True)
    fully_shard(model,mesh=mesh,mp_policy=mp,reshard_after_forward=True)
    disable_prefetch_and_bound_reduction(model)
    for name,p in model.named_parameters():p._qwen36_parameter_name=name
    params=list(model.parameters());store=GradientStore(retained_bytes=16*1024**2);store.register_hooks(params)
    audit=LifetimeAudit(model);model.train();torch.cuda.synchronize(device);torch.cuda.empty_cache()
    baseline=torch.cuda.memory_allocated(device);torch.cuda.reset_peak_memory_stats(device)
    result=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids,use_cache=False)
    logits=result.logits.detach().cpu();loss=result.loss.detach().item();result.loss.backward()
    torch.cuda.synchronize(device)
    values={name:store.slots[p].gradient.cpu().clone() for name,p in model.named_parameters()}
    if any(p.grad is not None for p in params):raise RuntimeError('Untracked gradients persisted')
    report=dict(mode=mode,loss=loss,baseline_allocated_bytes=baseline,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated(device)-baseline,
        gradient_store=store.audit(),lifetime=audit.result(),
        layer_checkpoint_audits=[dict(l._q38_bounded_checkpoint_audit) for l in model.model.layers] if mode=='bounded' else None)
    if not values or not any(v.count_nonzero().item() for v in values.values()):raise RuntimeError('No real gradients')
    audit.close();store.remove_hooks();store.clear()
    del result,store,params,model,audit
    gc.collect();torch.cuda.synchronize(device);torch.cuda.empty_cache();dist.barrier()
    return report,logits,values


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local_rank)
    device=torch.device('cuda',local_rank)
    from datetime import timedelta
    init_process_group('nccl',device_id=device,timeout=timedelta(seconds=180))
    if dist.get_world_size()!=4:raise ValueError('Authorized exactly node2 four-GPU qualification')
    torch.set_num_threads(2);torch.manual_seed(2881);torch.backends.cuda.enable_cudnn_sdp(False)
    original=json.loads((Path(__file__).resolve().parents[1] / 'qwen38_native_bootstrap/fixtures/config.json').read_text())
    c=tiny_config_dict(original);c.update(hidden_size=512,moe_intermediate_size=256,
        shared_expert_intermediate_size=256,num_experts=64,num_experts_per_tok=4)
    config=AutoConfig.for_model('qwen3_5_moe_text',**{k:v for k,v in c.items() if k!='model_type'})
    config._attn_implementation='sdpa';config._experts_implementation='eager'
    source=AutoModelForCausalLM.from_config(config,dtype=torch.bfloat16);convert(source)
    mesh=init_device_mesh('cuda',(4,))
    ids=torch.randint(4,289,(1,64),generator=torch.Generator().manual_seed(rank+1981)).to(device)
    a,alogits,ag=run('whole_decoder',source,mesh,ids,device)
    b,blogits,bg=run('bounded',source,mesh,ids,device)
    target=Path(args.output);target.mkdir(parents=True,exist_ok=True)
    (target/f'partial-lifetimes-rank{rank}.json').write_text(json.dumps(dict(old=a,new=b),indent=2)+'\n')
    logit_error=float((alogits.float()-blogits.float()).abs().max())
    maximum=max(float((ag[n]-bg[n]).abs().max()) for n in ag if ag[n].numel())
    relative=(sum((ag[n].double()-bg[n].double()).square().sum().item() for n in ag)/
              max(1e-100,sum(ag[n].double().square().sum().item() for n in ag)))**.5
    checks=dict(logits_max_abs_error=logit_error,gradient_max_abs_error=maximum,gradient_relative_L2=relative,
        old_retained_full_layer_expert_chunks=a['lifetime']['maximum_live_chunks_one_layer_during_backward']>=32,
        new_bounded_expert_chunks=b['lifetime']['maximum_live_chunks_during_backward']<=4,
        actual_gathered_storage_reduced=b['lifetime']['maximum_actual_unsharded_chunk_cuda_bytes']<a['lifetime']['maximum_actual_unsharded_chunk_cuda_bytes'])
    error=None
    if (logit_error>.05 or relative>.02 or not all(checks[k] for k in ['old_retained_full_layer_expert_chunks','new_bounded_expert_chunks','actual_gathered_storage_reduced'])):
        error=f'Lifetime/equivalence gate failed {checks}'
    statuses=[None]*4;dist.all_gather_object(statuses,dict(rank=rank,error=error))
    report=dict(schema='q38-real-HF-FSDP4-checkpoint-storage-lifetimes-v1',rank=rank,world_size=4,
        hostname=socket.gethostname(),pid=os.getpid(),gpu_uuid=str(torch.cuda.get_device_properties(device).uuid),
        passed=not any(s['error'] for s in statuses),checks=checks,old=a,new=b,
        small_synthetic_HF_config=c,official_weight_files_opened=False,fullmodel_RL_accepted=False,time=time.time())
    target=Path(args.output);target.mkdir(parents=True,exist_ok=True)
    (target/f'lifetimes-rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Q38_CHECKPOINT_LIFETIMES_RESULT',json.dumps(dict(rank=rank,passed=report['passed'],checks=checks,
        old_lifetime=a['lifetime'],new_lifetime=b['lifetime'],old_incremental_peak=a['incremental_peak_allocated_bytes'],
        new_incremental_peak=b['incremental_peak_allocated_bytes'])),flush=True)
    destroy_process_group()
    if not report['passed']:raise RuntimeError(str(statuses))


if __name__=='__main__':main()
