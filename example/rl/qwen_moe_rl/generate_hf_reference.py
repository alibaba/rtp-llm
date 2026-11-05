"""Independent full HF BF16 reference with bounded checkpoint materialization.

Only parameters start on meta; rotary buffers retain ordinary CPU initialization.
Official safetensors are read one CPU tensor at a time and assigned directly to
their final device. No model from_pretrained, training/RTP export, CPU parameter
offload or complete CUDA checkpoint copy is used.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from hashlib import sha256
import gc
import inspect
import json
import math
import os
from pathlib import Path
import time
from checkpoint_contract import default_model, identity, load_checkpoint, metadata

PROJECT=Path(__file__).resolve().parent
QUESTIONS=['What is 17 times 19? Reply only the number.',
           'Evaluate 2**10. Reply with just the answer.',
           'Give two primes whose sum is 30. Return the pair only.']
HEADROOM_BYTES=16*1024**3
MAX_FP32_PARAMETER_BYTES=128*1024


def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def validate_map(mapping):
    names=list(mapping)
    if '' in mapping or any(b.startswith(a+'.') for a in names for b in names if a!=b):
        raise ValueError('Use nonoverlapping terminal device blocks, without a root fallback')


def reference_device_map(layers):
    mapping={'model.language_model.embed_tokens':'cuda:0',
             'model.language_model.rotary_emb':'cuda:0',
             'model.visual':'cuda:3','model.language_model.norm':'cuda:3','lm_head':'cuda:3'}
    mapping.update({f'model.language_model.layers.{i}':f'cuda:{min(3,i*4//layers)}' for i in range(layers)})
    validate_map(mapping)
    return mapping


def target_device(name,mapping):
    matches=[device for prefix,device in mapping.items() if name==prefix or name.startswith(prefix+'.')]
    if len(matches)!=1:
        raise ValueError(f'{name}: expected exactly one final device block, found {matches}')
    return matches[0]


def parameter_plan(shapes,fp32_names,mapping):
    validate_map(mapping)
    fp32_names=set(fp32_names)
    if not fp32_names<=set(shapes):raise ValueError('FP32 source is outside the official model')
    rows=[]
    devices={device:dict(parameter_numel=0,parameter_bytes=0,fp32_numel=0,
                        maximum_single_parameter_bytes=0) for device in set(mapping.values())}
    for name,shape in shapes.items():
        if not shape or any(type(n) is not int or n<=0 for n in shape):raise ValueError('Invalid shape: '+name)
        device=target_device(name,mapping)
        numel=math.prod(shape)
        width=4 if name in fp32_names else 2
        if width==4 and numel*4>MAX_FP32_PARAMETER_BYTES:
            raise ValueError('A large weight may not silently become FP32: '+name)
        size=numel*width
        rows.append(dict(name=name,shape=shape,device=device,dtype='torch.float32' if width==4 else 'torch.bfloat16',
                         numel=numel,bytes=size))
        entry=devices[device]
        entry['parameter_numel']+=numel
        entry['parameter_bytes']+=size
        entry['fp32_numel']+=numel if width==4 else 0
        entry['maximum_single_parameter_bytes']=max(entry['maximum_single_parameter_bytes'],size)
    return dict(schema='qwen397-independent-hf-bounded-load-plan-v2',parameter_count=len(rows),
                parameter_numel=sum(row['numel'] for row in rows),total_parameter_bytes=sum(row['bytes'] for row in rows),
                fp32_parameter_names=sorted(fp32_names),per_device=devices,parameters=rows,
                minimum_free_after_load_bytes=HEADROOM_BYTES,checkpoint_gpu_staging_copies=0,
                maximum_live_cpu_source_tensors=1)


def declared_fp32_policy(model,expected_gdn):
    declarations={key:sorted(getattr(model,key,None) or []) for key in
                  ('_keep_in_fp32_modules','_keep_in_fp32_modules_strict')}
    strict=set(declarations['_keep_in_fp32_modules_strict'])
    # A complete component match cannot mistake gate_up_proj for a gate router.
    expanded={name for name,_ in model.named_parameters() for item in strict
              if item in name.split('.') or name==item or name.endswith('.'+item)}
    if expanded-set(expected_gdn):
        raise RuntimeError('Unexpected HF strict FP32 declaration: '+str(sorted(expanded-set(expected_gdn))))
    return set(expected_gdn),declarations


def sample_bytes(tensor,target_dtype):
    import torch
    count=min(64,tensor.numel())
    indices=torch.arange(count,dtype=torch.int64,device=tensor.device)
    if count>1:indices=indices*(tensor.numel()-1)//(count-1)
    sample=tensor.detach().reshape(-1)[indices].to(target_dtype).float().cpu()
    if not torch.isfinite(sample).all().item():raise RuntimeError('Nonfinite sampled checkpoint values')
    return sample.numpy().tobytes(),count


def assign_parameter(model,name,source,shape,device,dtype,setter):
    """Read one CPU source, directly assign one final parameter, check its bytes."""
    import torch
    prior=model.get_parameter(name)
    if not prior.is_meta or list(prior.shape)!=shape or list(source.shape)!=shape:
        raise RuntimeError(name+': source/empty parameter shape mismatch, or target already materialized')
    if source.device.type!='cpu' or source.dtype not in (torch.bfloat16,torch.float32):
        raise RuntimeError(name+': expected an ordinary BF16/F32 CPU safetensors source')
    expected,count=sample_bytes(source,dtype)
    kwargs=dict(value=source,dtype=dtype,non_blocking=False)
    if 'clear_cache' in inspect.signature(setter).parameters:kwargs['clear_cache']=False
    setter(model,name,device,**kwargs)
    actual=model.get_parameter(name)
    if actual.is_meta or str(actual.device)!=str(torch.device(device)) or actual.dtype!=dtype:
        raise RuntimeError(name+': final parameter device/dtype differs from its explicit plan')
    observed,_=sample_bytes(actual,dtype)
    if expected!=observed:raise RuntimeError(name+': actual loaded samples differ from the official source')
    return dict(name=name,shape=shape,device=str(actual.device),dtype=str(actual.dtype),
                source_dtype=str(source.dtype),source_bytes=source.numel()*source.element_size(),
                sample_count=count,sample_fp32_sha256=sha256(observed).hexdigest(),source_sample_match=True)


def buffer_inventory(model):
    import torch
    buffers=dict(model.named_buffers(remove_duplicate=False))
    by_id={id(value):name for name,value in buffers.items()}
    aliases=[]
    for name,value in buffers.items():
        if value.is_meta or value.device.type!='cpu' or value.numel()*value.element_size()>16*1024**2:
            raise RuntimeError('Expected a small ordinarily initialized CPU buffer: '+name)
    for module_name,module in model.named_modules():
        for attr,value in vars(module).items():
            if isinstance(value,torch.Tensor) and id(value) in by_id:
                aliases.append((module_name,attr,by_id[id(value)]))
    return buffers,aliases


def raw_bytes(tensor):
    import torch
    return tensor.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes()


def place_buffers(model,buffers,aliases,mapping,setter):
    loaded=[]
    for name,value in buffers.items():
        device=target_device(name,mapping)
        expected=raw_bytes(value)
        kwargs=dict(value=value,dtype=value.dtype,non_blocking=False)
        if 'clear_cache' in inspect.signature(setter).parameters:kwargs['clear_cache']=False
        setter(model,name,device,**kwargs)
        actual=model.get_buffer(name)
        if actual.dtype!=value.dtype or str(actual.device)!=device or raw_bytes(actual)!=expected:
            raise RuntimeError('Initialized HF buffer changed during final placement: '+name)
        loaded.append(dict(name=name,shape=list(actual.shape),dtype=str(actual.dtype),device=str(actual.device),
                           sha256=sha256(expected).hexdigest(),bytes=len(expected)))
    for module,attr,name in aliases:setattr(model.get_submodule(module),attr,model.get_buffer(name))
    return loaded


def storage_inventory(model):
    return {name:(str(p.device),str(p.dtype),list(p.shape),p.data_ptr(),p.untyped_storage().nbytes())
            for name,p in model.named_parameters()}


def require_storage_unchanged(before,model):
    after=storage_inventory(model)
    changed=[name for name in before if before[name]!=after.get(name)]
    if changed or set(before)!=set(after):raise RuntimeError('Dispatch moved/recopied final parameter storage: '+str(changed[:8]))


def guarded_dispatch(model,mapping):
    """Install ordinary Accelerate hooks, aborting relocation before allocation."""
    import torch
    from accelerate import dispatch_model,hooks
    original=hooks.set_module_tensor_to_device
    def stationary(module,tensor_name,device,*args,**kwargs):
        parent_name,_,leaf=tensor_name.rpartition('.')
        parent=module.get_submodule(parent_name) if parent_name else module
        old=getattr(parent,leaf)
        if old.is_meta or old.device!=torch.device(device):
            raise RuntimeError(f'Dispatch attempted forbidden relocation of {tensor_name}: {old.device} to {device}')
        if (args and args[0] is not None) or kwargs.get('value') is not None:
            raise RuntimeError('Dispatch attempted to rematerialize an already loaded parameter/buffer')
        pointer=old.data_ptr()
        result=original(module,tensor_name,device,*args,**kwargs)
        if getattr(parent,leaf).data_ptr()!=pointer:raise RuntimeError('Dispatch unexpectedly copied tensor storage')
        return result
    hooks.set_module_tensor_to_device=stationary
    try:
        return dispatch_model(model,device_map=mapping,main_device='cuda:0',offload_buffers=False,
                              skip_keys=getattr(model,'_skip_keys_device_placement',None),force_hooks=True)
    finally:
        hooks.set_module_tensor_to_device=original


def load_bounded(model,checkpoint,model_path,mapping,plan,output_dir):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    _,index=metadata(model_path)
    shapes=checkpoint.text|checkpoint.visual
    planned={row['name']:row for row in plan['parameters']}
    buffers,aliases=buffer_inventory(model)
    audit=[]
    maximum_cpu_source=0
    grouped=defaultdict(list)
    for name in shapes:grouped[index['weight_map'][name]].append(name)
    started=time.time()
    for filename,names in sorted(grouped.items()):
        with safe_open(str(model_path/filename),framework='pt',device='cpu') as shard:
            for name in sorted(names):
                source=shard.get_tensor(name)
                row=planned[name]
                dtype=torch.float32 if row['dtype']=='torch.float32' else torch.bfloat16
                proof=assign_parameter(model,name,source,shapes[name],row['device'],dtype,set_module_tensor_to_device)
                maximum_cpu_source=max(maximum_cpu_source,proof['source_bytes'])
                audit.append(dict(proof,checkpoint_file=filename))
                del source
                if len(audit)==1 or len(audit)%32==0 or len(audit)==len(shapes):
                    allocated={f'cuda:{i}':torch.cuda.memory_allocated(i) for i in range(4)}
                    progress=dict(schema='qwen397-hf-bounded-progress-v2',loaded_parameters=len(audit),
                        expected_parameters=len(shapes),last_name=name,allocated_bytes=allocated,
                        updated_unix=time.time())
                    write_json(output_dir/'load-progress.json',progress)
                    print('HF_BOUNDED_PARAMETER_LOADED',json.dumps(progress),flush=True)
        gc.collect()
    buffer_proofs=place_buffers(model,buffers,aliases,mapping,set_module_tensor_to_device)
    del buffers
    before=storage_inventory(model)
    before_buffers={name:(str(value.device),value.data_ptr()) for name,value in model.named_buffers()}
    model=guarded_dispatch(model,mapping)
    require_storage_unchanged(before,model)
    if before_buffers!={name:(str(value.device),value.data_ptr()) for name,value in model.named_buffers()}:
        raise RuntimeError('Dispatch unexpectedly moved initialized buffers')
    for module,attr,name in aliases:
        if getattr(model.get_submodule(module),attr) is not model.get_buffer(name):raise RuntimeError('HF rotary buffer alias changed')
    proof=dict(schema='qwen397-independent-hf-bounded-loader-v2',**identity(),passed=True,
        model_class=type(model).__name__,checkpoint=str(model_path),source='official_safetensors_CPU_one_tensor_at_a_time',
        used_training_or_rtp_export=False,cpu_parameter_offload=False,checkpoint_cuda_staging=False,
        maximum_live_cpu_source_tensors=1,maximum_cpu_source_bytes=maximum_cpu_source,
        parameters=audit,buffers=buffer_proofs,buffer_alias_count=len(aliases),
        dispatch_parameter_storage_unchanged=True,started_unix=started,finished_unix=time.time())
    write_json(output_dir/'bounded-loader.json',proof)
    return model,proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=default_model(PROJECT))
    parser.add_argument('--output-dir',type=Path,default=PROJECT/'acceptance/hf-reference')
    parser.add_argument('--plan-only',action='store_true',help='CPU meta shapes + official shard headers; no CUDA initialization')
    args=parser.parse_args()
    model_path=args.model.resolve(strict=True)
    text,_=metadata(model_path)
    device_map=reference_device_map(text['num_hidden_layers'])
    if args.plan_only:os.environ['CUDA_VISIBLE_DEVICES']=''
    destination=args.output_dir/'reference.json'
    if args.output_dir.exists() and any(args.output_dir.iterdir()):parser.error('Use a fresh independent reference output directory')
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig,AutoModelForImageTextToText,AutoTokenizer,GenerationConfig
    checkpoint=load_checkpoint(model_path)
    torch.set_num_threads(4)
    torch.manual_seed(1337)
    config=AutoConfig.from_pretrained(model_path,local_files_only=True)
    config.text_config.mtp_num_hidden_layers=0
    config.text_config._experts_implementation='eager'
    config._attn_implementation='sdpa'
    config.text_config._attn_implementation='sdpa'
    config.vision_config._attn_implementation='sdpa'
    started=time.time()
    with torch.device('cpu'),init_empty_weights(include_buffers=False):
        model=AutoModelForImageTextToText.from_config(config,dtype=torch.bfloat16)
    if any(not p.is_meta for p in model.parameters()):raise RuntimeError('All HF parameters must initialize on meta')
    shapes={name:list(param.shape) for name,param in model.named_parameters()}
    if shapes!=checkpoint.text|checkpoint.visual or len(shapes)!=1371:
        raise RuntimeError('Actual HF meta shapes differ from the complete 1038 text+333 visual official parameters')
    fp32_names,declared=declared_fp32_policy(model,checkpoint.gdn)
    if len(fp32_names)!=90:raise RuntimeError('Expected exactly90 small GDN FP32 scalar sources')
    plan=parameter_plan(shapes,fp32_names,device_map)
    if plan['parameter_numel']!=396802360816 or set(plan['per_device'])!={f'cuda:{i}' for i in range(4)}:
        raise RuntimeError('The load plan does not reconstruct the full model on four GPUs')
    plan.update(checkpoint_identity=identity(),device_map=device_map,hf_declared_fp32_modules=declared,
                meta_model_class=type(model).__name__,cuda_initialized=torch.cuda.is_initialized(),gpu_stages_executed=False)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.plan_only:
        if torch.cuda.is_initialized():raise RuntimeError('The CPU-only plan unexpectedly initialized CUDA')
        write_json(args.output_dir/'load-plan.json',plan)
        print('HF_BOUNDED_CPU_PLAN',json.dumps({k:v for k,v in plan.items() if k!='parameters'}),flush=True)
        return
    if torch.cuda.device_count()!=4:raise RuntimeError('Exactly four visible GPUs are required')
    capacities=[]
    for i in range(4):
        free,total=torch.cuda.mem_get_info(i)
        needed=plan['per_device'][f'cuda:{i}']['parameter_bytes']
        if needed+HEADROOM_BYTES>free:raise RuntimeError(f'cuda:{i}: parameters+16GiB headroom exceed actual free memory')
        capacities.append(dict(index=i,gpu_uuid=str(torch.cuda.get_device_properties(i).uuid),total_bytes=total,
                               free_before_bytes=free,planned_parameter_bytes=needed,projected_free_after_parameters=free-needed))
    plan['actual_cuda_capacity']=capacities
    write_json(args.output_dir/'load-plan.json',plan)
    print('HF_BOUNDED_LOAD_PLAN',json.dumps(dict(per_device=plan['per_device'],actual_capacity=capacities)),flush=True)
    model,loader_proof=load_bounded(model,checkpoint,model_path,device_map,plan,args.output_dir)
    model.eval().requires_grad_(False)
    model.generation_config=GenerationConfig.from_pretrained(model_path,local_files_only=True)
    placements={}
    by_device=defaultdict(int)
    dtype_counts=defaultdict(int)
    for name,p in model.named_parameters():
        expected=torch.float32 if name in fp32_names else torch.bfloat16
        if p.is_meta or p.device.type!='cuda' or p.dtype!=expected or str(p.device)!=target_device(name,device_map):
            raise RuntimeError('Final actual HF parameter residency/dtype differs from plan: '+name)
        placements[name]=str(p.device)
        by_device[str(p.device)]+=p.numel()
        dtype_counts[str(p.dtype)]+=p.numel()
    if sum(by_device.values())!=396802360816 or set(by_device)!={f'cuda:{i}' for i in range(4)}:
        raise RuntimeError('Final CUDA parameters do not reconstruct the full official model')
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    print('HF_FULL_CHECKPOINT_LOADED',time.time()-started,json.dumps(by_device),flush=True)
    cases=[]
    input_device=torch.device('cuda:0')
    for question in QUESTIONS:
        prompt=tokenizer.apply_chat_template([{'role':'user','content':question}],tokenize=False,
                    add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer.encode(prompt,add_special_tokens=False)
        inputs=torch.tensor([ids],device=input_device)
        with torch.inference_mode():
            result=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),do_sample=False,
                                  max_new_tokens=12,use_cache=True,pad_token_id=tokenizer.pad_token_id)
            targets=result[0,len(ids):]
            teacher=result[:,:-1]
            output=model(input_ids=teacher,attention_mask=torch.ones_like(teacher),use_cache=False)
            logits=output.logits[0,len(ids)-1:].float()
            values=logits.log_softmax(dim=-1).gather(-1,targets.to(logits.device)[:,None]).squeeze(-1).tolist()
        if not values or len(values)!=len(targets) or not all(math.isfinite(value) for value in values):
            raise RuntimeError('Empty, malformed or nonfinite HF selected-token scores')
        cases.append(dict(question=question,prompt_token_ids=ids,output_token_ids=targets.tolist(),raw_log_probs=values,
                          decoded=tokenizer.decode(targets.tolist(),skip_special_tokens=False)))
        print('HF_REFERENCE_CASE',json.dumps(cases[-1]),flush=True)
        del result,targets,teacher,output,logits,inputs
    devices=[]
    for i in range(4):
        torch.cuda.synchronize(i)
        devices.append(dict(index=i,device=f'cuda:{i}',gpu_uuid=str(torch.cuda.get_device_properties(i).uuid),
            cuda_numel=by_device[f'cuda:{i}'],allocated_bytes=torch.cuda.memory_allocated(i),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(i),peak_reserved_bytes=torch.cuda.max_memory_reserved(i)))
    receipt=dict(schema='qwen35-397b-hf-cold-reference-v1',**identity(),model_path=str(model_path),dtype='bfloat16',mtp=False,
        device='layer_sharded_cuda',input_device=str(input_device),device_map=device_map,devices=devices,
        parameter_device_sha256=sha256(json.dumps(placements,sort_keys=True).encode()).hexdigest(),parameter_placements=placements,
        parameter_dtype_numel=dict(dtype_counts),model_class=type(model).__name__,cuda_numel=sum(by_device.values()),
        cases=cases,finite=True,peak_allocated_bytes=sum(row['peak_allocated_bytes'] for row in devices),
        bounded_loader_schema=loader_proof['schema'],dispatch_parameter_storage_unchanged=True,started_unix=started,finished_unix=time.time())
    write_json(destination,receipt)
    print('HF_REFERENCE_COMPLETE',len(cases),flush=True)


if __name__=='__main__':main()
