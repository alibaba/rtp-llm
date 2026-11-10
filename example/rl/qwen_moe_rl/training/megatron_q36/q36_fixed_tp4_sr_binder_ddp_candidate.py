"""Bind fixed TP4 SR keys only from an exact, independently reviewed manifest.

This module does not generate, approve or extrapolate a manifest. All ranks
must call `bind_all_ranks` in the same order after MCore group initialization.
It is a private integration candidate; real 35B source/live coverage is pending.
"""
import hashlib
import json
from pathlib import Path
import torch
import torch.distributed as dist
from megatron.core import parallel_state as ps
from megatron.core.distributed import DistributedDataParallel as MCoreDDP
from megatron.core.transformer.module import Float16Module
from bf16_fixed_tp4_coordinates_candidate import make_fixed_tp4_coordinates


def _bind_local(model_chunks,manifest_path,expected_sha256,allow_test_only=False):
    path=Path(manifest_path)
    if path.is_symlink() or not path.is_file():raise ValueError('qualified manifest must be a regular file')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('manifest SHA mismatch')
    manifest=json.loads(raw)
    if (manifest.get('schema')!='q36-fixed-tp4-sr-provider-ddp-parameters-v2'
        or manifest.get('tp')!=4 or manifest.get('ep')!=1 or manifest.get('etp')!=4):
        raise ValueError('manifest does not describe exact TP4/EP1/ETP4')
    if manifest.get('binding_scope')!='full-provider-ddp-live-source-verified' and not allow_test_only:
        raise ValueError('test or extrapolated parameter manifest is not authorized for training')
    if (ps.get_tensor_model_parallel_world_size()!=4 or ps.get_expert_model_parallel_world_size()!=1
        or dist.get_world_size(ps.get_expert_tensor_parallel_group())!=4):
        raise ValueError('actual MCore parallel topology differs from manifest')
    entries=manifest.get('parameters')
    if not isinstance(entries,list):raise ValueError('manifest parameter list missing')
    expected={x['name']:x for x in entries}
    if len(expected)!=len(entries):raise ValueError('duplicate manifest names')
    params={};seen=set()
    for i,chunk in enumerate(model_chunks):
        if not isinstance(chunk,MCoreDDP):raise ValueError('expected actual MCore DDP wrapper')
        unwrapped=chunk.module
        if isinstance(unwrapped,Float16Module):unwrapped=unwrapped.module
        if isinstance(unwrapped,(MCoreDDP,Float16Module)):
            raise ValueError('unexpected nested model wrappers')
        wrapped_ids={id(p) for p in chunk.parameters() if p.requires_grad}
        unwrapped_ids={id(p) for p in unwrapped.parameters() if p.requires_grad}
        if wrapped_ids!=unwrapped_ids:raise ValueError('wrapper changes trainable parameter identity')
        for name,p in unwrapped.named_parameters():
            if not p.requires_grad or id(p) in seen:continue
            seen.add(id(p));key='chunk%d/%s'%(i,name)
            if key in params:raise ValueError('duplicate actual parameter name')
            params[key]=p
    if not params or set(params)!=set(expected):
        raise ValueError('reviewed manifest and actual trainable parameter coverage differ')
    result={};logical_names=set()
    for name,p in params.items():
        item=expected[name];kind=item.get('kind');logical=item.get('logical_name')
        if kind not in ('replicated','tp_shard') or not isinstance(logical,str) or not logical:
            raise ValueError('unreviewed category or logical name: '+name)
        if logical in logical_names:raise ValueError('duplicate logical name: '+logical)
        logical_names.add(logical)
        if p.device.type!='cuda' or p.dtype!=torch.bfloat16 or not p.is_contiguous():
            raise ValueError('expected contiguous CUDA BF16 trainable parameter: '+name)
        if (list(p.shape)!=item.get('shape') or list(p.stride())!=item.get('stride')
            or p.storage_offset()!=item.get('storage_offset')):
            raise ValueError('actual parameter geometry differs from reviewed manifest: '+name)
        attrs=item.get('actual_attributes');present=item.get('actual_attributes_present')
        keys=('tensor_model_parallel','partition_dim','partition_stride','allreduce','sequence_parallel')
        if (not isinstance(attrs,dict) or not isinstance(present,dict)
            or set(attrs)!=set(keys) or set(present)!=set(keys)):
            raise ValueError('exact reviewed TP/DP attributes missing: '+name)
        for key in keys:
            exists=hasattr(p,key)
            if type(present[key]) is not bool or exists!=present[key]:
                raise ValueError('actual parameter attribute presence differs: '+name+' '+key)
            if (getattr(p,key,None) if exists else None)!=attrs[key]:
                raise ValueError('actual parameter attribute value differs: '+name+' '+key)
        expert=not bool(getattr(p,'allreduce',True))
        group=ps.get_expert_tensor_parallel_group() if expert else ps.get_tensor_model_parallel_group()
        group_rank=dist.get_rank(group)
        if kind=='tp_shard':
            axis=attrs['partition_dim'];stride=attrs['partition_stride']
            if (not bool(getattr(p,'tensor_model_parallel',False)) or type(axis) is not int
                or (stride is not None and (type(stride) is not int or stride<1))
                or getattr(p,'partition_dim',None)!=axis
                or getattr(p,'partition_stride',None)!=stride):
                raise ValueError('actual TP shard metadata mismatch: '+name)
            result[id(p)]=make_fixed_tp4_coordinates(logical,tuple(p.shape),sharded=True,
                                                        group_rank=group_rank,axis=axis,stride=stride)
        else:
            if (bool(getattr(p,'tensor_model_parallel',False))
                or not present['partition_dim'] or attrs['partition_dim']!=-1
                or not present['partition_stride'] or attrs['partition_stride']!=1):
                raise ValueError('replica has TP shard metadata: '+name)
            result[id(p)]=make_fixed_tp4_coordinates(logical,tuple(p.shape),sharded=False,group_rank=group_rank)
    return result


def bind_all_ranks(model_chunks,manifest_path,expected_sha256,*,allow_test_only=False):
    error=None;bindings=None
    try:bindings=_bind_local(model_chunks,manifest_path,expected_sha256,allow_test_only)
    except Exception as exc:error=exc
    flag=torch.tensor(int(error is None),device=torch.cuda.current_device(),dtype=torch.int32)
    dist.all_reduce(flag,op=dist.ReduceOp.MIN)
    if not bool(flag.item()):raise RuntimeError('all-rank 35B fixed-TP4 SR manifest binding failed') from error
    return bindings
