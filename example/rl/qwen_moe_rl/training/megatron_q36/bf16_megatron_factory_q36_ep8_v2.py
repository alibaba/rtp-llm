"""35B TP4/EP8/ETP1 nonshared BF16/SR factory; private candidate.

The worker supplies every parameter's independently qualified logical coordinates.
This uses stable local-shard random keys, not topology-independent bit identity.
"""
import functools
import hashlib
import re
import torch
import torch.distributed as dist
from bf16_fixed_tp4_coordinates_candidate import FixedTP4Coordinates
from bf16_sr_coordinates_candidate import SRCoordinates
from bf16_streaming_optimizer_coords_candidate import StreamingBf16AdamWCoordinates


def install():
    import megatron.core.optimizer as factory
    from megatron.core import parallel_state as ps
    import bf16_megatron_factory_nonshared_candidate as base
    if getattr(factory,'_q38_candidate_installed',False):raise RuntimeError('factory already patched')
    base.StreamingBf16AdamW=StreamingBf16AdamWCoordinates
    base.install();installed=factory.get_megatron_optimizer

    @functools.wraps(installed)
    def get(config,model_chunks,*args,**kwargs):
        if not getattr(config,'q38_bf16_hdo',False):return installed(config,model_chunks,*args,**kwargs)
        records=[];seen=set();names=set()
        def world_gate(error,label):
            flag=torch.tensor(int(error is None),device=torch.cuda.current_device(),dtype=torch.int32)
            dist.all_reduce(flag,op=dist.ReduceOp.MIN)
            if not bool(flag.item()):raise RuntimeError('all-rank fixed-TP4 SR preflight failed: '+label) from error
        local_error=None
        try:
            if not getattr(config,'q38_nonsharded_bf16',False):raise ValueError('nonsharded BF16 required')
            base._check_config(config)
            if (dist.get_world_size()!=8 or ps.get_tensor_model_parallel_world_size()!=4
                or ps.get_expert_model_parallel_world_size()!=8
                or dist.get_world_size(ps.get_expert_tensor_parallel_group())!=1):
                raise ValueError('candidate requires TP4/EP8/ETP1 on eight ranks')
            if dist.get_process_group_ranks(ps.get_expert_tensor_parallel_group()) != [dist.get_rank()]:
                raise ValueError('ETP1 group must contain exactly this world rank')
            bindings=getattr(config,'q38_sr_bindings',None)
            if not isinstance(bindings,dict):raise ValueError('explicit SR bindings required')
            for chunk in model_chunks:
                if (chunk.ddp_config.grad_reduce_in_fp32 or chunk.ddp_config.use_distributed_optimizer
                    or getattr(chunk.ddp_config,'use_megatron_fsdp',False)):
                    raise ValueError('requires nonsharded BF16 DDP')
                for _,p in chunk.named_parameters():
                    if not p.requires_grad or id(p) in seen:continue
                    seen.add(id(p));coord=bindings.get(id(p))
                    if not isinstance(coord,SRCoordinates):raise ValueError('missing reviewed SR binding')
                    coord.validate()
                    if tuple(p.shape)!=coord.local_shape or not p.is_contiguous():
                        raise ValueError('actual local parameter shape/stride mismatch')
                    if coord.canonical_name in names:raise ValueError('duplicate logical parameter name')
                    names.add(coord.canonical_name)
                    expert=not getattr(p,'allreduce',True)
                    parallel=bool(getattr(p,'tensor_model_parallel',False))
                    if expert:
                        found=re.search(r'/global-expert-(\d+)/',coord.canonical_name)
                        ep_rank=ps.get_expert_model_parallel_rank()
                        if (found is None or not ep_rank*32 <= int(found.group(1)) < (ep_rank+1)*32
                            or coord.axis is not None):
                            raise ValueError('expert SR identity does not match live EP8 ownership')
                    else:
                        if not isinstance(coord,FixedTP4Coordinates):raise ValueError('dense TP4 binding required')
                        group=ps.get_tensor_model_parallel_group()
                        if dist.get_world_size(group)!=4 or coord.group_rank!=dist.get_rank(group):
                            raise ValueError('dense binding does not match actual TP group')
                        if parallel!=coord.sharded:raise ValueError('dense binding category differs from TP metadata')
                        if parallel and (coord.metadata_axis!=getattr(p,'partition_dim',None)
                            or coord.metadata_stride!=getattr(p,'partition_stride',None)):
                            raise ValueError('binding differs from actual TP partition metadata')
                    records.append((p,coord,expert))
            if set(bindings)!=seen:raise ValueError('missing or unowned SR bindings')
        except Exception as exc:local_error=exc
        world_gate(local_error,'local binding/config')
        for expert in (False,True):
            group=(ps.get_expert_data_parallel_group() if expert
                   else ps.get_data_parallel_group(with_context_parallel=True))
            schema=(sorted((c.canonical_name,c.local_shape) for _,c,e in records if e) if expert else
                    sorted((c.logical_name,c.canonical_name,c.local_shape,c.sharded,c.group_size,
                            c.group_rank,c.metadata_axis,c.metadata_stride) for _,c,e in records if not e))
            sig=hashlib.sha256(repr(schema).encode()).hexdigest();all_sigs=[None]*dist.get_world_size(group)
            dist.all_gather_object(all_sigs,sig,group=group)
            world_gate(None if len(set(all_sigs))==1 else ValueError('DP SR mismatch'),
                       'expert DP schema' if expert else 'dense DP schema')
        # EP8 ranks own different experts; compare only the dense TP schema.
        schema=sorted((c.logical_name,c.local_shape,c.sharded,c.group_size,c.metadata_axis,c.metadata_stride)
                      for _,c,expert in records if not expert)
        sig=hashlib.sha256(repr(schema).encode()).hexdigest();group=ps.get_tensor_model_parallel_group()
        all_sigs=[None]*dist.get_world_size(group);dist.all_gather_object(all_sigs,sig,group=group)
        world_gate(None if len(set(all_sigs))==1 else ValueError('TP logical schema mismatch'),'TP schema')
        for p,coord,_ in records:p._q38_sr_coordinates=coord
        return installed(config,model_chunks,*args,**kwargs)

    factory.get_megatron_optimizer=get
    factory._q38_sr_fixed_tp4_installed=True
