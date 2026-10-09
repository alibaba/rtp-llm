"""Experimental original-order expert ID permutation, integer/memory only."""

import torch
import triton
import triton.language as tl


@triton.jit
def _small_sort_kernel(IDS,OUT,COUNT:tl.constexpr):
    lane=tl.arange(0,32)
    valid=lane<COUNT
    keys=tl.load(IDS+lane,mask=valid,other=0)
    values=tl.where(valid,lane,0).to(tl.int64)
    for level in tl.static_range(1,6):
        for step in tl.static_range(level-1,-1,-1):
            stride=1<<step
            peer=lane^stride
            peer_key=tl.gather(keys,peer,axis=0)
            peer_value=tl.gather(values,peer,axis=0)
            peer_valid=tl.gather(valid,peer,axis=0)
            lower=(lane&stride)==0
            ka,kb=tl.where(lower,keys,peer_key),tl.where(lower,peer_key,keys)
            va,vb=tl.where(lower,valid,peer_valid),tl.where(lower,peer_valid,valid)
            ascending_cmp=((ka<kb)&va)|~vb
            if level<5:
                descending=(lane&(1<<level))!=0
            else:
                descending=tl.full((32,),False,tl.int1)
            exchange=ascending_cmp==descending
            keys=tl.where(exchange,peer_key,keys)
            values=tl.where(exchange,peer_value,values)
            valid=tl.where(exchange,peer_valid,valid)
    tl.store(OUT+lane,values,mask=lane<COUNT)


@triton.jit
def _medium_sort_kernel(IDS,OUT,COUNT:tl.constexpr,BLOCK:tl.constexpr,
                        PACK32:tl.constexpr):
    lane=tl.arange(0,BLOCK)
    expert=tl.load(IDS+lane,mask=lane<COUNT,other=0)
    if PACK32:
        encoded=tl.where(lane<COUNT,(expert.to(tl.int32)<<16)|lane,0x7FFFFFFF)
    else:
        encoded=tl.where(lane<COUNT,(expert<<32)|lane.to(tl.int64),0x7FFFFFFFFFFFFFFF)
    ordered=tl.sort(encoded,descending=False)
    if PACK32:
        tl.store(OUT+lane,(ordered&0xFFFF).to(tl.int64),mask=lane<COUNT)
    else:
        tl.store(OUT+lane,ordered&0xFFFFFFFF,mask=lane<COUNT)


@triton.jit
def _integer_max(a,b):
    return tl.maximum(a,b)


@triton.jit
def _chunk_sorted_rank_counts_kernel(IDS,RANKS,COUNTS,COUNT,CHUNKS,
                                     CHUNK:tl.constexpr):
    chunk=tl.program_id(0).to(tl.int64)
    lane=tl.arange(0,CHUNK)
    index=chunk*CHUNK+lane
    valid=index<COUNT
    ids=tl.load(IDS+index,mask=valid,other=0)
    packed=tl.where(valid,(ids<<32)|index,0x7FFFFFFFFFFFFFFF)
    ordered=tl.sort(packed,descending=False)
    experts=ordered>>32
    original=ordered&0xFFFFFFFF
    prior=tl.gather(experts,tl.maximum(lane-1,0),axis=0)
    starts=tl.where((lane==0)|(experts!=prior),lane,0)
    group_start=tl.associative_scan(starts,axis=0,combine_fn=_integer_max)
    local_rank=lane-group_start+1
    tl.store(RANKS+original,local_rank,mask=original<COUNT)
    histogram=tl.histogram(ids.to(tl.int32),128,mask=valid)
    all_experts=tl.arange(0,128)
    tl.store(COUNTS+all_experts.to(tl.int64)*CHUNKS+chunk,histogram)


@triton.jit
def _chunk_rank_counts_kernel(IDS,RANKS,COUNTS,COUNT,
                              CHUNKS,CHUNK:tl.constexpr,EXPERT_TILE:tl.constexpr,
                              TRANSPOSE:tl.constexpr):
    chunk=tl.program_id(0).to(tl.int64)
    group=tl.program_id(1)
    index=chunk*CHUNK+tl.arange(0,CHUNK)
    expert=group*EXPERT_TILE+tl.arange(0,EXPERT_TILE)
    ids=tl.load(IDS+index,mask=index<COUNT,other=0)
    same=(expert[:,None]==ids[None,:])&(index[None,:]<COUNT)
    flags=same.to(tl.int32)
    prefix=tl.associative_scan(flags,axis=1,combine_fn=_integer_add)
    rank=tl.sum(tl.where(same,prefix,0),axis=0)
    totals=tl.sum(flags,axis=1)
    tl.store(RANKS+index,rank,mask=(index<COUNT)&(ids//EXPERT_TILE==group))
    if TRANSPOSE:
        tl.store(COUNTS+expert.to(tl.int64)*CHUNKS+chunk,totals)
    else:
        tl.store(COUNTS+chunk*128+expert,totals)


@triton.jit
def _integer_add(a,b):
    return a+b


@triton.jit
def _chunk_prefix_kernel(COUNTS,PREFIX,TOTALS,CHUNKS,
                         BLOCK:tl.constexpr,TRANSPOSE:tl.constexpr):
    expert=tl.program_id(0)
    chunk=tl.arange(0,BLOCK)
    if TRANSPOSE:
        offset=expert.to(tl.int64)*CHUNKS+chunk
    else:
        offset=chunk*128+expert
    values=tl.load(COUNTS+offset,mask=chunk<CHUNKS,other=0)
    prefix=tl.associative_scan(values,axis=0,combine_fn=_integer_add)
    tl.store(PREFIX+offset,prefix-values,mask=chunk<CHUNKS)
    total=tl.sum(values,axis=0)
    tl.store(TOTALS+expert,total)


@triton.jit
def _expert_base_kernel(TOTALS,BASE):
    expert=tl.arange(0,128)
    count=tl.load(TOTALS+expert)
    offsets=tl.associative_scan(count,axis=0,combine_fn=_integer_add)
    tl.store(BASE+expert,offsets-count)


@triton.jit
def _scatter_permutation_kernel(IDS,RANKS,PREFIX,BASE,TOTALS,OUT,COUNT,CHUNKS,
                                CHUNK:tl.constexpr,BLOCK:tl.constexpr,
                                TRANSPOSE:tl.constexpr,FUSE_BASE:tl.constexpr):
    index=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
    expert=tl.load(IDS+index,mask=index<COUNT,other=0)
    local=tl.load(RANKS+index,mask=index<COUNT,other=0)
    if TRANSPOSE:
        prefix_index=expert*CHUNKS+index//CHUNK
    else:
        prefix_index=(index//CHUNK)*128+expert
    before=tl.load(PREFIX+prefix_index,mask=index<COUNT,other=0)
    if FUSE_BASE:
        all_experts=tl.arange(0,128)
        counts=tl.load(TOTALS+all_experts)
        bases=tl.associative_scan(counts,axis=0,combine_fn=_integer_add)-counts
        base=tl.gather(bases,expert.to(tl.int32),axis=0)
    else:
        base=tl.load(BASE+expert,mask=index<COUNT,other=0)
    destination=base.to(tl.int64)+before+local-1
    tl.store(OUT+destination,index,mask=index<COUNT)


def expert_permutation(ids,variant="initial"):
    if (
        not ids.is_cuda or ids.dtype!=torch.int64 or ids.dim()!=1
        or not ids.is_contiguous() or not 1<=ids.numel()<=1048576
        or variant not in ("initial","medium","coalesced","coalesced-fused","local-sort","packed32",
                           "packed32-w4","packed32-w16","packed32-w32","adaptive-packed32")
    ):
        return None
    # Caller provides valid0..127expert IDs from the original top-k. No host
    # extraction/repair or alternate selection happens here.
    count=ids.numel()
    output=torch.empty_like(ids)
    if count<=32:
        _small_sort_kernel[(1,)](ids,output,count,num_warps=4)
    elif count<=(1024 if variant=="local-sort" else 4096) and variant!="initial":
        block=triton.next_power_of_2(count)
        warps=4 if block<=1024 else 8
        if block>1024 and variant.startswith("packed32-w"):
            warps=int(variant.removeprefix("packed32-w"))
        if variant=="adaptive-packed32" and block>1024:
            warps=16 if block==2048 else 32
        packed=variant.startswith("packed32") or variant=="adaptive-packed32"
        _medium_sort_kernel[(1,)](ids,output,count,block,packed,num_warps=warps)
    else:
        chunks=triton.cdiv(count,256)
        ranks=torch.empty((count,),dtype=torch.int32,device=ids.device)
        counts=torch.empty((chunks,128),dtype=torch.int32,device=ids.device)
        prefix=torch.empty_like(counts)
        totals=torch.empty((128,),dtype=torch.int32,device=ids.device)
        base=torch.empty_like(totals)
        packed=variant.startswith("packed32") or variant=="adaptive-packed32"
        transpose=variant in ("coalesced","coalesced-fused","local-sort") or packed
        fuse=variant in ("coalesced-fused","local-sort") or packed
        if variant=="local-sort" or packed:
            _chunk_sorted_rank_counts_kernel[(chunks,)](ids,ranks,counts,count,chunks,256,num_warps=4)
        else:
            _chunk_rank_counts_kernel[(chunks,8)](ids,ranks,counts,count,chunks,256,16,transpose,num_warps=8)
        _chunk_prefix_kernel[(128,)](counts,prefix,totals,chunks,triton.next_power_of_2(chunks),transpose,num_warps=8 if chunks>=1024 else 4)
        if not fuse:
            _expert_base_kernel[(1,)](totals,base,num_warps=4)
        _scatter_permutation_kernel[(triton.cdiv(count,512),)](ids,ranks,prefix,base,totals,output,count,chunks,256,512,transpose,fuse,num_warps=4)
    return output
