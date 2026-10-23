"""Compute each head's chronological prefix once, then return owner states.

Summary ownership is temporarily transposed from tokens to heads. The merge
retains the original left-to-right FP32 recurrence, including hi/lo splitting;
it does not use an associative tree or change summary precision.
"""
import torch
import torch.distributed as dist
import triton
import triton.language as tl

from .kcp_bm_pipeline import scratch_buffer, scope
from .kcp_merge import _metadata


@triton.jit
def _pack_heads(SRC, DST, N: tl.constexpr, H: tl.constexpr,
                LH: tl.constexpr, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*BLOCK + tl.arange(0,BLOCK).to(tl.int64)
    valid = i < N*H*SIZE
    element = i % SIZE
    head = i // SIZE % LH
    segment = i // (SIZE*LH) % N
    destination = i // (SIZE*LH*N)
    value = tl.load(SRC + (segment*H+destination*LH+head)*SIZE+element, valid, 0)
    tl.store(DST+i,value,valid)


@triton.jit
def _merge_heads(HM,H0,OUT,PHYSICAL,STARTS,COUNTS,SEQS,
                 H:tl.constexpr,FULL_H:tl.constexpr,HEAD_START:tl.constexpr,
                 K:tl.constexpr,V:tl.constexpr,BV:tl.constexpr):
    group,head=tl.program_id(0),tl.program_id(1)
    rows=tl.arange(0,K)
    cols=tl.program_id(2)*BV+tl.arange(0,BV)
    seq=tl.load(SEQS+group).to(tl.int64)
    start,count=tl.load(STARTS+group),tl.load(COUNTS+group)
    state=tl.load(H0+((seq*FULL_H+HEAD_START+head)*K+rows[:,None])*V+cols[None,:]).to(tl.float32)
    for j in range(count):
        physical=tl.load(PHYSICAL+start+j).to(tl.int64)
        tl.store(OUT+((physical*H+head)*K+rows[:,None])*V+cols[None,:],state)
        if j<count-1:
            base=HM+(physical*H+head)*K*(V+K)
            transition=tl.load(base+rows[:,None]*(V+K)+V+rows[None,:])
            after=tl.load(base+rows[:,None]*(V+K)+cols[None,:]).to(tl.float32)
            high=state.to(tl.bfloat16)
            low=(state-high.to(tl.float32)).to(tl.bfloat16)
            product=tl.dot(transition,high,out_dtype=tl.float32)
            product=tl.dot(transition,low,product,out_dtype=tl.float32)
            state=after+product+state


@triton.jit
def _unpack_states(SRC,DST,N:tl.constexpr,PAD:tl.constexpr,H:tl.constexpr,
                   LH:tl.constexpr,SIZE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK).to(tl.int64)
    valid=i<N*H*SIZE
    element=i%SIZE
    head=i//SIZE%H
    segment=i//(SIZE*H)
    source=((head//LH*PAD+segment)*LH+head%LH)*SIZE+element
    value=tl.load(SRC+source,valid,0)
    tl.store(DST+i,value,valid)


def exchange_merge_local(local,h0,physical,plan,*,group,local_segments):
    """Accept contiguous local BF16 summaries, return local FP32 start states."""
    world,rank=dist.get_world_size(group),dist.get_rank(group)
    padded,heads,dim,packed=local.shape
    assert local.dtype==torch.bfloat16 and local.is_contiguous()
    assert heads%world==0 and dim==128 and packed==256
    local_heads=heads//world;device=local.device
    send=scratch_buffer(group,(world,padded,local_heads,dim,packed),local.dtype,device,'head_bm_send')
    incoming=scratch_buffer(group,send.shape,local.dtype,device,'head_bm_receive')
    with scope('kcp_bm/head_summary_a2a'):
        _pack_heads[(triton.cdiv(send.numel(),512),)](local,send,N=padded,H=heads,LH=local_heads,SIZE=dim*packed,BLOCK=512)
        dist.all_to_all_single(incoming.flatten(),send.flatten(),group=group)
    states=scratch_buffer(group,(world,padded,local_heads,dim,dim),torch.float32,device,'head_prefix_states')
    returned=scratch_buffer(group,states.shape,torch.float32,device,'head_prefix_return')
    starts,counts,_,seqs=_metadata(tuple(plan['merge_starts']),tuple(plan['merge_counts']),
        (0,)*len(plan['merge_seq_ids']),tuple(plan['merge_seq_ids']),device.index,
        int(torch.cuda.current_stream(device).cuda_stream))
    with scope('kcp_bm/head_prefix_merge'):
        _merge_heads[(len(plan['merge_seq_ids']),local_heads,2)](
            incoming,h0,states,physical,starts,counts,seqs,H=local_heads,FULL_H=heads,
            HEAD_START=rank*local_heads,K=dim,V=dim,BV=64,num_warps=4,num_stages=2)
    with scope('kcp_bm/head_state_a2a'):
        dist.all_to_all_single(returned.flatten(),states.flatten(),group=group)
        out=h0.new_empty(local_segments,heads,dim,dim)
        _unpack_states[(triton.cdiv(out.numel(),512),)](returned,out,N=local_segments,
            PAD=padded,H=heads,LH=local_heads,SIZE=dim*dim,BLOCK=512)
    return out


def summary_head_prefix(k,w,u,gk,cu_seqlens,h0,physical,plan,*,group,local_segments):
    from .cula_prescan_strided import chunk_delta_rule_pre_scan
    local=scratch_buffer(group,(plan['padded_segments'],k.shape[2],k.shape[3],u.shape[3]+k.shape[3]),
                         torch.bfloat16,k.device,'head_bm_local')
    with scope('kcp_bm/head_prescan'):
        if local_segments<len(local):local[local_segments:].zero_()
        chunk_delta_rule_pre_scan(k=k,w=w,u=u,gk=gk,cu_seqlens_split=cu_seqlens,
            S_split=local_segments,chunk_size=64,out=local[:local_segments],identity_residual=True)
    return exchange_merge_local(local,h0,physical,plan,group=group,local_segments=local_segments)
