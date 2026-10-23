"""Experimental same-node prefix with CUDA symmetric memory peer access."""
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from .kcp_bm_pipeline import scope
from .kcp_merge import _metadata

_workspaces={}

def workspace(group,padded,heads,dim,device):
    import torch.distributed._symmetric_memory as symm
    key=(id(group),device.index,int(torch.cuda.current_stream(device).cuda_stream),padded,heads,dim)
    if key not in _workspaces:
        backend=symm.get_backend(device)
        if backend is None:symm.set_backend('CUDA')
        elif backend!='CUDA':raise RuntimeError('Remote prefix requires the CUDA symmetric memory backend')
        local=symm.empty((padded,heads,dim,2*dim),device=device,dtype=torch.bfloat16)
        output=symm.empty((padded,heads,dim,dim),device=device,dtype=torch.float32)
        hm_handle=symm.rendezvous(local,group);out_handle=symm.rendezvous(output,group)
        hm_ptrs=torch.tensor(hm_handle.buffer_ptrs,device=device,dtype=torch.int64)
        out_ptrs=torch.tensor(out_handle.buffer_ptrs,device=device,dtype=torch.int64)
        _workspaces[key]=(local,output,hm_handle,out_handle,hm_ptrs,out_ptrs)
    return _workspaces[key]


@triton.jit
def _merge_remote(HM_PTRS,OUT_PTRS,H0,PHYSICAL,STARTS,COUNTS,SEQS,
                  H:tl.constexpr,LH:tl.constexpr,HEAD_START:tl.constexpr,
                  PAD:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BV:tl.constexpr):
    group,local_head=tl.program_id(0),tl.program_id(1)
    head=HEAD_START+local_head
    rows=tl.arange(0,K);cols=tl.program_id(2)*BV+tl.arange(0,BV)
    seq=tl.load(SEQS+group).to(tl.int64)
    start,count=tl.load(STARTS+group),tl.load(COUNTS+group)
    state=tl.load(H0+((seq*H+head)*K+rows[:,None])*V+cols[None,:]).to(tl.float32)
    for j in range(count):
        physical=tl.load(PHYSICAL+start+j).to(tl.int64)
        owner=physical//PAD;slot=physical%PAD
        target=tl.load(OUT_PTRS+owner).to(tl.pointer_type(tl.float32))
        tl.store(target+((slot*H+head)*K+rows[:,None])*V+cols[None,:],state)
        if j<count-1:
            source=tl.load(HM_PTRS+owner).to(tl.pointer_type(tl.bfloat16))
            base=source+(slot*H+head)*K*(V+K)
            transition=tl.load(base+rows[:,None]*(V+K)+V+rows[None,:])
            after=tl.load(base+rows[:,None]*(V+K)+cols[None,:]).to(tl.float32)
            high=state.to(tl.bfloat16);low=(state-high.to(tl.float32)).to(tl.bfloat16)
            product=tl.dot(transition,high,out_dtype=tl.float32)
            product=tl.dot(transition,low,product,out_dtype=tl.float32)
            state=after+product+state


def merge_remote(h0,physical,plan,*,group,local_segments):
    rank,world=dist.get_rank(group),dist.get_world_size(group)
    heads,dim=h0.shape[1:3];padded=plan['padded_segments']
    local,output,hm_handle,out_handle,hm_ptrs,out_ptrs=workspace(group,padded,heads,dim,h0.device)
    starts,counts,_,seqs=_metadata(tuple(plan['merge_starts']),tuple(plan['merge_counts']),
        (0,)*len(plan['merge_seq_ids']),tuple(plan['merge_seq_ids']),h0.device.index,
        int(torch.cuda.current_stream(h0.device).cuda_stream))
    with scope('kcp_bm/remote_prefix'):
        hm_handle.barrier(channel=0)
        _merge_remote[(len(plan['merge_seq_ids']),heads//world,2)](
            hm_ptrs,out_ptrs,h0,physical,starts,counts,seqs,H=heads,LH=heads//world,
            HEAD_START=rank*(heads//world),PAD=padded,K=dim,V=dim,BV=64,
            num_warps=4,num_stages=2)
        out_handle.barrier(channel=0)
    return output[:local_segments]


def summary_remote_prefix(k,w,u,gk,cu_seqlens,h0,physical,plan,*,group,local_segments):
    from .cula_prescan_strided import chunk_delta_rule_pre_scan
    local,*_=workspace(group,plan['padded_segments'],k.shape[2],k.shape[3],k.device)
    with scope('kcp_bm/remote_prescan'):
        chunk_delta_rule_pre_scan(k=k,w=w,u=u,gk=gk,cu_seqlens_split=cu_seqlens,
            S_split=local_segments,chunk_size=64,out=local[:local_segments],identity_residual=True)
    return merge_remote(h0,physical,plan,group=group,local_segments=local_segments)
