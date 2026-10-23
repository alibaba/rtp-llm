"""Peer copies into native TP-head checkpoint layout, retaining page indices."""
import math
import torch
import torch.distributed as dist
import triton
import triton.language as tl

_buffers={}

def output_buffer(shape,group,device):
    import torch.distributed._symmetric_memory as symm
    key=(id(group),device.index,int(torch.cuda.current_stream(device).cuda_stream))
    numel=math.prod(shape);entry=_buffers.get(key)
    if entry is None or entry[0].numel()<numel:
        backend=symm.get_backend(device)
        if backend is None:symm.set_backend('CUDA')
        elif backend!='CUDA':raise RuntimeError('Peer checkpoint requires CUDA symmetric memory')
        page_elements=(2*2**20)//4
        capacity=((numel+page_elements-1)//page_elements)*page_elements
        flat=symm.empty((capacity,),dtype=torch.float32,device=device)
        handle=symm.rendezvous(flat,group)
        ptrs=torch.tensor(handle.buffer_ptrs,dtype=torch.int64,device=device)
        entry=(flat,handle,ptrs);_buffers[key]=entry
    flat,handle,ptrs=entry
    return flat[:numel].view(shape),handle,ptrs


@triton.jit
def _invert(INDEX,INVERSE,COUNT:tl.constexpr,PAD:tl.constexpr,RANK:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    physical=tl.load(INDEX+i,i<COUNT,-1).to(tl.int64)
    valid=(i<COUNT)&(physical//PAD==RANK)
    tl.store(INVERSE+physical%PAD,i,valid)


@triton.jit
def _write(SRC,PTRS,INVERSE,COUNT:tl.constexpr,STATE:tl.constexpr,
           WORLD:tl.constexpr,RANK:tl.constexpr,BLOCK:tl.constexpr):
    pid=tl.program_id(0).to(tl.int64)
    owner=(pid%WORLD+RANK)%WORLD
    x=pid//WORLD*BLOCK+tl.arange(0,BLOCK).to(tl.int64)
    slot=x//STATE;valid=x<COUNT*STATE
    page=tl.load(INVERSE+slot,valid,-1).to(tl.int64)
    value=tl.load(SRC+(slot*WORLD+owner)*STATE+x%STATE,valid&(page>=0),0)
    target=tl.load(PTRS+owner).to(tl.pointer_type(tl.float32))
    tl.store(target+page*STATE+x%STATE,value,valid&(page>=0))


def return_checkpoints(local,index,output,padded,group):
    world,rank=dist.get_world_size(group),dist.get_rank(group)
    _,count,heads,k,v=local.shape
    assert local.dtype==output.dtype==torch.float32 and local.is_contiguous() and output.is_contiguous()
    if heads%world or output.numel()!=index.numel()*(heads//world)*k*v:
        raise ValueError('Checkpoint layout does not match the caller buffer')
    target,handle,ptrs=output_buffer(output.shape,group,output.device)
    inverse=torch.full((padded,),-1,device=local.device,dtype=torch.int32)
    _invert[(triton.cdiv(index.numel(),256),)](index,inverse,COUNT=index.numel(),PAD=padded,RANK=rank,BLOCK=256)
    # Reuse is safe even when a rank's previous consumer was delayed.
    handle.barrier(channel=0)
    state=(heads//world)*k*v
    _write[(triton.cdiv(count*state,1024)*world,)](
        local,ptrs,inverse,COUNT=count,STATE=state,WORLD=world,RANK=rank,BLOCK=1024)
    handle.barrier(channel=0)
    if output.data_ptr()!=target.data_ptr():output.copy_(target)
