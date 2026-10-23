"""Optional transport filtering for checkpoints the native cache publishes.

Local dense checkpoints and the generic dense-output API are retained. A caller
supplying publication masks declares that unselected output slots are unused.
"""
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from .kcp_peer_checkpoint import output_buffer,_invert,return_checkpoints as dense_return

@triton.jit
def _mask(MASK,INDEX,SEQUENCE,PAGE,MAP,N:tl.constexpr,MAX_PAGES:tl.constexpr,
          BLOCKS:tl.constexpr,ROW_STRIDE:tl.constexpr,COL_STRIDE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=i<N
    seq=tl.load(SEQUENCE+i,valid,0).to(tl.int64)
    page=tl.load(PAGE+i,valid,-1).to(tl.int64)
    block=tl.load(MAP+seq*ROW_STRIDE+page*COL_STRIDE,valid&(page>=0)&(page<MAX_PAGES),-1)
    checkpoint=tl.load(INDEX+i,valid,0).to(tl.int64)
    tl.store(MASK+checkpoint,1,valid&(block>0)&(block<BLOCKS))

def publication_masks(metadata,block_map,blocks,group):
    local=torch.zeros(metadata.total_checkpoints,device=block_map.device,dtype=torch.uint8)
    n=metadata.store_checkpoint_indices.numel()
    if n:
        _mask[(triton.cdiv(n,256),)](local,metadata.store_checkpoint_indices,
            metadata.store_sequence_indices,metadata.store_page_indices,block_map,
            N=n,MAX_PAGES=block_map.shape[1],BLOCKS=blocks,
            ROW_STRIDE=block_map.stride(0),COL_STRIDE=block_map.stride(1),BLOCK=256)
    masks=torch.empty(dist.get_world_size(group)*local.numel(),device=local.device,dtype=local.dtype)
    dist.all_gather_into_tensor(masks,local,group=group)
    return masks.view(dist.get_world_size(group),local.numel())

@triton.jit
def _write(SRC,PTRS,INVERSE,MASK,COUNT:tl.constexpr,STATE:tl.constexpr,
           PAGES:tl.constexpr,WORLD:tl.constexpr,RANK:tl.constexpr,BLOCK:tl.constexpr):
    pid=tl.program_id(0).to(tl.int64)
    owner=(pid%WORLD+RANK)%WORLD
    tl.static_assert(STATE%BLOCK==0)
    offset=pid//WORLD*BLOCK
    slot=offset//STATE
    page=tl.load(INVERSE+slot,slot<COUNT,-1).to(tl.int64)
    needed=tl.load(MASK+owner*PAGES+page,(slot<COUNT)&(page>=0),0)>0
    if needed:
        x=offset+tl.arange(0,BLOCK).to(tl.int64)
        value=tl.load(SRC+(slot*WORLD+owner)*STATE+x%STATE)
        target=tl.load(PTRS+owner).to(tl.pointer_type(tl.float32))
        tl.store(target+page*STATE+x%STATE,value)

def return_checkpoints(local,index,output,padded,group,publish_masks=None):
    if publish_masks is None:return dense_return(local,index,output,padded,group)
    world,rank=dist.get_world_size(group),dist.get_rank(group)
    _,count,heads,k,v=local.shape
    assert local.dtype==output.dtype==torch.float32 and local.is_contiguous() and output.is_contiguous()
    assert publish_masks.shape==(world,index.numel()) and publish_masks.dtype==torch.uint8
    assert heads%world==0 and output.numel()==index.numel()*(heads//world)*k*v
    target,handle,ptrs=output_buffer(output.shape,group,output.device)
    inverse=torch.full((padded,),-1,device=local.device,dtype=torch.int32)
    _invert[(triton.cdiv(index.numel(),256),)](index,inverse,COUNT=index.numel(),PAD=padded,RANK=rank,BLOCK=256)
    handle.barrier(channel=0)
    state=(heads//world)*k*v
    _write[(triton.cdiv(count*state,1024)*world,)](local,ptrs,inverse,publish_masks,
        COUNT=count,STATE=state,PAGES=index.numel(),WORLD=world,RANK=rank,BLOCK=1024)
    handle.barrier(channel=0)
    if output.data_ptr()!=target.data_ptr():output.copy_(target)
