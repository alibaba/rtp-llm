"""Fixed-split predecessor-to-successor halos; local history stays local."""
from functools import lru_cache
import torch
import torch.distributed as dist
from .kcp_plan import make_plan
from .kcp_conv_halo import halo_plan, _pack_tails
from .kcp_bm_pipeline import gather_buffer


@lru_cache(maxsize=128)
def directed_plan(cu_values,world,page,rank,history,device_index,stream):
    plan=make_plan(cu_values,world,page)
    kinds,indices,_,rows=halo_plan(cu_values,world,page,rank,history)
    chunks=plan['chunks'];slots={}
    for owner in range(world):
        owned=sorted((c for c in chunks if c[3]==owner),key=lambda c:c[4])
        assert len(owned)*history==rows
        for slot,chunk in enumerate(owned):slots[chunk[0]]=owner*len(owned)+slot
    edges=[(a[3],b[3],a) for a,b in zip(chunks,chunks[1:])
           if a[2]==b[2] and a[3]!=b[3]]
    sends=sorted((e for e in edges if e[0]==rank),key=lambda e:(e[1],e[2][0]))
    recvs=sorted((e for e in edges if e[1]==rank),key=lambda e:(e[0],e[2][0]))
    lookup={slots[e[2][0]]*history+j:i*history+j
            for i,e in enumerate(recvs) for j in range(history)}
    device=torch.device('cuda',device_index)
    send_index=torch.tensor([e[2][4]+e[2][1]-e[2][0]-history+j
        for e in sends for j in range(history)],device=device,dtype=torch.int64)
    history_index=torch.tensor([lookup[i] if k==1 else i for k,i in zip(kinds,indices)],
        device=device,dtype=torch.int64)
    send_rows=tuple(sum(e[1]==r for e in sends)*history for r in range(world))
    recv_rows=tuple(sum(e[0]==r for e in recvs)*history for r in range(world))
    assert (sum(send_rows)+sum(recv_rows))<=world*rows
    return send_index,history_index,send_rows,recv_rows,rows


def exchange_halo_directed(mixed,metadata,group):
    send_index,history_index,send_rows,recv_rows,rows=metadata
    width=mixed.shape[1]
    # Keep the existing workspace capacity so enabling this transport does not
    # introduce another large allocation near the configured memory budget.
    workspace=gather_buffer(group,(rows,width),mixed.dtype,mixed.device,
                            'conv_halo_chunks').flatten(0,1)
    nsend,nrecv=sum(send_rows),sum(recv_rows)
    outgoing=workspace[:nsend];incoming=workspace[nsend:nsend+nrecv]
    if nsend:
        import triton
        _pack_tails[(triton.cdiv(outgoing.numel(),512),)](mixed,send_index,outgoing,
            ROWS=nsend,WIDTH=width,BLOCK=512)
    dist.all_to_all_single(incoming.flatten(),outgoing.flatten(),
        output_split_sizes=[n*width for n in recv_rows],
        input_split_sizes=[n*width for n in send_rows],group=group)
    return incoming,history_index
