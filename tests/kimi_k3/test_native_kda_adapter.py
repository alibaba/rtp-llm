import json
import pytest
import torch
from rtp_llm.models_py.modules.kimi_k3.native_kda import plan_state_sequences, flash_kda_paged_prefill

def test_state_sequence_plans():
    # Exercise the actual CPU plan, including short, ragged and padded request batches.
    for batch in (1,2,3,7,8,9):
        lengths=[129+i for i in range(batch)]
        cu=[0]
        for n in lengths:cu.append(cu[-1]+n)
        plans=plan_state_sequences(cu,[0]*batch,[[2*i+1,2*i+2] for i in range(batch)],128)
        assert len(plans)==batch
        for i,p in enumerate(plans):
            assert [(s.start,s.end) for s in p.segments]==[(cu[i],cu[i]+128),(cu[i]+128,cu[i+1])]
    assert not plan_state_sequences([0,17],[0],[[0]],128)[0].segments
    for args in [([0,2],[-1],[[1]],128),([0,257],[0],[[1]],128),([0,129],[0],[[1,0]],128),([0,1],[128],[[0,2]],128)]:
        try:plan_state_sequences(*args)
        except ValueError:pass
        else:raise AssertionError('Invalid state plan accepted')



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flashkda_paged_state_layout():
    flash_kda = pytest.importorskip("flash_kda")
    torch.manual_seed(922)
    lengths=[4167,193,32];cu=[0,4167,4360,4392]
    heads,dim=2,128
    xs=[torch.randn(cu[-1],heads,dim,device='cuda',dtype=torch.bfloat16) for _ in range(4)]
    beta=torch.randn(cu[-1],heads,device='cuda',dtype=torch.bfloat16)
    alog=torch.randn(heads,device='cuda')*.1;bias=torch.randn(heads,dim,device='cuda')*.01
    # Non-contiguous outer stride matches RTP's combined conv/recurrent storage.
    storage=torch.randn(10,heads*dim*dim+512,device='cuda',dtype=torch.float32)
    cache=storage[:,:heads*dim*dim].view(10,heads,dim,dim)
    before=storage.clone()
    tables=[[1,2],[5,6],[0,0]]
    out=flash_kda_paged_prefill(*xs,beta,alog,bias,-5.,cache,cu,[0,4096,0],tables,4096)

    def native(request,end):
        start=cu[request];end=start+end;n=end-start
        q,k,v,g=[x[start:end].unsqueeze(0).contiguous() for x in xs]
        initial=torch.zeros(1,heads,dim,dim,device='cuda') if request==0 else before[5,:heads*dim*dim].view(heads,dim,dim).transpose(-1,-2).unsqueeze(0).contiguous()
        final=torch.empty_like(initial);output=torch.empty_like(v)
        workspace=torch.empty(flash_kda.get_workspace_size(n,heads,1),dtype=torch.uint8,device='cuda')
        torch.ops.flash_kda.fwd(q,k,v,g,beta[start:end].unsqueeze(0).contiguous(),dim**-.5,output,workspace,alog,bias,-5.,initial,final)
        return output[0],final[0].transpose(-1,-2)
    for request in (0,1):
        expected,state=native(request,lengths[request])
        torch.testing.assert_close(out[cu[request]:cu[request+1]],expected,rtol=0,atol=0)
        torch.testing.assert_close(cache[[2,6][request]],state,rtol=0,atol=0)
    _,boundary_state=native(0,4096)
    torch.testing.assert_close(cache[1],boundary_state,rtol=0,atol=0)
    assert torch.count_nonzero(out[cu[2]:]).item()==0
    for block in (0,3,4,5,7,8,9):torch.testing.assert_close(storage[block],before[block],rtol=0,atol=0)
    torch.testing.assert_close(storage[:,heads*dim*dim:],before[:,heads*dim*dim:],rtol=0,atol=0)
    print(json.dumps({'passed':True,'plan_batch_sizes':[1,2,3,7,8,9],'real_requests':2,'virtual_requests':1,'block_size':4096,'prefix_lengths':[0,4096,0],'ragged_lengths':lengths,'outputs_and_states':'bitwise identical to independent native full-request calls','unused_cache_and_conv_storage':'unchanged'},indent=2))
