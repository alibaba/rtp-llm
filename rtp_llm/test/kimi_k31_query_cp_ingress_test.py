"""Eight-rank CP producer/PageRR owner transport with real CPP permutations."""
import importlib.machinery,importlib.util,json,os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
from unittest.mock import patch

from rtp_llm.ops import ParallelismConfig,NcclCommConfig,CPRotateMethod,RoleType,compute_ops
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.modules.hybrid.query_layout import QueryLayout
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_page_rr_params import DsaPageRRParams
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_workspace import DsaWorkspaceOwner
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_ingress import DsaKVIngress
from rtp_llm.models_py.modules.base.cuda.bf16_indexer_op import Bf16DraftIndexerOp
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_page_rr_impl import PageRRSparseMlaOp, QueryCpLayout, flash_mla_sparse_fwd

def check_query_embedding(rank, parallel):
    from rtp_llm.models.kimi_k3.kimi_k31_weight import KimiK31Weight
    from rtp_llm.models_py.modules.base.common.embedding import Embedding
    from rtp_llm.model_loader.weight_module import CustomAtomicWeight
    from rtp_llm.utils.model_weight import W

    manifest = object.__new__(KimiK31Weight)
    manifest.prefill_query_cp = True
    embedding = next(x for x in manifest._global_weights() if x.name == W.embedding)
    assert isinstance(embedding, CustomAtomicWeight)
    weight = (torch.arange(256 * 128, device='cuda').reshape(256, 128) % 113).bfloat16()
    torch.testing.assert_close(embedding._get_split_func()(weight, tp=8, tp_rank=rank), weight, rtol=0, atol=0)
    manifest.prefill_query_cp = False
    decode = next(x for x in manifest._global_weights() if x.name == W.embedding)
    assert not isinstance(decode, CustomAtomicWeight)
    model = Embedding(SimpleNamespace(), parallel, weight, replicated=True)
    # Different tokens on every rank expose an accidental TP channel gather.
    ids = torch.tensor([rank, 64 + rank, 128 + rank], dtype=torch.int32, device='cuda')
    with patch('rtp_llm.models_py.modules.base.common.embedding.all_gather', side_effect=AssertionError('query CP embedding must not gather hidden channels')):
        actual = model(ids)
    torch.testing.assert_close(actual, weight[ids.long()], rtol=0, atol=0)
    print(json.dumps({'rank': rank, 'query_CP_embedding_full_hidden': True, 'different_rank_tokens': True}), flush=True)

def check_cached_chunks(rank, parallel, cpp):
    info = SimpleNamespace(
        prefill_actual_input_lengths_cpu=torch.tensor([17, 31], dtype=torch.int32),
        prefill_prefix_lengths_cpu=torch.tensor([8192, 4096], dtype=torch.int32),
        prefill_qkv_restore_indice=cpp.generate_qkv_restore_indices(torch.tensor([4,4],dtype=torch.int32),8).cuda(),
        prefill_qkv_padding_mask=cpp.generate_qkv_padding_mask(torch.tensor([4,4],dtype=torch.int32),torch.tensor([15,1],dtype=torch.int32),8).cuda(),
    )
    layout=QueryLayout(info,8,rank,torch.device('cuda',rank))
    owner=DsaWorkspaceOwner(device=torch.device('cuda',rank),tp_size=8)
    owner.prefill_query_cp=True;owner.query_layout=layout
    extents=[8209,4127];columns=((max(extents)+63)//64)*2
    table=torch.arange(1,2*columns+1,device='cuda',dtype=torch.int32).view(2,columns)
    params=DsaPageRRParams(SimpleNamespace(tokens_per_block=8,kernel_tokens_per_block=4),parallel,workspace_owner=owner)
    inputs=SimpleNamespace(is_prefill=True,is_target_verify=False,is_mtp_draft_update=False,kv_cache_kernel_block_id_device=table)
    params.prepare(inputs)
    index_cache=SimpleNamespace(kv_cache_base=torch.zeros(2*columns+1,4,256,dtype=torch.uint8,device='cuda'))
    latent_cache=torch.zeros(2*columns+1,4,576,dtype=torch.bfloat16,device='cuda')
    op=Bf16DraftIndexerOp(SimpleNamespace(indexer_use_fp8=False,index_rope_dim=0,index_head_dim=128,index_topk=2048))
    torch.manual_seed(3101)
    histories=[]
    for request,length in enumerate(extents):
        positions=torch.arange(length,device='cuda',dtype=torch.int32)
        requests=torch.full_like(positions,request)
        slots=params.cache_adapter.slot_mapping(positions,requests,table)
        owned=slots>=0
        keys=torch.zeros(length,128,device='cuda',dtype=torch.bfloat16)
        keys[:,0]=(positions%7).bfloat16()
        history=(torch.randn(length,576,device='cuda')*.1).bfloat16();histories.append(history)
        op.cache_view(index_cache,4).view(-1,128)[slots[owned]]=keys[owned]
        latent_cache.view(-1,576)[slots[owned]]=history[owned]
    query=torch.zeros(layout.local_rows,32,128,device='cuda',dtype=torch.bfloat16);query[:,:,0]=1
    gates=torch.full((layout.local_rows,32),1/32,device='cuda')
    value=torch.eye(512,128,device='cuda',dtype=torch.bfloat16)
    owner.value_weights[0]=value[None].expand(96,-1,-1).contiguous()
    with patch.dict(os.environ, {'RTP_PREFILL_KV_WORKSPACE_BUDGET_GIB': '0.00390625'}):
        index = params.prepare_execution('indexer')
        sfa = params.prepare_execution('sfa')
    assert index.plan.chunked and sfa.plan.chunked
    selected=op.compute_topk(query,gates,index_cache,params,inputs,params)
    for row,position in enumerate(layout.local_positions.tolist()):
        expected=torch.full((2048,),-1,device='cuda',dtype=torch.int32)
        if position>=0:
            candidates=torch.arange(position+1,device='cuda',dtype=torch.int32)
            chosen=(candidates%7).argsort(descending=True,stable=True)[:2048]
            expected[:chosen.numel()]=chosen.int()
        # Selection promises exact members (including the smaller-position
        # tie rule), not their order. SFA consumes this set; sorting belongs
        # only in this reference assertion, never the production path.
        torch.testing.assert_close(selected[row].sort().values,expected.sort().values,rtol=0,atol=0)
    absorbed=(torch.randn(layout.local_rows,96,576,device='cuda')*.01).bfloat16()
    sparse=PageRRSparseMlaOp(96,512,64,128,4,1.0,2048,parallelism_config=parallel,workspace_owner=owner)
    sparse.plan(params,table,inputs)
    with patch.object(torch,'bmm',wraps=torch.bmm) as v_up:
        actual=sparse.forward(absorbed,latent_cache.view(-1,1,576),selected)
        assert v_up.call_count==1
    reference=torch.zeros_like(actual)
    for row,request in enumerate(layout.local_requests.tolist()):
        chosen=selected[row][selected[row]>=0].long()
        if chosen.numel():
            history=histories[request][chosen].float()
            score=absorbed[row].float()@history.T/(192**.5)
            reference[row]=(score.softmax(-1)@history[:,:128]).bfloat16()
    torch.testing.assert_close(actual,reference,rtol=.03,atol=.0005)
    print(json.dumps({'rank':rank,'cached_chunk_passed':True,'local_Q':layout.local_rows,
                      'indexer_chunks':len(index.blocks),'SFA_chunks':len(sfa.blocks),
                      'TopK_mask_ties_exact':True,'V_up_calls':1,
                      'output_FP32_reference_max_abs':(actual.float()-reference.float()).abs().max().item()}),flush=True)


def main():
    rank=int(os.environ['RANK']);torch.cuda.set_device(rank)
    p=ParallelismConfig();p.tp_size=p.ep_size=p.world_size=p.local_world_size=8
    p.world_rank=p.local_rank=rank;p.ffn_sp_size=8;p.role_type=RoleType.PREFILL
    p.prefill_cp_config.method=CPRotateMethod.ALL_GATHER
    p.prefill_cp_config.preserve_tp_layout=True;p.prefill_cp_config.kv_cache_sharded=True
    nccl=NcclCommConfig();nccl.nccl_ip=os.environ['MASTER_ADDR']
    collective.init_distributed_environment(p,nccl,int(os.environ['MASTER_PORT']),disable_custom_all_reduce=True)
    check_query_embedding(rank, p)
    root=Path(__file__).resolve().parents[2]
    so=root/'bazel-bin/rtp_llm/cpp/models/context_parallel/test/libth_context_parallel_py_wrapper_test.so'
    loader=importlib.machinery.ExtensionFileLoader('libth_context_parallel_py_wrapper_test',str(so))
    spec=importlib.util.spec_from_loader(loader.name,loader);cpp=importlib.util.module_from_spec(spec);loader.exec_module(cpp)
    lengths=torch.tensor([29,45],dtype=torch.int32)
    chunks=torch.tensor([4,6],dtype=torch.int32)
    padding=torch.tensor([3,3],dtype=torch.int32)
    info=SimpleNamespace(prefill_actual_input_lengths_cpu=lengths,
                         prefill_prefix_lengths_cpu=torch.tensor([0,19],dtype=torch.int32),
                         prefill_qkv_restore_indice=cpp.generate_qkv_restore_indices(chunks,8).cuda(),
                         prefill_qkv_padding_mask=cpp.generate_qkv_padding_mask(chunks,padding,8).cuda())
    layout=QueryLayout(info,8,rank,torch.device('cuda',rank))
    owner=DsaWorkspaceOwner(device=torch.device('cuda',rank),tp_size=8)
    owner.prefill_query_cp=True;owner.query_layout=layout
    table=torch.zeros(2,2,dtype=torch.int32,device='cuda')
    if rank<4:table[0]=torch.tensor([1,2],device='cuda')
    table[1]=torch.tensor([3,4],device='cuda')
    params=DsaPageRRParams(SimpleNamespace(tokens_per_block=8,kernel_tokens_per_block=4),p,workspace_owner=owner)
    inputs=SimpleNamespace(is_prefill=True,is_target_verify=False,is_mtp_draft_update=False,
                           kv_cache_kernel_block_id_device=table)
    params.prepare(inputs)
    for phase, dim, capacity in [('indexer', 128, 4), ('sfa', 576, 8)]:
        fixed = layout.physical_rows * 24 + 64 * 1024 * 1024
        budget = fixed + capacity * (dim * 2 * 9 + 8 * 24)
        with patch.dict(os.environ, {'RTP_PREFILL_KV_WORKSPACE_BUDGET_GIB': str(budget / 2**30)}):
            params.ingress_plans[phase] = DsaKVIngress(params, phase, dim)
        assert params.ingress_plans[phase].capacity == capacity
    assert len(params.ingress_plans['indexer'].blocks)==3
    assert len(params.ingress_plans['sfa'].blocks)==2
    torch.manual_seed(3101)
    keys=(torch.randn(2,64,128,device='cuda')*.1).bfloat16()
    latents=(torch.randn(2,64,576,device='cuda')*.1).bfloat16()
    positions=layout.local_positions.long();requests=layout.local_requests.long()
    local_k=keys[requests,positions.clamp_min(0)]
    local_l=latents[requests,positions.clamp_min(0)]
    index_cache=SimpleNamespace(kv_cache_base=torch.zeros(5,4,256,dtype=torch.uint8,device='cuda'))
    sfa_cache=torch.zeros(5,4,576,dtype=torch.bfloat16,device='cuda')
    op=Bf16DraftIndexerOp(SimpleNamespace(indexer_use_fp8=False,index_rope_dim=0,index_head_dim=128,index_topk=2048))
    query=torch.zeros(layout.local_rows,32,128,dtype=torch.bfloat16,device='cuda')
    op.encode_q_k(query,local_k,index_cache,params,inputs,params)
    one=torch.ones((),device='cuda')
    params.ingress_plans['sfa'].run(local_l,lambda v,s:compute_ops.concat_and_cache_mla(v[:,:512],v[:,512:],sfa_cache,s,'auto',one))
    all_positions=torch.cat([torch.arange(29),torch.arange(19,64)]).cuda().int()
    all_requests=torch.cat([torch.zeros(29),torch.ones(45)]).cuda().int()
    slots=params.cache_adapter.slot_mapping(all_positions,all_requests,table)
    owned=slots>=0
    expected_k=keys[all_requests.long(),all_positions.long()][owned]
    expected_l=latents[all_requests.long(),all_positions.long()][owned]
    torch.testing.assert_close(op.cache_view(index_cache,4).view(-1,128)[slots[owned]],expected_k,rtol=0,atol=0)
    torch.testing.assert_close(sfa_cache.view(-1,576)[slots[owned]],expected_l,rtol=0,atol=0)
    assert params.query_positions.shape[0]==layout.local_rows
    assert (params.slot_mapping[~layout.local_valid]==-1).all()
    # Seed the cached prefix on its owners, then check the entire local
    # zigzag Q batch against an independent natural-history attention oracle.
    prefix_pos=torch.arange(19,device='cuda',dtype=torch.int32)
    prefix_req=torch.ones_like(prefix_pos)
    prefix_slots=params.cache_adapter.slot_mapping(prefix_pos,prefix_req,table)
    prefix_owned=prefix_slots>=0
    op.cache_view(index_cache,4).view(-1,128)[prefix_slots[prefix_owned]]=keys[1,prefix_pos[prefix_owned].long()]
    sfa_cache.view(-1,576)[prefix_slots[prefix_owned]]=latents[1,prefix_pos[prefix_owned].long()]
    query=(torch.randn(layout.local_rows,32,128,device='cuda')*.1).bfloat16()
    gates=torch.rand(layout.local_rows,32,device='cuda')*.01
    params.prepare_execution('indexer')
    selected=op.compute_topk(query,gates,index_cache,params,inputs,params)
    for row,pos in enumerate(positions.tolist()):
        chosen=selected[row][selected[row]>=0]
        expected=torch.arange(pos+1,device='cuda',dtype=torch.int32)
        torch.testing.assert_close(chosen.sort().values,expected,rtol=0,atol=0)
    value=torch.eye(512,128,device='cuda',dtype=torch.bfloat16)
    owner.value_weights[0]=value[None].expand(96,-1,-1).contiguous()
    params.prepare_execution('sfa')
    sparse=PageRRSparseMlaOp(96,512,64,128,4,1.0,2048,parallelism_config=p,workspace_owner=owner)
    assert isinstance(sparse.communicator,QueryCpLayout)
    sparse.plan(params,table,inputs)
    absorbed=(torch.randn(layout.local_rows,96,576,device='cuda')*.01).bfloat16()
    with patch.object(torch,'bmm',wraps=torch.bmm) as v_up:
        actual=sparse.forward(absorbed,sfa_cache.view(-1,1,576),selected)
        assert v_up.call_count==1
    complete=torch.cat((latents[0,:29],latents[1,:64]))
    ref_slots=torch.where(selected>=0,selected+params.query_kv_offsets[:,None],-1)
    native,_,_=flash_mla_sparse_fwd(
        torch.nn.functional.pad(absorbed,(0,0,0,32)).contiguous(),
        complete[:,None],ref_slots[:,None],192**-.5,d_v=512,
    )
    native=native[:,:96,:128]
    native=torch.where(layout.local_valid[:,None,None],native,0)
    torch.testing.assert_close(actual,native,rtol=0,atol=0)
    reference=torch.zeros_like(actual)
    for row,pos in enumerate(positions.tolist()):
        if pos>=0:
            history=latents[requests[row],:pos+1].float()
            score=absorbed[row].float()@history.T/(192**.5)
            reference[row]=(score.softmax(-1)@history[:,:128]).bfloat16()
    torch.testing.assert_close(actual,reference,rtol=.03,atol=.0005)
    dist.barrier();print(json.dumps({'rank':rank,'passed':True,'local_Q':layout.local_rows,
                                    'indexer_ingress_blocks':3,'SFA_ingress_blocks':2,
                                    'producer_owner_transport':'verified','padding_cache_write':False,
                                    'noncontiguous_query_causal_mask':True,'full_heads_SFA_reference':True,
                                    'SFA_native_dense_history_exact':True,
                                    'FP32_reference_max_abs':(actual.float()-reference.float()).abs().max().item(),
                                    'prefill_head_query_A2A':False,'V_up_calls':1}),flush=True)
    check_cached_chunks(rank,p,cpp)
    dist.destroy_process_group()

if __name__=='__main__':main()
