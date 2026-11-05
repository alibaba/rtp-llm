"""Numerical/reference tests for project BF16 compatibility API, CPU or CUDA.

This is not a full-model HF/native logprob acceptance test.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
import types

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import rtp_arm_bf16_rope as backend


def spawned_import(queue):
    import rtp_arm_bf16_import_hook as hook
    proof=hook.install()
    from rtp_kernel.fused_rope_kvcache import decode_fused_rope_kvcache
    import rtp_ops
    queue.put(dict(backend=rtp_ops.__backend__,external_wheel=proof['external_rtp_kernel_wheel'],
        explicit_lengths='sequence_lengths' in inspect.signature(decode_fused_rope_kvcache).parameters,
        context_cu='cu_seqlens' in inspect.signature(decode_fused_rope_kvcache).parameters))


def scalar_rope(qkv,positions,qh,kh,dim,rotary_dim,base,sections=None):
    """Independent scalar pair/axis formula, including explicit FP32 rounding."""
    output=qkv.cpu().clone().reshape(qkv.shape[0],qh+2*kh,dim)
    source=qkv.cpu().reshape_as(output)
    positions=positions.cpu().reshape(qkv.shape[0],-1)
    for token in range(qkv.shape[0]):
        for head in range(qh+kh):
            for pair in range(rotary_dim//2):
                axis=0
                if sections is not None:
                    if pair%3==1 and pair<3*sections[1]: axis=1
                    if pair%3==2 and pair<3*sections[2]: axis=2
                # Torch FP32 scalar arithmetic matches the declared eager boundary.
                exponent=torch.tensor(2*pair/rotary_dim,dtype=torch.float32)
                inv=1.0/(float(base)**exponent)
                angle=torch.tensor(float(positions[token,axis]),dtype=torch.float32)*inv
                cos,sin=angle.cos(),angle.sin()
                low=source[token,head,pair].float()
                high=source[token,head,pair+rotary_dim//2].float()
                output[token,head,pair]=(low*cos-high*sin).to(torch.bfloat16)
                output[token,head,pair+rotary_dim//2]=(high*cos+low*sin).to(torch.bfloat16)
    return output.reshape_as(qkv.cpu()).to(qkv.device)


def close(actual,expected,label,tolerance=0.0):
    error=float((actual.float()-expected.float()).abs().max()) if actual.numel() else 0.0
    if error>tolerance:
        raise AssertionError(f'{label}: max abs {error} > {tolerance}')
    return error


def expect_error(method,text):
    try: method()
    except (ValueError,RuntimeError,NotImplementedError) as error:
        if text not in str(error): raise AssertionError(f'Expected {text!r}, got {error!r}') from error
        return
    raise AssertionError(f'Expected rejection containing {text!r}')


def norm_zero_centered(qkv,qw,kw,qh,kh,dim,eps=1e-6):
    out=qkv.reshape(qkv.shape[0],qh+2*kh,dim).clone()
    for section,weight in [(slice(0,qh),qw),(slice(qh,qh+kh),kw)]:
        value=out[:,section].float()
        out[:,section]=(value*torch.rsqrt(value.square().mean(-1,keepdim=True)+eps)*(weight.float()+1)).bfloat16()
    return out.reshape_as(qkv)


def run(device):
    if device=='cpu': os.environ['RTP_ARM_BF16_ROPE_ALLOW_CPU_TESTS']='1'
    torch.manual_seed(271828)
    results=[]
    def case(name,method):
        started=time.time()
        values=method() or {}
        results.append(dict(name=name,passed=True,seconds=time.time()-started,**values))
    def prefill_base():
        qh,kh,dim,rotary=2,1,8,4
        raw=torch.randn(5,(qh+2*kh)*dim+8,device=device,dtype=torch.bfloat16)
        qkv=raw[:,:-8]  # row-gap layout produced by a packed projection view
        before=qkv.clone()
        weights=[torch.randn(dim,device=device,dtype=torch.bfloat16)*0.1 for _ in range(2)]
        normalized=norm_zero_centered(qkv,*weights,qh,kh,dim)
        cu=torch.tensor([0,3,5],device=device,dtype=torch.int32)
        prefix=torch.tensor([1,3],device=device,dtype=torch.int32)
        physical=torch.tensor([1,2,3,3,4],device=device)
        table=torch.tensor([[2,0,4],[1,3,5]],device=device,dtype=torch.int32)
        offsets=backend.convert_offset_to_block_array(table)
        cache=torch.full((6,2,kh,2,dim),-37,device=device,dtype=torch.bfloat16)
        expected_cache=cache.clone()
        expected=scalar_rope(normalized,physical,qh,kh,dim,rotary,10000)
        out=backend.prefill_fused_rope_kvcache(normalized,cu,2,3,qh,kh,dim,tokens_per_block=2,
            store_q_no_transpose=True,store_cache=True,kv_cache=cache,kv_cache_offset=offsets,
            rope_dim=rotary,rope_base=10000,prefix_prompt_lengths=prefix,max_prefix_length=3,count_length=True)
        packed=expected.reshape(5,qh+2*kh,dim)
        for token,seq in enumerate([0,0,0,1,1]):
            slot=int(physical[token]);page=int(table[seq,slot//2])
            expected_cache[page,0,:,slot%2]=packed[token,qh:qh+kh]
            expected_cache[page,1,:,slot%2]=packed[token,qh+kh:]
        error=close(out,packed[:,:qh],'normalized Base partial rotary',0.03125 if device=='cuda' else 0.0)
        cache_error=close(cache,expected_cache,'HND paged prefill cache',0.03125 if device=='cuda' else 0.0)
        close(qkv,before,'input read-only')
        return dict(max_abs_error=error,cache_max_abs_error=cache_error)
    case('QK FP32 norm boundary + Base partial RoPE + paged prefix cache',prefill_base)

    def prefill_mrope():
        qkv=torch.randn(3,32,device=device,dtype=torch.bfloat16)
        positions=torch.tensor([[8,3,1],[9,4,2],[10,5,3]],device=device,dtype=torch.int32)
        sections=(1,1,1)
        expected=scalar_rope(qkv,positions,2,1,8,6,10000,sections)
        out=backend.prefill_fused_rope_kvcache(qkv,torch.tensor([0,3],device=device,dtype=torch.int32),1,3,2,1,8,
            store_qkv=True,position_ids=positions,rope_style=types.SimpleNamespace(name='Mrope'),
            rope_dim=6,rope_base=10000,rope_index_factor=3,
            rope_mrope_dim1=1,rope_mrope_dim2=1,rope_mrope_dim3=1)
        return dict(max_abs_error=close(out,expected,'interleaved mRoPE partial tail',0.03125 if device=='cuda' else 0.0))
    case('Interleaved MRoPE and partial rotary tail',prefill_mrope)

    def decode_mrope():
        qh,kh,dim=2,1,12
        qkv=torch.randn(2,48,device=device,dtype=torch.bfloat16)
        before=qkv.clone()
        positions=torch.tensor([[11,5,2],[31,7,3]],device=device,dtype=torch.int32)
        physical=torch.tensor([3,6],device=device,dtype=torch.int32)
        table=torch.tensor([[2,0,4],[1,3,5]],device=device,dtype=torch.int32)
        cache=torch.full((6,2,kh,4,dim),-31,device=device,dtype=torch.bfloat16)
        expected_cache=cache.clone()
        expected=scalar_rope(qkv,positions,qh,kh,dim,12,10000,(2,2,2)).reshape(2,4,dim)
        out=backend.decode_fused_rope_kvcache(qkv,positions,sequence_lengths=physical,batch_size=2,
            head_num=qh,kv_head_num=kh,size_per_head=dim,kv_cache=cache,
            kv_cache_offset=backend.convert_offset_to_block_array(table),tokens_per_block=4,
            rope_style=types.SimpleNamespace(name='Mrope'),rope_dim=12,rope_base=10000,
            rope_index_factor=3,rope_mrope_dim1=2,rope_mrope_dim2=2,rope_mrope_dim3=2)
        for seq in range(2):
            slot=int(physical[seq]);page=int(table[seq,slot//4])
            expected_cache[page,0,:,slot%4]=expected[seq,qh:qh+kh]
            expected_cache[page,1,:,slot%4]=expected[seq,qh+kh:]
        close(qkv,before,'decode input read-only')
        signature=inspect.signature(backend.decode_fused_rope_kvcache).parameters
        assert 'sequence_lengths' in signature and 'cu_seqlens' not in signature
        return dict(max_abs_error=close(out,expected[:,:qh],'decode mRoPE',0.03125 if device=='cuda' else 0.0),
            cache_max_abs_error=close(cache,expected_cache,'physical decode slots independent of mRoPE',0.03125 if device=='cuda' else 0.0),
            context_zero_cu_cannot_enter_decode_api=True)
    case('Decode explicit physical length and independent MRoPE positions',decode_mrope)

    def no_rope_and_padding():
        qkv=torch.randn(4,32,device=device,dtype=torch.bfloat16)
        out=backend.prefill_fused_rope_kvcache(qkv,torch.tensor([0,3],device=device,dtype=torch.int32),1,3,2,1,8,
            store_qkv=True,rope_style=0,rope_dim=0)
        close(out[:3],qkv[:3],'No-RoPE prevents duplicate legacy rotation')
        close(out[3:],torch.zeros_like(out[3:]),'inactive query rows zeroed')
    case('Already-rotated No-RoPE and inactive packed rows',no_rope_and_padding)

    def extraction():
        kh,dim,page_size=2,4,4
        k=torch.arange(6*kh*page_size*dim,device=device).reshape(6,kh,page_size,dim).bfloat16()
        v=(k.float()+100).bfloat16()
        table=torch.tensor([[2,0],[1,3],[4,5]],device=device,dtype=torch.int32)
        lengths=torch.tensor([3,5,0],device=device,dtype=torch.int32)
        out=torch.empty(8,kh,dim,device=device,dtype=torch.bfloat16);outv=torch.empty_like(out)
        backend.ExtractKVFromPagedCachePrefill(k,v,lengths,table,out,outv,page_size)
        reference=torch.stack([k[int(table[seq,pos//page_size]),:,pos%page_size,:]
            for seq,length in enumerate([3,5,0]) for pos in range(length)])
        close(out,reference,'prefill KV extraction')
        close(outv,(reference.float()+100).bfloat16(),'prefill V extraction')
        lengths=torch.tensor([1,4,8],device=device,dtype=torch.int32)
        sink,tail=2,3
        out=torch.empty(3,5,kh,dim,device=device,dtype=torch.bfloat16);outv=torch.empty_like(out)
        backend.ExtractKVFromPagedCacheDecode(k,v,lengths,table,out,outv,page_size,sink,tail)
        reference=torch.zeros_like(out);referencev=torch.zeros_like(out)
        for seq,length in enumerate([1,4,8]):
            positions=list(range(min(sink,length)))+list(range(max(min(sink,length),length-tail),length))
            for column,pos in enumerate(positions):
                reference[seq,column]=k[int(table[seq,pos//page_size]),:,pos%page_size,:]
                referencev[seq,column]=v[int(table[seq,pos//page_size]),:,pos%page_size,:]
        close(out,reference,'nonoverlap sink/window extraction')
        close(outv,referencev,'window V extraction')
    case('Paged KV extraction prefill and decode sink/window',extraction)

    def rejections():
        qkv=torch.randn(2,32,device=device,dtype=torch.bfloat16)
        def prefill(**kwargs):
            args=dict(store_qkv=True,rope_style=0,rope_dim=0)
            args.update(kwargs)
            value=args.pop('qkv',qkv)
            return backend.prefill_fused_rope_kvcache(value,torch.tensor([0,2],device=device,dtype=torch.int32),1,2,2,1,8,**args)
        expect_error(lambda:prefill(qkv=qkv.half()),'must be BF16')
        expect_error(lambda:prefill(store_qkv_fp8=True),'FP8')
        expect_error(lambda:prefill(kv_cache_scale=torch.ones(1,device=device)),'scales')
        expect_error(lambda:prefill(rope_style=1,rope_dim=4,rope_scale=2),'unscaled')
        expect_error(lambda:prefill(use_logn_attn=True),'LogN')
        expect_error(lambda:prefill(rope_style='Yarn',rope_dim=4),'Unsupported')
        expect_error(lambda:prefill(store_q=True),'Only packed')
        expect_error(lambda:backend.get_yarn_rope_cache(8,10000,2,4096),'YaRN')
    case('Fail closed for unsupported dtype/quantization/rotary/output paths',rejections)

    def hook():
        import rtp_arm_bf16_import_hook as install
        os.environ['RTP_ARM_BF16_ROPE_BACKEND']='unsupported'
        expect_error(install.install,'Unsupported')
        os.environ['RTP_ARM_BF16_ROPE_BACKEND']='torch_bf16_v1'
        provenance=install.install()
        import rtp_ops
        from rtp_kernel import fused_rope_kvcache
        assert not provenance['external_rtp_kernel_wheel']
        assert rtp_ops.__backend__=='torch_bf16_v1'
        assert fused_rope_kvcache.prefill_fused_rope_kvcache is backend.prefill_fused_rope_kvcache
        assert fused_rope_kvcache.decode_fused_rope_kvcache is backend.decode_fused_rope_kvcache
        assert 'cu_seqlens' not in inspect.signature(fused_rope_kvcache.decode_fused_rope_kvcache).parameters
        return dict(external_wheel_identity_claimed=False)
    case('Explicit import API and provenance',hook)

    def fusion_and_dispatch_guard():
        import rtp_arm_bf16_import_hook as hook
        name='rtp_llm.models_py.triton_kernels.common.qk_norm_rope_cache'
        previous=sys.modules.get(name)
        module=types.ModuleType(name);module.__file__='canonical-Triton-source'
        called=[]
        def original(*args,**kwargs): called.append(True);return 'canonical-triton'
        module.fused_qk_norm_rope_cache=original
        module.maybe_fused_qk_norm_rope_cache=lambda *args:None
        sys.modules[name]=module
        try:
            proof=hook.install_canonical_fusion_guard()
            qkv=torch.zeros(1,32,device=device,dtype=torch.bfloat16)
            cache=torch.zeros(1,2,1,4,8,device=device,dtype=torch.bfloat16)
            output=module.fused_qk_norm_rope_cache(qkv,None,None,None,None,None,None,cache,None)
            assert output=='canonical-triton' and called==[True] and proof['canonical_triton_retained']
            expect_error(lambda:module.fused_qk_norm_rope_cache(qkv,None,None,None,None,None,None,cache.half(),None),'rejects')
            expect_error(lambda:module.maybe_fused_qk_norm_rope_cache(qkv,None,types.SimpleNamespace(native_fp8=True),None),'rejects')
        finally:
            if previous is None: del sys.modules[name]
            else: sys.modules[name]=previous
        dispatch_name='rtp_llm.ops.fused_rope_kvcache_op'
        previous=sys.modules.get(dispatch_name)
        caller=types.ModuleType(dispatch_name);caller.__file__='accepted-RTP-dispatch-source'
        caller._decode_uses_sequence_lengths=lambda method:'sequence_lengths' in inspect.signature(method).parameters
        caller._decode_uses_cu_seqlens=lambda method:'cu_seqlens' in inspect.signature(method).parameters
        sys.modules[dispatch_name]=caller
        try:
            assert hook.verify_rtp_decode_dispatch()['context_cu_seqlens_not_passed']
        finally:
            if previous is None: del sys.modules[dispatch_name]
            else: sys.modules[dispatch_name]=previous
    case('Canonical Triton retained with BF16 guard and explicit decode ABI',fusion_and_dispatch_guard)

    def spawn_contract():
        context=multiprocessing.get_context('spawn')
        queue=context.Queue()
        processes=[context.Process(target=spawned_import,args=(queue,)) for _ in range(2)]
        for process in processes: process.start()
        proofs=[queue.get(timeout=30) for process in processes]
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode==0
        for proof in proofs:
            assert proof==dict(backend='torch_bf16_v1',external_wheel=False,explicit_lengths=True,context_cu=False)
    case('Independent spawn children install the explicit compatibility API',spawn_contract)
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    started=time.time();results=run(args.device)
    if args.device=='cuda': torch.cuda.synchronize()
    import rtp_arm_bf16_import_hook as hook
    report=dict(passed=True,scope='standalone compatibility API numerical tests; not full-model HF/native acceptance',
        device=args.device,cuda_used=args.device=='cuda',torch=torch.__version__,tests=results,
        source_manifest=hook.source_manifest(),elapsed_seconds=time.time()-started)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,device=args.device,tests=len(results),output=str(args.output))))


if __name__=='__main__':main()
