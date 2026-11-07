"""MODEL1 sparse attention with the official 64-head reduction order."""

import torch,triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

# MODEL1 packed page sizes supported by the native decode kernel.
MODEL1_NATIVE_PAGES = (1, 2, 32, 64, 128, 130, 132, 134)

@triton.jit
def _schedule(Schedule, Splits, Lengths, ExtraLengths, Batch: tl.constexpr,
              Width: tl.constexpr, ExtraWidth: tl.constexpr,
              HasLengths: tl.constexpr, HasExtraLengths: tl.constexpr):
    row = tl.program_id(0)
    length = tl.load(Lengths + row) if HasLengths else Width
    if ExtraWidth:
        extra = tl.load(ExtraLengths + row) if HasExtraLengths else ExtraWidth
        end = tl.cdiv(length, 64) * 64 + extra
    else:
        end = length
    # Empty/fake requests still run one masked tile and produce zero output.
    end = tl.maximum(end, 1)
    columns = tl.arange(0, 8)
    values = tl.where((columns == 0) | (columns == 2), row,
                      tl.where(columns == 3, end, 0))
    tl.store(Schedule + row * 8 + columns, values)
    tl.store(Splits + row, row)
    if row == 0:
        tl.store(Splits + Batch, Batch)

@triton.jit
def native_sum64(p):
 lanes=tl.arange(0,8)
 partial=tl.full((16,8),0,tl.float32)
 for j in tl.static_range(8):
  columns=(lanes//4)*32+(lanes%4)*2+(j%4)*8+j//4
  partial=partial+tl.gather(p,tl.broadcast_to(columns[None,:],(16,8)),1)
 for offset in tl.static_range(3):
  partner=lanes^(4>>offset)
  partial=partial+tl.gather(partial,tl.broadcast_to(partner[None,:],(16,8)),1)
 return tl.sum(tl.where(lanes[None,:]==0,partial,0.),1)

@triton.jit
def sparse_dual(Q,KV,KV2,Sink,Idx,Idx2,Lengths,Lengths2,O,H:tl.constexpr,D:tl.constexpr,WA:tl.constexpr,WB:tl.constexpr,SCALE:tl.constexpr,PAGE:tl.constexpr,PAGE2:tl.constexpr,BLOCK_STRIDE:tl.constexpr,STRIDE2:tl.constexpr,QLEN:tl.constexpr,DIV_MODE:tl.constexpr,HAS_LENGTHS:tl.constexpr=True,HAS_LENGTHS2:tl.constexpr=True,BF16_KV:tl.constexpr=False):
 t=tl.program_id(0);hh=tl.program_id(1)*16+tl.arange(0,16);dd=tl.arange(0,D);kk=tl.arange(0,64)
 q=tl.load(Q+t*H*D+hh[:,None]*D+dd[None,:]);maximum=tl.full((16,),float('-inf'),tl.float32);denom=tl.zeros((16,),tl.float32);acc=tl.zeros((16,D),tl.float32)
 length=tl.load(Lengths+t//QLEN) if HAS_LENGTHS else WA
 length2=tl.load(Lengths2+t//QLEN) if HAS_LENGTHS2 else WB
 for block in range(triton.cdiv(WA+WB,64)):
  column=block*64+kk;use_first=column<WA
  ia=tl.load(Idx+t*WA+column,(column<WA)&(column<length),other=-1)
  ib=tl.load(Idx2+t*WB+column-WA,(column>=WA)&(column<WA+WB)&(column-WA<length2),other=-1)
  idx=tl.where(use_first,ia,ib)
  if BF16_KV:
   kv=tl.load(KV+idx[:,None]*D+dd[None,:],idx[:,None]>=0,other=0)
  else:
   words=tl.where(use_first[:,None],KV.to(tl.pointer_type(tl.uint16)),KV2.to(tl.pointer_type(tl.uint16)))
   page_id=tl.where(use_first,idx//PAGE,idx//PAGE2)
   row=tl.where(use_first,idx%PAGE,idx%PAGE2)
   stride=tl.where(use_first,BLOCK_STRIDE//2,STRIDE2//2)
   page=tl.where(use_first,PAGE,PAGE2)
   packed=words+page_id[:,None]*stride[:,None]+row[:,None]*288
   pair=tl.load(packed+dd[None,:]//2,(idx[:,None]>=0)&(dd[None,:]<448),other=0)
   byte=((pair>>(8*(dd[None,:]%2)))&255).to(tl.uint8)
   group=dd[None,:]//64
   scales=tl.load(words+page_id[:,None]*stride[:,None]+page[:,None]*288+row[:,None]*4+group//2,(idx[:,None]>=0)&(dd[None,:]<448),other=32639)
   scale=((scales>>(8*(group%2)))&255).to(tl.float32)
   fp8=byte.to(tl.float8e4nv,bitcast=True).to(tl.float32)*tl.exp2(scale-127.)
   rope=tl.load(packed+224+dd[None,:]-448,(idx[:,None]>=0)&(dd[None,:]>=448),other=0).to(tl.bfloat16,bitcast=True)
   kv=tl.where(dd[None,:]<448,fp8,rope.to(tl.float32)).to(tl.bfloat16)
  score=tl.dot(q,tl.trans(kv))*SCALE;score=tl.where(idx[None,:]>=0,score,float('-inf'))
  newmax=tl.maximum(maximum,tl.max(score,1));safe_max=tl.where(newmax==float('-inf'),0.0,newmax);alpha=libdevice.exp(maximum-safe_max);p=libdevice.exp(score-safe_max[:,None]);denom=tl.fma(denom,alpha,native_sum64(p));acc=acc*alpha[:,None];acc=tl.dot(p.to(tl.bfloat16),kv,acc);maximum=newmax
 sink=tl.load(Sink+hh);denom+=libdevice.exp(sink-maximum)
 if DIV_MODE==1:out=tl.div_rn(acc,denom[:,None])
 elif DIV_MODE==2:
  reciprocal=tl.div_rn(1.0,denom);quotient=acc*reciprocal[:,None];residual=tl.fma(-quotient,denom[:,None],acc);out=tl.fma(residual,reciprocal[:,None],quotient)
 else:out=acc/denom[:,None]
 out=tl.where(denom[:,None]==float('inf'),0.0,out)
 tl.store(O+t*H*D+hh[:,None]*D+dd[None,:],out)


def forward(q, kv_cache, attn_sink, topk_idxs, sched_meta=None,
            cache_seqlens=None, block_table=None, topk_length=None,
            extra_k_cache=None, extra_topk_idxs=None, extra_topk_length=None):
    # Preserve the official 64-head reduction order on MODEL1 packed pools.
    if q.ndim != 4 or q.shape[2:] != (64, 512) or q.dtype != torch.bfloat16:
        raise ValueError("expected BF16 [B,Q,64,512]")
    if not q.is_cuda or not q.is_contiguous():
        raise ValueError("expected contiguous device queries")
    b, nq, h, d = q.shape
    def check(t, dtype):
        if t.device != q.device or t.dtype != dtype or not t.is_contiguous():
            raise ValueError("tensor dtype/device/contiguity mismatch")
    check(attn_sink, torch.float32)
    if attn_sink.shape != (h,):
        raise ValueError("sink shape mismatch")
    def pool(cache, idx, length):
        if cache.ndim != 3 or cache.shape[2] != 584 or cache.shape[1] not in MODEL1_NATIVE_PAGES:
            raise ValueError("expected MODEL1 [blocks,page,584]")
        if cache.dtype != torch.uint8 or cache.device != q.device:
            raise ValueError("cache dtype/device mismatch")
        if cache.stride(2) != 1 or cache.stride(1) != 584 or cache.stride(0) < cache.shape[1]*584 or cache.stride(0)%2 or cache.data_ptr()%2:
            raise ValueError("unsupported packed cache layout")
        if idx.ndim != 3 or idx.shape[:2] != (b,nq):
            raise ValueError("indices shape mismatch")
        check(idx, torch.int32)
        if length is not None:
            check(length, torch.int32)
            if length.shape != (b,):
                raise ValueError("lengths must be per batch")
    pool(kv_cache, topk_idxs, topk_length)
    if (extra_k_cache is None) != (extra_topk_idxs is None):
        raise ValueError("secondary cache and indices must be paired")
    if extra_k_cache is None:
        if extra_topk_length is not None:
            raise ValueError("secondary lengths without pool")
        extra_k_cache, extra_topk_idxs = kv_cache, topk_idxs
        wb = 0
    else:
        pool(extra_k_cache, extra_topk_idxs, extra_topk_length)
        wb = extra_topk_idxs.shape[-1]
    out = torch.empty_like(q)
    if b*nq:
        sparse_dual[(b*nq,h//16)](
            q,kv_cache,extra_k_cache,attn_sink,topk_idxs,extra_topk_idxs,
            topk_length,extra_topk_length,out,h,d,topk_idxs.shape[-1],wb,
            512**-.5,kv_cache.shape[1],extra_k_cache.shape[1],
            kv_cache.stride(0),extra_k_cache.stride(0),nq,2,
            topk_length is not None,extra_topk_length is not None,
            num_warps=4,num_stages=2,enable_fp_fusion=False)
    return out


def forward_prefill(q, kv, indices, sm_scale, d_v=512, attn_sink=None,
                    topk_length=None, out=None):
    """Use the native reduction order with contiguous BF16 Prefill KV."""
    if q.ndim != 3 or q.shape[-1] != 512 or q.shape[1] not in (16, 32, 64, 128):
        raise ValueError("expected BF16 [M,H,512] with H divisible by 16")
    rows, heads, dim = q.shape

    def check(tensor, dtype):
        if tensor is None or tensor.device != q.device or tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError("Prefill dtype/device/contiguity mismatch")

    if not q.is_cuda or d_v != dim:
        raise ValueError("requires device queries and output head_dim 512")
    check(q, torch.bfloat16)
    check(kv, torch.bfloat16)
    check(indices, torch.int32)
    check(attn_sink, torch.float32)
    if (kv.ndim != 3 or kv.shape[1:] != (1, dim)
            or indices.ndim != 3 or indices.shape[:2] != (rows, 1)
            or attn_sink.shape != (heads,)):
        raise ValueError("Prefill KV/indices/sink shape mismatch")
    if topk_length is not None:
        check(topk_length, torch.int32)
        if topk_length.shape != (rows,):
            raise ValueError("Prefill lengths must be per Q row")
    if out is None:
        out = torch.empty_like(q)
    else:
        check(out, torch.bfloat16)
        if out.shape != q.shape:
            raise ValueError("Prefill output shape mismatch")
    if rows:
        sparse_dual[(rows, heads // 16)](
            q, kv, kv, attn_sink, indices, indices, topk_length, None, out,
            heads, dim, indices.shape[-1], 0, sm_scale, 1, 1,
            dim * 2, dim * 2, 1, 2,
            topk_length is not None, False, BF16_KV=True,
            num_warps=4, num_stages=2, enable_fp_fusion=False,
        )
    return out, None, None
