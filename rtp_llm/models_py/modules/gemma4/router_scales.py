"""Experimental normalization of existing BF16 top-k results without reordering."""

import torch
import triton
import triton.language as tl


@triton.jit
def _router_scales_kernel(P, IDS, WEIGHTS, OUT, SUM, ROWS,
                          PS: tl.constexpr, IS: tl.constexpr,
                          SAVE_SUM: tl.constexpr, BLOCK_R: tl.constexpr):
    row=tl.program_id(0).to(tl.int64)*BLOCK_R+tl.arange(0,BLOCK_R)
    p0=tl.load(P+row*PS,mask=row<ROWS,other=0).to(tl.float32)+0.0
    p1=tl.load(P+row*PS+1,mask=row<ROWS,other=0).to(tl.float32)+0.0
    p2=tl.load(P+row*PS+2,mask=row<ROWS,other=0).to(tl.float32)+0.0
    p3=tl.load(P+row*PS+3,mask=row<ROWS,other=0).to(tl.float32)+0.0
    p4=tl.load(P+row*PS+4,mask=row<ROWS,other=0).to(tl.float32)+0.0
    p5=tl.load(P+row*PS+5,mask=row<ROWS,other=0).to(tl.float32)+0.0
    p6=tl.load(P+row*PS+6,mask=row<ROWS,other=0).to(tl.float32)+0.0
    p7=tl.load(P+row*PS+7,mask=row<ROWS,other=0).to(tl.float32)+0.0
    a0,a1,a2,a3=p0+p4,p1+p5,p2+p6,p3+p7
    denominator=((a0+a2)+(a1+a3)).to(tl.bfloat16)
    if SAVE_SUM:
        tl.store(SUM+row,denominator,mask=row<ROWS)
    slot=tl.arange(0,8)
    values=tl.load(P+row[:,None]*PS+slot[None,:],mask=row[:,None]<ROWS,other=0).to(tl.float32)
    ids=tl.load(IDS+row[:,None]*IS+slot[None,:],mask=row[:,None]<ROWS,other=0)
    coefficients=tl.load(WEIGHTS+ids,mask=row[:,None]<ROWS,other=0).to(tl.float32)
    normalized=tl.div_rn(values,denominator[:,None].to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT+row[:,None]*8+slot[None,:],normalized*coefficients,mask=row[:,None]<ROWS)


def renormalize_topk(values,indices,per_expert_scale,return_sum=False):
    if (
        not values.is_cuda or values.dtype!=torch.bfloat16 or values.dim()!=2
        or values.shape[0]<=0 or values.shape[1]!=8
        or values.stride(1)!=1 or values.stride(0)<=0
        or indices.device!=values.device or indices.dtype!=torch.int64
        or indices.shape!=values.shape or indices.stride(1)!=1 or indices.stride(0)<=0
        or per_expert_scale.device!=values.device or per_expert_scale.dtype!=torch.bfloat16
        or per_expert_scale.dim()!=1 or not per_expert_scale.is_contiguous()
        or per_expert_scale.numel()<=0
    ):
        return None
    output=torch.empty(values.shape,device=values.device,dtype=values.dtype)
    denominator=torch.empty((values.shape[0],1),device=values.device,dtype=values.dtype) if return_sum else output
    _router_scales_kernel[(triton.cdiv(values.shape[0],64),)](
        values,indices,per_expert_scale,output,denominator,values.shape[0],
        values.stride(0),indices.stride(0),return_sum,64,
        num_warps=4,enable_fp_fusion=False,
    )
    return (output,denominator) if return_sum else output
