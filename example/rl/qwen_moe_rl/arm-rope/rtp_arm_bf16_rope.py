"""Explicit BF16-only PyTorch RoPE/HND-cache compatibility backend.

This is project source, not the rtp_kernel wheel or its CUDA implementation.
Qwen's Q/K normalization is performed by the canonical RTP norm/Triton fusion;
these fallbacks consume already-normalized packed QKV. No quantized path exists.
"""
from __future__ import annotations

import math
import os

import torch

BACKEND = 'torch_bf16_v1'


def _check(condition, message):
    if isinstance(condition, torch.Tensor):
        if condition.device.type == 'cuda':
            torch._assert_async(condition.all(), message)
        elif not bool(condition.all()):
            raise ValueError(message)
    elif not condition:
        raise ValueError(message)


def _device(tensor):
    if tensor.device.type == 'cuda':
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('torch_bf16_v1 fallback does not support CUDA graph capture')
    elif not (tensor.device.type == 'cpu' and os.getenv('RTP_ARM_BF16_ROPE_ALLOW_CPU_TESTS') == '1'):
        raise ValueError('torch_bf16_v1 requires CUDA; CPU is available only for explicit numerical tests')


def _bf16(tensor, name, device=None, *, contiguous=True):
    _check(isinstance(tensor, torch.Tensor), f'{name} must be a Tensor')
    _device(tensor)
    _check(tensor.dtype == torch.bfloat16, f'{name} must be BF16; FP8/FP16/quantized tensors are unsupported')
    _check(tensor.is_contiguous() if contiguous else tensor.stride(-1)==1,
           f'{name} must have contiguous storage' if contiguous else f'{name} must have contiguous token rows')
    if device is not None:
        _check(tensor.device == device, f'{name} must be on {device}')


def _ints(tensor, name, device, shape=None):
    _check(isinstance(tensor, torch.Tensor), f'{name} must be an integer Tensor')
    _check(tensor.dtype in (torch.int32, torch.int64), f'{name} must be int32/int64')
    if shape is not None:
        _check(tuple(tensor.shape) == tuple(shape), f'{name} shape must be {shape}')
    return tensor.to(device=device, dtype=torch.int64, non_blocking=True)


def _no_quant(kv_cache_scale, store_qkv_fp8=False):
    _check(not store_qkv_fp8, 'FP8 query/QKV output is unsupported')
    _check(kv_cache_scale is None or kv_cache_scale.numel() == 0,
           'KV scales/quantized KV caches are unsupported')


def _style(style):
    value = getattr(style, 'name', None)
    value = value() if callable(value) else value
    name = str(value if value is not None else style).rsplit('.', 1)[-1].lower()
    if name in ('no', 'none', '0'):
        return 'none'
    if name in ('base', '1'):
        return 'base'
    if name == 'mrope':
        return 'mrope'
    raise ValueError(f'Unsupported rotary style {style!r}; only No/Base/Qwen interleaved Mrope are supported')


def _rotary(qkv, positions, *, head_num, kv_head_num, size_per_head,
            rope_style, rope_dim, rope_base, rope_scale, rope_offset,
            rope_index_factor, rope_mrope_dim1, rope_mrope_dim2, rope_mrope_dim3,
            use_logn_attn=False, rope_cache=None):
    _bf16(qkv, 'qkv', contiguous=False)
    qh, kh, dim = int(head_num), int(kv_head_num), int(size_per_head)
    _check(qh > 0 and kh > 0 and dim > 0 and qkv.ndim == 2
           and qkv.shape[1] == (qh + 2*kh)*dim, 'Invalid packed QKV/head dimensions')
    _check(not use_logn_attn, 'LogN attention is unsupported')
    mode = _style(rope_style)
    if mode == 'none':
        _check(int(rope_dim) == 0, 'No-RoPE requires rotary_dim=0')
        return qkv.clone()
    rotary_dim = int(rope_dim)
    _check(0 < rotary_dim <= dim and rotary_dim % 2 == 0, 'Invalid partial rotary dimension')
    _check(math.isfinite(float(rope_base)) and float(rope_base) > 0
           and float(rope_scale) == 1.0 and int(rope_offset) == 0,
           'Only unscaled Base/MRoPE with offset=0 is supported')
    # Compute Base coefficients in FP32 from the declared base, as HF eager does.
    # The old CUDA wheel's precomputed cache layout is intentionally not assumed.
    if rope_cache is not None:
        _check(rope_cache.device == qkv.device and rope_cache.dtype in (torch.float32,torch.bfloat16),
               'Invalid optional rotary cache dtype/device')
    pairs = rotary_dim//2
    pos = _ints(positions, 'position_ids', qkv.device)
    if mode == 'mrope':
        sections = (int(rope_mrope_dim1),int(rope_mrope_dim2),int(rope_mrope_dim3))
        _check(pos.numel() == qkv.shape[0]*3 and sum(sections) == pairs and min(sections) >= 0,
               'MRoPE requires three positions/token and sections summing to rotary_dim/2')
        _check(int(rope_index_factor) == 3 and sections[1] <= (pairs+1)//3 and sections[2] <= pairs//3,
               'Only Qwen3.5 interleaved MRoPE is supported')
        pos = pos.reshape(-1,3)
        pair = torch.arange(pairs,device=qkv.device)
        axis = torch.where((pair % 3 == 1) & (pair < 3*sections[1]),1,0)
        axis = torch.where((pair % 3 == 2) & (pair < 3*sections[2]),2,axis)
        selected = pos[:,axis].to(torch.float32)
    else:
        _check(int(rope_index_factor) == 1 and pos.numel() == qkv.shape[0],
               'Base RoPE requires one position/token')
        selected = pos.reshape(-1,1).to(torch.float32)
    _check(pos >= 0, 'Negative rotary positions are invalid')
    inv = 1.0/(float(rope_base)**(torch.arange(0,rotary_dim,2,device=qkv.device,dtype=torch.float32)/rotary_dim))
    angle = selected*inv
    cos, sin = angle.cos()[:,None,:], angle.sin()[:,None,:]
    packed = qkv.reshape(qkv.shape[0],qh+2*kh,dim)
    output = packed.clone()
    qk = packed[:,:qh+kh,:].float()
    low, high = qk[...,:pairs], qk[...,pairs:rotary_dim]
    output[:,:qh+kh,:pairs] = (low*cos-high*sin).to(torch.bfloat16)
    output[:,:qh+kh,pairs:rotary_dim] = (high*cos+low*sin).to(torch.bfloat16)
    return output.reshape_as(qkv)


def _cache_append(packed, sequence, physical, active, kv_cache, kv_cache_offset,
                  head_num, kv_head_num, size_per_head, tokens_per_block):
    _bf16(kv_cache, 'kv_cache', packed.device)
    qh, kh, dim, page_size = map(int,(head_num,kv_head_num,size_per_head,tokens_per_block))
    _check(page_size > 0 and kv_cache.numel() % (2*kh*page_size*dim) == 0,
           'Cache must use contiguous RTP [page,2,KV-head,token,D] HND storage')
    offsets = _ints(kv_cache_offset,'kv_cache_offset',packed.device)
    _check(offsets.ndim == 4 and tuple(offsets.shape[1:3]) == (1,2),
           'Offsets must have [batch,1,2,max_pages] block-array shape')
    cache = kv_cache.view(-1,kh,page_size,dim)
    sequence, physical = sequence[active],physical[active]
    _check(physical >= 0, 'Negative physical cache position')
    block,slot = physical//page_size,physical%page_size
    _check(block < offsets.shape[-1], 'Physical cache position exceeds block table')
    _check(sequence < offsets.shape[0], 'Cache sequence exceeds block table')
    values = packed.reshape(packed.shape[0],qh+2*kh,dim)[active]
    for kind in (0,1):
        pages = offsets[sequence,0,kind,block]
        _check((pages >= 0) & (pages < cache.shape[0]), 'Invalid physical cache block')
        cache[pages,:,slot,:] = values[:,qh+kind*kh:qh+(kind+1)*kh,:]


def convert_offset_to_block_array(kv_cache_block_id):
    _device(kv_cache_block_id)
    _check(kv_cache_block_id.dtype == torch.int32 and kv_cache_block_id.ndim == 2,
           'Block IDs must be [batch,max_pages] int32')
    valid = kv_cache_block_id >= 0
    key = torch.where(valid,kv_cache_block_id*2,-1)
    value = torch.where(valid,kv_cache_block_id*2+1,-1)
    return torch.stack((key,value),dim=1).unsqueeze(1).contiguous()


def prefill_fused_rope_kvcache(qkv,cu_seqlens,batch_size,max_seq_len,head_num,kv_head_num,size_per_head,
        tokens_per_block=64,store_q_no_transpose=False,store_q=False,store_kv=False,store_qkv=False,
        store_qkv_fp8=False,store_cache=False,use_paged_fmha=False,kv_cache=None,kv_cache_scale=None,
        kv_cache_offset=None,kv_cache_offset_h=None,rope_cache=None,padding_offset=None,position_ids=None,
        use_logn_attn=False,rope_style=1,rope_dim=128,rope_base=1000000,rope_scale=1.0,
        rope_beta_slow=1.0,rope_beta_fast=1.0,rope_original_max_position_embeddings=0,
        rope_extrapolation_factor=1.0,rope_mscale=1.0,rope_offset=0,rope_index_factor=1,
        rope_mrope_dim1=0,rope_mrope_dim2=0,rope_mrope_dim3=0,prefix_prompt_lengths=None,
        max_prefix_length=0,count_length=False):
    _no_quant(kv_cache_scale,store_qkv_fp8)
    _check(not store_q and not store_kv and bool(store_q_no_transpose) != bool(store_qkv),
           'Only packed QOut or packed QKVOut BF16 output is supported')
    _check(float(rope_mscale) == 1.0 and float(rope_extrapolation_factor) == 1.0,
           'RoPE extrapolation/mscale is unsupported')
    _bf16(qkv,'qkv',contiguous=False)
    batch_size = int(batch_size)
    _check(batch_size > 0, 'Prefill requires at least one sequence')
    cu = _ints(cu_seqlens,'cu_seqlens',qkv.device,(batch_size+1,))
    _check((cu[0] == 0) & (cu[-1] <= qkv.shape[0]) & (cu[-1] >= 0), 'Invalid packed CU boundaries')
    _check(cu[1:] >= cu[:-1], 'CU boundaries must be monotonic')
    tokens = torch.arange(qkv.shape[0],device=qkv.device,dtype=torch.int64)
    sequence = torch.bucketize(tokens,cu[1:],right=True)
    active = (tokens < cu[-1]) & (sequence < batch_size)
    sequence = sequence.clamp(max=batch_size-1)
    prefix = (torch.zeros(batch_size,device=qkv.device,dtype=torch.int64) if prefix_prompt_lengths is None
              else _ints(prefix_prompt_lengths,'prefix_prompt_lengths',qkv.device,(batch_size,)))
    _check(prefix >= 0, 'Negative prefix lengths')
    physical = tokens-cu[sequence]+prefix[sequence]
    positions = position_ids if position_ids is not None else physical
    if _style(rope_style) == 'mrope' and position_ids is None:
        positions = physical[:,None].expand(-1,3)
    rotated = _rotary(qkv,positions,head_num=head_num,kv_head_num=kv_head_num,size_per_head=size_per_head,
        rope_style=rope_style,rope_dim=rope_dim,rope_base=rope_base,rope_scale=rope_scale,
        rope_offset=rope_offset,rope_index_factor=rope_index_factor,
        rope_mrope_dim1=rope_mrope_dim1,rope_mrope_dim2=rope_mrope_dim2,rope_mrope_dim3=rope_mrope_dim3,
        use_logn_attn=use_logn_attn,rope_cache=rope_cache)
    rotated = torch.where(active[:,None],rotated,torch.zeros((),device=qkv.device,dtype=qkv.dtype))
    if store_cache:
        _check(kv_cache is not None and kv_cache_offset is not None,'Cache and offset arrays are required')
        _cache_append(rotated,sequence,physical,active,kv_cache,kv_cache_offset,
            head_num,kv_head_num,size_per_head,tokens_per_block)
    elif kv_cache is not None:
        raise ValueError('kv_cache was supplied while store_cache=False')
    return rotated[:,:int(head_num)*int(size_per_head)].reshape(qkv.shape[0],int(head_num),int(size_per_head)).contiguous() if store_q_no_transpose else rotated


def decode_fused_rope_kvcache(qkv,position_ids=None,*,sequence_lengths,batch_size,head_num,kv_head_num,
        size_per_head,kv_cache,kv_cache_offset,tokens_per_block=64,store_kv=False,kv_cache_scale=None,
        kv_cache_offset_h=None,rope_cache=None,use_logn_attn=False,rope_style=1,rope_dim=128,
        rope_base=1000000,rope_scale=1.0,rope_beta_slow=1.0,rope_beta_fast=1.0,
        rope_original_max_position_embeddings=0,rope_extrapolation_factor=1.0,rope_mscale=1.0,
        rope_offset=0,rope_index_factor=1,rope_mrope_dim1=0,rope_mrope_dim2=0,rope_mrope_dim3=0):
    """One decode token/sequence; sequence_lengths is the physical next slot.

    This explicit new ABI intentionally has no cu_seqlens argument: RTP detects
    sequence_lengths and does not pass its zero-filled context CU array. MRoPE
    position IDs select rotary coordinates, never physical cache positions.
    """
    _no_quant(kv_cache_scale)
    _check(not store_kv, 'Contiguous K/V output is unsupported')
    _check(float(rope_mscale) == 1.0 and float(rope_extrapolation_factor) == 1.0,
           'RoPE extrapolation/mscale is unsupported')
    _check(qkv.shape[0] == int(batch_size), 'Decode requires exactly one token per sequence')
    physical = _ints(sequence_lengths,'sequence_lengths',qkv.device,(int(batch_size),))
    positions = position_ids if position_ids is not None else physical
    if _style(rope_style) == 'mrope' and position_ids is None:
        positions = physical[:,None].expand(-1,3)
    rotated = _rotary(qkv,positions,head_num=head_num,kv_head_num=kv_head_num,size_per_head=size_per_head,
        rope_style=rope_style,rope_dim=rope_dim,rope_base=rope_base,rope_scale=rope_scale,
        rope_offset=rope_offset,rope_index_factor=rope_index_factor,
        rope_mrope_dim1=rope_mrope_dim1,rope_mrope_dim2=rope_mrope_dim2,rope_mrope_dim3=rope_mrope_dim3,
        use_logn_attn=use_logn_attn,rope_cache=rope_cache)
    sequence = torch.arange(int(batch_size),device=qkv.device)
    active = torch.ones(int(batch_size),device=qkv.device,dtype=torch.bool)
    _cache_append(rotated,sequence,physical,active,kv_cache,kv_cache_offset,
        head_num,kv_head_num,size_per_head,tokens_per_block)
    return rotated[:,:int(head_num)*int(size_per_head)].reshape(int(batch_size),int(head_num),int(size_per_head)).contiguous()


def get_base_rope_cache(rope_dim,rope_base,rope_scale,max_position_embeddings,interleave=True):
    _check(int(rope_dim)>0 and int(rope_dim)%2==0 and float(rope_scale)==1.0,'Only unscaled Base RoPE cache is supported')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    probe=torch.empty(0,device=device,dtype=torch.bfloat16);_device(probe)
    inv=1.0/(float(rope_base)**(torch.arange(0,int(rope_dim),2,device=device,dtype=torch.float32)/int(rope_dim)))
    angle=torch.outer(torch.arange(int(max_position_embeddings),device=device,dtype=torch.float32),inv)
    cos,sin=angle.cos(),angle.sin()
    return torch.stack((cos,sin),dim=-1).flatten(1) if interleave else torch.cat((cos,sin),dim=-1)


def get_yarn_rope_cache(*args,**kwargs):
    raise NotImplementedError('torch_bf16_v1 does not implement YaRN/scaled RoPE')


def _extraction_inputs(k_cache,v_cache,prefix_seq_lens,block_tables,k_out,v_out,page_size):
    _bf16(k_cache,'k_cache');_bf16(v_cache,'v_cache',k_cache.device)
    _bf16(k_out,'k_out',k_cache.device);_bf16(v_out,'v_out',k_cache.device)
    _check(k_cache.shape == v_cache.shape and k_cache.ndim == 4
           and k_cache.shape[2] == int(page_size) and k_out.shape == v_out.shape,
           'KV extraction requires HND caches and equal output shapes')
    lengths=_ints(prefix_seq_lens,'prefix_seq_lens',k_cache.device)
    tables=_ints(block_tables,'block_tables',k_cache.device)
    _check(lengths.ndim == 1 and tables.ndim == 2 and tables.shape[0] == lengths.numel(),
           'Invalid extraction length/block table shape')
    _check((lengths >= 0) & (lengths <= tables.shape[1]*int(page_size)), 'Extraction length exceeds block table')
    return lengths,tables


def ExtractKVFromPagedCachePrefill(k_cache,v_cache,prefix_seq_lens,block_tables,k_out,v_out,page_size):
    lengths,tables=_extraction_inputs(k_cache,v_cache,prefix_seq_lens,block_tables,k_out,v_out,page_size)
    _check(k_out.ndim == 3 and tuple(k_out.shape[1:]) == (k_cache.shape[1],k_cache.shape[3]),'Invalid prefill extraction output')
    _check(lengths.sum() == k_out.shape[0], 'Prefill output does not match total sequence length')
    cu=torch.cat((lengths.new_zeros(1),lengths.cumsum(0)))
    token=torch.arange(k_out.shape[0],device=k_cache.device)
    sequence=torch.bucketize(token,cu[1:],right=True)
    position=token-cu[sequence]
    pages=tables[sequence,position//int(page_size)]
    _check((pages>=0)&(pages<k_cache.shape[0]),'Invalid extraction page')
    k_out.copy_(k_cache[pages,:,position%int(page_size),:])
    v_out.copy_(v_cache[pages,:,position%int(page_size),:])


def ExtractKVFromPagedCacheDecode(k_cache,v_cache,prefix_seq_lens,block_tables,k_out,v_out,page_size,first_n,last_m):
    lengths,tables=_extraction_inputs(k_cache,v_cache,prefix_seq_lens,block_tables,k_out,v_out,page_size)
    first_n,last_m=int(first_n),int(last_m)
    _check(first_n>=0 and last_m>=0 and k_out.shape == (lengths.numel(),first_n+last_m,k_cache.shape[1],k_cache.shape[3]),
           'Invalid decode extraction window/output')
    column=torch.arange(first_n+last_m,device=k_cache.device)[None,:]
    length=lengths[:,None]
    tail_start=torch.maximum(torch.minimum(length,torch.full_like(length,first_n)),length-last_m)
    positions=torch.where(column<first_n,column,tail_start+column-first_n)
    valid=(positions<length)&(column<torch.minimum(length,torch.full_like(length,first_n+last_m)))
    sequence=torch.arange(lengths.numel(),device=k_cache.device)[:,None].expand_as(positions)
    k_out.zero_();v_out.zero_()
    sequence,columns=sequence[valid],column.expand_as(positions)[valid]
    physical=positions[valid]
    pages=tables[sequence,physical//int(page_size)]
    _check((pages>=0)&(pages<k_cache.shape[0]),'Invalid extraction page')
    k_out[sequence,columns]=k_cache[pages,:,physical%int(page_size),:]
    v_out[sequence,columns]=v_cache[pages,:,physical%int(page_size),:]


def kvcache_extract_prefill(k_cache,v_cache,prefix_seq_lens,block_tables,k_out,v_out,page_size):
    ExtractKVFromPagedCachePrefill(k_cache,v_cache,prefix_seq_lens,block_tables,k_out,v_out,page_size)
    return k_out,v_out


def kvcache_extract_decode(k_cache,v_cache,prefix_seq_lens,block_tables,k_out,v_out,page_size,first_n,last_m):
    ExtractKVFromPagedCacheDecode(k_cache,v_cache,prefix_seq_lens,block_tables,k_out,v_out,page_size,first_n,last_m)
    return k_out,v_out
