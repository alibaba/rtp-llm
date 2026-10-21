"""HY4 Indexer Q tail with optional precomputed head-gate scaling.

Preserves a separate BF16 rounded query; input Q is never mutated. The first64
interleaved coordinates use the installed FlashInfer inline helper. The last64
remain unchanged. Group128 legacy quant writes the unpacked exponent-bit scale
as FP32 directly; zero input produces scale2**-33, not zero. No Hadamard.

K/cache must still use the original native path. Never call combined Q/K RoPE on
this already-rotated Q. Early notifications grant launch permission only; every
dependent consumer must GDC-wait before reading. Explicit apply_rope=False is a
quant-only diagnostic stage, never an automatic fallback.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def prepare_extension():
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('prepare_extension outside capture required')
    from torch.utils.cpp_extension import CUDA_HOME, load
    spec = find_spec('flashinfer')
    if spec is None or spec.origin is None or CUDA_HOME is None:
        raise RuntimeError('FlashInfer headers and CUDA13 required')
    include = Path(spec.origin).parent/'data/include'
    if not (include/'flashinfer/pos_enc.cuh').is_file():
        raise RuntimeError('actual FlashInfer RoPE header missing')
    version = Path(CUDA_HOME)/'version.json'
    if json.loads(version.read_text())['cuda']['version'].split('.')[0] != '13':
        raise RuntimeError('CUDA13 required')
    root = Path(__file__).resolve().parents[1]
    sources = [root/'cuda/indexer_tail.cpp', root/'cuda/indexer_tail.cu']
    # Preserve FlashInfer RoPE contraction/FTZ, but do not replace native quant
    # log2f/exp2f with fast intrinsics. Quant scale arguments are >=1e-10,
    # hence normal; subnormal query values quantize to signed FP8 zero.
    flags = ['-O3','-std=c++17','-lineinfo',
             '--fmad=true','--ftz=true','--prec-div=true','--prec-sqrt=true',
             '-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__',
             '-U__CUDA_NO_BFLOAT16_CONVERSIONS__','-U__CUDA_NO_HALF2_OPERATORS__',
             '-DFLASHINFER_ENABLE_F16','-DFLASHINFER_ENABLE_BF16',
             '-DFLASHINFER_ENABLE_FP8_E4M3','-DFLASHINFER_ENABLE_FP8_E5M2',
             '-gencode=arch=compute_103,code=sm_103']
    digest = hashlib.sha256(repr(flags).encode()+torch.__version__.encode()+version.read_bytes())
    for p in sources+[root/'cuda/indexer_tail.cuh']+sorted(p for p in include.rglob('*') if p.is_file()):
        digest.update(str(p).encode());digest.update(p.read_bytes())
    name = 'hy4_indexer_q_tail_'+digest.hexdigest()[:20]
    from torch.utils.cpp_extension import _get_build_directory
    build = Path(_get_build_directory(name, verbose=False))
    build.mkdir(parents=True,exist_ok=True)
    return load(name=name,sources=list(map(str,sources)),build_directory=str(build),
                extra_include_paths=[str(include)],extra_cflags=['-O3','-std=c++17'],
                extra_cuda_cflags=flags,verbose=True)


@dataclass(frozen=True)
class Outputs:
    query_bf16: torch.Tensor
    q_fp8: torch.Tensor
    q_scale: torch.Tensor


def allocate_outputs(query):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('allocate outside capture required')
    if (not query.is_cuda or query.dtype != torch.bfloat16 or not query.is_contiguous()
            or query.ndim != 3 or query.shape[1:] != (32,128) or not 1<=query.shape[0]<=256):
        raise ValueError('query must be contiguous CUDA BF16[M,32,128],1<=M<=256')
    return Outputs(torch.empty_like(query),torch.empty_like(query,dtype=torch.float8_e4m3fn),
                   torch.empty_strided((*query.shape[:2],1),(32,1,query.shape[0]*32),
                                       device=query.device,dtype=torch.float32))


def validate_positions(positions,cos_sin):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('validate metadata outside capture')
    if bool(((positions<0)|(positions>=cos_sin.shape[0])).any().item()):
        raise ValueError('position outside table')


def indexer_q_tail(query,positions,cos_sin,*,out,apply_rope=True,pdl=False,release='late',notify_event=None,
                   raw_gate=None,weighted_gate_out=None,gate_scale=1.0):
    if release not in ('early','late'):
        raise ValueError('release must be early or late')
    if os.environ.get('DSV4_FP8_QUANT_KERNEL','auto').strip().lower() not in ('auto','legacy'):
        raise RuntimeError('this bounded candidate targets original legacy query quantization')
    event=0
    if notify_event is not None:
        event=int(notify_event.cuda_event)
        if not event or notify_event.device!=query.device:
            raise ValueError('notify_event must be initialized on input device')
    prepare_extension().run(query,positions,cos_sin,out.query_bf16,out.q_fp8,out.q_scale,
                            bool(apply_rope),bool(pdl),release,event,raw_gate,weighted_gate_out,float(gate_scale))
    return out
