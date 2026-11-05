"""Opt-in project-owned compatibility API; never claims an old wheel identity."""
from __future__ import annotations

import hashlib
import importlib
from importlib.machinery import ModuleSpec
import json
import os
from pathlib import Path
import platform
import sys
import types

MODE_ENV = 'RTP_ARM_BF16_ROPE_BACKEND'
MODE = 'torch_bf16_v1'
_INSTALLED = False


def source_manifest():
    root=Path(__file__).resolve().parent
    return [dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted(root.glob('*.py'))]


def install():
    """Call at entrypoint module top-level, so every spawn child also installs."""
    global _INSTALLED
    mode=os.getenv(MODE_ENV,'')
    if not mode:
        return None
    if mode!=MODE:
        raise RuntimeError(f'Unsupported {MODE_ENV}={mode!r}')
    if _INSTALLED:
        return provenance()
    if any(name=='rtp_ops' or name=='rtp_kernel' or name.startswith('rtp_kernel.') for name in sys.modules):
        raise RuntimeError('Install explicit ARM BF16 backend before importing any rtp_kernel/rtp_ops module')
    backend=importlib.import_module('rtp_arm_bf16_rope')
    package=types.ModuleType('rtp_kernel',
        'Explicit project-owned BF16 compatibility API, not the rtp_kernel wheel.')
    package.__path__=[]
    package.__file__=backend.__file__
    package.__spec__=ModuleSpec('rtp_kernel',loader=None,is_package=True)
    package.__spec__.origin=backend.__file__
    package.__backend__=MODE
    package.__project_compatibility_api__=True
    wrapper=types.ModuleType('rtp_kernel.fused_rope_kvcache',backend.__doc__)
    wrapper.__file__=backend.__file__
    wrapper.__package__='rtp_kernel'
    wrapper.__spec__=ModuleSpec(wrapper.__name__,loader=None,origin=backend.__file__)
    wrapper.__backend__=MODE
    names=['prefill_fused_rope_kvcache','decode_fused_rope_kvcache',
           'convert_offset_to_block_array','get_base_rope_cache','get_yarn_rope_cache']
    for name in names:
        setattr(wrapper,name,getattr(backend,name))
    extraction=types.ModuleType('rtp_kernel.kvcache',backend.__doc__)
    extraction.__file__=backend.__file__
    extraction.__package__='rtp_kernel'
    extraction.__spec__=ModuleSpec(extraction.__name__,loader=None,origin=backend.__file__)
    extraction.__backend__=MODE
    for name in ['kvcache_extract_prefill','kvcache_extract_decode']:
        setattr(extraction,name,getattr(backend,name))
    ops=types.ModuleType('rtp_ops',
        'Explicit project-owned Torch BF16 operators; no binary CUDA wheel is loaded.')
    ops.__file__=backend.__file__
    ops.__spec__=ModuleSpec('rtp_ops',loader=None,origin=backend.__file__)
    ops.__backend__=MODE
    ops.__project_compatibility_api__=True
    for name in names[:3]+['ExtractKVFromPagedCachePrefill','ExtractKVFromPagedCacheDecode']:
        setattr(ops,name,getattr(backend,name))
    package.fused_rope_kvcache=wrapper
    package.kvcache=extraction
    sys.modules.update({'rtp_kernel':package,'rtp_kernel.fused_rope_kvcache':wrapper,
                        'rtp_kernel.kvcache':extraction,'rtp_ops':ops})
    _INSTALLED=True
    return provenance()


def install_canonical_fusion_guard():
    """Keep actual canonical Triton QK norm/RoPE fastpath, reject quantization."""
    if not _INSTALLED:
        return None
    module=importlib.import_module('rtp_llm.models_py.triton_kernels.common.qk_norm_rope_cache')
    if getattr(module,'_rtp_arm_bf16_guard_installed',False):
        return None
    import torch
    original=module.fused_qk_norm_rope_cache
    def fused(qkv,q_weight,k_weight,cos_sin_cache,position_ids,cu_seqlens,prefix_lengths,kv_cache,kv_offsets,**kwargs):
        if (qkv.dtype!=torch.bfloat16 or kv_cache.dtype!=torch.bfloat16
                or kwargs.get('fp8_query') or kwargs.get('kv_scales') is not None):
            raise ValueError('Explicit ARM BF16 backend rejects FP8/quantized QK fusion')
        return original(qkv,q_weight,k_weight,cos_sin_cache,position_ids,cu_seqlens,prefix_lengths,
                        kv_cache,kv_offsets,**kwargs)
    fused.__name__=original.__name__
    fused.__doc__=original.__doc__
    module.fused_qk_norm_rope_cache=fused
    original_dispatch=module.maybe_fused_qk_norm_rope_cache
    def dispatch(qkv,norm,impl,kv_cache):
        if (qkv.dtype!=torch.bfloat16 or getattr(impl,'native_fp8',False)
                or (kv_cache is not None and kv_cache.kv_cache_base.dtype!=torch.bfloat16)):
            raise ValueError('Explicit ARM BF16 backend rejects non-BF16 full attention')
        return original_dispatch(qkv,norm,impl,kv_cache)
    module.maybe_fused_qk_norm_rope_cache=dispatch
    module._rtp_arm_bf16_guard_installed=True
    source=Path(getattr(module,'__overlay_source__',module.__file__))
    return dict(module=module.__name__,canonical_source=str(source),
                canonical_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None,
                canonical_triton_retained=True,bf16_only_guard=True)


def verify_rtp_decode_dispatch():
    """Fail at startup if the deployed caller lacks the explicit-length ABI."""
    if not _INSTALLED:
        return None
    module=importlib.import_module('rtp_llm.ops.fused_rope_kvcache_op')
    import rtp_arm_bf16_rope as backend
    lengths=getattr(module,'_decode_uses_sequence_lengths',None)
    cumulative=getattr(module,'_decode_uses_cu_seqlens',None)
    if (lengths is None or cumulative is None or not lengths(backend.decode_fused_rope_kvcache)
            or cumulative(backend.decode_fused_rope_kvcache)):
        raise RuntimeError('ARM BF16 compatibility requires the accepted RTP decode caller with explicit sequence_lengths introspection')
    source=Path(module.__file__)
    return dict(caller_source=str(source),
        caller_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None,
        sequence_lengths_used=True,context_cu_seqlens_not_passed=True)


def provenance():
    import torch
    return dict(backend=MODE,explicit_project_source=True,external_rtp_kernel_wheel=False,
        architecture=platform.machine(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        source_manifest=source_manifest(),
        supported=['BF16 packed QKV','Base and Qwen3.5 interleaved MRoPE','partial NeoX rotary',
                   'RTP HND paged KV','one-token decode with explicit physical sequence_lengths'],
        rejected=['FP8','FP16','quantized KV scales','YaRN/scaled RoPE','LogN',
                  'padded Q output','speculative multi-token decode','fallback CUDA graph capture'])
