"""Single leased BF16 CUDA backing for bounded chunk-to-canonical RTP refit.

The consumer must acknowledge the real all-rank native copies before requesting
the next key. A failed refit retains its live source until native teardown.
"""
from __future__ import annotations

from collections import defaultdict
import gc
import hashlib
import math
import weakref
import torch
import torch.distributed as dist


def export_dtype(name):
    return torch.float32 if name.endswith(('.linear_attn.A_log','.linear_attn.dt_bias')) else torch.bfloat16


class CanonicalExport:
    def __init__(self, model, *, expected_world_size=72, collective_device=None):
        if not dist.is_initialized() or dist.get_world_size()!=expected_world_size:
            raise ValueError('Refit requires every actual training/export rank in the fixed collective order')
        self.model=model;self.world_size=expected_world_size
        self.device=(torch.device(collective_device) if collective_device is not None
                     else torch.device('cuda',torch.cuda.current_device()))
        self.sources=model._qwen38_expert_sources
        self.parameters=dict(model.named_parameters());self.groups=defaultdict(list)
        for name in self.parameters:
            mapped=self.sources.get(name)
            self.groups[mapped['canonical_name'] if mapped else name].append(name)
        self.order=sorted(self.groups)
        self._order_digest=hashlib.sha256('\0'.join(self.order).encode()).hexdigest()
        maximum=0
        for key,names in self.groups.items():
            mapped=self.sources.get(names[0])
            shape=mapped['source_shape'] if mapped else self.parameters[names[0]].shape
            bytes_per_value=4 if export_dtype(key)==torch.float32 else 2
            maximum=max(maximum,math.prod(shape)*bytes_per_value)
        if maximum%4:maximum+=4-maximum%4
        self.maximum_backing_bytes=maximum
        self.backing=None;self.pending=None;self.cursor=0;self.receipts=[];self._views=[]
        self.closed=False;self.release_receipt=None
        self._checked_local('constructor', self._validate_metadata)

    def _checked_local(self, boundary, action):
        result=None;error=None
        try:result=action()
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        statuses=[None for _ in range(self.world_size)]
        dist.all_gather_object(statuses, {'rank':dist.get_rank(),'boundary':boundary,'error':error})
        errors=[row for row in statuses if row['error'] is not None or row['boundary']!=boundary]
        if errors:raise RuntimeError(f'All-rank canonical source failure at {boundary}: {errors}')
        return result

    def _validate_metadata(self):
        from torch.distributed.tensor import DTensor, Shard
        for key,names in self.groups.items():
            for name in names:
                p=self.parameters[name]
                if not isinstance(p,DTensor) or p.device_mesh.ndim!=1 or p.device_mesh.size()!=self.world_size:
                    raise ValueError('Canonical source must be the actual single-world DTensor policy')
                if len(p.placements)!=1 or type(p.placements[0]) is not Shard or p.placements[0].dim!=0:
                    raise ValueError('Canonical source must preserve the reviewed first-axis shard layout')
                if p.dtype!=torch.bfloat16:
                    raise ValueError('Persistent policy source must be BF16 after stochastic rounding')
        return True

    def __iter__(self):return self

    @torch.no_grad()
    def __next__(self):
        # All callers enter this boundary in the same order. Failures before a
        # tensor collective are reconciled before another rank can enter it.
        def ready():
            if self.pending is not None:
                raise RuntimeError('Do not reuse leased CUDA storage before the real all-rank native ACK')
            if self.closed and self.cursor!=len(self.order):
                raise RuntimeError('An incomplete closed source iterator cannot resume')
            return self.cursor,self.closed,self._order_digest
        state=self._checked_local('source-next-ready',ready)
        states=[None for _ in range(self.world_size)];dist.all_gather_object(states,state)
        if any(value!=state for value in states):
            raise RuntimeError('Canonical source cursors or key orders differ across training ranks')
        if self.closed:raise StopIteration
        if self.cursor==len(self.order):raise StopIteration
        key=self.order[self.cursor];names=self.groups[key];mapped=self.sources.get(names[0])
        shape=tuple(mapped['source_shape']) if mapped else tuple(self.parameters[names[0]].shape)
        dtype=export_dtype(key)
        if self.backing is None:
            self.backing=self._checked_local('source-backing-allocation',lambda:
                torch.empty(self.maximum_backing_bytes//2,device=self.device,dtype=torch.bfloat16))
        typed=self.backing if dtype==torch.bfloat16 else self.backing.view(torch.float32)
        result=typed[:math.prod(shape)].view(shape)
        largest_chunk=0
        if mapped:
            vocab='vocab_row_begin' in mapped
            begin_key,end_key=('vocab_row_begin','vocab_row_end') if vocab else ('expert_begin','expert_end')
            ordered=sorted(names,key=lambda name:self.sources[name][begin_key])
            cursor=0
            for name in ordered:
                info=self.sources[name]
                if info[begin_key]!=cursor:
                    raise ValueError('Canonical expert assembly omitted or repeated an expert interval')
                parameter=self.parameters[name]
                # Cast the owned FP32 shard BEFORE any global materialization.
                # For chunk16 this gathers 1GiB BF16, never 64GiB FP32 packed.
                owned=self._checked_local('source-owned-'+name,lambda:
                    parameter.detach().to(device=self.device,dtype=dtype,non_blocking=False))
                whole=owned.full_tensor()
                count=info[end_key]-info[begin_key]
                def assemble():
                    result[info[begin_key]:info[end_key]].copy_(whole.view(count,*shape[1:]))
                    torch.cuda.synchronize(self.device)
                self._checked_local('source-assemble-'+name,assemble)
                largest_chunk=max(largest_chunk,whole.numel()*whole.element_size())
                del whole,owned
                cursor=info[end_key]
            if cursor!=shape[0]:raise ValueError('Incomplete canonical packed parameter')
        else:
            if len(names)!=1:raise ValueError('Unexpected many-to-one nonexpert canonical source')
            owned=self._checked_local('source-owned-'+key,lambda:
                self.parameters[names[0]].detach().to(device=self.device,dtype=dtype,non_blocking=False))
            whole=owned.full_tensor()
            def assemble():
                result.copy_(whole);torch.cuda.synchronize(self.device)
            self._checked_local('source-assemble-'+key,assemble)
            largest_chunk=whole.numel()*whole.element_size();del whole,owned
        self.pending=key
        self._views.append(weakref.ref(result))
        self.receipts.append({'name':key,'shape':list(shape),'dtype':str(dtype),
                              'backing_bytes':self.maximum_backing_bytes,
                              'maximum_full_chunk_bytes':largest_chunk,
                              'full_FP32_packed_parameter_created':False,
                              'native_all_rank_acknowledged':False})
        return key,result

    def acknowledge(self, key, *, successful_native_ranks):
        if key!=self.pending:raise ValueError('ACK does not match the outstanding source lease')
        ranks=list(successful_native_ranks)
        if len(ranks)!=self.world_size or set(ranks)!=set(range(self.world_size)):
            raise RuntimeError('Every unique native world rank must have actually copied the leased source')
        self.receipts[-1]['native_all_rank_acknowledged']=True
        self.pending=None;self.cursor+=1

    def close(self, *, all_owned_native_contexts_released=False):
        if self.closed:return self.release_receipt
        if ((self.pending is not None or self.cursor!=len(self.order))
                and all_owned_native_contexts_released is not True):
            raise RuntimeError('Failed/unacknowledged native refit must be torn down before freeing its borrowed source')
        live=sum(reference() is not None for reference in self._views)
        if live:raise RuntimeError('Delete every consumer CUDA source view before closing the refit backing')
        before=torch.cuda.memory_allocated(self.device) if self.backing is not None else 0
        reference=weakref.ref(self.backing) if self.backing is not None else lambda:None
        self.backing=None;self.pending=None;gc.collect()
        if reference() is not None:raise RuntimeError('An external reference still pins the CUDA refit backing')
        torch.cuda.synchronize(self.device);torch.cuda.empty_cache()
        after=torch.cuda.memory_allocated(self.device)
        self.release_receipt={'all_source_views_released':True,'backing_released':True,
            'source_backing_bytes':self.maximum_backing_bytes,
            'cuda_allocated_bytes_before_release':before,'cuda_allocated_bytes_after_release':after,
            'native_RTP_context_cleanup_proven_here':False}
        self.closed=True
        return self.release_receipt
