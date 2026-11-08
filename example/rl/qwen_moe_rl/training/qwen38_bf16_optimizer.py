"""BF16 policy/Adam storage with independent FP32 reduced-gradient capture.

Only bounded optimizer chunks are converted to FP32. Two persistent Adam
states live in CPU BF16 memory, with no mapped files or disk fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
import functools
import hashlib
import json
import inspect
import math
from pathlib import Path
import threading
import weakref

import torch

TORCH_FOREACH_REDUCE_SHA256 = 'cc2a17ca7325d89e14b7d262c7900b1d27187f0f7321b6b7c6e6bb11f8ef17f9'
_REDUCED_OWNERS = {}
_ORIGINAL_REDUCE = None


def _private_anonymous_mappings():
    """Linux-only conservative proof: do not credit file/shared-backed state."""
    rows=[]
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(maxsplit=5);left,right=(int(v,16) for v in fields[0].split('-'))
        path=fields[5] if len(fields)>5 else ''
        anonymous=(fields[4]=='0' and (not path or path=='[heap]' or path.startswith('[anon:')))
        rows.append((left,right,fields[1]=='rw-p' and anonymous))
    rows.sort();return rows,[row[0] for row in rows]


def _require_private_anonymous(pointer,size,maps):
    rows,starts=maps;index=bisect_right(starts,pointer)-1;end=pointer+size;current=pointer
    while current<end:
        if index<0 or index>=len(rows):raise RuntimeError('CPU storage is not fully mapped')
        left,right,safe=rows[index]
        if not safe or not left<=current<right:raise RuntimeError('CPU storage is not private anonymous memory')
        current=min(end,right);index+=1


def local(tensor):
    return tensor.to_local() if hasattr(tensor, 'to_local') else tensor


def _signed64(value):
    value = int(value) & ((1 << 64) - 1)
    return value if value < (1 << 63) else value - (1 << 64)


def rounding_key(name, step, role, seed=0):
    encoded = f'q38-bf16-sr-v1\0{seed}\0{name}\0{step}\0{role}'.encode()
    return _signed64(int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'little'))


def _logical_shift(value, bits):
    return (value >> bits) & ((1 << (64 - bits)) - 1)


def counter_random16(count, *, key, offset, device):
    """SplitMix64 counter permutation; output independent of chunk boundaries.

    Counter coordinates are global within one named runtime parameter, so
    changing its FSDP shard boundaries does not change its rounding decisions.
    """
    if count < 0 or offset < 0 or offset + count >= 1 << 63:
        raise ValueError('Invalid canonical parameter counter interval')
    value = torch.arange(offset, offset + count, dtype=torch.int64, device=device)
    value = value + _signed64(key)
    value = value + _signed64(0x9E3779B97F4A7C15)
    value = (value ^ _logical_shift(value, 30)) * _signed64(0xBF58476D1CE4E5B9)
    value = (value ^ _logical_shift(value, 27)) * _signed64(0x94D049BB133111EB)
    value = value ^ _logical_shift(value, 31)
    return _logical_shift(value, 48)


@torch.no_grad()
def stochastic_bf16(value, *, key, offset=0):
    """Round finite FP32 to neighboring BF16 values with unbiased probabilities."""
    if value.dtype != torch.float32 or not value.is_contiguous():
        raise TypeError('The bounded stochastic-rounding input must be contiguous FP32')
    if not torch.isfinite(value).all().item():
        raise FloatingPointError('Refusing a nonfinite BF16 policy or Adam state')
    shape = value.shape
    bits = value.view(torch.int32).reshape(-1).to(torch.int64) & 0xFFFFFFFF
    lower = bits & 0xFFFF
    upper = bits >> 16
    random = counter_random16(value.numel(), key=key, offset=offset, device=value.device)
    rounded = (upper + (random < lower).to(torch.int64)) & 0xFFFF
    result = rounded.to(torch.int16).view(torch.bfloat16).reshape(shape)
    if not torch.isfinite(result).all().item():
        raise FloatingPointError('Stochastic BF16 rounding overflowed; the update must abort')
    return result


@dataclass
class GradientSlot:
    parameter: object
    gradient: torch.Tensor
    tier: str
    name: str
    contributions: int = 1


def _install_reduction_capture():
    global _ORIGINAL_REDUCE
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives as collectives
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group as param_group
    if _ORIGINAL_REDUCE is not None:
        if collectives.foreach_reduce is not param_group.foreach_reduce:
            raise RuntimeError('Another hook replaced only one FSDP reduction entry point')
        return
    original = collectives.foreach_reduce
    if (hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
            != TORCH_FOREACH_REDUCE_SHA256 or param_group.foreach_reduce is not original):
        raise RuntimeError('Review the exact Torch2.13 reduction implementation before interception')
    signature = inspect.signature(original)

    @functools.wraps(original)
    def reduced(*args, **kwargs):
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        options = call.arguments
        params = options['fsdp_params']
        owners = [_REDUCED_OWNERS.get(id(param.sharded_param)) for param in params]
        if not any(owners):
            return original(*args, **kwargs)
        if not all(owners) or len({id(item[1]) for item in owners}) != 1:
            raise RuntimeError('A BF16 capture group must have one complete gradient-store owner')
        store = owners[0][1]
        if any(ref() is not param.sharded_param for (ref, _), param in zip(owners, params)):
            raise RuntimeError('A reduced parameter identity no longer matches its gradient owner')
        if (options['all_reduce_group'] is not None or options['all_reduce_hook'] is not None
                or options['partial_reduce_output'] is not None):
            raise ValueError('Initial BF16 qualification supports ordinary 1D FSDP without another all-reduce hook')
        if options['reduce_dtype'] != torch.float32:
            raise ValueError('BF16 parameter storage still requires real FP32 gradient reduction')
        world = options['reduce_scatter_group'].size() if options['reduce_scatter_group'] is not None else 1
        padded = [collectives._get_dim0_padded_size(grad.size(), world)
                  for grad in options['unsharded_grads']]
        if any(param.fsdp_placement.dim != 0 for param in params):
            raise ValueError('The reviewed BF16 policy requires first-axis FSDP shards')
        _, postdivide, _, _ = collectives._get_gradient_divide_factors(
            options['reduce_scatter_group'], None, torch.float32, options['device'].type,
            options['gradient_divide_factor'], options['force_sum_reduction_for_comms'])

        def capture_before_orig_dtype(buffer):
            if buffer.dtype != torch.float32:
                raise RuntimeError('The source reduction buffer was already downcast')
            if buffer.is_cuda:
                # The original implementation moves post-reduce work to this
                # stream for user hooks. Keep its RS-allocated buffer alive
                # until the hook and later dtype conversion have completed.
                buffer.record_stream(torch.cuda.current_stream(buffer.device))
            offset = 0
            for size, param in zip(padded, params):
                value = torch.as_strided(buffer, size=param.sharded_size,
                    stride=param.contiguous_sharded_stride, storage_offset=offset)
                store.capture_reduced(param.sharded_param, value, divide=postdivide)
                offset += size.numel() // world
            if offset != buffer.numel():
                raise RuntimeError('The captured FP32 shards do not cover the reduction buffer')

        options['all_reduce_hook'] = capture_before_orig_dtype
        return original(*call.args, **call.kwargs)

    reduced._q38_fp32_before_bf16_capture = True
    _ORIGINAL_REDUCE = original
    collectives.foreach_reduce = reduced
    param_group.foreach_reduce = reduced


class GradientStore:
    """Independent FP32 shards captured before Torch casts to BF16 .grad."""
    def __init__(self, *, retained_bytes, retained_device='cuda', norm_chunk_bytes=8 * 1024**2,
                 accumulation_chunk_bytes=8 * 1024**2):
        if retained_bytes < 0 or min(norm_chunk_bytes, accumulation_chunk_bytes) < 4:
            raise ValueError('Invalid bounded gradient storage')
        self.limit = retained_bytes
        self.retained_device = retained_device
        self.norm_chunk_numel = norm_chunk_bytes // 4
        self.accumulation_chunk_numel = accumulation_chunk_bytes // 4
        self.slots = {}
        self.retained_bytes = self.cpu_spilled_bytes = 0
        self.peak_retained_bytes = self.peak_spilled_bytes = 0
        self.generation = 0
        self.clipped_generation = None
        self.last_clip = None
        self._hooks = []
        self._params = []
        self._lock = threading.RLock()
        self._cleared_contributions = {}
        self._reuse_failed = False;self._reuse_frozen=False
        self._q38_capture_contract = None

    def bind_gpu_capture_contract(self, parameters, expected_gpu):
        """Freeze shard/owner inputs and derive an order-independent CPU cap."""
        from torch.distributed.fsdp._fully_shard import _fsdp_collectives as collectives
        from torch.distributed.fsdp._fully_shard import _fsdp_param_group as param_group
        with self._lock:
            if (self._q38_capture_contract is not None or self._reuse_failed or self._reuse_frozen
                    or self.slots or self.retained_bytes or self.cpu_spilled_bytes):
                raise RuntimeError('Capture contract requires a clean, unbound gradient store')
            params=tuple(parameters)
            if (not params or len({id(p) for p in params})!=len(params)
                    or len(self._params)!=len(params) or len(self._hooks)!=len(params)
                    or {id(p) for p in self._params}!={id(p) for p in params}
                    or self.retained_device!='cuda' or self.limit<0 or self.limit%4):
                raise ValueError('Captured parameters, hooks or CUDA cache limit disagree')
            if (collectives.foreach_reduce is not param_group.foreach_reduce
                    or not getattr(collectives.foreach_reduce,'_q38_fp32_before_bf16_capture',False)):
                raise RuntimeError('Reviewed FP32 FSDP reduction hook is not active')
            expected_gpu=torch.device(expected_gpu)
            if expected_gpu.type!='cuda' or expected_gpu.index is None:
                raise ValueError('Expected an explicit actor CUDA device')
            sizes={};total=largest=0
            for p in params:
                owner=_REDUCED_OWNERS.get(id(p));v=local(p.detach())
                if (not p.requires_grad or owner is None or owner[0]() is not p or owner[1] is not self
                        or v.device!=expected_gpu or not v.is_contiguous()
                        or v.dtype not in (torch.bfloat16,torch.float32)):
                    raise ValueError('Actual trainable shard identity, device or hook owner disagrees')
                size=v.numel()*4;sizes[p]=(tuple(v.shape),size);total+=size;largest=max(largest,size)
            # If any shard is rejected, before that rejection retained GPU
            # bytes exceed cache-largest; a never-rejected set spills zero.
            upper=min(total,max(0,total-self.limit+largest))
            self._q38_capture_contract=dict(params=sizes,limit=self.limit,device=expected_gpu,
                total_bytes=total,largest_bytes=largest,cpu_upper_bytes=upper,
                slot_ledger={},storage_ids=set(),expected_gpu_bytes=0,expected_cpu_bytes=0)
            return dict(total_bytes=total,largest_bytes=largest,cache_bytes=self.limit,cpu_upper_bytes=upper,
                parameter_count=len(params),device=str(expected_gpu),all_owners_verified=True)

    def finish_gpu_capture_contract(self):
        with self._lock:
            if (self._q38_capture_contract is None or self._reuse_failed or self._reuse_frozen
                    or self.slots or self.retained_bytes or self.cpu_spilled_bytes):
                raise RuntimeError('Successful batch did not release its captured gradients')
            self._q38_capture_contract=None

    def _check_gpu_capture_contract(self,parameter,gradient):
        contract=self._q38_capture_contract
        if contract is None:return
        owner=_REDUCED_OWNERS.get(id(parameter));entry=contract['params'].get(parameter)
        if (entry is None or owner is None or owner[0]() is not parameter or owner[1] is not self
                or not parameter.requires_grad or self.limit!=contract['limit'] or self.retained_device!='cuda'
                or gradient.device!=contract['device'] or gradient.dtype!=torch.float32
                or not gradient.is_contiguous() or tuple(gradient.shape)!=entry[0]
                or gradient.numel()*4!=entry[1] or entry[1]>contract['largest_bytes']
                or self.retained_bytes!=contract['expected_gpu_bytes']
                or self.cpu_spilled_bytes!=contract['expected_cpu_bytes']
                or self.retained_bytes>contract['limit'] or self.cpu_spilled_bytes>contract['cpu_upper_bytes']):
            raise RuntimeError('FP32 gradient violated the frozen CUDA capture contract')
        if parameter in self.slots:self._check_ledger_slot(parameter,self.slots[parameter])
        if (parameter not in self.slots and self.retained_bytes+entry[1]>self.limit
                ):
            self._check_all_ledger_slots()
            if self.cpu_spilled_bytes+entry[1]>contract['cpu_upper_bytes']:
                raise MemoryError('Captured CPU gradient would exceed its admitted bound')

    def _check_ledger_slot(self,parameter,slot):
        contract=self._q38_capture_contract;record=contract['slot_ledger'].get(parameter)
        if record is None or slot.gradient is None:raise RuntimeError('Missing captured-slot ledger entry')
        grad=slot.gradient;storage=grad.untyped_storage();identity=(int(storage._cdata),int(storage.data_ptr()))
        expected_device=contract['device'] if record['tier']=='gpu' else torch.device('cpu')
        if (slot.parameter is not parameter or slot.tier!=record['tier'] or grad.device!=expected_device
                or grad.dtype!=torch.float32 or not grad.is_contiguous() or tuple(grad.shape)!=record['shape']
                or grad.numel()*4!=record['bytes'] or id(grad)!=record['tensor_id']
                or identity!=record['storage'] or slot._q38_captured_storage!=identity):
            raise RuntimeError('Existing gradient slot differs from its capture ledger')

    def _check_all_ledger_slots(self):
        contract=self._q38_capture_contract
        if contract is None:return
        if len(self.slots)!=len(contract['slot_ledger']):raise RuntimeError('Gradient slot count changed')
        gpu=cpu=0;seen=set()
        for parameter,slot in self.slots.items():
            self._check_ledger_slot(parameter,slot);r=contract['slot_ledger'][parameter]
            if r['bytes']:
                if r['storage'][0] in seen:raise RuntimeError('Two slots share a gradient storage')
                seen.add(r['storage'][0])
            if r['tier']=='gpu':gpu+=r['bytes']
            else:cpu+=r['bytes']
        if (seen!=contract['storage_ids'] or gpu!=self.retained_bytes or cpu!=self.cpu_spilled_bytes
                or gpu!=contract['expected_gpu_bytes'] or cpu!=contract['expected_cpu_bytes']):
            raise RuntimeError('Gradient slot ledger accounting changed')

    def _wait_gpu_slot(self,parameter,slot):
        contract=self._q38_capture_contract
        if contract is None or slot.tier!='gpu':return
        event=contract['slot_ledger'][parameter]['event'];stream=torch.cuda.current_stream(contract['device'])
        stream.wait_event(event);slot.gradient.record_stream(stream)

    def _record_gpu_slot(self,parameter,slot):
        contract=self._q38_capture_contract
        if contract is None or slot.tier!='gpu':return
        stream=torch.cuda.current_stream(contract['device']);slot.gradient.record_stream(stream)
        event=torch.cuda.Event();event.record(stream);contract['slot_ledger'][parameter]['event']=event

    @torch.no_grad()
    def capture_reduced(self, parameter, gradient, *, divide=None):
        with self._lock:
            try:
                contract=self._q38_capture_contract;is_new=parameter not in self.slots
                self._check_gpu_capture_contract(parameter,gradient)
                if contract is not None and not is_new:self._wait_gpu_slot(parameter,self.slots[parameter])
                result=self._capture_reduced_impl(parameter,gradient,divide=divide)
                if contract is not None:
                    slot=self.slots[parameter]
                    if is_new:
                        grad=slot.gradient;storage=grad.untyped_storage();identity=(int(storage._cdata),int(storage.data_ptr()))
                        size=grad.numel()*4
                        if size and identity[0] in contract['storage_ids']:
                            raise RuntimeError('New captured slot aliases an existing gradient storage')
                        if size:contract['storage_ids'].add(identity[0])
                        contract['slot_ledger'][parameter]=dict(tensor_id=id(grad),shape=tuple(grad.shape),bytes=size,
                            tier=slot.tier,storage=identity,event=None)
                        key='expected_gpu_bytes' if slot.tier=='gpu' else 'expected_cpu_bytes';contract[key]+=size
                    self._check_ledger_slot(parameter,slot)
                    if (self.retained_bytes!=contract['expected_gpu_bytes']
                            or self.cpu_spilled_bytes!=contract['expected_cpu_bytes']):
                        raise RuntimeError('New gradient slot was not correctly charged')
                    self._record_gpu_slot(parameter,slot)
                return result
            except BaseException:
                if self._q38_capture_contract is not None:self._reuse_failed=True
                raise

    def _capture_reduced_impl(self, parameter, gradient, *, divide=None):
        value = local(parameter.detach())
        if gradient.dtype != torch.float32 or value.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError('Capture must retain FP32 reduction, with BF16 policy storage')
        if gradient.shape != value.shape or not gradient.is_contiguous():
            raise ValueError('Reduced gradient must match the real local parameter shard')
        size = gradient.numel() * 4
        with self._lock:
            if self._reuse_failed or self._reuse_frozen:raise RuntimeError('Frozen or poisoned gradient store cannot accept gradients')
            if self.clipped_generation == self.generation:
                raise RuntimeError('Backward cannot append to already clipped gradients')
            if parameter in self.slots:
                slot = self.slots[parameter]
                if slot.gradient is None:raise RuntimeError('Consumed gradient slot cannot accept gradients before clear')
                previous = slot.gradient.reshape(-1)
                incoming = gradient.reshape(-1)
                for begin in range(0, previous.numel(), self.accumulation_chunk_numel):
                    end = min(previous.numel(), begin + self.accumulation_chunk_numel)
                    piece = incoming[begin:end].to(previous.device, copy=True, non_blocking=False)
                    if divide is not None:
                        piece.div_(divide)
                    previous[begin:end].add_(piece)
                slot.contributions += 1
            else:
                keep = gradient.device.type == self.retained_device and self.retained_bytes + size <= self.limit
                captured = gradient.clone() if keep else gradient.to('cpu', copy=True, non_blocking=False)
                if divide is not None:
                    captured.div_(divide)
                name = getattr(parameter, '_qwen36_parameter_name', 'parameter-' + str(id(parameter)))
                slot=GradientSlot(parameter, captured, 'gpu' if keep else 'cpu', name)
                source_storage=captured.untyped_storage()
                slot._q38_captured_storage=(int(source_storage._cdata),int(source_storage.data_ptr()))
                self.slots[parameter] = slot
                if keep:
                    self.retained_bytes += size
                else:
                    self.cpu_spilled_bytes += size
            self.generation += 1
            self.clipped_generation = None
            self.peak_retained_bytes = max(self.peak_retained_bytes, self.retained_bytes)
            self.peak_spilled_bytes = max(self.peak_spilled_bytes, self.cpu_spilled_bytes)

    def _clear_shadow_gradient(self, parameter):
        with self._lock:return self._clear_shadow_gradient_locked(parameter)

    def _clear_shadow_gradient_locked(self, parameter):
        if self._reuse_failed or self._reuse_frozen:
            raise RuntimeError('Frozen or poisoned gradient store cannot clear shadow gradients')
        if parameter not in self.slots:
            raise RuntimeError('A BF16 .grad appeared without a preceding FP32 reduction capture')
        count = self.slots[parameter].contributions
        if count <= self._cleared_contributions.get(id(parameter), 0):
            raise RuntimeError('A post-accumulate hook has no fresh FP32 contribution')
        parameter.grad = None
        self._cleared_contributions[id(parameter)] = count

    def register_hooks(self, parameters, *, require_dtensor=True):
        from torch.distributed.tensor import DTensor
        _install_reduction_capture()
        for parameter in parameters:
            if require_dtensor and not isinstance(parameter, DTensor):
                raise ValueError('Production capture requires actual FSDP2 sharded parameters')
            if not parameter.requires_grad:
                continue
            previous = _REDUCED_OWNERS.get(id(parameter))
            if previous is not None:
                raise RuntimeError('A parameter already has a reduced-gradient owner')
            _REDUCED_OWNERS[id(parameter)] = (weakref.ref(parameter), self)
            self._params.append(parameter)
            self._hooks.append(parameter.register_post_accumulate_grad_hook(self._clear_shadow_gradient))
        return len(self._hooks)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        for parameter in self._params:
            owner = _REDUCED_OWNERS.get(id(parameter))
            if owner is not None and owner[1] is self:
                del _REDUCED_OWNERS[id(parameter)]
        self._hooks.clear()
        self._params.clear()

    @torch.no_grad()
    def clip_grad_norm(self, max_norm, *, expected_local_numel=None, expected_global_numel=None,
                       process_group=None, collective_device=None):
        with self._lock:
            try:
                self._check_all_ledger_slots()
                for parameter,slot in self.slots.items():self._wait_gpu_slot(parameter,slot)
                result=self._clip_grad_norm(max_norm,expected_local_numel=expected_local_numel,
                    expected_global_numel=expected_global_numel,process_group=process_group,collective_device=collective_device)
                for parameter,slot in self.slots.items():self._record_gpu_slot(parameter,slot)
                return result
            except BaseException:
                if self._q38_capture_contract is not None:self._reuse_failed=True
                raise

    @torch.no_grad()
    def _clip_grad_norm(self, max_norm, *, expected_local_numel=None, expected_global_numel=None,
                       process_group=None, collective_device=None):
        if self._reuse_failed or self._reuse_frozen or any(slot.gradient is None for slot in self.slots.values()):
            raise RuntimeError('Consumed or poisoned gradients cannot be clipped')
        if not math.isfinite(float(max_norm)) or max_norm <= 0:
            raise ValueError('A finite positive clipping bound is required')
        owned = sum(slot.gradient.numel() for slot in self.slots.values())
        if expected_local_numel is not None and owned != expected_local_numel:
            raise RuntimeError('Missing or extra local FP32 gradient elements before clipping')
        squares = 0.0
        for slot in self.slots.values():
            flat = slot.gradient.reshape(-1)
            for begin in range(0, flat.numel(), self.norm_chunk_numel):
                squares += float(flat[begin:begin+self.norm_chunk_numel].double().square().sum().item())
        local_norm = math.sqrt(squares)
        import torch.distributed as dist
        if dist.is_initialized():
            if collective_device is None:
                raise ValueError('The exact collective device must be explicit')
            scalar = torch.tensor([squares, owned], dtype=torch.float64, device=collective_device)
            dist.all_reduce(scalar, group=process_group)
            squares, global_owned = scalar.cpu().tolist()
            if expected_global_numel is None or global_owned != expected_global_numel:
                raise RuntimeError('Actual reduced shards do not cover the whole policy once')
        elif expected_global_numel is not None and owned != expected_global_numel:
            raise RuntimeError('A missing process group cannot prove whole-policy coverage')
        norm = math.sqrt(squares)
        if not math.isfinite(norm):
            raise FloatingPointError('Nonfinite global FP32 gradient norm')
        scale = min(1.0, float(max_norm) / (norm + 1e-6))
        for slot in self.slots.values():
            slot.gradient.mul_(scale)
        self.clipped_generation = self.generation
        self.last_clip = dict(local_accumulated_gradient_norm=local_norm,
                              global_accumulated_gradient_norm=norm, clip_bound=float(max_norm), scale=scale)
        return norm

    def clear(self):
        with self._lock:return self._clear_locked()

    def _clear_locked(self):
        if self._reuse_failed or self._reuse_frozen:raise RuntimeError('Frozen or poisoned gradient store cannot be reused')
        self._clear_contents_locked()

    def _clear_contents_locked(self):
        self.slots.clear()
        self._cleared_contributions.clear()
        self.retained_bytes = self.cpu_spilled_bytes = 0
        self.generation += 1
        self.clipped_generation = None

    def audit(self):
        with self._lock:return self._audit_locked()

    def _audit_locked(self):
        return dict(gradient_numel=sum(slot.gradient.numel() for slot in self.slots.values() if slot.gradient is not None),
                    gradient_count=sum(slot.gradient is not None for slot in self.slots.values()), retained_bytes=self.retained_bytes,
                    consumed_gradient_slots=sum(slot.gradient is None for slot in self.slots.values()),
                    cpu_spilled_bytes=self.cpu_spilled_bytes, peak_retained_bytes=self.peak_retained_bytes,
                    peak_cpu_spilled_bytes=self.peak_spilled_bytes, retained_budget_bytes=self.limit,
                    stored_gradient_dtype='float32', independent_gradient_storage=True,
                    captured_before_original_parameter_dtype_cast=True,
                    all_gradients_clipped=self.clipped_generation == self.generation,
                    microbatch_contributions_min=min((slot.contributions for slot in self.slots.values()), default=0),
                    microbatch_contributions_max=max((slot.contributions for slot in self.slots.values()), default=0),
                    last_clip=self.last_clip)


class CpuBf16Moments:
    def __init__(self, *, cpu_budget_bytes):
        if cpu_budget_bytes < 0:
            raise ValueError('Invalid CPU BF16 moment budget')
        self.limit = cpu_budget_bytes
        self.bytes = 0
        self.handles = []
        self.reused_gradient_bytes = 0

    def plan(self, named_parameters):
        elements = sum(local(parameter).numel() for _, parameter in named_parameters if parameter.requires_grad)
        size = elements * 4
        if size > self.limit:
            raise MemoryError('CPU BF16 Adam states exceed the hard budget; disk fallback is forbidden')
        return dict(planned_cpu_moment_bytes=size, planned_disk_moment_bytes=0,
                    fp32_persistent_moment_bytes=0, moment_elements_per_state=elements)

    def allocate(self, name, shape):
        count = math.prod(shape)
        size = count * 4
        if count < 1 or self.bytes + size > self.limit:
            raise MemoryError('CPU BF16 Adam states exceed the budget; no disk fallback exists')
        first = torch.empty(shape, dtype=torch.bfloat16, device='cpu')
        second = torch.empty_like(first)
        handle = dict(first=first, second=second, name=name, tier='cpu', dtype='bfloat16')
        self.bytes += size
        self.handles.append(handle)
        return handle

    def allocate_reusing_gradient(self, name, gradient):
        if gradient.device.type!='cpu' or gradient.dtype!=torch.float32 or not gradient.is_contiguous():
            raise ValueError('Reuse requires a contiguous CPU FP32 gradient')
        count=gradient.numel();size=count*4
        storage=gradient.untyped_storage()
        if (not count or self.bytes+size>self.limit or gradient.storage_offset()!=0
                or storage.nbytes()!=size or gradient.data_ptr()!=storage.data_ptr() or gradient.data_ptr()%4
                or gradient._base is not None):
            raise ValueError('Reuse requires one exclusive, aligned, whole-storage gradient')
        halves=gradient.view(-1).view(torch.bfloat16)
        first=halves[0::2].view(gradient.shape);second=halves[1::2].view(gradient.shape)
        handle=dict(first=first,second=second,name=name,tier='cpu_reused_gradient',dtype='bfloat16')
        self.bytes+=size;self.reused_gradient_bytes+=size;self.handles.append(handle)
        return handle

    def audit(self):
        return dict(cpu_moment_bytes=self.bytes, cpu_budget_bytes=self.limit,
                    cpu_moment_bytes_reusing_gradient=self.reused_gradient_bytes,
                    persistent_moment_dtype='bfloat16', disk_offload_bytes=0,
                    persistent_FP32_moment_bytes=0, handles=len(self.handles))


def global_parameter_offset(parameter):
    if not hasattr(parameter, 'device_mesh'):
        return int(getattr(parameter, '_q38_global_flat_offset', 0))
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    _, offset = compute_local_shape_and_global_offset(parameter.shape, parameter.device_mesh,
                                                     parameter.placements)
    if any(offset[1:]):
        raise ValueError('The counter layout requires first-axis FSDP shards')
    return int(offset[0]) * math.prod(parameter.shape[1:])


class Bf16CpuAdamW(torch.optim.Optimizer):
    """BF16 persistent weights/m/v, FP32 gradient and bounded update arithmetic."""
    def __init__(self, params, *, gradients, moments, lr=1e-6, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0, maximize=False, chunk_bytes=8 * 1024**2,
                 rounding_seed=20261106, require_clipped=True,reuse_cpu_spill=False,require_reuse_evidence=False):
        if chunk_bytes < 4 or chunk_bytes % 4:
            raise ValueError('The FP32 temporary chunk size must be a positive multiple of four')
        if not (0 <= lr and 0 <= eps and 0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError('Invalid AdamW hyperparameters')
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay, maximize=maximize))
        self.gradients = gradients
        self.moments = moments
        self.chunk_numel = chunk_bytes // 4
        self.rounding_seed = int(rounding_seed)
        self.require_clipped = require_clipped
        self.last_chunk_peak_numel = 0
        self.changed_weight_elements = 0
        self.reuse_cpu_spill=bool(reuse_cpu_spill);self._reuse_plan=None;self._reuse_failed=False
        self.require_reuse_evidence=bool(require_reuse_evidence);self._reuse_evidence=None;self._reuse_commit=None
        self._reuse_poststep_clear=False

    def prepare_reuse_plan(self):
        """Bind every slot to one optimizer-owned storage, without tensor data."""
        with self.gradients._lock:return self._prepare_reuse_plan_locked()

    def _prepare_reuse_plan_locked(self):
        if not self.reuse_cpu_spill or self._reuse_failed or self.gradients._reuse_failed:
            raise RuntimeError('First-step reuse is disabled or poisoned')
        if self.state or self.moments.bytes or self.gradients.clipped_generation!=self.gradients.generation:
            raise RuntimeError('Reuse requires first step and complete clipped gradients')
        params=[p for group in self.param_groups for p in group['params'] if p.requires_grad]
        if len(params)!=len(set(map(id,params))):raise RuntimeError('Duplicate optimizer parameter')
        required={p for p in params if local(p).numel()}
        actual=set(self.gradients.slots)
        if not required.issubset(actual) or not actual.issubset(set(params)):
            raise RuntimeError('Optimizer and gradient slot ownership differs')
        rows=[];seen=set();reusable=0
        maps=_private_anonymous_mappings()
        for parameter in params:
            value=local(parameter.detach());slot=self.gradients.slots.get(parameter)
            if not value.numel():
                if parameter.grad is not None:raise RuntimeError('Zero shard retains an unexpected shadow gradient')
                if slot is not None and (slot.parameter is not parameter or slot.gradient is None
                        or slot.gradient.shape!=value.shape or slot.gradient.dtype!=torch.float32
                        or slot.gradient.numel()!=0 or slot.contributions<1):
                    raise RuntimeError('Invalid zero-shard gradient slot')
                rows.append(dict(name=getattr(parameter,'_qwen36_parameter_name','parameter-'+str(id(parameter))),
                    tier='empty',numel=0,bytes=0,slot_present=slot is not None,
                    contributions=slot.contributions if slot is not None else None))
                continue
            grad=slot.gradient
            if (grad is None or parameter.grad is not None or grad.dtype!=torch.float32
                    or slot.parameter is not parameter or grad.shape!=value.shape or not grad.is_contiguous() or slot.contributions<1):
                raise RuntimeError('Invalid or incomplete clipped gradient slot')
            storage=grad.untyped_storage();identity=(str(grad.device),storage._cdata)
            if identity in seen:raise RuntimeError('Gradient slots alias one storage')
            seen.add(identity);size=grad.numel()*4
            storage_uses=torch._C._storage_Use_Count(storage._cdata)
            tensor_uses=grad._use_count()
            if (getattr(slot,'_q38_captured_storage',None)!=(int(storage._cdata),int(storage.data_ptr()))
                    or storage_uses!=2 or tensor_uses!=1):
                raise RuntimeError('Slot storage is not the unique capture-owned TensorImpl')
            if slot.tier=='cpu':
                if (grad.device.type!='cpu' or grad.storage_offset()!=0 or storage.nbytes()!=size
                        or grad.data_ptr()!=storage.data_ptr() or grad.data_ptr()%4 or grad._base is not None or storage.is_shared()):
                    raise RuntimeError('CPU slot is not an eligible whole owned allocation')
                _require_private_anonymous(storage.data_ptr(),size,maps)
                reusable+=size
            elif slot.tier!='gpu' or grad.device.type!='cuda':
                raise RuntimeError('Unexpected gradient tier/device')
            rows.append(dict(name=slot.name,tier=slot.tier,numel=grad.numel(),bytes=size,
                storage_id=int(storage._cdata),storage_pointer=int(storage.data_ptr()),
                storage_nbytes=storage.nbytes(),storage_offset=grad.storage_offset(),
                device=str(grad.device),shape=list(grad.shape),stride=list(grad.stride()),
                storage_use_count=storage_uses,tensor_use_count=tensor_uses,
                gradient_version=grad._version,contributions=slot.contributions))
        if reusable!=self.gradients.cpu_spilled_bytes:
            raise RuntimeError('Not every accounted CPU gradient is safely reusable')
        if len({row['name'] for row in rows})!=len(rows):raise RuntimeError('Ambiguous gradient names')
        return dict(schema='q38-first-adam-reuse-cpu-slot-plan-v1',generation=self.gradients.generation,
                    reusable_bytes=reusable,planned_moment_bytes=sum(r['bytes'] for r in rows),slots=rows)

    def bind_reuse_plan(self,plan):
        with self.gradients._lock:
            if self._prepare_reuse_plan_locked()!=plan:raise RuntimeError('First-step reuse plan changed before binding')
            self._reuse_plan=plan;self.gradients._reuse_frozen=True

    def attach_reuse_evidence(self,sha256,path):
        with self.gradients._lock:
            if self._reuse_plan is None or not self.gradients._reuse_frozen:
                raise RuntimeError('Reuse evidence requires a frozen plan')
            if not isinstance(sha256,str) or len(sha256)!=64 or any(c not in '0123456789abcdef' for c in sha256):
                raise ValueError('Invalid pre-update evidence SHA')
            self._reuse_evidence=dict(pre_update_sha256=sha256,pre_update_path=str(path))

    @torch.no_grad()
    def step(self, closure=None):
        if self._reuse_failed:raise RuntimeError('Poisoned first Adam cannot be replayed')
        if any(slot.gradient is None for slot in self.gradients.slots.values()):
            raise RuntimeError('Consumed gradients must be cleared before another optimizer step')
        first_reuse=self.reuse_cpu_spill and not self.state
        if first_reuse:
            if self._reuse_plan is None or self.prepare_reuse_plan()!=self._reuse_plan:
                raise RuntimeError('Unbound or changed first-Adam reuse plan')
            if self.require_reuse_evidence and self._reuse_evidence is None:
                raise RuntimeError('Missing complete pre-update gradient evidence')
        try:
            return self._step_impl(closure,first_reuse=first_reuse)
        except BaseException:
            if self.reuse_cpu_spill:self._reuse_failed=True;self.gradients._reuse_failed=True
            raise

    @torch.no_grad()
    def _step_impl(self, closure=None, *, first_reuse=False):
        if closure is not None:
            raise ValueError('This GRPO actor path has no optimizer closure')
        if self.require_clipped and self.gradients.clipped_generation != self.gradients.generation:
            raise RuntimeError('Complete global FP32 gradient clipping must precede this update')
        self.changed_weight_elements = 0
        self.last_chunk_peak_numel = 0
        consumed=[]
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for parameter in group['params']:
                value = local(parameter.detach())
                if value.dtype != torch.bfloat16 or not value.is_contiguous():
                    raise ValueError('Every persistent policy master must be contiguous BF16')
                if parameter.grad is not None:
                    raise RuntimeError('An untracked parameter.grad escaped FP32 capture')
                slot = self.gradients.slots.get(parameter)
                if slot is None:
                    if parameter.requires_grad and value.numel():
                        raise RuntimeError('A trainable local parameter is missing its FP32 reduced gradient')
                    continue
                if slot.gradient.dtype != torch.float32 or slot.gradient.shape != value.shape:
                    raise ValueError('Every optimizer input must be the actual matching FP32 reduced shard')
                if not value.numel():
                    continue
                if self.gradients._q38_capture_contract is not None:
                    self.gradients._check_ledger_slot(parameter,slot)
                    self.gradients._wait_gpu_slot(parameter,slot)
                state = self.state[parameter]
                if not state:
                    state['step'] = 0
                    state['moments'] = (self.moments.allocate_reusing_gradient(slot.name,slot.gradient)
                        if first_reuse and slot.tier=='cpu' else self.moments.allocate(slot.name, value.shape))
                previous = state['step']
                step = previous + 1
                handle = state['moments']
                first, second = handle['first'].view(-1), handle['second'].view(-1)
                weights = value.view(-1)
                source_grad = slot.gradient.reshape(-1)
                offset = global_parameter_offset(parameter)
                keys = {role: rounding_key(slot.name, step, role, self.rounding_seed)
                        for role in ('weight', 'exp_avg', 'exp_avg_sq')}
                for begin in range(0, value.numel(), self.chunk_numel):
                    end = min(value.numel(), begin + self.chunk_numel)
                    self.last_chunk_peak_numel = max(self.last_chunk_peak_numel, end - begin)
                    active_weight = weights[begin:end].float()
                    active_gradient = source_grad[begin:end].to(value.device, copy=True, non_blocking=False)
                    if group['maximize']:
                        active_gradient.neg_()
                    mean = (torch.zeros_like(active_weight) if previous == 0 else
                            first[begin:end].to(value.device, dtype=torch.float32, copy=True))
                    variance = (torch.zeros_like(active_weight) if previous == 0 else
                                second[begin:end].to(value.device, dtype=torch.float32, copy=True))
                    active_weight.mul_(1 - group['lr'] * group['weight_decay'])
                    mean.lerp_(active_gradient, 1 - beta1)
                    variance.mul_(beta2).addcmul_(active_gradient, active_gradient, value=1-beta2)
                    denominator = variance.sqrt().div_(math.sqrt(1-beta2**step)).add_(group['eps'])
                    active_weight.addcdiv_(mean, denominator, value=-group['lr'] / (1-beta1**step))
                    rounded_weight = stochastic_bf16(active_weight.contiguous(), key=keys['weight'], offset=offset+begin)
                    rounded_first = stochastic_bf16(mean.contiguous(), key=keys['exp_avg'], offset=offset+begin)
                    rounded_second = stochastic_bf16(variance.contiguous(), key=keys['exp_avg_sq'], offset=offset+begin)
                    self.changed_weight_elements += int((weights[begin:end] != rounded_weight).sum().item())
                    weights[begin:end].copy_(rounded_weight)
                    first[begin:end].copy_(rounded_first, non_blocking=False)
                    second[begin:end].copy_(rounded_second, non_blocking=False)
                    del active_weight, active_gradient, mean, variance, denominator
                    del rounded_weight, rounded_first, rounded_second
                state['step'] = step
                if first_reuse and slot.tier=='cpu':
                    storage=slot.gradient.untyped_storage();size=slot.gradient.numel()*4
                    if (first.untyped_storage()._cdata!=storage._cdata or second.untyped_storage()._cdata!=storage._cdata
                            or first.data_ptr()!=storage.data_ptr() or second.data_ptr()!=storage.data_ptr()+2):
                        raise RuntimeError('Reused BF16 states do not bind their original FP32 storage')
                    consumed.append(dict(name=slot.name,gradient_bytes=size,storage_id=int(storage._cdata),
                        storage_pointer=int(storage.data_ptr()),first_dtype='bfloat16',second_dtype='bfloat16',
                        first_stride=list(first.stride()),second_stride=list(second.stride()),
                        all_chunks_read_before_overwrite=True,contributions=slot.contributions))
                    self.gradients.cpu_spilled_bytes-=slot.gradient.numel()*4
                    slot.gradient=None
        if first_reuse:
            expected=[row for row in self._reuse_plan['slots'] if row['tier']=='cpu']
            if ([row['name'] for row in consumed]!=[row['name'] for row in expected]
                    or sum(row['gradient_bytes'] for row in consumed)!=self._reuse_plan['reusable_bytes']):
                raise RuntimeError('Consumed slots do not match frozen first-step plan')
            self._reuse_commit=dict(schema='q38-first-adam-reuse-commit-v1',complete=True,
                pre_update_evidence=self._reuse_evidence,pre_update_generation=self._reuse_plan['generation'],
                consumed_cpu_gradient_bytes=self._reuse_plan['reusable_bytes'],consumed_slots=consumed,
                consumed_gradients_still_resident_as_fp32=False)
            self._reuse_plan=None
        return None

    def clear_first_reuse_before_world_vote(self):
        with self.gradients._lock:
            if (self._reuse_failed or self.gradients._reuse_failed or not self.gradients._reuse_frozen
                    or self._reuse_commit is None or self._reuse_poststep_clear):
                raise RuntimeError('First-reuse internal clear is not authorized')
            super().zero_grad(set_to_none=True)
            self.gradients._clear_contents_locked();self._reuse_poststep_clear=True

    def release_first_reuse_after_world_commit(self):
        with self.gradients._lock:
            if (self._reuse_failed or self.gradients._reuse_failed or not self.gradients._reuse_frozen
                    or not self._reuse_poststep_clear or self.gradients.slots):
                raise RuntimeError('First-reuse freeze cannot be released')
            self.gradients._reuse_frozen=False

    def zero_grad(self, set_to_none=True):
        if not set_to_none:
            raise ValueError('The explicit gradient store must be released with set_to_none=True')
        super().zero_grad(set_to_none=True)
        self.gradients.clear()

    def state_dict(self):
        raise NotImplementedError('Use a bounded per-rank BF16 weight/state checkpoint writer')

    def audit(self):
        return dict(implementation='BF16 persistent AdamW with counter stochastic rounding',
                    master_dtype='bfloat16', gradient_dtype='float32', moment_dtype='bfloat16',
                    steps=sorted({state['step'] for state in self.state.values()}),
                    maximum_FP32_temporary_chunk_elements=self.last_chunk_peak_numel,
                    configured_FP32_chunk_bytes=self.chunk_numel * 4,
                    persistent_FP32_master_bytes=0, persistent_FP32_moment_bytes=0,
                    disk_offload_bytes=0, rounding_seed=self.rounding_seed,
                    rounding='SplitMix64(global runtime-parameter coordinate, named state, step, seed)',
                    changed_weight_elements_last_step=self.changed_weight_elements,
                    gradient_store=self.gradients.audit(), moments=self.moments.audit(),
                    first_adam_reuse_enabled=self.reuse_cpu_spill,first_adam_reuse_commit=self._reuse_commit)
