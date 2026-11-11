"""Zero-copy CUDA views for host-pinned V4.1 KV pools.

``DSV4_FIXED_POOL_USE_MEMORY=1`` moves the fixed-size pools (swa_kv,
decoder_swa_kv, csa_state) into pinned host RAM — the branch design for
HBM-constrained devices keeps bounded-window state off-device and out of the
paged KV budget. The pool buffers are torch ``pin_memory()`` allocations
(``DeviceBlockPool::initializePinnedCpuBuffer``), so under unified addressing
they are fully device-accessible; native SM100 kernels consume the raw
pointers C++-side and never notice. Python-side consumers (Triton fallback
kernels, torch indexing) instead receive CPU tensors and reject them
("Pointer argument cannot be accessed from Triton", "indices should be ...
same device as the indexed tensor (cpu)").

``install_pinned_pool_cuda_views`` patches the process ``KVCache`` binding so
every ``get_layer_cache``/``get_layer_cache_groups`` result presents pinned
pool storage as a zero-copy CUDA tensor view (``cudaHostGetDevicePointer`` +
``__cuda_array_interface__`` — the mechanism the Engram host projections
already proved on this hardware). No GPU-resident copy is created: the view
aliases the same host pages, so the pool stays host-RAM-only per placement
policy, and staging rows for computation never becomes a persistent device
allocation.
"""

import logging
import threading

import torch

logger = logging.getLogger(__name__)

_VIEW_LOCK = threading.Lock()
# data_ptr -> (cuda_view, cpu_tensor). Holding the CPU tensor keeps the CAI
# owner chain honest; the C++ pool owns the underlying storage for the process
# lifetime regardless.
_VIEWS: dict = {}
_PATCHED_CLASSES: set = set()


def _cuda_view_of_pinned(cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    from cuda.bindings import runtime

    from rtp_llm.models_py.modules.dsv41.engram import _CudaArrayView

    with torch.cuda.device(device):
        result, pointer = runtime.cudaHostGetDevicePointer(cpu.data_ptr(), 0)
    if int(result) != 0:
        raise RuntimeError(
            f"pinned-pool cudaHostGetDevicePointer failed for "
            f"{tuple(cpu.shape)}: {result}"
        )
    # Flat bytes survive any element dtype; the views below restore layout.
    flat = torch.as_tensor(
        _CudaArrayView(
            cpu,
            cpu,
            pointer,
            shape=(cpu.numel() * cpu.element_size(),),
            strides=(1,),
        ),
        device=device,
    )
    return flat.view(cpu.dtype).view(cpu.shape)


def _view_for(cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    key = (cpu.data_ptr(), tuple(cpu.shape), cpu.dtype)
    with _VIEW_LOCK:
        hit = _VIEWS.get(key)
    if hit is not None:
        return hit[0]
    if not cpu.is_pinned():
        raise RuntimeError(
            "host-resident KV pool tensor is not pinned; UVA views require "
            "cudaHostAlloc-backed memory (DeviceBlockPool HOST_PINNED backing)"
        )
    view = _cuda_view_of_pinned(cpu, device)
    with _VIEW_LOCK:
        _VIEWS.setdefault(key, (view, cpu))
    logger.info(
        "[pinned-pools] zero-copy CUDA view for host pool tensor shape=%s dtype=%s ptr=%#x",
        tuple(cpu.shape),
        cpu.dtype,
        cpu.data_ptr(),
    )
    return view


def _maybe_swap(lkv, device: torch.device):
    """Point one LayerKVCache copy's tensors at zero-copy CUDA views."""
    base = lkv.kv_cache_base
    if (
        base is not None
        and torch.is_tensor(base)
        and base.numel() > 0
        and base.device.type == "cpu"
    ):
        lkv.kv_cache_base = _view_for(base, device)
    scale = lkv.kv_scale_base
    if (
        scale is not None
        and torch.is_tensor(scale)
        and scale.numel() > 0
        and scale.device.type == "cpu"
    ):
        lkv.kv_scale_base = _view_for(scale, device)
    return lkv


def install_pinned_pool_cuda_views(kv_cache, device=None) -> None:
    """Patch the KVCache binding class so pinned pools surface as CUDA views.

    Class-level and idempotent: every consumer in the process (pool views,
    JIT warmups, CED, indexer) goes through ``get_layer_cache``/
    ``get_layer_cache_groups``, and ``LayerKVCache.kv_cache_base`` is
    writable, so wrapping the returned copies needs no C++ change. GPU-resident
    pools pass through untouched, keeping default deployments bit-identical.
    """
    if kv_cache is None:
        return
    cls = type(kv_cache)
    if id(cls) in _PATCHED_CLASSES:
        return
    if device is None:
        device = torch.cuda.current_device()
    device = torch.device(device)

    orig_get = cls.get_layer_cache
    orig_groups = getattr(cls, "get_layer_cache_groups", None)

    def get_layer_cache(self, layer_id, tag=None):
        lkv = (
            orig_get(self, layer_id)
            if tag is None
            else orig_get(self, layer_id, tag)
        )
        return _maybe_swap(lkv, device)

    cls.get_layer_cache = get_layer_cache
    if orig_groups is not None:

        def get_layer_cache_groups(self, layer_id):
            return [
                _maybe_swap(item, device)
                for item in orig_groups(self, layer_id)
            ]

        cls.get_layer_cache_groups = get_layer_cache_groups

    _PATCHED_CLASSES.add(id(cls))
    logger.info(
        "[pinned-pools] installed zero-copy CUDA pool views on %s (device=%s)",
        cls.__name__,
        device,
    )
