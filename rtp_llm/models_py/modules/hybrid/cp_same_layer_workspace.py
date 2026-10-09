"""CP4 eager-prefill scratch for the same-layer three-all-gather schedule.

This helper owns storage and its reuse fences, not the collective schedule.
The caller must pass globally identical padded row counts on every rank.
"""

import contextlib
import logging
import os
from dataclasses import dataclass, field


@dataclass(eq=False)
class Lease:
    kind: str
    rows: int
    generation: int
    tensors: dict
    consumer_waited: bool = False


@dataclass
class _Slot:
    capacity: int = 0
    generation: int = 0
    pool: object = None
    storage: dict = field(default_factory=dict)
    done: object = None
    active: object = None


class SameLayerCPWorkspace:
    """One reusable prefix slot and one reusable suffix slot per TP_SIDE group.

    main_stream and side_stream are fixed for this owner. Instantiate once at
    model/group scope, never per layer. backend is the TP_SIDE NCCL backend:
    get_group(Group.TP_SIDE)._get_backend(device). effective_cta_policy must be
    the actual communicator configuration, not an assumption from environment.

    This class intentionally has no async destructor: close() must happen
    collectively, before process-group destruction, after every lease retires.
    """

    def __init__(self, device, backend, main_stream, side_stream,
                 effective_cta_policy=0, *, torch_module=None, environ=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.device = device
        self.backend = backend
        self.main_stream = main_stream
        self.side_stream = side_stream
        self.slots = {"prefix": _Slot(), "suffix": _Slot()}
        self.closed = False
        env = os.environ if environ is None else environ
        if env.get("RTP_LLM_CP_PACKED_KV_OVERLAP", "0") != "1":
            raise ValueError("instantiate only for CP4 native same-layer overlap")
        self.cta_policy = int(env.get("NCCL_CTA_POLICY", "0"))
        if int(effective_cta_policy) != self.cta_policy:
            raise ValueError("TP_SIDE effective CTA policy differs from selector")
        self.registered = self.cta_policy == 2
        logging.getLogger(__name__).info(
            "M3.1 CP4 workspace allocator=%s cta_policy=%d symmetric=%s",
            "ncclMemAlloc" if self.registered else "torch",
            self.cta_policy, self.registered,
        )

    def _check_context(self):
        if self.closed:
            raise RuntimeError("workspace is closed")
        with self.torch.cuda.device(self.device):
            if self.torch.cuda.is_current_stream_capturing():
                raise RuntimeError("same-layer communication workspace is eager-only")
        if (self.torch.cuda.current_stream(self.device).cuda_stream
                != self.main_stream.cuda_stream):
            raise RuntimeError("workspace main stream changed")

    def _layout(self, kind):
        # Fixed allocation order, independent of local rank or live row count.
        if kind == "prefix":
            return (("side", 17408, self.torch.uint8),
                    ("values", 65536, self.torch.uint8))
        return (("kv", 1152, self.torch.bfloat16),)

    def _synchronize_registration(self):
        """Finish every slot before communicator-wide registered-pool changes."""
        with self.torch.cuda.device(self.device):
            self.side_stream.synchronize()
            for previous in self.slots.values():
                if previous.done is not None:
                    previous.done.synchronize()

    def _drop_idle(self, slot):
        # done includes the side-stream collective plus all main-stream reads.
        # A CPU wait only occurs on growth/close, never on steady-state reuse.
        if slot.done is not None:
            slot.done.synchronize()
        if slot.pool is not None:
            self.backend.deregister_mem_pool(slot.pool)
        slot.storage.clear()
        slot.pool = None
        slot.capacity = 0
        slot.done = None

    def _acquire(self, kind, rows):
        self._check_context()
        if type(rows) is not int or rows < 0:
            raise ValueError("rows must be a nonnegative globally padded integer")
        slot = self.slots[kind]
        if slot.active is not None:
            raise RuntimeError(f"{kind} slot still has an active consumer")
        if rows == 0:
            # Only legal when ALL CP ranks skip this prefix. No registration,
            # resize, event, or collective is issued for a zero-count stage.
            return None
        if rows > slot.capacity:
            if self.registered:
                # Registration affects the communicator, so another slot's
                # in-flight CE must also finish before changing registered pools.
                # Only cold allocation/growth takes this CPU fence; steady-state
                # reuse retains asynchronous stream/event dependencies.
                self._synchronize_registration()
            self._drop_idle(slot)
            pool = (self.torch.cuda.MemPool(self.backend.mem_allocator)
                    if self.registered else None)
            ctx = (self.torch.cuda.use_mem_pool(pool)
                   if pool is not None else contextlib.nullcontext())
            storage = {}
            with self.torch.cuda.device(self.device), ctx:
                for name, width, dtype in self._layout(kind):
                    # Flat leading views keep rank-packed layout contiguous
                    # when a request uses fewer rows than the high-water mark.
                    for direction, multiplier in (("send", 1), ("recv", 4)):
                        storage[name + "_" + direction] = self.torch.empty(
                            rows * width * multiplier + 1024,
                            dtype=dtype, device=self.device,
                        )
            # All ranks call in identical prefix/suffix schedule and allocation
            # order. Registration failures are fatal; no rank-local fallback.
            if pool is not None:
                self.backend.register_mem_pool(pool, symm=True)
            slot.storage, slot.pool, slot.capacity = storage, pool, rows
        elif slot.done is not None:
            self.side_stream.wait_event(slot.done)

        # Orders newly allocated storage and this stage's main-stream producers
        # before side-stream writes. For prefix call before suffix projections.
        self.side_stream.wait_stream(self.main_stream)
        tensors = {}
        for name, width, _dtype in self._layout(kind):
            tensors[name + "_send"] = slot.storage[name + "_send"][:rows * width].view(rows, width)
            tensors[name + "_recv"] = slot.storage[name + "_recv"][:rows * width * 4].view(rows * 4, width)
        slot.generation += 1
        lease = Lease(kind, rows, slot.generation, tensors)
        slot.active = lease
        return lease

    def acquire_prefix(self, global_padded_rows):
        return self._acquire("prefix", global_padded_rows)

    def acquire_suffix(self, global_padded_rows):
        return self._acquire("suffix", global_padded_rows)

    def _slot_for(self, lease):
        self._check_context()
        slot = self.slots[lease.kind]
        if slot.active is not lease or slot.generation != lease.generation:
            raise RuntimeError("expired or foreign workspace lease")
        return slot

    def wait_for_consumer(self, lease, ready_event):
        """Call on main before restoring/reading gathered results.

        ready_event must be recorded AFTER the final collective on side_stream;
        for prefix it is recorded before waiting for suffix projections.
        """
        self._slot_for(lease)
        self.main_stream.wait_event(ready_event)
        lease.consumer_waited = True

    def retire(self, lease):
        """Call on main AFTER enqueueing every read of this lease's tensors."""
        slot = self._slot_for(lease)
        if not lease.consumer_waited:
            raise RuntimeError("retire requires the collective readiness dependency")
        done = self.torch.cuda.Event()
        done.record(self.main_stream)
        slot.done = done
        slot.active = None
        # Do not keep references to old windows in expired leases. Caller must
        # also release saved views, and must not enqueue any reads after retire.
        lease.tensors.clear()

    def payload_bytes(self):
        """Owned tensor bytes, excluding NCCL/CUDA allocator segment overhead."""
        return sum(t.numel() * t.element_size()
                   for slot in self.slots.values() for t in slot.storage.values())

    def close(self):
        """All CP ranks call in the same order before communicator teardown."""
        self._check_context()
        if any(slot.active is not None for slot in self.slots.values()):
            raise RuntimeError("retire all leases before closing workspace")
        if self.registered:
            self._synchronize_registration()
        for kind in ("prefix", "suffix"):
            self._drop_idle(self.slots[kind])
        self.closed = True


@dataclass
class PrefixGather:
    values: object
    side: object
    ready: object
    working_planes: tuple
    workspace: SameLayerCPWorkspace
    lease: Lease


def _effective_cta_policy(backend, environ=None):
    """Older Torch may omit CTA config; explicit CE must still prove policy 2."""
    env = os.environ if environ is None else environ
    selected = int(env.get("NCCL_CTA_POLICY", "0"))
    config = getattr(getattr(backend, "options", None), "config", None)
    actual = getattr(config, "cta_policy", None)
    if selected == 2:
        if actual != 2:
            raise RuntimeError("TP_SIDE actual CTA policy=2 is required for registered CE")
        return 2
    # NCCL_CONFIG_UNDEF_INT is the ordinary/default communicator policy.
    return selected if actual is None or int(actual) < 0 else int(actual)


def get_workspace(device, main_stream, side_stream):
    from rtp_llm.models_py.distributed import collective_torch
    group = collective_torch._get_group(collective_torch.Group.TP_SIDE)
    key = (group, str(device))
    workspace = collective_torch._owned_cp_workspaces.get(key)
    if workspace is None or workspace.closed:
        backend = group._get_backend(device)
        workspace = SameLayerCPWorkspace(
            device, backend, main_stream, side_stream,
            effective_cta_policy=_effective_cta_policy(backend),
        )
        collective_torch._owned_cp_workspaces[key] = workspace
    if workspace.side_stream.cuda_stream != side_stream.cuda_stream:
        raise RuntimeError("same-layer workspace side stream changed")
    workspace._check_context()
    return workspace
