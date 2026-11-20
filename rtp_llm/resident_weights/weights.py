"""Keep post-processed CPU and GPU weights alive across engine restarts.

Three phases make up a cold start. ``T1`` copies the loader output into sealed CPU memfds or
exportable CUDA VMM memory (``move_to_vmm``), ``T2`` is the C++ ``WeightsConverter``, and ``T3`` is the python model's
``initialize()``, where weight transforms run through :func:`resident_init_tensors`,
:func:`resident_init_scalar` and :func:`should_consume_loaded`. After the first successful
``initialize()`` the storages that are still alive are handed to the per-user store as FDs.

A restart maps the same storages read-only: ``T1`` puts the ``loaded`` tensors back into the
weight dicts and ``T3`` hands each transform its recorded result without running it again.

Which of the two a manager is in is explicit: exactly one of ``_recorder`` (cold start) and
``_replay`` (restart) is set, and both are released by ``finish()``.
"""

import gc
import hashlib
import json
import logging
import os
import resource
import socket
import time
import weakref
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch

from rtp_llm.resident_weights import store
from rtp_llm.resident_weights.cpu import CpuAllocation
from rtp_llm.resident_weights.store import (
    GLOBAL_SCOPE,
    AllocationSpec,
    AttachStatus,
    Manifest,
    MemoryKind,
    ScalarValue,
    ScopeKey,
    SlotRecord,
    StorageSpec,
    TensorSpec,
    WeightOrigin,
)
from rtp_llm.resident_weights.vmm import (
    VmmAllocation,
    _CudaArrayBridge,
    round_up_to_granularity,
)

if TYPE_CHECKING:
    # The loader imports this package, so these stay annotations only.
    from rtp_llm.model_loader.loader import ModelLoader
    from rtp_llm.model_loader.model_weight_info import ModelWeights

# Safety ceiling on client-side polling for an in-flight restore; slightly past the store's own
# 600 s move_timeout_s so the store aborts and reports cleanly first.
_RESTORE_WAIT_S = 660.0


class _RestoreWaitTimeoutError(RuntimeError):
    """Client-side restore wait exceeded the store's move bound; handled like RESTORE_FAILED."""


# Three-state attach consensus across WORLD ranks (combined with ReduceOp.MIN):
# restore failure wins over cold miss, which wins over warm hit.
_ATTACH_RESTORE_FAILED = 0
_ATTACH_MISS = 1
_ATTACH_HIT = 2
# Storages this large get their own allocation, so the memory of a consumed input is released.
_DEDICATED_MIN_BYTES = 1 << 20
_CHUNK_BYTES = 16 << 20
# torch's caching allocator returns 512-byte aligned blocks; packed storages keep that alignment.
_ALIGN = 512

_Allocation = Union[VmmAllocation, CpuAllocation]
_StorageOwner = Union[_CudaArrayBridge, CpuAllocation]

_managers: "weakref.WeakSet[ResidentWeights]" = weakref.WeakSet()


@dataclass(frozen=True)
class _StorageAddressRange:
    """Where a torch storage lives right now. Identifies a resident storage while it is alive."""

    ptr: int
    nbytes: int
    device: torch.device

    @classmethod
    def of(cls, storage: torch.UntypedStorage) -> "_StorageAddressRange":
        return cls(storage.data_ptr(), storage.nbytes(), storage.device)


@dataclass(frozen=True)
class _ViewSignature:
    """Identifies one tensor view well enough to recognise it again under another name.

    A transform is given the name the model is using now; this is how the loader names behind
    that one are found when the model renamed a tensor before consuming it.
    """

    storage_range: _StorageAddressRange
    storage_offset: int
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]

    @classmethod
    def of(cls, tensor: torch.Tensor) -> "_ViewSignature":
        return cls(
            _StorageAddressRange.of(tensor.untyped_storage()),
            tensor.storage_offset(),
            tuple(tensor.size()),
            tuple(tensor.stride()),
        )


@dataclass(frozen=True)
class RestoredTensor:
    """One loader weight a restart mapped back, for ``T1`` to put into the weight dict again."""

    scope: ScopeKey
    name: str
    tensor: torch.Tensor


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _bytes_of(storage: torch.UntypedStorage) -> torch.Tensor:
    return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def _view(
    storage: torch.UntypedStorage, dtype, storage_offset, shape, strides
) -> torch.Tensor:
    return torch.empty(0, dtype=dtype, device=storage.device).set_(
        storage, storage_offset, shape, strides
    )


def _checksum_total(storage: torch.UntypedStorage) -> torch.Tensor:
    """Position-weighted 64-bit sum of the bytes, left on their device as an int64 scalar.

    Verify mode compares it across runs. The caller pulls every storage's total back in one
    transfer: a dsv41 target has several hundred of them and a sync each would be visible.
    """
    data = _bytes_of(storage)
    total = torch.zeros((), dtype=torch.int64, device=data.device)
    step = 1 << 24
    for start in range(0, data.numel(), step):
        chunk = data[start : start + step].to(torch.int64)
        position = torch.arange(
            start + 1, start + 1 + chunk.numel(), dtype=torch.int64, device=data.device
        )
        total += (chunk * (position * 0x9E3779B1 + 1)).sum()  # int64 arithmetic wraps
    return total


def _format_checksum(total: int) -> str:
    return f"{total & ((1 << 64) - 1):016x}"


class WeightDict(dict):
    """One layer's or the global weight dict, carrying its scope and its resident manager.

    It subclasses the builtin dict, so ``pybind11::isinstance<py::dict>`` in the C++ weights
    converter and every existing python model see exactly what they saw before. The scope and the
    weak link to the manager replace a module-global ``id(dict) -> manager`` table, which could
    not tell a dead dict from a new one allocated at the same address.
    """

    def __init__(self, scope: ScopeKey = GLOBAL_SCOPE, mapping=()):
        super().__init__(mapping)
        self.scope: ScopeKey = scope
        self._manager_ref: "Optional[weakref.ref[ResidentWeights]]" = None

    @property
    def manager(self) -> "Optional[ResidentWeights]":
        return self._manager_ref() if self._manager_ref is not None else None

    def bind(self, manager: "ResidentWeights") -> None:
        # A weak reference: a manager that is dropped mid-initialization must not be kept alive
        # by the dicts it was recording, and no cycle is formed for the collector to break.
        self._manager_ref = weakref.ref(manager)

    def unbind(self) -> None:
        self._manager_ref = None


@dataclass
class _TrackedStorage:
    """One resident storage, alive exactly as long as some tensor still views it.

    The weak owner is either a CUDA array bridge holding its VMM allocation, or the
    CPU mmap buffer itself. A live owner proves the storage is still in use.
    """

    offset_bytes: int
    nbytes: int
    bridge_ref: "weakref.ref[_StorageOwner]"

    def is_alive(self) -> bool:
        return self.bridge_ref() is not None

    def allocation(self) -> Optional[_Allocation]:
        bridge = self.bridge_ref()
        return None if bridge is None else bridge.allocation


@dataclass
class _VmmCopyTarget:
    """One tensor to move into resident memory: ``container[key]`` is replaced in place."""

    container: object  # a WeightDict, or the output list of an init-time transform
    key: object  # a weight name, or an index into that list
    scope: ScopeKey
    name: str
    origin: WeightOrigin


class _VmmPacker:
    """Places storages: large ones in their own allocation, small ones packed into shared chunks."""

    def __init__(self, device: int):
        self.device = device
        self.chunk: Optional[VmmAllocation] = None
        self.used = 0

    def place(self, nbytes: int) -> Tuple[VmmAllocation, int, bool]:
        """Returns (allocation, offset_bytes, whether the allocation is new)."""
        if nbytes >= _DEDICATED_MIN_BYTES:
            return (
                _create_allocation(
                    round_up_to_granularity(nbytes, self.device), self.device
                ),
                0,
                True,
            )
        offset = (self.used + _ALIGN - 1) // _ALIGN * _ALIGN
        new = self.chunk is None or offset + nbytes > self.chunk.size_bytes
        if new:
            self.chunk = _create_allocation(
                round_up_to_granularity(_CHUNK_BYTES, self.device), self.device
            )
            offset = 0
        self.used = offset + nbytes
        return self.chunk, offset, new


def _create_allocation(size_bytes: int, device: int) -> VmmAllocation:
    try:
        return VmmAllocation.create(size_bytes, device)
    except RuntimeError:
        # Blocks freed by earlier relocations sit in torch's cache until they are returned.
        torch.cuda.empty_cache()
        return VmmAllocation.create(size_bytes, device)


@dataclass
class _ColdRecorder:
    """Everything a cold start accumulates before it can publish. Absent on a restart."""

    device: int
    packer: _VmmPacker
    storages: List[_TrackedStorage] = field(default_factory=list)
    # (storage data_ptr, storage nbytes) -> index into ``storages``. Each entry is removed by the
    # weakref callback of its own lease, so a reused address can never resolve to a dead storage.
    storage_index_by_address_range: Dict[_StorageAddressRange, int] = field(
        default_factory=dict
    )
    # What T1 moved: the loader's own names and the layout they had right after relocation.
    initial_loaded_specs: Dict[Tuple[ScopeKey, str], TensorSpec] = field(
        default_factory=dict
    )
    # (scope, view signature) -> the loader names that viewed it after T1. A transform that
    # consumes a renamed tensor settles the original names along with it.
    loaded_names_by_view: Dict[Tuple[ScopeKey, _ViewSignature], List[str]] = field(
        default_factory=dict
    )
    # Loader names a transform reported consuming: exempt from the publish-time check that every
    # loaded name is still accounted for. Reporting does not delete anything itself.
    consumed_loaded_keys: Set[Tuple[ScopeKey, str]] = field(default_factory=set)
    init_tensor_specs: List[TensorSpec] = field(default_factory=list)
    transform_returned_tensors: Dict[ScopeKey, Dict[str, bool]] = field(
        default_factory=dict
    )
    init_scalars: Dict[ScopeKey, Dict[str, ScalarValue]] = field(default_factory=dict)
    weight_dicts: List[WeightDict] = field(default_factory=list)
    excluded_names: FrozenSet[str] = frozenset()
    # Resident GPU bytes when the loaded model went to the engine, before initialize().
    gpu_bytes_at_engine_start: Optional[int] = None


@dataclass
class _AttachedReplay:
    """What a restart hands back instead of running the transforms. Absent on a cold start.

    ``finalize()`` requires all three tables to be empty: an unrequested key means this run took a
    different branch than the run that published, and the record no longer describes it.
    """

    transform_returned_tensors: Dict[Tuple[ScopeKey, str], bool]
    init_tensors: Dict[Tuple[ScopeKey, str], torch.Tensor]
    init_scalars: Dict[Tuple[ScopeKey, str], ScalarValue]

    def unrequested_keys(self) -> List[str]:
        return sorted(
            f"{scope}/{key}"
            for table in (
                self.transform_returned_tensors,
                self.init_tensors,
                self.init_scalars,
            )
            for scope, key in table
        )


class ResidentWeights:
    """Resident weights of one model role on one GPU, for compatible engine restarts.

    The caller owns checkpoint/code/dependency compatibility and must drop old records after
    incompatible changes. The fingerprint detects registered configuration drift, not arbitrary
    source edits or in-place checkpoint replacement.
    """

    def __init__(
        self,
        role: str,
        device: int,
        fingerprint: Dict[str, str],
        verify: bool,
        store_settings: Optional[store.StoreSettings] = None,
    ):
        self.device = device
        self._store_settings = store_settings or store.StoreSettings()
        # Open while this process uses the record; closed on drop/release or process exit.
        self._hold: Optional[socket.socket] = None
        # Generation token observed on attach; engine-side cleanup drops only this generation.
        self._seen_generation: Optional[str] = None
        self._attach_attempted = False
        # Set when attach() gets RESTORE_FAILED; raised after multi-rank consensus.
        self._restore_error: Optional[str] = None
        self.role = role
        self._gpu_uuid = str(torch.cuda.get_device_properties(device).uuid)
        self.slot_key = f"{self._gpu_uuid}/{role}"
        self.fingerprint_text = "\n".join(
            f"{k}: {v}" for k, v in sorted(fingerprint.items())
        )
        self.fingerprint = hashlib.sha256(self.fingerprint_text.encode()).hexdigest()
        self.summary = f"{fingerprint.get('model_type', '?')} {fingerprint.get('checkpoint_path', '?')}"
        self.verify = verify
        self.eplb_placement: Optional[Dict[int, List[int]]] = None
        self._store_name = store.default_store_name()
        # Exactly one of these is set while the model is being built, and neither afterwards.
        self._recorder: Optional[_ColdRecorder] = None
        self._replay: Optional[_AttachedReplay] = None
        # Every allocation this process owns, published or attached. Weak: the leases on the
        # live storages are what keep them mapped, and EPLB only needs the ones still in use.
        self._allocation_refs: "List[weakref.ref[_Allocation]]" = []
        self._attached = False
        self._model_weights: Optional["weakref.ref"] = None
        self._published = False
        self._finalized = False
        self._verify_against: Optional[SlotRecord] = None
        # The attached record's initialize_gpu_growth_bytes; 0 on a cold start.
        self.initialize_gpu_growth_bytes = 0
        _managers.add(self)

    # ---- store access -------------------------------------------------------------------

    def ensure_store_running(self) -> bool:
        # Publish and attach hold every allocation FD of this model at once (550 for the dsv41
        # target) on top of the engine's own files; lift the soft limit as the store does.
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            running = store.ensure_running(
                self._store_name, settings=self._store_settings
            )
            if running != self._store_settings:
                logging.warning(
                    "resident weight store %s keeps its current settings (%s), "
                    "not %s; use `store.py config` to update them live or stop it",
                    self._store_name,
                    running,
                    self._store_settings,
                )
            return True
        except (OSError, store.StoreError) as error:
            logging.warning(
                "resident weights disabled: store %s unavailable: %s",
                self._store_name,
                error,
            )
            return False

    def drop(self, expected_generation: Optional[str] = None) -> None:
        """Release this slot's record, so the next start loads from the checkpoint."""
        self._release_hold()
        store.drop_slot(
            self._store_name, self.slot_key, expected_generation=expected_generation
        )

    def _release_hold(self) -> None:
        hold = getattr(self, "_hold", None)
        if hold is not None:
            hold.close()
            self._hold = None

    def __del__(self) -> None:
        self._release_hold()

    def _drop_quietly(self, expected_generation: Optional[str] = None) -> None:
        """Drop on an error path, where the error being handled is what matters."""
        try:
            self.drop(expected_generation=expected_generation)
        except (OSError, store.StoreError) as error:
            logging.warning(
                "resident weights %s: could not drop the record: %s",
                self.slot_key,
                error,
            )

    # ---- loading ------------------------------------------------------------------------

    def attach_or_load_weights(
        self,
        loader: "ModelLoader",
        device: str,
        global_weight_aliases: Mapping[str, torch.Tensor],
        excluded_names: Sequence[str] = (),
        synchronize_ranks: bool = False,
    ) -> "ModelWeights":
        """This model's weights: the published record when there is one, else the checkpoint.

        ``global_weight_aliases`` are global weights another model owns and ``excluded_names``
        are weights the loader rebuilds on every start; neither belongs in this record, so a
        cold start leaves both where they are. Verify mode always loads from the checkpoint,
        because what it verifies is the published record against a fresh one.
        """
        self._restore_error = None
        if self.verify:
            self.begin_verify()
            restored = None
        else:
            restored = self.attach()
        if self._restore_error is not None:
            local_state = _ATTACH_RESTORE_FAILED
        elif restored is not None:
            local_state = _ATTACH_HIT
        else:
            local_state = _ATTACH_MISS
        # fastsafetensors broadcasts on WORLD. All participating ranks must agree on whether to
        # attach, cold-load, or fail on a host restore error before anyone enters load_weights.
        state = (
            all_ranks_min_state(local_state, device)
            if synchronize_ranks
            else local_state
        )
        if state == _ATTACH_RESTORE_FAILED:
            self.release_attach()
            raise RuntimeError(
                self._restore_error
                or f"resident weights {self.slot_key}: another rank failed to restore its "
                f"resident weights from host memory; free GPU memory on that rank and start "
                f"again, or drop the record to load from the checkpoint"
            )
        if synchronize_ranks and state == _ATTACH_MISS:
            restored = None
            self.discard_attach()
        if restored is not None:
            weights = loader.assemble_weights(device, global_weight_aliases, restored)
            self.bind(weights)
            return weights
        weights = loader.load_weights(
            device=device, global_weight_aliases=global_weight_aliases
        )
        self.move_to_vmm(weights, tuple(global_weight_aliases) + tuple(excluded_names))
        return weights

    def release_attach(self) -> None:
        """Drop this process's local mappings and hold without deleting the store's record."""
        self._replay = None
        self._attached = False
        self._allocation_refs.clear()
        self.initialize_gpu_growth_bytes = 0
        self._release_hold()

    def discard_attach(self) -> None:
        """Drop a local hit before a coordinated cold load releases its GPU memory."""
        gen = getattr(self, "_seen_generation", None)
        attempted = getattr(self, "_attach_attempted", False)
        self._seen_generation = None
        self.release_attach()
        if attempted and gen is None:
            # ABSENT or already dropped: never delete a record another client published meanwhile.
            return
        self._drop_quietly(expected_generation=gen)

    def _attach_slot(self) -> store.AttachResult:
        """attach_slot, waiting while the store moves an idle record back to the GPU."""
        try:
            result = store.attach_slot(
                self._store_name, self.slot_key, self.fingerprint, hold=True
            )
        except TimeoutError as error:
            # The store may have already received ATTACH and started to_gpu; do not cold-load.
            raise _RestoreWaitTimeoutError(
                f"timed out waiting for store attach reply: {error}"
            ) from error
        except ConnectionError as error:
            if not isinstance(error, ConnectionRefusedError):
                # Only fall back to cold loading when the store is definitely gone
                # (ConnectionRefusedError); any other probe outcome leaves restore state unknown.
                try:
                    store.list_store(self._store_name)
                except (ConnectionRefusedError, FileNotFoundError):
                    raise error from None
                except (OSError, store.StoreError) as probe_error:
                    raise _RestoreWaitTimeoutError(
                        f"lost connection to store during attach ({error}); "
                        f"store probe failed ({probe_error})"
                    ) from error
                raise _RestoreWaitTimeoutError(
                    f"lost connection to store during attach: {error}"
                ) from error
            raise
        if result.status is not AttachStatus.MOVING:
            return result
        logging.info(
            "resident weights %s: moving the record from host memory back to the GPU",
            self.slot_key,
        )
        start = time.monotonic()
        wait_s = _RESTORE_WAIT_S
        while result.status is AttachStatus.MOVING:
            if result.settings is not None:
                wait_s = max(_RESTORE_WAIT_S, result.settings.move_timeout_s + 60.0)
            if time.monotonic() - start > wait_s:
                # Raise a restore-wait error rather than StoreError so attach() reports
                # RESTORE_FAILED through multi-rank consensus instead of dropping & cold-loading.
                raise _RestoreWaitTimeoutError(
                    f"the record was still moving back to the GPU after {wait_s:.0f} s"
                )
            time.sleep(1.0)
            try:
                result = store.attach_slot(
                    self._store_name, self.slot_key, self.fingerprint, hold=True
                )
            except (OSError, store.StoreError) as error:
                # Once MOVING was observed, a poll timeout/disconnect must not fall back to
                # cold loading while a to_gpu mover may still own GPU memory.
                raise _RestoreWaitTimeoutError(
                    f"lost contact with the store while moving back to the GPU: {error}"
                ) from error
        return result

    # ---- restart ------------------------------------------------------------------------

    def attach(self) -> Optional[List[RestoredTensor]]:
        """Map the published weights read-only.

        Returns the loader's own tensors for ``T1`` to put back into the weight dicts, or None
        when this process must load from the checkpoint instead (or when ``_restore_error`` was
        recorded for the caller to raise after multi-rank consensus).
        """
        self._restore_error = None
        self._seen_generation = None
        self._attach_attempted = True
        try:
            result = self._attach_slot()
        except _RestoreWaitTimeoutError as error:
            self._restore_error = (
                f"resident weights {self.slot_key}: {error}. The record stays in the store: "
                f"wait for the move to finish or check the store log, then retry or drop it"
            )
            return None
        except (OSError, store.StoreError) as error:
            logging.warning(
                "resident weights %s: attach failed, loading from the checkpoint: %s",
                self.slot_key,
                error,
            )
            return None
        gen = getattr(result, "generation", None) or None
        if result.status is AttachStatus.MISMATCH:
            logging.warning(
                "resident weights on %s were published for another configuration; dropping them "
                "and loading from the checkpoint. Changed fields: %s",
                self.slot_key,
                _changed_fields(
                    result.published_fingerprint_text, self.fingerprint_text
                ),
            )
            self._drop_quietly(expected_generation=gen)
            return None
        if result.status is AttachStatus.RESTORE_FAILED:
            # Record for attach_or_load_weights() to raise after multi-rank consensus: a
            # checkpoint load needs the same GPU memory, so it would fail too, and throwing
            # before all_ranks_min_state() would hang peer ranks in the collective.
            self._restore_error = (
                f"resident weights {self.slot_key}: {result.restore_error}. The record stays in "
                f"host memory: free GPU memory and start again, or drop it to load from the "
                f"checkpoint"
            )
            return None
        if result.status is AttachStatus.EPLB_DIRTY:
            logging.warning(
                "resident weights on %s were left mid EPLB rearrangement; dropping them",
                self.slot_key,
            )
            self._drop_quietly(expected_generation=gen)
            return None
        if result.status is not AttachStatus.HIT:
            logging.info(
                "resident weights %s: nothing published, loading from the checkpoint",
                self.slot_key,
            )
            return None
        self._seen_generation = gen
        record = result.record
        assert record is not None  # a hit always carries one
        try:
            loaded = self._import(record, result.allocation_fds)
        except BaseException:
            if result.hold_sock is not None:
                result.hold_sock.close()
            raise
        finally:
            # from_fd imports the handle; the engine's own FDs are never needed afterwards, and a
            # failure half-way through must not leave the rest of them pinned here.
            _close_fds(result.allocation_fds)
        self.eplb_placement = {
            int(k): v for k, v in record.eplb_placement.items()
        } or None
        self.initialize_gpu_growth_bytes = record.initialize_gpu_growth_bytes
        self._release_hold()
        self._hold = result.hold_sock
        logging.info(
            "attached resident weights %s: %.2f GiB in %d allocations; the cold start's "
            "initialize() changed resident GPU memory by %+.2f GiB",
            self.slot_key,
            record.total_bytes / (1 << 30),
            len(record.manifest.allocations),
            record.initialize_gpu_growth_bytes / (1 << 30),
        )
        return loaded

    def _import(self, record: SlotRecord, fds: Sequence[int]) -> List[RestoredTensor]:
        manifest = record.manifest
        if len(fds) != len(manifest.allocations):
            self._drop_quietly(
                expected_generation=getattr(self, "_seen_generation", None)
            )
            raise RuntimeError(
                f"resident weights {self.slot_key}: the store sent {len(fds)} descriptors for "
                f"{len(manifest.allocations)} allocations; the record has been dropped, "
                f"restart to load from the checkpoint"
            )
        allocations = [
            (
                CpuAllocation.from_fd(
                    fd, spec.size_bytes, self.device, spec.cuda_access_required
                )
                if spec.kind is MemoryKind.CPU
                else VmmAllocation.from_fd(
                    fd, spec.size_bytes, self.device, writable=False
                )
            )
            for fd, spec in zip(fds, manifest.allocations)
        ]
        self._allocation_refs = [weakref.ref(a) for a in allocations]
        storages = [
            allocations[s.allocation_index].as_torch_storage(s.offset_bytes, s.nbytes)[
                0
            ]
            for s in manifest.storages
        ]
        loaded: List[RestoredTensor] = []
        init_tensors: Dict[Tuple[ScopeKey, str], torch.Tensor] = {}
        for spec in manifest.tensors:
            tensor = self._restore_tensor(storages, spec)
            if spec.origin is WeightOrigin.LOADED:
                loaded.append(RestoredTensor(spec.scope, spec.name, tensor))
            else:
                init_tensors[(spec.scope, spec.name)] = tensor
        self._replay = _AttachedReplay(
            transform_returned_tensors={
                (scope, key): value
                for scope, groups in manifest.transform_returned_tensors.items()
                for key, value in groups.items()
            },
            init_tensors=init_tensors,
            init_scalars={
                (scope, key): value
                for scope, scalars in manifest.init_scalars.items()
                for key, value in scalars.items()
            },
        )
        self._attached = True
        return loaded

    def _restore_tensor(self, storages, spec: TensorSpec) -> torch.Tensor:
        with torch.inference_mode(spec.is_inference):
            if spec.storage_index < 0:
                return torch.empty_strided(
                    spec.shape,
                    spec.strides,
                    dtype=_dtype(spec.dtype),
                    device=(
                        "cpu"
                        if spec.device is MemoryKind.CPU
                        else f"cuda:{self.device}"
                    ),
                )
            return _view(
                storages[spec.storage_index],
                _dtype(spec.dtype),
                spec.storage_offset,
                spec.shape,
                spec.strides,
            )

    # ---- binding the weight dicts -------------------------------------------------------

    def bind(self, model_weights) -> None:
        """Make the weight dicts of ``model_weights`` reachable from the three entry functions.

        Dicts that are not already a ``WeightDict`` are wrapped in place: ``_load_from_ft_style``
        and the unit tests build plain dicts, and the local list it built is gone by the time this
        runs, so there is no second reference left pointing at the unwrapped dict.
        """
        bound: List[WeightDict] = []
        for scope in list(range(len(model_weights.weights))) + [GLOBAL_SCOPE]:
            current = (
                model_weights.global_weights
                if scope == GLOBAL_SCOPE
                else model_weights.weights[scope]
            )
            if not isinstance(current, WeightDict):
                current = WeightDict(scope, current)
                if scope == GLOBAL_SCOPE:
                    model_weights.global_weights = current
                else:
                    model_weights.weights[scope] = current
            current.scope = scope
            current.bind(self)
            bound.append(current)
        self._model_weights = weakref.ref(model_weights)
        if self._recorder is not None:
            self._recorder.weight_dicts = bound

    def _unbind(self) -> None:
        model_weights = (
            self._model_weights() if self._model_weights is not None else None
        )
        if model_weights is None:
            return
        for weight_dict in list(model_weights.weights) + [model_weights.global_weights]:
            if isinstance(weight_dict, WeightDict):
                weight_dict.unbind()

    # ---- cold start ---------------------------------------------------------------------

    def move_to_vmm(self, model_weights, excluded_names: Sequence[str] = ()) -> None:
        """Move every dense CPU/CUDA tensor of ``model_weights`` into exportable memory in place.

        ``excluded_names`` names tensors that stay private: aliases owned by another model
        and tensors that are rewritten in place at runtime.
        """
        excluded = frozenset(excluded_names)
        named = [
            (scope, name, weight_dict)
            for scope, weight_dict in _scoped_dicts(model_weights)
            for name in weight_dict
            if name not in excluded
        ]
        self._recorder = _ColdRecorder(
            device=self.device,
            packer=_VmmPacker(self.device),
            excluded_names=excluded,
        )
        targets = [
            _VmmCopyTarget(weight_dict, name, scope, name, WeightOrigin.LOADED)
            for scope, name, weight_dict in named
        ]
        for spec in self._copy_tensors_to_vmm(targets):
            self._recorder.initial_loaded_specs[(spec.scope, spec.name)] = spec
        for target in targets:
            tensor = target.container[target.key]
            if tensor.untyped_storage().nbytes() == 0:
                continue  # a zero-byte tensor is recreated, not viewed; it has no signature
            self._recorder.loaded_names_by_view.setdefault(
                (target.scope, _ViewSignature.of(tensor)), []
            ).append(target.name)
        self.bind(model_weights)

    def _copy_tensors_to_vmm(
        self, targets: Sequence[_VmmCopyTarget]
    ) -> List[TensorSpec]:
        """Replace each ``target.container[target.key]`` with a view of resident memory.

        Storages are copied one by one, keeping offsets, strides and storage sharing; an original
        is freed as soon as its last view has moved, so the peak is about one storage above the
        weights themselves.
        """
        recorder = self._recorder
        assert recorder is not None
        # Whatever produced these tensors may still be reading resident memory: an
        # initialization-time transform reads its inputs out of the read-only mapping this call
        # is about to re-protect, and re-protecting pages a running kernel reads faults the
        # context. One sync per call covers every set_writable below.
        torch.cuda.synchronize(self.device)
        writable_allocations: List[VmmAllocation] = []
        torch_storages_by_index: Dict[int, torch.UntypedStorage] = {}
        # (data_ptr, nbytes) of a *source* storage -> the resident storage it was copied into. A
        # source stays alive while views of it remain to move, so its address cannot be reused.
        index_by_source_span: Dict[_StorageAddressRange, int] = {}
        specs: List[TensorSpec] = []
        for target in targets:
            tensor = target.container[target.key]
            old = tensor.untyped_storage()
            source_span = _StorageAddressRange.of(old)
            index = index_by_source_span.get(
                source_span, self._resident_storage_index(old)
            )
            if old.nbytes() == 0:
                index = -1
            elif index is not None:
                torch_storages_by_index.setdefault(index, old)
            else:
                index = self._copy_into_resident(
                    old, writable_allocations, torch_storages_by_index
                )
                index_by_source_span[source_span] = index
            specs.append(
                TensorSpec(
                    scope=target.scope,
                    name=target.name,
                    origin=target.origin,
                    storage_index=index,
                    dtype=_dtype_name(tensor.dtype),
                    shape=list(tensor.size()),
                    strides=list(tensor.stride()),
                    storage_offset=tensor.storage_offset(),
                    is_inference=tensor.is_inference(),
                    device=MemoryKind(tensor.device.type),
                )
            )
            if index >= 0:
                with torch.inference_mode(tensor.is_inference()):
                    target.container[target.key] = _view(
                        torch_storages_by_index[index],
                        tensor.dtype,
                        tensor.storage_offset(),
                        tensor.size(),
                        tensor.stride(),
                    )
            del tensor, old
        # The copies must land before the mappings become read-only.
        torch.cuda.synchronize(self.device)
        for allocation in writable_allocations:
            allocation.set_writable(False)
        return specs

    def _copy_into_resident(
        self,
        old: torch.UntypedStorage,
        writable_allocations: List[VmmAllocation],
        torch_storages_by_index: Dict[int, torch.UntypedStorage],
    ) -> int:
        recorder = self._recorder
        assert recorder is not None
        if old.device.type == "cpu":
            allocation = CpuAllocation.create(old, self.device)
            offset = 0
            self._allocation_refs.append(weakref.ref(allocation))
            storage, bridge = allocation.as_torch_storage(0, old.nbytes())
        else:
            allocation, offset, is_new = recorder.packer.place(old.nbytes())
            if is_new:
                self._allocation_refs.append(weakref.ref(allocation))
            if allocation not in writable_allocations:
                allocation.set_writable(True)
                writable_allocations.append(allocation)
            storage, bridge = allocation.as_torch_storage(offset, old.nbytes())
            _bytes_of(storage).copy_(_bytes_of(old))
        index = len(recorder.storages)
        address_range = _StorageAddressRange.of(storage)
        index_by_range = recorder.storage_index_by_address_range

        def forget(_ref, ranges=index_by_range, key=address_range, index=index):
            # Only this storage's own entry: the address may already belong to a newer one.
            if ranges.get(key) == index:
                del ranges[key]

        recorder.storages.append(
            _TrackedStorage(offset, old.nbytes(), weakref.ref(bridge, forget))
        )
        index_by_range[address_range] = index
        torch_storages_by_index[index] = storage
        return index

    def _resident_storage_index(self, storage: torch.UntypedStorage) -> Optional[int]:
        recorder = self._recorder
        if recorder is None:
            return None
        return recorder.storage_index_by_address_range.get(
            _StorageAddressRange.of(storage)
        )

    # ---- initialization-time transforms -------------------------------------------------

    def init_tensors(
        self,
        weight_dict: Mapping[str, torch.Tensor],
        scope: ScopeKey,
        transform_key: str,
        output_names: Sequence[str],
        consumed_names: Sequence[str],
        builder: Callable[[], Optional[Sequence[torch.Tensor]]],
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        if self._replay is not None:
            return self._replay_tensors(scope, transform_key, output_names)
        if self._recorder is None:
            out = builder()
            return None if out is None else tuple(out)
        return self._record_tensors(
            weight_dict, scope, transform_key, output_names, consumed_names, builder
        )

    def _replay_tensors(
        self, scope: ScopeKey, transform_key: str, output_names: Sequence[str]
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """Hand back what the publishing run produced. ``consumed_names`` is deliberately ignored: those
        names were consumed before the record was published and are not in the dict any more.
        """
        replay = self._replay
        assert replay is not None
        if (scope, transform_key) not in replay.transform_returned_tensors:
            self._fail_replay(scope, transform_key, "transform group")
        if not replay.transform_returned_tensors.pop((scope, transform_key)):
            return None  # the publishing run took the degenerate branch and produced nothing
        out = []
        for name in output_names:
            tensor = replay.init_tensors.pop((scope, name), None)
            if tensor is None:
                self._fail_replay(scope, name, "init tensor")
            out.append(tensor)
        return tuple(out)

    def _record_tensors(
        self,
        weight_dict: Mapping[str, torch.Tensor],
        scope: ScopeKey,
        transform_key: str,
        output_names: Sequence[str],
        consumed_names: Sequence[str],
        builder: Callable[[], Optional[Sequence[torch.Tensor]]],
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        recorder = self._recorder
        assert recorder is not None
        # Before the transform runs: the model may have renamed a loaded tensor before consuming
        # it (a checkpoint name to the canonical fused-MoE name), and only the view identifies
        # which loaded names the transform is about to settle.
        signatures = [
            _ViewSignature.of(weight_dict[name])
            for name in consumed_names
            if isinstance(weight_dict.get(name), torch.Tensor)
        ]
        out = builder()
        if out is None:
            # A degenerate branch consumed nothing: the inputs stay in the dict for the fallback
            # path, and the publish-time check still holds them to account.
            recorder.transform_returned_tensors.setdefault(scope, {})[
                transform_key
            ] = False
            return None
        items = list(out)
        if len(items) != len(output_names) or not all(
            isinstance(item, torch.Tensor) for item in items
        ):
            raise RuntimeError(
                f"resident weights {self.slot_key}: transform {transform_key!r} for scope "
                f"{scope} returned {len(items)} values for {len(output_names)} output names; it "
                f"must return None or exactly one tensor per output name"
            )
        del out  # let each output be freed once it has moved
        self._report_consumed(scope, consumed_names, signatures)
        recorder.init_tensor_specs.extend(
            self._copy_tensors_to_vmm(
                [
                    _VmmCopyTarget(items, i, scope, name, WeightOrigin.INIT)
                    for i, name in enumerate(output_names)
                ]
            )
        )
        recorder.transform_returned_tensors.setdefault(scope, {})[transform_key] = True
        return tuple(items)

    def init_scalar(
        self, scope: ScopeKey, scalar_key: str, builder: Callable[[], ScalarValue]
    ) -> ScalarValue:
        if self._replay is not None:
            if (scope, scalar_key) not in self._replay.init_scalars:
                self._fail_replay(scope, scalar_key, "init scalar")
            return self._replay.init_scalars.pop((scope, scalar_key))
        value = builder()
        if self._recorder is None:
            return value
        if not isinstance(value, (bool, int, float, str, type(None))):
            raise RuntimeError(
                f"resident weights {self.slot_key}: init scalar {scalar_key!r} for scope "
                f"{scope} is a "
                f"{type(value).__name__}; only JSON scalars can be published"
            )
        self._recorder.init_scalars.setdefault(scope, {})[scalar_key] = value
        return value

    def should_consume(
        self,
        weight_dict: Mapping[str, torch.Tensor],
        scope: ScopeKey,
        consumed_names: Sequence[str],
    ) -> bool:
        if self._replay is not None:
            return (
                False  # the published weights already went through this preprocessing
            )
        if self._recorder is None:
            return True
        self._report_consumed(
            scope,
            consumed_names,
            [
                _ViewSignature.of(weight_dict[name])
                for name in consumed_names
                if isinstance(weight_dict.get(name), torch.Tensor)
            ],
        )
        return True

    def _report_consumed(
        self,
        scope: ScopeKey,
        names: Sequence[str],
        signatures: Sequence[_ViewSignature],
    ) -> None:
        """Exempt these loader names from the publish-time "still accounted for" check.

        The names the transform was given, plus every loader name that viewed the same memory:
        a renamed tensor settles its original name with it.
        """
        recorder = self._recorder
        assert recorder is not None
        recorder.consumed_loaded_keys.update((scope, name) for name in names)
        for signature in signatures:
            for original in recorder.loaded_names_by_view.get((scope, signature), ()):
                recorder.consumed_loaded_keys.add((scope, original))

    def _fail_replay(self, scope: ScopeKey, key: str, what: str) -> None:
        """A replay asked for something the published record does not have: give up on it."""
        self._drop_quietly(expected_generation=getattr(self, "_seen_generation", None))
        raise RuntimeError(
            f"resident weights {self.slot_key}: the published record has no {what} {key!r} for "
            f"scope {scope}. This initialization took a different branch than the run that "
            f"published; add the environment variable controlling that branch to "
            f"ModelDeployWeightInfo.resident_weights_env_vars or to the config fingerprint. The "
            f"record has been dropped; the next start loads from the checkpoint."
        )

    # ---- publish ------------------------------------------------------------------------

    def finalize_after_initialize(self, py_model) -> None:
        """Publish now, or after ``initialize()`` for a model that builds its layers there.

        ``py_model`` is None when this engine has no python model at all, and for the roles
        that skip building one; there is nothing left to wait for, so publish immediately.
        """
        if self._recorder is not None:
            self._recorder.gpu_bytes_at_engine_start = self._live_gpu_bytes()
        if py_model is None:
            self.finalize()
        else:
            self._wrap_model_initialize(py_model)

    def _live_gpu_bytes(self) -> int:
        """Size of the resident GPU allocations some tensor of this cold start still views."""
        recorder = self._recorder
        assert recorder is not None
        gc.collect()
        allocations: List[_Allocation] = []
        for tracked in recorder.storages:
            allocation = tracked.allocation()
            if isinstance(allocation, VmmAllocation) and allocation not in allocations:
                allocations.append(allocation)
        return sum(a.size_bytes for a in allocations)

    def _wrap_model_initialize(self, py_model) -> None:
        """Publish after the first successful ``initialize()``, when the model holds its final
        weights; an initialization that fails leaves the dicts unbound instead."""
        original = py_model.initialize

        def initialize(*args, **kwargs):
            try:
                result = original(*args, **kwargs)
            except BaseException:
                self._abandon()
                raise
            if not result:
                self._abandon()  # the model reported failure; its weights are not final
            elif not self._finalized:
                self.finalize()
            return result

        py_model.initialize = initialize

    def _abandon(self) -> None:
        """Give up on this start without publishing: nothing recorded describes final weights."""
        self._finalized = True
        self._unbind()
        self._recorder = None
        self._replay = None

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._unbind()
        try:
            if self._replay is not None:
                unrequested = self._replay.unrequested_keys()
                if unrequested:
                    self._fail_unrequested_keys(unrequested)
            elif self._recorder is not None:
                self._publish()
        finally:
            self._replay = None
            self._recorder = None

    def _fail_unrequested_keys(self, unrequested: List[str]) -> None:
        self._drop_quietly(expected_generation=getattr(self, "_seen_generation", None))
        raise RuntimeError(
            f"resident weights {self.slot_key}: {len(unrequested)} published results were "
            f"never asked for: {', '.join(unrequested[:20])}. This initialization took a different branch "
            f"than the run that published; add the environment variable controlling that branch "
            f"to ModelDeployWeightInfo.resident_weights_env_vars or to the config fingerprint. The "
            f"record has been dropped; the next start loads from the checkpoint."
        )

    def _publish(self) -> None:
        recorder = self._recorder
        assert recorder is not None
        gc.collect()
        alive = [tracked.is_alive() for tracked in recorder.storages]
        resolved, problems = self._resolve_loaded_specs(alive)
        problems += sorted(
            f"{spec.scope}/{spec.name}: transform output dropped after initialization"
            for spec in recorder.init_tensor_specs
            if spec.storage_index >= 0 and not alive[spec.storage_index]
        )
        if problems:
            logging.error(
                "resident weights %s not published; a restart could not rebuild these weights: %s",
                self.slot_key,
                "; ".join(problems[:20]),
            )
            return
        fds: List[int] = []
        try:
            try:
                manifest, allocations = self._build_manifest(alive, resolved)
                for allocation in allocations:
                    fds.append(allocation.export_fd())
            except Exception:
                # Publishing only speeds up the next start: running short of memory for the
                # checksums or of descriptors must not fail this one.
                logging.exception("resident weights %s not published", self.slot_key)
                return
            total_bytes = sum(a.size_bytes for a in allocations)
            gpu_bytes = sum(
                a.size_bytes for a in manifest.allocations if a.kind is MemoryKind.CUDA
            )
            engine_start = recorder.gpu_bytes_at_engine_start
            record = SlotRecord(
                fingerprint=self.fingerprint,
                fingerprint_text=self.fingerprint_text,
                summary=self.summary,
                total_bytes=total_bytes,
                publisher_pid=os.getpid(),
                manifest=manifest,
                initialize_gpu_growth_bytes=(
                    0 if engine_start is None else gpu_bytes - engine_start
                ),
            )
            if self.verify:
                self._compare_with_store(manifest)
            try:
                hold_sock = store.publish_slot(
                    self._store_name, self.slot_key, record, fds, hold=True
                )
            except (OSError, store.StoreError) as error:
                logging.warning(
                    "resident weights %s not published: %s", self.slot_key, error
                )
                return
        finally:
            _close_fds(fds)
        self._published = True
        self._release_hold()
        self._hold = hold_sock
        logging.info(
            "published resident weights %s: %.2f GiB in %d allocations",
            self.slot_key,
            total_bytes / (1 << 30),
            len(fds),
        )

    def _resolve_loaded_specs(
        self, alive: List[bool]
    ) -> Tuple[List[TensorSpec], List[str]]:
        """The loaded weights a restart must put back, as they look at the end of initialization.

        ``initial_loaded_specs`` is only the snapshot ``T1`` took. Publishing it unchanged would lose
        every weight a transform re-pointed: ``merge_v41_qkv_weights`` leaves ``weights[q_weight]``
        viewing a slice of its merged output, and the storage the loader name used is dead by now.
        """
        recorder = self._recorder
        assert recorder is not None
        problems = self._check_weight_dicts_unchanged()
        # 1. Names whose original storage is still alive keep their T1 layout. This covers the
        #    untouched weights, the ones popped into a module attribute, and plain renames.
        resolved = [
            spec
            for spec in recorder.initial_loaded_specs.values()
            if spec.storage_index < 0 or alive[spec.storage_index]
        ]
        # 2. A loader name still in the dict whose original storage is dead now views something
        #    else; publish it with the view it has, so a restart reads the same shape back.
        for weight_dict in recorder.weight_dicts:
            for name, tensor in weight_dict.items():
                spec = recorder.initial_loaded_specs.get((weight_dict.scope, name))
                if (
                    name in recorder.excluded_names
                    or spec is None
                    or spec.storage_index < 0
                    or alive[spec.storage_index]
                ):
                    continue
                storage = tensor.untyped_storage()
                index = recorder.storage_index_by_address_range.get(
                    _StorageAddressRange.of(storage)
                )
                if index is None:
                    problems.append(
                        f"{weight_dict.scope}/{name}: replaced by a tensor outside resident memory"
                    )
                    continue
                resolved.append(
                    TensorSpec(
                        scope=weight_dict.scope,
                        name=name,
                        origin=WeightOrigin.LOADED,
                        storage_index=index,
                        dtype=_dtype_name(tensor.dtype),
                        shape=list(tensor.size()),
                        strides=list(tensor.stride()),
                        storage_offset=tensor.storage_offset(),
                        is_inference=tensor.is_inference(),
                        device=MemoryKind(tensor.device.type),
                    )
                )
        # 3. Anything else is gone without a transform reporting that it consumed it: a restart
        #    would silently miss it, so nothing is published at all.
        published = {(spec.scope, spec.name) for spec in resolved}
        problems += sorted(
            f"{scope}/{name}: consumed outside resident_init_tensors/should_consume_loaded"
            for scope, name in recorder.initial_loaded_specs
            if (scope, name) not in published
            and (scope, name) not in recorder.consumed_loaded_keys
        )
        return resolved, problems

    def _check_weight_dicts_unchanged(self) -> List[str]:
        """No path in the engine replaces a weight dict after ``bind()``; make sure of it."""
        recorder = self._recorder
        assert recorder is not None
        model_weights = (
            self._model_weights() if self._model_weights is not None else None
        )
        if model_weights is None:
            return []
        live = list(model_weights.weights) + [model_weights.global_weights]
        if len(live) != len(recorder.weight_dicts):
            return [
                f"ModelWeights held {len(recorder.weight_dicts)} weight dicts at load time and "
                f"{len(live)} after initialization"
            ]
        return [
            f"{bound.scope}: the weight dict was replaced during initialization"
            for bound, current in zip(recorder.weight_dicts, live)
            if current is not bound
        ]

    def _build_manifest(
        self, alive: List[bool], loaded_specs: List[TensorSpec]
    ) -> Tuple[Manifest, List[_Allocation]]:
        recorder = self._recorder
        assert recorder is not None
        allocations: List[_Allocation] = []
        manifest_index: Dict[int, int] = {}
        storages: List[StorageSpec] = []
        totals: List[torch.Tensor] = []
        for index, tracked in enumerate(recorder.storages):
            if not alive[index]:
                continue
            allocation = tracked.allocation()
            assert allocation is not None  # alive means its lease still holds one
            if allocation not in allocations:
                allocations.append(allocation)
            manifest_index[index] = len(storages)
            data, _bridge = allocation.as_torch_storage(
                tracked.offset_bytes, tracked.nbytes
            )
            totals.append(_checksum_total(data))
            storages.append(
                StorageSpec(
                    allocation_index=allocations.index(allocation),
                    offset_bytes=tracked.offset_bytes,
                    nbytes=tracked.nbytes,
                    checksum="",
                )
            )
        # Keep the GPU checksums batched without moving CPU weight bytes to the GPU.
        for device in {total.device for total in totals}:
            indices = [i for i, total in enumerate(totals) if total.device == device]
            values = torch.stack([totals[i] for i in indices]).cpu().tolist()
            for i, value in zip(indices, values):
                storages[i].checksum = _format_checksum(value)
        manifest = Manifest(
            allocations=[
                (
                    AllocationSpec(MemoryKind.CPU, a.size_bytes, a.cuda_access_required)
                    if isinstance(a, CpuAllocation)
                    else AllocationSpec(MemoryKind.CUDA, a.size_bytes)
                )
                for a in allocations
            ],
            storages=storages,
            tensors=[
                replace(spec, storage_index=manifest_index.get(spec.storage_index, -1))
                for spec in list(loaded_specs) + recorder.init_tensor_specs
            ],
            transform_returned_tensors={
                scope: dict(groups)
                for scope, groups in recorder.transform_returned_tensors.items()
            },
            init_scalars={
                scope: dict(scalars) for scope, scalars in recorder.init_scalars.items()
            },
        )
        return manifest, allocations

    # ---- verify mode --------------------------------------------------------------------

    def begin_verify(self) -> None:
        """Take this role's old record for comparison, releasing its allocation FDs."""
        try:
            record = store.drop_slot(self._store_name, self.slot_key)
            if record is not None:
                self._verify_against = record
        except (OSError, store.StoreError) as error:
            logging.warning(
                "resident weights verify %s: could not take the published record: %s",
                self.slot_key,
                error,
            )

    def _compare_with_store(self, manifest: Manifest) -> None:
        old, self._verify_against = self._verify_against, None
        if old is None:
            logging.info(
                "resident weights verify %s: nothing published before", self.slot_key
            )
            return
        if old.fingerprint != self.fingerprint:
            logging.warning(
                "resident weights verify %s: the published record has another fingerprint; "
                "changed fields: %s",
                self.slot_key,
                _changed_fields(old.fingerprint_text, self.fingerprint_text),
            )
            return
        now = _tensor_layouts(manifest)
        differ = (
            _differing(_tensor_layouts(old.manifest), now)
            + _differing(
                _by_label(old.manifest.transform_returned_tensors),
                _by_label(manifest.transform_returned_tensors),
            )
            + _differing(
                _by_label(old.manifest.init_scalars), _by_label(manifest.init_scalars)
            )
        )
        if differ:
            logging.error(
                "resident weights verify %s: %d published values differ from the previous run: %s",
                self.slot_key,
                len(differ),
                ", ".join(differ[:50]),
            )
        else:
            logging.info(
                "resident weights verify %s: all %d weights match the published ones",
                self.slot_key,
                len(now),
            )

    # ---- writes by EPLB and host tables -------------------------------------------------

    def allocation_of(self, tensor: torch.Tensor) -> Optional[_Allocation]:
        ptr = tensor.untyped_storage().data_ptr()
        for ref in self._allocation_refs:
            allocation = ref()
            if (
                allocation is not None
                and (tensor.device.type == "cpu")
                == isinstance(allocation, CpuAllocation)
                and allocation.ptr <= ptr < allocation.ptr + allocation.size_bytes
            ):
                return allocation
        return None

    def eplb_begin(self) -> None:
        if self._published or self._attached:
            self._notify(
                lambda: store.eplb_begin(self._store_name, self.slot_key),
                "eplb_begin",
            )

    def eplb_commit(self, eplb_placement: Dict[int, List[int]]) -> None:
        if self._published or self._attached:
            self._notify(
                lambda: store.eplb_commit(
                    self._store_name,
                    self.slot_key,
                    {str(k): v for k, v in eplb_placement.items()},
                ),
                "eplb_commit",
            )

    def retain_host_file(self, file_path: str, lock_path: str) -> None:
        def retain():
            if not store.retain_host_file(self._store_name, file_path, lock_path):
                logging.warning("resident weight store could not retain %s", file_path)

        self._notify(retain, "retain_host_file")

    def _notify(self, call: Callable[[], None], what: str) -> None:
        """A request made while serving: a store that is gone holds nothing to keep consistent."""
        try:
            call()
        except (OSError, store.StoreError) as error:
            logging.warning(
                "resident weight store %s: %s skipped: %s",
                self._store_name,
                what,
                error,
            )


def _scoped_dicts(model_weights) -> List[Tuple[ScopeKey, dict]]:
    return [(i, layer) for i, layer in enumerate(model_weights.weights)] + [
        (GLOBAL_SCOPE, model_weights.global_weights)
    ]


def _close_fds(fds: Sequence[int]) -> None:
    for fd in fds:
        os.close(fd)


@dataclass(frozen=True)
class _TensorLayout:
    """What verify mode compares across two runs: a tensor's identity and its bytes."""

    origin: WeightOrigin
    device: MemoryKind
    dtype: str
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    storage_offset: int
    checksum: Optional[str]  # None for a zero-byte tensor, which has no storage


def _tensor_layouts(manifest: Manifest) -> Dict[str, _TensorLayout]:
    return {
        f"{spec.scope}/{spec.name}": _TensorLayout(
            spec.origin,
            spec.device,
            spec.dtype,
            tuple(spec.shape),
            tuple(spec.strides),
            spec.storage_offset,
            (
                manifest.storages[spec.storage_index].checksum
                if spec.storage_index >= 0
                else None
            ),
        )
        for spec in manifest.tensors
    }


def _by_label(by_scope: Mapping[ScopeKey, Mapping[str, object]]) -> Dict[str, object]:
    return {
        f"{scope}/{key}": value
        for scope, values in by_scope.items()
        for key, value in values.items()
    }


def _differing(before: Mapping[str, object], now: Mapping[str, object]) -> List[str]:
    return sorted(
        key for key in before.keys() | now.keys() if before.get(key) != now.get(key)
    )


def _changed_fields(before: str, now: str) -> str:
    def fields(text):
        current, values = "", {}
        for line in text.splitlines():
            if line and not line.startswith(" ") and ": " in line:
                current = line.split(": ", 1)[0]
            values[current] = values.get(current, "") + line
        return values

    a, b = fields(before), fields(now)
    return (
        ", ".join(
            sorted(key for key in a.keys() | b.keys() if a.get(key) != b.get(key))
        )
        or "unknown"
    )


def _get_manager_and_scope(weights) -> Tuple[Optional[ResidentWeights], ScopeKey]:
    """The manager recording or replaying this weight dict, if resident weights are active."""
    if isinstance(weights, WeightDict):
        manager = weights.manager
        if manager is not None:
            return manager, weights.scope
    return None, GLOBAL_SCOPE


def resident_init_tensors(
    weight_dict: Mapping[str, torch.Tensor],
    transform_key: str,
    output_names: Sequence[str],
    consumed_names: Sequence[str],
    builder: Callable[[], Optional[Sequence[torch.Tensor]]],
) -> Optional[Tuple[torch.Tensor, ...]]:
    """Record or replay an initialization-time weight transform.

    ``weight_dict`` is the dict the transform reads from (one layer or the globals),
    ``transform_key`` must encode every setting the result depends on, ``output_names`` names the
    outputs so a restart can hand them back, ``consumed_names`` names the entries of
    ``weight_dict`` the transform pops or drops, and ``builder`` returns one tensor per output
    name or None for a degenerate branch.

    With resident weights off this is ``builder()``. On a cold start the outputs are moved into
    read-only resident memory and published; on a restart the published outputs are returned
    without calling ``builder`` -- and without touching ``consumed_names``, which the publishing
    run consumed and which are not in the dict at all.
    """
    manager, scope = _get_manager_and_scope(weight_dict)
    if manager is None:
        out = builder()
        return None if out is None else tuple(out)
    return manager.init_tensors(
        weight_dict, scope, transform_key, output_names, consumed_names, builder
    )


def resident_init_scalar(
    weight_dict: Mapping[str, torch.Tensor],
    scalar_key: str,
    builder: Callable[[], ScalarValue],
) -> ScalarValue:
    """Record or replay a non-tensor value derived from weights a transform is about to consume."""
    manager, scope = _get_manager_and_scope(weight_dict)
    if manager is None:
        return builder()
    return manager.init_scalar(scope, scalar_key, builder)


def should_consume_loaded(
    weight_dict: Mapping[str, torch.Tensor], consumed_names: Sequence[str]
) -> bool:
    """Whether the caller must run cold-start preprocessing that consumes loaded weights.

    True with resident weights off and on a cold start (where the loader names are also recorded
    as settled). False on a restart: the published weights already went through this
    preprocessing, those loaded names are gone, and the caller must skip it.
    """
    manager, scope = _get_manager_and_scope(weight_dict)
    if manager is None:
        return True
    return manager.should_consume(weight_dict, scope, consumed_names)


def prepare_host_for_cuda(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Prepare a CPU weight for GPU reads, retaining shared backing when managed.

    Ordinary CPU weights stay pageable until a consumer requests access. Each new
    process registers its own mapping; only that requirement is published.
    """
    for manager in list(_managers):
        allocation = manager.allocation_of(tensor)
        if isinstance(allocation, CpuAllocation):
            allocation.ensure_cuda_access(device)
            return tensor
    return tensor if tensor.is_pinned() else tensor.pin_memory()


def writable_view(tensor: torch.Tensor) -> torch.Tensor:
    """A read-write view of a resident tensor (same layout, second mapping); others unchanged."""
    for manager in list(_managers):
        allocation = manager.allocation_of(tensor)
        if allocation is None:
            continue
        if isinstance(allocation, CpuAllocation):
            raise PermissionError(
                "resident CPU weights are sealed and cannot have a writable view"
            )
        storage = tensor.untyped_storage()
        offset = storage.data_ptr() - allocation.ptr
        data, _bridge = allocation.as_torch_storage(
            offset, storage.nbytes(), base_ptr=allocation.writable_alias_ptr()
        )
        return _view(
            data,
            tensor.dtype,
            tensor.storage_offset(),
            tensor.size(),
            tensor.stride(),
        )
    return tensor


def resident_weights_of(tensor: torch.Tensor) -> Optional[ResidentWeights]:
    for manager in list(_managers):
        if manager.allocation_of(tensor) is not None:
            return manager
    return None


def retain_host_file(file_path: str, lock_path: str) -> None:
    """Keep a host-memory table file (held under LOCK_SH by its users) until the store drops it.

    ``lock_path`` is the file's creation lock; the store takes it before deleting the file.
    """
    for manager in list(_managers):
        manager.retain_host_file(file_path, lock_path)
        return


def checkpoint_digest(path: str, follow_index: bool = False) -> str:
    """Cheap guard of file names, sizes and mtimes, not a content checksum.

    Covers the top-level files and, with ``follow_index``, the subdirectory shards the
    safetensors index points at: the main checkpoint's loader reads those, while p-tuning and
    LoRA directories are read by loaders that ignore the index. Users must explicitly
    invalidate after a same-size rewrite that preserves mtime. Do not read full checkpoints on
    the warm path.
    """
    entries = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                stat = entry.stat()
                entries.append(f"{entry.name}:{stat.st_size}:{stat.st_mtime_ns}")
    index = os.path.join(path, "model.safetensors.index.json")
    if follow_index and os.path.isfile(index):
        with open(index) as f:
            shards = set(json.load(f).get("weight_map", {}).values())
        for shard in sorted(
            name for name in shards if os.sep in os.path.normpath(name)
        ):
            stat = os.stat(os.path.join(path, shard))
            entries.append(f"{shard}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(sorted(entries)).encode()).hexdigest()


def all_ranks_min_state(state: int, device: str) -> int:
    """Agree on the minimum integer state across participating WORLD ranks."""
    dist = torch.distributed
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return state
    tensor_device = device if dist.get_backend() == "nccl" else "cpu"
    flag = torch.tensor(int(state), dtype=torch.int32, device=tensor_device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return int(flag.item())


def all_ranks_true(value: bool, device: str) -> bool:
    """Agree before entering or skipping checkpoint-loader WORLD collectives."""
    return bool(all_ranks_min_state(int(value), device))
