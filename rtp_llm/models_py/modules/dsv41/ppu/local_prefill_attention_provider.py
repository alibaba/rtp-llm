"""Opt-in reuse of the existing, isolated 4305 BF16 prefill provider.

The receipt certifies candidate identity only. Numeric qualification belongs
to the unchanged component gates; this adapter never claims it from a decode
receipt, changes cache encoding, or caches an attention payload.
"""

import hashlib
import importlib
import inspect
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import weakref

import torch

from rtp_llm.models_py.modules.dsv41._profiler import record_function_range
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu


ENV = "RTP_V41_PPU_LOCAL_PREFILL_ATTENTION_PROVIDER"
RECEIPT_ENV = "RTP_V41_PPU_LOCAL_PREFILL_ATTENTION_PROVIDER_RECEIPT"
SOURCE_REVISION = "4305d1bdaeb8451eb5d78c67a5122ecf38f5014d"
UPSTREAM_INTERFACE_SHA256 = "8861f77da7d7900ffeb60cb2f652176d4342c947d516fbcc0928eba81e952d43"
PACKAGE_SHA256 = "270a9b68485d187c9cc405c712413347841b1e80ec7281b85748932bc9c62b3f"
WRAPPER_SHA256 = "fa5b209f087c29d9784959174f006baf9a86778488e94fb898273eeccb0f294e"
EXTENSION_SHA256 = "f6c930dc908bd9e57d076ad7983b4368fb266e80448d9671e130b379047ad7ba"
MODULE = "flash_mla_v41_packed"
EXTENSION_MODULE = MODULE + ".flash_mla_cuda"


def candidate_receipt():
    """Exact candidate schema; deliberately contains no numeric PASS field."""
    return dict(
        format="rtp-v41-local-prefill-attention-provider-v1",
        qualification="IDENTITY_VERIFIED_NUMERIC_PENDING",
        source_revision=SOURCE_REVISION,
        upstream_interface_sha256=UPSTREAM_INTERFACE_SHA256,
        module=MODULE, extension_module=EXTENSION_MODULE,
        native_entry="sparse_prefill_fwd", package_sha256=PACKAGE_SHA256,
        wrapper_sha256=WRAPPER_SHA256, extension_sha256=EXTENSION_SHA256,
    )


def receipt_reason(receipt):
    if not isinstance(receipt, dict):
        return "attention_provider_receipt_not_object"
    for name, expected in candidate_receipt().items():
        if receipt.get(name) != expected:
            return "attention_provider_receipt_mismatch:" + name
    return None


@dataclass(frozen=True)
class _Provider:
    native: object
    extension: object
    entry: object
    identity: dict


@lru_cache(maxsize=1)
def _import_provider():
    """Hash the fixed import once per process, never replace a global provider."""
    module = importlib.import_module(MODULE)
    extension = importlib.import_module(EXTENSION_MODULE)
    native = getattr(module, "flash_mla_sparse_fwd", None)
    entry = getattr(extension, "sparse_prefill_fwd", None)
    if (not callable(native) or not callable(entry)
            or getattr(native, "__globals__", {}).get("flash_mla_cuda") is not extension
            or getattr(extension, "__name__", None) != EXTENSION_MODULE):
        raise ValueError("attention_provider_native_binding_mismatch")
    paths = {
        "package": Path(module.__file__).resolve(),
        "wrapper": Path(inspect.getfile(native)).resolve(),
        "extension": Path(extension.__file__).resolve(),
    }
    for name, expected in (("package", PACKAGE_SHA256), ("wrapper", WRAPPER_SHA256),
                           ("extension", EXTENSION_SHA256)):
        if hashlib.sha256(paths[name].read_bytes()).hexdigest() != expected:
            raise ValueError("attention_provider_import_hash_mismatch:" + name)
    identity = candidate_receipt()
    identity.update({name + "_path": str(path) for name, path in paths.items()})
    return _Provider(native, extension, entry, identity)


def load_provider(receipt_path):
    """Validate each forward's explicit receipt before using the cached import."""
    path = Path(receipt_path)
    if not path.is_absolute():
        raise ValueError("attention_provider_absolute_receipt_required")
    receipt = json.loads(path.read_text())
    reason = receipt_reason(receipt)
    if reason is not None:
        raise ValueError(reason)
    return _import_provider()


def _normal(tensor):
    return (isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided
            and not tensor.is_neg() and not tensor.is_conj())


def _key(tensor):
    if tensor is None:
        return None
    return (tensor._cdata, tensor.data_ptr(), tensor.shape, tensor.stride(),
            tensor.storage_offset(), tensor.dtype, tensor.device,
            None if tensor.is_inference() else tensor._version)


def _metadata_snapshot(state):
    inputs, attn, cache = (getattr(state, name, None) for name in ("inputs", "attn", "cache"))
    values = (getattr(inputs, "input_ids", None), getattr(inputs, "combo_position_ids", None),
              getattr(attn, "cu_seqlens", None))
    if (not _normal(values[0]) or not _normal(values[2])
            or any(value is not None and not _normal(value) for value in values)):
        return None
    lengths, prefixes = tuple(getattr(state, "lengths", ())), tuple(getattr(state, "query_prefixes", ()))
    totals = tuple(getattr(cache, "lengths", ()))
    if (len(lengths) != 1 or len(prefixes) != 1 or len(totals) != 1
            or any(type(value) is not int or value < 0 for value in (*lengths, *prefixes, *totals))
            or totals[0] != lengths[0] + prefixes[0]):
        return None
    runtime = getattr(state, "runtime", None)
    return ((id(inputs), id(attn), id(runtime), id(getattr(runtime, "model", None)),
             lengths, prefixes, totals, tuple(_key(value) for value in values)), values)


def _device_reason(q):
    if q.device.type != "cuda" or not is_ppu():
        return "not_ppu"
    if torch.cuda.get_device_capability(q.device) != (8, 9):
        return "device_arch"
    if (torch.cuda.current_device() != q.device.index
            or torch.cuda.is_current_stream_capturing()
            or torch.cuda.current_stream(q.device) != torch.cuda.default_stream(q.device)):
        return "stream_or_capture"
    return None


class LocalPrefillAttentionProvider:
    """One forward's admission and host evidence, with no retained payload."""

    def __init__(self, state):
        flag = os.environ.get(ENV, "0")
        if flag not in ("0", "1"):
            raise ValueError(ENV + " must be explicitly 0 or 1")
        self.enabled = flag == "1"
        self.stats = dict(enabled=self.enabled, decision="pending" if self.enabled else "disabled",
                          calls=0, attempts=0, native_errors=0, layers=[], rows=[], identity={},
                          operands=[], fallbacks={}, closed=False)
        self._receipt = os.environ.get(RECEIPT_ENV, "")
        self._binding = self._state = self._cache = self._birth = None
        self._metadata_owners = ()
        self._invalid = None
        self._unavailable = False
        self._last_layer = -1
        if self.enabled:
            try:
                self._state, self._cache = weakref.ref(state), weakref.ref(getattr(state, "cache", None))
            except TypeError:
                self._invalid = "birth_owner_unsupported"
            snapshot = _metadata_snapshot(state)
            if snapshot is None:
                self._invalid = "birth_metadata_unavailable"
            else:
                self._birth, self._metadata_owners = snapshot

    def _decline(self, reason):
        if self.enabled:
            counts = self.stats["fallbacks"]
            counts[reason] = counts.get(reason, 0) + 1
            if not self.stats["calls"]:
                self.stats["decision"] = reason
        return None

    def _reason(self, state, layer_id, geometry, q, kv, indices, lengths, sink):
        if self.stats["closed"]:
            return "closed"
        if not self.enabled:
            return "disabled_at_birth"
        if self._invalid:
            return self._invalid
        if self._state() is not state or self._cache() is not state.cache:
            self._invalid = "forward_owner_changed"
            return self._invalid
        if os.environ.get(ENV, "0") != "1":
            return "flag_disabled"
        if os.environ.get(RECEIPT_ENV, "") != self._receipt:
            self._invalid = "receipt_changed"
            return self._invalid
        snapshot = _metadata_snapshot(state)
        if snapshot is None or snapshot[0] != self._birth:
            self._invalid = "metadata_changed"
            return self._invalid
        from rtp_llm.models_py.modules.dsv41.ppu.layers import _local_prefill_business_phase

        if (getattr(state.runtime, "current", None) is not state
                or not _local_prefill_business_phase(state)
                or getattr(state, "compacted", False) or getattr(state, "restoring_aux", False)):
            return "phase"
        cp = getattr(state, "context", None)
        if cp is not None and (getattr(cp, "gather_restore_positions", None) is not None
                               or getattr(cp, "swa_replay_start", None) is not None
                               or getattr(cp, "swa_replay_starts_host", None) is not None):
            return "compact_context"
        if type(layer_id) is not int or not 0 <= layer_id < 20:
            return "layer"
        if layer_id <= self._last_layer:
            return "layer_order"
        operands = q, kv, indices, lengths, sink
        if any(not _normal(value) or value.requires_grad for value in operands if value is not None):
            return "unresolved_layout"
        if any(value is None for value in (q, kv, indices, lengths)):
            return "missing_operand"
        rows = state.lengths[0]
        if (rows not in (4096, 4097) or geometry.head_dim != 512 or geometry.n_heads != 64
                or geometry.window != 128 or type(geometry.sm_scale()) not in (float, int)
                or geometry.sm_scale() != 512 ** -0.5):
            return "geometry"
        ratio = geometry.ratio(layer_id)
        if ratio not in (0, 2) or (ratio and geometry.kv_source_for(layer_id) is None):
            return "ratio"
        prefix, total = state.query_prefixes[0], state.cache.lengths[0]
        bank = rows + min(prefix, 127) + (total // ratio if ratio else 0)
        shapes = ((rows, 64, 512), (bank, 1, 512), (rows, 1, 640), (rows,))
        dtypes = torch.bfloat16, torch.bfloat16, torch.int32, torch.int32
        if (any(value.shape != shape or value.dtype != dtype or not value.is_contiguous()
                or value.device != q.device for value, shape, dtype in zip(operands[:4], shapes, dtypes))
                or (sink is not None and (sink.shape != (64,) or sink.dtype != torch.float32
                                         or sink.device != q.device or not sink.is_contiguous()))
                or q.device != state.inputs.input_ids.device):
            return "tensor_contract"
        return _device_reason(q)

    def try_run(self, state, layer_id, geometry, q, kv, indices, lengths, attn_sink):
        reason = self._reason(state, layer_id, geometry, q, kv, indices, lengths, attn_sink)
        if reason is not None:
            return self._decline(reason)
        if not self._receipt:
            return self._decline("receipt_missing_at_birth")
        if self._unavailable:
            return self._decline("provider_unavailable")
        if self._binding is None:
            try:
                self._binding = load_provider(self._receipt)
            except ImportError as error:
                self._unavailable = True
                self.stats["unavailable_detail"] = type(error).__name__ + ": " + str(error)
                return self._decline("provider_unavailable")
            self.stats["identity"] = dict(self._binding.identity)
            self.stats["identity"]["receipt_path"] = self._receipt
        binding = self._binding
        if (binding.native.__globals__.get("flash_mla_cuda") is not binding.extension
                or getattr(binding.extension, "sparse_prefill_fwd", None) is not binding.entry):
            raise ValueError("attention_provider_live_binding_changed")
        self.stats["attempts"] += 1
        self._last_layer = layer_id
        def descriptor(value):
            if value is None:
                return None
            return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                        device=str(value.device), storage_offset=value.storage_offset())
        self.stats["operands"].append(dict(
            layer=layer_id, q=descriptor(q), kv=descriptor(kv), indices=descriptor(indices),
            lengths=descriptor(lengths), sink=descriptor(attn_sink), scale=geometry.sm_scale(),
        ))
        with record_function_range(f"dsv41.ppu.local_attention_provider.L{layer_id:02d}.full"):
            try:
                result = binding.native(q, kv, indices, geometry.sm_scale(), d_v=512,
                                        attn_sink=attn_sink, topk_length=lengths)
            except Exception:
                self.stats["native_errors"] += 1
                self._invalid = "native_error"
                raise
        self.stats["calls"] += 1
        self.stats["decision"] = "active"
        self.stats["layers"].append(layer_id)
        self.stats["rows"].append(q.shape[0])
        return result

    def close(self):
        self._binding = self._state = self._cache = self._birth = None
        self._metadata_owners = ()
        self.stats["closed"] = True
