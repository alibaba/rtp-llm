# SPDX-License-Identifier: Apache-2.0
"""Private local-cache bootstrap for process-lifetime external Engram.

Startup RPC materializes this cache automatically; the explicit offline launcher
can also transfer the small private files over an authenticated channel.
This module does not discover an Owner, load a checkpoint table, run RPC, compile
native code, or permit live generation replacement. The Owner must outlive all
consumer processes. Experimental decode Graph requires the model attachment's
permanent process-lifetime pin; in-process detach is not supported.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import threading
from uuid import UUID

from .engram_manifest import ExpectedLayout, Limits, parse_manifest, validate_for_attach
from .engram_storage import CleanupPendingError, FabricEngramStorage
from .engram_capacity import CapacityError, capacity_from_environ


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PROCESS_STORAGES = []
_PROCESS_LOCK = threading.Lock()
_STAGED_PACKAGE = None
_STAGED_LOCK = threading.Lock()


def stage_package(package):
    """Hand the exchanged package to the model attach in this process.

    The startup exchange and the model attach share one rank process, so the
    package is passed in memory and never materialized as bundle files.
    """
    global _STAGED_PACKAGE
    with _STAGED_LOCK:
        if _STAGED_PACKAGE is not None and _STAGED_PACKAGE != package:
            raise BootstrapError("PACKAGE_ALREADY_STAGED")
        _STAGED_PACKAGE = package


def staged_package():
    with _STAGED_LOCK:
        return _STAGED_PACKAGE


def clear_staged_package():
    global _STAGED_PACKAGE
    with _STAGED_LOCK:
        _STAGED_PACKAGE = None


class BootstrapError(ValueError):
    """Bounded error category; never formats capability bytes or payloads."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _configured_limits(environ=None):
    try:
        allocation, total = capacity_from_environ(environ)
    except CapacityError as error:
        raise BootstrapError(error.code) from None
    return Limits(max_manifest_bytes=1 << 20, max_tensors=1024,
                  max_allocation_bytes=allocation, max_total_allocation_bytes=total)


def retain_for_process(storage):
    """Prevent Python object destruction from unmapping a live engine resource.

    No atexit/GC close is registered: CUDA/OS process teardown owns final cleanup
    until the real engine has an explicit, verified drain/destroy contract.
    """
    with _PROCESS_LOCK:
        if not any(item is storage for item in _PROCESS_STORAGES):
            _PROCESS_STORAGES.append(storage)


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise BootstrapError("INVALID_JSON")
            result[key] = value
        return result

    def invalid(_):
        raise BootstrapError("INVALID_JSON")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, UnicodeError, RecursionError):
        raise BootstrapError("INVALID_JSON") from None


def _private_read(path, limit):
    """Open a regular, current-UID private file without following its symlink."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) & 0o077):
                raise BootstrapError("UNSAFE_BOOTSTRAP_FILE")
            if info.st_size > limit:
                raise BootstrapError("BOOTSTRAP_TOO_LARGE")
            data = stream.read(limit + 1)
            if len(data) > limit:
                raise BootstrapError("BOOTSTRAP_TOO_LARGE")
            return data
    except OSError:
        raise BootstrapError("BOOTSTRAP_FILE_UNAVAILABLE") from None


def _digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def hash_state_digest(hash_state):
    """Bind the actual tokenizer mapping and all lookup hash constants.

    Algorithm semantics are versioned independently of file names. Constants
    remain on CPU in NgramHashState; computing this digest must not pull a model
    tensor from the GPU or require remote checkpoint table files.
    """
    import torch

    layout = hash_state.layout
    metadata = dict(
        algorithm="dsv41-xor-int64-boundary-cumsum-v1",
        layer_ids=list(layout.layer_ids), num_embeddings=list(layout.num_embeddings),
        head_dim=layout.head_dim, max_ngram_size=layout.max_ngram_size,
        n_heads=layout.n_heads, compressed_vocab_size=layout.compressed_vocab_size,
        pad_token_id=layout.pad_token_id, compressed_pad_id=hash_state.pad_id,
    )
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode("ascii"))
    for name, tensor in (("token_map", hash_state.token_map),
                         ("multipliers", hash_state.multipliers),
                         ("primes", hash_state.primes), ("offsets", layout.offsets)):
        if tensor.device.type != "cpu" or tensor.dtype != torch.int64:
            raise BootstrapError("HASH_CONSTANTS_MUST_BE_CPU_INT64")
        descriptor = json.dumps([name, list(tensor.shape)], separators=(",", ":")).encode("ascii")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(tensor.detach().contiguous().numpy().astype("<i8", copy=False).tobytes())
    return "sha256:" + digest.hexdigest()


def _load_native():
    try:
        return importlib.import_module("engram_native_dev_492ecf45")
    except ImportError:
        raise BootstrapError("PREBUILT_ENGRAM_NATIVE_MODULE_REQUIRED") from None


def _verify_samples(views, validated, samples, device):
    import torch

    for tensor in validated.manifest.tensors:
        sample = samples[tensor.name]
        indices = torch.tensor(sample["row_ids"], dtype=torch.int64, device=device)
        selected = views[tensor.name].index_select(0, indices).cpu()
        if _digest(selected.numpy().tobytes()) != sample["sha256"]:
            raise BootstrapError("GPU_READ_SAMPLE_MISMATCH")
    torch.cuda.synchronize(device)
    return True


def _attach_validated(raw, dataset_id, generation, model_revision, handles, samples,
                      *, layout, hash_state, device, environ=None):
    """Shared attach tail: validate, map once and verify before READY."""
    import torch

    target = torch.device(device)
    if target.type != "cuda":
        raise BootstrapError("CUDA_DEVICE_REQUIRED")
    limits = _configured_limits(environ)
    actual_hash_abi = hash_state_digest(hash_state)
    expected = ExpectedLayout(
        layers=tuple(zip(layout.layer_ids, layout.num_embeddings)),
        head_dim=layout.head_dim, model_revision=model_revision,
        hash_abi=actual_hash_abi)
    ordinal = target.index if target.index is not None else torch.cuda.current_device()
    # Reject an older/smaller bridge before importing even the first allocation.
    parsed = parse_manifest(raw, limits)
    if parsed.hash_abi != actual_hash_abi:
        raise BootstrapError("HASH_ABI_MISMATCH")
    validated = validate_for_attach(parsed, expected, dataset_id, generation, limits)
    manifest = validated.manifest
    from .engram_shard_config import ShardConfig
    ShardConfig.from_environ(environ, inferred_count=len(manifest.tensors[0].shards)).validate_manifest(manifest)
    native = _load_native()
    native_limit = getattr(native, "max_allocation_bytes", None)
    if (type(native_limit) is not int or native_limit <= 0
            or any(shard.allocation_bytes > native_limit
                   for t in manifest.tensors for shard in t.shards)):
        raise BootstrapError("NATIVE_ALLOCATION_LIMIT")
    try:
        storage = FabricEngramStorage.attach(
            raw, expected_layout=expected, dataset_id=dataset_id,
            generation=generation, handles=handles, device=ordinal,
            native=native, limits=limits)
    except CleanupPendingError as error:
        retain_for_process(error.resource)
        raise
    del handles
    try:
        storage.verify(lambda views, verified: _verify_samples(views, verified, samples, target))
        return storage
    except BaseException as original:
        try:
            storage.close()
        except BaseException:
            retain_for_process(storage)
            raise CleanupPendingError("external Engram verification cleanup incomplete", storage) from original
        raise


_PACKAGE_FIELDS = frozenset((
    "raw", "dataset_id", "generation", "model_revision", "capabilities", "samples",
))


def attach_from_package(package, *, layout, hash_state, device, environ=None):
    """Attach from an in-memory startup package. Serving never reads bundle files."""
    if type(package) is not dict or set(package) != _PACKAGE_FIELDS:
        raise BootstrapError("INVALID_PACKAGE")
    raw = package["raw"]
    if type(raw) is not bytes or not raw:
        raise BootstrapError("INVALID_PACKAGE")
    for key in ("dataset_id", "generation", "model_revision"):
        if type(package[key]) is not str or not package[key]:
            raise BootstrapError("INVALID_PACKAGE")
    if not _DIGEST.fullmatch(package["dataset_id"]):
        raise BootstrapError("INVALID_PACKAGE_DIGEST")
    try:
        if str(UUID(package["generation"])) != package["generation"]:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise BootstrapError("INVALID_GENERATION") from None
    limits = _configured_limits(environ)
    manifest = parse_manifest(raw, limits)
    names = {tensor.name for tensor in manifest.tensors}
    samples = package["samples"]
    if type(samples) is not dict or samples.keys() != names:
        raise BootstrapError("INCOMPLETE_READ_SAMPLES")
    for tensor in manifest.tensors:
        sample = samples[tensor.name]
        required = sorted({0, tensor.shape[0] // 2, tensor.shape[0] - 1} |
                          {row for part in tensor.shards for row in (part.row_begin, part.row_end - 1)})
        if (type(sample) is not dict or sample.keys() != {"row_ids", "sha256"}
                or type(sample["row_ids"]) is not list
                or any(type(row) is not int for row in sample["row_ids"])
                or sample["row_ids"] != required
                or type(sample["sha256"]) is not str or not _DIGEST.fullmatch(sample["sha256"])):
            raise BootstrapError("INVALID_READ_SAMPLE")
    capabilities = package["capabilities"]
    allocation_ids = {shard.allocation_id for tensor in manifest.tensors for shard in tensor.shards}
    if type(capabilities) is not dict or capabilities.keys() != allocation_ids:
        raise BootstrapError("CAPABILITY_SET_MISMATCH")
    for value in capabilities.values():
        if type(value) is not bytes or not 0 < len(value) <= 1024:
            raise BootstrapError("INVALID_CAPABILITY_ENCODING")
    return _attach_validated(
        raw, package["dataset_id"], package["generation"], package["model_revision"],
        capabilities, samples,
        layout=layout, hash_state=hash_state, device=device, environ=environ)
