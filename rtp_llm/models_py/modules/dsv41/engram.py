# SPDX-License-Identifier: Apache-2.0
# Hashing and gating adapted from vLLM (Copyright contributors to the vLLM project).
"""DeepSeek V4.1 Engram with shared, CPU-resident lookup tables.

Hashing and the gated residual match vLLM's DeepSeek V4.1 implementation.
The large FP8 embedding tensors remain in host memory throughout their
lifetime. Only selected rows and the small projection are copied to CUDA.
Named shared-memory mappings share physical pages across colocated CP/DP
workers. CUDA registers these pages as pinned host RAM before graph capture;
the GPU directly gathers selected rows through UVA.

Token windows are explicit, rather than cached by request ID or KV slot.
The caller supplies ``[current, previous, previous-2, previous-3]`` for each
token, including CP boundary tokens, PD prompt history, and speculative
candidate tokens. This avoids stale history after prefix reuse or rollback.
The CUDA hash and lookup paths perform no host transfers during capture.
"""

import ctypes
import fcntl
import hashlib
import json
import logging
import math
import mmap
import os
import shutil
import uuid
import weakref
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)
DEAD_ID = -1


def _config_value(config, name):
    return config[name] if isinstance(config, Mapping) else getattr(config, name)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % prime == 0:
            return n == prime
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        value = pow(a, d, n)
        if value in (1, n - 1):
            continue
        for _ in range(r - 1):
            value = value * value % n
            if value == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Use the checkpoint tokenizer's exact training-time normalization."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    keys = {}
    lookup = []
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            key = normalizer.normalize_str(text) or text
        if key not in keys:
            keys[key] = len(keys)
        lookup.append(keys[key])
    return lookup, len(keys)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, compressed_vocab_size: int
) -> torch.Tensor:
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(0, bound, size=max_ngram_size, dtype=np.int64)
        rows.append(torch.from_numpy(values * 2 + 1))
    return torch.stack(rows)


class EngramLayout:
    def __init__(self, config):
        self.layer_ids = tuple(_config_value(config, "engram_layer_ids"))
        self.num_embeddings = tuple(_config_value(config, "engram_num_embeddings"))
        self.max_ngram_size = int(_config_value(config, "engram_max_ngram_size"))
        self.n_heads = int(_config_value(config, "engram_n_heads"))
        self.head_dim = int(_config_value(config, "engram_head_dim"))
        self.compressed_vocab_size = int(
            _config_value(config, "engram_compressed_vocab_size")
        )
        self.pad_token_id = int(_config_value(config, "engram_pad_token_id"))
        if len(self.layer_ids) != len(self.num_embeddings):
            raise ValueError("Engram layer IDs and embedding counts must match")
        if self.max_ngram_size < 2 or self.n_heads < 1:
            raise ValueError("Engram requires at least one head and a 2-gram")
        seen, primes = set(), []
        for _ in self.layer_ids:
            per_layer = []
            for _ in range(self.max_ngram_size - 1):
                current = int(_config_value(config, "engram_vocab_size")) - 1
                sizes = []
                for _ in range(self.n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_layer.append(tuple(sizes))
            primes.append(tuple(per_layer))
        self.primes = tuple(primes)
        self.n_hash_cols = (self.max_ngram_size - 1) * self.n_heads
        flat = [[p for group in layer for p in group] for layer in primes]
        self.offsets = torch.tensor(
            [np.cumsum([0, *sizes[:-1]]).tolist() for sizes in flat],
            dtype=torch.int64,
            device="cpu",
        )
        for count, sizes in zip(self.num_embeddings, flat):
            if sum(sizes) > count:
                raise ValueError("Engram prime buckets exceed checkpoint table size")


def make_token_windows(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lookback_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Build windows for packed request chunks, without crossing requests.

    ``lookback_token_ids[b]`` is ordered most recent first and contains the
    tokens immediately preceding chunk b, or -1 before the sequence starts.
    A CP shard supplies its own boundary history, not the global chunk's.
    """
    ids = input_ids.detach().to(device="cpu", dtype=torch.int64).flatten()
    boundaries = cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
    lookback = lookback_token_ids.detach().to(device="cpu", dtype=torch.int64)
    depth = lookback.shape[1]
    windows = torch.empty((ids.numel(), depth + 1), dtype=torch.int64, device="cpu")
    for request, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if start == end:
            continue
        history = torch.cat((lookback[request].flip(0), ids[start:end]))
        # unfold presents oldest first; hashing uses current first.
        windows[start:end] = history.unfold(0, depth + 1, 1).flip(1)
    return windows


class NgramHashState:
    """Stateless CPU/CUDA hashing; explicit windows make rollback unambiguous."""

    def __init__(
        self,
        layout: EngramLayout,
        tokenizer_path: Optional[str] = None,
        *,
        tokenizer=None,
        token_map: Optional[list[int]] = None,
        device=None,
    ):
        self.layout = layout
        if token_map is None:
            if tokenizer is None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path, trust_remote_code=True
                )
            token_map, size = build_compressed_token_map(tokenizer)
        else:
            size = max(token_map) + 1
        if size != layout.compressed_vocab_size:
            raise ValueError(
                f"Engram compressed vocab mismatch: tokenizer={size}, "
                f"checkpoint={layout.compressed_vocab_size}"
            )
        self.token_map = torch.tensor(token_map, dtype=torch.int64, device="cpu")
        self.pad_id = int(self.token_map[layout.pad_token_id])
        self.multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, size
        )
        self.primes = torch.tensor(layout.primes, dtype=torch.int64, device="cpu")
        self._device_buffers = {}
        if device is not None and torch.device(device).type == "cuda":
            self.prepare_device(device)

    def prepare_device(self, device):
        device = torch.device(device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device not in self._device_buffers:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Engram hash constants must be initialized before capture"
                )
            self._device_buffers[device] = tuple(
                tensor.to(device)
                for tensor in (
                    self.token_map,
                    self.multipliers,
                    self.primes,
                    self.layout.offsets,
                )
            )
        return self._device_buffers[device]

    def hash_token_windows(
        self, token_windows: torch.Tensor, dead_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if token_windows.is_cuda:
            from rtp_llm.models_py.modules.dsv41 import _engram_triton

            return _engram_triton.hash_token_windows(
                token_windows,
                dead_mask,
                self.prepare_device(token_windows.device),
                self.layout,
                self.pad_id,
            )
        windows = token_windows.detach().to(device="cpu", dtype=torch.int64)
        blocked = windows < 0
        if dead_mask is not None:
            blocked = blocked | dead_mask.to(device="cpu", dtype=torch.bool)
        # An image or sequence boundary terminates every longer n-gram.
        blocked = blocked.to(torch.int32).cumsum(dim=1) > 0
        mapped = self.token_map[windows.clamp_min(0)]
        mapped = torch.where(blocked, self.pad_id, mapped)
        hashes = torch.empty(
            (windows.shape[0], len(self.layout.layer_ids), self.layout.n_hash_cols),
            dtype=torch.int64,
            device="cpu",
        )
        rolling = torch.zeros(
            (windows.shape[0], len(self.layout.layer_ids)),
            dtype=torch.int64,
            device="cpu",
        )
        for shift in range(self.layout.max_ngram_size):
            rolling ^= mapped[:, shift, None] * self.multipliers[None, :, shift]
            if shift:
                start = (shift - 1) * self.layout.n_heads
                end = start + self.layout.n_heads
                hashes[:, :, start:end] = (
                    rolling[:, :, None] % self.primes[None, :, shift - 1]
                    + self.layout.offsets[None, :, start:end]
                )
        return hashes

    def __call__(self, token_windows, dead_mask=None):
        return self.hash_token_windows(token_windows, dead_mask)


def _scale_to_float(scales: torch.Tensor) -> torch.Tensor:
    if scales.dtype == torch.uint8:
        # Match the reference kernel's exponent-bit interpretation, including 0.
        return (scales.to(torch.int32) << 23).view(torch.float32)
    return scales.to(torch.float32)


def _prefault_host_tensor(tensor: torch.Tensor, name: str) -> None:
    """Read mapped pages into RAM without making a private table copy."""
    nbytes = tensor.numel() * tensor.element_size()
    if not nbytes:
        return
    page = os.sysconf("SC_PAGE_SIZE")
    pointer = tensor.data_ptr()
    start = pointer - pointer % page
    span = pointer + nbytes - start
    libc = ctypes.CDLL(None, use_errno=True)
    # MADV_WILLNEED starts read-ahead; touching every page waits for completion.
    libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(span), ctypes.c_int(3))
    raw = tensor.view(torch.uint8).reshape(-1).numpy()
    int(raw[::page].sum(dtype=np.uint64))
    int(raw[-1])
    logger.info(
        "Engram %s: %.2f GiB prefaulted into shared host RAM (no CUDA table copy)",
        name,
        nbytes / 1024**3,
    )


def _live_leases(data_path: Path):
    leases = []
    for lease in data_path.parent.glob(data_path.name + ".lease.*"):
        try:
            pid = int(lease.name.split(".lease.", 1)[1].split(".", 1)[0])
            os.kill(pid, 0)
        except ProcessLookupError:
            lease.unlink(missing_ok=True)
        except (ValueError, PermissionError):
            leases.append(lease)
        else:
            leases.append(lease)
    return leases


def _release_shared_table(data_path, lock_path, lease_path):
    with open(lock_path, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        Path(lease_path).unlink(missing_ok=True)
        if not _live_leases(Path(data_path)):
            Path(data_path).unlink(missing_ok=True)


class _CudaArrayView:
    def __init__(self, owner, tensor, pointer):
        self.owner = owner
        self.__cuda_array_interface__ = {
            "shape": tuple(tensor.shape),
            "strides": tuple(tensor.stride()),
            "typestr": "|u1",
            "data": (pointer, False),
            "version": 3,
        }


def _register_shared_host_memory(runtime, pointer, nbytes):
    """Bound concurrent large host registrations across local model ranks.

    On H20/CUDA13, concurrent registration of two full Engram tables by
    eight ranks has failed with cudaErrorInvalidValue. Serialization bounds
    registration concurrency but does not guarantee success: later full-table
    probes still failed on some ranks while registering the second table.
    The creation lock is per table and cannot protect overlapping calls
    for different layers. This lock covers registration only, never lookup.
    """
    path = Path("/tmp") / f"rtp_llm_engram_{os.getuid()}_cuda_register.lock"
    with open(path, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return runtime.cudaHostRegister(pointer, nbytes, 3)  # Portable | Mapped


def _register_shared_host_ranges(runtime, pointer, nbytes, chunk_bytes):
    """Register bounded ranges while retaining a contiguous device view.

    Only the last range may end within a host page. No table bytes move.
    A failed call rolls back all successful ranges; never fall back to CPU.
    The caller must keep the host storage alive until every range is released.
    """
    if nbytes <= 0 or pointer % mmap.PAGESIZE:
        raise ValueError("Engram registration needs nonempty page-aligned storage")
    if chunk_bytes < 0 or chunk_bytes % mmap.PAGESIZE:
        raise ValueError("Engram registration chunk must be a nonnegative page multiple")
    registered = []
    device_base = None
    try:
        for offset in range(0, nbytes, chunk_bytes or nbytes):
            address = pointer + offset
            size = min(chunk_bytes or nbytes, nbytes - offset)
            (result,) = _register_shared_host_memory(runtime, address, size)
            if int(result) != 0:
                _, name = runtime.cudaGetErrorName(result)
                raise RuntimeError(
                    f"Engram cudaHostRegister failed: {result} ({name}); "
                    f"offset={offset}, bytes={size}, total_bytes={nbytes}"
                )
            registered.append(address)
            result, device_pointer = runtime.cudaHostGetDevicePointer(address, 0)
            if int(result) != 0:
                raise RuntimeError(f"Engram chunk device pointer failed: {result}")
            if device_base is None:
                device_base = device_pointer
            if device_pointer != device_base + offset:
                raise RuntimeError("Engram registered ranges have noncontiguous device addresses")
        return tuple(registered)
    except BaseException:
        for address in reversed(registered):
            (result,) = runtime.cudaHostUnregister(address)
            if int(result) != 0:
                logger.error("Engram registration rollback failed: ptr=%#x error=%s", address, result)
        raise


class _PinnedSharedTable:
    """One physical tmpfs allocation shared by independent prefill/decode groups.

    Disk-backed MAP_SHARED mappings cannot be CUDA-registered on all drivers.
    A tmpfs mapping supports registration without the private COW copies that
    pinning a safetensors MAP_PRIVATE mapping could introduce. A lock protects
    one-time creation. Per-process leases remove the file after the last reader
    exits; stale leases from SIGKILL are pruned on the next access.
    """

    def __init__(self, weight, scales, source_identity, device, *, register=True):
        digest = hashlib.sha256(source_identity.encode()).hexdigest()[:24]
        data_path = Path("/dev/shm") / f"rtp_llm_engram_{os.getuid()}_{digest}"
        lock_path = Path("/tmp") / (data_path.name + ".lock")
        lease_path = Path(str(data_path) + f".lease.{os.getpid()}.{uuid.uuid4().hex}")
        weight_bytes = weight.numel()
        total_bytes = weight_bytes + scales.numel()
        if weight.element_size() != 1 or scales.element_size() != 1:
            raise ValueError("Pinned Engram tables require native FP8/E8M0 bytes")
        with open(lock_path, "a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _live_leases(data_path)
            for partial in data_path.parent.glob(data_path.name + ".partial.*"):
                try:
                    os.kill(int(partial.name.rsplit(".", 1)[1]), 0)
                except ProcessLookupError:
                    partial.unlink(missing_ok=True)
            if not data_path.exists():
                free = shutil.disk_usage(data_path.parent).free
                if free < total_bytes:
                    raise RuntimeError(
                        f"Engram needs {total_bytes / 1024**3:.2f} GiB shared RAM, "
                        f"but /dev/shm has {free / 1024**3:.2f} GiB free"
                    )
                partial = Path(str(data_path) + f".partial.{os.getpid()}")
                try:
                    with open(partial, "w+b") as file:
                        os.chmod(partial, 0o600)
                        file.truncate(total_bytes)
                        mapping = mmap.mmap(file.fileno(), total_bytes)
                        try:
                            destination = np.frombuffer(mapping, dtype=np.uint8)
                            offset = 0
                            for tensor in (weight, scales):
                                source = tensor.view(torch.uint8).reshape(-1).numpy()
                                for begin in range(0, source.size, 64 * 1024**2):
                                    end = min(begin + 64 * 1024**2, source.size)
                                    destination[offset + begin : offset + end] = source[
                                        begin:end
                                    ]
                                offset += source.size
                            del destination, source
                        finally:
                            mapping.close()
                    os.replace(partial, data_path)
                    logger.info(
                        "Created shared host Engram table %s (%.2f GiB)",
                        data_path,
                        total_bytes / 1024**3,
                    )
                except BaseException:
                    partial.unlink(missing_ok=True)
                    raise
            if data_path.stat().st_size != total_bytes:
                raise RuntimeError(
                    f"Engram shared table has an invalid size: {data_path}"
                )
            with open(data_path, "r+b") as file:
                mapping = mmap.mmap(file.fileno(), total_bytes)
            lease_path.touch(mode=0o600)
        self.storage = torch.frombuffer(mapping, dtype=torch.uint8)
        self.weight = self.storage[:weight_bytes].view(weight.dtype).view(weight.shape)
        self.scales = self.storage[weight_bytes:].view(torch.uint8).view(scales.shape)
        self._lease_finalizer = weakref.finalize(
            self, _release_shared_table, str(data_path), str(lock_path), str(lease_path)
        )
        self.device = torch.device(device)
        self.registered = False
        if not register:
            logger.warning(
                "Engram shared CPU lookup enabled for %s on %s; "
                "no CUDA host registration, eager-only with host transfers",
                data_path,
                self.device,
            )
            return
        from cuda.bindings import runtime

        with torch.cuda.device(self.device):
            # Limit per-call pinning metadata on Hopper hosts. A 94 GiB single
            # range can require large GUP/VMA arrays on older Linux kernels.
            # Other architectures retain the original one-range policy.
            chunk_mb = int(os.environ.get(
                "DSV41_ENGRAM_REGISTER_CHUNK_MB",
                "256" if torch.cuda.get_device_capability(self.device) == (9, 0) else "0",
            ))
            if chunk_mb < 0:
                raise ValueError("DSV41_ENGRAM_REGISTER_CHUNK_MB must be nonnegative")
            (prior_error,) = runtime.cudaPeekAtLastError()
            logger.info(
                "Engram register path=%s device=%s ptr=%#x bytes=%d prior_error=%s chunk_mb=%d",
                data_path,
                self.device,
                self.storage.data_ptr(),
                total_bytes,
                prior_error,
                chunk_mb,
            )
            try:
                registered_ranges = _register_shared_host_ranges(
                    runtime, self.storage.data_ptr(), total_bytes, chunk_mb * 1024**2
                )
            except BaseException:
                self._lease_finalizer()
                raise
        self._registration_finalizer = weakref.finalize(
            self, self._unregister, runtime, self.storage, registered_ranges
        )
        if not self.storage.is_pinned():
            self._registration_finalizer()
            self._lease_finalizer()
            raise RuntimeError("CUDA did not recognize the Engram shared pinned pages")
        self.runtime = runtime
        self.registered = True
        logger.info(
            "Engram CUDA-registered %.2f GiB shared pinned host RAM on %s (%d ranges)",
            total_bytes / 1024**3,
            self.device,
            len(registered_ranges),
        )

    @staticmethod
    def _unregister(runtime, storage, registered_ranges):
        # storage is intentionally retained by the finalizer until all ranges
        # are unregistered, including when the owning module is collected.
        for address in reversed(registered_ranges):
            (result,) = runtime.cudaHostUnregister(address)
            if int(result) != 0:
                logger.warning("Engram cudaHostUnregister failed: ptr=%#x error=%s", address, result)

    def cuda_view(self, tensor):
        if not self.registered:
            raise RuntimeError("Engram CPU lookup table has no registered CUDA view")
        with torch.cuda.device(self.device):
            result, pointer = self.runtime.cudaHostGetDevicePointer(
                tensor.data_ptr(), 0
            )
            if int(result) != 0:
                raise RuntimeError(f"Engram cudaHostGetDevicePointer failed: {result}")
            return torch.as_tensor(
                _CudaArrayView(self, tensor, pointer), device=self.device
            )


class HostEngramEmbedding:
    """Native FP8/E8M0 host tensors deliberately not registered as buffers.

    ``nn.Module.cuda()`` must never move these multi-GiB tables to a GPU.
    Checkpoint tensor storage retains the safetensors mapping after close.
    """

    def __init__(
        self, weight: torch.Tensor, scales: torch.Tensor, block_size=32, pinned=None
    ):
        if weight.device.type != "cpu" or scales.device.type != "cpu":
            raise ValueError("Engram embedding tables must remain in host RAM")
        if weight.ndim != 2 or scales.shape != (
            weight.shape[0],
            weight.shape[1] // block_size,
        ):
            raise ValueError("Engram embedding scales do not match table geometry")
        self.weight = weight
        # CPU index_select does not implement Float8; select its unchanged bytes.
        self.scales = scales.view(torch.uint8) if scales.element_size() == 1 else scales
        self.block_size = block_size
        self._pinned = pinned
        self._uva = None
        if pinned is not None:
            # Startup warmup also uses the device geometry for CPU lookup.
            self._num_sms = torch.cuda.get_device_properties(
                pinned.device
            ).multi_processor_count
            if pinned.registered:
                self._uva = (
                    pinned.cuda_view(self.weight),
                    pinned.cuda_view(self.scales),
                )

    def __call__(self, indices: torch.Tensor, device, *, out=None) -> torch.Tensor:
        if indices.is_cuda and self._uva is not None:
            from rtp_llm.models_py.modules.dsv41 import _engram_triton

            return _engram_triton.lookup_host_rows(
                *self._uva, indices, self._num_sms, out=out
            )
        if out is not None:
            raise RuntimeError("Preallocated Engram lookup requires the CUDA UVA path")
        if indices.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Engram CPU lookup requires eager execution, not CUDA Graph"
            )
        indices = indices.detach().to(device="cpu", dtype=torch.int64)
        valid = (indices >= 0) & (indices < self.weight.shape[0])
        flat = indices.masked_fill(~valid, 0).flatten()
        if self.weight.element_size() == 1:
            rows = (
                self.weight.view(torch.uint8)
                .index_select(0, flat)
                .view(self.weight.dtype)
            )
        else:
            rows = self.weight.index_select(0, flat)
        scales = _scale_to_float(self.scales.index_select(0, flat))
        rows = rows.float() * scales.repeat_interleave(self.block_size, dim=1)
        rows = rows.to(torch.bfloat16).view(*indices.shape, self.weight.shape[1])
        rows.masked_fill_(~valid[..., None], 0)
        return rows.to(device=device)


class UvaEngramPrefetch:
    """Bounded all-layer lookup on a graph-capturable auxiliary stream.

    Construct before capture. ``begin`` forks from the calling stream after
    hashes are ready; each consumer waits its own ready event. ``finish`` must
    run in a finally block on that same calling stream, including failures.
    Eager record_stream plus the final join protect allocation lifetime; graph
    capture retains fixed tensor addresses through its private allocation pool.
    This object is request-local in use, not safe for concurrent forwards.
    """

    def __init__(self, device, layer_ids):
        self.device = torch.device(device)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Create UVA prefetch streams before graph capture")
        self.stream = torch.cuda.Stream(device=self.device)
        self.fork = torch.cuda.Event()
        self.done = torch.cuda.Event()
        self.ready = {layer_id: torch.cuda.Event() for layer_id in layer_ids}
        self.rows = {}
        self._hashes = ()
        self._origin = None

    def begin(self, lookups):
        if self._origin is not None:
            raise RuntimeError("Previous UVA Engram prefetch has not been joined")
        lookups = tuple(lookups)
        if not lookups:
            return
        # Validate and allocate on the consumer stream before the fork. No
        # large embedding table is copied; only this bounded batch's rows.
        for layer_id, embedding, indices in lookups:
            if layer_id not in self.ready or embedding._uva is None:
                raise RuntimeError("UVA Engram prefetch requires registered layer tables")
            if not indices.is_cuda or indices.device != self.device:
                raise ValueError("UVA Engram prefetch hashes must reside on its CUDA device")
        self.rows = {
            layer_id: torch.empty(
                (*indices.shape, embedding.weight.shape[1]),
                dtype=torch.bfloat16, device=self.device,
            )
            for layer_id, embedding, indices in lookups
        }
        self._hashes = tuple(indices for _, _, indices in lookups)
        origin = torch.cuda.current_stream(self.device)
        capturing = torch.cuda.is_current_stream_capturing()
        self._origin = origin
        try:
            self.fork.record(origin)
            self.stream.wait_event(self.fork)
            with torch.cuda.stream(self.stream):
                try:
                    for layer_id, embedding, indices in lookups:
                        if not capturing:
                            indices.record_stream(self.stream)
                        embedding(indices, self.device, out=self.rows[layer_id])
                        self.ready[layer_id].record(self.stream)
                finally:
                    # Join all enqueued side work even if a later lookup fails.
                    self.done.record(self.stream)
        except BaseException:
            self.finish()
            raise

    def wait(self, layer_id):
        if self._origin is None or layer_id not in self.rows:
            raise RuntimeError("No active UVA Engram prefetch for this layer")
        current = torch.cuda.current_stream(self.device)
        current.wait_event(self.ready[layer_id])
        rows = self.rows[layer_id]
        if not torch.cuda.is_current_stream_capturing():
            rows.record_stream(current)
        return rows

    def finish(self):
        if self._origin is None:
            return
        try:
            current = torch.cuda.current_stream(self.device)
            if current != self._origin:
                raise RuntimeError("Join UVA Engram prefetch on its originating stream")
            current.wait_event(self.done)
        finally:
            self.rows = {}
            self._hashes = ()
            self._origin = None


def dequantize_block_weight(
    weight: torch.Tensor, scales: torch.Tensor, block_size: int = 32
) -> torch.Tensor:
    """Materialize only the small WKV projection, never an embedding table."""
    if weight.ndim != 2:
        raise ValueError("Engram WKV must be a matrix")
    expected = tuple(math.ceil(size / block_size) for size in weight.shape)
    if tuple(scales.shape) != expected:
        raise ValueError(f"Engram WKV scales {scales.shape} do not match {expected}")
    output = torch.empty_like(weight, dtype=torch.bfloat16)
    # Bound startup scratch memory for the 25,600 x 6,144 V4.1 projection.
    for begin in range(0, weight.shape[0], 1024):
        end = min(begin + 1024, weight.shape[0])
        scale = _scale_to_float(
            scales[begin // block_size : math.ceil(end / block_size)]
        )
        scale = scale.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
        output[begin:end] = (
            weight[begin:end].float() * scale[: end - begin, : weight.shape[1]]
        )
    return output


def mxfp8_activation_reference(x: torch.Tensor) -> torch.Tensor:
    """V4.1 group-32 E4M3/E8M0 activation rounding before WKV matmul."""
    blocked = x.float().reshape(*x.shape[:-1], -1, 32)
    amax = blocked.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
    exponent = (torch.ceil(torch.log2(amax / 448.0)) + 127).clamp(0, 254)
    scale = torch.exp2(exponent - 127)[..., None]
    values = (blocked / scale).to(torch.float8_e4m3fn).float() * scale
    return values.reshape(x.shape).to(x.dtype)


def _engram_inject_supported(hidden, q_weight, k_weight, token_mask):
    return (
        not torch.is_grad_enabled()
        and hidden.is_cuda
        and hidden.ndim == 3
        and hidden.numel() > 0
        and hidden.shape[-2:] == (4, 5120)
        and q_weight.shape == k_weight.shape == (4, 5120)
        and all(
            value.dtype == torch.bfloat16
            and value.device == hidden.device
            and value.is_contiguous()
            for value in (hidden, q_weight, k_weight)
        )
        and (
            token_mask is None
            or (
                token_mask.device == hidden.device
                and token_mask.dtype == torch.bool
                and token_mask.shape == hidden.shape[:-2]
                and token_mask.is_contiguous()
            )
        )
    )


def gated_engram_residual(
    hidden: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    token_mask: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact normalized signed-square-root gate used by the reference.

    In inference, ``out`` may exactly alias ``hidden``. The native kernel
    owns disjoint rows; the reference materializes its result before copying.
    The caller supplies a contiguous output disjoint from the other inputs.
    """
    if (
        _engram_inject_supported(hidden, q_weight, k_weight, token_mask)
        and kv.shape == hidden.shape[:-2] + (5 * 5120,)
        and kv.dtype == torch.bfloat16
        and kv.device == hidden.device
        and kv.is_contiguous()
    ):
        from rtp_llm.models_py.modules.dsv41._engram_inject_triton import (
            engram_inject_kernel,
        )

        if out is None:
            out = torch.empty_like(hidden)
        engram_inject_kernel[(hidden.numel() // (4 * 5120), 4)](
            hidden,
            kv,
            q_weight,
            k_weight,
            hidden if token_mask is None else token_mask,
            out,
            DIM=5120,
            HC=4,
            HAS_MASK=token_mask is not None,
            EPS=eps,
            BLOCK=8192,
            enable_fp_fusion=False,
        )
        return out
    tokens, copies, dim = hidden.shape
    key = kv[:, : copies * dim].reshape(tokens, copies, dim).float()
    value = kv[:, copies * dim :].reshape(tokens, 1, dim).float()
    h = hidden.float()
    dot = (h * q_weight.float() * k_weight.float() * key).sum(-1)
    dot *= torch.rsqrt(h.square().mean(-1) + eps)
    dot *= torch.rsqrt(key.square().mean(-1) + eps) / math.sqrt(dim)
    gate_input = dot.abs().clamp_min(1e-6).sqrt()
    gate_input = torch.where(dot < 0, -gate_input, gate_input)
    gate = torch.sigmoid(gate_input)
    if token_mask is not None:
        gate = torch.where(token_mask.to(device=hidden.device)[:, None], gate, 0.0)
    result = (h + gate[..., None] * value).to(hidden.dtype)
    return result if out is None else out.copy_(result)


class Engram(nn.Module):
    def __init__(
        self,
        layout: EngramLayout,
        layer_hash_index: int,
        embedding: HostEngramEmbedding,
        wkv: nn.Module,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
    ):
        super().__init__()
        self.layout = layout
        self.layer_hash_index = layer_hash_index
        self.embed_tokens = embedding
        self.wkv = wkv
        self.register_buffer("q_weight", q_weight, persistent=False)
        self.register_buffer("k_weight", k_weight, persistent=False)
        self.eps = eps

    @classmethod
    def from_checkpoint(cls, config, layer_id: int, checkpoint_path: str, device):
        from safetensors import safe_open

        layout = EngramLayout(config)
        layer_hash_index = layout.layer_ids.index(layer_id)
        root = Path(checkpoint_path)
        with (root / "model.safetensors.index.json").open() as file:
            weight_map = json.load(file)["weight_map"]
        prefix = f"layers.{layer_id}.engram."

        def load(suffix):
            name = prefix + suffix
            if name not in weight_map:
                raise KeyError(f"V4.1 checkpoint is missing {name}")
            with safe_open(
                root / weight_map[name], framework="pt", device="cpu"
            ) as file:
                return file.get_tensor(name)

        table, scales = load("embed.weight"), load("embed.scale")
        expected = (layout.num_embeddings[layer_hash_index], layout.head_dim)
        if tuple(table.shape) != expected:
            raise ValueError(
                f"Engram table shape {table.shape} does not match {expected}"
            )
        pinned = None
        if torch.device(device).type == "cuda":
            from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

            identity = [str(root.resolve()), prefix]
            for suffix in ("embed.weight", "embed.scale"):
                path = root / weight_map[prefix + suffix]
                stat = path.stat()
                identity.extend(
                    [str(path.resolve()), str(stat.st_size), str(stat.st_mtime_ns)]
                )
            host_lookup = os.environ.get("DSV41_ENGRAM_HOST_LOOKUP", "uva")
            if host_lookup not in ("uva", "cpu"):
                raise ValueError("DSV41_ENGRAM_HOST_LOOKUP must be uva or cpu")
            pinned = _PinnedSharedTable(
                table,
                scales,
                "|".join(identity),
                device,
                register=host_lookup == "uva",
            )
            table, scales = pinned.weight, pinned.scales
            projection = V41MXFP8Linear(
                load("wkv.weight").to(device), load("wkv.scale").to(device)
            )
        else:
            _prefault_host_tensor(table, prefix + "embed.weight")
            _prefault_host_tensor(scales, prefix + "embed.scale")
            projection = _ReferenceWKV(
                dequantize_block_weight(load("wkv.weight"), load("wkv.scale"))
            )
        return cls(
            layout,
            layer_hash_index,
            HostEngramEmbedding(table, scales, pinned=pinned),
            projection,
            load("q_weight").to(device),
            load("k_weight").to(device),
            float(_config_value(config, "rms_norm_eps")),
        )

    def forward(self, hidden, hash_ids, token_mask=None, *, embedding_rows=None):
        if embedding_rows is not None and (
            embedding_rows.shape != (hidden.shape[0], self.layout.n_hash_cols, self.layout.head_dim)
            or embedding_rows.dtype != torch.bfloat16
            or embedding_rows.device != hidden.device
            or not embedding_rows.is_contiguous()
        ):
            raise ValueError("Prefetched Engram rows must be contiguous same-device BF16 [T,H,D]")
        if hidden.shape[0] == 0:
            return hidden
        # Engram is token-local after hashing. Bound lookup/projection/gate
        # temporaries for packed long prompts without changing the decode path.
        chunk_rows = 32768
        inplace = _engram_inject_supported(
            hidden, self.q_weight, self.k_weight, token_mask
        )
        if hidden.shape[0] > chunk_rows:
            output = (
                hidden
                if inplace
                else torch.empty_like(hidden, memory_format=torch.contiguous_format)
            )
            for begin in range(0, hidden.shape[0], chunk_rows):
                end = min(begin + chunk_rows, hidden.shape[0])
                self._forward_rows(
                    hidden[begin:end],
                    hash_ids[begin:end],
                    None if token_mask is None else token_mask[begin:end],
                    out=output[begin:end],
                    embedding_rows=None if embedding_rows is None else embedding_rows[begin:end],
                )
            return output
        return self._forward_rows(
            hidden, hash_ids, token_mask, out=hidden if inplace else None,
            embedding_rows=embedding_rows,
        )

    def _forward_rows(self, hidden, hash_ids, token_mask, *, out=None, embedding_rows=None):
        rows = (
            self.embed_tokens(hash_ids, hidden.device)
            if embedding_rows is None else embedding_rows
        ).flatten(-2)
        kv = self.wkv(rows)
        return gated_engram_residual(
            hidden, kv, self.q_weight, self.k_weight, self.eps, token_mask, out=out
        )


class _ReferenceWKV(nn.Module):
    """CPU validation path for small synthetic checkpoints."""

    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, rows):
        return F.linear(mxfp8_activation_reference(rows), self.weight)
