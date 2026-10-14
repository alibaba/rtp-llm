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
import threading
import time
import uuid
import weakref
from concurrent.futures import CancelledError, ThreadPoolExecutor, TimeoutError
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


class _PinnedSharedTable:
    """One physical tmpfs allocation shared by independent prefill/decode groups.

    Disk-backed MAP_SHARED mappings cannot be CUDA-registered on all drivers.
    A tmpfs mapping supports registration without the private COW copies that
    pinning a safetensors MAP_PRIVATE mapping could introduce. A lock protects
    one-time creation. Per-process leases remove the file after the last reader
    exits; stale leases from SIGKILL are pruned on the next access.
    """

    def __init__(self, weight, scales, source_identity, device):
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
        from cuda.bindings import runtime

        with torch.cuda.device(self.device):
            (result,) = runtime.cudaHostRegister(
                self.storage.data_ptr(), total_bytes, 3  # Portable | Mapped
            )
            if int(result) != 0:
                self._lease_finalizer()
                raise RuntimeError(f"Engram cudaHostRegister failed: {result}")
        self._lookup_events = []
        self._registration_finalizer = weakref.finalize(
            self, self._unregister, runtime, self.storage, self._lookup_events
        )
        if not self.storage.is_pinned():
            self._registration_finalizer()
            self._lease_finalizer()
            raise RuntimeError("CUDA did not recognize the Engram shared pinned pages")
        self.runtime = runtime
        logger.info(
            "Engram CUDA-registered %.2f GiB shared pinned host RAM on %s",
            total_bytes / 1024**3,
            self.device,
        )

    @staticmethod
    def _unregister(runtime, storage, lookup_events):
        # Unlike CUDA allocator tensors, mapped host storage has no record_stream
        # support. Keep its registration until every asynchronous reader exits.
        for event in lookup_events:
            event.synchronize()
        (result,) = runtime.cudaHostUnregister(storage.data_ptr())
        if int(result) != 0:
            logger.warning("Engram cudaHostUnregister failed: %s", result)

    def cuda_view(self, tensor):
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
            self._uva = (pinned.cuda_view(self.weight), pinned.cuda_view(self.scales))
            self._num_sms = torch.cuda.get_device_properties(
                pinned.device
            ).multi_processor_count

    def warmup_prefetch(self, indices: torch.Tensor, device):
        """Compile and configure the asynchronous lookup during model startup."""
        from rtp_llm.models_py.modules.dsv41._engram_triton import (
            warmup_lookup_prefetch_rows,
        )

        return warmup_lookup_prefetch_rows(*self._uva, indices, self._num_sms)

    def __call__(
        self, indices: torch.Tensor, device, *, prefetch=False
    ) -> torch.Tensor:
        if indices.is_cuda:
            from rtp_llm.models_py.modules.dsv41 import _engram_triton

            lookup = (
                _engram_triton.lookup_prefetch_rows
                if prefetch
                else _engram_triton.lookup_host_rows
            )
            return lookup(*self._uva, indices, self._num_sms)
        indices = indices.detach().to(device="cpu", dtype=torch.int64)
        valid = indices >= 0
        flat = indices.clamp_min(0).flatten()
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


def engram_lookup_prefetch_enabled(*, is_decode_role=False):
    return (
        not engram_cpu_lookup_enabled(is_decode_role=is_decode_role)
        and os.getenv("DSV41_ASYNC_ENGRAM_LOOKUP", "0") == "1"
    )


def engram_cpu_lookup_enabled(*, is_decode_role=False):
    """Resolve the role default without mutating process-wide environment.

    A configured CPU lookup still only runs on ordinary target prefill.
    Decode verification, draft, and CUDA graph capture keep the GPU path.
    """
    default = "0" if is_decode_role else "1"
    return os.getenv("DSV41_CPU_ENGRAM_LOOKUP", default) == "1"


def _hash_token_windows_cpu(state, windows, image_token, num_threads=None):
    """Native token-parallel hash; NumPy remains an independent CPU oracle."""
    if num_threads is not None:
        from rtp_llm.models_py.modules.dsv41.engram_cpu import hash_token_windows_cpu

        return hash_token_windows_cpu(
            windows,
            state.token_map,
            state.multipliers,
            state.primes,
            state.layout.offsets,
            pad_id=state.pad_id,
            image_token=image_token,
            num_threads=num_threads,
        )
    values = windows.detach().numpy().astype(np.int64, copy=False)
    token_map = state.token_map.numpy()
    blocked = np.logical_or.accumulate(
        (values < 0) | (values >= token_map.size) | (values == image_token), axis=1
    )
    mapped = token_map[np.clip(values, 0, token_map.size - 1)]
    mapped = np.where(blocked, state.pad_id, mapped)
    multipliers = state.multipliers.numpy()
    primes = state.primes.numpy()
    offsets = state.layout.offsets.numpy()
    layout = state.layout
    hashes = np.empty(
        (values.shape[0], len(layout.layer_ids), layout.n_hash_cols), dtype=np.int64
    )
    rolling = np.zeros((values.shape[0], len(layout.layer_ids)), dtype=np.int64)
    for shift in range(layout.max_ngram_size):
        # NumPy int64 multiply wraps like the CPU/CUDA tensor hash, and signed
        # remainder uses the same nonnegative result for each positive prime.
        rolling ^= mapped[:, shift, None] * multipliers[None, :, shift]
        if shift:
            start, end = (shift - 1) * layout.n_heads, shift * layout.n_heads
            hashes[:, :, start:end] = (
                rolling[:, :, None] % primes[None, :, shift - 1]
                + offsets[None, :, start:end]
            )
    return torch.from_numpy(hashes)


def _timed_cpu_hash(state, windows, image_token, num_threads, timing):
    if timing is not None:
        timing["hash_started_ns"] = time.perf_counter_ns()
    try:
        return _hash_token_windows_cpu(state, windows, image_token, num_threads)
    finally:
        if timing is not None:
            timing["hash_finished_ns"] = time.perf_counter_ns()


class _EngramCPULookupSlot:
    """Fixed host storage; request outputs belong to their allocation stream."""

    def __init__(self, module, capacity):
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

        self.embedding = module.embed_tokens
        self.device = module.q_weight.device
        self.hash_index = module.layer_hash_index
        self.heads = module.layout.n_hash_cols
        self.capacity = capacity
        self.num_threads = int(os.getenv("DSV41_CPU_ENGRAM_LOOKUP_THREADS", "16"))
        if not 1 <= self.num_threads <= 16:
            raise ValueError("DSV41_CPU_ENGRAM_LOOKUP_THREADS must be between 1 and 16")
        self.collect_timing = os.getenv("DSV41_CPU_ENGRAM_LOOKUP_TIMING", "0") == "1"
        self.last_timing = None
        self.deadline_enabled = True
        self.packed = type(module.wkv) is V41MXFP8Linear
        self.head_dim = module.layout.head_dim
        self.width = self.heads * (
            self.head_dim + self.head_dim // 32 if self.packed else self.head_dim
        )
        self.dtype = torch.uint8 if self.packed else torch.bfloat16
        self.host = torch.empty(
            (capacity, self.width), dtype=self.dtype, device="cpu", pin_memory=True
        )
        self.stream = torch.cuda.Stream(device=self.device)
        self.copied = torch.cuda.Event()
        self.allocated = torch.cuda.Event()
        current = torch.cuda.current_stream(self.device)
        self.allocated.record(current)
        self.stream.wait_event(self.allocated)
        self.copied.record(self.stream)

    def allocate_output(self, count):
        current = torch.cuda.current_stream(self.device)
        output = torch.empty(
            (count, self.width),
            dtype=getattr(self, "dtype", torch.bfloat16),
            device=self.device,
        )
        # Record before submitting the worker. A later worker-side recording
        # could include the consumer's wait for copied and create a cycle.
        self.allocated.record(current)
        return output, current

    def run(self, hashes, output, submitted_ns=None, work=None):
        from rtp_llm.models_py.modules.dsv41.engram_cpu import lookup_host_rows_cpu

        with torch.inference_mode(), torch.cuda.device(self.device):
            try:
                timing = (
                    {"submitted_ns": submitted_ns, "started_ns": time.perf_counter_ns()}
                    if self.collect_timing
                    else None
                )
                if work is not None:
                    if work.timing is not None:
                        work.timing.update(timing)
                    timing = work.timing
                    self.last_timing = timing
                    if work.abandoned:
                        return None
                ids = torch.from_numpy(
                    np.ascontiguousarray(hashes.result().numpy()[:, self.hash_index])
                )
                if timing is not None:
                    timing.update(getattr(hashes, "engram_timing", None) or {})
                if work is not None and work.abandoned:
                    return None
                # The host buffer may be overwritten once its preceding H2D
                # finishes, independently of the previous GPU consumer.
                self.copied.synchronize()
                host = self.host[: output.shape[0]]
                if timing is not None:
                    timing["lookup_started_ns"] = time.perf_counter_ns()
                lookup = lookup_host_rows_cpu
                if getattr(self, "packed", False):
                    from rtp_llm.models_py.modules.dsv41.engram_cpu import (
                        lookup_host_rows_cpu_packed,
                    )

                    lookup = lookup_host_rows_cpu_packed
                lookup(
                    self.embedding.weight,
                    self.embedding.scales,
                    ids,
                    host,
                    num_threads=self.num_threads,
                )
                if timing is not None:
                    timing["lookup_finished_ns"] = time.perf_counter_ns()
                # Abandonment and copy submission must be one atomic decision.
                # Once H2D is submitted, consuming it is cheaper than starting
                # a competing UVA lookup, and its owning stream must join it.
                if work is None:
                    self._copy(output, host)
                else:
                    with work.lock:
                        if work.abandoned:
                            self.last_timing = timing
                            return None
                        self._copy(output, host)
                        work.copy_submitted = True
                if timing is not None:
                    timing["copy_submitted_ns"] = time.perf_counter_ns()
                self.last_timing = timing
                return output
            except BaseException:
                self.stream.synchronize()
                raise

    def _copy(self, output, host):
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.allocated)
            output.copy_(host, non_blocking=True)
            self.copied.record(self.stream)

    def close(self):
        self.stream.synchronize()


def _close_cpu_lookup(executor, slots):
    executor.shutdown(wait=True)
    for slot in slots:
        slot.close()


class EngramCPULookupRuntime:
    """Startup-owned worker and bounded buffers; request code never compiles."""

    def __init__(self, model, capacity):
        from rtp_llm.models_py.modules.dsv41.engram_cpu import prepare_cpu_lookup

        prepare_cpu_lookup()
        self.capacity = min(int(capacity), 32768)
        self.num_threads = int(os.getenv("DSV41_CPU_ENGRAM_LOOKUP_THREADS", "16"))
        if not 1 <= self.num_threads <= 16:
            raise ValueError("DSV41_CPU_ENGRAM_LOOKUP_THREADS must be between 1 and 16")
        self.retired = []
        self.slots = {}
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="engram-cpu-lookup"
        )
        try:
            for layer_id, _ in model._engram_layers:
                module = model.v4.layers[layer_id].engram
                if layer_id < 20 and module.q_weight.is_cuda:
                    self.slots[layer_id] = _EngramCPULookupSlot(module, self.capacity)
            # Initialize the Python worker, native thread team and H2D stream /
            # events before readiness. Use real CPU hashes and each real table.
            state = model._engram_hash_state
            windows = torch.zeros(
                (self.capacity, state.layout.max_ngram_size), dtype=torch.int32
            )
            hashes = self.executor.submit(
                _hash_token_windows_cpu, state, windows, -2, self.num_threads
            )
            for slot in self.slots.values():
                output, _ = slot.allocate_output(self.capacity)
                future = self.executor.submit(slot.run, hashes, output)
                future.result()
                slot.copied.synchronize()
                if getattr(slot, "packed", False):
                    from rtp_llm.models_py.modules.dsv41._engram_packed_triton import (
                        unpack_packed_rows,
                    )

                    pair = unpack_packed_rows(output, slot.heads, slot.head_dim)
                    if pair is None:
                        raise RuntimeError(
                            "Engram packed output failed startup qualification"
                        )
                    torch.cuda.current_stream(slot.device).synchronize()
                    del pair
                # Warm the allocation, copy and events without retaining a
                # device tensor or a Future holding it after startup.
                del future, output
        except BaseException:
            _close_cpu_lookup(self.executor, self.slots.values())
            raise
        self._finalizer = weakref.finalize(
            self, _close_cpu_lookup, self.executor, tuple(self.slots.values())
        )

    def close(self):
        self._finalizer()
        self.reap()

    def reap(self):
        """Bound abandoned work to one request; never queue behind old lookups."""
        pending = []
        error = None
        for future in self.retired:
            if future.done():
                try:
                    future.result()
                except BaseException as exc:
                    error = error or exc
            else:
                pending.append(future)
        self.retired = pending
        if error is not None:
            raise error  # Surface background errors once, including at shutdown.
        return not pending

    def retire(self, work):
        self.retired.append(self.executor.submit(work._drain))


def prepare_engram_cpu_lookup(model, *, max_m):
    is_decode_role = bool(getattr(model, "_is_decode_role", False))
    if (
        is_decode_role
        or not engram_cpu_lookup_enabled(is_decode_role=is_decode_role)
        or max_m <= 0
    ):
        return
    runtime = getattr(model, "_engram_cpu_runtime", None)
    if runtime is not None and runtime.capacity >= min(max_m, 32768):
        return
    if runtime is not None:
        runtime.close()
        model._engram_cpu_runtime = None
    runtime = EngramCPULookupRuntime(model, max_m)
    model._engram_cpu_runtime = runtime
    logger.info(
        "[DSV41 Engram] CPU lookup ready=True capacity=%d slots=%d "
        "workers=1 threads=%s pinned_bytes=%d device_bytes=%d",
        runtime.capacity,
        len(runtime.slots),
        tuple(slot.num_threads for slot in runtime.slots.values()),
        sum(
            slot.host.numel() * slot.host.element_size()
            for slot in runtime.slots.values()
        ),
        0,  # Device outputs are allocated per request and released after WKV.
    )


class _EngramCPULookupWork:
    def __init__(self, slot, ids, runtime=None):
        self.slot, self.ids = slot, ids
        # The model/request plan owns the runtime. A queued worker must not
        # become its final owner: finalization on that same worker would call
        # executor.shutdown(wait=True) and try to join its own thread.
        self.runtime = weakref.proxy(runtime) if runtime is not None else None
        self.lock = threading.Lock()
        self.abandoned = self.copy_submitted = self.retired = False
        self.timing = None
        self.future = None
        self.allocation_stream = None
        self.signature = (ids.shape, ids.stride(), ids.storage_offset(), ids.dtype)
        self.version = _tensor_version(ids)
        self.consumed = False

    def submit(self, executor, hashes, count, submitted_ns):
        if getattr(self.slot, "collect_timing", False):
            self.timing = {"submitted_ns": submitted_ns}
            self.slot.last_timing = self.timing
        output, self.allocation_stream = self.slot.allocate_output(count)
        if self.runtime is None:
            self.future = executor.submit(self.slot.run, hashes, output, submitted_ns)
        else:
            self.future = executor.submit(
                self.slot.run, hashes, output, submitted_ns, self
            )

    def _abandon_before_copy(self):
        with self.lock:
            if self.copy_submitted:
                return False
            self.abandoned = True
            self.future.cancel()
            return True

    def consume(self, ids, hidden):
        if not (
            self.ids is ids
            and self.signature
            == (ids.shape, ids.stride(), ids.storage_offset(), ids.dtype)
            and self.version == _tensor_version(ids)
            and hidden.shape[0] == ids.shape[0]
            and hidden.device == self.slot.device
            and torch.cuda.current_stream(hidden.device) == self.allocation_stream
        ):
            self.clear()
            return None
        started = time.perf_counter_ns() if self.slot.collect_timing else None
        current = torch.cuda.current_stream(hidden.device)
        polls = 0
        if self.runtime is not None and self.slot.deadline_enabled:
            while not self.future.done():
                # Query before submitting WKV: completion means the GPU has
                # reached this consumer, not merely that a wall-clock estimate
                # expired. No extra event or GPU kernel is needed. Yield the
                # GIL between probes rather than busy-polling the driver.
                polls += 1
                if current.query() and self._abandon_before_copy():
                    if self.timing is not None:
                        self.timing.update(
                            outcome="gpu_deadline_fallback",
                            consumer_wait_started_ns=started,
                            consumer_wait_finished_ns=time.perf_counter_ns(),
                            consumer_stream_queries=polls,
                        )
                    self.clear()
                    return None
                try:
                    self.future.result(timeout=0.001)
                except TimeoutError:
                    if self.future.done():
                        # Completion can race the timeout. Retrieve the actual
                        # result/worker error instead of rethrowing a stale
                        # wait timeout after a successful completion.
                        self.future.result()
                        break
        rows = self.future.result()
        if self.slot.last_timing is not None:
            self.slot.last_timing["consumer_wait_started_ns"] = started
            self.slot.last_timing["consumer_wait_finished_ns"] = time.perf_counter_ns()
            self.slot.last_timing["consumer_stream_queries"] = polls
            self.slot.last_timing["outcome"] = "consumed"
        current.wait_event(self.slot.copied)
        self.consumed = True
        # Join the copy on the allocation's owning stream before returning it.
        # Dropping the Future lets WKV release the tensor before forward ends;
        # stream-ordered reuse needs no record_stream allocator polling.
        self.ids = self.future = None
        return rows

    def clear(self):
        self.ids = None
        if self.runtime is not None and self.future is not None:
            if not self.retired:
                self._abandon_before_copy()
                self.retired = True
                self.runtime.retire(self)
            return
        self._drain()

    def _drain(self):
        try:
            if self.future is not None:
                try:
                    self.future.result()
                except CancelledError:
                    if not self.abandoned:
                        raise
                if not self.consumed and (self.copy_submitted or not self.abandoned):
                    self.slot.copied.synchronize()
        finally:
            self.ids = self.future = None


class EngramCPULookupPrefetch:
    """Queue CPU work immediately; one worker serves layers in model order."""

    def __init__(self, model, windows, image_token):
        self.runtime = getattr(model, "_engram_cpu_runtime", None)
        self.work = []
        self.hashes = None
        if (
            self.runtime is None
            or windows is None
            or windows.is_cuda
            or windows.dtype not in (torch.int32, torch.int64)
            or windows.ndim != 2
            or windows.shape[1] != model._engram_hash_state.layout.max_ngram_size
            or not 0 < windows.shape[0] <= self.runtime.capacity
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        if isinstance(self.runtime, EngramCPULookupRuntime) and not self.runtime.reap():
            return
        timing = (
            {"hash_submitted_ns": time.perf_counter_ns()}
            if any(slot.collect_timing for slot in self.runtime.slots.values())
            else None
        )
        self.hashes = self.runtime.executor.submit(
            _timed_cpu_hash,
            model._engram_hash_state,
            windows,
            image_token,
            getattr(self.runtime, "num_threads", None),
            timing,
        )
        self.hashes.engram_timing = timing
        try:
            for layer_id, slot in self.runtime.slots.items():
                layer = model.v4.layers[layer_id]
                ids = layer.engram_hashes
                if not (
                    ids is not None
                    and ids.is_cuda
                    and ids.device == slot.device
                    and ids.dtype == torch.int64
                    and ids.shape == (windows.shape[0], slot.heads)
                ):
                    continue
                if layer.engram._cpu_lookup_work is not None:
                    raise RuntimeError("Engram CPU lookup is already pending")
                work = _EngramCPULookupWork(
                    slot,
                    layer.engram_hashes,
                    (
                        self.runtime
                        if isinstance(self.runtime, EngramCPULookupRuntime)
                        else None
                    ),
                )
                self.work.append((layer.engram, work))
                work.submit(
                    self.runtime.executor,
                    self.hashes,
                    windows.shape[0],
                    time.perf_counter_ns() if slot.collect_timing else None,
                )
                layer.engram._cpu_lookup_work = work
        except BaseException:
            self.clear()
            raise

    def before_layer(self, index):
        pass

    def clear(self):
        error = None
        for module, work in self.work:
            module._cpu_lookup_work = None
            try:
                work.clear()
            except BaseException as exc:
                error = error or exc
        if self.hashes is not None:
            try:
                if isinstance(self.runtime, EngramCPULookupRuntime):
                    # Hold CPU windows/state until hashing finishes without
                    # blocking the fallback consumer or allowing another batch
                    # to pile work behind this one.
                    if not self.hashes.done():
                        self.runtime.retired.append(self.hashes)
                    else:
                        self.hashes.result()
                else:
                    self.hashes.result()
            except BaseException as exc:
                error = error or exc
        self.hashes = None
        self.work = []
        if error is not None:
            raise error


def _tensor_version(tensor):
    # Production hashes are fresh inference tensors, immutable for this forward.
    # Ordinary tensors additionally let us reject an in-place edit before consume.
    if tensor.is_inference():
        return None
    return tensor._version


class _EngramLookupWork:
    """One forward's lookup; CUDA dependencies never imply a host-side wait."""

    def __init__(self, embedding, ids, rows, done):
        self.embedding = embedding  # Own the custom UVA registration while pending.
        self.ids = ids
        self.signature = (ids.shape, ids.stride(), ids.storage_offset(), ids.dtype)
        self.version = _tensor_version(ids)
        self.rows = rows
        self.done = done

    def consume(self, ids, hidden):
        valid = (
            self.ids is ids
            and self.signature
            == (ids.shape, ids.stride(), ids.storage_offset(), ids.dtype)
            and self.version == _tensor_version(ids)
            and self.rows.shape[0] == hidden.shape[0]
            and self.rows.device == hidden.device
        )
        if not valid:
            self.clear()
            return None
        current = torch.cuda.current_stream(hidden.device)
        current.wait_event(self.done)
        rows = self.rows
        rows.record_stream(current)
        self.rows = self.ids = self.embedding = None
        return rows

    def clear(self):
        if self.rows is not None:
            # Only abandoned/invalid work waits on the CPU. The normal consumer
            # joins on its stream and record_stream covers the output allocator.
            self.done.synchronize()
        self.rows = self.ids = self.embedding = None


class EngramLookupPrefetch:
    """Forward-local, two-layer lookahead with at most one pending lookup.

    Hash tensors must remain immutable for this forward. No lookup result or
    hash is reused by a later request, including requests reusing tensor storage.
    Layers in the CED query domain are conservatively left synchronous.
    """

    def __init__(self, layers):
        self.layers = tuple(
            (index, layer.engram, layer.engram_hashes)
            for index, layer in layers
            if index < 20 and layer.engram is not None
        )
        self.cursor = 0
        self.pending = None

    def before_layer(self, index):
        if self.pending is not None:
            target, module, work = self.pending
            if work.rows is None:
                self.pending = None
            elif target < index:
                module.clear_lookup_prefetch()
                self.pending = None
            else:
                return
        while self.cursor < len(self.layers):
            target, module, ids = self.layers[self.cursor]
            if target > index + 2:
                return
            self.cursor += 1
            if target < index:
                continue
            from rtp_llm.models_py.modules.dsv41 import _profiler

            with _profiler.record_function_range(f"dsv41.engram.prefetch.L{target}"):
                work = module.prefetch_lookup(ids)
            if work is not None:
                self.pending = (target, module, work)
                return

    def clear(self):
        if self.pending is not None:
            _, module, work = self.pending
            # The module can already have consumed the work; clear is idempotent.
            module.clear_lookup_prefetch()
            work.clear()
            self.pending = None
        self.layers = ()


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
        self._lookup_stream = None
        self._lookup_done = None
        self._lookup_work = None
        self._cpu_lookup_work = None

    def prepare_lookup_prefetch(self):
        """Called after lookup JIT warmup, never from a request or capture."""
        if (
            not engram_lookup_prefetch_enabled()
            or not self.q_weight.is_cuda
            or self.embed_tokens._uva is None
            or self._lookup_stream is not None
        ):
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Engram prefetch must be initialized before capture")
        device = self.q_weight.device
        stream = torch.cuda.Stream(device=device, priority=0)
        done = torch.cuda.Event()
        stream.wait_stream(torch.cuda.current_stream(device))
        done.record(stream)
        torch.cuda.current_stream(device).wait_event(done)
        # Register the event once, not once per request. Its latest recording
        # covers all earlier reads on this stream when the table is destroyed.
        self.embed_tokens._pinned._lookup_events.append(done)
        self._lookup_stream, self._lookup_done = stream, done

    def prefetch_lookup(self, hash_ids):
        if (
            self._lookup_stream is None
            or self._lookup_work is not None
            or not hash_ids.is_cuda
            or hash_ids.device != self.q_weight.device
            or hash_ids.dtype != torch.int64
            or hash_ids.ndim != 2
            or hash_ids.shape[1] != self.layout.n_hash_cols
            or not 0 < hash_ids.shape[0] <= 32768
            or not hash_ids.is_contiguous()
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        stream = self._lookup_stream
        stream.wait_stream(torch.cuda.current_stream(hash_ids.device))
        hash_ids.record_stream(stream)
        try:
            with torch.cuda.stream(stream):
                rows = self.embed_tokens(hash_ids, hash_ids.device, prefetch=True)
                self._lookup_done.record(stream)
        except BaseException:
            # Preserve owners until queued reads have finished, then propagate
            # the launch/runtime error instead of silently running a fallback.
            stream.synchronize()
            raise
        work = _EngramLookupWork(self.embed_tokens, hash_ids, rows, self._lookup_done)
        self._lookup_work = work
        return work

    def clear_lookup_prefetch(self):
        work = self._lookup_work
        self._lookup_work = None
        if work is not None:
            work.clear()

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
            pinned = _PinnedSharedTable(table, scales, "|".join(identity), device)
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

    def forward(self, hidden, hash_ids, token_mask=None):
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
                )
            return output
        return self._forward_rows(
            hidden, hash_ids, token_mask, out=hidden if inplace else None
        )

    def _forward_rows(self, hidden, hash_ids, token_mask, *, out=None):
        cpu_work = self._cpu_lookup_work
        if cpu_work is not None:
            self._cpu_lookup_work = None
            rows = cpu_work.consume(hash_ids, hidden)
            if rows is not None:
                if rows.dtype == torch.uint8:
                    from rtp_llm.models_py.modules.dsv41._engram_packed_triton import (
                        unpack_packed_rows,
                    )

                    pair = unpack_packed_rows(
                        rows, self.layout.n_hash_cols, self.layout.head_dim
                    )
                    if pair is None:
                        del rows
                        return self._forward_rows(hidden, hash_ids, token_mask, out=out)
                    kv = self.wkv.forward_quantized(*pair)
                    del pair
                else:
                    kv = self.wkv(rows)
                del rows
                return gated_engram_residual(
                    hidden,
                    kv,
                    self.q_weight,
                    self.k_weight,
                    self.eps,
                    token_mask,
                    out=out,
                )
        work = self._lookup_work
        rows = None
        if work is not None:
            self._lookup_work = None
            rows = work.consume(hash_ids, hidden)
        if rows is None:
            rows = self.embed_tokens(hash_ids, hidden.device)
        rows = rows.flatten(-2)
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
