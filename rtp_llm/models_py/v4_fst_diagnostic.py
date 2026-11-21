"""Private opt-in markers for the full V4 xgrammar FST smoke.

Fences are device-local, not collective. A fence observes *all* work submitted
on this device so overlapping producer/consumer batches are candidates, not
proof of the identity of a failing kernel.
"""

import ctypes
import hashlib
import logging
import os
import re
import time
import weakref
from contextlib import contextmanager
from pathlib import Path

import torch

log = logging.getLogger(__name__)


class V4RouterRangeError(ValueError):
    """This table failed the existing model gate ABI validation, not a retry."""

    def __init__(self, layer, num_experts, observed, lo, hi):
        self.layer, self.num_experts = layer, num_experts
        self.observed, self.lo, self.hi = observed, lo, hi
        super().__init__(
            f"gate.tid2eid contains an out-of-range expert index: layer={layer} "
            f"dtype={observed.dtype} shape={tuple(observed.shape)} "
            f"observed=[{lo},{hi}] valid=[0,{num_experts})"
        )


def _pci(ordinal):
    # CUDA ordinals respect the prefill role's CUDA_VISIBLE_DEVICES remapping.
    # Do not use the NVML index as if it were this process's CUDA ordinal.
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        dev = ctypes.c_int()
        bus = ctypes.create_string_buffer(32)
        if driver.cuDeviceGet(ctypes.byref(dev), ordinal) == 0:
            if driver.cuDeviceGetPCIBusId(bus, len(bus), dev) == 0:
                return bus.value.decode("ascii")
    except (AttributeError, OSError, UnicodeError):
        pass
    return "unknown"


def _namespace_pids():
    # If procfs is container-local Linux need not expose the host PID. Never
    # label an inner namespace PID as the host PID.
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("NSpid:"):
                return (
                    "/".join(v for v in line.split()[1:] if v.isdecimal()) or "unknown"
                )
    except OSError:
        pass
    return "unknown"


class V4FstTrace:
    _wo_a_key = "layers.0.attn.wo_a.scale"

    def bind_wo_a_route(self, model, catalog, selected_files):
        """Fenced smoke only: independent CPU oracle for the one failing leaf."""
        from safetensors import safe_open

        selected = {os.path.realpath(p) for p in selected_files}
        paths = [
            p for k, p in catalog.entries
            if k == self._wo_a_key and os.path.realpath(p) in selected
        ]
        modules = [
            m for m in model.modules()
            if type(m).__name__ == "AttentionFP8" and m._checkpoint_owned
            and m.layer_id == 0
        ]
        if len(paths) != 1 or len(modules) != 1:
            raise RuntimeError("V4FST ambiguous layer 0 wo_a scale route")
        module = modules[0]
        owner = module.wo_a_s
        if (
            owner.ndim != 2
            or owner.dtype != torch.float8_e8m0fnu
            or not 0 < owner.numel() * module.tp_size <= 1024**2
            or not 0 <= module.tp_rank < module.tp_size
        ):
            raise RuntimeError("V4FST invalid layer 0 wo_a scale geometry")
        with safe_open(paths[0], framework="pt", device="cpu") as f:
            field = f.get_slice(self._wo_a_key)
            rows, cols = owner.shape
            if field.get_dtype() != "F8_E8M0" or tuple(field.get_shape()) != (
                rows * module.tp_size, cols
            ):
                raise RuntimeError("V4FST invalid layer 0 wo_a source geometry")
            self._wo_a_full = field[:].view(torch.uint8).clone()
        start = module.tp_rank * rows
        self._wo_a_local = self._wo_a_full[start : start + rows]
        self._wo_a_module = weakref.ref(module)

    def _wo_a_sample(self, stage, tensor, *, full=False):
        reference = self._wo_a_full if full else self._wo_a_local
        if tensor.dtype != torch.float8_e8m0fnu or tensor.shape != reference.shape:
            log.error(
                "V4FST wo_a rank=%s stage=%s geometry_mismatch dtype=%s shape=%s",
                self.rank, stage, tensor.dtype, tuple(tensor.shape),
            )
            return
        bits = tensor.detach().view(torch.uint8).cpu().reshape(-1)
        different = (bits != reference.reshape(-1)).nonzero().flatten()
        # Do not abort here: the file-owner rank inspects its source BEFORE
        # broadcast, so a local raise would strand other ranks in NCCL.
        log.info(
            "V4FST wo_a rank=%s stage=%s dtype=%s bytes=%d mismatches=%d first_offset=%d",
            self.rank, stage, tensor.dtype, bits.numel(), different.numel(),
            int(different[0]) if different.numel() else -1,
        )

    def observe_wo_a_owner(self, stage):
        self._wo_a_sample(stage, self._wo_a_module().wo_a_s)

    def observe_wo_a_reader(self, file_buffer, key, spec, scatter):
        # Same actual operations as the normal reader: never issue a second
        # FST read (it may consume/release the file). Source only exists on
        # the owner; non-owner ranks verify the actual collective result.
        owner_rank, idx = file_buffer.key_to_rank_lidx[key]
        if file_buffer.pg.rank() == owner_rank:
            native = file_buffer.rank_loaders[owner_rank][idx].tensors[key]
            self._wo_a_sample("native_file_buffer", native.get_raw(), full=True)
        if scatter:
            raw = file_buffer.get_sharded(key, spec.shard_dim)
            self._wo_a_sample("get_sharded", raw)
            return spec.wrap_sharded_tensor(raw)
        raw = file_buffer.get_tensor(key)
        self._wo_a_sample("get_tensor", raw, full=True)
        return spec.materialize(raw)

    def wo_a_dispatch(self, weights):
        from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor

        for key, value in weights:
            if key == self._wo_a_key:
                self._wo_a_sample("after_readspec", unwrap_checkpoint_tensor(value)[0])
                yield key, value
                # The model consumed this item before asking for the next one.
                self.observe_wo_a_owner("after_dispatch")
            else:
                yield key, value

    def verify_wo_a_owners(self, model, catalog, selected_files):
        """Check the actual FST destinations before packing; never repair from CPU."""
        from safetensors import safe_open

        selected = {os.path.realpath(p) for p in selected_files}
        sources = {}
        for key, path in catalog.entries:
            if re.fullmatch(r"layers\.\d+\.attn\.wo_a\.scale", key):
                path = os.path.realpath(path)
                if path in selected:
                    sources.setdefault(key, []).append(path)
        checked = 0
        for module in model.modules():
            if type(module).__name__ != "AttentionFP8" or not module._checkpoint_owned:
                continue
            layer = module.layer_id
            paths = sources.get(f"layers.{layer}.attn.wo_a.scale", ())
            owner = module.wo_a_s.detach()
            if (
                len(paths) != 1
                or owner.ndim != 2
                or owner.dtype != torch.float8_e8m0fnu
                or not 0 < owner.numel() <= 1024**2
            ):
                raise RuntimeError(f"V4FST invalid wo_a audit input layer={layer}")
            rows, cols = owner.shape
            tp, rank = module.tp_size, module.tp_rank
            if not 0 <= rank < tp:
                raise RuntimeError(f"V4FST invalid wo_a owner rank layer={layer}")
            # All checkpoint names here come from the same selected FST
            # manifest. Read just this owner's rows, not an entire shard.
            try:
                with safe_open(paths[0], framework="pt", device="cpu") as f:
                    field = f.get_slice(f"layers.{layer}.attn.wo_a.scale")
                    if field.get_dtype() != "F8_E8M0" or tuple(field.get_shape()) != (
                        tp * rows,
                        cols,
                    ):
                        raise ValueError("incompatible source")
                    source = field[rank * rows : (rank + 1) * rows]
            except Exception as exc:
                raise RuntimeError(
                    f"V4FST wo_a reference unavailable layer={layer} reason={type(exc).__name__}"
                ) from None
            if layer == 0 and hasattr(self, "_wo_a_local"):
                self.observe_wo_a_owner("before_pack")
            if not torch.equal(owner.view(torch.uint8).cpu(), source.view(torch.uint8)):
                raise RuntimeError(
                    f"V4FST wo_a source/owner mismatch rank={self.rank} layer={layer}"
                )
            checked += 1
        if not checked:
            raise RuntimeError("V4FST did not verify any wo_a scale owner")
        log.info(
            "V4FST wo_a owner/source bytes match rank=%s layers=%d", self.rank, checked
        )

    def compare_bad_router(self, error, catalog_fn, selected_files):
        """Compare at most 8 MiB on the failed worker; never use it for loading."""
        if not isinstance(error, V4RouterRangeError):
            return
        try:
            observed = error.observed
            if (
                type(error.layer) is not int
                or not 0 <= error.layer < 128
                or observed.dim() != 2
                or observed.numel() == 0
                or observed.numel() * observed.element_size() > 8 * 1024**2
                or observed.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("unsupported table")
            # A validated weight index from this loader, not a path fabricated
            # from a tensor name, message, environment variable or exception.
            name = f"layers.{error.layer}.ffn.gate.tid2eid"
            files = {os.path.realpath(p) for p in selected_files}
            sources = [
                p
                for k, p in catalog_fn().entries
                if k == name and os.path.realpath(p) in files
            ]
            if len(sources) != 1:
                raise ValueError("ambiguous source")
            from safetensors import safe_open

            with safe_open(sources[0], framework="pt", device="cpu") as reader:
                metadata = reader.get_slice(name)
                expected = "I32" if observed.dtype == torch.int32 else "I64"
                if (
                    tuple(metadata.get_shape()) != tuple(observed.shape)
                    or metadata.get_dtype() != expected
                ):
                    raise ValueError("source metadata mismatch")
                reference = reader.get_tensor(name)
                actual = observed.detach().to("cpu").contiguous()

                def digest(value):
                    return hashlib.sha256(
                        memoryview(value.view(torch.uint8).numpy())
                    ).hexdigest()

                log.error(
                    "V4FST failed_router rank=%s layer=%d dtype=%s shape=%s "
                    "observed_range=[%d,%d] source_range=[%d,%d] "
                    "equal=%s observed_sha256=%s source_sha256=%s",
                    self.rank,
                    error.layer,
                    observed.dtype,
                    tuple(observed.shape),
                    error.lo,
                    error.hi,
                    reference.min().item(),
                    reference.max().item(),
                    torch.equal(actual, reference),
                    digest(actual),
                    digest(reference),
                )
        except Exception as diagnostic_error:
            # Even a poisoned CUDA context or inaccessible file must preserve
            # the *original* validation rejection. Exception strings can leak paths.
            log.error(
                "V4FST failed_router unavailable rank=%s layer=%s reason=%s",
                self.rank,
                error.layer,
                type(diagnostic_error).__name__,
            )

    def __init__(self, mode, rank, device):
        self.mode, self.rank, self.device = mode, rank, device
        self._batches = {}

    @classmethod
    def from_env(cls, device):
        mode = os.environ.get("RTP_V4_XGRAMMAR_FST_DIAG")
        if mode not in ("trace", "fenced"):
            raise ValueError("Expected trace or fenced for V4 FST diagnostics")
        device = torch.device(device)
        ordinal = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        device = torch.device("cuda", ordinal)
        trace = cls(mode, torch.distributed.get_rank(), device)
        log.info(
            "V4FST mode=%s role=prefill rank=%s pid=%d nspid=%s cuda=%d pci=%s monotonic=%.6f",
            mode,
            trace.rank,
            os.getpid(),
            _namespace_pids(),
            ordinal,
            _pci(ordinal),
            time.monotonic(),
        )
        return trace

    def point(self, stage, batch=None, *, layer=None, fence=True):
        # On long successful startups, only report the first attention layer
        # and an actual failed fence, not per-layer success hundreds of times.
        emit = layer is None or layer == 0
        fields = (
            self.mode,
            self.rank,
            stage,
            batch,
            self._batches.get(batch, "-"),
            layer,
        )
        fmt = "V4FST mode=%s rank=%s stage=%s batch=%s shards=%s layer=%s"
        if emit:
            log.info(fmt + " marker=host", *fields)
        if self.mode == "fenced" and fence:
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                log.error(fmt + " marker=fence_failed", *fields)
                raise
            if emit:
                log.info(fmt + " marker=fence_ok", *fields)

    @contextmanager
    def watch_keys(self, batch, keys):
        """Identify native-read/consumer progress without recording checkpoint keys."""
        ordinal = None

        def indexed():
            nonlocal ordinal
            report_after = 0.0
            for ordinal, key in enumerate(keys):
                now = time.monotonic()
                if now >= report_after:
                    log.info(
                        "V4FST consumer_progress rank=%s batch=%s key_ordinal=%s",
                        self.rank, batch, ordinal,
                    )
                    report_after = now + 0.5
                yield key

        try:
            yield indexed()
        except (Exception, GeneratorExit) as exc:
            # GeneratorExit also occurs when external module dispatch fails (or
            # stops intentionally). This ordinal is a location, not a verdict.
            log.warning(
                "V4FST consumer_stopped rank=%s batch=%s key_ordinal=%s reason=%s",
                self.rank, batch, ordinal, type(exc).__name__,
            )
            raise

    def reader_class(self, base, files):
        index = {os.fspath(path): idx for idx, path in enumerate(sorted(files), 1)}
        trace = self

        def shard_id(path):
            # Emit only a numerical checkpoint ordinal, never paths, tensor
            # names, metadata, or the contents of the checkpoint.
            m = re.search(r"-(\d+)-of-(\d+)\.safetensors$", os.path.basename(path))
            return (
                f"{int(m[1])}/{int(m[2])}"
                if m
                else f"selected:{index.get(os.fspath(path), 0)}"
            )

        class TracedReader(base):
            _rtp_v4_fst_trace = trace

            def _load_single_batch(self, batch_id, file_list):
                trace._batches[batch_id] = ",".join(shard_id(p) for p in file_list)
                # Producer thread: no device sync, no touching its CUDA context
                # or changing when the native producer may enqueue the batch.
                trace.point("producer_start", batch_id, fence=False)
                try:
                    result = super()._load_single_batch(batch_id, file_list)
                except Exception:
                    trace.point("producer_raised", batch_id, fence=False)
                    raise
                # Some FST versions hand off the exception through the batch
                # queue instead of raising it on this thread. Do not confuse
                # that normal Python return with successful native I/O.
                error_event = getattr(self, "error_event", None)
                stage = (
                    "producer_returned_error_event"
                    if error_event is not None and error_event.is_set()
                    else "producer_returned"
                )
                trace.point(stage, batch_id, fence=False)
                return result

        return TracedReader
