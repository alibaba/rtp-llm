"""Bounded S5 installed-surface and shifted BF16 Mega-mHC diagnostics.

Run through the two manual Bazel sh_tests and the qualified published wrapper.
The surface mode intentionally imports no RTP package: models_py/__init__.py
calls get_device_type(), which queries CUDA availability even for metadata IO.
Surface collection makes no explicit device queries except safe is_initialized.
An imported third-party package can still have side effects; report them.

Source contracts, not runtime claims:
* DeepGEMM-for-sail 07674e672082c2e06ea9f1ec1993d79684444a1c,
  csrc/apis/mega_mhc.hpp and tests/test_mega_mhc.py (MIT, T-Head/DeepSeek).
* RTP hc/delayed.py, hc/fallback_impl.py and ppu/prefill.py define the
  independent equations below. No production helper or kernel is used as oracle.
* SGLang-for-sail b313eda591b53f263c811ac001239855506f64ba,
  mhc_mega.py / deepseek_v4_mhc.py (Apache-2.0) identify the shifted boundary;
  no SGLang implementation is imported or copied here.

The first diagnostic freezes exact equality for all five outputs before launch.
TF32 projection/FMA/M-dependent splitting can fail that IEEE FP32 comparator:
record FAIL and the discrepancy, never select a tolerance from observed output.
The upstream calc_diff metric is reported separately, never used to pass.
No production dispatch, attention call, Graph, timing, or promotion is performed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import stat
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

SCHEMA = "s5-installed-kernel-probe-v1"
EXPECTED_DEEPGEMM_SHA256 = (
    "816ff22272d7636c8ee57b960189ada869910af74016f35d53638248fcdc3942"
)
DEEPGEMM_SOURCE = "07674e672082c2e06ea9f1ec1993d79684444a1c"
FLASHMLA_SOURCE = "4305d1bdaeb8451eb5d78c67a5122ecf38f5014d"
BASE_ROWS = (1, 3, 4, 5, 6, 64, 65, 129, 1025, 8192, 16384)
HC, HIDDEN, MIX = 4, 5120, 24
CHUNK_ROWS = 64
GUARD_ELEMENTS = 128
MAX_WORKING_BYTES = 8 * 2**30
MAX_EVIDENCE_ROWS = 16
SEED = 20261106
FIELD_SEEDS = {
    "x": 101,
    "residual": 211,
    "shifted_prev_mix": 307,
    "post_mix": 401,
    "comb_res_mix": 503,
    "fn": 601,
    "mix_scales": 701,
    "mix_bases": 809,
    "rmsnorm_weight": 907,
}
OUTPUTS = (
    "new_residual",
    "new_prev_mix",
    "new_post_mix",
    "new_comb_res_mix",
    "y_bf16",
)
_QUALIFICATION_LIMIT = 5e-5
_QUALIFICATION_PROFILE = "FIXED_UPSTREAM_CALC_DIFF_EACH_ROW_CROSS_M_04"
_QUALIFICATION_CONTRACT_SHA256 = (
    "a6bf5ed6a937180e7df536ed03660f1116beb3298a2e07c65f4afd55a4339215"
)
_QUALIFICATION_CASES = 26
_QUALIFICATION_CALLS = 28
_QUALIFICATION_CROSS_PAIRS = 22
ROW_INPUTS = (
    "x",
    "residual",
    "shifted_prev_mix",
    "post_mix",
    "comb_res_mix",
)
SURFACE_MODULES = (
    "torch",
    "deep_gemm",
    "deep_gemm.deep_gemm_cpp",
    "flash_mla",
    "flash_mla.flash_mla_interface",
    "flash_mla_cuda",
)
SURFACE_SYMBOLS = {
    "deep_gemm": ("mega_mhc", "get_num_sms"),
    "flash_mla": (
        "flash_mla_sparse_fwd",
        "flash_mla_with_kvcache",
        "get_mla_metadata",
        "FlashMLASchedMeta",
        "flash_mla_sparse_quantized_fwd",
        "flash_mla_sparse_quantized_decode_fwd",
        "QuantizedAttentionConfig",
    ),
    "flash_mla.flash_mla_interface": (
        "flash_mla_sparse_fwd",
        "flash_mla_with_kvcache",
        "get_mla_metadata",
        "FlashMLASchedMeta",
        "flash_mla_sparse_quantized_fwd",
        "flash_mla_sparse_quantized_decode_fwd",
        "QuantizedAttentionConfig",
    ),
}


def _error(error):
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as reader:
        for block in iter(lambda: reader.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path):
    if not path:
        return {"status": "UNAVAILABLE", "path": None}
    path = Path(path)
    try:
        info = path.stat()
        return {
            "status": "RECORDED",
            "path": str(path),
            "resolved_path": str(path.resolve()),
            "sha256": _sha256(path),
            "bytes": info.st_size,
            "mode": oct(stat.S_IMODE(info.st_mode)),
        }
    except (OSError, ValueError) as error:
        return {"status": "UNAVAILABLE", "path": str(path), "error": _error(error)}


def _write_json(root, name, value):
    # All probe artifacts, including intermediate receipts, stay in Bazel output.
    if Path(name).name != name:
        raise ValueError("artifact name must not escape the Bazel output directory")
    path = root / name
    temporary = root / (name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)
    return {"path": name, "sha256": _sha256(path), "bytes": path.stat().st_size}


def _snapshot_identity():
    root_text = os.environ.get("RTP_V41_TEST_SOURCE_ROOT")
    if not root_text:
        raise ValueError("RTP_V41_TEST_SOURCE_ROOT is required")
    root = Path(root_text).resolve()
    path = root / ".v41_snapshot.json"
    manifest = json.loads(path.read_text())
    entries = manifest["files"]
    source_sha256 = hashlib.sha256(
        json.dumps(entries, sort_keys=True).encode()
    ).hexdigest()
    if manifest.get("mode") != "snapshot" or source_sha256 != manifest["source_sha256"]:
        raise ValueError("invalid committed source snapshot manifest")
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("snapshot entry escapes source root")
        source = root / relative
        actual = (
            os.readlink(source) if entry["kind"] == "symlink" else _sha256(source)
        )
        if actual != entry["value"]:
            raise ValueError(f"published source changed: {entry['path']}")
    return {
        "source_root": str(root),
        "snapshot_id": manifest["snapshot_id"],
        "source_sha256": source_sha256,
        "manifest": _file_record(path),
        "verified_entries": len(entries),
        "verification": "stdlib file/symlink hashes, no RTP package import",
        "manifest_repository_identity": {
            key: value
            for key, value in manifest.items()
            if key not in ("files", "source_sha256")
        },
    }


def _initialization_state():
    torch = sys.modules.get("torch")
    if torch is None:
        return {"status": "NOT_IMPORTED", "initialized": None}
    cuda = vars(torch).get("cuda")
    check = vars(cuda).get("is_initialized") if cuda is not None else None
    if not callable(check):
        return {"status": "UNKNOWN", "initialized": None}
    try:
        return {"status": "RECORDED", "initialized": bool(check())}
    except Exception as error:
        return {"status": "UNKNOWN", "initialized": None, "error": _error(error)}


def _distribution_record(name):
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"name": name, "status": "UNAVAILABLE"}
    files = distribution.files or ()
    metadata_files = {}
    for entry in files:
        if entry.name in ("METADATA", "RECORD", "direct_url.json", "WHEEL"):
            metadata_files[str(entry)] = _file_record(distribution.locate_file(entry))
    return {
        "name": name,
        "status": "RECORDED",
        "version": distribution.version,
        "root": str(distribution.locate_file("")),
        "metadata_files": metadata_files,
        "meaning": (
            "distribution lookup does not prove the selected module or extension"
        ),
    }


def _symbol_record(module, name):
    value = vars(module).get(name)
    if value is None:
        return {
            "status": (
                "UNKNOWN_LAZY_EXPORT"
                if callable(vars(module).get("__getattr__"))
                else "UNAVAILABLE"
            ),
            "callable": False,
        }
    result = {
        "status": "AVAILABLE",
        "callable": callable(value),
        "declaring_module": getattr(value, "__module__", None),
        "qualname": getattr(value, "__qualname__", None),
        "doc": inspect.getdoc(value),
    }
    try:
        result["signature"] = str(inspect.signature(value))
    except (TypeError, ValueError) as error:
        result["signature"] = None
        result["signature_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    try:
        source = inspect.getsourcefile(value)
        result["python_source"] = _file_record(source) if source else None
    except (TypeError, OSError) as error:
        result["source_error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def _loaded_elf_records():
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return {"status": "UNKNOWN", "reason": "/proc/self/maps unavailable"}
    paths = set()
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and "x" in fields[1] and fields[5].startswith("/"):
            paths.add(fields[5])
    records = []
    for path in sorted(paths):
        try:
            with open(path, "rb") as reader:
                is_elf = reader.read(4) == b"\x7fELF"
            if is_elf:
                records.append(_file_record(path))
        except OSError as error:
            records.append({"path": path, "status": "UNKNOWN", "error": _error(error)})
    return {
        "status": "RECORDED",
        "executable_mapping_count": len(paths),
        "files": records,
    }


@contextmanager
def _surface_import_guard(torch, attempts):
    """Block Python query/init/allocation paths during third-party import only.

    A compiled module initializer can bypass Python; its native side effects
    remain UNKNOWN even when is_initialized stays false. No installed file is
    changed, and every successfully replaced attribute is restored in finally.
    """
    restored = []

    def reject(qualified_name):
        def blocked(*args, **kwargs):
            attempts.append({"api": qualified_name, "original_called": False})
            raise RuntimeError(
                f"surface import blocked forbidden API: {qualified_name}"
            )

        return blocked

    cuda = vars(torch).get("cuda")
    native = vars(torch).get("_C")
    factories = (
        "empty",
        "empty_like",
        "empty_strided",
        "zeros",
        "zeros_like",
        "ones",
        "ones_like",
        "full",
        "full_like",
        "tensor",
        "as_tensor",
        "asarray",
        "rand",
        "rand_like",
        "randn",
        "randn_like",
        "randint",
        "randperm",
        "arange",
        "range",
        "linspace",
        "logspace",
        "eye",
        "from_numpy",
    )
    cuda_names = (
        "init",
        "_lazy_init",
        "is_available",
        "device_count",
        "current_device",
        "get_device_properties",
        "get_device_name",
        "get_device_capability",
        "set_device",
        "synchronize",
        "current_stream",
        "default_stream",
        "set_stream",
        "Stream",
        "ExternalStream",
        "Event",
        "CUDAGraph",
        "device",
        "device_of",
        "stream",
        "graph",
        "caching_allocator_alloc",
        "memory_allocated",
        "memory_reserved",
        "mem_get_info",
        "empty_cache",
    )
    tensor_names = (
        "cuda",
        "to",
        "new",
        "new_empty",
        "new_zeros",
        "new_ones",
        "new_full",
        "new_tensor",
    )
    targets = [(torch, name, "torch." + name) for name in factories]
    if cuda is not None:
        targets.extend((cuda, name, "torch.cuda." + name) for name in cuda_names)
        targets.extend(
            (cuda, name, "torch.cuda." + name)
            for name in vars(cuda)
            if name.endswith("Tensor")
        )
    tensor_type = vars(torch).get("Tensor")
    if tensor_type is not None:
        targets.extend(
            (tensor_type, name, "torch.Tensor." + name) for name in tensor_names
        )
    if native is not None:
        targets.extend(
            (native, name, "torch._C." + name)
            for name in vars(native)
            if name.startswith(("_cuda_", "_accelerator_", "_ppu_", "_hggc_"))
            and name != "_cuda_isInBadFork"
        )
    try:
        for owner, name, qualified in targets:
            original = getattr(owner, name, None)
            if callable(original):
                setattr(owner, name, reject(qualified))
                restored.append((owner, name, original))
        yield
    finally:
        for owner, name, original in reversed(restored):
            setattr(owner, name, original)


def collect_surface(guard_imports=True, module_names=SURFACE_MODULES):
    """Collect imports and signatures only; never query or allocate a device."""
    result = {
        "collection_status": "COLLECTING",
        "surface_status": "UNKNOWN",
        "correctness": "NOT_RUN",
        "performance": "NOT_RUN",
        "attention_invocations": 0,
        "python_import_guards_enabled": guard_imports,
        "collection_policy": (
            "GPU0 guarded signature/provenance only"
            if guard_imports
            else (
                "GPU1 fresh normal qualified imports; "
                "no FlashMLA import or invocation requested"
            )
        ),
        "python": {
            "executable": _file_record(sys.executable),
            "version": sys.version,
            "sys_path": list(sys.path),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
        "initialization_before": _initialization_state(),
        "modules": {},
        "distributions": {},
        "surface_policy": (
            "GPU0: no allocation/init/properties/device_count/is_available calls"
            if guard_imports
            else (
                "GPU1: normal qualified imports; "
                "device compatibility checked separately"
            )
        ),
        "wrapper_routing": (
            "surface early branch precedes runtime_identity; models_py import would "
            "call device_type.get_device_type -> torch.cuda.is_available"
        ),
    }
    for name in module_names:
        before = _initialization_state()
        attempts = []
        try:
            if name == "torch":
                module = importlib.import_module(name)
            elif guard_imports:
                torch = sys.modules.get("torch")
                if torch is None:
                    raise RuntimeError(
                        "cannot guard third-party imports without imported torch"
                    )
                with _surface_import_guard(torch, attempts):
                    module = importlib.import_module(name)
            else:
                # Only GPU_COUNT=1 uses unrestricted native imports, under the
                # standard lock. They are not a GPU0 collection fallback.
                module = importlib.import_module(name)
        except Exception as error:
            entry = {
                "status": "UNKNOWN_IMPORT_BLOCKED" if attempts else "UNAVAILABLE",
                "import_error": _error(error),
            }
        else:
            entry = {
                "status": "IMPORTED",
                "file": _file_record(vars(module).get("__file__")),
                "symbols": {
                    symbol: _symbol_record(module, symbol)
                    for symbol in SURFACE_SYMBOLS.get(name, ())
                },
            }
        entry["initialization_before"] = before
        entry["initialization_after"] = _initialization_state()
        entry["blocked_import_attempts"] = attempts
        result["modules"][name] = entry
    for name in ("torch", "triton", "deep-gemm", "deep-gemm-mega", "flash-mla"):
        result["distributions"][name] = _distribution_record(name)
    selected = result["modules"].get("deep_gemm.deep_gemm_cpp", {}).get("file", {})
    result["selected_deepgemm_extension"] = selected
    result["expected_deepgemm_extension_sha256"] = EXPECTED_DEEPGEMM_SHA256
    result["selected_extension_matches_qualified_baseline"] = (
        selected.get("sha256") == EXPECTED_DEEPGEMM_SHA256
    )
    # Capture only actually imported related Python/extension files, not guesses
    # from another wheel or a scan that imports previously unloaded modules.
    related = {}
    for name, module in list(sys.modules.items()):
        if module is not None and name.startswith(("deep_gemm", "flash_mla")):
            path = vars(module).get("__file__")
            if path:
                related[name] = _file_record(path)
    result["loaded_related_module_files"] = related
    result["loaded_elf"] = _loaded_elf_records()
    result["initialization_after"] = _initialization_state()
    initialized = any(
        entry["initialization_after"].get("initialized") is True
        for entry in result["modules"].values()
    )
    unknown = any(
        entry["initialization_after"]["status"] != "RECORDED"
        for entry in result["modules"].values()
    )
    blocked = any(
        entry["blocked_import_attempts"] for entry in result["modules"].values()
    )
    missing = any(
        entry["status"] == "UNAVAILABLE" for entry in result["modules"].values()
    )
    result["surface_policy_violated"] = guard_imports and initialized
    result["native_import_side_effects"] = (
        "UNKNOWN: Python guards cannot observe/prevent direct compiled initializer "
        "driver calls or native allocations; is_initialized is not a no-driver proof"
    )
    if not guard_imports:
        result["surface_status"] = "GPU1_NORMAL_IMPORT_PROVENANCE_ONLY"
    elif initialized:
        result["surface_status"] = "UNKNOWN_IMPORT_INITIALIZED_DEVICE"
    elif unknown:
        result["surface_status"] = "UNKNOWN_INITIALIZATION_OBSERVATION"
    elif blocked:
        result["surface_status"] = "UNKNOWN_IMPORT_POLICY_BLOCKED"
    elif missing:
        result["surface_status"] = "UNAVAILABLE"
    else:
        result["surface_status"] = "UNKNOWN_NATIVE_IMPORT_SIDE_EFFECTS"
    result["collection_status"] = "COLLECTED"
    return result


def _tensor_bytes(value):
    return value.numel() * value.element_size()


def _tensor_sha256(torch, value):
    digest = hashlib.sha256()
    flat = value.detach().reshape(-1)
    elements = max(1, 2**20 // value.element_size())
    for start in range(0, flat.numel(), elements):
        raw = flat[start : start + elements].contiguous().view(torch.uint8).cpu()
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def _tensor_record(value):
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "stride": list(value.stride()),
        "contiguous": value.is_contiguous(),
        "data_ptr": value.data_ptr(),
        "bytes": _tensor_bytes(value),
    }


def _guarded(torch, shape, dtype, device, fill=None):
    elements = math.prod(shape)
    storage = torch.empty(elements + 2 * GUARD_ELEMENTS, dtype=dtype, device=device)
    storage[:GUARD_ELEMENTS].fill_(123.5)
    storage[-GUARD_ELEMENTS:].fill_(-87.0)
    payload = storage[GUARD_ELEMENTS : GUARD_ELEMENTS + elements].view(shape)
    if fill is not None:
        payload.fill_(fill)
    return {"storage": storage, "tensor": payload}


def _canaries(torch, buffers):
    result = {}
    for name, buffer in buffers.items():
        storage = buffer["storage"]
        result[name] = {
            "prefix": bool((storage[:GUARD_ELEMENTS] == 123.5).all().item()),
            "suffix": bool((storage[-GUARD_ELEMENTS:] == -87.0).all().item()),
        }
    return result


def _finite(torch, value):
    flat = value.reshape(-1)
    for start in range(0, flat.numel(), 2**20):
        if not bool(torch.isfinite(flat[start : start + 2**20]).all().item()):
            return False
    return True


def _check_nonoverlap(inputs, outputs):
    def byte_range(value):
        start = value.data_ptr()
        return start, start + _tensor_bytes(value)

    named = list(inputs.items()) + list(outputs.items())
    for index, (name, value) in enumerate(named):
        begin, end = byte_range(value)
        for other_name, other_value in named[:index]:
            other_begin, other_end = byte_range(other_value)
            if begin < other_end and other_begin < end:
                raise ValueError(f"overlapping tensor payloads: {name}, {other_name}")


def _contracts(torch, rows):
    return {
        "x": ((rows, HIDDEN), torch.bfloat16),
        "residual": ((rows, HC, HIDDEN), torch.bfloat16),
        "shifted_prev_mix": ((rows, HC, 1), torch.float32),
        "post_mix": ((rows, HC, 1), torch.float32),
        "comb_res_mix": ((rows, HC, HC), torch.float32),
        "fn": ((MIX, HC * HIDDEN), torch.float32),
        "mix_scales": ((3,), torch.float32),
        "mix_bases": ((MIX,), torch.float32),
        "rmsnorm_weight": ((HIDDEN,), torch.bfloat16),
        "new_residual": ((rows, HC, HIDDEN), torch.bfloat16),
        "new_prev_mix": ((rows, HC, 1), torch.float32),
        "new_post_mix": ((rows, HC, 1), torch.float32),
        "new_comb_res_mix": ((rows, HC, HC), torch.float32),
        "y_bf16": ((rows, HIDDEN), torch.bfloat16),
    }


def _preflight(torch, inputs, outputs, scalars, device):
    rows = inputs["x"].shape[0]
    if not 0 < rows <= max(BASE_ROWS):
        raise ValueError("row count exceeds the bounded diagnostic contract")
    for name, (shape, dtype) in _contracts(torch, rows).items():
        value = (outputs if name in OUTPUTS else inputs)[name]
        if (
            tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device != device
            or not value.is_contiguous()
            or value.data_ptr() % 16
        ):
            raise ValueError(f"invalid shape/dtype/device/stride/alignment: {name}")
    if scalars["hc_mult"] != HC or scalars["num_sinkhorn_iters"] != 20:
        raise ValueError("only HC4 and Sinkhorn20 are in this probe")
    if any(not math.isfinite(value) for value in scalars.values()):
        raise ValueError("nonfinite scalar")
    for name in ("hc_norm_eps", "sinkhorn_eps", "rmsnorm_eps"):
        if scalars[name] <= 0:
            raise ValueError(f"nonpositive epsilon: {name}")
    _check_nonoverlap(inputs, outputs)
    for name, value in inputs.items():
        if not _finite(torch, value):
            raise ValueError(f"nonfinite input: {name}")


def _cpu_values(torch, name, shape, generator):
    value = torch.randn(shape, generator=generator, dtype=torch.float32)
    if name in ("x", "residual"):
        return value * 0.1
    if name == "shifted_prev_mix":
        # Deliberately different from a freshly projected near-0.5 pre mix.
        return value.sigmoid() * 0.4 + torch.tensor([0.1, 0.2, 0.3, 0.4]).view(1, HC, 1)
    if name == "post_mix":
        return value.sigmoid() * 2.0
    if name == "comb_res_mix":
        value = value.softmax(dim=-1) + 1e-6
        for _ in range(20):
            value = value / (value.sum(-2, keepdim=True) + 1e-6)
            value = value / (value.sum(-1, keepdim=True) + 1e-6)
        return value
    if name == "fn":
        return value * 0.01
    if name == "mix_scales":
        return torch.tensor([0.7, 0.9, 0.5], dtype=torch.float32)
    if name == "mix_bases":
        return value * 0.1
    if name == "rmsnorm_weight":
        return value * 0.01 + 1.0
    raise ValueError(f"unknown fixture field {name}")


def _make_case(torch, rows, device, variant):
    inputs, outputs, buffers = {}, {}, {}
    for name, (shape, dtype) in _contracts(torch, rows).items():
        buffer = _guarded(
            torch, shape, dtype, device, float("nan") if name in OUTPUTS else None
        )
        buffers[name] = buffer
        value = buffer["tensor"]
        if name in OUTPUTS:
            outputs[name] = value
            continue
        inputs[name] = value
        generator = torch.Generator(device="cpu").manual_seed(SEED + FIELD_SEEDS[name])
        if name in ROW_INPUTS:
            for start in range(0, rows, CHUNK_ROWS):
                count = min(CHUNK_ROWS, rows - start)
                chunk = _cpu_values(torch, name, (count,) + shape[1:], generator)
                value[start : start + count].copy_(chunk.to(dtype=dtype, device=device))
            # Row0 is generated separately with an identical field-specific seed
            # for every M. Weights/scalars are also identical across M.
            row_generator = torch.Generator(device="cpu").manual_seed(
                SEED + FIELD_SEEDS[name] + 10000
            )
            first = _cpu_values(torch, name, (1,) + shape[1:], row_generator)
            value[:1].copy_(first.to(dtype=dtype, device=device))
        else:
            value.copy_(
                _cpu_values(torch, name, shape, generator).to(
                    dtype=dtype, device=device
                )
            )
    duplicates = sorted(
        {index for index in (3, 64, 1024, rows - 1) if 0 < index < rows}
    )
    if variant in ("zero", "tiny"):
        factor = 0.0 if variant == "zero" else 2.0**-30
        for name in ("x", "residual"):
            inputs[name].mul_(factor)
    for name in ROW_INPUTS:
        for index in duplicates:
            inputs[name][index].copy_(inputs[name][0])
    return inputs, outputs, buffers, duplicates


def independent_transition_reference(torch, inputs, scalars):
    """Unfused, chunked IEEE FP32 equations, with both BF16 landing points."""
    rows = inputs["x"].shape[0]
    expected = {
        name: torch.empty(shape, dtype=dtype, device=inputs["x"].device)
        for name, (shape, dtype) in _contracts(torch, rows).items()
        if name in OUTPUTS
    }
    fn = inputs["fn"].float()
    scale = inputs["mix_scales"].float()
    base = inputs["mix_bases"].float()
    weight = inputs["rmsnorm_weight"].float()
    for start in range(0, rows, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, rows)
        residual = inputs["residual"][start:end].float()
        branch = inputs["x"][start:end].float()
        updated = (
            branch.unsqueeze(1) * inputs["post_mix"][start:end].float()
            + torch.einsum(
                "tij,tih->tjh", inputs["comb_res_mix"][start:end].float(), residual
            )
        ).to(torch.bfloat16)
        # Collapse uses the carried pre, never the pre produced below.
        collapsed = (
            updated.float() * inputs["shifted_prev_mix"][start:end].float()
        ).sum(dim=1).to(torch.bfloat16)
        flat = updated.flatten(1).float()
        mixes = (flat @ fn.T) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + scalars["hc_norm_eps"]
        )
        pre = (mixes[:, :HC] * scale[0] + base[:HC]).sigmoid() + scalars["hc_pre_eps"]
        post = (
            mixes[:, HC : 2 * HC] * scale[1] + base[HC : 2 * HC]
        ).sigmoid() * scalars["hc_post_scale"]
        comb = (
            mixes[:, 2 * HC :] * scale[2] + base[2 * HC :]
        ).reshape(-1, HC, HC)
        comb = comb.softmax(dim=-1) + scalars["sinkhorn_eps"]
        comb = comb / (comb.sum(-2, keepdim=True) + scalars["sinkhorn_eps"])
        for _ in range(scalars["num_sinkhorn_iters"] - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + scalars["sinkhorn_eps"])
            comb = comb / (comb.sum(-2, keepdim=True) + scalars["sinkhorn_eps"])
        norm_input = collapsed.float()
        normalized = (
            norm_input
            * torch.rsqrt(
                norm_input.square().mean(-1, keepdim=True) + scalars["rmsnorm_eps"]
            )
            * (weight * scalars["rmsnorm_scale"])
        ).to(torch.bfloat16)
        for name, value in (
            ("new_residual", updated),
            ("new_prev_mix", pre.unsqueeze(-1)),
            ("new_post_mix", post.unsqueeze(-1)),
            ("new_comb_res_mix", comb),
            ("y_bf16", normalized),
        ):
            expected[name][start:end].copy_(value)
    return expected


def _json_number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def _qualification_ratio(numerator, denominator, both_zero):
    valid = (
        numerator is not None
        and denominator is not None
        and math.isfinite(numerator)
        and math.isfinite(denominator)
        and numerator >= 0.0
        and denominator >= 0.0
    )
    score = None
    if valid:
        if denominator == 0.0:
            valid = both_zero is True and numerator == 0.0
            if valid:
                score = 0.0
        else:
            score = numerator / denominator
            valid = math.isfinite(score) and score >= 0.0
    return {
        "valid": valid,
        "qualified_status": (
            "PASS" if valid and score < _QUALIFICATION_LIMIT else "FAIL"
        ),
        "numerator": numerator if valid else None,
        "denominator": denominator if valid else None,
        "score": score if valid else None,
        "both_zero": both_zero,
    }


def _qualification_metric(torch, actual, expected):
    """Stable SSE/energy, bounded CPU FP64 rows and fsum global energies."""
    rows = actual.shape[0] if actual.dim() else 0
    result = {
        "profile": _QUALIFICATION_PROFILE,
        "limit_strictly_less_than": _QUALIFICATION_LIMIT,
        "expected_rows": rows,
        "rows_compared": 0,
        "row_metrics": [],
        "qualified_status": "FAIL",
        "arithmetic": "CPU FP64 chunked row sums; math.fsum row N/D globally",
        "formula": "sum((a-b)^2)/sum(a^2+b^2), no cancellation or clamp",
        "zero_policy": "D=0 valid only for observed all-zero a and b; score0",
        "relative_error_policy": (
            "diagnostic only over nonzero references; zero-reference absolute "
            "error/count separate, no relative floor used for acceptance"
        ),
    }
    if rows <= 0 or not actual.numel() or actual.shape != expected.shape:
        result["invalid_reason"] = "empty or mismatched tensor coverage"
        result["global"] = _qualification_ratio(None, None, False)
        return result
    for start in range(0, rows, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, rows)
        a = actual[start:end].detach().cpu().double().reshape(end - start, -1)
        b = expected[start:end].detach().cpu().double().reshape(end - start, -1)
        finite = (torch.isfinite(a).all(-1) & torch.isfinite(b).all(-1)).tolist()
        absolute = (a - b).abs()
        numerators = (a - b).square().sum(-1).tolist()
        denominators = (a.square() + b.square()).sum(-1).tolist()
        all_zero = ((a == 0) & (b == 0)).all(-1).tolist()
        maximum_abs = absolute.max(-1).values.tolist()
        nonzero_ref = b != 0
        relative_denominator = torch.where(nonzero_ref, b.abs(), torch.ones_like(b))
        relative = torch.where(
            nonzero_ref, absolute / relative_denominator, torch.zeros_like(b)
        )
        maximum_rel = relative.max(-1).values.tolist()
        zero_ref_errors = ((b == 0) & (a != 0)).sum(-1).tolist()
        zero_ref_abs = torch.where(b == 0, absolute, torch.zeros_like(b))
        zero_ref_max = zero_ref_abs.max(-1).values.tolist()
        for offset in range(end - start):
            row = start + offset
            ratio = _qualification_ratio(
                numerators[offset] if finite[offset] else None,
                denominators[offset] if finite[offset] else None,
                bool(all_zero[offset]),
            )
            result["row_metrics"].append({
                "row": row,
                **ratio,
                "finite": bool(finite[offset]),
                "max_absolute_error": (
                    _json_number(maximum_abs[offset]) if finite[offset] else None
                ),
                "max_relative_error_nonzero_reference": (
                    _json_number(maximum_rel[offset]) if finite[offset] else None
                ),
                "zero_reference_nonzero_actual_count": zero_ref_errors[offset],
                "max_absolute_at_zero_reference": (
                    _json_number(zero_ref_max[offset]) if finite[offset] else None
                ),
            })
    metrics = result["row_metrics"]
    result["rows_compared"] = len(metrics)
    valid = len(metrics) == rows and all(item["valid"] for item in metrics)
    result["global"] = _qualification_ratio(
        math.fsum(item["numerator"] for item in metrics) if valid else None,
        math.fsum(item["denominator"] for item in metrics) if valid else None,
        valid and all(item["both_zero"] for item in metrics),
    )
    valid_scores = [item for item in metrics if item["valid"]]
    worst = max(valid_scores, key=lambda item: item["score"]) if valid_scores else None
    result["worst_row"] = None if worst is None else worst["row"]
    result["worst_row_score"] = None if worst is None else worst["score"]
    result["qualified_status"] = (
        "PASS"
        if valid
        and result["global"]["qualified_status"] == "PASS"
        and all(item["qualified_status"] == "PASS" for item in metrics)
        else "FAIL"
    )
    return result


def _qualification_field_complete(metric, expected_rows):
    if not isinstance(metric, dict) or expected_rows <= 0:
        return False
    rows = metric.get("row_metrics")
    return (
        metric.get("profile") == _QUALIFICATION_PROFILE
        and metric.get("expected_rows") == expected_rows
        and metric.get("rows_compared") == expected_rows
        and isinstance(rows, list)
        and len(rows) == expected_rows
        and [item.get("row") for item in rows] == list(range(expected_rows))
        and isinstance(metric.get("global"), dict)
    )


def _qualification_field_pass(metric, expected_rows):
    if not _qualification_field_complete(metric, expected_rows):
        return False
    for item in [metric["global"], *metric["row_metrics"]]:
        checked = _qualification_ratio(
            item.get("numerator"), item.get("denominator"), item.get("both_zero")
        )
        if (
            item.get("valid") is not True
            or checked["qualified_status"] != "PASS"
            or item.get("score") != checked["score"]
            or item.get("qualified_status") != "PASS"
        ):
            return False
    return metric.get("qualified_status") == "PASS"


def _qualification_self_check(torch):
    result = {"profile": _QUALIFICATION_PROFILE, "new_Mega_calls": 0, "checks": []}
    fixtures = (
        ("both_zero", [[0.0]], [[-0.0]], True),
        ("opposite", [[1.0]], [[-1.0]], False),
        ("row_failure_hidden_by_global", [[1000.0], [1.0]], [[1000.0], [-1.0]], False),
        ("NaN", [[float("nan")]], [[0.0]], False),
        ("Inf", [[float("inf")]], [[0.0]], False),
        ("empty", [], [], False),
    )
    for name, left, right, expected_pass in fixtures:
        a = torch.tensor(left, dtype=torch.float32).reshape(-1, 1)
        b = torch.tensor(right, dtype=torch.float32).reshape(-1, 1)
        metric = _qualification_metric(torch, a, b)
        actual_pass = _qualification_field_pass(metric, len(left))
        extra = True
        if name == "both_zero":
            extra = metric["global"]["score"] == 0.0
        elif name == "opposite":
            extra = metric["global"]["score"] == 2.0
        elif name == "row_failure_hidden_by_global":
            extra = (
                metric["global"]["score"] == 4.0 / 2000002.0
                and metric["global"]["qualified_status"] == "PASS"
                and metric["row_metrics"][1]["score"] == 2.0
            )
        result["checks"].append({
            "name": name, "expected_pass": expected_pass,
            "actual_pass": actual_pass, "global": metric["global"],
            "assertion_passed": actual_pass == expected_pass and extra,
        })
    a = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    valid = _qualification_metric(torch, a, a)
    missing = dict(valid, row_metrics=valid["row_metrics"][:1])
    result["checks"].append({
        "name": "missing_row", "expected_rows": 2,
        "assertion_passed": not _qualification_field_pass(missing, 2),
    })
    result["status"] = (
        "PASS" if len(result["checks"]) == 7
        and all(item["assertion_passed"] for item in result["checks"]) else "FAIL"
    )
    return result


def _compare(torch, actual, expected, qualification=False):
    """Exact per-element comparator; bounded temporaries and per-row receipts."""
    rows = actual.shape[0]
    width = actual.numel() // rows
    mismatch_count, byte_count, dot, denominator = 0, 0, 0.0, 0.0
    max_abs, max_rel, nonfinite = 0.0, 0.0, 0
    first, worst_row, per_row = None, 0, []
    for start in range(0, rows, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, rows)
        original_a, original_b = actual[start:end], expected[start:end]
        a = original_a.float().reshape(end - start, width)
        b = original_b.float().reshape(end - start, width)
        unequal = a != b
        counts = unequal.sum(-1)
        count_list = counts.cpu().tolist()
        finite_a, finite_b = torch.isfinite(a), torch.isfinite(b)
        finite = finite_a & finite_b
        # Finite FP32/BF16 values can overflow a FP32 subtraction/division.
        # Evaluate error metrics in FP64 so even catastrophic failures retain
        # a valid JSON receipt rather than throwing in allow_nan=False.
        ad, bd = a.double(), b.double()
        absolute = torch.where(finite, (ad - bd).abs(), 0.0)
        relative = absolute / bd.abs().clamp_min(1e-8)
        relative = torch.where(finite, relative, 0.0)
        abs_list = absolute.max(-1).values.cpu().tolist()
        rel_list = relative.max(-1).values.cpu().tolist()
        bad_a = (~finite_a).sum(-1).cpu().tolist()
        bad_b = (~finite_b).sum(-1).cpu().tolist()
        first_columns = unequal.to(torch.int32).argmax(-1)
        a_first = a.gather(1, first_columns[:, None]).flatten().cpu().tolist()
        b_first = b.gather(1, first_columns[:, None]).flatten().cpu().tolist()
        columns = first_columns.cpu().tolist()
        byte_count += int(
            (
                original_a.contiguous().view(torch.uint8)
                != original_b.contiguous().view(torch.uint8)
            ).sum().item()
        )
        for offset, count in enumerate(count_list):
            row = start + offset
            difference = (
                {
                    "flat_column": columns[offset],
                    "actual": _json_number(a_first[offset]),
                    "expected": _json_number(b_first[offset]),
                }
                if count
                else None
            )
            per_row.append(
                {
                    "row": row,
                    "mismatched_elements": count,
                    "first_difference": difference,
                    "max_finite_absolute_error": _json_number(abs_list[offset]),
                    "max_finite_relative_error": _json_number(rel_list[offset]),
                    "nonfinite_actual": bad_a[offset],
                    "nonfinite_expected": bad_b[offset],
                }
            )
            if count and first is None:
                first = {"row": row, **difference}
            if abs_list[offset] > max_abs:
                max_abs, worst_row = abs_list[offset], row
            max_rel = max(max_rel, rel_list[offset])
            nonfinite += bad_a[offset] + bad_b[offset]
        mismatch_count += sum(count_list)
        if not bool(finite.all().item()):
            continue
        dot += float((ad * bd).sum().item())
        denominator += float((ad.square() + bd.square()).sum().item())
    if nonfinite or not math.isfinite(dot) or not math.isfinite(denominator):
        metric = {"status": "UNDEFINED_NONFINITE", "value": None}
    elif denominator == 0.0:
        metric = {"status": "BOTH_ZERO", "value": 0.0}
    else:
        metric = {"status": "RECORDED", "value": 1.0 - 2.0 * dot / denominator}
    result = {
        "strict_status": "PASS" if mismatch_count == 0 and nonfinite == 0 else "FAIL",
        "rtol": 0.0,
        "atol": 0.0,
        "mismatched_elements": mismatch_count,
        "mismatched_bytes": byte_count,
        "nonfinite_count": nonfinite,
        "first_difference": first,
        "worst_finite_absolute_error_row": worst_row,
        "max_finite_absolute_error": _json_number(max_abs),
        "max_finite_relative_error": _json_number(max_rel),
        "upstream_calc_diff_not_a_gate": metric,
        "per_row": per_row,
    }
    if qualification:
        result["qualification"] = _qualification_metric(torch, actual, expected)
    return result


def _row_inputs_identity(torch, inputs):
    return {
        name: _tensor_sha256(torch, value[:1] if name in ROW_INPUTS else value)
        for name, value in inputs.items()
    }


def _cpu_row_outputs(outputs):
    return {name: value[:1].detach().cpu().clone() for name, value in outputs.items()}


def _bitwise_comparison(torch, left, right):
    return {
        name: bool(
            torch.equal(
                left[name].contiguous().view(torch.uint8),
                right[name].contiguous().view(torch.uint8),
            )
        )
        for name in OUTPUTS
    }


def _alias_rejection_receipt(inputs, outputs):
    results = {}
    for name, conflicting in (
        ("output_input", dict(outputs, new_residual=inputs["residual"])),
        ("output_output", dict(outputs, new_prev_mix=outputs["new_post_mix"])),
        (
            "partial_storage_view",
            dict(outputs, new_prev_mix=outputs["new_post_mix"].reshape(-1)[1:]),
        ),
    ):
        try:
            _check_nonoverlap(inputs, conflicting)
        except ValueError as error:
            results[name] = {"status": "REJECTED_BEFORE_DISPATCH", "reason": str(error)}
        else:
            raise AssertionError(f"host alias validation failed to reject {name}")
    return results


def _case_memory(torch, rows):
    contracts = _contracts(torch, rows)
    sizes = {
        name: math.prod(shape) * (2 if dtype == torch.bfloat16 else 4)
        for name, (shape, dtype) in contracts.items()
    }
    guards = sum(
        2 * GUARD_ELEMENTS * (2 if dtype == torch.bfloat16 else 4)
        for _, dtype in contracts.values()
    )
    owned = sum(sizes.values()) + sum(sizes[name] for name in OUTPUTS) + guards
    estimate = owned + 128 * 2**20  # Bounded oracle/compare/CPU transfer chunks.
    if estimate > MAX_WORKING_BYTES:
        raise ValueError("case exceeds the fixed probe-owned working-memory budget")
    return {
        "payload_bytes_by_tensor": sizes,
        "owned_tensors_and_reference_bytes": owned,
        "bounded_scratch_allowance_bytes": 128 * 2**20,
        "estimated_probe_owned_max_bytes": estimate,
        "budget_bytes": MAX_WORKING_BYTES,
        "native_JIT_workspace": (
            "not included; allocated by qualified installed extension"
        ),
    }


def _failure_tensors(torch, root, case, inputs, outputs, expected):
    rows = inputs["x"].shape[0]
    selected = {0}
    for metrics in case["outputs"].values():
        if metrics["first_difference"] is not None:
            selected.add(metrics["first_difference"]["row"])
        selected.add(metrics["worst_finite_absolute_error_row"])
    if rows <= MAX_EVIDENCE_ROWS:
        selected = set(range(rows))
    selected = sorted(selected)[:MAX_EVIDENCE_ROWS]
    indices = torch.tensor(selected, dtype=torch.int64, device=inputs["x"].device)

    def host_rows(values, inputs_mapping=False):
        return {
            name: (
                value.index_select(0, indices)
                if not inputs_mapping or name in ROW_INPUTS
                else value
            ).detach().cpu()
            for name, value in values.items()
        }

    name = case["id"] + "-first-differences.pt"
    path = root / name
    torch.save(
        {
            "rows": selected,
            "coverage": (
                "all rows" if len(selected) == rows else "bounded first/worst rows"
            ),
            "inputs": host_rows(inputs, True),
            "actual": host_rows(outputs),
            "expected": host_rows(expected),
            "scalars": case["scalars"],
            "fixture": case["fixture"],
        },
        path,
    )
    return {
        "path": name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "selected_rows": selected,
        "full_case_reconstruction": (
            "fixed per-field seeds, variant, M and scalars in case receipt"
        ),
    }


def _qualification_bits_complete(values):
    return (
        isinstance(values, dict)
        and set(values) == set(OUTPUTS)
        and all(value is True for value in values.values())
    )


def _qualification_guards_complete(values, names):
    return (
        isinstance(values, dict)
        and set(values) == names
        and all(
            isinstance(item, dict)
            and set(item) == {"prefix", "suffix"}
            and item["prefix"] is True and item["suffix"] is True
            for item in values.values()
        )
    )


def _qualification_case(case):
    rows = case["rows"]
    names = set(FIELD_SEEDS) | set(OUTPUTS)
    inputs = case.get("input_hashes_before", {})
    valid_hashes = (
        set(inputs) == set(FIELD_SEEDS)
        and all(
            isinstance(value, str) and len(value) == 64 for value in inputs.values()
        )
    )
    metrics = {
        name: value.get("qualification")
        for name, value in case.get("outputs", {}).items()
    }
    fields_complete = set(metrics) == set(OUTPUTS) and all(
        _qualification_field_complete(value, rows) for value in metrics.values()
    )
    field_pass = fields_complete and all(
        _qualification_field_pass(value, rows) for value in metrics.values()
    )
    expected_pairs = set()
    if case["variant"] == "normal" and rows != 1:
        expected_pairs.add("alone")
    if case["variant"] == "normal" and rows == 65:
        expected_pairs.add("M64")
    pairs = case.get("cross_M", {})
    pair_complete = isinstance(pairs, dict) and set(pairs) == expected_pairs
    pair_identity, pair_pass = pair_complete, pair_complete
    if expected_pairs and pair_complete:
        for name, pair in pairs.items():
            fields = pair.get("qualification", {})
            complete = set(fields) == set(OUTPUTS) and all(
                _qualification_field_complete(value, 1) for value in fields.values()
            )
            pair_complete = pair_complete and complete
            pair_identity = pair_identity and (
                pair.get("identical_inputs_weights_scalars") is True
                and pair.get("previous_M") == (1 if name == "alone" else 64)
                and pair.get("current_M") == rows
                and set(pair.get("previous_input_hashes", {})) == set(FIELD_SEEDS)
                and pair.get("previous_input_hashes")
                == pair.get("current_input_hashes")
                and pair.get("current_input_hashes")
                == case.get("row0_and_weight_hashes")
            )
            pair_pass = pair_pass and complete and all(
                _qualification_field_pass(value, 1) for value in fields.values()
            )
    planned = 2 if rows == 3 and case["variant"] == "normal" else 1
    repeat = (
        _qualification_bits_complete(case.get("planned_repeat_bitwise"))
        if planned == 2 else "planned_repeat_bitwise" not in case
    )
    expected_duplicates = sorted({
        value for value in (3, 64, 1024, rows - 1) if 0 < value < rows
    })
    duplicates = case.get("duplicate_rows_bitwise")
    duplicate_complete = (
        isinstance(duplicates, dict)
        and set(duplicates) == {str(value) for value in expected_duplicates}
        and case.get("fixture", {}).get("duplicate_rows_from_row0")
        == expected_duplicates
    )
    duplicate_pass = duplicate_complete and (
        all(_qualification_bits_complete(value) for value in duplicates.values())
        if expected_duplicates else duplicates == {}
    )
    invocations = case.get("invocation_safety", [])
    invocation_complete = len(invocations) == planned and [
        value.get("invocation") for value in invocations
    ] == list(range(planned))
    invocation_pass = invocation_complete and all(
        value.get("status") == "PASS"
        and _qualification_guards_complete(value.get("canaries"), names)
        and value.get("input_hashes") == inputs
        for value in invocations
    )
    kwargs = list(FIELD_SEEDS) + [
        "hc_mult", "hc_norm_eps", "hc_pre_eps", "hc_post_scale", "sinkhorn_eps",
        "num_sinkhorn_iters", "rmsnorm_eps", "rmsnorm_scale",
    ] + list(OUTPUTS)
    hard = {
        "preflight": case.get("qualification_preflight") is True,
        "capability_preflight": case.get("capability") == "AVAILABLE",
        "native_shape": case.get("shape_proof") == "ACCEPTED_BY_NATIVE_API",
        "tensor_metadata_coverage": set(case.get("validated_tensors", {})) == names,
        "ABI22_kwargs": case.get("explicit_kernel_kwargs") == kwargs,
        "execution": case.get("execution") == "COMPLETED",
        "planned_calls": (
            case.get("planned_dispatch_calls") == planned
            and case.get("dispatch_calls") == planned
        ),
        "safety": case.get("safety") == "PASS",
        "input_immutability": (
            valid_hashes and case.get("inputs_immutable") is True
            and case.get("input_hashes_after") == inputs
        ),
        "canaries_before": _qualification_guards_complete(
            case.get("canaries_before"), names
        ),
        "canaries_after": _qualification_guards_complete(
            case.get("canaries_after"), names
        ),
        "invocation_safety": invocation_pass,
        "finite_outputs": _qualification_bits_complete(case.get("outputs_finite")),
        "same_M_repeat_bitwise": repeat,
        "same_batch_duplicate_bitwise": duplicate_pass,
        "cross_M_input_identity": pair_identity,
    }
    complete = fields_complete and pair_complete
    return {
        "profile": _QUALIFICATION_PROFILE,
        "complete": complete,
        "output_count": len(metrics),
        "expected_rows_per_output": rows,
        "cross_pair_count": len(pairs),
        "expected_cross_pair_count": len(expected_pairs),
        "cross_M_raw_bitwise_gate": False,
        "hard_gates": hard,
        "qualified_status": (
            "INCOMPLETE" if not complete else
            "PASS" if field_pass and pair_pass and all(hard.values()) else "FAIL"
        ),
    }


def _qualification_outcomes(report):
    cases = report["cases"]
    self_check = report.get("qualification_self_check", {})
    checks = self_check.get("checks", []) if isinstance(self_check, dict) else []
    expected_checks = (
        "both_zero", "opposite", "row_failure_hidden_by_global",
        "NaN", "Inf", "empty", "missing_row",
    )
    self_check_complete = (
        isinstance(self_check, dict)
        and self_check.get("profile") == _QUALIFICATION_PROFILE
        and self_check.get("status") == "PASS"
        and type(self_check.get("new_Mega_calls")) is int
        and self_check.get("new_Mega_calls") == 0
        and isinstance(checks, list) and len(checks) == len(expected_checks)
        and all(isinstance(item, dict) for item in checks)
        and [item.get("name") for item in checks] == list(expected_checks)
        and all(item.get("assertion_passed") is True for item in checks)
        and all(
            item.get("expected_pass") is (item["name"] == "both_zero")
            and item.get("actual_pass") is (item["name"] == "both_zero")
            for item in checks[:6]
        )
        and checks[-1].get("expected_rows") == 2
    )
    report["qualification_self_check_complete"] = self_check_complete
    expected = [
        (label, rows, variant)
        for label in ("geometry_1e-20", "generic_1e-6")
        for rows, variant in (
            [(value, "normal") for value in BASE_ROWS] + [(1, "zero"), (1, "tiny")]
        )
    ]
    actual = [
        (case["epsilon_contract"], case["rows"], case["variant"]) for case in cases
    ]
    coverage = (
        self_check_complete and bool(cases) and len(cases) == _QUALIFICATION_CASES
        and report.get("expected_case_count") == _QUALIFICATION_CASES
        and actual == expected and report.get("matrix_complete") is True
        and all(case.get("qualification", {}).get("complete") is True for case in cases)
        and all(
            case.get("qualification", {}).get("output_count") == len(OUTPUTS)
            for case in cases
        )
        and sum(
            case.get("qualification", {}).get("cross_pair_count", 0) for case in cases
        )
        == _QUALIFICATION_CROSS_PAIRS
        and sum(case["dispatch_calls"] for case in cases) == _QUALIFICATION_CALLS
    )
    report["qualification_complete"] = coverage
    if report.get("error") or any(
        case.get("qualified_status") == "FAIL" for case in cases
    ):
        status = "FAIL"
    elif coverage and all(case.get("qualified_status") == "PASS" for case in cases):
        status = (
            "PASS" if report.get("execution") == "PASS"
            and report.get("safety") == "PASS"
            and report.get("capability") == "AVAILABLE" else "FAIL"
        )
    else:
        status = "INCOMPLETE"
    report["qualified_status"] = status


def _case_summary(case, artifact, qualification=False):
    result = {
        "id": case["id"],
        "rows": case["rows"],
        "epsilon_contract": case["epsilon_contract"],
        "variant": case["variant"],
        "status": case.get("status", "FAIL"),
        "shape_proof": case.get("shape_proof", "NOT_RUN"),
        "dispatch_calls": case.get("dispatch_calls", 0),
        "strict_correctness": case.get("strict_correctness", "NOT_RUN"),
        "comparison_outputs_completed": len(case.get("outputs", {})),
        "capability": case.get("capability", "UNKNOWN"),
        "execution": case.get("execution", "NOT_RUN"),
        "safety": case.get("safety", "NOT_RUN"),
        "failure_stage": case.get("failure_stage"),
        "failure_kind": case.get("failure_kind"),
        "max_probe_owned_bytes": case.get("memory", {}).get(
            "estimated_probe_owned_max_bytes"
        ),
        "artifact": artifact,
    }
    if qualification:
        result["qualified_status"] = case.get("qualified_status", "INCOMPLETE")
        result["qualification"] = case.get("qualification", {})
    return result


def _update_outcomes(report, matrix_complete=False, qualification=False):
    """Aggregate recorded verdicts without inventing mathematical evidence."""
    cases = report["cases"]
    complete = (
        matrix_complete
        and bool(cases)
        and len(cases) == report.get("expected_case_count")
    )
    report["matrix_complete"] = complete
    if complete and all(
        case["shape_proof"] == "ACCEPTED_BY_NATIVE_API" for case in cases
    ):
        report["shape_proof"] = "ACCEPTED_BY_NATIVE_API"
    elif any(case["shape_proof"] == "ACCEPTED_BY_NATIVE_API" for case in cases):
        report["shape_proof"] = "INCOMPLETE"
    else:
        report["shape_proof"] = "NOT_RUN"
    if any(case["strict_correctness"] == "FAIL" for case in cases):
        mathematical = "FAIL"
    elif complete and all(
        case["strict_correctness"] == "PASS"
        and case["comparison_outputs_completed"] == len(OUTPUTS)
        for case in cases
    ):
        mathematical = "PASS"
    elif any(case["comparison_outputs_completed"] for case in cases):
        mathematical = "INCOMPLETE"
    else:
        mathematical = "NOT_RUN"
    report["strict_correctness"] = mathematical
    report["correctness"] = mathematical
    if any(case["execution"] == "FAIL" for case in cases):
        report["execution"] = "FAIL"
    elif complete and all(case["execution"] == "COMPLETED" for case in cases):
        report["execution"] = "PASS"
    elif any(case["dispatch_calls"] for case in cases):
        report["execution"] = "INCOMPLETE"
    else:
        report["execution"] = "NOT_RUN"
    if any(case["safety"] == "FAIL" for case in cases):
        report["safety"] = "FAIL"
    elif complete and all(case["safety"] == "PASS" for case in cases):
        report["safety"] = "PASS"
    elif any(case["dispatch_calls"] for case in cases):
        report["safety"] = "INCOMPLETE"
    else:
        report["safety"] = "NOT_RUN"
    if qualification:
        _qualification_outcomes(report)


def _run_case(torch, kernel, device, root, case, invariance, qualification=False):
    case["phase"] = "memory_budget"
    rows, variant, epsilon = case["rows"], case["variant"], case["epsilon"]
    case["memory"] = _case_memory(torch, rows)
    case["phase"] = "fixture"
    inputs, outputs, buffers, duplicates = _make_case(torch, rows, device, variant)
    scalars = {
        "hc_mult": HC,
        "hc_norm_eps": epsilon,
        "hc_pre_eps": 1e-6,
        "hc_post_scale": 2.0,
        "sinkhorn_eps": 1e-6,
        "num_sinkhorn_iters": 20,
        "rmsnorm_eps": epsilon,
        "rmsnorm_scale": 1.0,
    }
    case["scalars"] = scalars
    case["fixture"] = {
        "seed": SEED,
        "field_seed_offsets": FIELD_SEEDS,
        "row0_seed_addition": 10000,
        "row0_overridden_identically_for_all_M": True,
        "duplicate_rows_from_row0": duplicates,
        "oracle_rows_per_chunk": CHUNK_ROWS,
        "norm_applications": 1,
        "bf16_landing_points": [
            "updated residual",
            "carried-pre collapse",
            "normalized output",
        ],
    }
    case["phase"] = "preflight"
    _preflight(torch, inputs, outputs, scalars, device)
    if qualification:
        case["qualification_preflight"] = True
    case["validated_tensors"] = {
        name: _tensor_record(value) for name, value in {**inputs, **outputs}.items()
    }
    case["host_alias_rejections"] = _alias_rejection_receipt(inputs, outputs)
    case["phase"] = "input_identity"
    case["input_hashes_before"] = {
        name: _tensor_sha256(torch, value) for name, value in inputs.items()
    }
    case["row0_and_weight_hashes"] = _row_inputs_identity(torch, inputs)
    case["canaries_before"] = _canaries(torch, buffers)
    if not all(all(value.values()) for value in case["canaries_before"].values()):
        case["safety"] = "FAIL"
        raise AssertionError("fixture guard corruption before dispatch")
    case["phase"] = "independent_reference"
    expected = independent_transition_reference(torch, inputs, scalars)
    arguments = {
        "x": inputs["x"],
        "residual": inputs["residual"],
        "shifted_prev_mix": inputs["shifted_prev_mix"],
        "post_mix": inputs["post_mix"],
        "comb_res_mix": inputs["comb_res_mix"],
        "fn": inputs["fn"],
        "mix_scales": inputs["mix_scales"],
        "mix_bases": inputs["mix_bases"],
        "rmsnorm_weight": inputs["rmsnorm_weight"],
        "hc_mult": scalars["hc_mult"],
        "hc_norm_eps": scalars["hc_norm_eps"],
        "hc_pre_eps": scalars["hc_pre_eps"],
        "hc_post_scale": scalars["hc_post_scale"],
        "sinkhorn_eps": scalars["sinkhorn_eps"],
        "num_sinkhorn_iters": scalars["num_sinkhorn_iters"],
        "rmsnorm_eps": scalars["rmsnorm_eps"],
        "rmsnorm_scale": scalars["rmsnorm_scale"],
        "new_residual": outputs["new_residual"],
        "new_prev_mix": outputs["new_prev_mix"],
        "new_post_mix": outputs["new_post_mix"],
        "new_comb_res_mix": outputs["new_comb_res_mix"],
        "y_bf16": outputs["y_bf16"],
    }
    case["explicit_kernel_kwargs"] = list(arguments)
    case["dispatch_calls"] = 0
    case["planned_dispatch_calls"] = 2 if rows == 3 and variant == "normal" else 1
    case["possible_writes"] = False
    case["dispatch_state"] = "PREFLIGHT_VALIDATED"
    case["phase"] = "pre_dispatch_receipt"
    _write_json(root, case["id"] + ".json", case)
    first_cpu = None
    for invocation in range(case["planned_dispatch_calls"]):
        if invocation:
            case["phase"] = "repeat_preflight"
            for value in outputs.values():
                value.fill_(float("nan"))
            _preflight(torch, inputs, outputs, scalars, device)
        case["dispatch_calls"] += 1
        case["possible_writes"] = True
        case["dispatch_state"] = "ATTEMPTED"
        case["phase"] = "native_dispatch"
        case["execution"] = "ATTEMPTED"
        # Any native call or its synchronization exception escapes. The outer
        # driver records it and stops ALL later dispatch: never retry/fallback.
        kernel(**arguments)
        case["phase"] = "synchronization"
        torch.cuda.synchronize(device)
        case["dispatch_state"] = "RETURNED_AND_SYNCHRONIZED"
        case["execution"] = "RETURNED_AND_SYNCHRONIZED"
        case["shape_proof"] = "ACCEPTED_BY_NATIVE_API"
        case["phase"] = "invocation_safety"
        invocation_guards = _canaries(torch, buffers)
        invocation_hashes = {
            name: _tensor_sha256(torch, value) for name, value in inputs.items()
        }
        invocation_safe = (
            all(all(value.values()) for value in invocation_guards.values())
            and invocation_hashes == case["input_hashes_before"]
        )
        case.setdefault("invocation_safety", []).append(
            {
                "invocation": invocation,
                "canaries": invocation_guards,
                "input_hashes": invocation_hashes,
                "status": "PASS" if invocation_safe else "FAIL",
            }
        )
        if not invocation_safe:
            case["safety"] = "FAIL"
            raise AssertionError(
                "guard/input mutation after dispatch; stop before any next invocation"
            )
        if invocation == 0:
            case["phase"] = "mathematical_comparison"
            case["outputs"] = {}
            for name in OUTPUTS:
                case["outputs"][name] = _compare(
                    torch, outputs[name], expected[name], qualification=qualification
                )
                case["strict_correctness"] = (
                    "FAIL"
                    if any(
                        value["strict_status"] == "FAIL"
                        for value in case["outputs"].values()
                    )
                    else "INCOMPLETE"
                )
            first_cpu = (
                {
                    name: value.detach().cpu().clone()
                    for name, value in outputs.items()
                }
                if case["planned_dispatch_calls"] > 1
                else None
            )
        else:
            case["phase"] = "repeat_comparison"
            repeated = {name: value.detach().cpu() for name, value in outputs.items()}
            case["planned_repeat_bitwise"] = _bitwise_comparison(
                torch, first_cpu, repeated
            )
    case["execution"] = "COMPLETED"
    case["phase"] = "post_dispatch_safety"
    case["canaries_after"] = _canaries(torch, buffers)
    case["input_hashes_after"] = {
        name: _tensor_sha256(torch, value) for name, value in inputs.items()
    }
    case["inputs_immutable"] = case["input_hashes_before"] == case["input_hashes_after"]
    case["outputs_finite"] = {
        name: _finite(torch, value) for name, value in outputs.items()
    }
    guards_ok = all(all(value.values()) for value in case["canaries_after"].values())
    case["safety"] = "PASS" if guards_ok and case["inputs_immutable"] else "FAIL"
    case["phase"] = "state_invariance"
    case["duplicate_rows_bitwise"] = {
        str(index): _bitwise_comparison(
            torch,
            _cpu_row_outputs(outputs),
            {
                name: value[index : index + 1].detach().cpu()
                for name, value in outputs.items()
            },
        )
        for index in duplicates
    }
    if variant == "normal":
        key = case["epsilon_contract"]
        current = {
            "inputs": case["row0_and_weight_hashes"],
            "scalars": scalars,
            "outputs": _cpu_row_outputs(outputs),
            "rows": rows,
        }
        for other_key in ("alone", "M64"):
            previous = invariance.get((key, other_key))
            if previous is not None and (other_key == "alone" or rows == 65):
                identical = (
                    previous["inputs"] == current["inputs"]
                    and previous["scalars"] == scalars
                )
                case.setdefault("cross_M", {})[other_key] = {
                    "previous_M": previous["rows"],
                    "current_M": rows,
                    "identical_inputs_weights_scalars": identical,
                    "previous_input_hashes": previous["inputs"],
                    "current_input_hashes": current["inputs"],
                    "bitwise_outputs": (
                        _bitwise_comparison(
                            torch, previous["outputs"], current["outputs"]
                        )
                        if identical
                        else None
                    ),
                }
                if qualification and identical:
                    case["cross_M"][other_key]["qualification"] = {
                        name: _qualification_metric(
                            torch, previous["outputs"][name], current["outputs"][name]
                        )
                        for name in OUTPUTS
                    }
        if rows == 1:
            invariance[(key, "alone")] = current
        if rows == 64:
            invariance[(key, "M64")] = current
    case["phase"] = "verdict_aggregation"
    numerical_ok = all(
        metrics["strict_status"] == "PASS" for metrics in case["outputs"].values()
    )
    repeated_ok = all(case.get("planned_repeat_bitwise", {}).values())
    duplicates_ok = all(
        all(value.values()) for value in case["duplicate_rows_bitwise"].values()
    )
    cross_ok = all(
        value["identical_inputs_weights_scalars"]
        and all(value["bitwise_outputs"].values())
        for value in case.get("cross_M", {}).values()
    )
    case["strict_correctness"] = "PASS" if numerical_ok else "FAIL"
    case["state_invariance"] = (
        "PASS" if repeated_ok and duplicates_ok and cross_ok else "FAIL"
    )
    case["status"] = (
        "PASS"
        if numerical_ok
        and repeated_ok
        and duplicates_ok
        and cross_ok
        and case["safety"] == "PASS"
        and all(case["outputs_finite"].values())
        else "FAIL"
    )
    if qualification:
        case["qualification"] = _qualification_case(case)
        case["qualified_status"] = case["qualification"]["qualified_status"]
    if case["status"] == "FAIL":
        case["failure_stage"] = (
            "mathematical_comparison" if not numerical_ok else "state_invariance"
        )
        case["failure_kind"] = "mathematical" if not numerical_ok else "state"
        case["phase"] = "failure_evidence"
        case["failure_tensor_evidence"] = _failure_tensors(
            torch, root, case, inputs, outputs, expected
        )
    if case["safety"] != "PASS":
        case["phase"] = "post_dispatch_safety"
        raise AssertionError(
            "canary or input immutability failure; no further dispatch is safe"
        )
    # Only bounded CPU row0 snapshots survive this call. Large fixtures, output
    # buffers, and the independent reference are released before the next M.
    del inputs, outputs, buffers, expected, arguments, first_cpu
    gc.collect()
    case["phase"] = "complete"


def run_shape_probe(root, report, gamma, query_width, qualification=False):
    """Run the explicitly bounded, eager real-M890 diagnostic, never timing."""
    report_filename = "mhc-qualified04.json" if qualification else "mhc-mega.json"
    report["phase"] = "configuration"
    if gamma is None or query_width is None:
        raise ValueError(
            "explicit RTP_V41_S5_GAMMA and RTP_V41_S5_QUERY_WIDTH "
            "(or CLI flags) are required"
        )
    if not 1 <= gamma <= 64 or query_width not in (gamma, gamma + 1):
        raise ValueError(
            "bounded gamma must be1..64; query_width must equal gamma or gamma+1"
        )
    if qualification and (gamma, query_width) != (3, 4):
        raise ValueError("fixed qualification profile requires gamma3/query_width4")
    report["phase"] = "installed_symbol"
    torch = sys.modules.get("torch")
    if qualification and torch is not None:
        report["phase"] = "qualification_self_check"
        report["qualification_self_check"] = _qualification_self_check(torch)
        _write_json(
            root, "mhc-qualified04-self-check.json", report["qualification_self_check"]
        )
        if report["qualification_self_check"]["status"] != "PASS":
            raise AssertionError(
                "fixed numerical-gate self-check failed before dispatch"
            )
        report["phase"] = "installed_symbol"
    deep_gemm = sys.modules.get("deep_gemm")
    kernel = vars(deep_gemm).get("mega_mhc") if deep_gemm is not None else None
    if torch is None or not callable(kernel):
        report["capability"] = "UNAVAILABLE"
        report["capability_reason"] = (
            "required module/callable absent in selected runtime"
        )
        raise RuntimeError(
            "required installed deep_gemm.mega_mhc unavailable; no skip or fallback"
        )
    report["phase"] = "extension_identity"
    if not report["surface"]["selected_extension_matches_qualified_baseline"]:
        report["capability_reason"] = "qualified extension identity not established"
        raise RuntimeError(
            "selected DeepGEMM extension is not the qualified816ff222 baseline"
        )
    report["phase"] = "device_initialization"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    report["phase"] = "device_properties"
    properties = torch.cuda.get_device_properties(device)
    report["device"] = {
        "name": properties.name,
        "major": properties.major,
        "minor": properties.minor,
        "total_memory": properties.total_memory,
        "visible_selection": "cuda:0 within the standard gpu_lock visibility",
    }
    if "M890" not in properties.name or (properties.major, properties.minor) != (8, 9):
        report["capability"] = "UNAVAILABLE"
        report["capability_reason"] = (
            "observed card violates fixed M890 capability8.9 contract"
        )
        raise RuntimeError("fixed source contract requires real M890 capability8.9")
    report["capability"] = "AVAILABLE"
    report["capability_reason"] = (
        "required callable, qualified extension hash and card preflight only; "
        "shape acceptance and mathematical correctness remain independent"
    )
    rows = sorted(set(BASE_ROWS) | {gamma, query_width})
    report["expected_case_count"] = 2 * (len(rows) + 2)
    report["configuration"] = {
        "gamma": gamma,
        "query_width": query_width,
        "rows": rows,
        "source": (
            "explicit synthetic probe configuration, no checkpoint/model/group loaded"
        ),
        "target_verify_geometry": (
            "causal stages use active B, not one B*(gamma+1) kernel call"
        ),
        "large_rows": (
            "8192/16384 are synthetic local-row representatives, "
            "not64K/128K CP acceptance"
        ),
    }
    report["reference"] = {
        "math": (
            "independent unfused IEEE FP32 Torch/einsum/Sinkhorn;"
            "64-row evaluation chunks"
        ),
        "candidate_math_source": (
            "TF32 MMA/FP32 accumulation; residual/collapse BF16; M-dependent splits"
        ),
        "strict_budget_frozen_before_output": {
            "rtol": 0.0,
            "atol": 0.0,
            "outputs": list(OUTPUTS),
        },
        "upstream_metric": (
            "1-2*sum(x*y)/sum(x*x+y*y), float64, zero/nonfinite guarded; "
            "separate only"
        ),
        "norm": "one norm, after BF16 collapse; actual1e-20 and generic1e-6 separate",
    }
    report["phase"] = "reference_policy"
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    report["reference"]["previous_allow_tf32"] = old_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    invariance = {}
    try:
        for label, epsilon in (("geometry_1e-20", 1e-20), ("generic_1e-6", 1e-6)):
            plans = [(count, "normal") for count in rows] + [(1, "zero"), (1, "tiny")]
            for count, variant in plans:
                report["phase"] = "case"
                case = {
                    "id": (
                        ("qualification04-" if qualification else "")
                        + f"case-{len(report['cases']):03d}-{label}-M{count}-{variant}"
                    ),
                    "rows": count,
                    "epsilon_contract": label,
                    "epsilon": epsilon,
                    "variant": variant,
                    "status": "RUNNING",
                    "strict_correctness": "NOT_RUN",
                    "capability": report["capability"],
                    "execution": "NOT_RUN",
                    "safety": "NOT_RUN",
                    "performance": "NOT_RUN",
                    "promotion": "NOT_REQUESTED",
                }
                try:
                    with torch.no_grad():
                        _run_case(
                            torch, kernel, device, root, case, invariance,
                            qualification=qualification,
                        )
                except Exception as error:
                    case["status"] = "FAIL"
                    if qualification:
                        case["qualified_status"] = "FAIL"
                    case["error"] = _error(error)
                    case["failure_stage"] = case.get("phase", "case_setup")
                    if case["failure_stage"] in ("native_dispatch", "synchronization"):
                        case["execution"] = "FAIL"
                        case["failure_kind"] = "execution"
                    elif case["safety"] == "FAIL":
                        case["failure_kind"] = "safety"
                    elif case["failure_stage"] in (
                        "invocation_safety",
                        "post_dispatch_safety",
                    ):
                        case["failure_kind"] = "safety"
                        case["safety"] = "UNKNOWN"
                    elif case.get("dispatch_calls", 0):
                        case["failure_kind"] = "evidence"
                    else:
                        case["failure_kind"] = "pre_dispatch"
                    if case.get("dispatch_calls", 0) and case["safety"] == "NOT_RUN":
                        case["safety"] = "UNKNOWN"
                    case["remaining_dispatch"] = (
                        "STOPPED; no retry/fallback after possible writes"
                    )
                    summary = _case_summary(case, None, qualification=qualification)
                    report["cases"].append(summary)
                    report["failure_stage"] = case["failure_stage"]
                    report["failure_kind"] = case["failure_kind"]
                    _update_outcomes(report, qualification=qualification)
                    try:
                        report["phase"] = "case_receipt"
                        summary["artifact"] = _write_json(
                            root, case["id"] + ".json", case
                        )
                        report["phase"] = "matrix_receipt"
                        _write_json(root, report_filename, report)
                    except Exception:
                        report["failure_stage"] = report["phase"]
                        report["failure_kind"] = "evidence"
                        raise
                    raise
                summary = _case_summary(case, None, qualification=qualification)
                report["cases"].append(summary)
                _update_outcomes(report, qualification=qualification)
                report["phase"] = "case_receipt"
                summary["artifact"] = _write_json(root, case["id"] + ".json", case)
                report["phase"] = "matrix_receipt"
                _write_json(root, report_filename, report)
    finally:
        if qualification:
            pending_exception = sys.exc_info()
            pending_phase = report["phase"]
        try:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
        except Exception as restore_error:
            if qualification and pending_exception[1] is not None:
                if report.get("failure_stage") is None:
                    report["failure_stage"] = pending_phase
                    report["failure_kind"] = "evidence"
                report["qualification_primary_error"] = {
                    "type": type(pending_exception[1]).__name__,
                    "message": str(pending_exception[1]),
                    "traceback": "".join(
                        traceback.format_exception(*pending_exception)
                    ),
                }
                report["qualification_restore_error"] = _error(restore_error)
                raise pending_exception[1].with_traceback(
                    pending_exception[2]
                ) from restore_error
            elif qualification and report.get("failure_stage") is not None:
                report["qualification_restore_error"] = _error(restore_error)
            else:
                report["phase"] = "reference_policy_restore"
                report["failure_stage"] = "reference_policy_restore"
                report["failure_kind"] = "evidence"
            raise
    report["phase"] = "matrix_aggregation"
    _update_outcomes(report, matrix_complete=True, qualification=qualification)
    report["shape_proof"] = (
        "ACCEPTED_BY_NATIVE_API"
        if report["matrix_complete"] and all(
            case["shape_proof"] == "ACCEPTED_BY_NATIVE_API" for case in report["cases"]
        )
        else "FAIL"
    )
    report["status"] = (
        "PASS_STRICT_DIAGNOSTIC_ONLY"
        if report["matrix_complete"]
        and report["strict_correctness"] == "PASS"
        and all(case["status"] == "PASS" for case in report["cases"])
        else "FAIL"
    )
    if qualification:
        report["status"] = (
            "PASS_NUMERICAL_QUALIFICATION"
            if report["qualified_status"] == "PASS" else "FAIL"
        )
    report["phase"] = "complete"
    for case in report["cases"]:
        if case["status"] == "FAIL":
            if qualification:
                report["strict_diagnostic_failure_stage"] = case["failure_stage"]
                report["strict_diagnostic_failure_kind"] = case["failure_kind"]
            else:
                report["failure_stage"] = case["failure_stage"]
                report["failure_kind"] = case["failure_kind"]
            break
    if qualification and report["qualified_status"] != "PASS":
        report["failure_stage"] = "numerical_qualification"
        report["failure_kind"] = "qualification"


# Scope03 binds only these five original P/mhc01 receipts and safe PT payloads.
# JSON byte counts are read-only directory observations; SHA256 binds contents.
_EVIDENCE_ZIP_SHA256 = (
    "995531ea9991ad1a6bafcba40c87896b6dc703d14e1f7f58ee25487639e139ab"
)
_EVIDENCE_ZIP_BYTES = 72539963
_EVIDENCE_SPECS = (
    (
        "case-000-geometry_1e-20-M1-normal", 1, [0], 18500,
        "5419dc810510fc3dd9b37ad4714295c33b390a906ac1fd31d7aadf16bc43212f",
        2138422,
        "d3efb2792a8c25c0baef5bcac6da0c73c7cabfab6cfc5f10a542d2fb25526f3a",
    ),
    (
        "case-002-geometry_1e-20-M4-normal", 4, [0, 1, 2, 3], 26672,
        "22d431ab4324473df2bfbf04708e5631442a3aee75922d3e1331e03481ba17b2",
        2600182,
        "259f4ed073a3a8347b4780a83b24356bbfe0ef1682239d2b90a0c7ff16c5fa2f",
    ),
    (
        "case-005-geometry_1e-20-M64-normal", 64, [0, 1, 4, 7, 22, 27, 41], 143692,
        "78dfb62456b4bea4b09f643e869bfa40b1f9264c0cd196f9295e4f86f611ec5d",
        3062031,
        "cc8b948abfcf04f8d0f9357a011cba6d28c439efab9f327f4f4bf00edb050c0d",
    ),
    (
        "case-006-geometry_1e-20-M65-normal", 65, [0, 1, 4, 7, 27, 41, 63], 147553,
        "72345ea8ecbbafea344a9e90e9cda746c8b33b47729c9299a3f1f1b896df827b",
        3062031,
        "353697c9147dcd117c84a19b71b8c1428ce9a8d6f06d30a2877462329ab66689",
    ),
    (
        "case-011-geometry_1e-20-M1-zero", 1, [0], 18498,
        "cd2678d40070d9c8bc2b72fd3ef551ad1d89798690991410690a88411088ad6f",
        2138372,
        "3fe4b2cd20fe8df5feca4a21ec0c7e52cc979426489697ed9d0b39f4692c11f7",
    ),
)


def _evidence_archive(path):
    import io
    import zipfile

    path = Path(path)
    if not path.is_file() or path.stat().st_size != _EVIDENCE_ZIP_BYTES:
        raise ValueError("evidence ZIP size/path differs from the fixed original")
    if _sha256(path) != _EVIDENCE_ZIP_SHA256:
        raise ValueError("evidence ZIP SHA256 differs from the fixed original")
    verified, raw = {}, {}
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
        names = archive.namelist()
        for case_id, _, _, json_bytes, json_sha, pt_bytes, pt_sha in _EVIDENCE_SPECS:
            for name, size, digest in (
                (case_id + ".json", json_bytes, json_sha),
                (case_id + "-first-differences.pt", pt_bytes, pt_sha),
            ):
                if names.count(name) != 1:
                    raise ValueError(f"missing/duplicate selected ZIP member: {name}")
                info = archive.getinfo(name)
                if info.file_size != size:
                    raise ValueError(f"selected ZIP member size mismatch: {name}")
                value = archive.read(name)  # ZIP checks its CRC; never extract.
                if len(value) != size or hashlib.sha256(value).hexdigest() != digest:
                    raise ValueError(f"selected ZIP member checksum mismatch: {name}")
                raw[name] = value
                verified[name] = {"bytes": size, "sha256": digest, "CRC": info.CRC}
    # No JSON interpretation or PT loading occurs before ALL ten checks pass.
    return raw, {
        "archive": _file_record(path),
        "members": verified,
        "selected_PT_count": 5,
        "extraction": "NONE; ZIP/BytesIO only",
    }


def _evidence_payloads(torch, raw):
    import io

    payloads, receipts = {}, {}
    scalar_contract = {
        "hc_mult": 4, "hc_norm_eps": 1e-20, "hc_pre_eps": 1e-6,
        "hc_post_scale": 2.0, "sinkhorn_eps": 1e-6, "num_sinkhorn_iters": 20,
        "rmsnorm_eps": 1e-20, "rmsnorm_scale": 1.0,
    }
    for case_id, full_rows, selected, _, _, _, _ in _EVIDENCE_SPECS:
        receipt = json.loads(raw[case_id + ".json"])
        evidence_name = case_id + "-first-differences.pt"
        payload = torch.load(
            io.BytesIO(raw[evidence_name]), weights_only=True, map_location="cpu"
        )
        required = {
            "rows", "coverage", "inputs", "actual", "expected", "scalars", "fixture"
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError(f"invalid safe PT mapping: {case_id}")
        if (
            payload["rows"] != selected
            or receipt["id"] != case_id
            or receipt["rows"] != full_rows
            or receipt["failure_tensor_evidence"]["selected_rows"] != selected
            or payload["scalars"] != scalar_contract
            or payload["scalars"] != receipt["scalars"]
            or payload["fixture"] != receipt["fixture"]
            or receipt["strict_correctness"] != "FAIL"
        ):
            raise ValueError(f"PT/JSON row/scalar/fixture identity mismatch: {case_id}")
        if payload["coverage"] != (
            "all rows" if len(selected) == full_rows else "bounded first/worst rows"
        ):
            raise ValueError(f"invalid saved subset coverage: {case_id}")
        contracts = _contracts(torch, len(selected))
        for group, names in (
            ("inputs", tuple(FIELD_SEEDS)), ("actual", OUTPUTS), ("expected", OUTPUTS)
        ):
            if (
                not isinstance(payload[group], dict)
                or set(payload[group]) != set(names)
            ):
                raise ValueError(f"invalid tensor keys in {case_id}/{group}")
            for name in names:
                value = payload[group][name]
                shape, dtype = contracts[name]
                if (
                    not isinstance(value, torch.Tensor)
                    or tuple(value.shape) != shape
                    or value.dtype != dtype
                    or value.device.type != "cpu"
                    or value.layout != torch.strided
                    or not value.is_contiguous()
                ):
                    raise ValueError(
                        f"invalid safe PT tensor: {case_id}/{group}/{name}"
                    )
        computed = _row_inputs_identity(torch, payload["inputs"])
        if computed != receipt["row0_and_weight_hashes"]:
            raise ValueError(f"saved row0/weight hash mismatch: {case_id}")
        if len(selected) == full_rows:
            full_hashes = {
                name: _tensor_sha256(torch, value)
                for name, value in payload["inputs"].items()
            }
            if full_hashes != receipt["input_hashes_before"]:
                raise ValueError(f"full input hash mismatch: {case_id}")
        payloads[case_id] = payload
        receipts[case_id] = {
            "original_M": full_rows,
            "saved_original_row_ids": selected,
            "original_id_to_storage_index": {
                str(row): i for i, row in enumerate(selected)
            },
            "row0_and_weight_hashes": computed,
            "scalars": receipt["scalars"],
            "original_strict_correctness": receipt["strict_correctness"],
            "full_input_identity": (
                "VERIFIED" if len(selected) == full_rows else "UNKNOWN_UNSAVED_ROWS"
            ),
            "original_full_input_hashes": receipt["input_hashes_before"],
        }
    return payloads, receipts


def _evidence_bits(torch, value):
    integer = torch.int16 if value.dtype == torch.bfloat16 else torch.int32
    mask = 0xFFFF if value.dtype == torch.bfloat16 else 0xFFFFFFFF
    return [
        int(item) & mask
        for item in value.contiguous().view(integer).flatten().tolist()
    ]


def _evidence_scalar_bits(value):
    import struct

    return struct.unpack("<I", struct.pack("<f", value))[0]


def _evidence_match(torch, value, observed):
    left, right = _evidence_bits(torch, value), _evidence_bits(torch, observed)
    differences = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
    return {
        "control_sha256": _tensor_sha256(torch, value),
        "observed_sha256": _tensor_sha256(torch, observed),
        "numerically_equal": bool(torch.equal(value, observed)),
        "different_raw_elements": len(differences),
        "first_raw_differences": [
            {"flat_index": i, "control_bits": left[i], "observed_bits": right[i]}
            for i in differences[:16]
        ],
        "meaning": "selected explanatory equality only; no kernel PASS or promotion",
    }


def _evidence_fmaf():
    import ctypes

    try:
        if not sys.platform.startswith("linux") or ctypes.sizeof(ctypes.c_float) != 4:
            raise RuntimeError("no declared Linux binary32 libm ABI")
        library = ctypes.CDLL("libm.so.6")
        function = library.fmaf
        function.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
        function.restype = ctypes.c_float
        round_query = library.fegetround
        round_query.argtypes = ()
        round_query.restype = ctypes.c_int
        rounding = round_query()
        if rounding != 0:
            raise RuntimeError("existing fenv is not Linux FE_TONEAREST(0)")
        address = ctypes.cast(function, ctypes.c_void_p).value
        path = None
        for line in Path("/proc/self/maps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            begin, end = (int(part, 16) for part in fields[0].split("-"))
            if (
                len(fields) == 6
                and begin <= address < end
                and fields[5].startswith("/")
            ):
                path = fields[5]
                break
        identity = _file_record(path)
        if identity.get("status") != "RECORDED":
            raise RuntimeError("cannot identify the actual mapped fmaf library")
        return function, {
            "status": "AVAILABLE_CONTROL_PRIMITIVE",
            "primitive": (
                "existing C libm fmaf, one correctly rounded binary32 operation"
            ),
            "library": identity, "symbol_address": address,
            "round_mode": rounding, "declared_FE_TONEAREST": 0,
            "native_PPU_rounding": "UNKNOWN; this is an explicit host control",
        }
    except Exception as error:
        return None, {
            "status": "UNKNOWN", "reason": _error(error), "substitute": "NONE"
        }


def _evidence_residual_order(torch, payload, fmaf):
    row, lane, column = payload["rows"].index(1), 1, 943
    inputs = payload["inputs"]
    branch = inputs["x"][row, column].float().item()
    post = inputs["post_mix"][row, lane, 0].item()
    terms = [
        (
            inputs["residual"][row, j, column].float().item(),
            inputs["comb_res_mix"][row, j, lane].item(),
        )
        for j in range(HC)
    ]
    result = {
        "original_row": 1, "storage_index": row, "lane": lane, "column": column,
        "branch_FP32_bits": _evidence_scalar_bits(branch),
        "post_FP32_bits": _evidence_scalar_bits(post),
        "terms_FP32_bits": [
            {"source_lane": j, "residual": _evidence_scalar_bits(a),
             "comb_j_i": _evidence_scalar_bits(b)}
            for j, (a, b) in enumerate(terms)
        ],
        "observed_native_BF16_bits": _evidence_bits(
            torch, payload["actual"]["new_residual"][row, lane, column]
        )[0],
        "recorded_oracle_BF16_bits": _evidence_bits(
            torch, payload["expected"]["new_residual"][row, lane, column]
        )[0],
        "native_pre_landing_accumulator": "UNKNOWN",
    }
    if fmaf is None:
        return {**result, "control_status": "UNKNOWN_NO_EXACT_FMAF"}
    # Initial multiplication is a separate, declared Torch CPU FP32 operation.
    accumulator = (
        torch.tensor(branch, dtype=torch.float32)
        * torch.tensor(post, dtype=torch.float32)
    ).item()
    trace = [{
        "stage": "post_first_multiply",
        "FP32_bits": _evidence_scalar_bits(accumulator),
    }]
    for j, (residual, comb) in enumerate(terms):
        accumulator = fmaf(residual, comb, accumulator)
        trace.append({
            "source_lane": j, "FP32_bits": _evidence_scalar_bits(accumulator)
        })
    landing = torch.tensor(accumulator, dtype=torch.float32).to(torch.bfloat16)
    result.update({
        "control_status": "RECORDED_EXPLANATORY_ONLY",
        "arithmetic": (
            "post*x rounded FP32; then four source-ordered libm fmaf; BF16 once"
        ),
        "trace": trace,
        "before_BF16_FP32_bits": _evidence_scalar_bits(accumulator),
        "control_BF16_bits": _evidence_bits(torch, landing)[0],
    })
    return result


def _evidence_collapse_norm(torch, payload, fmaf):
    row, focus = payload["rows"].index(7), 2841
    pre = payload["inputs"]["shifted_prev_mix"][row, :, 0]
    weight = payload["inputs"]["rmsnorm_weight"]
    controls = []
    for residual_name in ("expected", "actual"):
        residual = payload[residual_name]["new_residual"][row]
        for arithmetic in ("separate_FP32_mul_add", "source_ordered_fmaf"):
            entry = {
                "residual_input": residual_name + "_saved_R_control_not_golden",
                "original_row": 7, "storage_index": row, "focus_column": focus,
                "input_sha256": _tensor_sha256(torch, residual),
                "carried_pre_FP32_bits": _evidence_bits(torch, pre),
                "arithmetic": arithmetic,
                "native_collapse_and_inverse": "UNKNOWN",
            }
            if arithmetic == "source_ordered_fmaf" and fmaf is None:
                controls.append({**entry, "status": "UNKNOWN_NO_EXACT_FMAF"})
                continue
            if arithmetic == "source_ordered_fmaf":
                values = residual.float().tolist()
                coefficients = pre.tolist()
                accumulator = [0.0] * HIDDEN
                for lane in range(HC):
                    accumulator = [
                        fmaf(values[lane][i], coefficients[lane], accumulator[i])
                        for i in range(HIDDEN)
                    ]
                collapse = torch.tensor(accumulator, dtype=torch.float32)
            else:
                collapse = torch.zeros(HIDDEN, dtype=torch.float32)
                for lane in range(HC):
                    product = residual[lane].float() * pre[lane].float()
                    collapse = collapse + product
            landed = collapse.to(torch.bfloat16)
            norm_input = landed.float()
            square = norm_input.square()
            square_sum, mean = square.sum(), square.mean()
            inverse = torch.rsqrt(mean + payload["scalars"]["rmsnorm_eps"])
            normalized = (norm_input * inverse * weight.float()).to(torch.bfloat16)
            entry.update({
                "status": "RECORDED_EXPLANATORY_ONLY",
                "collapse_before_BF16_FP32_bits": _evidence_bits(torch, collapse),
                "collapse_after_BF16_bits": _evidence_bits(torch, landed),
                "collapse_after_BF16_sha256": _tensor_sha256(torch, landed),
                "norm_policy": (
                    "one Torch CPU FP32 square/mean/rsqrt/multiply; BF16 output"
                ),
                "reduction_tree": (
                    "Torch CPU implementation, not an observed native tree"
                ),
                "square_sum_FP32_bits": _evidence_bits(torch, square_sum),
                "mean_FP32_bits": _evidence_bits(torch, mean),
                "inverse_FP32_bits": _evidence_bits(torch, inverse),
                "norm_applications": 1,
                "focus_output_BF16_bits": _evidence_bits(torch, normalized[focus]),
                "focus_native_BF16_bits": _evidence_bits(
                    torch, payload["actual"]["y_bf16"][row, focus]
                ),
                "focus_recorded_oracle_BF16_bits": _evidence_bits(
                    torch, payload["expected"]["y_bf16"][row, focus]
                ),
                "native_y_comparison": _evidence_match(
                    torch, normalized, payload["actual"]["y_bf16"][row]
                ),
                "recorded_oracle_y_comparison": _evidence_match(
                    torch, normalized, payload["expected"]["y_bf16"][row]
                ),
            })
            controls.append(entry)
    return controls


def _evidence_coefficients(torch, values, row=0):
    return sum(
        (_evidence_bits(torch, values[name][row]) for name in OUTPUTS[1:4]), []
    )


def _evidence_zero(torch, payload):
    observed = payload["actual"]["new_residual"][0]
    oracle = payload["expected"]["new_residual"][0]
    signs = {}
    for label, value in (("native", observed), ("recorded_oracle", oracle)):
        bits = _evidence_bits(torch, value)
        signs[label] = {
            "positive_zero": bits.count(0), "negative_zero": bits.count(0x8000),
            "other_raw_values": len(bits) - bits.count(0) - bits.count(0x8000),
            "sha256": _tensor_sha256(torch, value),
        }
    base = payload["inputs"]["mix_bases"]
    eps = payload["scalars"]["sinkhorn_eps"]
    pre = base[:HC].sigmoid() + payload["scalars"]["hc_pre_eps"]
    post = base[HC:2 * HC].sigmoid() * payload["scalars"]["hc_post_scale"]
    comb = base[2 * HC:].reshape(HC, HC).softmax(-1)
    stages = [{"stage": "Torch_CPU_softmax", "FP32_bits": _evidence_bits(torch, comb)}]
    comb = comb + eps
    stages.append({
        "stage": "softmax_plus_eps", "FP32_bits": _evidence_bits(torch, comb)
    })
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    stages.append({
        "stage": "first_column_norm", "FP32_bits": _evidence_bits(torch, comb)
    })
    for iteration in range(payload["scalars"]["num_sinkhorn_iters"] - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        stages.append({"iteration": iteration + 1, "stage": "row_norm",
                       "FP32_bits": _evidence_bits(torch, comb)})
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        stages.append({"iteration": iteration + 1, "stage": "column_norm",
                       "FP32_bits": _evidence_bits(torch, comb)})
    comparison = _evidence_match(torch, observed, oracle)
    for difference in comparison["first_raw_differences"]:
        difference["lane"] = difference["flat_index"] // HIDDEN
        difference["column"] = difference["flat_index"] % HIDDEN
    return {
        "original_row": 0,
        "raw_uint16_sign_counts": signs,
        "zero_residual_comparison": comparison,
        "bases_FP32_bits": _evidence_bits(torch, base),
        "scales_FP32_bits": _evidence_bits(torch, payload["inputs"]["mix_scales"]),
        "bias_only_policy": "named Torch CPU sigmoid/softmax/reductions; FP32",
        "control_pre_FP32_bits": _evidence_bits(torch, pre),
        "control_post_FP32_bits": _evidence_bits(torch, post),
        "comb_normalization_boundaries": stages,
        "native_final_24_FP32_bits": _evidence_coefficients(torch, payload["actual"]),
        "recorded_oracle_24_FP32_bits": _evidence_coefficients(
            torch, payload["expected"]
        ),
        "native_sgmdf_rcpf_logits_and_intermediates": "UNKNOWN",
        "causal_conclusion": "NONE; intrinsic presence is not a cause observation",
    }


def _evidence_backend_metadata():
    try:
        module = importlib.import_module("deep_gemm")
        native = sys.modules.get("deep_gemm.deep_gemm_cpp")
        result = {
            "module": _file_record(vars(module).get("__file__")),
            "extension": _file_record(vars(native).get("__file__") if native else None),
            "import_native_side_effects": "UNKNOWN; no Mega-mHC invocation requested",
            "active_native_specialization_partials_inverse": "UNKNOWN",
            "source_split_flags": "DERIVED only if evaluated; not inferred here",
        }
        getter = vars(module).get("get_num_sms")
        result["effective_num_sms"] = getter() if callable(getter) else "UNKNOWN"
        package = Path(module.__file__).parent
        result["installed_header_JIT_identity"] = {
            name: _file_record(package / name)
            for name in (
                "include/deep_gemm/impls/ppu_mega_mhc_bf16.cuh",
                "jit/compiler.py", "jit/runtime.py",
            )
        }
        result["active_JIT_specialization"] = "UNKNOWN; no cache probe or kernel replay"
        return result
    except Exception as error:
        return {"status": "UNKNOWN", "reason": _error(error)}


def _evidence_row_shapes(torch, payloads, receipts):
    ids = [_EVIDENCE_SPECS[i][0] for i in (0, 2, 3)]
    baseline = receipts[ids[0]]
    original = {}
    for case_id in ids:
        receipt, payload = receipts[case_id], payloads[case_id]
        identical = (
            receipt["row0_and_weight_hashes"] == baseline["row0_and_weight_hashes"]
            and receipt["scalars"] == baseline["scalars"]
        )
        if not identical:
            raise ValueError(
                "original row-shape inputs/weights/scalars are not identical"
            )
        original[case_id] = {
            "row0_and_weight_hashes": receipt["row0_and_weight_hashes"],
            "scalars": receipt["scalars"], "identical_to_M1": identical,
            "native_24_FP32_bits": _evidence_coefficients(torch, payload["actual"]),
            "recorded_oracle_24_FP32_bits": _evidence_coefficients(
                torch, payload["expected"]
            ),
        }
    source = payloads[ids[0]]
    controls = []
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        for device in (torch.device("cpu"), torch.device("cuda:0")):
            for rows in (1, 64):
                inputs = {
                    name: (
                        value[:1].expand((rows,) + tuple(value.shape[1:])).contiguous()
                        if name in ROW_INPUTS else value
                    ).to(device)
                    for name, value in source["inputs"].items()
                }
                reference = independent_transition_reference(
                    torch, inputs, source["scalars"]
                )
                host = {name: value[:1].cpu() for name, value in reference.items()}
                controls.append({
                    "device": str(device), "rows": rows, "allow_tf32": False,
                    "backend": "Torch eager IEEE reference; no candidate calls",
                    "input_scope": (
                        "saved immutable M1 row0 repeated; NOT original M64 batch"
                    ),
                    "row0_and_weight_hashes": _row_inputs_identity(torch, inputs),
                    "reference_chunk_rows": CHUNK_ROWS,
                    "control_24_FP32_bits": _evidence_coefficients(torch, host),
                    "row0_output_hashes": {
                        name: _tensor_sha256(torch, value)
                        for name, value in host.items()
                    },
                    "interpretation": (
                        "new shape control, not an original oracle or kernel PASS"
                    ),
                })
                del inputs, reference, host
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return {"original_evidence": original, "new_reference_controls": controls,
            "allow_tf32_saved_value": previous, "allow_tf32_restored": True}


def run_evidence_diagnostic(root, report, archive_path):
    """Exactly five trusted payloads; explanatory controls, zero Mega dispatch."""
    diagnostic = {
        "status": "COLLECTING", "new_Mega_mHC_dispatches": 0,
        "primary_correctness": "NOT_RUN",
        "original_mhc01": "26 strict FAIL per P/D unchanged",
        "promotion": "NOT_REQUESTED", "Graph_performance_full_model": "NOT_RUN",
        "native_unobserved_partials_logits_inverse_activation": "UNKNOWN",
    }
    report["evidence_diagnostic"] = diagnostic
    try:
        report["phase"] = "evidence_archive_identity"
        raw, diagnostic["archive_identity"] = _evidence_archive(archive_path)
        report["phase"] = "evidence_torch_import"
        torch = importlib.import_module("torch")
        diagnostic["torch_module"] = _file_record(torch.__file__)
        report["phase"] = "evidence_safe_PT_validation"
        payloads, diagnostic["payload_identity"] = _evidence_payloads(torch, raw)
        del raw
        report["phase"] = "evidence_control_device"
        properties = torch.cuda.get_device_properties(torch.device("cuda:0"))
        diagnostic["reference_control_device"] = {
            "name": properties.name,
            "major": properties.major,
            "minor": properties.minor,
        }
        if (
            "M890" not in properties.name
            or (properties.major, properties.minor) != (8, 9)
        ):
            raise RuntimeError(
                "evidence PPU controls require the approved real M890 device"
            )
        fmaf, diagnostic["exact_fmaf_primitive"] = _evidence_fmaf()
        report["phase"] = "evidence_residual_order"
        diagnostic["residual_order"] = _evidence_residual_order(
            torch, payloads[_EVIDENCE_SPECS[1][0]], fmaf
        )
        report["phase"] = "evidence_collapse_norm"
        diagnostic["collapse_norm"] = _evidence_collapse_norm(
            torch, payloads[_EVIDENCE_SPECS[2][0]], fmaf
        )
        report["phase"] = "evidence_bias_only_zero"
        diagnostic["bias_only_zero"] = _evidence_zero(
            torch, payloads[_EVIDENCE_SPECS[4][0]]
        )
        report["phase"] = "evidence_row_shape_coefficients"
        diagnostic["row_shape_coefficients"] = _evidence_row_shapes(
            torch, payloads, diagnostic["payload_identity"]
        )
        diagnostic["native_metadata"] = _evidence_backend_metadata()
        diagnostic["status"] = "COLLECTED_EXPLANATORY_ONLY"
        diagnostic["acceptance"] = (
            "control matches do not replace the original strict FAIL"
        )
        report["status"] = "COLLECTED_EXPLANATORY_ONLY"
        report["phase"] = "evidence_complete"
        _write_json(root, "mhc-evidence-controls.json", diagnostic)
    except Exception:
        diagnostic["status"] = "FAIL"
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("surface", "mhc-mega"), required=True)
    parser.add_argument("--gamma", type=int, default=os.environ.get("RTP_V41_S5_GAMMA"))
    parser.add_argument(
        "--query-width", type=int, default=os.environ.get("RTP_V41_S5_QUERY_WIDTH")
    )
    parser.add_argument("--evidence-zip")
    parser.add_argument("--qualify-numerical", action="store_true")
    args = parser.parse_args()
    if args.evidence_zip is not None and args.mode != "mhc-mega":
        parser.error("--evidence-zip is allowed only with --mode mhc-mega")
    if args.qualify_numerical and (
        args.mode != "mhc-mega" or args.evidence_zip is not None
    ):
        parser.error("--qualify-numerical requires mhc-mega without --evidence-zip")
    output = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if not output:
        raise ValueError(
            "TEST_UNDECLARED_OUTPUTS_DIR is required; no other write destination"
        )
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": SCHEMA,
        "mode": args.mode,
        "status": "STARTED",
        "correctness": "NOT_RUN",
        "strict_correctness": "NOT_RUN",
        "shape_proof": "NOT_RUN",
        "capability": "UNKNOWN",
        "capability_reason": "preflight not completed",
        "execution": "NOT_RUN",
        "safety": "NOT_RUN",
        "matrix_complete": False,
        "phase": "startup",
        "failure_stage": None,
        "failure_kind": None,
        "performance": "NOT_RUN",
        "graph": "NOT_RUN",
        "promotion": "NOT_REQUESTED",
        "fixed_primary_source": {
            "DeepGEMM": DEEPGEMM_SOURCE,
            "FlashMLA": FLASHMLA_SOURCE,
        },
        "cases": [],
    }
    filename = (
        "mhc-qualified04.json" if args.qualify_numerical else
        "kernel-surface.json" if args.mode == "surface" else "mhc-mega.json"
    )
    if args.qualify_numerical:
        report["qualified_status"] = "NOT_RUN"
        report["qualification_profile"] = {
            "name": _QUALIFICATION_PROFILE,
            "contract_sha256": _QUALIFICATION_CONTRACT_SHA256,
            "limit_strictly_less_than": _QUALIFICATION_LIMIT,
            "upstream_source": DEEPGEMM_SOURCE,
            "threshold_source": "tests/test_mega_mhc.py:145-153",
            "metric_source": "deep_gemm/testing/numeric.py:5-11",
            "stable_equivalent": "FP64 SSE/energy; no cancellation or clipping",
            "strict_history": "unchanged diagnostics; old mhc01 FAIL is not rewritten",
        }
    _write_json(root, filename, report)
    try:
        report["phase"] = "snapshot"
        report["snapshot"] = _snapshot_identity()
        if args.evidence_zip is not None:
            run_evidence_diagnostic(root, report, args.evidence_zip)
            _write_json(root, filename, report)
            print(json.dumps({
                "status": report["status"], "report": filename,
                "primary_correctness": "NOT_RUN", "new_Mega_mHC_dispatches": 0,
            }))
            return 0
        report["phase"] = "imports"
        if args.mode == "surface":
            report["surface"] = collect_surface()
        else:
            report["surface"] = collect_surface(
                guard_imports=False,
                module_names=("torch", "deep_gemm", "deep_gemm.deep_gemm_cpp"),
            )
        _write_json(root, filename, report)
        if args.mode == "surface":
            if report["surface"]["surface_status"] == "UNAVAILABLE":
                report["capability"] = "UNAVAILABLE"
            report["capability_reason"] = (
                "surface/import evidence only; no device/shape qualification"
            )
            report["status"] = (
                "FAIL"
                if report["surface"]["surface_policy_violated"]
                else "COLLECTED_SURFACE_ONLY"
            )
            if report["status"] == "FAIL":
                report["failure_stage"] = "surface_import_policy"
                report["failure_kind"] = "surface_policy"
            report["phase"] = "complete"
        else:
            run_shape_probe(
                root, report, args.gamma, args.query_width,
                qualification=args.qualify_numerical,
            )
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = _error(error)
        if report["failure_stage"] is None:
            report["failure_stage"] = report["phase"]
            report["failure_kind"] = (
                "evidence"
                if any(case["dispatch_calls"] for case in report["cases"])
                else "pre_dispatch"
            )
        if args.mode == "mhc-mega":
            _update_outcomes(report, qualification=args.qualify_numerical)
        _write_json(root, filename, report)
        print(
            json.dumps(
                {"status": "FAIL", "report": filename, "error": report["error"]}
            ),
            file=sys.stderr,
        )
        return 1
    _write_json(root, filename, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": filename,
                "correctness": report["correctness"],
                "performance": "NOT_RUN",
            }
        )
    )
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
