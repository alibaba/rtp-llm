"""Validate captures or replay CUDA operators through statically registered contracts.

Schema3 dispatches by (contract.name, contract.version). Registration is explicit
Python code; capture metadata never names a module to import or a DAG to execute.
Unknown contracts remain loadable but cannot be validated or replayed here.
Legacy schema1 TopK/schema2 MQA+TopK delegate to the Indexer adapter unchanged.

    /opt/conda310/bin/python3 -m rtp_llm.utils.tensor_capture_replay MANIFEST
    /opt/conda310/bin/python3 -m rtp_llm.utils.tensor_capture_replay MANIFEST --device cuda

CPU mode checks the current source contract, NOT execution of the captured kernel.
This is the stable-index contract, not a promise about arbitrary older binaries:
DSV4 insertion compares actual token indices on equal scores; radix embeds the
same index priority in its key. Legacy insertion used candidate arrival slots.
A mismatch requires checking the capture/loaded library provenance; the loader
cannot establish a binary's source revision from an operator name or timestamp.
Both backends select by descending FP32 bit-order key, then ascending relative
index; long-row output order is unspecified. Short rows emit 0..length-1, -1.
V3's effective candidate length is min(row_end-row_start, max_seq_len).
Signed zeros are distinct. Valid-domain NaNs are explicitly unsupported: their
histogram/backend semantics are not a numeric maximum-set contract. Infinities
are supported. Invalid-domain snapshots must contain the capture protocol's zeros.

Schema2 CPU mode validates files, shapes, metadata and the captured TopK contract;
it does not evaluate MQA. CUDA calls DeepGEMM with the saved, already query-scaled
weights, then checks both independent and chained TopK. --mqa-atol/--mqa-rtol
(default 0) apply only to valid-domain logits, never to index-set comparisons.
Neither schema attests a source revision for the loaded native libraries.

Schema3 Indexer version2 validates indexed CP rows against frozen global requests
using the capture adapter's shared mapping validator. Global query IDs are CPU
validation metadata, not native operator arguments. CUDA restores the saved local
query/full key inputs and runs the same fixed kernels, never CP collectives.

Schema3 Indexer version3 validates single-token paged decode. Its TopK kernels
select descending FP32 keys, but boundary ties and long-row order are unspecified.
CUDA remaps captured physical pages to a compact pool, never allocating by the
original pool size. Schedules are reused only for the same verified DeepGEMM ABI,
device capability and SM count; otherwise a verified scheduler rebuilds them.

Reading requires no capture authorization configuration and sends no data out.
The loader verifies checksums, not authenticity: coordinated metadata/payload
rewrites cannot be detected. JSON summaries omit tensors and identities; joint
summaries include aggregate numerical errors and a bounded list of score thresholds.
Exit codes: 0 validated, 1 invalid/failed, 2 unsupported, 3 CUDA unavailable.
"""

import argparse
import importlib
import json
import math
import re
import sys

import torch

from rtp_llm.utils import tensor_capture_indexer
from rtp_llm.utils.tensor_capture import load_capture


class UnsupportedCapture(ValueError):
    """The artifact cannot be checked/replayed under the supported contract."""


class ReplayUnavailable(RuntimeError):
    """CUDA or the real operator library is unavailable (not a passing replay)."""


class ReplayMismatch(ValueError):
    """A completed joint replay failed; retain measurements for CLI consumers."""

    def __init__(self, summary):
        super().__init__("MQA/TopK replay mismatch; see comparison summary")
        self.summary = summary


_MAX_STORAGE_BYTES = 1024**3
_MQA_TENSORS = {"q_fp8", "k_fp8", "k_scale", "weights"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**31 - 1


def _storage_elements(desc):
    # A conservative nonoverlap proof admits dense, transposed and padded views.
    # Exotic interleaved layouts not proven by this rule are explicitly refused.
    shape, strides = desc["shape"], desc["original_stride"]
    span = 1
    for stride, size in sorted(zip(strides, shape)):
        if size > 1:
            if stride < span:
                raise UnsupportedCapture("overlapping or unsupported source strides")
            span += (size - 1) * stride
    return desc["storage_offset"] + span


def _contiguous(shape, strides):
    expected = 1
    for size, stride in reversed(list(zip(shape, strides))):
        if size > 1 and stride != expected:
            return False
        expected *= max(size, 1)
    return True


def _is_paged(manifest):
    return (
        manifest.get("schema_version") == 3
        and manifest.get("contract", {}).get("version") == 3
    )


def _row_bounds(tensors, row):
    if "lengths" in tensors:
        return 0, int(tensors["lengths"][row])
    return int(tensors["row_starts"][row]), int(tensors["row_ends"][row])


def _check_paged_schema(manifest, tensors):
    tensor_capture_indexer.validate_paged_capture(manifest, tensors)
    # Source offsets/strides and workspace are untrusted even when logical files
    # are small. Physical page IDs are metadata, never allocation dimensions.
    storage_bytes = manifest["parameters"]["workspace_bytes"]
    for name, tensor in tensors.items():
        storage_bytes += (
            _storage_elements(manifest["tensors"][name]) * tensor.element_size()
        )
    if storage_bytes > _MAX_STORAGE_BYTES:
        raise UnsupportedCapture("reconstructed source storage exceeds 1 GiB limit")
    for row, length in enumerate(tensors["lengths"].tolist()):
        if bool(torch.isnan(tensors["logits"][row, :length]).any()):
            raise UnsupportedCapture(
                "valid-domain NaN: paged TopK semantics unsupported by this validator"
            )
    if manifest["op"] == "indexer.mqa_topk":
        inputs = [tensors["q_fp8"], tensors["weights"]]
        block = manifest["cache_layout"]["block_size"]
        dimension = manifest["cache_layout"]["head_dim"]
        # Packed bytes are payload-then-scales per page, NOT interleaved tokens.
        for page, count in zip(
            tensors["packed_kv_pages"], tensors["page_valid_tokens"].tolist()
        ):
            blob = page.view(-1)
            inputs.extend(
                (
                    blob[: count * dimension].view(torch.float8_e4m3fn),
                    blob[block * dimension : block * dimension + count * 4].view(
                        torch.float32
                    ),
                )
            )
        for tensor in inputs:
            for chunk in tensor.reshape(-1).split(1024 * 1024):
                if not bool(torch.isfinite(chunk.float()).all()):
                    raise UnsupportedCapture("nonfinite paged MQA input unsupported")
    return manifest["parameters"]


def _check_schema(manifest, tensors):
    if _is_paged(manifest):
        return _check_paged_schema(manifest, tensors)
    joint = manifest.get("op") == "indexer.mqa_topk"
    indexed = manifest.get("schema_version") == 3
    _require(manifest.get("schema_version") in (1, 2, 3), "unsupported schema_version")
    if indexed:
        _require(
            manifest.get("contract") == dict(name=manifest.get("op"), version=2),
            "unsupported Indexer contract",
        )
        inputs = {"row_starts", "row_ends", "query_global_ids"} | (
            _MQA_TENSORS if joint else {"logits"}
        )
        outputs = {"logits", "indices"} if joint else {"indices"}
        _require(
            set(manifest.get("inputs", ())) == inputs
            and set(manifest.get("outputs", ())) == outputs,
            "invalid indexed named tensor interface",
        )
        if not joint:
            _require(
                "operations" not in manifest and "mqa_parameters" not in manifest,
                "unexpected joint metadata for TopK",
            )
    else:
        _require(
            manifest["schema_version"] == (2 if joint else 1),
            "invalid legacy Indexer schema/op",
        )
    if "server_role" in manifest:
        _require(
            manifest["server_role"] in ("PREFILL", "PDFUSION"),
            "unsupported server_role; DECODE capture unsupported",
        )
    for field, expected in (
        ("op", "indexer.mqa_topk" if joint else "indexer.topk"),
        ("phase", "prefill"),
        ("invalid_logits", "zero_filled_snapshot_only"),
        ("nonfinite_policy", "valid_domain_bits_preserved_backend_semantics"),
        ("output_contract", "raw_relative_indices_minus_one_padding_backend_order"),
    ):
        _require(manifest.get(field) == expected, "unsupported " + field)
    _require(
        manifest.get("finite_values_checked") is False, "invalid finite_values_checked"
    )
    backend = manifest.get("backend")
    if backend not in ("dsv4_per_row", "topk_v3_tie_break"):
        error = ReplayUnavailable if joint else UnsupportedCapture
        raise error("unsupported backend: " + str(backend))
    names = {"logits", "row_starts", "row_ends", "indices"}
    if joint:
        names |= _MQA_TENSORS
    if indexed:
        names.add("query_global_ids")
    _require(set(tensors) == names, "unexpected/missing capture tensors")
    logits = tensors["logits"]
    _require(
        logits.dtype == torch.float32 and logits.ndim == 2,
        "logits must be float32 [rows, columns]",
    )
    rows, columns = logits.shape
    _require(rows > 0 and columns > 0, "capture requires nonempty logits dimensions")
    if joint:
        _check_mqa_schema(manifest, tensors, rows, columns)
    params = manifest.get("parameters")
    _require(isinstance(params, dict), "missing backend parameters")
    common = {"k", "index_space", "stride0", "stride1"}
    expected = common | (
        {"num_rows", "force_radix_sort"}
        if backend == "dsv4_per_row"
        else {"max_seq_len"}
    )
    _require(set(params) == expected, "missing/unknown backend parameters")
    k = params["k"]
    _require(_integer(k, 1), "k must be a positive int32")
    _require(params["index_space"] == "relative_to_row_start", "invalid index_space")
    for name in ("row_starts", "row_ends", "indices"):
        tensor = tensors[name]
        expected_shape = (rows, k) if name == "indices" else (rows,)
        _require(
            tensor.dtype == torch.int32 and tuple(tensor.shape) == expected_shape,
            name + " must be int32 with shape " + str(expected_shape),
        )
    if indexed:
        # The capture adapter owns the frozen request/prefix mapping, including
        # indexed IDs and CP coverage. Never reinterpret it as a contiguous chunk.
        tensor_capture_indexer.validate_query_mapping(manifest, tensors, manifest)
    else:
        for name in ("query_row_start", "query_row_count", "total_query_rows"):
            _require(_integer(manifest.get(name)), "invalid " + name)
        _require(
            manifest["query_row_count"] == rows
            and manifest["query_row_start"] + rows <= manifest["total_query_rows"],
            "inconsistent query chunk",
        )
    desc = manifest["tensors"]["logits"]
    for axis in (0, 1):
        key = "stride" + str(axis)
        _require(
            _integer(params[key]) and params[key] == desc["original_stride"][axis],
            "parameter/source " + key + " mismatch",
        )
    if backend == "dsv4_per_row":
        _require(
            _integer(params["num_rows"]) and params["num_rows"] == rows,
            "num_rows mismatch",
        )
        _require(
            type(params["force_radix_sort"]) is bool, "force_radix_sort must be boolean"
        )
    else:
        if k not in (512, 1024, 2048):
            raise UnsupportedCapture("topk_v3_tie_break supports K=512/1024/2048")
        _require(
            _integer(params["max_seq_len"], 1) and params["max_seq_len"] <= columns,
            "invalid max_seq_len",
        )
        if params["stride1"] != 1:
            raise UnsupportedCapture("topk_v3_tie_break requires source stride1=1")
    storage_bytes = 0
    for name, tensor in tensors.items():
        desc = manifest["tensors"][name]
        storage_bytes += _storage_elements(desc) * tensor.element_size()
        if name in ("row_starts", "row_ends", "indices") and not _contiguous(
            desc["shape"], desc["original_stride"]
        ):
            raise UnsupportedCapture(
                name + " must have contiguous source layout for these kernels"
            )
    if storage_bytes > _MAX_STORAGE_BYTES:
        raise UnsupportedCapture("reconstructed source storage exceeds 1 GiB limit")
    starts, ends = tensors["row_starts"], tensors["row_ends"]
    _require(
        bool(((starts >= 0) & (ends >= starts) & (ends <= columns)).all()),
        "invalid row bounds",
    )
    for row in range(rows):
        start, end = int(starts[row]), int(ends[row])
        _require(
            bool((logits[row, :start] == 0).all())
            and bool((logits[row, end:] == 0).all()),
            "invalid-domain logits are not zero-filled",
        )
        if bool(torch.isnan(logits[row, start:end]).any()):
            raise UnsupportedCapture(
                "valid-domain NaN: backend histogram semantics unsupported by this validator"
            )
    return params


def _check_mqa_schema(manifest, tensors, rows, columns):
    expected_operations = [
        dict(
            op="indexer.mqa_logits",
            inputs=["q_fp8", "k_fp8", "k_scale", "weights", "row_starts", "row_ends"],
            outputs=["logits"],
        ),
        dict(
            op="indexer.topk",
            inputs=["logits", "row_starts", "row_ends"],
            outputs=["indices"],
        ),
    ]
    _require(
        manifest.get("operations") == expected_operations,
        "invalid MQA/TopK operation wiring",
    )
    params = manifest.get("mqa_parameters")
    _require(isinstance(params, dict), "missing mqa_parameters")
    if params.get("backend") != "deep_gemm.fp8_mqa_logits":
        raise ReplayUnavailable(
            "unsupported MQA backend: " + str(params.get("backend"))
        )
    _require(
        set(params)
        == {
            "backend",
            "clean_logits",
            "weights_include_query_scale",
            "key_scale_format",
        },
        "missing/unknown MQA parameters",
    )
    _require(params["clean_logits"] is False, "MQA requires clean_logits=false")
    _require(
        params["weights_include_query_scale"] is True,
        "MQA weights must already include query scale",
    )
    _require(
        params["key_scale_format"] == "float32_per_128_channels",
        "unsupported key_scale_format",
    )
    q, k, scale, weights = (
        tensors[name] for name in ("q_fp8", "k_fp8", "k_scale", "weights")
    )
    _require(
        q.dtype == torch.float8_e4m3fn and q.ndim == 3 and q.shape[0] == rows,
        "q_fp8 must be e4m3fn [Q,H,D]",
    )
    heads, dimension = q.shape[1:]
    _require(
        heads > 0 and dimension > 0 and dimension % 128 == 0,
        "invalid MQA H/D (D must be a multiple of 128)",
    )
    _require(
        k.dtype == torch.float8_e4m3fn and tuple(k.shape) == (columns, dimension),
        "k_fp8 must be e4m3fn [N,D]",
    )
    scale_shape = (columns,) if dimension == 128 else (columns, dimension // 128)
    _require(
        scale.dtype == torch.float32 and tuple(scale.shape) == scale_shape,
        "k_scale must be float32 with shape " + str(scale_shape),
    )
    _require(
        weights.dtype == torch.float32 and tuple(weights.shape) == (rows, heads),
        "weights must be float32 [Q,H]",
    )
    for name in sorted(_MQA_TENSORS):
        # Bound the temporary FP32 expansion of large FP8 payloads.
        for chunk in tensors[name].reshape(-1).split(1024 * 1024):
            if not bool(torch.isfinite(chunk.float()).all()):
                raise UnsupportedCapture(
                    name + ": nonfinite MQA inputs (including NaN) unsupported"
                )


def _score_keys(values):
    # Compare exact FP32 keys, including signed zero, without floating tolerance.
    bits = values.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where(
        (bits & 0x80000000) != 0, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000
    )


def _check_indices(manifest, tensors, indices):
    params = manifest["parameters"]
    k = params["k"]
    counts = {
        "rows": indices.shape[0],
        "empty_rows": 0,
        "short_rows": 0,
        "selected_indices": 0,
        "padding_indices": 0,
        "boundary_tie_rows": 0,
    }
    _require(
        indices.dtype == torch.int32 and indices.shape == tensors["indices"].shape,
        "invalid TopK output dtype/shape",
    )
    for row, output in enumerate(indices):
        start, end = _row_bounds(tensors, row)
        length = end - start
        if manifest["backend"] == "topk_v3_tie_break":
            length = min(length, params["max_seq_len"])
        count = min(k, length)
        selected = output[:count].to(torch.int64)
        prefix = "row %d: " % row
        _require(
            bool(((selected >= 0) & (selected < length)).all()),
            prefix + "missing/out-of-range index",
        )
        _require(bool((output[count:] == -1).all()), prefix + "invalid -1 padding")
        _require(selected.unique().numel() == count, prefix + "duplicate index")
        if length <= k:
            _require(
                torch.equal(selected, torch.arange(length)),
                prefix + "short-row indices must be sequential",
            )
        else:
            keys = _score_keys(tensors["logits"][row, start : start + length])
            selected_keys = keys[selected]
            threshold = selected_keys.min()
            # No unselected candidate may outrank the weakest selected candidate.
            # Equal keys use relative index, NOT torch.topk's arbitrary tie order.
            last_tie = selected[selected_keys == threshold].max()
            chosen = torch.zeros(length, dtype=torch.bool)
            chosen[selected] = True
            relative = torch.arange(length)
            _require(
                not bool(((keys > threshold) & ~chosen).any()),
                prefix + "not the maximal candidate set (score)",
            )
            better_tie = (keys == threshold) & (relative < last_tie)
            _require(
                _is_paged(manifest) or not bool((better_tie & ~chosen).any()),
                prefix
                + "not the maximal candidate set (index tie-break under current-source contract; "
                "verify capture/loaded library provenance)",
            )
            counts["boundary_tie_rows"] += int(int((keys == threshold).sum()) > 1)
        counts["empty_rows"] += int(length == 0)
        counts["short_rows"] += int(length < k)
        counts["selected_indices"] += count
        counts["padding_indices"] += k - count
    return counts


def _restore(tensor, desc, device):
    storage = torch.empty(_storage_elements(desc), dtype=tensor.dtype, device=device)
    result = storage.as_strided(
        desc["shape"], desc["original_stride"], desc["storage_offset"]
    )
    result.copy_(tensor)
    return result


def _cuda_operator(backend):
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise ReplayUnavailable("CUDA unavailable; no kernel replay performed")
    name = "dsv4_top_k_per_row_prefill" if backend == "dsv4_per_row" else backend
    try:
        ops = importlib.import_module("rtp_llm.ops").rtp_llm_ops
        operator = getattr(ops, name)
        if not callable(operator):
            raise AttributeError("operator is not callable")
    except Exception as error:
        raise ReplayUnavailable(
            "real CUDA operator unavailable: " + str(error)
        ) from error
    return operator


def _mqa_operator():
    try:
        module = importlib.import_module("deep_gemm")
        operator = getattr(module, "fp8_mqa_logits")
        if not callable(operator):
            raise AttributeError("fp8_mqa_logits is not callable")
    except Exception as error:
        raise ReplayUnavailable(
            "real DeepGEMM MQA unavailable: " + str(error)
        ) from error
    return operator


def _run_topk(operator, manifest, restored):
    params = manifest["parameters"]
    output = restored["indices"]
    output.fill_(-2)
    if _is_paged(manifest):
        operator(
            restored["logits"],
            restored["lengths"],
            output,
            restored["workspace"],
            params["k"],
            params["max_seq_len"],
        )
        torch.cuda.synchronize()
        return output.to(device="cpu", copy=True)
    args = (restored["logits"], restored["row_starts"], restored["row_ends"], output)
    if manifest["backend"] == "dsv4_per_row":
        operator(
            *args,
            params["num_rows"],
            params["stride0"],
            params["stride1"],
            params["k"],
            params["force_radix_sort"]
        )
    else:
        operator(*args, params["k"], params["max_seq_len"])
    torch.cuda.synchronize()
    return output.to(device="cpu", copy=True)


def _paged_score_sets_equal(tensors, actual):
    """Tied keys may select different tokens; tolerances never apply to TopK."""
    if actual.dtype != torch.int32 or actual.shape != tensors["indices"].shape:
        return False
    for row, length in enumerate(tensors["lengths"].tolist()):
        count = min(length, actual.shape[1])
        selected = actual[row, :count].long()
        if not bool(((selected >= 0) & (selected < length)).all()):
            return False
        keys = _score_keys(tensors["logits"][row, :length])
        captured = tensors["indices"][row, :count].long()
        if not torch.equal(keys[selected].sort().values, keys[captured].sort().values):
            return False
    return True


def _restore_paged(manifest, tensors, device):
    names = {"logits", "lengths", "indices"}
    joint = manifest["op"] == "indexer.mqa_topk"
    if joint:
        names |= {"q_fp8", "weights", "kvlen_2d"}
    restored = {
        name: _restore(tensors[name], manifest["tensors"][name], device)
        for name in names
    }
    restored["workspace"] = torch.empty(
        manifest["parameters"]["workspace_bytes"], dtype=torch.uint8, device=device
    )
    if joint:
        ids = [page for page in tensors["page_ids"].tolist() if page >= 0]
        slots = {page: slot for slot, page in enumerate(ids)}
        # O(referenced table size), independent of the largest physical page ID.
        table = torch.tensor(
            [
                [slots[page] if page >= 0 else -1 for page in row]
                for row in tensors["block_table"].tolist()
            ],
            dtype=torch.int32,
        )
        restored["block_table"] = _restore(
            table, manifest["tensors"]["block_table"], device
        )
        restored["packed_kv_pages"] = tensors["packed_kv_pages"][: len(ids)].to(
            device=device, copy=True
        )
    return restored


def _call_paged_mqa(manifest, tensors, restored):
    try:
        module = importlib.import_module("deep_gemm")
        mqa = module.fp8_paged_mqa_logits
        scheduler = module.get_paged_mqa_logits_metadata
        get_num_sms = module.get_num_sms
        if not all(callable(fn) for fn in (mqa, scheduler, get_num_sms)):
            raise AttributeError("paged MQA/scheduler is not callable")
    except Exception as error:
        raise ReplayUnavailable(
            "real DeepGEMM paged MQA unavailable: " + str(error)
        ) from error
    try:
        version = tensor_capture_indexer.paged_schedule_version(module)
    except ValueError as error:
        raise UnsupportedCapture(
            "unverified runtime paged schedule ABI: " + str(error)
        ) from error
    device = restored["q_fp8"].device
    environment = dict(
        deep_gemm_version=version,
        device_capability=list(torch.cuda.get_device_capability(device)),
        num_sms=get_num_sms(),
    )
    block = manifest["cache_layout"]["block_size"]
    if (
        type(environment["num_sms"]) is not int
        or not 1 <= environment["num_sms"] <= 4096
    ):
        raise UnsupportedCapture("invalid runtime paged MQA SM count")
    try:
        # Validate architecture/block/head support BEFORE invoking a native scheduler.
        probe = torch.empty((environment["num_sms"] + 1, 2), dtype=torch.int32)
        tensor_capture_indexer._validate_paged_schedule(environment, probe, block)
        tensor_capture_indexer._validate_paged_heads(
            restored["q_fp8"].shape[2], environment
        )
    except ValueError as error:
        raise UnsupportedCapture(
            "unsupported runtime paged MQA environment: " + str(error)
        ) from error
    reused = environment == manifest["schedule_environment"]
    if reused:
        schedule = tensors["schedule_metadata"].to(device=device, copy=True)
    else:
        schedule = scheduler(restored["kvlen_2d"], block, environment["num_sms"])
        _require(
            isinstance(schedule, torch.Tensor) and schedule.device == device,
            "paged scheduler returned invalid device/tensor",
        )
        tensor_capture_indexer._validate_paged_schedule(environment, schedule, block)
        tensor_capture_indexer._validate_paged_schedule_values(
            schedule.detach().cpu(), tensors["lengths"].tolist()
        )
    try:
        logits = mqa(
            restored["q_fp8"],
            restored["packed_kv_pages"],
            restored["weights"],
            restored["kvlen_2d"],
            restored["block_table"],
            schedule,
            manifest["parameters"]["max_seq_len"],
            clean_logits=False,
        )
    except (AssertionError, NotImplementedError, RuntimeError) as error:
        if isinstance(error, RuntimeError) and not isinstance(
            error, NotImplementedError
        ):
            if not _is_mqa_architecture_rejection(str(error)):
                raise
        raise UnsupportedCapture(
            "DeepGEMM rejected paged MQA input/device contract: " + str(error)
        ) from error
    return logits, dict(
        mode="reused" if reused else "rebuilt",
        environment=environment,
        compact_pages=restored["packed_kv_pages"].shape[0],
    )


def _json_number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def _compare_mqa(tensors, actual, atol, rtol):
    expected = tensors["logits"]
    _require(
        actual.dtype == torch.float32 and actual.shape == expected.shape,
        "DeepGEMM returned invalid logits dtype/shape",
    )
    max_abs = max_rel = 0.0
    matched, compared = True, 0
    for row in range(expected.shape[0]):
        start, end = _row_bounds(tensors, row)
        reference, replayed = (
            expected[row, start:end].double(),
            actual[row, start:end].double(),
        )
        if bool(torch.isnan(replayed).any()):
            raise UnsupportedCapture("replayed valid-domain MQA NaN unsupported")
        if reference.numel() == 0:
            continue
        difference = torch.where(
            reference == replayed, 0.0, (reference - replayed).abs()
        )
        relative = torch.where(difference == 0, 0.0, difference / reference.abs())
        # Unequal infinities have no finite relative error; do not serialize NaN.
        relative = torch.nan_to_num(relative, nan=float("inf"), posinf=float("inf"))
        max_abs = max(max_abs, float(difference.max()))
        max_rel = max(max_rel, float(relative.max()))
        matched &= bool(torch.isclose(replayed, reference, atol=atol, rtol=rtol).all())
        compared += reference.numel()
    return dict(
        allclose=matched,
        atol=atol,
        rtol=rtol,
        valid_values=compared,
        max_abs=_json_number(max_abs),
        max_rel=_json_number(max_rel),
    )


def _chain_comparison(manifest, tensors, actual):
    captured = tensors["indices"]
    different_rows = different_indices = 0
    examples = []
    min_gap = None
    k = manifest["parameters"]["k"]
    for row in range(captured.shape[0]):
        old, new = set(captured[row].tolist()) - {-1}, set(actual[row].tolist()) - {-1}
        delta = len(old ^ new)
        different_rows += int(delta > 0)
        different_indices += delta
        start, end = _row_bounds(tensors, row)
        length = end - start
        if manifest["backend"] == "topk_v3_tie_break":
            length = min(length, manifest["parameters"]["max_seq_len"])
        threshold = gap = None
        if length > k:
            values = tensors["logits"][row, start : start + length].double()
            selected = captured[row].long()
            threshold = float(values[selected].min())
            remaining = torch.ones(length, dtype=torch.bool)
            remaining[selected] = False
            runner_up = float(values[remaining].max())
            gap = 0.0 if threshold == runner_up else threshold - runner_up
            min_gap = gap if min_gap is None else min(min_gap, gap)
        if len(examples) < 16 and (delta or gap is not None):
            examples.append(
                dict(
                    row=row,
                    symmetric_difference_count=delta,
                    captured_threshold=(
                        None if threshold is None else _json_number(threshold)
                    ),
                    captured_score_gap=None if gap is None else _json_number(gap),
                )
            )
    return dict(
        selected_sets_equal=different_rows == 0,
        differing_rows=different_rows,
        symmetric_difference_count=different_indices,
        min_captured_score_gap=None if min_gap is None else _json_number(min_gap),
        row_examples=examples,
        row_examples_limit=16,
    )


def _is_mqa_architecture_rejection(message):
    # DGException is translated to RuntimeError. Inspect only the diagnostic,
    # not stack frames, and do not misclassify mixed shape/scale assertions.
    diagnostic = message.splitlines()[0] if message else ""
    if diagnostic == "Unsupported architecture" or diagnostic.startswith(
        "Unsupported architecture: "
    ):
        return True
    assertion = re.fullmatch(r"Assertion error \([^\n]+:[0-9]+\): (.+)", diagnostic)
    if assertion is None:
        return False
    return assertion.group(1) in {
        "Unsupported architecture",
        "false: Unsupported architecture",
        "arch_major == 10",
        "jit->device.get_arch_major() == 9",
        "jit->device.get_arch_major() == 10",
    }


def _call_mqa(mqa, restored):
    # weights already include query scale; do not quantize or scale again.
    try:
        return mqa(
            restored["q_fp8"],
            (restored["k_fp8"], restored["k_scale"]),
            restored["weights"],
            restored["row_starts"],
            restored["row_ends"],
            clean_logits=False,
        )
    except (AssertionError, NotImplementedError, RuntimeError) as error:
        if isinstance(error, RuntimeError) and not isinstance(
            error, NotImplementedError
        ):
            if not _is_mqa_architecture_rejection(str(error)):
                raise
        layout = "; ".join(
            name
            + "_shape="
            + str(tuple(restored[name].shape))
            + "; "
            + name
            + "_stride="
            + str(restored[name].stride())
            for name in ("q_fp8", "k_fp8", "k_scale", "weights")
        )
        raise UnsupportedCapture(
            "DeepGEMM rejected the saved input/device contract: "
            + str(error)
            + "; "
            + layout
        ) from error


def _replay_mqa(manifest, tensors, restored, operator, summary, atol, rtol):
    if _is_paged(manifest):
        logits, summary["schedule"] = _call_paged_mqa(manifest, tensors, restored)
    else:
        logits = _call_mqa(_mqa_operator(), restored)
    torch.cuda.synchronize()
    _require(isinstance(logits, torch.Tensor), "DeepGEMM must return a logits tensor")
    actual_logits = logits.detach().cpu()
    comparison = _compare_mqa(tensors, actual_logits, atol, rtol)
    # Reuse the source-layout restoration path, preserving all returned values,
    # including unspecified invalid-domain values. Only valid ranges are read.
    chained = dict(restored)
    chained["logits"] = _restore(logits, manifest["tensors"]["logits"], "cuda")
    indices = _run_topk(operator, manifest, chained)
    chain = _chain_comparison(manifest, tensors, indices)
    if _is_paged(manifest):
        chain["captured_score_keys_equal"] = _paged_score_sets_equal(tensors, indices)
    try:
        _check_indices(manifest, dict(tensors, logits=actual_logits), indices)
        chain["contract_valid"] = True
    except ValueError as error:
        chain.update(contract_valid=False, contract_error=str(error))
    summary.update(
        mode="cuda_mqa_topk_replay",
        mqa_kernel_replayed=True,
        independent_topk=summary.get(
            "independent_topk", dict(contract_valid=True, selected_sets_equal=True)
        ),
        mqa_comparison=comparison,
        chained_topk=chain,
    )
    selection_equal = chain.get(
        "captured_score_keys_equal", chain["selected_sets_equal"]
    )
    if not comparison["allclose"] or not selection_equal or not chain["contract_valid"]:
        summary["status"] = "failed"
        raise ReplayMismatch(summary)
    return summary


class _IndexerReplayAdapter:
    """Indexer legacy and indexed contracts share fixed kernels and comparisons.

    Version2 validates the indexed CP mapping; version3 validates paged decode.
    Both use the capture adapter's validators. Native operation wiring and index
    semantics belong here, not in the generic dispatcher.
    """

    @staticmethod
    def validate_options(options):
        for tolerance in (options["mqa_atol"], options["mqa_rtol"]):
            _require(
                type(tolerance) in (int, float)
                and math.isfinite(tolerance)
                and tolerance >= 0,
                "MQA tolerances must be finite and nonnegative",
            )

    @staticmethod
    def validate(manifest, tensors):
        params = _check_schema(manifest, tensors)
        counts = _check_indices(manifest, tensors, tensors["indices"])
        summary = dict(
            status="validated",
            mode="cpu_contract_validation",
            kernel_replayed=False,
            backend=manifest["backend"],
            k=params["k"],
            counts=counts,
            selection_contract="descending_fp32_bit_key_then_ascending_relative_index",
            nan_policy="unsupported_in_valid_domain",
        )
        if _is_paged(manifest):
            summary.update(
                phase="decode",
                selection_contract="descending_fp32_bit_key_boundary_ties_unspecified",
            )
        if manifest["op"] == "indexer.mqa_topk":
            summary.update(
                schema_version=manifest["schema_version"],
                op=manifest["op"],
                mqa_kernel_replayed=False,
                mqa_backend=manifest["mqa_parameters"]["backend"],
                contract_source="current_source_not_binary_revision_attestation",
            )
        return summary

    @staticmethod
    def replay(manifest, tensors, *, device, summary, options):
        operator = _cuda_operator(manifest["backend"])
        if _is_paged(manifest):
            restored = _restore_paged(manifest, tensors, device)
        else:
            restored = {
                name: _restore(tensor, manifest["tensors"][name], device)
                for name, tensor in tensors.items()
                if name != "query_global_ids"
            }
        actual = _run_topk(operator, manifest, restored)
        _check_indices(manifest, tensors, actual)
        if _is_paged(manifest):
            sets_equal = torch.equal(
                actual.sort(dim=1).values, tensors["indices"].sort(dim=1).values
            )
            scores_equal = _paged_score_sets_equal(tensors, actual)
            _require(scores_equal, "CUDA/capture selected score keys differ")
            summary["independent_topk"] = dict(
                contract_valid=True,
                selected_sets_equal=sets_equal,
                captured_score_keys_equal=scores_equal,
            )
        summary.update(
            mode="cuda_kernel_replay",
            kernel_replayed=True,
            comparison="selected_sets_and_short_row_order",
            raw_order_equal=torch.equal(actual, tensors["indices"]),
        )
        if _is_paged(manifest):
            summary["comparison"] = "selected_score_keys_and_short_row_order"
        if manifest["op"] == "indexer.mqa_topk":
            return _replay_mqa(
                manifest,
                tensors,
                restored,
                operator,
                summary,
                options["mqa_atol"],
                options["mqa_rtol"],
            )
        return summary


_REPLAY_CONTRACTS = {}
_LEGACY_CONTRACTS = {
    1: ("indexer.topk", 1),
    2: ("indexer.mqa_topk", 1),
}


def register_replay_contract(
    name, version, *, validator, replayer=None, validate_options=None
):
    """Register trusted Python callbacks, never code or imports from an artifact.

    Call once at startup for each stable (name, positive version); duplicates are
    errors. The validator(manifest, tensors) must check the backend, parameters,
    named inputs/outputs and numerical contract, returning a JSON-safe summary.
    The loader owns file, descriptor and schema-envelope validation. Tensors are
    CPU-contiguous logical snapshots; source-layout restoration is adapter-owned.

    CPU only invokes the validator and always reports kernel_replayed=False.
    CUDA first validates, then calls replayer(manifest, tensors, *, device,
    summary, options). The replayer owns execution, output comparisons and its
    truthful kernel_replayed flag. Omit it for validation-only contracts.
    options carries the legacy CLI mqa_atol/mqa_rtol; other adapters may ignore
    them. Optional validate_options(options) runs before validation/execution;
    only the Indexer adapter enforces MQA tolerance semantics by default.
    Callbacks are trusted application code, not a sandbox or proof that a native
    kernel ran. Artifact metadata cannot authorize callbacks or set replay state.
    Unsupported contracts/backends should raise UnsupportedCapture;
    unavailable native implementations should raise ReplayUnavailable.
    """
    _require(
        isinstance(name, str)
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", name) is not None,
        "invalid replay contract name",
    )
    _require(type(version) is int and version > 0, "invalid replay contract version")
    _require(callable(validator), "replay validator must be callable")
    _require(replayer is None or callable(replayer), "replay replayer must be callable")
    _require(
        validate_options is None or callable(validate_options),
        "replay options validator must be callable",
    )
    key = (name, version)
    _require(
        key not in _REPLAY_CONTRACTS, "replay contract already registered: " + repr(key)
    )
    _REPLAY_CONTRACTS[key] = (validator, replayer, validate_options)


for _name, _version in (
    *_LEGACY_CONTRACTS.values(),
    ("indexer.topk", 2),
    ("indexer.mqa_topk", 2),
    ("indexer.topk", 3),
    ("indexer.mqa_topk", 3),
):
    register_replay_contract(
        _name,
        _version,
        validator=_IndexerReplayAdapter.validate,
        replayer=_IndexerReplayAdapter.replay,
        validate_options=_IndexerReplayAdapter.validate_options,
    )


def replay_capture(path, device="cpu", *, mqa_atol=0.0, mqa_rtol=0.0):
    """Validate or replay; ReplayMismatch retains joint numerical measurements."""
    _require(device in ("cpu", "cuda"), "device must be cpu or cuda")
    options = dict(mqa_atol=mqa_atol, mqa_rtol=mqa_rtol)
    manifest, tensors = load_capture(path)
    if manifest["schema_version"] == 3:
        contract = manifest["contract"]
        key = (contract["name"], contract["version"])
    else:
        key = _LEGACY_CONTRACTS.get(manifest["schema_version"])
    callbacks = _REPLAY_CONTRACTS.get(key)
    if callbacks is None:
        raise UnsupportedCapture("unsupported replay contract: " + repr(key))
    validator, replayer, validate_options = callbacks
    if validate_options is not None:
        validate_options(options)
    summary = dict(validator(manifest, tensors))
    summary.setdefault("status", "validated")
    summary["mode"] = "cpu_contract_validation"
    summary.setdefault("backend", manifest["backend"])
    summary["kernel_replayed"] = False
    if manifest["schema_version"] == 3:
        summary.update(schema_version=3, op=manifest["op"], contract=dict(contract))
    if device == "cpu":
        return summary
    if replayer is None:
        raise ReplayUnavailable(
            "no CUDA replayer registered for contract: " + repr(key)
        )
    return replayer(manifest, tensors, device=device, summary=summary, options=options)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mqa-atol", type=float, default=0.0)
    parser.add_argument("--mqa-rtol", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        summary = replay_capture(
            args.manifest, args.device, mqa_atol=args.mqa_atol, mqa_rtol=args.mqa_rtol
        )
    except ReplayMismatch as error:
        print(
            json.dumps(error.summary, sort_keys=True, allow_nan=False), file=sys.stderr
        )
        return 1
    except (ValueError, RuntimeError, OSError) as error:
        if isinstance(error, ReplayUnavailable):
            status, code = "unavailable", 3
        elif isinstance(error, UnsupportedCapture):
            status, code = "unsupported", 2
        else:
            status, code = "failed", 1
        print(
            json.dumps(dict(status=status, device=args.device, error=str(error))),
            file=sys.stderr,
        )
        return code
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
