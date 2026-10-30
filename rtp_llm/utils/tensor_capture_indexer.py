"""Indexer capture contracts. Kernel/layout semantics do not belong to the writer.

The adapter retains the original v1 TopK and v2 MQA/TopK artifacts. Both use the
same named-tensor reservation, snapshot, publication and cancellation lifecycle
as v3 contracts; a joint capture never falls back to a TopK-only sample.
"""

import re
from bisect import bisect_right
from collections.abc import Mapping
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Optional

import torch

from rtp_llm.utils import tensor_capture as capture

_TOPK = "indexer.topk"
_JOINT = "indexer.mqa_topk"


def _validate_mqa_inputs(inputs):
    q_fp8, k_fp8, k_scale, weights = (
        inputs[name] for name in ("q_fp8", "k_fp8", "k_scale", "weights")
    )
    if q_fp8.dtype != torch.float8_e4m3fn or q_fp8.ndim != 3:
        raise ValueError("unsupported query FP8 contract")
    rows, heads, dimension = q_fp8.shape
    if (
        not capture._integer(heads, 1)
        or not capture._integer(dimension, 128)
        or dimension % 128
    ):
        raise ValueError("unsupported query shape")
    if k_fp8.dtype != q_fp8.dtype or k_fp8.ndim != 2 or k_fp8.shape[1] != dimension:
        raise ValueError("unsupported key FP8 contract")
    columns = k_fp8.shape[0]
    scale_shape = (columns,) if dimension == 128 else (columns, dimension // 128)
    if k_scale.dtype != torch.float32 or k_scale.shape != scale_shape:
        raise ValueError("unsupported key scale format")
    if weights.dtype != torch.float32 or weights.shape != (rows, heads):
        raise ValueError("unsupported weights contract")
    if q_fp8.device.type not in ("cpu", "cuda") or any(
        t.device != q_fp8.device for t in inputs.values()
    ):
        raise ValueError("MQA input devices must match")
    return rows, columns


def _parameters(parameters, k):
    clean = capture._parameters(parameters)
    if (
        clean.get("k", k) != k
        or clean.get("index_space", "relative_to_row_start") != "relative_to_row_start"
    ):
        raise ValueError("inconsistent topk parameters")
    clean.update(k=k, index_space="relative_to_row_start")
    if len(capture._json_bytes(clean)) > capture._MAX_PARAMETERS_BYTES:
        raise ValueError("backend parameters exceed reserved metadata bound")
    return clean


def _snapshot_logits(logits, tensors):
    tensors["logits"] = capture._snapshot(logits)
    for row in range(logits.shape[0]):
        start = int(tensors["row_starts"][row])
        stop = int(tensors["row_ends"][row])
        tensors["logits"][row, :start].zero_()
        tensors["logits"][row, stop:].zero_()


_INDEXED_FIELDS = frozenset(
    {"query_mapping", "key_domain", "execution_layout", "capture_coverage"}
)


def _validate_query_metadata(identity, metadata):
    """Validate bounded scalar mapping metadata without reading tensor values."""
    requests = identity.get("requests")
    if not isinstance(requests, (list, tuple)) or not 1 <= len(requests) <= 4096:
        raise ValueError("invalid query requests")
    q_offsets, key_offsets, prefixes = [0], [0], []
    for request in requests:
        if (
            not isinstance(request, Mapping)
            or not capture._integer(request.get("q_len"), 1)
            or not capture._integer(request.get("kv_len"))
        ):
            raise ValueError("invalid request lengths")
        q_offsets.append(q_offsets[-1] + request["q_len"])
        key_offsets.append(key_offsets[-1] + request["q_len"] + request["kv_len"])
        prefixes.append(request["kv_len"])
    rows, total = metadata.get("query_row_count"), metadata.get("total_query_rows")
    if (
        not capture._integer(rows, 1)
        or not capture._integer(total, 1)
        or total != q_offsets[-1]
        or rows > total
        or key_offsets[-1] > capture._MAX_DIM
    ):
        raise ValueError("incomplete query mapping")
    indexed = bool(_INDEXED_FIELDS.intersection(metadata))
    if indexed:
        if "query_row_start" in metadata or not _INDEXED_FIELDS.issubset(metadata):
            raise ValueError("indexed mapping requires exclusive complete metadata")
        if metadata["query_mapping"] != dict(
            kind="indexed",
            tensor="query_global_ids",
            index_space="request_order_query_rows",
        ) or metadata["key_domain"] != dict(
            layout="request_order_prefix_then_query", coverage="full"
        ):
            raise ValueError("unsupported indexed query/key mapping")
        layout, coverage = metadata["execution_layout"], metadata["capture_coverage"]
        if (
            not isinstance(layout, Mapping)
            or set(layout) != {"cp_rank", "cp_size", "kv_cache_sharded"}
            or not capture._integer(layout.get("cp_size"), 1)
            or not capture._integer(layout.get("cp_rank"))
            or layout["cp_rank"] >= layout["cp_size"]
            or type(layout.get("kv_cache_sharded")) is not bool
        ):
            raise ValueError("invalid execution layout")
        if (
            not isinstance(coverage, Mapping)
            or set(coverage) != {"scope", "local_query_rows", "local_query_row_start"}
            or coverage.get("scope") != "record_owner_local"
            or not capture._integer(coverage.get("local_query_rows"), 1)
            or not capture._integer(coverage.get("local_query_row_start"))
            or coverage["local_query_rows"] > total
            or coverage["local_query_row_start"] + rows > coverage["local_query_rows"]
        ):
            raise ValueError("invalid local query coverage")
    else:
        offset = metadata.get("query_row_start")
        if not capture._integer(offset) or offset + rows > total:
            raise ValueError("invalid query chunk")
    return indexed, q_offsets, key_offsets, prefixes


def _validate_mapping(identity, rows, columns, starts, ends, k, metadata, ids=None):
    if not capture._integer(k, 1):
        raise ValueError("invalid topk dimensions")
    indexed, _, key_offsets, _ = _validate_query_metadata(identity, metadata)
    if rows != metadata["query_row_count"] or any(
        x.dtype != torch.int32 or x.shape != (rows,) for x in (starts, ends)
    ):
        raise ValueError("only int32 row ranges supported")
    if columns != key_offsets[-1]:
        raise ValueError("incomplete key domain")
    if indexed:
        if (
            not isinstance(ids, torch.Tensor)
            or ids.dtype != torch.int64
            or ids.shape != (rows,)
        ):
            raise ValueError("query_global_ids must be int64 [rows]")
    elif ids is not None:
        raise ValueError("legacy mapping cannot contain query_global_ids")


def validate_query_mapping(identity, tensors, metadata):
    """Validate contiguous or indexed request/prefix mapping; return None or raise ValueError.

    Capture passes owned CPU mapping snapshots, replay passes loaded CPU tensors.
    Any supplied logits/K must span the full key domain of every request, even
    requests with no selected query rows. Mapping-only snapshots are supported
    during capture before large payload copies. No IO, CUDA work, mutation,
    sorting or large payload copies occur here. Other tensor shapes/backends are
    the caller's responsibility. Indexed IDs identify request-order query rows,
    not K columns or rank-local storage offsets.
    """
    indexed, q_offsets, key_offsets, prefixes = _validate_query_metadata(
        identity, metadata
    )
    for name, key_axis in (("logits", 1), ("k_fp8", 0)):
        if name in tensors:
            tensor = tensors[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 2
                or tensor.shape[key_axis] != key_offsets[-1]
            ):
                raise ValueError(name + " does not cover the complete key domain")
    rows = metadata["query_row_count"]
    for name in ("row_starts", "row_ends"):
        tensor = tensors.get(name)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.int32
            or tensor.shape != (rows,)
            or tensor.device.type != "cpu"
        ):
            raise ValueError("row ranges must be CPU int32 [rows]")
    ids = tensors.get("query_global_ids")
    if indexed:
        if (
            not isinstance(ids, torch.Tensor)
            or ids.dtype != torch.int64
            or ids.shape != (rows,)
            or ids.device.type != "cpu"
        ):
            raise ValueError("query_global_ids must be CPU int64 [rows]")
        global_rows = ids.tolist()
        if any(g < 0 or g >= q_offsets[-1] for g in global_rows):
            raise ValueError("query_global_ids out of range")
        if len(set(global_rows)) != rows:
            raise ValueError("duplicate query_global_ids")
    else:
        if "query_global_ids" in tensors:
            raise ValueError("legacy mapping cannot contain query_global_ids")
        global_rows = range(
            metadata["query_row_start"], metadata["query_row_start"] + rows
        )
    for row, global_row in enumerate(global_rows):
        request = bisect_right(q_offsets, global_row) - 1
        start = key_offsets[request]
        stop = start + prefixes[request] + global_row - q_offsets[request] + 1
        if (
            int(tensors["row_starts"][row]) != start
            or int(tensors["row_ends"][row]) != stop
        ):
            raise ValueError("row range differs from frozen prefix mapping")


class _IndexerCaptureToken(capture.CaptureToken):
    """Adapt Indexer arguments/stage checks; CaptureToken owns all state transitions."""

    def prepare_topk(self, logits, *, backend, parameters):
        if self._contract.version == 3:
            try:
                _validate_paged_topk_parameters(parameters, logits.shape[0])
                if backend not in _PAGED_BACKENDS:
                    raise ValueError("unsupported paged TopK backend")
                if (
                    parameters["max_seq_len"] != self._outputs["logits"].shape[1]
                    or parameters["k"] != self._outputs["indices"].shape[1]
                    or tuple(logits.stride())
                    != (parameters["stride0"], parameters["stride1"])
                ):
                    raise ValueError(
                        "paged TopK stage parameters differ from actual tensors"
                    )
            except (ValueError, KeyError, IndexError, TypeError):
                self.cancel("invalid_paged_topk_contract")
                return False
        return self.capture_outputs(
            {"logits": logits}, backend=backend, parameters=parameters
        )

    def finish(self, output):
        if self._contract.version == 3:
            try:
                _paged_require(
                    self._manifest is not None
                    and self._manifest["backend"] in _PAGED_BACKENDS,
                    "paged TopK stage was not captured",
                )
                _validate_paged_topk_parameters(
                    self._manifest["parameters"], self._outputs["indices"].shape[0]
                )
            except (ValueError, KeyError, TypeError):
                self.cancel("invalid_paged_topk_contract")
                return False
        return super().finish(
            output if isinstance(output, dict) else {"indices": output}
        )


class _IndexerContract(capture.CaptureContract):
    def clean_parameters(self, parameters, outputs):
        if self.version == 2:
            expected = {"indices"} if self.name == _TOPK else {"indices", "logits"}
            if set(outputs) != expected:
                raise ValueError("unexpected indexed outputs")
            indices = outputs["indices"]
            if (
                indices.dtype != torch.int32
                or len(indices.shape) != 2
                or min(indices.shape) < 1
            ):
                raise ValueError("indices must be nonempty int32 [rows, k]")
            if self.name == _JOINT:
                logits = outputs["logits"]
                if (
                    logits.dtype != torch.float32
                    or len(logits.shape) != 2
                    or logits.shape[0] != indices.shape[0]
                    or logits.shape[1] < 1
                ):
                    raise ValueError("logits must be float32 [rows, columns]")
        return _parameters(parameters, outputs["indices"].shape[1])

    def snapshot(self, name, tensor, tensors):
        if self.version == 2 and name in ("logits", "indices"):
            if tensor.ndim != 2 or tensor.shape[0] != tensors["row_starts"].numel():
                raise ValueError("indexed output row count mismatch")
            if (
                name == "logits"
                and self.name == _JOINT
                and tensor.shape[1] != tensors["k_fp8"].shape[0]
            ):
                raise ValueError("indexed logits must cover full keys")
        if name == "logits":
            _snapshot_logits(tensor, tensors)
        else:
            super().snapshot(name, tensor, tensors)

    def snapshot_inputs(self, inputs, tensors, identity, metadata):
        # Check the frozen row mapping before copying large logits/FP8 payloads.
        try:
            if self.version == 2:
                expected = {"query_global_ids", "row_starts", "row_ends"}
                expected |= (
                    {"logits"}
                    if self.name == _TOPK
                    else {"q_fp8", "k_fp8", "k_scale", "weights"}
                )
                if set(inputs) != expected or not _INDEXED_FIELDS.issubset(metadata):
                    raise ValueError(
                        "indexed contract requires indexed named inputs and metadata"
                    )
                operand = inputs["logits"] if self.name == _TOPK else inputs["q_fp8"]
                if self.name == _TOPK:
                    if operand.dtype != torch.float32 or operand.ndim != 2:
                        raise ValueError("logits must be float32 [rows, columns]")
                    rows, columns = operand.shape
                else:
                    rows, columns = _validate_mqa_inputs(inputs)
                _validate_mapping(
                    identity,
                    rows,
                    columns,
                    inputs["row_starts"],
                    inputs["row_ends"],
                    1,
                    metadata,
                    inputs["query_global_ids"],
                )
                if any(t.device != operand.device for t in inputs.values()):
                    raise ValueError("indexed input devices must match")
            elif _INDEXED_FIELDS.intersection(metadata) or "query_global_ids" in inputs:
                raise ValueError("legacy contract requires contiguous mapping")
        except ValueError as error:
            raise capture.CaptureContractError(str(error)) from error
        names = (
            ("query_global_ids", "row_starts", "row_ends")
            if self.version == 2
            else ("row_starts", "row_ends")
        )
        for name in names:
            self.snapshot(name, inputs[name], tensors)
        try:
            validate_query_mapping(identity, tensors, metadata)
        except ValueError as error:
            raise capture.CaptureContractError(str(error)) from error
        for name, tensor in inputs.items():
            if name not in tensors:
                self.snapshot(name, tensor, tensors)

    def error_reason(self, stage):
        if self.name == _TOPK and stage in ("input", "contract"):
            return "invalid_topk_or_snapshot_error"
        return {
            "contract": "invalid_mqa_contract",
            "input": "input_snapshot_error",
            "stage_contract": "invalid_mqa_contract",
            "stage": "logits_snapshot_error",
            "state": "invalid_mqa_state",
            "output": "output_snapshot_error",
        }.get(stage, super().error_reason(stage))


_METADATA_FIELDS = frozenset(
    {
        "query_row_start",
        "total_query_rows",
        "query_row_count",
        "invalid_logits",
        "finite_values_checked",
        "nonfinite_policy",
        "output_contract",
        "mqa_parameters",
        "operations",
    }
)


_PAGED_BACKENDS = frozenset({"topk_v3", "dsv4_persistent_topk"})
_PAGED_QUERY_MAPPING = dict(kind="batch_slot_order", next_n=1, coverage="full")
_PAGED_METADATA_FIELDS = frozenset(
    {
        "query_mapping",
        "total_query_rows",
        "invalid_logits",
        "finite_values_checked",
        "nonfinite_policy",
        "output_contract",
        "cache_layout",
        "mqa_parameters",
        "schedule_environment",
    }
)
_PAGED_INPUTS = frozenset(
    {
        "q_fp8",
        "weights",
        "kvlen_2d",
        "lengths",
        "block_table",
        "page_ids",
        "page_valid_tokens",
        "packed_kv_pages",
        "schedule_metadata",
    }
)
# Verified against the wheels' attention.hpp and SM90/SM100 paged schedulers
# for non-varlen next_n=1. Generic 2.8.0 remains readable in older artifacts,
# but is not sufficient provenance for a new runtime capture.
_PAGED_SCHEDULE_VERSIONS = frozenset(
    {"2.8.0", "2.8.0+6d6aaad.cu132", "2.8.0+122e18b.cu132"}
)


def paged_schedule_version(module):
    """Resolve an inspected build belonging to the imported package, not sys.path.

    Bazel may omit RECORD or all dist-info runfiles. Locate sibling metadata
    without RECORD; only an exact module build version can replace absent metadata.
    Never let the generic module __version__ mask an unknown distribution build.
    """
    filename = getattr(module, "__file__", None)
    if not isinstance(filename, str):
        raise ValueError("missing DeepGEMM package location")
    location = Path(filename).resolve()
    distributions = [
        dist
        for dist in package_metadata.distributions(path=[str(location.parent.parent)])
        if re.sub(r"[-_.]+", "-", dist.metadata.get("Name", "")).lower() == "deep-gemm"
    ]
    module_version = getattr(module, "__version__", None)
    if distributions:
        if len(distributions) != 1:
            raise ValueError("ambiguous DeepGEMM distribution metadata")
        dist = distributions[0]
        if Path(dist.locate_file("deep_gemm/__init__.py")).resolve() != location:
            raise ValueError("DeepGEMM distribution does not own imported package")
        version = dist.version
        if isinstance(module_version, str) and (
            module_version.split("+")[0] != str(version).split("+")[0]
            or ("+" in module_version and module_version != version)
        ):
            raise ValueError("conflicting DeepGEMM build versions")
    else:
        version = module_version
    if (
        not isinstance(version, str)
        or "+" not in version
        or version not in _PAGED_SCHEDULE_VERSIONS
    ):
        raise ValueError("unknown paged schedule ABI build")
    return version


def _paged_require(condition, message):
    if not condition:
        raise ValueError(message)


def _paged_expected_lengths(identity, total_query_rows):
    _paged_require(
        identity.get("model_role") == "target"
        and identity.get("phase", "decode") == "decode"
        and identity.get("server_role") in ("DECODE", "PDFUSION"),
        "paged capture requires target decode identity",
    )
    for name in ("session_id", "owner_id", "replica_id"):
        _paged_require(capture._identifier(identity.get(name)), "invalid paged " + name)
    for name in ("world_rank", "dp_rank"):
        _paged_require(capture._integer(identity.get(name)), "invalid paged " + name)
    _paged_require(
        isinstance(identity.get("exec_id"), str)
        and re.fullmatch(r"[0-9]{1,40}", identity["exec_id"]) is not None,
        "invalid paged exec_id",
    )
    requests = identity.get("requests")
    _paged_require(
        isinstance(requests, (list, tuple))
        and 1 <= len(requests) <= 4096
        and capture._integer(total_query_rows, 1)
        and len(requests) == total_query_rows,
        "paged capture requires full batch",
    )
    expected = []
    for row in requests:
        _paged_require(
            isinstance(row, Mapping)
            and row.get("phase") == "decode"
            and type(row.get("q_len")) is int
            and row["q_len"] == 1
            and row.get("is_fake", False) is False,
            "paged capture requires real single-token decode rows",
        )
        _paged_require(
            isinstance(row.get("request_id"), str)
            and re.fullmatch(r"r[0-9]{1,40}", row["request_id"]) is not None,
            "invalid paged request_id",
        )
        _paged_require(
            capture._integer(row.get("kv_len"), 0, 2**31 - 2),
            "unknown or invalid decode prior length",
        )
        if "sequence_id" in row:
            _paged_require(capture._integer(row["sequence_id"]), "invalid sequence_id")
        expected.append(row["kv_len"] + 1)
    return expected


def _paged_metadata(rows):
    return dict(
        query_mapping=dict(_PAGED_QUERY_MAPPING),
        total_query_rows=rows,
        invalid_logits="zero_filled_snapshot_only",
        finite_values_checked=False,
        nonfinite_policy="valid_domain_bits_preserved_backend_semantics",
        output_contract="raw_relative_indices_minus_one_padding_backend_order",
    )


def _validate_paged_topk_parameters(parameters, rows):
    _paged_require(
        isinstance(parameters, Mapping)
        and set(parameters)
        == {"k", "max_seq_len", "index_space", "workspace_bytes", "stride0", "stride1"},
        "invalid paged TopK parameters",
    )
    _paged_require(
        type(parameters["k"]) is int and parameters["k"] in (512, 1024, 2048),
        "unsupported paged TopK k",
    )
    for name in ("max_seq_len", "stride0", "workspace_bytes"):
        _paged_require(capture._integer(parameters[name], 1), "invalid paged " + name)
    _paged_require(
        parameters["index_space"] == "sequence_relative"
        and type(parameters["stride1"]) is int
        and parameters["stride1"] == 1
        and parameters["stride0"] >= parameters["max_seq_len"]
        and parameters["workspace_bytes"] >= 1 << 20,
        "invalid paged TopK layout or workspace",
    )
    _paged_require(
        rows * parameters["stride0"] * 4 <= capture._MAX_FILE_BYTES,
        "paged logits source layout exceeds limit",
    )


def _paged_tensor(tensor, dtype, shape, name, *, contiguous=True):
    _paged_require(
        isinstance(tensor, torch.Tensor)
        and tensor.dtype == dtype
        and tuple(tensor.shape) == tuple(shape)
        and (not contiguous or tensor.is_contiguous()),
        "invalid paged tensor " + name,
    )


def _validate_paged_schedule(environment, schedule, block_size):
    _paged_require(
        isinstance(environment, Mapping)
        and set(environment) == {"deep_gemm_version", "device_capability", "num_sms"},
        "invalid paged schedule environment",
    )
    _paged_require(
        isinstance(environment["deep_gemm_version"], str)
        and environment["deep_gemm_version"] in _PAGED_SCHEDULE_VERSIONS,
        "unknown paged schedule ABI",
    )
    capability = environment["device_capability"]
    _paged_require(
        isinstance(capability, (list, tuple))
        and len(capability) == 2
        and all(capture._integer(x) for x in capability)
        and capability[0] in (9, 10),
        "unsupported paged device capability",
    )
    _paged_require(
        type(block_size) is int
        and (
            (capability[0] == 9 and block_size == 64)
            or (capability[0] == 10 and block_size in (32, 64, 128))
        ),
        "unsupported paged block size",
    )
    _paged_require(capture._integer(environment["num_sms"], 1, 4096), "invalid num_sms")
    _paged_tensor(
        schedule, torch.int32, (environment["num_sms"] + 1, 2), "schedule_metadata"
    )


def _validate_paged_schedule_values(schedule, expected):
    # Both verified schedulers encode (query row, 256-token split), ending at
    # (batch_size, 0). Partitioning differs by architecture; do not assume SM90
    # and SM100 distribute idle SMs the same way.
    entries = schedule.tolist()
    rows = len(expected)
    _paged_require(
        entries[0] == [0, 0] and entries[-1] == [rows, 0],
        "invalid paged schedule endpoints",
    )
    previous = (0, 0)
    for row, split in entries:
        _paged_require(
            0 <= row <= rows
            and split >= 0
            and (split == 0 if row == rows else split < (expected[row] + 255) // 256)
            and (row, split) >= previous,
            "invalid paged schedule range",
        )
        previous = (row, split)


def _validate_paged_heads(heads, environment):
    valid = (
        heads in (32, 64)
        if environment["device_capability"][0] == 9
        else (0 < heads <= 128 and heads % 4 == 0)
    )
    _paged_require(valid, "unsupported paged query heads")


def _validate_paged_layout(layout, expected):
    _paged_require(
        isinstance(layout, Mapping)
        and set(layout)
        == {
            "version",
            "block_size",
            "head_dim",
            "scale_group_size",
            "page_capacity",
            "source_page_count",
            "layout",
        },
        "invalid paged cache layout",
    )
    for name in (
        "version",
        "block_size",
        "head_dim",
        "scale_group_size",
        "page_capacity",
        "source_page_count",
    ):
        _paged_require(
            capture._integer(layout[name], 1), "invalid cache layout " + name
        )
    _paged_require(
        layout["version"] == 1
        and layout["head_dim"] == 128
        and layout["scale_group_size"] == 128
        and layout["layout"] == "page_payload_then_float32_scales",
        "unsupported packed cache layout",
    )
    block = layout["block_size"]
    capacity = sum((length + block - 1) // block for length in expected)
    _paged_require(layout["page_capacity"] == capacity, "invalid page capacity")
    return block, layout["head_dim"], capacity


def _paged_page_union(table, expected, block, source_pages, *, sanitized):
    union = {}
    for row, length in enumerate(expected):
        count = (length + block - 1) // block
        _paged_require(
            count <= table.shape[1], "block table shorter than valid sequence"
        )
        for column in range(count):
            page = int(table[row, column])
            _paged_require(0 <= page < source_pages, "invalid referenced cache page")
            union[page] = max(union.get(page, 0), min(block, length - column * block))
        if sanitized:
            _paged_require(
                bool(torch.all(table[row, count:] == -1)),
                "unsanitized block table tail",
            )
    return union


def _paged_source_devices(inputs):
    device = next(iter(inputs.values())).device
    _paged_require(
        device.type in ("cpu", "cuda")
        and all(t.device == device for t in inputs.values()),
        "paged input devices must match",
    )
    if device.type == "cuda":
        with torch.cuda.device(device):
            _paged_require(
                not torch.cuda.is_current_stream_capturing(),
                "CUDA graph capture is unsupported",
            )


def _paged_snapshot_logits(logits, tensors):
    lengths = tensors["lengths"]
    _paged_tensor(
        logits,
        torch.float32,
        (lengths.numel(), logits.shape[1]),
        "logits",
        contiguous=False,
    )
    _paged_require(logits.stride(1) == 1, "invalid logits inner stride")
    snapshot = torch.zeros(tuple(logits.shape), dtype=torch.float32, device="cpu")
    for row, length in enumerate(lengths.tolist()):
        _paged_require(0 < length <= logits.shape[1], "invalid logits length")
        # Do not transfer uninitialized logits tails, even transiently.
        snapshot[row, :length].copy_(logits[row, :length].detach(), non_blocking=False)
    tensors["logits"] = snapshot


class _PagedIndexerContract(capture.CaptureContract):
    def input_specs(self, inputs, identity, metadata):
        expected = _paged_expected_lengths(identity, metadata["total_query_rows"])
        if self.name == _TOPK:
            return {}
        block, dimension, capacity = _validate_paged_layout(
            metadata["cache_layout"], expected
        )
        stride = dimension + 4 * (dimension // 128)
        return {
            "packed_kv_pages": capture.TensorSpec(
                (capacity, block, 1, stride), torch.uint8
            ),
            "page_ids": capture.TensorSpec((capacity,), torch.int64),
            "page_valid_tokens": capture.TensorSpec((capacity,), torch.int32),
            "block_table": capture.TensorSpec(inputs["block_table"].shape, torch.int32),
        }

    def clean_parameters(self, parameters, outputs):
        joint = self.name == _JOINT
        _paged_require(
            set(outputs) == ({"logits", "indices"} if joint else {"indices"}),
            "invalid paged outputs",
        )
        indices = outputs["indices"]
        _paged_require(
            indices.dtype == torch.int32
            and len(indices.shape) == 2
            and indices.shape[0] > 0
            and indices.shape[1] in (512, 1024, 2048),
            "invalid paged indices specification",
        )
        if joint:
            logits = outputs["logits"]
            _paged_require(
                logits.dtype == torch.float32
                and len(logits.shape) == 2
                and logits.shape[0] == indices.shape[0]
                and logits.shape[1] > 0,
                "invalid paged logits specification",
            )
            if not parameters:
                return {}  # Backend/stride/workspace are supplied by prepare_topk.
        _validate_paged_topk_parameters(parameters, indices.shape[0])
        _paged_require(
            parameters["k"] == indices.shape[1]
            and (not joint or parameters["max_seq_len"] == outputs["logits"].shape[1]),
            "paged parameters differ from reserved outputs",
        )
        return capture._parameters(parameters)

    def snapshot(self, name, tensor, tensors):
        if name == "logits":
            _paged_snapshot_logits(tensor, tensors)
        else:
            super().snapshot(name, tensor, tensors)

    def snapshot_inputs(self, inputs, tensors, identity, metadata):
        try:
            expected = _paged_expected_lengths(identity, metadata["total_query_rows"])
            rows = len(expected)
            _paged_source_devices(inputs)
            _paged_require(
                metadata["query_mapping"] == _PAGED_QUERY_MAPPING,
                "invalid paged query mapping",
            )
            _paged_tensor(inputs["lengths"], torch.int32, (rows,), "lengths")
            self.snapshot("lengths", inputs["lengths"], tensors)
            _paged_require(
                tensors["lengths"].tolist() == expected,
                "runtime lengths differ from frozen decode batch",
            )
            if self.name == _TOPK:
                _paged_require(
                    set(inputs) == {"logits", "lengths"}, "invalid paged TopK inputs"
                )
                self.snapshot("logits", inputs["logits"], tensors)
                return
            _paged_require(set(inputs) == _PAGED_INPUTS, "invalid paged MQA inputs")
            layout = metadata["cache_layout"]
            block, dimension, capacity = _validate_paged_layout(layout, expected)
            max_seq_len = metadata["mqa_parameters"]["max_seq_len"]
            _paged_require(
                capture._integer(max_seq_len, 1)
                and max_seq_len % block == 0
                and max(expected) <= max_seq_len,
                "invalid paged sequence capacity",
            )
            q = inputs["q_fp8"]
            _paged_require(q.ndim == 4 and q.shape[2] > 0, "invalid paged query")
            _paged_tensor(
                q, torch.float8_e4m3fn, (rows, 1, q.shape[2], dimension), "q_fp8"
            )
            _paged_tensor(
                inputs["weights"],
                torch.float32,
                (rows, q.shape[2]),
                "weights",
                contiguous=False,
            )
            _paged_require(inputs["weights"].stride(1) == 1, "invalid weights stride")
            _paged_tensor(inputs["kvlen_2d"], torch.int32, (rows, 1), "kvlen_2d")
            _paged_tensor(
                inputs["block_table"],
                torch.int32,
                (rows, max_seq_len // block),
                "block_table",
                contiguous=False,
            )
            _paged_require(
                inputs["block_table"].stride(1) == 1, "invalid block table stride"
            )
            _validate_paged_schedule(
                metadata["schedule_environment"], inputs["schedule_metadata"], block
            )
            _validate_paged_heads(q.shape[2], metadata["schedule_environment"])
            self.snapshot("kvlen_2d", inputs["kvlen_2d"], tensors)
            _paged_require(
                tensors["kvlen_2d"].flatten().tolist() == expected,
                "MQA context lengths differ from TopK lengths",
            )
            self.snapshot("schedule_metadata", inputs["schedule_metadata"], tensors)
            _validate_paged_schedule_values(tensors["schedule_metadata"], expected)
            # Only table entries that the kernel is allowed to consume are copied.
            table = torch.full(
                tuple(inputs["block_table"].shape), -1, dtype=torch.int32
            )
            for row, length in enumerate(expected):
                count = (length + block - 1) // block
                _paged_require(
                    count <= table.shape[1], "insufficient block table capacity"
                )
                table[row, :count].copy_(inputs["block_table"][row, :count].detach())
            union = _paged_page_union(
                table, expected, block, layout["source_page_count"], sanitized=True
            )
            tensors["block_table"] = table
            stride = dimension + 4 * (dimension // 128)
            source = inputs["packed_kv_pages"]
            _paged_tensor(
                source,
                torch.uint8,
                (layout["source_page_count"], block, 1, stride),
                "packed_kv_pages source",
            )
            pages = torch.zeros((capacity, block, 1, stride), dtype=torch.uint8)
            ids = torch.full((capacity,), -1, dtype=torch.int64)
            valid = torch.zeros((capacity,), dtype=torch.int32)
            # First-appearance ordering is deterministic; union lengths were fully
            # computed before any payload copy, including longer shared references.
            for slot, (page, count) in enumerate(union.items()):
                ids[slot], valid[slot] = page, count
                src = source[page].view(-1)
                dst = pages[slot].view(-1)
                dst[: count * dimension].copy_(src[: count * dimension].detach())
                start = block * dimension
                end = start + count * (dimension // 128) * 4
                dst[start:end].copy_(src[start:end].detach())
            tensors.update(page_ids=ids, page_valid_tokens=valid, packed_kv_pages=pages)
            for name in ("q_fp8", "weights"):
                self.snapshot(name, inputs[name], tensors)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            raise capture.CaptureContractError(str(error)) from error


def validate_paged_capture(manifest, tensors):
    """Validate CPU paged artifacts; numerical TopK comparison belongs to replay."""
    _paged_require(
        isinstance(manifest, dict) and isinstance(tensors, dict),
        "paged manifest and tensors must be mappings",
    )
    joint = manifest.get("op") == _JOINT
    allowed = {
        "schema_version",
        "contract",
        "complete",
        "session_id",
        "owner_id",
        "replica_id",
        "world_rank",
        "dp_rank",
        "exec_id",
        "execution_id",
        "model_role",
        "server_role",
        "requests",
        "call_id",
        "layer",
        "rank",
        "phase",
        "op",
        "inputs",
        "outputs",
        "tensors",
        "backend",
        "parameters",
    } | _PAGED_METADATA_FIELDS
    _paged_require(not manifest.keys() - allowed, "unknown paged manifest fields")
    _paged_require(
        type(manifest.get("schema_version")) is int
        and manifest["schema_version"] == 3
        and manifest.get("op") in (_TOPK, _JOINT)
        and manifest.get("contract") == {"name": manifest["op"], "version": 3}
        and type(manifest["contract"]["version"]) is int
        and manifest.get("complete") is True
        and manifest.get("phase") == "decode",
        "invalid paged capture envelope",
    )
    expected = _paged_expected_lengths(manifest, manifest.get("total_query_rows"))
    rows = len(expected)
    for request in manifest["requests"]:
        _paged_require(
            not request.keys()
            - {"phase", "request_id", "q_len", "kv_len", "sequence_id"},
            "unknown paged request fields",
        )
    _paged_require(
        manifest.get("execution_id") == manifest["exec_id"]
        and type(manifest.get("rank")) is int
        and manifest["rank"] == manifest["world_rank"]
        and capture._integer(manifest.get("layer"))
        and capture._integer(manifest.get("call_id")),
        "invalid paged execution identity",
    )
    for name, value in _paged_metadata(rows).items():
        _paged_require(manifest.get(name) == value, "invalid paged " + name)
    _paged_require(
        manifest.get("finite_values_checked") is False
        and type(manifest["query_mapping"].get("next_n")) is int,
        "invalid paged policy types",
    )
    _paged_require(
        isinstance(manifest.get("backend"), str)
        and manifest["backend"] in _PAGED_BACKENDS,
        "unsupported paged backend",
    )
    parameters = manifest.get("parameters")
    _validate_paged_topk_parameters(parameters, rows)
    columns, k = parameters["max_seq_len"], parameters["k"]
    input_names = _PAGED_INPUTS if joint else {"logits", "lengths"}
    output_names = {"logits", "indices"} if joint else {"indices"}
    for field, names in (("inputs", input_names), ("outputs", output_names)):
        _paged_require(
            isinstance(manifest.get(field), list)
            and len(manifest[field]) == len(names)
            and all(isinstance(name, str) for name in manifest[field])
            and set(manifest[field]) == names,
            "invalid paged named " + field,
        )
    _paged_require(
        set(tensors) == input_names | output_names
        and isinstance(manifest.get("tensors"), dict)
        and set(manifest["tensors"]) == set(tensors),
        "invalid paged tensor set",
    )
    for name, tensor in tensors.items():
        _paged_require(
            isinstance(tensor, torch.Tensor)
            and tensor.device.type == "cpu"
            and tensor.is_contiguous(),
            "paged validation requires CPU snapshots",
        )
        desc = manifest["tensors"][name]
        _paged_require(
            isinstance(desc, dict)
            and desc.get("shape") == list(tensor.shape)
            and all(capture._integer(x) for x in desc["shape"])
            and desc.get("dtype") == str(tensor.dtype).removeprefix("torch.")
            and capture._integer(desc.get("nbytes"), 0, capture._MAX_FILE_BYTES)
            and desc["nbytes"] == tensor.numel() * tensor.element_size()
            and desc.get("serialization") == "contiguous_row_major"
            and desc.get("byte_order") == "little"
            and isinstance(desc.get("original_stride"), list)
            and len(desc["original_stride"]) == tensor.ndim
            and all(
                capture._integer(x, 0, capture._MAX_FILE_BYTES)
                for x in desc["original_stride"]
            )
            and capture._integer(desc.get("storage_offset"), 0, 2**63 - 1),
            "paged descriptor mismatch: " + name,
        )
        strides = desc["original_stride"]
        if name in ("logits", "weights", "block_table"):
            _paged_require(
                tensor.ndim == 2 and strides[1] == 1 and strides[0] >= tensor.shape[1],
                "invalid paged source row layout",
            )
        else:
            size = 1
            for dimension, stride in reversed(list(zip(tensor.shape, strides))):
                _paged_require(
                    dimension <= 1 or stride == size,
                    "noncontiguous paged source layout",
                )
                size *= max(1, dimension)
        if name in ("page_ids", "page_valid_tokens", "packed_kv_pages", "block_table"):
            _paged_require(
                desc["storage_offset"] == 0,
                "compact input must have zero storage offset",
            )
    _paged_tensor(tensors["lengths"], torch.int32, (rows,), "lengths")
    _paged_require(
        tensors["lengths"].tolist() == expected and max(expected) <= columns,
        "invalid paged runtime lengths",
    )
    _paged_tensor(tensors["logits"], torch.float32, (rows, columns), "logits")
    _paged_tensor(tensors["indices"], torch.int32, (rows, k), "indices")
    _paged_require(
        manifest["tensors"]["logits"]["original_stride"] == [parameters["stride0"], 1],
        "paged logits source stride mismatch",
    )
    for row, length in enumerate(expected):
        _paged_require(
            bool(torch.all(tensors["logits"][row, length:].view(torch.uint8) == 0)),
            "unsanitized logits tail",
        )
    joint_fields = {"cache_layout", "mqa_parameters", "schedule_environment"}
    _paged_require(
        not (
            {
                "operations",
                "row_starts",
                "row_ends",
                "query_row_start",
                "key_domain",
                "execution_layout",
                "capture_coverage",
            }
            & manifest.keys()
        ),
        "unexpected prefill metadata in paged capture",
    )
    if not joint:
        _paged_require(
            not joint_fields & manifest.keys(), "joint metadata on TopK capture"
        )
    else:
        layout = manifest.get("cache_layout")
        block, dimension, capacity = _validate_paged_layout(layout, expected)
        _paged_require(
            columns % block == 0
            and manifest.get("mqa_parameters")
            == dict(
                backend="deep_gemm.fp8_paged_mqa_logits",
                clean_logits=False,
                max_seq_len=columns,
                weights_include_query_scale=True,
            ),
            "invalid paged MQA parameters",
        )
        _paged_require(
            manifest["mqa_parameters"]["clean_logits"] is False
            and manifest["mqa_parameters"]["weights_include_query_scale"] is True,
            "invalid paged MQA policy types",
        )
        q = tensors["q_fp8"]
        _paged_require(q.ndim == 4 and q.shape[2] > 0, "invalid paged query shape")
        _paged_tensor(q, torch.float8_e4m3fn, (rows, 1, q.shape[2], dimension), "q_fp8")
        _paged_tensor(tensors["weights"], torch.float32, (rows, q.shape[2]), "weights")
        _paged_tensor(tensors["kvlen_2d"], torch.int32, (rows, 1), "kvlen_2d")
        _paged_require(
            tensors["kvlen_2d"].flatten().tolist() == expected,
            "paged context length mismatch",
        )
        _validate_paged_schedule(
            manifest.get("schedule_environment"), tensors["schedule_metadata"], block
        )
        _validate_paged_heads(q.shape[2], manifest["schedule_environment"])
        _validate_paged_schedule_values(tensors["schedule_metadata"], expected)
        _paged_tensor(
            tensors["block_table"], torch.int32, (rows, columns // block), "block_table"
        )
        union = _paged_page_union(
            tensors["block_table"],
            expected,
            block,
            layout["source_page_count"],
            sanitized=True,
        )
        _paged_tensor(tensors["page_ids"], torch.int64, (capacity,), "page_ids")
        _paged_tensor(
            tensors["page_valid_tokens"], torch.int32, (capacity,), "page_valid_tokens"
        )
        stride = dimension + 4 * (dimension // 128)
        _paged_tensor(
            tensors["packed_kv_pages"],
            torch.uint8,
            (capacity, block, 1, stride),
            "packed_kv_pages",
        )
        ids, valid = tensors["page_ids"].tolist(), tensors["page_valid_tokens"].tolist()
        used = len(union)
        _paged_require(
            len(set(ids[:used])) == used
            and set(ids[:used]) == set(union)
            and ids[used:] == [-1] * (capacity - used)
            and valid[used:] == [0] * (capacity - used),
            "invalid compact page mapping",
        )
        for slot in range(capacity):
            count = union[ids[slot]] if slot < used else 0
            _paged_require(
                valid[slot] == count, "page valid tokens differ from reference union"
            )
            blob = tensors["packed_kv_pages"][slot].view(-1)
            scale = block * dimension
            _paged_require(
                bool(torch.all(blob[count * dimension : scale] == 0))
                and bool(
                    torch.all(blob[scale + count * (dimension // 128) * 4 :] == 0)
                ),
                "unsanitized packed page padding",
            )
    return dict(
        rows=rows,
        columns=columns,
        k=k,
        backend=manifest["backend"],
        phase="decode",
        mqa_kernel_replayed=False,
    )


def register_contracts():
    # Called explicitly by the core's builtin-registration entry point, never by JSON.
    for name, version, aliases in (
        (_TOPK, 1, ()),
        (_JOINT, 2, (("indexer.mqa_logits", "indexer.topk"),)),
    ):
        for contract_version, envelope, fields in (
            (1, version, _METADATA_FIELDS),
            (2, 3, (_METADATA_FIELDS - {"query_row_start"}) | _INDEXED_FIELDS),
        ):
            capture._register_contract(
                _IndexerContract(
                    name=name,
                    version=contract_version,
                    envelope_version=envelope,
                    aliases=aliases,
                    selection_group="indexer.capture",
                    metadata_fields=fields,
                    token_type=_IndexerCaptureToken,
                )
            )
        capture._register_contract(
            _PagedIndexerContract(
                name=name,
                version=3,
                envelope_version=3,
                aliases=aliases,
                selection_group="indexer.capture",
                metadata_fields=_PAGED_METADATA_FIELDS,
                token_type=_IndexerCaptureToken,
            )
        )


class IndexerCaptureContext:
    def __init__(self, context):
        self.context = context

    @property
    def _recorder(self):
        return self.context._recorder

    @property
    def identity(self):
        return self.context.identity

    @property
    def mqa_enabled(self):
        return self.context.is_selected(_JOINT)

    def selected(self, layer):
        return self.context.selected(_JOINT if self.mqa_enabled else _TOPK, layer)

    def note_skip(self, reason, layer=None):
        self.context.note_skip(reason, layer)

    def _manifest(
        self,
        rows,
        query_row_start,
        total_query_rows,
        query_global_ids,
        local_query_row_start,
        local_query_rows,
        cp_rank,
        cp_size,
        kv_cache_sharded,
    ):
        metadata = dict(
            query_row_start=query_row_start,
            total_query_rows=total_query_rows,
            query_row_count=rows,
            invalid_logits="zero_filled_snapshot_only",
            finite_values_checked=False,
            nonfinite_policy="valid_domain_bits_preserved_backend_semantics",
            output_contract="raw_relative_indices_minus_one_padding_backend_order",
        )
        if query_global_ids is not None:
            if query_row_start is not None:
                raise ValueError("indexed mapping cannot specify query_row_start")
            del metadata["query_row_start"]
            metadata.update(
                query_mapping=dict(
                    kind="indexed",
                    tensor="query_global_ids",
                    index_space="request_order_query_rows",
                ),
                key_domain=dict(
                    layout="request_order_prefix_then_query", coverage="full"
                ),
                execution_layout=dict(
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                    kv_cache_sharded=kv_cache_sharded,
                ),
                capture_coverage=dict(
                    scope="record_owner_local",
                    local_query_rows=local_query_rows,
                    local_query_row_start=local_query_row_start,
                ),
            )
        elif (
            local_query_row_start is not None
            or local_query_rows is not None
            or type(cp_rank) is not int
            or cp_rank != 0
            or type(cp_size) is not int
            or cp_size != 1
            or kv_cache_sharded is not False
        ):
            raise ValueError("CP mapping requires query_global_ids")
        return metadata

    def begin_paged_topk(
        self,
        *,
        layer,
        logits,
        lengths,
        k,
        backend,
        parameters,
        total_query_rows,
    ):
        if self.mqa_enabled:
            self.note_skip("joint_capture_required", layer)
            return None
        if not self.context.selected(_TOPK, layer):
            return None
        try:
            expected = _paged_expected_lengths(self.identity, total_query_rows)
            _paged_require(backend in _PAGED_BACKENDS, "unsupported paged TopK backend")
            _validate_paged_topk_parameters(parameters, len(expected))
            _paged_require(type(k) is int and k == parameters["k"], "paged k mismatch")
            _paged_tensor(
                logits,
                torch.float32,
                (len(expected), parameters["max_seq_len"]),
                "logits",
                contiguous=False,
            )
            _paged_tensor(lengths, torch.int32, (len(expected),), "lengths")
            _paged_require(
                tuple(logits.stride()) == (parameters["stride0"], parameters["stride1"])
                and max(expected) <= parameters["max_seq_len"],
                "paged logits layout mismatch",
            )
            _paged_source_devices({"logits": logits, "lengths": lengths})
            return self.context.begin(
                op=_TOPK,
                contract_version=3,
                layer=layer,
                inputs=dict(logits=logits, lengths=lengths),
                outputs={
                    "indices": capture.TensorSpec(
                        (len(expected), k), torch.int32, logits.device
                    )
                },
                backend=backend,
                parameters=parameters,
                metadata=_paged_metadata(len(expected)),
            )
        except Exception:
            self.note_skip("invalid_paged_topk_contract", layer)
            return None

    def begin_paged_mqa(
        self,
        *,
        layer,
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        lengths,
        k,
        block_size,
        max_seq_len,
        total_query_rows,
        schedule_environment,
    ):
        if not self.mqa_enabled:
            self.note_skip("mqa_disabled", layer)
            return None
        if not self.context.selected(_JOINT, layer):
            return None
        try:
            expected = _paged_expected_lengths(self.identity, total_query_rows)
            rows = len(expected)
            _paged_require(
                type(k) is int and k in (512, 1024, 2048), "unsupported paged k"
            )
            _paged_require(
                capture._integer(block_size, 1)
                and capture._integer(max_seq_len, 1)
                and max_seq_len % block_size == 0
                and max(expected) <= max_seq_len,
                "invalid paged sequence capacity",
            )
            _paged_require(
                isinstance(q_fp8, torch.Tensor)
                and q_fp8.ndim == 4
                and q_fp8.shape[2] > 0,
                "invalid paged query",
            )
            dimension = q_fp8.shape[3]
            _paged_tensor(
                q_fp8,
                torch.float8_e4m3fn,
                (rows, 1, q_fp8.shape[2], dimension),
                "q_fp8",
            )
            _paged_require(dimension == 128, "unsupported paged head dimension")
            _paged_tensor(
                weights,
                torch.float32,
                (rows, q_fp8.shape[2]),
                "weights",
                contiguous=False,
            )
            _paged_require(weights.stride(1) == 1, "invalid weights stride")
            _paged_tensor(context_lens, torch.int32, (rows, 1), "kvlen_2d")
            _paged_tensor(lengths, torch.int32, (rows,), "lengths")
            _paged_tensor(
                block_table,
                torch.int32,
                (rows, max_seq_len // block_size),
                "block_table",
                contiguous=False,
            )
            _paged_require(block_table.stride(1) == 1, "invalid block table stride")
            _validate_paged_schedule(
                schedule_environment, schedule_metadata, block_size
            )
            _validate_paged_heads(q_fp8.shape[2], schedule_environment)
            _paged_require(
                isinstance(kv_cache_fp8, torch.Tensor)
                and kv_cache_fp8.ndim == 4
                and kv_cache_fp8.shape[0] > 0,
                "invalid packed cache",
            )
            stride = dimension + 4 * (dimension // 128)
            _paged_tensor(
                kv_cache_fp8,
                torch.uint8,
                (kv_cache_fp8.shape[0], block_size, 1, stride),
                "packed cache",
            )
            capacity = sum(
                (length + block_size - 1) // block_size for length in expected
            )
            metadata = _paged_metadata(rows)
            metadata.update(
                cache_layout=dict(
                    version=1,
                    block_size=block_size,
                    head_dim=dimension,
                    scale_group_size=128,
                    page_capacity=capacity,
                    source_page_count=kv_cache_fp8.shape[0],
                    layout="page_payload_then_float32_scales",
                ),
                mqa_parameters=dict(
                    backend="deep_gemm.fp8_paged_mqa_logits",
                    clean_logits=False,
                    max_seq_len=max_seq_len,
                    weights_include_query_scale=True,
                ),
                schedule_environment=dict(schedule_environment),
            )
            # Borrow only until begin returns. The contract transforms these sources
            # under the generic reservation; no pool descriptor enters the artifact.
            inputs = dict(
                q_fp8=q_fp8,
                weights=weights,
                kvlen_2d=context_lens,
                lengths=lengths,
                block_table=block_table,
                schedule_metadata=schedule_metadata,
                packed_kv_pages=kv_cache_fp8,
                page_ids=block_table,
                page_valid_tokens=lengths,
            )
            _paged_source_devices(inputs)
            return self.context.begin(
                op=_JOINT,
                contract_version=3,
                layer=layer,
                inputs=inputs,
                outputs={
                    "logits": capture.TensorSpec(
                        (rows, max_seq_len), torch.float32, q_fp8.device
                    ),
                    "indices": capture.TensorSpec((rows, k), torch.int32, q_fp8.device),
                },
                backend="pending",
                parameters={},
                metadata=metadata,
            )
        except Exception:
            self.note_skip("invalid_paged_mqa_contract", layer)
            return None

    def begin_topk(
        self,
        *,
        layer,
        logits,
        row_starts,
        row_ends,
        k,
        backend,
        parameters,
        query_row_start,
        total_query_rows,
        query_global_ids=None,
        local_query_row_start=None,
        local_query_rows=None,
        cp_rank=0,
        cp_size=1,
        kv_cache_sharded=False,
    ):
        if self.mqa_enabled:
            self.note_skip("joint_capture_required", layer)
            return None
        if not self.context.selected(_TOPK, layer):
            return None
        try:
            if (
                logits.dtype != torch.float32
                or logits.ndim != 2
                or logits.shape[0] == 0
            ):
                raise ValueError("only nonempty float32 logits supported")
            rows, columns = logits.shape
            metadata = self._manifest(
                rows,
                query_row_start,
                total_query_rows,
                query_global_ids,
                local_query_row_start,
                local_query_rows,
                cp_rank,
                cp_size,
                kv_cache_sharded,
            )
            _validate_mapping(
                self.identity,
                rows,
                columns,
                row_starts,
                row_ends,
                k,
                metadata,
                query_global_ids,
            )
            inputs = dict(row_starts=row_starts, row_ends=row_ends, logits=logits)
            if query_global_ids is not None:
                inputs["query_global_ids"] = query_global_ids
                if any(t.device != logits.device for t in inputs.values()):
                    raise ValueError("indexed TopK input devices must match")
            return self.context.begin(
                op=_TOPK,
                contract_version=2 if query_global_ids is not None else 1,
                layer=layer,
                inputs=inputs,
                outputs={"indices": capture.TensorSpec((rows, k), torch.int32)},
                backend=backend,
                parameters=parameters,
                metadata=metadata,
            )
        except Exception:
            self.note_skip("invalid_topk_or_snapshot_error", layer)
            return None

    def begin_mqa(
        self,
        *,
        layer,
        q_fp8,
        k_fp8,
        k_scale,
        weights,
        row_starts,
        row_ends,
        k,
        query_row_start,
        total_query_rows,
        query_global_ids=None,
        local_query_row_start=None,
        local_query_rows=None,
        cp_rank=0,
        cp_size=1,
        kv_cache_sharded=False,
    ):
        if not self.mqa_enabled:
            self.note_skip("mqa_disabled", layer)
            return None
        if not self.context.selected(_JOINT, layer):
            return None
        try:
            inputs = dict(
                row_starts=row_starts,
                row_ends=row_ends,
                q_fp8=q_fp8,
                k_fp8=k_fp8,
                k_scale=k_scale,
                weights=weights,
            )
            if query_global_ids is not None:
                inputs["query_global_ids"] = query_global_ids
            rows, columns = _validate_mqa_inputs(inputs)
            metadata = self._manifest(
                rows,
                query_row_start,
                total_query_rows,
                query_global_ids,
                local_query_row_start,
                local_query_rows,
                cp_rank,
                cp_size,
                kv_cache_sharded,
            )
            _validate_mapping(
                self.identity,
                rows,
                columns,
                row_starts,
                row_ends,
                k,
                metadata,
                query_global_ids,
            )
            metadata.update(
                mqa_parameters=dict(
                    backend="deep_gemm.fp8_mqa_logits",
                    clean_logits=False,
                    weights_include_query_scale=True,
                    key_scale_format="float32_per_128_channels",
                ),
                operations=[
                    dict(
                        op="indexer.mqa_logits",
                        inputs=[
                            "q_fp8",
                            "k_fp8",
                            "k_scale",
                            "weights",
                            "row_starts",
                            "row_ends",
                        ],
                        outputs=["logits"],
                    ),
                    dict(
                        op="indexer.topk",
                        inputs=["logits", "row_starts", "row_ends"],
                        outputs=["indices"],
                    ),
                ],
            )
            return self.context.begin(
                op=_JOINT,
                contract_version=2 if query_global_ids is not None else 1,
                layer=layer,
                inputs=inputs,
                outputs={
                    "logits": capture.TensorSpec(
                        (rows, columns), torch.float32, q_fp8.device
                    ),
                    "indices": capture.TensorSpec((rows, k), torch.int32, q_fp8.device),
                },
                backend="pending",
                parameters={},
                metadata=metadata,
            )
        except Exception:
            self.note_skip("invalid_mqa_contract", layer)
            return None


def for_context(context) -> Optional[IndexerCaptureContext]:
    if context is None or isinstance(context, IndexerCaptureContext):
        return context
    return IndexerCaptureContext(context)
