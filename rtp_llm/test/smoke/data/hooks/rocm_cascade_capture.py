"""Observation-only hot hooks for the manual MI308 real-tensor capture test.

The controller arms one sequential request after service readiness. Only its
first decode is captured. This is eager model evidence, not a scheduler batch
or a performance measurement. No prompt, input token IDs or response is saved.
"""

import hashlib
import importlib.metadata
import json
import os
import sys
import threading
import time
from pathlib import Path

import torch

_state = threading.local()
_capture_lock = threading.Lock()


class _CaptureFailure(Exception):
    """Carries only a static, data-free validation code."""


def _arg(ctx, position, name, default=None):
    return (
        ctx.args[position]
        if len(ctx.args) > position
        else ctx.kwargs.get(name, default)
    )


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu(value):
    # Never serialize a view's unrelated backing storage (e.g. the packed QKV).
    return value.detach().cpu().clone(memory_format=torch.contiguous_format)


def _tensor_hash(value):
    return hashlib.sha256(
        value.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _tensor_metadata(value):
    span = (
        0
        if not value.numel()
        else 1
        + sum((size - 1) * stride for size, stride in zip(value.shape, value.stride()))
    )
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "storage_offset": value.storage_offset(),
        "dtype": str(value.dtype),
        "element_size": value.element_size(),
        "numel_bytes": value.numel() * value.element_size(),
        "view_span_bytes": span * value.element_size(),
        "storage_nbytes": value.untyped_storage().nbytes(),
        "data_ptr_mod_256": value.data_ptr() % 256,
    }


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return getattr(sys.modules.get(package), "__version__", "unavailable")


def enter_layer(ctx):
    stack = getattr(_state, "stack", None)
    if stack is None:
        stack = _state.stack = []
    stack.append(None)
    control = json.loads(Path(ctx.config["control_file"]).read_text())
    if not control.get("armed"):
        return None
    layer = int(_arg(ctx, 3, "layer_idx", 0))
    if layer in control["layers"]:
        stack[-1] = (layer, control, _arg(ctx, 0, "self"))
    return None


def leave_layer(ctx):
    stack = getattr(_state, "stack", None)
    if stack:
        stack.pop()
    return None


def _require(condition, code):
    if not condition:
        # Only static error codes; tensor/payload repr must never enter logs.
        raise _CaptureFailure(code)


def _capture(ctx, layer, control, outer, destination):
    op = _arg(ctx, 0, "self")
    query = _arg(ctx, 1, "query")
    cache = _arg(ctx, 2, "kv_cache")
    params = _arg(ctx, 3, "fmha_params")
    heads, kv_heads, dim, page = (
        op.head_num,
        op.head_num_kv,
        op.head_dim,
        op.tokens_per_block,
    )
    _require(
        (heads, kv_heads, dim, page) == (16, 8, 128, 16), "unsupported_attention_shape"
    )
    _require(
        not op.linear_v and not op.enable_cuda_graph, "expected_eager_vectorized_v"
    )
    _require(not getattr(outer.attn_inputs, "is_prefill", True), "expected_decode")
    _require(query.is_cuda and torch.version.hip is not None, "expected_rocm_query")
    _require(query.dtype == torch.bfloat16, "expected_bf16_query")
    rows = query.shape[0]
    _require(rows == control["expected_decode_rows"], "unexpected_decode_rows")
    _require(query.numel() == rows * heads * dim, "unexpected_query_elements")
    paged = op.reshape_kv_cache(cache.kv_cache_base)
    _require(paged.dtype == torch.bfloat16, "expected_bf16_cache")
    _require(
        tuple(paged.shape[1:]) == (2, kv_heads, page, dim), "unexpected_paged_layout"
    )
    scales = cache.kv_scale_base
    _require(
        scales is None or (isinstance(scales, torch.Tensor) and scales.numel() == 0),
        "unexpected_kv_scales",
    )
    _require(params.kv_scale is None, "unexpected_query_scales")

    seq_lens, history = _cpu(params.seq_lens), _cpu(params.sequence_lengths)
    table = _cpu(params.kv_cache_block_id_device)
    _require(seq_lens.ndim == 1 and seq_lens.numel() == rows, "invalid_seq_lens")
    _require(torch.equal(seq_lens, history + 1), "history_plus_one_mismatch")
    _require(table.ndim == 2 and table.shape[0] == rows, "invalid_block_table")
    _require(bool((seq_lens > 0).all()), "invalid_attended_lengths")
    _require(int(seq_lens.max()) <= table.shape[1] * page, "insufficient_block_table")
    positions = torch.arange(table.shape[1], dtype=torch.int64)[None, :] * page
    used_tokens = (seq_lens.to(torch.int64)[:, None] - positions).clamp(0, page)
    valid = used_tokens > 0
    physical_ids, inverse = torch.unique(
        table[valid].to(torch.int64),
        sorted=True,
        return_inverse=True,
    )
    _require(
        int(physical_ids.min()) >= 0 and int(physical_ids.max()) < paged.shape[0],
        "physical_page_out_of_range",
    )
    logical_table = torch.full_like(table, -1)
    logical_table[valid] = inverse.to(table.dtype)
    live_pages = _cpu(paged.index_select(0, physical_ids.to(paged.device)))
    live_tokens_per_page = torch.zeros(physical_ids.numel(), dtype=torch.int64)
    live_tokens_per_page.scatter_reduce_(
        0,
        inverse,
        used_tokens[valid],
        reduce="amax",
        include_self=True,
    )

    # Check only attended positions: the unused tail of a real page can contain
    # uninitialized bytes. Independently unpack the actual vectorized BF16 KV.
    count = physical_ids.numel()
    key = live_pages[:, 0].reshape(count, kv_heads, dim // 8, page, 8)
    key = key.permute(0, 1, 3, 2, 4).reshape(count, kv_heads, page, dim)
    value = live_pages[:, 1].reshape(count, kv_heads, page // 8, dim, 8)
    value = value.permute(0, 1, 2, 4, 3).reshape(count, kv_heads, page, dim)
    used = torch.arange(page)[None, :] < live_tokens_per_page[:, None]
    _require(
        bool(torch.isfinite(key.permute(0, 2, 1, 3)[used]).all()), "nonfinite_live_keys"
    )
    _require(
        bool(torch.isfinite(value.permute(0, 2, 1, 3)[used]).all()),
        "nonfinite_live_values",
    )
    q, off_output = _cpu(query), _cpu(ctx.result)
    _require(off_output.numel() == q.numel(), "unexpected_off_output_elements")
    _require(
        bool(torch.isfinite(q).all()) and bool(torch.isfinite(off_output).all()),
        "nonfinite_q_or_output",
    )

    tensors = {
        "q": q,
        "kv_pages": live_pages,
        "physical_page_ids": physical_ids,
        "live_tokens_per_page": live_tokens_per_page,
        "block_table": table,
        "logical_block_table": logical_table,
        "seq_lens": seq_lens,
        "history": history,
        "input_lengths": _cpu(outer.attn_inputs.input_lengths),
        "off_output": off_output,
    }
    tensor_metadata = {
        name: {"shape": list(t.shape), "dtype": str(t.dtype), "sha256": _tensor_hash(t)}
        for name, t in tensors.items()
    }
    temporary = destination / "tensors.pt.tmp"
    torch.save(tensors, temporary)
    temporary.chmod(0o600)
    os.replace(temporary, destination / "tensors.pt")
    properties = torch.cuda.get_device_properties(query.device)
    manifest = {
        "schema_version": 1,
        "evidence": "RTP_MEASURED",
        "fixture_kind": "real_model_beam_capture",
        "capture_mode": "eager",
        "coverage": "first_decode_of_one_sequential_request; representative_layer",
        "capture_time_unix": time.time(),
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "request_ordinal": control["request_ordinal"],
        "request_sha256": control["request_sha256"],
        "provenance_sha256": control["provenance_sha256"],
        "layer_idx": layer,
        "model_num_layers": control["model_num_layers"],
        "kv_cache_mem_mb_configured": control["kv_cache_mem_mb_configured"],
        "decode_step": 0,
        "rows": rows,
        "request_boundaries": [0, rows],
        "head_num": heads,
        "kv_head_num": kv_heads,
        "head_dim": dim,
        "tokens_per_block": page,
        "linear_v": False,
        "kv_layout": "K[N,H,D/8,P,8];V[N,H,P/8,D,8];BF16",
        "query_metadata": _tensor_metadata(query),
        "kv_base_metadata": _tensor_metadata(cache.kv_cache_base),
        "paged_kv_metadata": _tensor_metadata(paged),
        "off_output_metadata": _tensor_metadata(ctx.result),
        "max_seq_len": int(params.max_seq_len),
        "configured_max_seq_len": int(op.max_seq_len),
        "history_lengths": sorted(set(history.tolist())),
        "attended_lengths": sorted(set(seq_lens.tolist())),
        "live_page_count": count,
        "minimum_live_byte_offset": int(physical_ids.min())
        * paged.stride(0)
        * paged.element_size(),
        "maximum_live_page_byte_offset": int(physical_ids.max())
        * paged.stride(0)
        * paged.element_size(),
        "finite_attended_kv": True,
        "tensors": tensor_metadata,
        "tensor_file_sha256": _sha_file(destination / "tensors.pt"),
        "runtime": {
            "torch": torch.__version__,
            "rocm": torch.version.hip,
            "triton": _version("triton"),
            "aiter": _version("aiter"),
            "device_name": properties.name,
            "gcn_arch": getattr(properties, "gcnArchName", None),
            "device_index": query.device.index,
            "HIP_VISIBLE_DEVICES": os.getenv("HIP_VISIBLE_DEVICES"),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
        },
        "kernel_replay_accuracy": "NOT_RUN",
        "end_to_end_on_off_accuracy": "NOT_RUN",
    }
    # The manifest is the completion marker and is published only after tensors.
    _write_json(destination / "manifest.json", manifest)


def capture_after(ctx):
    stack = getattr(_state, "stack", None)
    if not stack or stack[-1] is None:
        return None
    layer, control, outer = stack[-1]
    root = Path(ctx.config["capture_root"])
    destination = (
        root / f"request_{control['request_ordinal']:04d}" / f"layer_{layer:03d}"
    )
    with _capture_lock:
        if (destination / "manifest.json").exists() or (
            destination / "error.json"
        ).exists():
            return None
        destination.mkdir(parents=True, mode=0o700, exist_ok=True)
        destination.chmod(0o700)
        try:
            _capture(ctx, layer, control, outer, destination)
        except Exception as error:
            _write_json(
                destination / "error.json",
                {
                    "evidence": "FAIL",
                    "exception_type": type(error).__name__,
                    "code": (
                        str(error)
                        if isinstance(error, _CaptureFailure)
                        else "capture_failed"
                    ),
                    "layer_idx": layer,
                    "request_ordinal": control["request_ordinal"],
                },
            )
    return None
