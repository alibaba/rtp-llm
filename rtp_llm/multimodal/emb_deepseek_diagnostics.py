"""Opt-in, process-local evidence for DeepSeek-VL2's (embedding, None).

Snapshots never provide model operands. Only one in-flight image, TP1 and eager
execution are supported; missing or ambiguous observations fail closed. This
module never reads a writer's files, image contents, or a previous-run ledger.
"""

import contextlib
import hashlib
import json
import logging
import math
import os
import struct
import threading
import time
from pathlib import Path

import torch

from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate

_MODEL = "deepseek_vl_v2"
_MAX_BYTES = 256 * 1024 * 1024
_observer = None
_lock = threading.RLock()
_local = threading.local()


def enabled(model_type=None):
    return (
        os.environ.get("DEEPSEEK_EMB_DIAGNOSTICS") == "1"
        and bool(os.environ.get("DEEPSEEK_EMB_DIAGNOSTICS_DIR"))
        and model_type in (None, _MODEL)
    )


def _sha(cpu):
    raw = cpu.contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(memoryview(raw.numpy())).hexdigest()


def _snapshot(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise ValueError("unsupported diagnostic tensor")
    nbytes = tensor.numel() * tensor.element_size()
    if nbytes > _MAX_BYTES:
        raise ValueError("diagnostic tensor limit")
    with cuda_graph_gate.operation():
        if tensor.is_cuda:
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("CUDA capture outside diagnostic scope")
            torch.cuda.synchronize(tensor.device)
        cpu = tensor.detach().to("cpu").contiguous().clone()
    return cpu, dict(
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        nbytes=nbytes,
        sha256=_sha(cpu),
    )


def _same(a, b):
    return all(a[k] == b[k] for k in ("shape", "dtype", "nbytes", "sha256"))


def _get():
    global _observer
    identity = (os.getpid(), os.environ["DEEPSEEK_EMB_DIAGNOSTICS_DIR"])
    with _lock:
        if _observer is None or _observer.identity != identity:
            _observer = Observer(identity)
        return _observer


def _call(method, *args):
    if not enabled():
        return None
    try:
        observer = _get()
        with observer.lock:
            return getattr(observer, method)(*args)
    except Exception as error:
        try:
            observer = _get()
            with observer.lock:
                traces = list(observer.requests) or [None]
                for trace in traces:
                    if trace in observer.requests:
                        observer.requests[trace]["unproven"] = True
                for trace in traces:
                    observer.emit(
                        "unproven",
                        trace,
                        reason="diagnostic_exception",
                        hook=method,
                        error_type=type(error).__name__,
                    )
        except Exception:
            try:
                logging.warning(
                    "DeepSeek EMB diagnostics unavailable (%s)", type(error).__name__
                )
            except Exception:
                pass
        return None


class Observer:
    def __init__(self, identity):
        self.identity = identity
        self.lock = threading.RLock()
        directory = Path(identity[1])
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "emb-chain.jsonl"
        self.run_id = os.environ.get("DEEPSEEK_EMB_DIAGNOSTICS_RUN_ID", directory.name)
        try:
            self.start_ticks = (
                Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
            )
        except (OSError, IndexError):
            self.start_ticks = None
        self.requests = {}

    def emit(self, event, trace_id=None, **fields):
        record = dict(
            schema="rtp-deepseek-emb-chain-v1",
            event=event,
            timestamp_ns=time.time_ns(),
            pid=self.identity[0],
            proc_start_ticks=self.start_ticks,
            run_id=self.run_id,
            trace_id=trace_id,
            key=self.requests.get(trace_id, {}).get("key"),
            **fields
        )
        data = memoryview(
            (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
        )
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            while data:
                written = os.write(fd, data)
                if written <= 0:
                    raise OSError("diagnostic log write failed")
                data = data[written:]
        finally:
            os.close(fd)

    def fail(self, trace, reason, **fields):
        if trace in self.requests:
            self.requests[trace]["unproven"] = True
        self.emit("unproven", trace, reason=reason, **fields)

    def start(self, trace, count):
        if os.environ.get("ENABLE_CUDA_GRAPH", "false").lower() not in ("false", "0"):
            self.fail(trace, "cuda_graph_configuration_unsupported")
            return
        if self.requests:
            for previous in list(self.requests):
                self.fail(previous, "concurrent_or_unfinished_request")
            self.fail(trace, "concurrent_or_unfinished_request")
            return
        if count != 1:
            self.fail(trace, "single_image_required", image_count=count)
            return
        self.requests[trace] = {"unproven": False, "key": None}
        self.emit("request_start", trace, model_type=_MODEL, image_count=count)

    def bind(self, trace, key):
        if trace in self.requests:
            self.requests[trace]["key"] = key

    def vision(self, trace, features, pixels, vision):
        if trace not in self.requests:
            return
        _, info = _snapshot(features)
        self.requests[trace]["vision"] = info
        self.emit(
            "vision_output",
            trace,
            tensor=info,
            vision_class=type(vision).__name__,
            pixel_values=dict(
                shape=list(pixels.shape),
                dtype=str(pixels.dtype),
                device=str(pixels.device),
            ),
        )

    def projector(
        self,
        trace,
        raw,
        projected,
        newline,
        separator,
        width_tiles,
        height_tiles,
        global_view_pos,
        tile_tag,
    ):
        if trace not in self.requests:
            return
        state = self.requests[trace]
        _, raw_info = _snapshot(raw)
        image, info = _snapshot(projected)
        nl, nl_info = _snapshot(newline)
        sep, sep_info = _snapshot(separator)
        if "vision" not in state or not _same(state["vision"], raw_info):
            self.fail(trace, "projector_input_differs_from_vision_output")
        if (
            tile_tag != "2D"
            or global_view_pos not in ("head", "tail")
            or width_tiles <= 0
            or height_tiles <= 0
            or image.ndim != 3
            or image.shape[0] != 1 + width_tiles * height_tiles
        ):
            self.fail(trace, "unsupported_tiling_contract")
            return
        count, hw, hidden = image.shape
        side = math.isqrt(hw)
        if (
            side * side != hw
            or tuple(nl.shape) != (hidden,)
            or tuple(sep.shape) != (hidden,)
        ):
            self.fail(trace, "invalid_projector_shape")
            return
        # Independent CPU assembly of the existing 2D tile layout. These
        # diagnostic copies never flow back into the model's actual result.
        global_image = image[0].reshape(side, side, hidden)
        global_image = torch.cat(
            [global_image, nl.reshape(1, 1, hidden).expand(side, 1, hidden)], dim=1
        ).reshape(-1, hidden)
        local_image = image[1:].reshape(height_tiles, width_tiles, side, side, hidden)
        local_image = local_image.permute(0, 2, 1, 3, 4).reshape(
            height_tiles * side, width_tiles * side, hidden
        )
        local_image = torch.cat(
            [
                local_image,
                nl.reshape(1, 1, hidden).expand(height_tiles * side, 1, hidden),
            ],
            dim=1,
        ).reshape(-1, hidden)
        pieces = (
            [global_image, sep.reshape(1, hidden), local_image]
            if global_view_pos == "head"
            else [local_image, sep.reshape(1, hidden), global_image]
        )
        expected = torch.cat(pieces)
        state["assembled_expected"] = _snapshot(expected)[1]
        self.emit(
            "projector_output",
            trace,
            tensor=info,
            input_tensor=raw_info,
            input_matches_vision="vision" in state and _same(state["vision"], raw_info),
            image_newline=nl_info,
            view_separator=sep_info,
            width_tiles=width_tiles,
            height_tiles=height_tiles,
            global_view_pos=global_view_pos,
            tile_tag=tile_tag,
            assembled_expected=state["assembled_expected"],
        )

    def result(self, trace, result, event, source):
        if trace not in self.requests:
            return
        state = self.requests[trace]
        if not isinstance(result, tuple) or len(result) != 2 or result[1] is not None:
            self.fail(trace, "expected_embedding_none_tuple")
            return
        value = result[0]
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or not all(value.shape)
        ):
            self.fail(trace, "expected_nonempty_embedding_matrix")
            return
        _, info = _snapshot(value)
        previous = (
            state.get("assembled_expected")
            if event == "encoder_tuple"
            else state.get("restored" if source == "remote" else "encoded")
        )
        exact = previous is not None and _same(previous, info)
        if not exact:
            self.fail(trace, "tuple_origin_mismatch", stage=event)
        if event == "encoder_tuple":
            state["encoded"] = info
            state.pop("assembled_expected", None)
        else:
            state["delivered"] = info
        self.emit(
            event,
            trace,
            tensor=info,
            positions=None,
            source=source,
            expected_sha256=None if previous is None else previous["sha256"],
            exact=exact,
        )

    def slices(self, trace, plan, buffer, loading):
        if trace not in self.requests:
            return
        state = self.requests[trace]
        magic, length = struct.unpack("<8sQ", bytes(buffer[:16].numpy()))
        if magic != b"RTPMM001" or length > 65536:
            raise ValueError("invalid diagnostic header")
        header = json.loads(bytes(buffer[16 : 16 + length].numpy()))
        offset = (16 + length + 63) // 64 * 64
        if (
            header["tree"] != ["tuple", [["tensor", 0], ["scalar", None]]]
            or len(header["tensors"]) != 1
            or len(plan.tensors) != 1
        ):
            self.fail(trace, "serialized_tuple_contract_mismatch")
            return
        spec = header["tensors"][0]
        tensor = plan.tensors[0]
        _, info = _snapshot(tensor)
        begin = offset + spec["offset"]
        if (
            spec["offset"] != 0
            or offset != plan.payload_offset
            or buffer.numel() != offset + spec["nbytes"]
            or buffer.numel() != plan.nbytes
        ):
            self.fail(trace, "serialized_slice_range_mismatch")
            return
        wire = buffer[begin : begin + spec["nbytes"]]
        wire_sha = _sha(wire)
        exact = (
            info["sha256"] == wire_sha
            and info["nbytes"] == spec["nbytes"]
            and info["shape"] == spec["shape"]
            and info["dtype"] == spec["dtype"]
            and tensor.device.type == spec["device"]
        )
        origin = state.get("encoded")
        origin_exact = None if loading else origin is not None and _same(origin, info)
        if not exact or origin_exact is False:
            self.fail(trace, "serialized_slice_mismatch")
        if loading:
            state["restored"] = info
        self.emit(
            "unpack_slices" if loading else "pack_slices",
            trace,
            tensor=info,
            positions=None,
            slice_sha256=wire_sha,
            exact=exact,
            origin_sha256=None if origin is None else origin["sha256"],
            origin_exact=origin_exact,
            payload_offset=offset,
            offset=begin,
            payload_nbytes=buffer.numel(),
            payload_sha256=_sha(buffer),
        )

    def model_begin(self, features, locs, tp_size):
        if not features:
            return None
        if len(self.requests) != 1:
            self.fail(None, "ambiguous_model_request", candidates=len(self.requests))
            return None
        trace, state = next(iter(self.requests.items()))
        if tp_size != 1 or len(features) != 1:
            self.fail(trace, "single_image_segment_tp1_required")
            return None
        locations = _snapshot(locs)[0].reshape(-1).tolist()
        if len(locations) != 1 or locations[0] < 0:
            self.fail(trace, "invalid_image_location")
            return None
        _, info = _snapshot(features[0])
        original = state.get("delivered")
        exact = original is not None and _same(original, info)
        if not exact:
            self.fail(trace, "llm_input_differs_from_delivered_tuple")
        self.emit(
            "llm_inputs",
            trace,
            tensor=info,
            location=locations[0],
            expected_sha256=None if original is None else original["sha256"],
            exact=exact,
        )
        return dict(trace=trace, tensor=info, location=locations[0])

    def copied(self, ticket, hidden):
        if ticket is None:
            return
        trace = ticket["trace"]
        source = ticket["tensor"]
        _, actual = _snapshot(hidden.narrow(0, ticket["location"], source["shape"][0]))
        exact = _same(source, actual)
        if not exact:
            self.fail(trace, "llm_embedding_copy_mismatch")
        self.emit(
            "llm_copy",
            trace,
            tensor=actual,
            location=ticket["location"],
            expected_sha256=source["sha256"],
            exact=exact,
            unproven=self.requests[trace]["unproven"],
        )
        del self.requests[trace]

    def eviction_origin(self, trace):
        state = self.requests.get(trace)
        if state is None:
            return None
        values = state.get("encoded") or state.get("restored")
        if values is None:
            return None
        return dict(
            key=state.get("key"),
            tensors=[dict(values)],
            origin_event="encoder_tuple" if "encoded" in state else "unpack_slices",
            unproven=state.get("unproven", True),
        )

    def eviction_slices(self, trace, origin, plan, buffer):
        if origin is None:
            self.emit("unproven", trace, reason="missing_eviction_tensor_origin")
            return
        magic, length = struct.unpack("<8sQ", bytes(buffer[:16].numpy()))
        if magic != b"RTPMM001" or length > 65536:
            raise ValueError("invalid eviction diagnostic header")
        header = json.loads(bytes(buffer[16 : 16 + length].numpy()))
        offset = (16 + length + 63) // 64 * 64
        prior = origin["tensors"]
        expected_count = 1
        if (
            header["tree"] != ["tuple", [["tensor", 0], ["scalar", None]]]
            or len(prior) != expected_count
            or len(plan.tensors) != expected_count
            or len(header["tensors"]) != expected_count
            or offset != plan.payload_offset
            or buffer.numel() != plan.nbytes
        ):
            self.emit("unproven", trace, reason="eviction_tuple_contract_mismatch")
            return
        comparisons = []
        previous_end = 0
        for tensor, spec, expected in zip(plan.tensors, header["tensors"], prior):
            _, actual = _snapshot(tensor)
            begin = offset + spec["offset"]
            end = begin + spec["nbytes"]
            wire_sha = _sha(buffer[begin:end])
            exact = (
                tensor.device.type == "cpu"
                and spec["offset"] == (previous_end + 63) // 64 * 64
                and end <= buffer.numel()
                and actual["sha256"] == wire_sha == expected["sha256"]
                and all(
                    actual[key] == expected[key] == spec[key]
                    for key in ("shape", "dtype", "nbytes")
                )
                and expected["device"].split(":", 1)[0] == spec["device"]
            )
            comparisons.append(
                dict(
                    origin_tensor=expected,
                    staging_tensor=actual,
                    slice_sha256=wire_sha,
                    offset=begin,
                    nbytes=spec["nbytes"],
                    restore_device=spec["device"],
                    exact=exact,
                )
            )
            previous_end = spec["offset"] + spec["nbytes"]
        exact = (
            all(item["exact"] for item in comparisons)
            and offset + previous_end == buffer.numel()
            and not origin["unproven"]
        )
        self.emit(
            "eviction_pack_slices",
            trace,
            origin_key=origin["key"],
            origin_event=origin["origin_event"],
            tensors=comparisons,
            payload_offset=offset,
            payload_nbytes=buffer.numel(),
            payload_sha256=_sha(buffer),
            exact=exact,
        )


def request_start(trace, count, model_type):
    if enabled(model_type) and model_type == _MODEL:
        _call("start", trace, count)


def bind_key(trace, key):
    _call("bind", trace, key)


@contextlib.contextmanager
def encoder_scope(items):
    if not enabled():
        yield
        return
    previous = getattr(_local, "trace", None)
    _local.trace = (
        getattr(items[0], "deepseek_diagnostic_trace", None)
        if len(items) == 1
        else None
    )
    try:
        yield
    finally:
        _local.trace = previous


def vision_output(features, pixels, vision):
    _call("vision", getattr(_local, "trace", None), features, pixels, vision)


def projector_output(
    raw,
    projected,
    newline,
    separator,
    width_tiles,
    height_tiles,
    global_view_pos,
    tile_tag,
):
    _call(
        "projector",
        getattr(_local, "trace", None),
        raw,
        projected,
        newline,
        separator,
        width_tiles,
        height_tiles,
        global_view_pos,
        tile_tag,
    )


def encoder_tuple(trace, result):
    _call("result", trace, result, "encoder_tuple", "encoder")


def delivered(trace, result, source):
    _call("result", trace, result, "tuple_delivered", source)


def serialized(trace, plan, buffer, loading=False):
    _call("slices", trace, plan, buffer, loading)


def model_begin(features, locs, model_type, tp_size):
    if enabled(model_type) and model_type == _MODEL:
        return _call("model_begin", features, locs, tp_size)
    return None


def embedding_copied(ticket, hidden):
    _call("copied", ticket, hidden)


def eviction_origin(trace):
    return _call("eviction_origin", trace)


def eviction_serialized(trace, origin, plan, buffer):
    _call("eviction_slices", trace, origin, plan, buffer)
