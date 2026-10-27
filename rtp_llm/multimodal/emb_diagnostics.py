"""Opt-in evidence only. CPU snapshots never supply operands to inference.

Supported proof scope: one in-flight image, Qwen3-VL, TP/CP=1, eager execution.
Each process maintains its own bounded ledger; no previous-run file is read.
Failures are recorded as unproven and never change the normal model result.
"""

import contextlib
import hashlib
import json
import logging
import os
import struct
import threading
import time
from pathlib import Path

import torch

from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate

_LABELS = ("embedding", "positions", "deepstack")
_MAX_BYTES = 256 * 1024 * 1024
_local = threading.local()
_lock = threading.RLock()
_observer = None


def enabled():
    return os.environ.get("MM_EMB_DIAGNOSTICS") == "1" and bool(
        os.environ.get("MM_EMB_DIAGNOSTICS_DIR")
    )


def _get():
    global _observer
    if not enabled():
        return None
    identity = (os.getpid(), os.environ["MM_EMB_DIAGNOSTICS_DIR"])
    with _lock:
        if _observer is None or _observer.identity != identity:
            _observer = Observer(identity)
        return _observer


def _call(method, *args, **kwargs):
    if not enabled():
        return None
    try:
        observer = _get()
        with observer.lock:
            return getattr(observer, method)(*args, **kwargs)
    except Exception as error:
        # Evidence failure must not become an inference failure. Do not include
        # exception messages: they may contain user content or tensor data.
        try:
            observer = _get()
            with observer.lock:
                traces = list(observer.requests) or [None]
                # A log failure after a ticket's seen flag must never permit a
                # later completion to claim this request was fully observed.
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
                    "EMB diagnostics unavailable (%s)", type(error).__name__
                )
            except Exception:
                pass
        return None


def _sha(cpu):
    raw = cpu.contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(memoryview(raw.numpy())).hexdigest()


def _snapshot(value):
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise ValueError("unsupported diagnostic tensor")
    if value.numel() * value.element_size() > _MAX_BYTES:
        raise ValueError("diagnostic tensor limit")
    with cuda_graph_gate.operation():
        if value.is_cuda:
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("capture is outside diagnostic scope")
            torch.cuda.synchronize(value.device)
        cpu = value.detach().to(device="cpu").contiguous().clone()
    return cpu, dict(
        shape=list(value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
        nbytes=value.numel() * value.element_size(),
        sha256=_sha(cpu),
    )


def _equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and _sha(a) == _sha(b)


class Observer:
    def __init__(self, identity):
        self.identity = identity
        self.lock = threading.RLock()
        directory = Path(identity[1])
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "emb-chain.jsonl"
        self.run_id = os.environ.get("MM_EMB_DIAGNOSTICS_RUN_ID", directory.name)
        try:
            self.start_ticks = (
                Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
            )
        except (OSError, IndexError):
            self.start_ticks = None
        self.requests = {}

    def emit(self, event, trace_id=None, **fields):
        state = self.requests.get(trace_id, {})
        record = dict(
            schema="rtp-emb-chain-v1",
            event=event,
            timestamp_ns=time.time_ns(),
            pid=self.identity[0],
            proc_start_ticks=self.start_ticks,
            run_id=self.run_id,
            trace_id=trace_id,
            key=state.get("key"),
            **fields
        )
        data = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            remaining = memoryview(data)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError("diagnostic log write failed")
                remaining = remaining[written:]
        finally:
            os.close(fd)

    def fail(self, trace_id, reason, **fields):
        if trace_id in self.requests:
            self.requests[trace_id]["unproven"] = True
        self.emit("unproven", trace_id, reason=reason, **fields)

    def start(self, trace_id, image_count):
        if os.environ.get("ENABLE_CUDA_GRAPH", "false").lower() not in ("false", "0"):
            self.emit(
                "unproven", trace_id, reason="cuda_graph_configuration_unsupported"
            )
            return
        if self.requests:
            for previous in list(self.requests):
                self.fail(previous, "concurrent_or_unfinished_request")
            self.emit("unproven", trace_id, reason="concurrent_or_unfinished_request")
            return
        if image_count != 1:
            self.emit(
                "unproven",
                trace_id,
                reason="single_image_required",
                image_count=image_count,
            )
            return
        self.requests[trace_id] = {"key": None, "unproven": False}
        self.emit("request_start", trace_id, image_count=image_count)

    def bind(self, trace_id, key):
        if trace_id in self.requests:
            self.requests[trace_id]["key"] = key

    def vision(
        self, trace_id, output, grid, merge_size, pixel_values=None, visual=None
    ):
        if trace_id not in self.requests:
            return
        tensors = [output.pooler_output, *output.deepstack_features]
        if len(tensors) != 4 or tuple(grid.shape) != (1, 3):
            self.fail(trace_id, "vision_requires_single_image_three_deepstack")
            return
        snaps = [_snapshot(t) for t in tensors]
        grid_cpu, grid_info = _snapshot(grid)
        self.requests[trace_id]["vision"] = [p[0] for p in snaps]
        self.requests[trace_id]["grid"] = grid_cpu
        self.requests[trace_id]["merge"] = int(merge_size)
        pixel_info = (
            None
            if pixel_values is None
            else dict(
                shape=list(pixel_values.shape),
                dtype=str(pixel_values.dtype),
                device=str(pixel_values.device),
            )
        )
        self.emit(
            "vision_output",
            trace_id,
            pixel_values=pixel_info,
            vision_class=None if visual is None else type(visual).__name__,
            attention_implementation=getattr(
                getattr(visual, "config", None), "_attn_implementation", None
            ),
            tensors=[
                dict(label=label, **info)
                for label, (_, info) in zip(
                    ("pooler", "deepstack.0", "deepstack.1", "deepstack.2"), snaps
                )
            ],
            grid=grid_info,
            grid_thw=grid_cpu.tolist(),
            spatial_merge_size=int(merge_size),
        )

    def tuple_snapshot(self, trace_id, result, event, source=None):
        if trace_id not in self.requests:
            return
        state = self.requests[trace_id]
        if len(result) != 3 or not all(isinstance(t, torch.Tensor) for t in result):
            self.fail(trace_id, "three_tensor_tuple_required")
            return
        snaps = [_snapshot(t) for t in result]
        metadata = [
            dict(label=label, **info) for label, (_, info) in zip(_LABELS, snaps)
        ]
        comparisons = []
        if event == "encoder_tuple":
            vision = state.pop("vision", None)
            if vision is None:
                self.fail(trace_id, "missing_raw_vision_output")
            else:
                expected_embedding = vision[0].to(snaps[0][0].dtype)
                expected_deepstack = (
                    torch.stack(vision[1:]).to(snaps[2][0].dtype).flatten()
                )
                grid = state["grid"][0].tolist()
                t, h, w = grid[0], grid[1] // state["merge"], grid[2] // state["merge"]
                expected_positions = (
                    torch.stack(
                        torch.meshgrid(
                            torch.arange(t),
                            torch.arange(h),
                            torch.arange(w),
                            indexing="ij",
                        ),
                        dim=-1,
                    )
                    .reshape(-1, 3)
                    .to(snaps[1][0].dtype)
                )
                for label, expected, (actual, _) in zip(
                    _LABELS,
                    (expected_embedding, expected_positions, expected_deepstack),
                    snaps,
                ):
                    comparisons.append(
                        dict(
                            label=label,
                            expected_sha256=_sha(expected),
                            actual_sha256=_sha(actual),
                            exact=_equal(expected, actual),
                        )
                    )
                if not all(c["exact"] for c in comparisons):
                    self.fail(trace_id, "vision_to_tuple_mismatch")
        if event == "encoder_tuple":
            state["encoder_tuple"] = metadata
        if event == "tuple_delivered":
            prior = (
                state.get("unpack_slices")
                if source == "remote"
                else state.get("encoder_tuple")
            )
            if prior is None:
                self.fail(trace_id, "missing_tuple_origin", source=source)
            else:
                for label, previous, current in zip(_LABELS, prior, metadata):
                    exact = all(
                        previous[k] == current[k]
                        for k in ("sha256", "dtype", "shape", "nbytes")
                    )
                    comparisons.append(
                        dict(
                            label=label,
                            expected_sha256=previous["sha256"],
                            actual_sha256=current["sha256"],
                            exact=exact,
                        )
                    )
                if not all(c["exact"] for c in comparisons):
                    self.fail(trace_id, "tuple_changed_before_delivery")
            state["delivered"] = metadata
            # Only positions are retained for native offset validation. The
            # ledger is never read by any inference operator.
            state["positions"] = snaps[1][0]
            state["source"] = source
        self.emit(
            event, trace_id, tensors=metadata, comparisons=comparisons, source=source
        )

    def slices(self, trace_id, plan, buffer, result, event):
        if trace_id not in self.requests:
            return
        # Read the actual validated wire header, not a guessed re-serialization
        # offset: the reader checks bytes it really received from native load.
        prefix = bytes(buffer[:16].numpy())
        magic, header_len = struct.unpack("<8sQ", prefix)
        if magic != b"RTPMM001" or header_len > 65536:
            raise ValueError("invalid diagnostic object header")
        header = json.loads(bytes(buffer[16 : 16 + header_len].numpy()))
        offset = (16 + header_len + 63) // 64 * 64
        if len(header["tensors"]) != 3 or len(plan.tensors) != 3:
            self.fail(trace_id, "three_serialized_tensors_required")
            return
        comparisons = []
        metadata = []
        for label, tensor, spec in zip(_LABELS, plan.tensors, header["tensors"]):
            cpu, info = _snapshot(tensor)
            metadata.append(info)
            begin = offset + spec["offset"]
            raw = buffer[begin : begin + spec["nbytes"]]
            raw_sha = _sha(raw)
            exact = (
                info["sha256"] == raw_sha
                and info["nbytes"] == spec["nbytes"]
                and info["shape"] == spec["shape"]
                and info["dtype"] == spec["dtype"]
                and tensor.device.type == spec["device"]
            )
            comparisons.append(
                dict(
                    label=label,
                    tensor=info,
                    slice_sha256=raw_sha,
                    offset=begin,
                    nbytes=spec["nbytes"],
                    exact=exact,
                )
            )
        if event == "pack_slices":
            previous = self.requests[trace_id].get("encoder_tuple")
            if previous is None or len(previous) != len(metadata):
                self.fail(trace_id, "missing_encoder_tuple_before_pack")
            else:
                for current, origin, comparison in zip(metadata, previous, comparisons):
                    comparison["origin_sha256"] = origin["sha256"]
                    comparison["origin_exact"] = all(
                        current[k] == origin[k]
                        for k in ("sha256", "shape", "dtype", "nbytes")
                    )
                if not all(c["origin_exact"] for c in comparisons):
                    self.fail(trace_id, "tensor_changed_before_pack")
        else:
            self.requests[trace_id]["unpack_slices"] = metadata
        if not all(c["exact"] for c in comparisons):
            self.fail(trace_id, "serialized_tensor_mismatch", stage=event)
        self.emit(
            event,
            trace_id,
            comparisons=comparisons,
            payload_sha256=_sha(buffer),
            nbytes=buffer.numel(),
            payload_offset=offset,
        )

    def model_begin(self, features, deepstack, locs, positions, tp_size):
        if not features:
            return None
        if tp_size != 1:
            self.emit("unproven", reason="tp_cp_one_required", tp_size=tp_size)
            return None
        if len(self.requests) != 1:
            self.emit(
                "unproven",
                reason="ambiguous_model_request",
                candidates=len(self.requests),
            )
            return None
        trace_id, state = next(iter(self.requests.items()))
        if "delivered" not in state or len(features) != len(deepstack):
            self.fail(trace_id, "missing_delivery_or_deepstack")
            return None
        # CP1 may split a contiguous image range. Verify its reassembly and
        # contiguous token locations before making any identity claim.
        cpu_features = [_snapshot(t)[0] for t in features]
        cpu_deep = [_snapshot(t)[0] for t in deepstack]
        locations = _snapshot(locs)[0].flatten().tolist()
        lengths = [t.shape[0] for t in cpu_features]
        if len(locations) != len(lengths) or any(
            locations[i] != locations[i - 1] + lengths[i - 1]
            for i in range(1, len(lengths))
        ):
            self.fail(trace_id, "noncontiguous_cp_image_segments")
            return None
        joined = torch.cat(cpu_features)
        joined_deep = torch.cat(cpu_deep, dim=1)
        if joined_deep.shape[0] != 3:
            self.fail(trace_id, "three_deepstack_layers_required")
            return None
        expected = state["delivered"]
        comparisons = [
            dict(
                label="embedding",
                actual_sha256=_sha(joined),
                expected_sha256=expected[0]["sha256"],
                exact=(
                    _sha(joined) == expected[0]["sha256"]
                    and list(joined.shape) == expected[0]["shape"]
                    and str(joined.dtype) == expected[0]["dtype"]
                ),
            ),
            dict(
                label="deepstack",
                actual_sha256=_sha(joined_deep.flatten()),
                expected_sha256=expected[2]["sha256"],
                exact=(
                    _sha(joined_deep.flatten()) == expected[2]["sha256"]
                    and list(joined_deep.flatten().shape) == expected[2]["shape"]
                    and str(joined_deep.dtype) == expected[2]["dtype"]
                ),
            ),
        ]
        actual_positions = _snapshot(positions)[0].reshape(-1, 3)
        image_positions = torch.cat(
            [
                actual_positions[loc : loc + length]
                for loc, length in zip(locations, lengths)
            ]
        )
        # First/single image, no prefix reuse: preceding text advances every
        # mRoPE axis once. CP1's verified contiguous image starts at this offset.
        offset_positions = state["positions"] + locations[0]
        comparisons.append(
            dict(
                label="positions",
                offset=locations[0],
                actual_sha256=_sha(image_positions),
                expected_sha256=_sha(offset_positions),
                exact=_equal(image_positions, offset_positions),
            )
        )
        if not all(c["exact"] for c in comparisons):
            self.fail(trace_id, "llm_input_mismatch")
        ticket = dict(
            trace_id=trace_id,
            features=cpu_features,
            deepstack=cpu_deep,
            locs=locations,
            lengths=lengths,
            positions=actual_positions,
            additions=[],
            rope_seen=False,
            embedding_seen=False,
        )
        self.emit(
            "llm_inputs",
            trace_id,
            comparisons=comparisons,
            segment_locations=locations,
            segment_lengths=lengths,
            tensors=[
                dict(label=label, **_snapshot(t)[1])
                for label, t in (("embedding", joined), ("deepstack", joined_deep))
            ],
        )
        return ticket

    def embedding(self, ticket, hidden):
        if ticket is None:
            return
        checks = []
        for index, (loc, feature) in enumerate(zip(ticket["locs"], ticket["features"])):
            actual = _snapshot(hidden.narrow(0, loc, feature.shape[0]))[0]
            checks.append(
                dict(
                    segment=index,
                    expected_sha256=_sha(feature),
                    actual_sha256=_sha(actual),
                    exact=_equal(feature, actual),
                )
            )
        ticket["embedding_seen"] = True
        if not all(c["exact"] for c in checks):
            self.fail(ticket["trace_id"], "embedding_copy_mismatch")
        self.emit("embedding_copy", ticket["trace_id"], comparisons=checks)

    def before_add(self, ticket, hidden, layer):
        if ticket is None or layer >= 3:
            return None
        return [
            _snapshot(hidden.narrow(0, loc, length))[0]
            for loc, length in zip(ticket["locs"], ticket["lengths"])
        ]

    def after_add(self, ticket, before, hidden, layer):
        if ticket is None or before is None or layer >= 3:
            return
        checks = []
        for i, (loc, previous, deep) in enumerate(
            zip(ticket["locs"], before, ticket["deepstack"])
        ):
            # Independent CPU two-operand oracle, rounded once to original
            # dtype. Never assign this value to the model's hidden states.
            operand = deep[layer]
            expected = (previous.float() + operand.float()).to(previous.dtype)
            actual = _snapshot(hidden.narrow(0, loc, previous.shape[0]))[0]
            checks.append(
                dict(
                    segment=i,
                    before_sha256=_sha(previous),
                    input_sha256=_sha(operand),
                    expected_sha256=_sha(expected),
                    actual_sha256=_sha(actual),
                    exact=_equal(expected, actual),
                )
            )
        ticket["additions"].append(layer)
        if not all(c["exact"] for c in checks):
            self.fail(ticket["trace_id"], "deepstack_add_mismatch", layer=layer)
        self.emit("deepstack_add", ticket["trace_id"], layer=layer, comparisons=checks)

    def rope(self, ticket, positions, config, prefix_lengths):
        if ticket is None or ticket["rope_seen"]:
            return
        ticket["rope_seen"] = True
        actual, info = _snapshot(positions)
        expected = ticket["positions"]
        actual = actual.reshape(-1, 3)
        exact = _equal(actual, expected)
        prefixes = _snapshot(prefix_lengths)[0]
        if bool(prefixes.any()):
            self.fail(ticket["trace_id"], "prefix_reuse_outside_diagnostic_scope")
        fields = (
            "style",
            "dim",
            "base",
            "scale",
            "index_factor",
            "mrope_dim1",
            "mrope_dim2",
            "mrope_dim3",
            "mrope_interleaved",
        )
        params = {
            name: (
                int(getattr(config, name))
                if name == "style"
                else getattr(config, name, None)
            )
            for name in fields
        }
        if not exact:
            self.fail(ticket["trace_id"], "rope_position_mismatch")
        self.emit(
            "rope_input",
            ticket["trace_id"],
            tensor=info,
            expected_sha256=_sha(expected),
            exact=exact,
            parameters=params,
        )

    def abort(self, ticket):
        if ticket is not None:
            self.fail(ticket["trace_id"], "model_forward_did_not_complete")

    def complete(self, ticket):
        if ticket is None:
            return
        trace_id = ticket["trace_id"]
        if (
            not ticket["embedding_seen"]
            or not ticket["rope_seen"]
            or ticket["additions"] != [0, 1, 2]
        ):
            self.fail(
                trace_id,
                "missing_llm_observation",
                embedding=ticket["embedding_seen"],
                rope=ticket["rope_seen"],
                additions=ticket["additions"],
            )
        self.emit(
            "llm_complete",
            trace_id,
            unproven=self.requests[trace_id]["unproven"],
            additions=ticket["additions"],
            rope_seen=ticket["rope_seen"],
        )
        del self.requests[trace_id]

    def eviction_origin(self, trace):
        state = self.requests.get(trace)
        if state is None:
            return None
        values = state.get("encoder_tuple") or state.get("unpack_slices")
        if values is None:
            return None
        return dict(
            key=state.get("key"),
            tensors=[dict(value) for value in values],
            origin_event=(
                "encoder_tuple" if "encoder_tuple" in state else "unpack_slices"
            ),
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
        expected_count = 3
        if (
            header["tree"] != ["tuple", [["tensor", 0], ["tensor", 1], ["tensor", 2]]]
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


def request_start(trace_id, image_count):
    _call("start", trace_id, image_count)


def bind_key(trace_id, key):
    _call("bind", trace_id, key)


@contextlib.contextmanager
def encoder_scope(items):
    if not enabled():
        yield
        return
    previous = getattr(_local, "encoder_trace", None)
    trace = getattr(items[0], "emb_diagnostic_trace", None) if len(items) == 1 else None
    _local.encoder_trace = trace
    try:
        yield
    finally:
        _local.encoder_trace = previous


def vision_output(output, grid, merge_size, pixel_values=None, visual=None):
    _call(
        "vision",
        getattr(_local, "encoder_trace", None),
        output,
        grid,
        merge_size,
        pixel_values,
        visual,
    )


def encoder_tuple(trace_id, result):
    _call("tuple_snapshot", trace_id, result, "encoder_tuple")


def delivered(trace_id, result, source):
    _call("tuple_snapshot", trace_id, result, "tuple_delivered", source)


def serialized(trace_id, plan, buffer, result, *, loading=False):
    _call(
        "slices",
        trace_id,
        plan,
        buffer,
        result,
        "unpack_slices" if loading else "pack_slices",
    )


@contextlib.contextmanager
def model_scope(features, deepstack, locs, positions, tp_size):
    if not enabled():
        yield None
        return
    ticket = _call("model_begin", features, deepstack, locs, positions, tp_size)
    previous = getattr(_local, "model_ticket", None)
    _local.model_ticket = ticket
    completed = False
    try:
        yield ticket
        completed = True
    finally:
        if not completed:
            _call("abort", ticket)
        _call("complete", ticket)
        _local.model_ticket = previous


def embedding_copied(ticket, hidden):
    _call("embedding", ticket, hidden)


def before_deepstack(ticket, hidden, layer):
    return _call("before_add", ticket, hidden, layer)


def after_deepstack(ticket, before, hidden, layer):
    _call("after_add", ticket, before, hidden, layer)


def rope_input(positions, config, prefix_lengths):
    _call(
        "rope", getattr(_local, "model_ticket", None), positions, config, prefix_lengths
    )


def eviction_origin(trace):
    return _call("eviction_origin", trace)


def eviction_serialized(trace, origin, plan, buffer):
    _call("eviction_slices", trace, origin, plan, buffer)
