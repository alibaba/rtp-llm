"""Versioned single-object representation of an entire multimodal result.

The object is opaque to KVCM. Tensor bytes retain their dtype, list/tuple/dict
boundaries and repeated tensor identities. No pickle or executable metadata.
"""

import hashlib
import json
import math
import os
import struct
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate

_MAGIC_V1 = b"RTPMM001"
_MAGIC_V2 = b"RTPMM002"
_PREFIX = struct.Struct("<8sQ")
_MAX_HEADER = 64 * 1024
_CUDA_HEADER_READ_BYTES = 1024
_MAX_NODES = 4096
_MAX_DEPTH = 32
_ALIGNMENT = 64
_USE_NATIVE_RESTORE = os.environ.get("MM_EMB_NATIVE_RESTORE") == "1"
_USE_NATIVE_PACK = os.environ.get("MM_EMB_NATIVE_PACK") == "1"
_USE_NATIVE_PLAN = os.environ.get("MM_EMB_NATIVE_PLAN") == "1"
_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
}


def _align(size):
    return (size + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


@dataclass
class TensorObjectPlan:
    header: dict
    tensors: list
    header_bytes: bytes
    payload_offset: int
    nbytes: int
    checksum: bool


def _describe_object(value, devices, max_bytes):
    tensors, specs, identities = [], [], {}
    payload_size = nodes = 0

    def visit(item, target, depth=0):
        nonlocal payload_size, nodes
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("multimodal object structure exceeds limits")
        if isinstance(item, torch.Tensor):
            if item.layout != torch.strided or str(item.dtype) not in _DTYPES:
                raise ValueError("unsupported multimodal tensor layout or dtype")
            device = torch.device(target) if target is not None else item.device
            if item.device.type not in ("cpu", "cuda") or device.type not in (
                "cpu",
                "cuda",
            ):
                raise ValueError("unsupported multimodal tensor device")
            identity = (id(item), device.type)
            if identity not in identities:
                index = identities[identity] = len(tensors)
                nbytes = item.numel() * item.element_size()
                offset = _align(payload_size)
                payload_size = offset + nbytes
                if payload_size > max_bytes:
                    raise ValueError("multimodal object exceeds byte limit")
                tensors.append(item)
                specs.append(
                    dict(
                        shape=list(item.shape),
                        dtype=str(item.dtype),
                        device=device.type,
                        offset=offset,
                        nbytes=nbytes,
                    )
                )
            return ["tensor", identities[identity]]
        if isinstance(item, (tuple, list)):
            if target is not None and (
                not isinstance(target, (tuple, list)) or len(target) != len(item)
            ):
                raise ValueError("multimodal device tree mismatch")
            return [
                "tuple" if isinstance(item, tuple) else "list",
                [
                    visit(v, target[i] if target is not None else None, depth + 1)
                    for i, v in enumerate(item)
                ],
            ]
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise ValueError("multimodal object dictionary keys must be strings")
            if target is not None and (
                not isinstance(target, dict) or target.keys() != item.keys()
            ):
                raise ValueError("multimodal device tree mismatch")
            return [
                "dict",
                [
                    [k, visit(v, target[k] if target is not None else None, depth + 1)]
                    for k, v in item.items()
                ],
            ]
        if item is None or type(item) in (bool, int, str):
            return ["scalar", item]
        if type(item) is float and math.isfinite(item):
            return ["scalar", item]
        raise ValueError(f"unsupported multimodal value: {type(item).__name__}")

    try:
        tree = visit(value, devices)
    finally:
        # The recursive closure captures both itself and the tensor list. Break
        # that cycle so releasing the plan releases tensors without waiting for GC.
        del visit
    return dict(tree=tree, tensors=specs), tensors, payload_size


def plan_object(
    value: Any, *, devices=None, max_bytes=1024**3, checksum=True
) -> TensorObjectPlan:
    """Describe without copying/allocating tensor data; used for admission."""
    description = None
    if (
        _USE_NATIVE_PLAN
        and devices is None
        and type(max_bytes) is int
        and 0 <= max_bytes <= (1 << 63) - 1 - _ALIGNMENT
    ):
        from librtp_compute_ops import plan_emb_tensor_object

        description = plan_emb_tensor_object(value, max_bytes)
    if description is None:
        description = _describe_object(value, devices, max_bytes)
    header, tensors, payload_size = description
    if checksum:
        header["sha256"] = "0" * 64
    encoded = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > _MAX_HEADER:
        raise ValueError("multimodal object metadata exceeds limit")
    offset = _align(_PREFIX.size + len(encoded))
    nbytes = offset + payload_size
    if nbytes > max_bytes:
        raise ValueError("multimodal object exceeds byte limit")
    return TensorObjectPlan(header, tensors, encoded, offset, nbytes, checksum)


def pack_object(
    plan: TensorObjectPlan,
    *,
    out: torch.Tensor | None = None,
    observe_stage=None,
    payload_fence=None,
    blocking_cuda_events=False,
) -> torch.Tensor:
    """Pack bytes directly into a supplied host or registered CUDA buffer."""

    def stage(name):
        return (
            observe_stage("write", name) if observe_stage is not None else nullcontext()
        )

    if len(plan.tensors) != len(plan.header["tensors"]):
        raise ValueError("packing plan no longer owns its tensors")
    with stage("pack_allocate"):
        if out is None:
            output = torch.empty(plan.nbytes, dtype=torch.uint8)
        else:
            if (
                out.device.type not in ("cpu", "cuda")
                or (plan.checksum and out.is_cuda)
                or out.dtype != torch.uint8
                or out.ndim != 1
                or not out.is_contiguous()
                or out.numel() < plan.nbytes
            ):
                raise ValueError("invalid packing buffer")
            output = out[:plan.nbytes]

    def write_header(magic, header):
        prefix = _PREFIX.pack(magic, len(header)) + header
        with stage("h2d_header") if output.is_cuda else nullcontext():
            output[: len(prefix)].copy_(
                torch.frombuffer(bytearray(prefix), dtype=torch.uint8)
            )
            output[len(prefix) : plan.payload_offset].zero_()

    with cuda_graph_gate.operation() if output.is_cuda else nullcontext():
        if output.is_cuda and _USE_NATIVE_PACK:
            import librtp_compute_ops

            fenced_pack = None
            if payload_fence is not None and blocking_cuda_events:
                fenced_pack = getattr(
                    librtp_compute_ops, "pack_emb_cuda_tensor_slices_and_wait", None
                )
            pack_slices = fenced_pack or librtp_compute_ops.pack_emb_tensor_slices
            with stage("native_pack_fenced" if fenced_pack else "native_pack"):
                pack_slices(
                    output,
                    _PREFIX.pack(_MAGIC_V2, len(plan.header_bytes)) + plan.header_bytes,
                    plan.payload_offset,
                    [
                        (plan.payload_offset + spec["offset"], spec["nbytes"], tensor)
                        for tensor, spec in zip(plan.tensors, plan.header["tensors"])
                    ],
                )
            if payload_fence is not None and fenced_pack is None:
                with stage("pack_ready_wait"):
                    payload_fence()
            return output
        if not plan.checksum:
            write_header(_MAGIC_V2, plan.header_bytes)
        with stage("pack_payload"):
            end = plan.payload_offset
            for tensor, spec in zip(plan.tensors, plan.header["tensors"]):
                start = plan.payload_offset + spec["offset"]
                if start > end:
                    output[end:start].zero_()
                end = start + spec["nbytes"]
                if spec["nbytes"]:
                    with cuda_graph_gate.operation():
                        target = (
                            output[start:end].view(tensor.dtype).reshape(tensor.shape)
                        )
                        if tensor.is_cuda and output.is_cuda:
                            tensor.record_stream(
                                torch.cuda.current_stream(tensor.device)
                            )
                        target.copy_(tensor.detach(), non_blocking=output.is_cuda)
        if plan.checksum:
            # A memoryview avoids a second whole-payload copy for legacy SHA-256.
            with stage("pack_checksum"):
                digest = hashlib.sha256(plan.header_bytes)
                digest.update(memoryview(output.numpy())[plan.payload_offset :])
                header = dict(plan.header, sha256=digest.hexdigest())
                encoded = json.dumps(
                    header, separators=(",", ":"), allow_nan=False
                ).encode()
            write_header(_MAGIC_V1, encoded)
        if payload_fence is not None:
            with stage("pack_ready_wait"):
                payload_fence()
    return output


def _rebuild_tree(tree, tensor_count, referenced, tensors=None):
    nodes = 0

    def rebuild(node, depth=0):
        nonlocal nodes
        nodes += 1
        if (
            nodes > _MAX_NODES
            or depth > _MAX_DEPTH
            or not isinstance(node, list)
            or len(node) != 2
        ):
            raise ValueError("invalid multimodal structure")
        kind, content = node
        if kind == "tensor":
            if type(content) is not int or not 0 <= content < tensor_count:
                raise ValueError("invalid multimodal tensor reference")
            referenced.add(content)
            return tensors[content] if tensors is not None else None
        if kind in ("list", "tuple") and isinstance(content, list):
            result = [rebuild(v, depth + 1) for v in content]
            return tuple(result) if kind == "tuple" else result
        if kind == "dict" and isinstance(content, list):
            result = {}
            for pair in content:
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or not isinstance(pair[0], str)
                    or pair[0] in result
                ):
                    raise ValueError("invalid multimodal dictionary")
                result[pair[0]] = rebuild(pair[1], depth + 1)
            return result
        if kind == "scalar" and (
            content is None
            or type(content) in (bool, int, str)
            or type(content) is float
            and math.isfinite(content)
        ):
            return content
        raise ValueError("invalid multimodal node")

    try:
        return rebuild(tree)
    finally:
        # A recursive closure otherwise keeps the metadata and tensors alive
        # until cyclic GC. Release both on parse errors and successful reads.
        del rebuild


def unpack_object(
    buffer: torch.Tensor,
    *,
    cuda_device=None,
    restore_devices=True,
    max_bytes=1024**3,
    observe_stage=None,
    consumer_stream=None,
    expected_key=None,
    blocking_cuda_events=False,
    payload_fence=None,
):
    """Validate the layout before materializing any destination tensor.

    CUDA callers own the source and must wait for all submitted copies before
    reusing it. A caller can request completion here by providing its stream
    fence. CUDA results are allocated independently on the current stream.
    """

    def stage(name):
        return (
            observe_stage("read", name) if observe_stage is not None else nullcontext()
        )

    size = buffer.numel()
    if (
        buffer.dtype != torch.uint8
        or buffer.device.type not in ("cpu", "cuda")
        or buffer.ndim != 1
        or not buffer.is_contiguous()
        or not _PREFIX.size <= size <= max_bytes
    ):
        raise ValueError("invalid multimodal object buffer")
    read_native_header = None
    if buffer.is_cuda and _USE_NATIVE_RESTORE and blocking_cuda_events:
        import librtp_compute_ops

        read_native_header = getattr(librtp_compute_ops, "read_emb_cuda_header", None)

    def copy_header(start, end):
        with stage("d2h_header"):
            if read_native_header is not None:
                return read_native_header(buffer, start, end - start)
            return buffer[start:end].cpu()

    if buffer.is_cuda:
        # Fetch the usual prefix and metadata together. Large headers still
        # copy only the missing suffix after validating the declared length.
        head = copy_header(0, min(size, _CUDA_HEADER_READ_BYTES))
    else:
        head = buffer
    view = memoryview(head.numpy())
    magic, length = _PREFIX.unpack_from(view)
    header_end = _PREFIX.size + length
    offset = _align(header_end)
    if (
        (magic != _MAGIC_V2 and (magic != _MAGIC_V1 or buffer.is_cuda))
        or length > _MAX_HEADER
        or offset > size
    ):
        raise ValueError("invalid multimodal object header")
    raw_header = bytes(view[_PREFIX.size : header_end])
    if header_end > len(view):
        suffix = copy_header(len(view), header_end)
        raw_header += suffix.numpy().tobytes()
    with stage("decode_header"):
        try:
            header = json.loads(raw_header)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise ValueError("invalid multimodal object metadata") from error
        fields = {"tree", "tensors"}
        if magic == _MAGIC_V1:
            fields.add("sha256")
        if not isinstance(header, dict) or set(header) != fields:
            raise ValueError("invalid multimodal object schema")
    if magic == _MAGIC_V1:
        with stage("checksum"):
            expected_digest = header["sha256"]
            header["sha256"] = "0" * 64
            metadata = json.dumps(
                header, separators=(",", ":"), allow_nan=False
            ).encode()
            digest = hashlib.sha256(metadata)
            digest.update(view[offset:])
            if digest.hexdigest() != expected_digest:
                raise ValueError("multimodal object checksum mismatch")
    with stage("decode_layout"):
        specs = header["tensors"]
        if not isinstance(specs, list) or len(specs) > _MAX_NODES:
            raise ValueError("invalid multimodal tensor table")
        end = 0
        for spec in specs:
            if not isinstance(spec, dict) or set(spec) != {
                "shape",
                "dtype",
                "device",
                "offset",
                "nbytes",
            }:
                raise ValueError("invalid multimodal tensor descriptor")
            shape = spec["shape"]
            if (
                not isinstance(shape, list)
                or len(shape) > 32
                or any(type(s) is not int or s < 0 or s > 2**31 for s in shape)
                or not isinstance(spec["dtype"], str)
                or spec["dtype"] not in _DTYPES
                or spec["device"] not in ("cpu", "cuda")
                or type(spec["offset"]) is not int
                or type(spec["nbytes"]) is not int
            ):
                raise ValueError("invalid multimodal tensor metadata")
            width = _DTYPES[spec["dtype"]].itemsize
            if (
                spec["nbytes"] != math.prod(shape) * width
                or spec["offset"] != _align(end)
                or spec["offset"] + spec["nbytes"] > size - offset
            ):
                raise ValueError("invalid multimodal tensor range")
            end = spec["offset"] + spec["nbytes"]
        if end != size - offset:
            raise ValueError("invalid multimodal object length")
        referenced = set()
        layout = _rebuild_tree(header["tree"], len(specs), referenced)
        if magic == _MAGIC_V2 and expected_key is not None:
            if not isinstance(layout, dict) or layout.get("cache_key") != expected_key:
                raise ValueError(
                    "remote embedding cache key mismatch or missing binding"
                )
        if referenced != set(range(len(specs))):
            raise ValueError("unreferenced multimodal tensor")
    def tensor_view(spec):
        start = offset + spec["offset"]
        return (
            buffer[start : start + spec["nbytes"]]
            .view(_DTYPES[spec["dtype"]])
            .reshape(spec["shape"])
        )

    tensors = []
    if buffer.is_cuda and _USE_NATIVE_RESTORE:
        import librtp_compute_ops

        gpu = torch.device(cuda_device or buffer.device)
        cpu = torch.device("cpu")
        restore_slices = librtp_compute_ops.restore_emb_tensor_slices
        fenced_restore = None
        if (
            payload_fence is not None
            and blocking_cuda_events
            and gpu.type == "cuda"
            and (gpu.index is None or gpu == buffer.device)
        ):
            fenced_restore = getattr(
                librtp_compute_ops, "restore_emb_cuda_tensor_slices_and_wait", None
            )
        restore_stage = (
            "native_restore_fenced" if fenced_restore else "native_restore_payload"
        )
        with stage(restore_stage):
            slices = [
                (
                    offset + spec["offset"],
                    spec["nbytes"],
                    _DTYPES[spec["dtype"]],
                    spec["shape"],
                    gpu if restore_devices and spec["device"] == "cuda" else cpu,
                )
                for spec in specs
            ]
            tensors = (fenced_restore or restore_slices)(buffer, slices)
            if consumer_stream is not None:
                for tensor in tensors:
                    if tensor.is_cuda:
                        tensor.record_stream(consumer_stream)
        if fenced_restore is not None:
            payload_fence = None
    elif buffer.is_cuda:
        for spec in specs:
            target_cuda = restore_devices and spec["device"] == "cuda"
            target = (cuda_device or buffer.device) if target_cuda else "cpu"
            with stage("d2d_restore" if target_cuda else "d2h_restore"):
                tensor = (
                    tensor_view(spec).to(target, copy=True, non_blocking=True)
                    if spec["nbytes"]
                    else torch.empty(
                        spec["shape"], dtype=_DTYPES[spec["dtype"]], device=target
                    )
                )
                if tensor.is_cuda and consumer_stream is not None:
                    tensor.record_stream(consumer_stream)
                tensors.append(tensor)
    else:
        with stage("cpu_rebuild"):
            for spec in specs:
                tensors.append(
                    tensor_view(spec)
                    if spec["nbytes"]
                    else torch.empty(spec["shape"], dtype=_DTYPES[spec["dtype"]])
                )
        if restore_devices and any(spec["device"] == "cuda" for spec in specs):
            with stage("h2d_restore"), cuda_graph_gate.operation():
                for index, spec in enumerate(specs):
                    if spec["device"] == "cuda":
                        tensors[index] = tensors[index].to(cuda_device or "cuda")
    with stage("tree_restore"):
        value = _rebuild_tree(header["tree"], len(specs), referenced, tensors)
    if payload_fence is not None:
        with stage("restore_ready_wait"):
            payload_fence()
    return value
