"""Versioned single-object representation of an entire multimodal result.

The object is opaque to KVCM. Tensor bytes retain their dtype, list/tuple/dict
boundaries and repeated tensor identities. No pickle or executable metadata.
"""

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Any

import torch

from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate

_MAGIC = b"RTPMM001"
_PREFIX = struct.Struct("<8sQ")
_MAX_HEADER = 64 * 1024
_MAX_NODES = 4096
_MAX_DEPTH = 32
_ALIGNMENT = 64
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


def plan_object(value: Any, *, devices=None, max_bytes=1024**3) -> TensorObjectPlan:
    """Describe without copying/allocating tensor data; used for admission."""
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

    tree = visit(value, devices)
    header = dict(tree=tree, tensors=specs, sha256="0" * 64)
    encoded = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > _MAX_HEADER:
        raise ValueError("multimodal object metadata exceeds limit")
    offset = _align(_PREFIX.size + len(encoded))
    nbytes = offset + payload_size
    if nbytes > max_bytes:
        raise ValueError("multimodal object exceeds byte limit")
    return TensorObjectPlan(header, tensors, encoded, offset, nbytes)


def pack_object(plan: TensorObjectPlan) -> torch.Tensor:
    """CPU staging compatibility path for the existing one-buffer SDK."""
    output = torch.zeros(plan.nbytes, dtype=torch.uint8)
    for tensor, spec in zip(plan.tensors, plan.header["tensors"]):
        if not spec["nbytes"]:
            continue
        start = plan.payload_offset + spec["offset"]
        with cuda_graph_gate.operation():
            source = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
            output[start : start + spec["nbytes"]].copy_(source, non_blocking=False)
    # A memoryview avoids another full bytes copy during checksumming.
    plan.header["sha256"] = "0" * 64
    metadata = json.dumps(plan.header, separators=(",", ":"), allow_nan=False).encode()
    digest = hashlib.sha256(metadata)
    digest.update(memoryview(output.numpy())[plan.payload_offset :])
    plan.header["sha256"] = digest.hexdigest()
    header = json.dumps(plan.header, separators=(",", ":"), allow_nan=False).encode()
    prefix = _PREFIX.pack(_MAGIC, len(header)) + header
    output[: len(prefix)].copy_(torch.frombuffer(bytearray(prefix), dtype=torch.uint8))
    return output


def unpack_object(
    buffer: torch.Tensor, *, cuda_device=None, restore_devices=True, max_bytes=1024**3
):
    """Validate the entire object before materializing any destination tensor."""
    if (
        buffer.dtype != torch.uint8
        or buffer.device.type != "cpu"
        or buffer.ndim != 1
        or not buffer.is_contiguous()
        or not _PREFIX.size <= buffer.numel() <= max_bytes
    ):
        raise ValueError("invalid multimodal object buffer")
    view = memoryview(buffer.numpy())
    magic, length = _PREFIX.unpack_from(view)
    offset = _align(_PREFIX.size + length)
    if magic != _MAGIC or length > _MAX_HEADER or offset > len(view):
        raise ValueError("invalid multimodal object header")
    try:
        header = json.loads(bytes(view[_PREFIX.size : _PREFIX.size + length]))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid multimodal object metadata") from error
    if not isinstance(header, dict) or set(header) != {"tree", "tensors", "sha256"}:
        raise ValueError("invalid multimodal object schema")
    expected_digest = header["sha256"]
    header["sha256"] = "0" * 64
    metadata = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    digest = hashlib.sha256(metadata)
    digest.update(view[offset:])
    if digest.hexdigest() != expected_digest:
        raise ValueError("multimodal object checksum mismatch")
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
            or spec["dtype"] not in _DTYPES
            or spec["device"] not in ("cpu", "cuda")
            or type(spec["offset"]) is not int
            or type(spec["nbytes"]) is not int
        ):
            raise ValueError("invalid multimodal tensor metadata")
        width = torch.empty((), dtype=_DTYPES[spec["dtype"]]).element_size()
        if (
            spec["nbytes"] != math.prod(shape) * width
            or spec["offset"] != _align(end)
            or spec["offset"] + spec["nbytes"] > len(view) - offset
        ):
            raise ValueError("invalid multimodal tensor range")
        end = spec["offset"] + spec["nbytes"]
    if end != len(view) - offset:
        raise ValueError("invalid multimodal object length")
    nodes, referenced = 0, set()

    def rebuild(node, tensors=None, depth=0):
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
            if type(content) is not int or not 0 <= content < len(specs):
                raise ValueError("invalid multimodal tensor reference")
            referenced.add(content)
            return tensors[content] if tensors is not None else None
        if kind in ("list", "tuple") and isinstance(content, list):
            result = [rebuild(v, tensors, depth + 1) for v in content]
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
                result[pair[0]] = rebuild(pair[1], tensors, depth + 1)
            return result
        if kind == "scalar" and (
            content is None
            or type(content) in (bool, int, str)
            or type(content) is float
            and math.isfinite(content)
        ):
            return content
        raise ValueError("invalid multimodal node")

    rebuild(header["tree"])
    if referenced != set(range(len(specs))):
        raise ValueError("unreferenced multimodal tensor")
    tensors = []
    for spec in specs:
        dtype = _DTYPES[spec["dtype"]]
        start = offset + spec["offset"]
        tensor = (
            buffer[start : start + spec["nbytes"]].view(dtype).reshape(spec["shape"])
            if spec["nbytes"]
            else torch.empty(spec["shape"], dtype=dtype)
        )
        if restore_devices and spec["device"] == "cuda":
            with cuda_graph_gate.operation():
                tensor = tensor.to(
                    cuda_device or torch.device("cuda", torch.cuda.current_device())
                )
        tensors.append(tensor)
    nodes = 0
    return rebuild(header["tree"], tensors)
