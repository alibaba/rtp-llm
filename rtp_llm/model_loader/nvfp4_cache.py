"""Content-addressed, portable NVFP4 expert artifacts before backend swizzling."""

import hashlib
import inspect
import json
import math
import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
    _E2M1_BOUNDS,
    _E2M1_ODD_BOUNDS,
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    _cast_to_e2m1,
    quantize_expert_to_nvfp4,
)

FORMAT = "rtp-nvfp4-expert-v1"
ALGORITHM_SHA = hashlib.sha256(
    (
        repr(
            (
                FLOAT4_E2M1_MAX,
                FLOAT8_E4M3_MAX,
                _E2M1_BOUNDS.tolist(),
                _E2M1_ODD_BOUNDS.tolist(),
            )
        )
        + inspect.getsource(_cast_to_e2m1)
        + inspect.getsource(quantize_expert_to_nvfp4)
    ).encode()
).hexdigest()


def tensor_sha(tensor):
    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


class Nvfp4ExpertCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def get_or_quantize(self, weight, *, block_size, input_scale, identity):
        if block_size != 16 or not math.isfinite(input_scale) or input_scale <= 0:
            raise ValueError(
                "NVFP4 cache requires block16 and positive finite input scale"
            )
        contract = dict(
            format=FORMAT,
            algorithm_sha256=ALGORITHM_SHA,
            source_sha256=tensor_sha(weight),
            source_dtype=str(weight.dtype),
            shape=list(weight.shape),
            block_size=block_size,
            input_scale=float(input_scale),
            identity=identity,
            packed_layout="row-major-low-nibble-first",
            scale_layout="linear-float8-e4m3-block16",
        )
        raw = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(raw.encode()).hexdigest()
        path = self.directory / (key + ".safetensors")
        if path.exists():
            with safe_open(str(path), framework="pt", device="cpu") as f:
                metadata = f.metadata()
                if metadata.get("contract") != raw:
                    raise ValueError("NVFP4 cache contract mismatch")
                tensors = {name: f.get_tensor(name) for name in f.keys()}
            self._validate(tensors, weight.shape, block_size)
            checksums = json.loads(metadata["tensor_sha256"])
            if checksums != {
                name: tensor_sha(value) for name, value in tensors.items()
            }:
                raise ValueError("NVFP4 cache payload checksum mismatch")
            return (
                tensors["packed"].to(weight.device),
                tensors["block_scale"].to(weight.device),
                tensors["global_scale"].to(weight.device).reshape(()),
            )
        packed, scale, global_scale = quantize_expert_to_nvfp4(weight, block_size)
        tensors = {
            "packed": packed,
            "block_scale": scale,
            "global_scale": global_scale.reshape(1),
        }
        self._validate(tensors, weight.shape, block_size)
        tensors = {
            name: value.detach().contiguous().cpu() for name, value in tensors.items()
        }
        metadata = dict(
            contract=raw,
            tensor_sha256=json.dumps(
                {name: tensor_sha(value) for name, value in tensors.items()},
                sort_keys=True,
            ),
        )
        fd, temporary = tempfile.mkstemp(
            prefix=".nvfp4-", suffix=".safetensors", dir=self.directory
        )
        os.close(fd)
        try:
            save_file(tensors, temporary, metadata=metadata)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return packed, scale, global_scale

    @staticmethod
    def _validate(tensors, shape, block_size):
        n, k = shape
        expected = dict(
            packed=((n, k // 2), torch.uint8),
            block_scale=((n, k // block_size), torch.float8_e4m3fn),
            global_scale=((1,), torch.float32),
        )
        if set(tensors) != set(expected):
            raise ValueError("NVFP4 cache tensor inventory mismatch")
        for name, (dimensions, dtype) in expected.items():
            value = tensors[name]
            if tuple(value.shape) != dimensions or value.dtype != dtype:
                raise ValueError(f"NVFP4 cache dtype/shape mismatch: {name}")
        if (
            not bool(torch.isfinite(tensors["block_scale"].float()).all())
            or not bool(torch.isfinite(tensors["global_scale"]).all())
            or not bool((tensors["global_scale"] > 0).all())
        ):
            raise ValueError(
                "NVFP4 cache scales must be finite and global scale positive"
            )

    def manifest(self):
        rows = []
        for path in sorted(self.directory.glob("*.safetensors")):
            with safe_open(str(path), framework="pt", device="cpu") as f:
                contract = json.loads(f.metadata()["contract"])
            rows.append(
                dict(
                    file=path.name,
                    bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    contract=contract,
                )
            )
        return dict(
            format=FORMAT,
            algorithm_sha256=ALGORITHM_SHA,
            entries=rows,
            total_bytes=sum(row["bytes"] for row in rows),
        )
