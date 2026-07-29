"""Sharded, checksummed tensor artifacts for Kimi K3 accuracy traces.

Accuracy capture is disabled in normal serving.  Long-context semantic traces
can nevertheless exceed the practical size of one safetensors file, so this
module gives both the RTP recorder and the independent dummy runner one small
artifact ABI.  A logical ``foo.safetensors`` is either a regular safetensors
file or a set of ``foo-XXXXX-of-YYYYY.safetensors`` files accompanied by
``foo.safetensors.index.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from safetensors import safe_open
from safetensors.torch import save_file
import torch


TRACE_ARTIFACT_FORMAT = "kimi-k3-tensor-artifact-v1"
DEFAULT_MAX_SHARD_SIZE_BYTES = 3 << 30


def file_sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        while chunk := reader.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_index_path(logical_path: Path) -> Path:
    return logical_path.with_suffix(logical_path.suffix + ".index.json")


def artifact_exists(logical_path: Path) -> bool:
    return logical_path.is_file() or artifact_index_path(logical_path).is_file()


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _plan_shards(
    tensors: Mapping[str, torch.Tensor], max_shard_size_bytes: int
) -> list[list[str]]:
    if max_shard_size_bytes <= 0:
        raise ValueError("max_shard_size_bytes must be positive")
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name in sorted(tensors):
        tensor_size = _tensor_nbytes(tensors[name])
        if current and current_size + tensor_size > max_shard_size_bytes:
            shards.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += tensor_size
    if current:
        shards.append(current)
    return shards


def _shard_name(logical_path: Path, shard: int, count: int) -> str:
    suffix = logical_path.suffix or ".safetensors"
    stem = logical_path.name[: -len(suffix)] if suffix else logical_path.name
    return f"{stem}-{shard:05d}-of-{count:05d}{suffix}"


def save_tensor_artifact(
    tensors: Mapping[str, torch.Tensor],
    logical_path: Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    max_shard_size_bytes: int = DEFAULT_MAX_SHARD_SIZE_BYTES,
) -> Path:
    """Save tensors without silently overwriting an existing logical artifact."""

    if artifact_exists(logical_path):
        raise FileExistsError(f"tensor artifact already exists: {logical_path}")
    logical_path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = {
        name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()
    }
    tensor_bytes = sum(_tensor_nbytes(tensor) for tensor in contiguous.values())
    common_metadata = {
        "format": TRACE_ARTIFACT_FORMAT,
        **({} if metadata is None else {key: str(value) for key, value in metadata.items()}),
    }
    shards = _plan_shards(contiguous, max_shard_size_bytes)
    if len(shards) <= 1:
        save_file(contiguous, str(logical_path), metadata=common_metadata)
        logical_path.with_suffix(logical_path.suffix + ".sha256").write_text(
            file_sha256(logical_path) + "\n", encoding="utf-8"
        )
        return logical_path

    weight_map: dict[str, str] = {}
    shard_manifest: dict[str, dict[str, Any]] = {}
    for shard_index, names in enumerate(shards, start=1):
        shard_name = _shard_name(logical_path, shard_index, len(shards))
        shard_path = logical_path.parent / shard_name
        shard_tensors = {name: contiguous[name] for name in names}
        save_file(shard_tensors, str(shard_path), metadata=common_metadata)
        for name in names:
            weight_map[name] = shard_name
        shard_manifest[shard_name] = {
            "tensor_count": len(names),
            "tensor_nbytes": sum(_tensor_nbytes(contiguous[name]) for name in names),
            "file_nbytes": shard_path.stat().st_size,
            "sha256": file_sha256(shard_path),
        }

    index = {
        "format": TRACE_ARTIFACT_FORMAT,
        "logical_name": logical_path.name,
        "tensor_count": len(contiguous),
        "tensor_nbytes": tensor_bytes,
        "max_shard_size_bytes": max_shard_size_bytes,
        "metadata": dict(metadata or {}),
        "weight_map": weight_map,
        "shards": shard_manifest,
    }
    index_path = artifact_index_path(logical_path)
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    index_path.with_suffix(index_path.suffix + ".sha256").write_text(
        file_sha256(index_path) + "\n", encoding="utf-8"
    )
    return index_path


class TensorArtifactReader:
    """Read one logical tensor artifact without loading unrelated shards."""

    def __init__(self, logical_path: Path, *, verify: bool = False) -> None:
        self.logical_path = logical_path
        self.index_path = artifact_index_path(logical_path)
        self.index: dict[str, Any] | None = None
        if logical_path.is_file():
            with safe_open(str(logical_path), framework="pt", device="cpu") as handle:
                self.weight_map = {name: logical_path.name for name in handle.keys()}
        elif self.index_path.is_file():
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
            if self.index.get("format") != TRACE_ARTIFACT_FORMAT:
                raise ValueError(f"unsupported tensor artifact: {self.index_path}")
            self.weight_map = dict(self.index["weight_map"])
        else:
            raise FileNotFoundError(f"tensor artifact does not exist: {logical_path}")
        if verify:
            errors = self.validate()
            if errors:
                raise ValueError(f"invalid tensor artifact {logical_path}: {errors}")

    def keys(self) -> set[str]:
        return set(self.weight_map)

    def _path_for(self, name: str) -> Path:
        try:
            shard = self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"tensor artifact has no key {name!r}") from error
        return self.logical_path.parent / shard

    def tensor(self, name: str) -> torch.Tensor:
        path = self._path_for(name)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def tensors(self, names: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
        selected = sorted(self.keys() if names is None else set(names))
        by_shard: dict[Path, list[str]] = {}
        for name in selected:
            by_shard.setdefault(self._path_for(name), []).append(name)
        result: dict[str, torch.Tensor] = {}
        for path, shard_names in by_shard.items():
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for name in shard_names:
                    result[name] = handle.get_tensor(name)
        return result

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.index is None:
            checksum_path = self.logical_path.with_suffix(
                self.logical_path.suffix + ".sha256"
            )
            if not checksum_path.is_file():
                errors.append("missing file checksum")
            elif checksum_path.read_text(encoding="utf-8").strip() != file_sha256(
                self.logical_path
            ):
                errors.append("file checksum mismatch")
            try:
                with safe_open(
                    str(self.logical_path), framework="pt", device="cpu"
                ) as handle:
                    if set(handle.keys()) != self.keys():
                        errors.append("single-file key set changed during validation")
            except Exception as error:  # pragma: no cover - safetensors owns details
                errors.append(f"cannot open single-file artifact: {error}")
            return errors

        checksum_path = self.index_path.with_suffix(self.index_path.suffix + ".sha256")
        if not checksum_path.is_file():
            errors.append("missing index checksum")
        elif checksum_path.read_text(encoding="utf-8").strip() != file_sha256(
            self.index_path
        ):
            errors.append("index checksum mismatch")
        observed: set[str] = set()
        for shard_name, expected in self.index["shards"].items():
            shard_path = self.logical_path.parent / shard_name
            if not shard_path.is_file():
                errors.append(f"missing shard: {shard_name}")
                continue
            if shard_path.stat().st_size != int(expected["file_nbytes"]):
                errors.append(f"file size mismatch: {shard_name}")
            if file_sha256(shard_path) != expected["sha256"]:
                errors.append(f"checksum mismatch: {shard_name}")
                continue
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                shard_keys = set(handle.keys())
            expected_keys = {
                name for name, mapped_shard in self.weight_map.items()
                if mapped_shard == shard_name
            }
            if shard_keys != expected_keys:
                errors.append(f"key set mismatch: {shard_name}")
            observed.update(shard_keys)
        if observed != self.keys():
            errors.append("observed key set differs from index")
        return errors


__all__ = [
    "DEFAULT_MAX_SHARD_SIZE_BYTES",
    "TRACE_ARTIFACT_FORMAT",
    "TensorArtifactReader",
    "artifact_exists",
    "artifact_index_path",
    "file_sha256",
    "save_tensor_artifact",
]
