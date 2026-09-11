"""Persistent host backing with atomic publication and explicit reader leases.

GPU registration is deliberately a separate consumer contract. Mapping this
store is not proof of mapped-pinned or ATS GPU lookup support.
"""

import fcntl
import hashlib
import json
import mmap
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SharedWeightSlice:
    name: str
    source: Path
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str


class HostSharedWeights:
    def __init__(self, directory: Path, lease, manifest: dict):
        self.directory = directory
        self.manifest = manifest
        self._lease = lease
        self._mappings = {}
        self._writable_mappings = set()

    def view(self, name: str) -> memoryview:
        if self._lease is None:
            raise RuntimeError("shared weight lease is closed")
        if name not in self.manifest["tensors"]:
            raise KeyError(name)
        if name not in self._mappings:
            item = self.manifest["tensors"][name]
            with (self.directory / item["file"]).open("rb") as reader:
                self._mappings[name] = mmap.mmap(
                    reader.fileno(), 0, access=mmap.ACCESS_READ
                )
        return memoryview(self._mappings[name]).toreadonly()

    def _cuda_view(self, name: str, *, writable_mapping: bool) -> memoryview:
        if not writable_mapping:
            return self.view(name)
        if self._lease is None or name not in self.manifest["tensors"]:
            raise RuntimeError("CUDA registration requires a live shared weight lease")
        if name not in self._writable_mappings:
            if name in self._mappings:
                self._mappings[name].close()
                del self._mappings[name]
            path = self.directory / self.manifest["tensors"][name]["file"]
            with path.open("r+b") as reader:
                self._mappings[name] = mmap.mmap(
                    reader.fileno(), 0, access=mmap.ACCESS_WRITE
                )
            self._writable_mappings.add(name)
        # ACCESS_WRITE is MAP_SHARED, not MAP_PRIVATE or a COW table copy.
        # CUDA needs writable pages when ReadOnly registration is unsupported;
        # the CPU mapping is writable, while the exported/query API is read-only.
        return memoryview(self._mappings[name]).toreadonly()

    def register_cuda(self, device=None):
        from rtp_llm.model_loader.host_shared_cuda import SharedEngramLookup

        return SharedEngramLookup(self, device=device)

    def close(self) -> None:
        # Exported views must be released after GPU completion and unregister.
        # mmap.close() raises BufferError while a live view still holds it.
        for name in list(self._mappings):
            self._mappings[name].close()
            del self._mappings[name]
            self._writable_mappings.discard(name)
        if self._lease is not None:
            self._lease.close()
            self._lease = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class HostSharedWeightStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def identity(revision: str, slices: list[SharedWeightSlice]) -> str:
        if len(revision) != 40 or any(
            char not in "0123456789abcdef" for char in revision
        ):
            raise ValueError("shared weights require an immutable HF revision SHA")
        layout = [
            (item.name, item.nbytes, item.shape, item.dtype)
            for item in sorted(slices, key=lambda item: item.name)
        ]
        return hashlib.sha256(
            json.dumps([revision, layout], sort_keys=True).encode()
        ).hexdigest()

    def open_or_publish(
        self,
        revision: str,
        slices: list[SharedWeightSlice],
        *,
        chunk_bytes: int = 16 * 1024 * 1024,
    ) -> HostSharedWeights:
        if (
            chunk_bytes <= 0
            or not slices
            or len({item.name for item in slices}) != len(slices)
        ):
            raise ValueError(
                "shared weights require nonempty unique slices and bounded chunks"
            )
        for item in slices:
            if (
                item.offset < 0
                or item.nbytes <= 0
                or item.offset + item.nbytes > item.source.stat().st_size
            ):
                raise ValueError(f"invalid source extent for {item.name}")
        slices = sorted(slices, key=lambda item: item.name)
        key = self.identity(revision, slices)
        destination = self.root / key
        with (self.root / (key + ".lock")).open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for abandoned in self.root.glob(key + ".loading-*"):
                if abandoned.is_dir() and not abandoned.is_symlink():
                    shutil.rmtree(abandoned)
            if not destination.exists():
                staging = Path(
                    tempfile.mkdtemp(prefix=key + ".loading-", dir=self.root)
                )
                try:
                    tensors = {}
                    for index, item in enumerate(slices):
                        filename = f"{index}.bin"
                        digest = hashlib.sha256()
                        with item.source.open("rb") as source, (
                            staging / filename
                        ).open("xb") as target:
                            source.seek(item.offset)
                            remaining = item.nbytes
                            while remaining:
                                data = source.read(min(remaining, chunk_bytes))
                                if not data:
                                    raise ValueError(
                                        f"truncated shared tensor {item.name}"
                                    )
                                target.write(data)
                                digest.update(data)
                                remaining -= len(data)
                            target.flush()
                            os.fsync(target.fileno())
                        tensors[item.name] = {
                            "file": filename,
                            "nbytes": item.nbytes,
                            "shape": item.shape,
                            "dtype": item.dtype,
                            "sha256": digest.hexdigest(),
                        }
                    manifest = {
                        "revision": revision,
                        "identity": key,
                        "tensors": tensors,
                    }
                    with (staging / "READY.json").open("x", encoding="utf-8") as writer:
                        json.dump(manifest, writer, sort_keys=True)
                        writer.flush()
                        os.fsync(writer.fileno())
                    (staging / "lease").touch(exist_ok=False)
                    staging_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(staging_fd)
                    finally:
                        os.close(staging_fd)
                    os.rename(staging, destination)
                    directory_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except BaseException:
                    shutil.rmtree(staging, ignore_errors=True)
                    raise
            with (destination / "READY.json").open(encoding="utf-8") as reader:
                manifest = json.load(reader)
            if manifest.get("identity") != key or manifest.get("revision") != revision:
                raise ValueError(
                    "shared backing identity does not match the requested revision/layout"
                )
            for index, item in enumerate(slices):
                record = manifest["tensors"].get(item.name)
                if (
                    record is None
                    or record["file"] != f"{index}.bin"
                    or record["nbytes"] != item.nbytes
                    or record["shape"] != list(item.shape)
                    or record["dtype"] != item.dtype
                ):
                    raise ValueError(f"shared backing manifest mismatch: {item.name}")
                if (destination / record["file"]).stat().st_size != item.nbytes:
                    raise ValueError(f"shared backing size mismatch: {item.name}")
            lease = (destination / "lease").open("rb")
            fcntl.flock(lease, fcntl.LOCK_SH)
        return HostSharedWeights(destination, lease, manifest)

    def remove_if_unused(self, identity: str) -> bool:
        if len(identity) != 64 or any(
            char not in "0123456789abcdef" for char in identity
        ):
            raise ValueError("invalid shared backing identity")
        destination = self.root / identity
        with (self.root / (identity + ".lock")).open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not destination.exists():
                return True
            with (destination / "lease").open("rb") as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return False
                shutil.rmtree(destination)
        return True


def engram_checkpoint_slices(checkpoint, config) -> list[SharedWeightSlice]:
    """Find the four complete Engram payload extents without loading tensors."""
    checkpoint = Path(checkpoint)
    with (checkpoint / "model.safetensors.index.json").open(encoding="utf-8") as reader:
        mapping = json.load(reader)["weight_map"]
    expected = {}
    dim = config.text["engram_head_dim"]
    for layer, rows in zip(
        config.text["engram_layer_ids"], config.text["engram_num_embeddings"]
    ):
        prefix = f"layers.{layer}.engram.embed."
        expected[prefix + "weight"] = ((rows, dim), "F8_E4M3")
        expected[prefix + "scale"] = ((rows, dim // 32), "F8_E8M0")
    if len(expected) != 4:
        raise ValueError("both full Engram tables and their scales are required")
    headers = {}
    result = []
    for name, (shape, dtype) in sorted(expected.items()):
        shard = mapping[name]
        if Path(shard).name != shard:
            raise ValueError("Engram shards must be local checkpoint basenames")
        path = checkpoint / shard
        if shard not in headers:
            with path.open("rb") as reader:
                encoded_length = reader.read(8)
                if len(encoded_length) != 8:
                    raise ValueError("truncated Engram safetensors header")
                length = struct.unpack("<Q", encoded_length)[0]
                if not 2 <= length <= 64 * 1024 * 1024:
                    raise ValueError("invalid Engram safetensors header length")
                raw = reader.read(length)
                if len(raw) != length:
                    raise ValueError("truncated Engram safetensors header")
            headers[shard] = (8 + length, json.loads(raw))
        offset, header = headers[shard]
        tensor = header[name]
        if tuple(tensor["shape"]) != shape or tensor["dtype"] != dtype:
            raise ValueError(f"Engram shape/dtype mismatch for {name}")
        begin, end = tensor["data_offsets"]
        nbytes = shape[0] * shape[1]
        if (
            type(begin) is not int
            or type(end) is not int
            or begin < 0
            or end - begin != nbytes
            or offset + end > path.stat().st_size
        ):
            raise ValueError(f"invalid or incomplete Engram tensor extent: {name}")
        result.append(
            SharedWeightSlice(name, path, offset + begin, nbytes, shape, dtype)
        )
    return result
