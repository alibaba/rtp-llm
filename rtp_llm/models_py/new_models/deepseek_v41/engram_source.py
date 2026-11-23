"""Per-layer checkpoint source for the V4.1 Engram runtime.

The production checkpoint places both large host tensors *and* the small
projection/gates in one shard. A collective whole-file FST load would put a
~94 GiB host table on every GPU. Native FST reads small ranges into owned CPU
tensors and streams the big pair directly into the shared cache on a miss.
"""

import ctypes
import json
import mmap
import os
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open


class EngramSource:
    def __init__(self, directory, layer_id, specs, load_method):
        self.method = getattr(load_method, "value", load_method)
        if self.method not in ("fastsafetensors", "scratch"):
            raise ValueError("Engram requires the resolved FST or scratch load method")
        self.root = Path(directory).resolve()
        self.prefix = f"layers.{layer_id}.engram."
        with (self.root / "model.safetensors.index.json").open() as stream:
            index = json.load(stream)
        if index.get("is_ft_style_weight") is True:
            self.root = self.root / "v41_engram"
            sidecar = self.root / "model.safetensors.index.json"
            if not sidecar.is_file():
                raise ValueError(
                    "This V4.1 FT archive lacks its host Engram tables; re-export "
                    "from the original indexed HF checkpoint"
                )
            with sidecar.open() as stream:
                index = json.load(stream)
        index = index["weight_map"]
        self.paths = {}
        for name in specs:
            shard = Path(index[self.prefix + name])
            if (
                shard.is_absolute()
                or ".." in shard.parts
                or shard.suffix != ".safetensors"
            ):
                raise ValueError(f"Invalid Engram checkpoint shard name: {shard}")
            self.paths[name] = self.root / shard
        self.specs = specs
        self.frames = {}
        self.stamps = {}
        # Validate the entire selected layer before any native allocation. In
        # particular, warm cache hits must still reject malformed table metadata.
        for path in dict.fromkeys(self.paths.values()):
            with path.open("rb") as stream:
                stamp = os.fstat(stream.fileno())
                self.stamps[path] = (
                    stamp.st_dev,
                    stamp.st_ino,
                    stamp.st_size,
                    stamp.st_mtime_ns,
                )
                # Rust validates offsets, contiguity, total size and dtypes on
                # the very same open file. No table payload is materialized.
                with safe_open(
                    f"/proc/self/fd/{stream.fileno()}", framework="pt", device="cpu"
                ) as meta:
                    n = int.from_bytes(stream.read(8), "little")
                    header = json.loads(stream.read(n))
                    for name, source_path in self.paths.items():
                        if source_path != path:
                            continue
                        key = self.prefix + name
                        view = meta.get_slice(key)
                        shape, tag, _ = specs[name]
                        if (
                            tuple(view.get_shape()) != tuple(shape)
                            or view.get_dtype() != tag
                        ):
                            raise ValueError(
                                f"Invalid Engram checkpoint layout for {key}"
                            )
                        start, end = header[key]["data_offsets"]
                        self.frames[name] = (8 + n + start, end - start)

    def cache_identity(self):
        # Same path+mtime identity as main's production cache; byte layout and
        # consumer are identical, so Old/New and separate workers can attach.
        parts = [str(self.root), self.prefix, "staging-v2"]
        for key in ("embed.weight", "embed.scale"):
            path = self.paths[key]
            _, _, size, ns = self.stamps[path]
            parts.extend((str(path.resolve()), str(size), str(ns)))
        return "|".join(parts)

    @contextmanager
    def _open(self, name):
        path = self.paths[name]
        with path.open("rb", buffering=0) as stream:
            fd = stream.fileno()
            st = os.fstat(fd)
            if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) != self.stamps[path]:
                raise RuntimeError(f"Engram shard changed during loading: {path}")
            yield fd

    def _read_native(self, name, fd, address):
        from fastsafetensors import cpp

        start, size = self.frames[name]
        if size:
            reader = cpp.nogds_file_reader(False, 1024, 1, False)
            dest = cpp.gds_device_buffer(address, size, False)
            chunk = 64 * 1024 * 1024
            for off in range(0, size, chunk):
                req = reader.submit_read(
                    fd, dest, start + off, min(size - off, chunk), off
                )
                if req < 0 or reader.wait_read(req) <= 0:
                    raise OSError(f"Engram native FST read failed: {self.prefix}{name}")

    def fill_shared_table(self, mapping: mmap.mmap):
        """Fill the two validated native byte ranges in the final tmpfs mapping."""
        if self.method != "fastsafetensors":
            raise ValueError("Engram direct shared staging requires native FST")
        names = ("embed.weight", "embed.scale")
        if len(mapping) != sum(self.frames[name][1] for name in names):
            raise ValueError("Incorrect shared Engram staging size")
        # Keep the caller-owned mapping alive for all synchronous wait_read calls.
        # The temporary ctypes view must not outlive this address lookup: failed
        # staging must also be able to close and unlink the unpublished mapping.
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        for name in names:
            with self._open(name) as fd:
                self._read_native(name, fd, address)
            address += self.frames[name][1]

    def load(self, name):
        with self._open(name) as fd:
            if self.method == "scratch":
                with safe_open(
                    f"/proc/self/fd/{fd}", framework="pt", device="cpu"
                ) as source:
                    return source.get_tensor(self.prefix + name)

            shape, _, dtype = self.specs[name]
            output = torch.empty(shape, dtype=dtype, device="cpu")
            if output.nbytes != self.frames[name][1]:
                raise ValueError(f"Invalid byte count for {self.prefix}{name}")
            self._read_native(name, fd, output.data_ptr())
            return output
