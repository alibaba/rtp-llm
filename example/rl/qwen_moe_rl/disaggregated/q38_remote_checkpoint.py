"""A bounded reader for the pinned flat Qwen3.8 BF16 checkpoint."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import weakref

from q38_checkpoint_relay import CONFIG_SHA256, INDEX_SHA256, MAX_RANGE_BYTES, REVISION, Range
from q38_checkpoint_relay import fetch, read_local


POLICY_SHA256 = "c60d0be889e462b7665922001abfc5a57e3846f33330e33ebfdc044e6b7d3075"
OBJECT_SHA256 = "987eb6586ccecf6cfbb29b751ebdd711b1a2ec41cde6933f166317ef0b86651a"
HISTORIC_CATALOG_SHA256 = "34b083dd3e39e32b511e5459a3cfdc23ee554a9d02de99198e418913a09a6aa2"
PARAMETERS = 2_419_804_697_984
EXPERTS = 512
TRAIN_EP = 64
TRAIN_TP = 8
LAYERS = 92
MAX_HEADER_BYTES = 1024 * 1024
MAX_DENSE_SOURCE_BYTES = 5 * 1024**3
GATE_SHAPE = (4096, 8192)
DOWN_SHAPE = (8192, 2048)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def decode_json(raw):
    return json.loads(raw, object_pairs_hook=_unique_object)


def checked_json(raw, expected, label):
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    return decode_json(raw)


def _routed(name):
    return ".mlp.experts." in name


@dataclass(frozen=True)
class TensorLocation:
    name: str
    shard: str
    object_size: int
    shape: tuple
    offset: int

    def ranges(self, start, stop):
        if (type(start) is not int or type(stop) is not int or
                not 0 <= start < stop <= self.shape[0]):
            raise ValueError(f"invalid first-axis interval for {self.name}")
        row_bytes = 2 * math.prod(self.shape[1:])
        length = (stop-start) * row_bytes
        base = self.offset + start*row_bytes
        for relative in range(0, length, MAX_RANGE_BYTES):
            yield Range(self.shard, self.object_size, base+relative,
                        min(MAX_RANGE_BYTES, length-relative))


class Metadata:
    def __init__(self, config_bytes, index_bytes, object_bytes, policy_bytes):
        self.config = checked_json(config_bytes, CONFIG_SHA256, "config")
        self.index = checked_json(index_bytes, INDEX_SHA256, "index")["weight_map"]
        objects = checked_json(object_bytes, OBJECT_SHA256, "object manifest")
        policy = checked_json(policy_bytes, POLICY_SHA256, "policy manifest")
        if objects["revision"] != REVISION or policy["revision"] != REVISION:
            raise ValueError("checkpoint revision mismatch")
        self.sizes = {row["rfilename"]: row["size"] for row in objects["files"]
                      if row["rfilename"].endswith(".safetensors")}
        self.policy = policy["parameters"]
        self.by_file = {name: set() for name in self.sizes}
        for key, name in self.index.items():
            self.by_file[name].add(key)
        if (len(self.sizes) != 213 or any(not names for names in self.by_file.values())
                or len(self.policy) != 1590 or len(self.index) != 1609
                or sum(math.prod(row["shape"]) for row in self.policy.values()) != PARAMETERS
                or set(self.policy) != {name for name in self.index if not name.startswith("mtp.")}):
            raise ValueError("incomplete pinned checkpoint geometry")
        for name, entry in self.policy.items():
            if entry["checkpoint_file"] != self.index[name]:
                raise ValueError(f"pinned index and shape manifest disagree for {name}")

    @classmethod
    def from_files(cls, model_metadata, object_manifest, policy_manifest):
        root = Path(model_metadata)
        return cls((root/"config.json").read_bytes(),
                   (root/"model.safetensors.index.json").read_bytes(),
                   Path(object_manifest).read_bytes(), Path(policy_manifest).read_bytes())

    def scan_header(self, shard, read):
        size = self.sizes[shard]
        length_raw = read(Range(shard, size, 0, 8))
        if len(length_raw) != 8:
            raise ValueError(f"short header length in {shard}")
        length = struct.unpack("<Q", length_raw)[0]
        if not 2 <= length <= min(MAX_HEADER_BYTES, size - 8):
            raise ValueError(f"invalid header length for {shard}")
        raw = read(Range(shard, size, 8, length))
        if len(raw) != length:
            raise ValueError(f"short header in {shard}")
        return raw

    def parse_header(self, shard, raw):
        return self.parse_header_record(shard, decode_json(raw), len(raw))

    def parse_header_record(self, shard, header, length):
        size = self.sizes[shard]
        if type(length) is not int or not 2 <= length <= min(MAX_HEADER_BYTES, size - 8):
            raise ValueError(f"invalid catalog header length for {shard}")
        header = {key: value for key, value in header.items() if key != "__metadata__"}
        if set(header) != self.by_file[shard]:
            raise ValueError(f"checkpoint index and header keys differ for {shard}")
        data_base = 8 + length
        result = {}
        ordered = []
        for name, row in header.items():
            shape = tuple(row["shape"])
            offsets = row["data_offsets"]
            if (row["dtype"] != "BF16" or not shape
                    or any(type(axis) is not int or axis <= 0 for axis in shape)
                    or len(offsets) != 2 or any(type(x) is not int for x in offsets)):
                raise ValueError(f"invalid BF16 safetensors entry for {name}")
            first, last = offsets
            if not 0 <= first < last <= size-data_base or last-first != 2*math.prod(shape):
                raise ValueError(f"invalid data offsets for {name}")
            if name in self.policy and shape != tuple(self.policy[name]["shape"]):
                raise ValueError(f"pinned shape mismatch for {name}")
            ordered.append((first, last))
            result[name] = TensorLocation(name, shard, size, shape, data_base + first)
        cursor = 0
        for first, last in sorted(ordered):
            if cursor != first:
                raise ValueError(f"overlapping or incomplete offsets in {shard}")
            cursor = last
        if data_base + cursor != size:
            raise ValueError(f"incomplete safetensors data in {shard}")
        return result


class Checkpoint:
    def __init__(self, metadata, headers, read, *, endpoints=None):
        if set(headers) != set(metadata.sizes):
            raise ValueError("header catalog does not cover the 213 pinned shards")
        self.metadata = metadata
        self.locations = {}
        for shard, raw in headers.items():
            if isinstance(raw, bytes):
                self.locations.update(metadata.parse_header(shard, raw))
            else:
                self.locations.update(metadata.parse_header_record(shard, *raw))
        self.read = read
        self.endpoints = endpoints
        self.bytes_read = 0

    @classmethod
    def from_catalog(cls, metadata, catalog_path, expected_sha256, endpoints, *, local_ip):
        headers = load_headers(metadata, catalog_path, expected_sha256)
        if set(endpoints) != set(metadata.sizes):
            raise ValueError("relay sources do not match the pinned checkpoint")

        def read(selected):
            host, port = endpoints[selected.name]
            data, _ = fetch(host, port, selected, local_ip=local_ip, timeout=90)
            return data

        return cls(metadata, headers, read, endpoints=endpoints)

    def read_into(self, name, start, stop, destination):
        """Copy one bounded byte range at a time into existing BF16 storage."""
        import torch

        location = self.locations[name]
        if name not in self.metadata.policy:
            raise ValueError(f"tensor {name} is outside the main policy")
        if (destination.dtype != torch.bfloat16 or not destination.is_contiguous() or
                tuple(destination.shape) != (stop-start, *location.shape[1:])):
            raise ValueError(f"live BF16 destination shape mismatch for {name}")
        pointer = destination.data_ptr()
        destination_bytes = destination.detach().view(torch.uint8).reshape(-1)
        cursor = 0
        sha = hashlib.sha256()
        for selected in location.ranges(start, stop):
            raw = self.read(selected)
            if len(raw) != selected.length:
                raise ValueError(f"short tensor range for {name}")
            sha.update(raw)
            source = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            destination_bytes.narrow(0, cursor, len(raw)).copy_(source, non_blocking=False)
            cursor += len(raw)
            self.bytes_read += len(raw)
            del source, raw
        if cursor != destination_bytes.numel() or destination.data_ptr() != pointer:
            raise ValueError(f"incomplete copy to {name}")
        return sha.hexdigest()


def load_headers(metadata, catalog_path, expected_sha256):
    catalog = checked_json(Path(catalog_path).read_bytes(), expected_sha256, "header catalog")
    if isinstance(catalog["headers"], dict) and catalog.get("revision") == REVISION:
        return {shard: header.encode("utf-8") for shard, header in catalog["headers"].items()}
    if (catalog.get("index_sha256") != INDEX_SHA256 or catalog.get("tensor_payload_read") is not False
            or not isinstance(catalog["headers"], list)):
        raise ValueError("header catalog does not belong to the pinned checkpoint")
    headers = {}
    for row in catalog["headers"]:
        shard = row["file"]
        if row["file_bytes"] != metadata.sizes[shard] or shard in headers:
            raise ValueError(f"catalog object size or identity mismatch for {shard}")
        headers[shard] = (row["header"], row["header_length"])
    return headers


class DenseBridgeState(Mapping):
    """Present pinned HF names to Bridge and materialize bounded dense inputs."""

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.source = self
        self.seen = set()
        self._live = []
        self.max_live_bytes = 0

    def __len__(self):
        return len(self.checkpoint.metadata.policy)

    def __iter__(self):
        return iter(self.checkpoint.metadata.policy)

    def __contains__(self, key):
        return key in self.checkpoint.metadata.policy

    def get_all_keys(self):
        return list(self)

    def __getitem__(self, name):
        import torch

        if _routed(name):
            raise ValueError(f"routed tensor {name} requires rank-local expert reads")
        location = self.checkpoint.locations[name]
        size = 2 * math.prod(location.shape)
        self._live = [(ref, n) for ref, n in self._live if ref() is not None]
        live_bytes = sum(n for _, n in self._live) + size
        if live_bytes > MAX_DENSE_SOURCE_BYTES:
            raise MemoryError(f"dense Bridge inputs exceed source budget at {name}")
        result = torch.empty(location.shape, dtype=torch.bfloat16, device="cpu")
        self.checkpoint.read_into(name, 0, location.shape[0], result)
        self._live.append((weakref.ref(result), size))
        self.max_live_bytes = max(self.max_live_bytes, live_bytes)
        self.seen.add(name)
        return result


def scan_catalog(metadata, mounted_root):
    headers = {}
    for shard in sorted(metadata.sizes):
        raw = metadata.scan_header(shard, lambda r: read_local(mounted_root, r))
        metadata.parse_header(shard, raw)
        headers[shard] = raw.decode("utf-8")
    return {"revision": REVISION, "headers": headers}


def main():
    if len(sys.argv) != 5:
        raise SystemExit("usage: q38_remote_checkpoint MOUNT_ROOT OBJECTS POLICY HEADER_CATALOG")
    root, objects, policy, output = sys.argv[1:]
    metadata = Metadata.from_files(root, objects, policy)
    raw = json.dumps(scan_catalog(metadata, root), sort_keys=True, separators=(",", ":")).encode()
    with Path(output).open("xb") as handle:
        handle.write(raw)
    print(hashlib.sha256(raw).hexdigest(), len(raw), len(metadata.index))


if __name__ == "__main__":
    main()
