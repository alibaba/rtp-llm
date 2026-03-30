"""Merkle tree: convert a set of files into REAPI Directory/FileNode structure.

Memory-efficient: only computes hashes by streaming, does not load file contents into memory.
File data is read on-demand during upload.
"""
import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from . import remote_execution_pb2 as re_pb2

HASH_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for streaming hash


def sha256_digest(data: bytes) -> re_pb2.Digest:
    return re_pb2.Digest(hash=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def sha256_file(path: Path) -> Tuple[str, int]:
    """Compute SHA256 of a file by streaming. Returns (hex_hash, size)."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


@dataclass
class MerkleResult:
    root_digest: re_pb2.Digest
    # hash -> absolute file path (for on-demand reading during upload)
    file_map: Dict[str, Path]
    # hash -> serialized Directory proto bytes (small, kept in memory)
    dir_blobs: Dict[str, bytes]
    # hash -> size (for all blobs: files + dirs)
    sizes: Dict[str, int]


def build_merkle_tree(root: Path, files: List[str]) -> MerkleResult:
    """Build a Merkle tree from a list of relative file paths.

    Memory-efficient: only hashes files by streaming, stores paths not data.
    """
    file_map: Dict[str, Path] = {}  # hash -> abs path
    sizes: Dict[str, int] = {}
    dir_blobs: Dict[str, bytes] = {}

    dir_children: Dict[str, list] = defaultdict(list)
    seen_dirs: Dict[str, set] = defaultdict(set)

    for rel_path in sorted(set(files)):
        abs_path = root / rel_path
        if not abs_path.exists():
            continue

        parts = Path(rel_path).parts

        for i in range(len(parts) - 1):
            parent = "/".join(parts[:i]) if i > 0 else ""
            child_dir = parts[i]
            if child_dir not in seen_dirs[parent]:
                seen_dirs[parent].add(child_dir)
                dir_children[parent].append((child_dir, False, None, False))

        parent = "/".join(parts[:-1])
        filename = parts[-1]

        # Stream-hash the file (no full read into memory)
        file_hash, file_size = sha256_file(abs_path)
        digest = re_pb2.Digest(hash=file_hash, size_bytes=file_size)
        file_map[file_hash] = abs_path
        sizes[file_hash] = file_size
        is_exec = os.access(abs_path, os.X_OK)
        dir_children[parent].append((filename, True, digest, is_exec))

    # Build Directory protos bottom-up (these are small)
    dir_digests: Dict[str, re_pb2.Digest] = {}

    def _build(dir_path: str) -> re_pb2.Digest:
        if dir_path in dir_digests:
            return dir_digests[dir_path]

        d = re_pb2.Directory()
        for name, is_file, digest, is_exec in sorted(dir_children.get(dir_path, []), key=lambda e: e[0]):
            if is_file:
                d.files.append(re_pb2.FileNode(name=name, digest=digest, is_executable=is_exec))
            else:
                child = f"{dir_path}/{name}" if dir_path else name
                child_digest = _build(child)
                d.directories.append(re_pb2.DirectoryNode(name=name, digest=child_digest))

        data = d.SerializeToString()
        dg = sha256_digest(data)
        dir_blobs[dg.hash] = data
        sizes[dg.hash] = len(data)
        dir_digests[dir_path] = dg
        return dg

    root_digest = _build("")
    return MerkleResult(root_digest=root_digest, file_map=file_map, dir_blobs=dir_blobs, sizes=sizes)
