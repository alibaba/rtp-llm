from pathlib import Path


def resolve_safetensors_shard(root: Path, filename: str) -> Path:
    """Accept regular local shards and a snapshot's own content-addressed blobs."""
    name = Path(filename)
    if name.is_absolute() or ".." in name.parts or name.suffix != ".safetensors":
        raise ValueError(f"Invalid checkpoint shard name: {filename!r}")
    path = (root / name).resolve()
    if root not in path.parents:
        # A snapshot can point into its own repository's blobs directory, but
        # not into another repository or through a redirected blobs directory.
        if root.parent.name != "snapshots" or path.parent != root.parent.parent / "blobs":
            raise ValueError(f"Shard escapes checkpoint directory: {filename!r}")
    if not path.is_file():
        raise ValueError(f"Missing checkpoint shard: {filename!r}")
    return path
