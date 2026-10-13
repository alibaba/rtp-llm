"""Release stopped TSDB copies only after verifying their lossless attachment."""

import shutil
from pathlib import Path


def prune_tsdb(directory, artifacts, deadline):
    from artifacts.experiment import ExperimentReader
    from file_digest import sha256_file

    directory = Path(directory).resolve()
    roots = sorted((directory / "telemetry").glob("*/data"))
    if not roots:
        return
    candidates = []
    for root in roots:
        if root.resolve() != root or not root.is_dir():
            raise ValueError("TSDB retention requires owned directories")
        for path in root.rglob("*"):
            deadline.check()
            if path.is_symlink():
                raise ValueError("TSDB retention refuses symbolic links")
            if path.is_file():
                candidates.append(path)
    with ExperimentReader(artifacts["experiment"]) as primary, ExperimentReader(artifacts["raw"]) as raw:
        reference = primary.read_json("raw-attachment.json")
        if (raw.kind != "raw" or Path(artifacts["raw"]).stat().st_size != reference["size"]
                or sha256_file(artifacts["raw"]) != reference["sha256"]):
            raise ValueError("raw attachment checksum mismatch")
        for path in candidates:
            deadline.check()
            name = "run/" + path.relative_to(directory).as_posix()
            owner = primary if name in primary.entries else raw
            if name not in owner.entries:
                raise ValueError("TSDB file was not archived: " + name)
            entry = owner.entries[name]
            if path.stat().st_size != entry.size or sha256_file(path) != entry.sha256:
                raise ValueError("TSDB source differs from archive: " + name)
            with owner.open(name):
                pass  # Verify compressed payload, not only manifest metadata.
    deadline.check()
    # Verify the entire candidate set before removing any directory.
    for root in roots:
        shutil.rmtree(root)
