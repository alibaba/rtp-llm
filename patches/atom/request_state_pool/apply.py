"""Apply the additive request-state patches to exactly reviewed dependencies.

Run after apply_qwen38_dependency_patches.py. No dependency package is imported.
--root is a source-fixture override for CPU tests; normal installation resolves
each package via its installed distribution metadata and checks pinned versions.
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes() if path.exists() else b"").hexdigest()


def apply(root=None, verify_only=False):
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory / "manifest.json").read_text())
    prepared = []
    for name, spec in manifest["components"].items():
        package = "atom" if name == "atom" else "amd-aiter"
        if root is None:
            if importlib.metadata.version(package) != manifest[name + "_version"]:
                raise RuntimeError(
                    f"Unexpected {package} version for request-state patch"
                )
            destination = Path(importlib.metadata.distribution(package).locate_file(""))
        else:
            destination = Path(root).resolve()
        patch = directory / spec["patch"]
        if digest(patch) != spec["patch_sha256"]:
            raise RuntimeError(f"{name} patch differs from its reviewed manifest")
        names = [
            line.split()[2][2:]
            for line in patch.read_text().splitlines()
            if line.startswith("diff --git ")
        ]
        if (
            len(names) != len(set(names))
            or set(names) != set(spec["source_files_sha256"])
            or set(names) != set(spec["patched_files_sha256"])
            or any(
                Path(path).is_absolute() or ".." in Path(path).parts for path in names
            )
        ):
            raise RuntimeError(f"Invalid {name} patch file manifest")
        current = {path: digest(destination / path) for path in names}
        if current == spec["patched_files_sha256"]:
            prepared.append((name, destination, patch, spec, False))
            continue
        if verify_only or current != spec["source_files_sha256"]:
            raise RuntimeError(
                f"{name} source is not the reviewed request-state baseline"
            )
        # An overlay may live inside an unrelated Git worktree. Explicitly
        # anchor paths here; otherwise git apply silently skips patch paths
        # outside that worktree's inferred subdirectory prefix.
        subprocess.run(
            ["git", f"--work-tree={destination}", "apply", "--check", str(patch)],
            cwd=destination,
            check=True,
        )
        prepared.append((name, destination, patch, spec, True))
    # Validate both packages before touching either. Do not overwrite unknown
    # edits or quietly apply this on top of a different upstream revision.
    for name, destination, patch, spec, needed in prepared:
        if needed:
            subprocess.run(
                ["git", f"--work-tree={destination}", "apply", str(patch)],
                cwd=destination,
                check=True,
            )
        actual = {
            path: digest(destination / path) for path in spec["patched_files_sha256"]
        }
        if actual != spec["patched_files_sha256"]:
            raise RuntimeError(
                f"{name} output differs from reviewed request-state source"
            )
        print(
            f"Request-state {name} patch verified ({'applied' if needed else 'present'})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    apply(arguments.root, arguments.verify_only)
