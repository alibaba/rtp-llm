"""Apply the version-scoped ATOM main compatibility patch after wheel install."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in ("atom", "transformers"):
        expected = manifest[name + "_version"]
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"Expected {name} {expected}, found {actual}")
    patch = directory / "compat.patch"
    if digest(patch) != manifest["patch_sha256"]:
        raise RuntimeError("Compatibility patch hash mismatch")
    site = Path(importlib.metadata.distribution("atom").locate_file(""))
    before = manifest["source_files_sha256"]
    after = manifest["patched_files_sha256"]
    current = {name: digest(site / name) for name in before}
    if current == after:
        print("ATOM main compatibility patch verified")
        return
    if current != before:
        raise RuntimeError("ATOM files differ from both wheel and patched manifest")
    if args.verify_only:
        raise RuntimeError("ATOM main compatibility patch is not installed")
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=site, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=site, check=True)
    if {name: digest(site / name) for name in after} != after:
        raise RuntimeError("Patched ATOM files failed hash verification")
    print("ATOM main compatibility patch applied and verified")


if __name__ == "__main__":
    main()
