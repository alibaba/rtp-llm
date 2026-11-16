"""Apply the default-on ROCm RTP-ATOM optimization patch to the pinned wheel."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import tempfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory / "manifest.json").read_text())
    patch = directory / "mtp_hc.patch"
    if digest(patch.read_bytes()) != manifest["patch_sha256"]:
        raise RuntimeError("ATOM optimization patch hash mismatch")
    package = importlib.metadata.distribution("atom")
    if package.version != manifest["atom_version"]:
        raise RuntimeError(f"Unsupported ATOM version: {package.version}")
    targets = {name: Path(package.locate_file(name)) for name in manifest["files"]}
    current = {name: path.read_bytes() if path.exists() else None for name, path in targets.items()}
    hashes = {name: digest(data) if data is not None else None for name, data in current.items()}
    if all(hashes[name] == meta["patched_sha256"] for name, meta in manifest["files"].items()):
        print("ATOM optimization patch verified")
        return
    if args.verify_only:
        raise RuntimeError("Default-on patch is missing")
    if all(hashes[name] == meta["previous_sha256"] for name, meta in manifest["files"].items()):
        patch = directory / "upgrade.patch"
        if digest(patch.read_bytes()) != manifest["upgrade_sha256"]:
            raise RuntimeError("Upgrade patch hash mismatch")
    elif any(hashes[name] != meta["source_sha256"] for name, meta in manifest["files"].items()):
        raise RuntimeError("Unknown or partially patched ATOM files; refusing to overwrite")
    with tempfile.TemporaryDirectory(prefix="atom_topk512_") as temporary:
        root = Path(temporary)
        for name, data in current.items():
            if data is not None:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
        payloads = {name: (root / name).read_bytes() for name in targets}
        for name, data in payloads.items():
            if digest(data) != manifest["files"][name]["patched_sha256"]:
                raise RuntimeError(f"Patched hash mismatch: {name}")
            compile(data, name, "exec")
        for name, data in payloads.items():
            targets[name].write_bytes(data)
    print("Patch installed; restart workers to use ROCm RTP-ATOM defaults")


if __name__ == "__main__":
    main()
