"""Assemble a checked native runtime and generated protos outside the checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as reader:
        for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.source_root.resolve(), args.output.resolve()
    if output.exists() or output == root or root in output.parents:
        parser.error(
            "output must be a new artifact directory outside the source checkout"
        )
    snapshot = json.loads((root / ".v41_snapshot.json").read_text())
    build = json.loads(args.build_manifest.read_text())
    digest = hashlib.sha256(
        json.dumps(snapshot["files"], sort_keys=True).encode()
    ).hexdigest()
    if snapshot.get("mode") != "snapshot" or digest != snapshot["source_sha256"]:
        raise ValueError("invalid committed source snapshot")
    if any(
        build[key] != snapshot[key]
        for key in ("snapshot_id", "commit", "source_sha256")
    ):
        raise ValueError("native build and source snapshot identities differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".native-runtime-", dir=output.parent))
    try:
        for entry in snapshot["files"]:
            source, target = root / entry["path"], temporary / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            actual = (
                os.readlink(source) if entry["kind"] == "symlink" else sha256(source)
            )
            if actual != entry["value"]:
                raise ValueError(f"published source changed: {entry['path']}")
            if entry["kind"] == "symlink":
                target.symlink_to(actual)
            else:
                shutil.copy2(source, target)
        shutil.copy2(root / ".v41_snapshot.json", temporary / ".v41_snapshot.json")
        runtime_libraries = {}
        for name, library in build["libraries"].items():
            source = Path(library["path"])
            if sha256(source) != library["sha256"]:
                raise ValueError(f"native library changed after build: {name}")
            relative = Path("rtp_llm/libs") / name
            target = temporary / relative
            # Bazel can replace the output on the next build. Keep this run's
            # checked binary with its snapshot instead of linking that cache.
            shutil.copy2(source, target)
            if sha256(target) != library["sha256"]:
                raise ValueError(f"native library copy differs from build: {name}")
            runtime_libraries[name] = {
                **library,
                "runtime_path": str(relative),
                "storage": "snapshot_copy",
            }
        # Generate Python stubs as build artifacts. Managed source files stay
        # untouched, including during subsequent snapshot publication.
        # Python clients use these two gRPC protocols. CacheStore's ARPC
        # protocols are compiled into the native library and need no Python
        # stubs or dependency on the external ARPC proto include tree.
        cpp_protos = [
            root / "rtp_llm/cpp/model_rpc/proto" / name
            for name in ("model_rpc_service.proto", "flexlb_schedule_service.proto")
        ]
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{root}",
                f"--python_out={temporary}",
                f"--grpc_python_out={temporary}",
                *[str(path) for path in cpp_protos],
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(root / "rtp_llm/dash_sc/proto/create_grpc_proto.py"),
                str(temporary / "rtp_llm/dash_sc/proto"),
            ],
            check=True,
        )
        generated = {
            str(path.relative_to(temporary)): sha256(path)
            for path in sorted(temporary.rglob("*_pb2*.py"))
        }
        manifest = {
            "snapshot_id": snapshot["snapshot_id"],
            "commit": snapshot["commit"],
            "source_sha256": digest,
            "libraries": runtime_libraries,
            "generated_python": generated,
            "python": sys.version,
        }
        (temporary / "native-runtime.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        temporary.rename(output)
        print(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
