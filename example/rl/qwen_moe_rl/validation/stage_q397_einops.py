"""Stage the pinned official pure-Python wheel; never invokes pip or imports Torch."""
from __future__ import annotations
import argparse
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

WHEEL_SHA = "54058201ac7087911181bfec4af6091bb59380360f069276601256a76af08193"
MANIFEST_SHA = "8941c4a78f1b3bced364b869cfeda6e20aabaa8465e11df200a9095d7cf68169"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_verified(path, expected, size):
    error = None
    for delay in (0, 0.3, 1, 2, 4, 8):
        if delay:
            time.sleep(delay)
        try:
            data = path.read_bytes()
            if len(data) != size or sha(data) != expected:
                raise ValueError(f"Existing bytes differ: {path}")
            return
        except OSError as exc:
            error = exc
    raise error


def exclusive_file(path, data, *, oss=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as out:
            out.write(data)
            out.flush()
            try:
                os.fsync(out.fileno())
            except OSError as exc:
                if not oss or exc.errno not in (errno.EPERM, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
                    raise
                print("OSS_FSYNC_UNSUPPORTED", str(path), exc.errno, flush=True)
    except FileExistsError:
        pass
    read_verified(path, sha(data), len(data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--persist-oss", action="store_true")
    parser.add_argument("--oss-directory", type=Path, help="Explicit immutable wheel archive directory")
    args = parser.parse_args()
    if args.persist_oss and args.oss_directory is None:
        parser.error("--persist-oss requires --oss-directory")
    wheel, raw_manifest = Path(args.wheel).read_bytes(), Path(args.manifest).read_bytes()
    assert len(wheel) == 65638 and sha(wheel) == WHEEL_SHA
    assert sha(raw_manifest) == MANIFEST_SHA
    manifest = json.loads(raw_manifest)
    expected = {row["path"]: row for row in manifest["files"]}
    destination = Path(args.project_root) / "logs/dependencies/einops-0.8.2"
    files = {}
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative = Path(member.filename)
            assert not relative.is_absolute() and ".." not in relative.parts
            assert member.filename.startswith(("einops/", "einops-0.8.2.dist-info/"))
            assert member.filename in expected and member.filename not in files
            data = archive.read(member)
            row = expected[member.filename]
            assert len(data) == row["bytes"] and sha(data) == row["sha256"]
            files[member.filename] = data
    assert set(files) == set(expected)
    exclusive_file(destination / "einops-0.8.2-py3-none-any.whl", wheel)
    for relative, data in files.items():
        exclusive_file(destination / "extracted" / relative, data)
    exclusive_file(destination / "manifest.json", raw_manifest)
    oss_receipt = None
    if args.persist_oss:
        oss = args.oss_directory
        exclusive_file(oss / "einops-0.8.2-py3-none-any.whl", wheel, oss=True)
        exclusive_file(oss / "manifest.json", raw_manifest, oss=True)
        index = {"schema": "q397-einops-official-wheel-archive-v1", "files": [
            {"path": "einops-0.8.2-py3-none-any.whl", "bytes": len(wheel), "sha256": WHEEL_SHA},
            {"path": "manifest.json", "bytes": len(raw_manifest), "sha256": MANIFEST_SHA}]}
        raw_index = (json.dumps(index, sort_keys=True, indent=2) + "\n").encode()
        exclusive_file(oss / "INDEX.json", raw_index, oss=True)
        ready = (json.dumps({"schema": "q397-einops-official-wheel-ready-v1", "index_sha256": sha(raw_index)}, sort_keys=True, indent=2) + "\n").encode()
        exclusive_file(oss / "READY.json", ready, oss=True)
        oss_receipt = dict(path=str(oss), index_sha256=sha(raw_index), ready_sha256=sha(ready))
    print("Q397_EINOPS_STAGED " + json.dumps(dict(destination=str(destination), files=len(files),
          wheel_sha256=WHEEL_SHA, manifest_sha256=MANIFEST_SHA, oss=oss_receipt)), flush=True)


if __name__ == "__main__":
    main()
