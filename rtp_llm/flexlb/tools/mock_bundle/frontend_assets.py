"""Stage frontend configuration/tokenizer assets when a CPU Pod has no FUSE sidecar.

The remote model URI and its existing credentials come from Whale's environment.
Weight files are never downloaded. This helper is for the dedicated Mock frontend.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from urllib.parse import parse_qs, urlsplit


SUFFIXES = {".json", ".txt", ".model", ".tiktoken", ".py", ".jinja", ".vocab", ".merges"}
MAX_FILE = 64 * 1024 * 1024
MAX_TOTAL = 512 * 1024 * 1024


def asset_name(key, prefix):
    if not key.startswith(prefix):
        return None
    name = key[len(prefix):]
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or path.suffix not in SUFFIXES:
        return None
    return name


def stage(uri, root):
    parsed = urlsplit(uri)
    if not parsed.scheme:
        if not Path(uri).is_dir():
            raise ValueError("frontend asset directory is missing")
        return uri
    if parsed.scheme != "oss":
        raise ValueError("frontend asset staging requires a local directory or OSS URI")
    import oss2
    query = parse_qs(parsed.query)
    endpoint = query["OSS_ENDPOINT"][0]
    if not endpoint.startswith(("http://", "https://")):
        endpoint = "https://" + endpoint
    bucket = oss2.Bucket(oss2.Auth(query["OSS_ACCESS_ID"][0], query["OSS_ACCESS_KEY"][0]),
                         endpoint, parsed.netloc, connect_timeout=15)
    canonical = parsed._replace(query="", fragment="").geturl()
    destination = root / hashlib.sha256(canonical.encode()).hexdigest()[:24]
    manifest = destination / "frontend-assets.json"
    if manifest.is_file():
        recorded = json.loads(manifest.read_text())
        if recorded.get("source") == canonical and all(
                (destination / name).is_file() and
                hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
                for name, digest in recorded["files"].items()):
            return str(destination)
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="staging-", dir=root))
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    files, total = {}, 0
    try:
        for count, item in enumerate(oss2.ObjectIterator(bucket, prefix=prefix)):
            if count >= 10000:
                raise ValueError("frontend asset listing exceeds its bound")
            name = asset_name(item.key, prefix)
            if name is None:
                continue
            total += item.size
            if item.size > MAX_FILE or total > MAX_TOTAL:
                raise ValueError("frontend metadata exceeds its download budget")
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            bucket.get_object_to_file(item.key, str(target))
            files[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        if "config.json" not in files or not any("tokenizer" in n or "vocab" in n for n in files):
            raise ValueError("frontend model config or tokenizer assets are missing")
        (temporary / manifest.name).write_text(json.dumps({"source": canonical, "files": files}))
        if destination.exists():
            shutil.rmtree(destination)
        temporary.rename(destination)
        print(json.dumps({"frontend_assets": str(destination), "files": len(files), "bytes": total}), flush=True)
        return str(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    root = Path(os.environ.get("HIPPO_PROC_WORKDIR", ".")) / "frontend-assets"
    for key in ("CHECKPOINT_PATH", "TOKENIZER_PATH"):
        uri = os.environ.get(key)
        if not uri:
            raise ValueError(key + " is required")
        os.environ[key] = stage(uri, root)
    os.execv("/bin/bash", ["bash", "/usr/bin/maga_start.sh"])


if __name__ == "__main__":
    main()
