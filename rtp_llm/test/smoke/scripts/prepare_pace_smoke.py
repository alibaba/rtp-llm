import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from functools import partial
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_FILES = {
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "generation_config.json": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
CRANE_ARCHIVE = "go-containerregistry_Linux_x86_64.tar.gz"
CRANE_VERSION = "v0.22.1"
CRANE_SHA256 = "0ab7a1d6932a213aed964ce97666c3077fe691c8606413674a8b3e0b9ec4cda0"


def matches_sha256(path, expected):
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest() == expected


def cache_file(path, digest, populate):
    if matches_sha256(path, digest):
        return
    print(f"Preparing {path.name}", file=sys.stderr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        staging = Path(directory) / path.name
        populate(staging)
        if not matches_sha256(staging, digest):
            raise RuntimeError(f"SHA256 mismatch for {path.name}")
        staging.replace(path)


def download(url, destination):
    result = subprocess.run(
        [
            "curl", "--fail", "--silent", "--show-error", "--location",
            "--netrc-optional", "--retry", "2", "--connect-timeout", "15",
            "--max-time", "600", "--output", str(destination), url,
        ],
    )
    if result.returncode:
        raise RuntimeError(f"Download failed for {destination.name} (curl exit {result.returncode})")


def extract_file(archive_path, member, destination):
    with tarfile.open(archive_path) as archive:
        if not archive.getmember(member).isfile():
            raise RuntimeError(f"{member} is not a regular file in {archive_path.name}")
        with archive.extractfile(member) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)


def crane_binary(cache_dir):
    installed = shutil.which("crane")
    if installed:
        return installed
    tools = cache_dir / "crane" / CRANE_VERSION
    binary = tools / "crane"
    if not binary.is_file():
        archive = tools / CRANE_ARCHIVE
        url = os.environ.get("PACE_SMOKE_CRANE_URL") or (
            f"https://github.com/google/go-containerregistry/releases/download/{CRANE_VERSION}/{CRANE_ARCHIVE}"
        )
        cache_file(archive, CRANE_SHA256, partial(download, url))
        with tempfile.TemporaryDirectory(dir=tools) as directory:
            staging = Path(directory) / "crane"
            extract_file(archive, "crane", staging)
            staging.chmod(0o755)
            staging.replace(binary)
    return str(binary)


def open_anonymous_blob(reference):
    image, digest = reference.rsplit("@", 1)
    registry, repository = image.split("/", 1)
    url = f"https://{registry}/v2/{repository}/blobs/{digest}"
    try:
        return urlopen(url, timeout=30)
    except HTTPError as error:
        scheme, _, challenge = error.headers.get("WWW-Authenticate", "").partition(" ")
        if error.code != 401 or scheme.lower() != "bearer":
            raise
        fields = dict(re.findall(r'(\w+)="([^"]+)"', challenge))
        realm = fields.pop("realm", "")
        if urlsplit(realm).scheme != "https":
            raise
    separator = "&" if urlsplit(realm).query else "?"
    with urlopen(f"{realm}{separator}{urlencode(fields)}", timeout=30) as response:
        reply = json.load(response)
    token = reply.get("token") or reply.get("access_token")
    if not token:
        return None
    request = Request(url)
    request.add_unredirected_header("Authorization", f"Bearer {token}")
    return urlopen(request, timeout=30)


def registry_blob(cache_dir, reference, destination):
    try:
        stream = open_anonymous_blob(reference)
    except HTTPError as error:
        if error.code not in (401, 403):
            raise
        stream = None
    if stream is not None:
        with stream, destination.open("wb") as output:
            shutil.copyfileobj(stream, output, length=1024 * 1024)
        return
    with destination.open("wb") as output:
        subprocess.run(
            [crane_binary(cache_dir), "blob", reference],
            stdout=output,
            check=True,
            timeout=600,
        )


def registry_rpm(cache_dir, reference, member, destination):
    layer = destination.with_name("layer.tar")
    registry_blob(cache_dir, reference, layer)
    extract_file(layer, member, destination)


def prepare_artifacts(cache_dir):
    repo = Path(__file__).resolve().parents[4]
    repositories = repo / "3rdparty/pace_test_runtime/repositories.bzl"
    manifest = next(
        assignment.value
        for assignment in ast.parse(repositories.read_text()).body
        if isinstance(assignment, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PACE_TEST_ARTIFACTS"
                for target in assignment.targets)
    )
    artifacts = ast.literal_eval(manifest)
    distfiles = cache_dir / "distfiles"
    for name, (url, digest) in {**artifacts["rpms"], **artifacts["runtime_rpms"]}.items():
        filename = Path(urlsplit(url).path).name
        if not filename.endswith(".x86_64.rpm"):
            continue
        layer = artifacts["rpm_layers"].get(name)
        populate = partial(registry_rpm, cache_dir, *layer) if layer else partial(download, url)
        cache_file(distfiles / filename, digest, populate)

    digest = artifacts["manager_layers"]["x86"]
    blob_name = f"sha256:{digest}"
    reference = f"{artifacts['manager_repo']}@{blob_name}"
    cache_file(distfiles / blob_name, digest, partial(registry_blob, cache_dir, reference))


def prepare_model(cache_dir):
    model_name = MODEL_ID.rsplit("/", 1)[-1]
    requested = os.environ.get("CHECKPOINT_PATH")
    mounted = Path(requested or f"/mnt/nas1/hf/{model_name}").resolve()
    mismatch = next(
        (name for name, digest in MODEL_FILES.items() if not matches_sha256(mounted / name, digest)),
        None,
    )
    if mismatch is None:
        return mounted
    if requested:
        raise RuntimeError(f"CHECKPOINT_PATH does not match the pinned model: {mounted / mismatch}")

    checkpoint = cache_dir / model_name / MODEL_REVISION
    endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    for name, digest in MODEL_FILES.items():
        url = f"{endpoint}/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
        cache_file(checkpoint / name, digest, partial(download, url))
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Prepare pinned PACE smoke inputs.")
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--artifacts", action="store_true")
    parser.add_argument("--model", action="store_true")
    args = parser.parse_args()
    if not (args.artifacts or args.model):
        parser.error("select --artifacts, --model, or both")
    cache_dir = args.cache_dir.resolve()
    try:
        if args.artifacts:
            prepare_artifacts(cache_dir)
        if args.model:
            print(prepare_model(cache_dir))
    except (OSError, RuntimeError, ValueError, KeyError, StopIteration, tarfile.TarError,
            subprocess.SubprocessError) as error:
        print(f"PACE input preparation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
