"""Read-only PPU/Pouch build checks for a committed dual snapshot."""

import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from rtp_llm.models_py.modules.dsv41.ppu.pd_handoff import runtime_identity
from rtp_llm.tools.v41.preflight import SDK_VERSION


def staged_vipserver(root, source):
    """Verify an offline copy against the snapshot's unchanged dependency pin."""
    if not source:
        return None
    if not Path(source).is_absolute():
        raise ValueError("staged vipserver path must be absolute")
    root = Path(root).resolve(strict=True)
    source = Path(source).resolve(strict=True)
    if source == root or root in source.parents:
        raise ValueError("staged vipserver must be outside managed source")
    declarations = ast.parse((root / "internal_source/impls.bzl").read_text())
    pins = []
    for call in ast.walk(declarations):
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Name) or call.func.id != "new_git_repository":
            continue
        values = {
            item.arg: ast.literal_eval(item.value)
            for item in call.keywords
            if item.arg in ("name", "remote", "commit")
        }
        if values.get("name") == "vipserver":
            pins.append(values)
    if len(pins) != 1:
        raise ValueError("snapshot must declare exactly one vipserver pin")
    manifest_path = source / ".v41_dependency.json"
    manifest = json.loads(manifest_path.read_text())
    if any(manifest.get(key) != value for key, value in pins[0].items()):
        raise ValueError("staged vipserver differs from the snapshot's declared pin")
    actual = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix()
        if relative == manifest_path.name:
            continue
        if path.is_symlink():
            if not path.resolve().is_relative_to(source):
                raise ValueError(f"staged vipserver has an escaping link: {relative}")
            actual[relative] = {"kind": "symlink", "target": os.readlink(path)}
        elif path.is_file():
            actual[relative] = {
                "kind": "file",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "executable": bool(path.stat().st_mode & 0o111),
            }
        elif not path.is_dir():
            raise ValueError(f"unsupported staged vipserver entry: {relative}")
    if actual != manifest.get("files"):
        raise ValueError("staged vipserver file hashes or file set changed")
    template = root / "3rdparty/vipserver/vipserver.BUILD"
    if (source / "BUILD.bazel").read_bytes() != template.read_bytes():
        raise ValueError("staged vipserver BUILD differs from the committed template")
    return {
        "path": str(source),
        "commit": pins[0]["commit"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "file_count": len(actual),
    }


def owns_bazel_client(argv, paths):
    if not argv or "bazel" not in Path(argv[0]).name.lower():
        return False
    # Bazel rewrites the JVM's argv[0] to bazel(workspace). Keep the persistent
    # server; only another client using these exact output paths is a conflict.
    if "-jar" in argv and any(Path(token).name == "A-server.jar" for token in argv):
        return False
    for option, path in (("--output_user_root", paths[0]), ("--output_base", paths[1])):
        if f"{option}={path}" in argv or any(
            token == option and i + 1 < len(argv) and argv[i + 1] == str(path)
            for i, token in enumerate(argv)
        ):
            return True
    return False


def check(root, output_user_root, output_base, repository_cache):
    root = Path(root).resolve(strict=True)
    paths = [
        Path(p).resolve() for p in (output_user_root, output_base, repository_cache)
    ]
    if any(
        not Path(p).is_absolute()
        for p in (output_user_root, output_base, repository_cache)
    ):
        raise ValueError("Bazel/cache paths must be absolute")
    if any(p == root or root in p.parents for p in paths):
        raise ValueError("Bazel/cache paths must be outside managed snapshot source")
    manifest = json.loads((root / ".v41_snapshot.json").read_text())
    if not all(manifest.get(k) for k in ("public_commit", "internal_commit")):
        raise ValueError("native builds require both committed repository identities")
    identity = runtime_identity(root)
    vipserver = staged_vipserver(root, os.environ.get("RTP_V41_VIPSERVER_SOURCE"))
    if sys.version_info[:2] != (3, 12):
        raise ValueError("PPU native build requires Python 3.12")
    sdk = Path(os.environ.get("PPU_HOME", "/usr/local/PPU_SDK"))
    if (sdk / "release.yaml").read_text().strip() != f"version: {SDK_VERSION}":
        raise ValueError("unexpected PPU SDK version")
    if not shutil.which("ppu-smi") or not (
        shutil.which("bazel") or shutil.which("bazelisk")
    ):
        raise ValueError("PPU SDK and Bazel must be on PATH")
    overlay = root / "internal_source/.internal_bazelrc"
    settings = overlay.read_text()
    if (
        "--override_repository=rtp_deps=" in settings
        or "--override_repository=arch_config=" in settings
    ):
        raise ValueError("legacy closed dependency/platform overrides are still active")
    if "--override_repository=internal_impls=" not in settings:
        raise ValueError("closed implementation seam is absent")
    busy = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode().strip("\0").split("\0")
            if owns_bazel_client(argv, paths):
                busy.append({"pid": int(entry.name), "argv": argv})
        except (OSError, UnicodeError):
            continue
    if busy:
        raise ValueError(f"Bazel clients already own this task's output paths: {busy}")
    for path in paths:
        parent = path
        while not parent.exists():
            parent = parent.parent
        if not os.access(parent, os.W_OK | os.X_OK):
            raise ValueError(f"Bazel/cache parent is not writable: {parent}")
        if shutil.disk_usage(parent).free < 10 * 1024**3:
            raise ValueError(f"less than 10 GiB available for build/cache: {parent}")
    return {
        "status": "pass",
        "uid": os.getuid(),
        "python": sys.version,
        "public_commit": manifest["public_commit"],
        "internal_commit": manifest["internal_commit"],
        "identity": identity,
        "output_user_root": str(paths[0]),
        "output_base": str(paths[1]),
        "repository_cache": str(paths[2]),
        "staged_vipserver": vipserver,
        "cache_mode": os.environ.get("RTP_V41_BAZEL_CACHE_MODE", "remote"),
        "cache_reason": os.environ.get("RTP_V41_BAZEL_CACHE_REASON", ""),
        "execution": "PPU in the explicitly selected existing Pouch container",
        "gpu_allocation": "none; build only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-user-root", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--repository-cache", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = check(
        args.root, args.output_user_root, args.output_base, args.repository_cache
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
