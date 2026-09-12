"""Create a reviewable source overlay, patch and compact result bundle."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--base", default="1f57ca1b5b73467575702fbeebbd23ddebf18a83")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    base_commit = (
        git(repo, "rev-parse", "--verify", args.base + "^{commit}").decode().strip()
    )
    tracked = (
        git(
            repo, "diff", "--diff-filter=M", "--name-only", base_commit, "--", "rtp_llm"
        )
        .decode()
        .splitlines()
    )
    added = (
        git(
            repo,
            "diff",
            "--diff-filter=A",
            "--name-only",
            base_commit,
            "--",
            "rtp_llm/models_py/modules/dsv4",
        )
        .decode()
        .splitlines()
    )
    untracked = [
        p
        for p in git(
            repo,
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "rtp_llm/models_py/modules/dsv4",
        )
        .decode()
        .splitlines()
        if p.endswith((".py", "/BUILD"))
    ]
    new_source = sorted(set(added + untracked))
    tools = sorted(
        str(p.relative_to(repo))
        for p in (repo / "dsv4_migration/harness").iterdir()
        if p.suffix in (".py", ".sh")
    )
    docs = sorted(
        str(p.relative_to(repo)) for p in (repo / "dsv4_migration").glob("*.md")
    )
    metadata = ["dsv4_migration/.gitignore"]
    files = sorted(set(tracked + new_source + tools + docs + metadata))
    patch = (
        git(repo, "diff", "--binary", base_commit, "--", *tracked) if tracked else b""
    )
    for name in new_source + tools + docs + metadata:
        result = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--", "/dev/null", name],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode())
        patch += result.stdout
    patch_path = destination / "dsv4-main-offload.patch"
    patch_path.write_bytes(patch)
    for name in files:
        target = destination / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, target)
    for name in docs:
        shutil.copy2(repo / name, destination / Path(name).name)
    # Reconstruct only touched base files, apply the patch, and verify every byte.
    with tempfile.TemporaryDirectory(prefix="dsv4-patch-check-") as directory:
        base = Path(directory)
        if tracked:
            archive = git(repo, "archive", base_commit, "--", *tracked)
            with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
                for name in tracked:
                    target = base / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with handle.extractfile(name) as source:
                        target.write_bytes(source.read())
        subprocess.run(
            ["git", "apply", "--check", str(patch_path)], cwd=base, check=True
        )
        subprocess.run(["git", "apply", str(patch_path)], cwd=base, check=True)
        for name in files:
            if (base / name).read_bytes() != (repo / name).read_bytes():
                raise RuntimeError(f"patch reconstruction differs: {name}")
    for source in sorted((repo / "dsv4_migration/results").rglob("*")):
        relative = source.relative_to(repo / "dsv4_migration")
        if not source.is_file() or "server" in relative.parts:
            continue
        if source.suffix not in (".json", ".md"):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for original, name in (
        ("/tmp/dsv4-final-gpu-tests.log", "gpu-tests.log"),
        ("/tmp/dsv4-build-calibration.log", "native-build.log"),
    ):
        if Path(original).exists():
            shutil.copy2(original, destination / name)
    evidence = {}
    for log in sorted(
        (repo / "dsv4_migration/results").glob("*/server/logs/main_*.log")
    ):
        selected = []
        memory_lines = 0
        with log.open(errors="replace") as handle:
            for line in handle:
                if "DSV4 CSA" in line:
                    selected.append(line.rstrip())
                elif "BENCH_REAL_PREFILL_MEMORY" in line and memory_lines < 2:
                    selected.append(line.rstrip())
                    memory_lines += 1
        if selected:
            evidence[str(log.relative_to(repo / "dsv4_migration"))] = selected
    (destination / "runtime-evidence.json").write_text(json.dumps(evidence, indent=2))
    manifest = {
        "base_commit": base_commit,
        "git_head": git(repo, "rev-parse", "HEAD").decode().strip(),
        "branch": git(repo, "branch", "--show-current").decode().strip(),
        "patch_applied_and_source_bytes_verified": True,
        "source_files": files,
        "excluded": [
            "model weights",
            "trace prompts",
            "verbose server logs",
            "generated protobuf files",
            "machine-specific internal_source symlink",
        ],
        "files_sha256": {
            str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(destination.rglob("*"))
            if p.is_file()
        },
    }
    (destination / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "destination": str(destination),
                "source_files": len(files),
                "artifacts": len(manifest["files_sha256"]),
                "patch_verified": True,
            }
        )
    )


if __name__ == "__main__":
    main()
