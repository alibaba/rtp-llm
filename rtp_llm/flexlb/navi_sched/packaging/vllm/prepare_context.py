"""Keep vLLM source and NaviSched install inputs separate in the image context."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlsplit, urlunsplit


def prepare_context(source, bundle, destination):
    source, bundle, destination = (
        path.resolve() for path in (source, bundle, destination)
    )
    if destination.exists():
        raise RuntimeError("image context already exists: " + str(destination))
    excluded = {
        ".navi_sources",
        "navi_sched_dist",
        destination.name,
        ".venv",
        "cpu-results",
        "mediafetch-results",
        "__pycache__",
        ".git",
    }
    destination.mkdir()

    def ignore(directory, names):
        return sorted(set(names) & excluded) if Path(directory) == source else []

    def copy_file(src, dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return dst

    code = destination / "vllm_source"
    if (source / ".git").exists():
        # Keep SCM version resolution without copying CI checkout credentials.
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--no-checkout",
                "--no-hardlinks",
                "--dissociate",
                str(source),
                str(code),
            ],
            check=True,
        )
        remote = subprocess.check_output(
            ["git", "-C", str(source), "remote", "get-url", "origin"], text=True
        ).strip()
        if remote.startswith(("http://", "https://")):
            url = urlsplit(remote)
            remote = urlunsplit(
                (url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", "")
            )
        subprocess.run(
            ["git", "-C", str(code), "remote", "set-url", "origin", remote], check=True
        )
    shutil.copytree(
        source,
        code,
        ignore=ignore,
        dirs_exist_ok=True,
        symlinks=True,
        copy_function=copy_file,
    )
    shutil.copytree(
        bundle, destination / "navi_sched_dist", symlinks=True, copy_function=copy_file
    )
    patterns = []
    for line in (source / ".dockerignore").read_text().splitlines():
        if line.startswith("/"):
            line = "/vllm_source/" + line[1:]
        elif "/" in line and not line.startswith(("#", "!")):
            line = "vllm_source/" + line
        patterns.append(line)
    (destination / ".dockerignore").write_text("\n".join(patterns) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare_context(args.source, args.bundle, args.destination)
