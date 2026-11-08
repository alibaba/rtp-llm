"""Start the traditional worker with the selected server's Python/DSO closure."""

import argparse
import os
import sys
from pathlib import Path


def runtime_pythonpath(source, server):
    runfiles = Path(str(server) + ".runfiles")
    packages = sorted(runfiles.glob("*/site-packages"))
    if not packages:
        raise ValueError("The selected server has no Python dependency runfiles")
    cutlass = [
        package / "nvidia_cutlass_dsl/python_packages"
        for package in packages
        if (package / "nvidia_cutlass_dsl/python_packages/cutlass").is_dir()
    ]
    if not cutlass:
        raise ValueError("The selected server has no CUTLASS Python dependency")
    # CUTLASS lives below python_packages, outside the normal site-packages
    # import root. Publish it before any spawned rank can import a global copy.
    return os.pathsep.join(map(str, [source, runfiles, *cutlass, *packages]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source.resolve(strict=True)
    server = args.server.resolve(strict=True)
    sys.path.insert(0, str(source / "rtp_llm/test/smoke/kimi_k3"))
    from two_host_launcher import runtime_environment

    environment = runtime_environment(source, server)
    environment["PYTHONPATH"] = runtime_pythonpath(source, server)
    root = args.artifact_root
    if (
        not root.is_absolute()
        or root.is_symlink()
        or root.stat().st_uid != os.geteuid()
    ):
        raise ValueError("Runtime artifact root must belong to the non-root worker")
    home = root / ".runtime-home"
    home.mkdir(mode=0o700, exist_ok=False)
    environment["HOME"] = str(home)
    # LD_LIBRARY_PATH must be in the new process environment before its dynamic
    # loader starts; changing it after Python imports does not establish this.
    entry = source / "rtp_llm/test/smoke/kimi_k3/k31_two_host_smoke.py"
    os.execve(
        sys.executable, [sys.executable, str(entry), "--worker-control"], environment
    )


if __name__ == "__main__":
    main()
