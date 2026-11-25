#!/usr/bin/env python3

import argparse
import fcntl
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def commit_sha(value):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise argparse.ArgumentTypeError("expected a full 40-character Git commit SHA")
    return value.lower()


def run(*command, cwd, capture=False, env=None):
    return subprocess.run(
        [str(arg) for arg in command],
        cwd=cwd,
        env=env,
        check=True,
        universal_newlines=True,
        stdout=subprocess.PIPE if capture else None,
    ).stdout


def stage_file(source, destination):
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        pending = Path(staging) / destination.name
        shutil.copy2(source, pending)
        pending.replace(destination)


def checkout(path, repo, sha):
    try:
        if (path / ".git").is_dir():
            actual = run("git", "rev-parse", "--verify", "HEAD", cwd=path, capture=True).strip()
            if actual != sha:
                raise RuntimeError(f"{path} is at {actual}, expected {sha}")
            return
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f"cannot initialize checkout in nonempty directory {path}")
        with tempfile.TemporaryDirectory(dir=path.parent, prefix=f".{path.name}-") as staging:
            pending = Path(staging) / "repo"
            pending.mkdir()
            run("git", "init", "--quiet", cwd=pending)
            run("git", "remote", "add", "origin", repo, cwd=pending)
            run("git", "fetch", "--no-tags", "--depth=1", "origin", sha, cwd=pending)
            run("git", "checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=pending)
            actual = run("git", "rev-parse", "HEAD", cwd=pending, capture=True).strip()
            if actual != sha:
                raise RuntimeError(f"{path} fetched {actual}, expected {sha}")
            pending.replace(path)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"failed to check out {path.name} at {sha}; check Git access, the commit, and any incomplete cache at {path}"
        ) from error


def build_pace(args, source, output):
    poco = run("git", "ls-tree", "HEAD", "lib/poco", cwd=source, capture=True).split()
    git_options = []
    if poco[:3] == ["160000", "commit", "02c86d4cddb14de293eabdb795e4a7c61745eb2b"]:
        git_options = ["-c", "submodule.lib/poco.url=https://github.com/pocoproject/poco.git"]
    run("git", *git_options, "submodule", "update", "--init", "--recursive", "--depth=1", cwd=source)
    build_dir = output / "pace-build"
    run(
        "cmake", "-S", source, "-B", build_dir, *args.cmake_arg,
        "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_CUDA=OFF", "-DBUILD_CLIENT=OFF",
        "-DENABLE_MPI=OFF", "-DENABLE_OTEL=OFF", "-DTENT_ENABLE_BAREX=OFF",
        "-DTENT_ENABLE_ETCD_META=OFF", "-DTENT_ENABLE_REDIS_META=OFF",
        "-DTENT_ENABLE_RDMA=ON", "-DTENT_BUILD_SHARED=ON",
        cwd=source,
    )
    cache = (build_dir / "CMakeCache.txt").read_text()
    for variable in ("IBVERBS_LIB:FILEPATH", "IBVERBS_INCLUDE:PATH"):
        if not re.search(rf"^{variable}=(?!.*-NOTFOUND$).+$", cache, re.MULTILINE):
            raise RuntimeError(f"PACE native RDMA requires {variable}; install libibverbs development files")
    run(
        "cmake", "--build", build_dir, "--parallel", args.jobs,
        "--target", "tair_mempool_meta_service", "tair_mempool_server", "tent_shared",
        cwd=source,
    )
    bin_dir = output / "pace" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for binary in ("tair_mempool_meta_service", "tair_mempool_server"):
        stage_file(build_dir / "bin" / binary, bin_dir / binary)
    libraries = sorted((build_dir / "lib").glob("libtent_shared.so*"))
    if not libraries:
        raise RuntimeError("PACE build did not produce libtent_shared")
    for library in libraries:
        stage_file(library, bin_dir / library.name)


def build_kvcm(args, public, mempool, output):
    stub = public / "stub_source"
    if not stub.is_symlink():
        raise RuntimeError(f"expected the KVCM stub_source symlink at {stub}")
    if os.readlink(stub) != "../internal_source":
        stub.unlink()
        stub.symlink_to("../internal_source")

    env = os.environ.copy()
    legacy_havenask = re.search(
        r'name\s*=\s*"havenask"\s*,\s*remote\s*=\s*"(git@gitlab\.alibaba-inc\.com:[^"]+)"\s*,\s*'
        r'commit\s*=\s*"f7317cb70a2e04c62d3b6a5f7f3c6ae75ab2149c"',
        (stub / "deps/git.bzl").read_text(),
    )
    if legacy_havenask:
        index = int(env.get("GIT_CONFIG_COUNT", "0"))
        env[f"GIT_CONFIG_KEY_{index}"] = "url.https://github.com/alibaba/havenask.git.insteadOf"
        env[f"GIT_CONFIG_VALUE_{index}"] = legacy_havenask[1]
        env["GIT_CONFIG_COUNT"] = str(index + 1)

    override = output / "tair_mempool"
    override.mkdir(exist_ok=True)
    for entry in mempool.iterdir():
        if entry.name not in (".git", "BUILD", "WORKSPACE") and not (override / entry.name).is_symlink():
            (override / entry.name).symlink_to(entry)
    shutil.copy2(public / "3rdparty/tair_mempool/tair_mempool.BUILD", override / "BUILD")
    (override / "WORKSPACE").write_text('workspace(name = "tair_mempool")\n')

    bazel = [
        args.bazel, f"--output_user_root={output / 'bazel'}", "build",
        f"--jobs={args.jobs}", *args.bazel_arg, f"--override_repository=tair_mempool={override}",
    ]
    sdk = output / "kvcm-sdk" / "file"
    sdk.mkdir(parents=True, exist_ok=True)
    run(
        *bazel, "--config=client_with_cuda", "--config=client_only_header",
        f"--@rules_cuda//cuda:archs={args.cuda_archs}",
        "//package/rpm/client:kv-cache-manager-client-rpm",
        cwd=public,
        env=env,
    )
    stage_file(public / "bazel-bin/package/rpm/client/kv-cache-manager-client.rpm", sdk / "sdk.rpm")
    (sdk / "BUILD.bazel").write_text(
        'filegroup(name = "file", srcs = ["sdk.rpm"], visibility = ["//visibility:public"])\n'
    )
    (sdk.parent / "WORKSPACE").write_text("")

    manager = output / "kvcm-manager"
    run(*bazel, "--action_env=PATH", "//kv_cache_manager:kv_cache_manager_bin", cwd=public, env=env)
    (manager / "bin").mkdir(parents=True, exist_ok=True)
    stage_file(
        public / "bazel-bin/kv_cache_manager/kv_cache_manager_bin",
        manager / "bin/kv_cache_manager_bin",
    )
    (manager / "etc").mkdir(exist_ok=True)
    for config in (public / "package/etc").iterdir():
        if config.suffix in (".conf", ".json"):
            shutil.copy2(config, manager / "etc" / config.name)
    if not (manager / "etc/default_startup_config.json").is_file():
        raise RuntimeError("missing KVCM default_startup_config.json")
    (manager / "BUILD.bazel").write_text('exports_files(["bin/kv_cache_manager_bin"])\n')
    (manager / "WORKSPACE").write_text("")
    is_arm = platform.machine() == "aarch64"
    sdk_repo = "remote_kv_cache_manager_client_rpm_cuda130_" + ("arm" if is_arm else "x86")
    manager_repo = "remote_kv_cache_manager_server_cuda130"
    (output / "rtp-smoke.bazelrc").write_text(
        f"build:pace_source {shlex.quote(f'--override_repository={sdk_repo}={sdk.parent}')}\n"
        f"build:pace_source {shlex.quote(f'--override_repository={manager_repo}={manager}')}\n"
        "test:pace_source --test_env=KVCM_PACE_FIXTURE\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Build the PACE smoke dependencies from three exact commits.")
    parser.add_argument("--kvcm-public-commit", required=True, type=commit_sha)
    parser.add_argument("--kvcm-internal-commit", required=True, type=commit_sha)
    parser.add_argument("--mempool-commit", required=True, type=commit_sha)
    parser.add_argument("--kvcm-public-repo", default="https://github.com/alibaba/tair-kvcache.git")
    parser.add_argument("--kvcm-internal-repo", help="Git URL or path for the KVCM internal overlay")
    parser.add_argument("--mempool-repo", required=True, help="Git URL or path for the PACE source")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cuda-archs", required=True, help="rules_cuda architecture list for the KVCM SDK")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--bazel", default="bazelisk")
    parser.add_argument("--cmake-arg", action="append", default=[])
    parser.add_argument("--bazel-arg", action="append", default=[])
    parser.add_argument("--component", choices=("all", "pace", "kvcm"), default="all")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.component != "pace" and not args.kvcm_internal_repo:
        parser.error("--kvcm-internal-repo is required when building KVCM")
    os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")
    os.environ.setdefault("GIT_HTTP_LOW_SPEED_LIMIT", "1")
    os.environ.setdefault("GIT_HTTP_LOW_SPEED_TIME", "120")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = {
        "kvcm_public": args.kvcm_public_commit,
        "kvcm_internal": args.kvcm_internal_commit,
        "mempool": args.mempool_commit,
        "machine": platform.machine(),
        "cuda_archs": args.cuda_archs,
        "cmake_args": args.cmake_arg,
        "bazel_args": args.bazel_arg,
    }
    with (output / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = output / "sources.json"
        if manifest.exists() and json.loads(manifest.read_text()) != spec:
            raise RuntimeError(f"{output} already belongs to a different build")
        if not manifest.exists():
            manifest.write_text(json.dumps(spec, indent=2) + "\n")
        artifacts = {
            "pace": (
                "pace/bin/tair_mempool_meta_service",
                "pace/bin/tair_mempool_server",
                "pace/bin/libtent_shared.so.5",
            ),
            "kvcm": (
                "kvcm-sdk/file/sdk.rpm",
                "kvcm-manager/bin/kv_cache_manager_bin",
                "kvcm-manager/etc/default_startup_config.json",
                "rtp-smoke.bazelrc",
            ),
        }
        for component in artifacts if args.component == "all" else (args.component,):
            ready = output / (component + ".ready")
            if ready.is_file() and all((output / artifact).is_file() for artifact in artifacts[component]):
                print(f"Reusing {component} artifacts: {output}")
                continue
            if ready.exists():
                ready.unlink()
            mempool = output / "mempool-src"
            checkout(mempool, args.mempool_repo, args.mempool_commit)
            if component == "pace":
                build_pace(args, mempool, output)
            else:
                if not shutil.which("rpmbuild"):
                    raise RuntimeError("install rpm-build in the KVCM build environment first")
                internal = output / "kvcm-src"
                public = internal / "github-opensource"
                checkout(internal, args.kvcm_internal_repo, args.kvcm_internal_commit)
                checkout(public, args.kvcm_public_repo, args.kvcm_public_commit)
                build_kvcm(args, public, mempool, output)
            if not all((output / artifact).is_file() for artifact in artifacts[component]):
                raise RuntimeError(f"{component} build did not produce the required artifacts in {output}")
            ready.touch()
    print(f"Build artifacts: {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"PACE smoke build failed: {error}", file=sys.stderr)
        sys.exit(1)
