"""Shared K3.1 server launch and bounded core evidence."""

import json
import resource
import os
import stat
from pathlib import Path

from rtp_llm.test.smoke.kimi_k3 import server_launcher as shared


def configure_core_dump_limit(limit_mib):
    if type(limit_mib) is not int or not 1 <= limit_mib <= 1024:
        raise ValueError("Core dump limit must be between 1 and 1024 MiB")
    before = resource.getrlimit(resource.RLIMIT_CORE)
    requested = limit_mib * 1024**2
    soft = requested if before[0] == resource.RLIM_INFINITY else min(before[0], requested)
    resource.setrlimit(resource.RLIMIT_CORE, (soft, before[1]))
    return {"requested_bytes": requested, "previous_limits": list(before),
            "effective_limits": list(resource.getrlimit(resource.RLIMIT_CORE)),
            "core_files_remain_test_failures": True}


def extend_parser(parser):
    parser.add_argument("--core-dump-limit-mib", type=int, default=64)
    parser.add_argument("--block-tree-transfer-worker-count", type=int)
    parser.add_argument("--traditional-artifact-root", type=Path)
    parser.add_argument("--draft-allow-checkpoint-root")


def validate_destination(path, args):
    if args.traditional_artifact_root is None:
        return shared.require_bazel_destination(path)
    root = args.traditional_artifact_root
    if not root.is_absolute() or root.is_symlink() or root == Path(root.anchor):
        raise ValueError("Traditional artifact root must be an explicit private directory")
    root = root.resolve(strict=True)
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("Traditional output escapes the selected artifact root")
    for directory in (path, *path.parents):
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o022:
            raise ValueError("Traditional output must be owned and private")
        if directory == root:
            return


def launch_config(args):
    environment, command = shared.launch_config(args)
    # Preserve per-rank event files inside this Bazel action's own outputs.
    if args.model_type == "kimi_k31":
        environment["FT_SERVER_TEST"] = "0"
    environment["K31_CI_CORE_DUMP_POLICY"] = json.dumps(
        configure_core_dump_limit(args.core_dump_limit_mib)
    )
    return environment, command


if __name__ == "__main__":
    shared.main(configure=launch_config, extend_parser=extend_parser,
                destination_validator=validate_destination)
