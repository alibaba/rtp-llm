"""CI resource controls around the traditional smoke's server configuration."""

import json
import resource

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
    shared.main(configure=launch_config, extend_parser=extend_parser)
