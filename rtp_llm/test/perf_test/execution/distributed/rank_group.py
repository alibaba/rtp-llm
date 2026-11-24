"""Own one host's local engine rank group; importing it has no engine effects."""

from __future__ import annotations

import logging
import os
import pathlib
import re
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from rtp_llm.test.perf_test.execution.launch_boundary import (
    RUNTIME_ENVIRONMENT_KEYS,
    configure_perf_environment,
    server_environment_exclusions,
)
from rtp_llm.test.perf_test.plan import ExecutionPlan

if TYPE_CHECKING:
    from rtp_llm.distribute.worker_info import WorkerInfo
    from rtp_llm.test.utils.maga_server_manager import MagaServerManager


@dataclass
class LocalRankGroupContext:
    """Per-host context needed to start one local rank group."""

    port: int
    world_rank: int
    log_dir_path: str
    profiler_directory: pathlib.Path
    plan: ExecutionPlan


@dataclass
class ServerHandle:
    """Owning handle for one host's local rank-group server.

    Wraps the MagaServerManager together with the startup byproducts the
    caller needs: the locally resolved tokenizer path (rank zero only),
    whether this process became its own process group leader during
    startup, and the member's log dir (where the engine drops raw chrome
    traces through TORCH_CUDA_PROFILER_DIR). The caller launches and stops
    the server through the handle; the caller owns cancellation and cleanup.
    """

    manager: MagaServerManager
    tokenizer_path: Optional[str]
    pgrp_set: bool
    log_dir: str

    @property
    def log_file_path(self) -> Optional[str]:
        return self.manager.log_file_path

    def start_server(self) -> bool:
        return self.manager.start_server()

    def print_process_log(self) -> None:
        self.manager.print_process_log()

    def stop_server(self) -> None:
        self.manager.stop_server()


def _host_engine_environment(environment: Dict[str, str]) -> Dict[str, str]:
    """Keep launcher-owned host assignments out of custom engine overrides."""
    runtime_keys = RUNTIME_ENVIRONMENT_KEYS
    return {
        **{key: value for key, value in environment.items() if key not in runtime_keys},
        **{key: os.environ[key] for key in runtime_keys if key in os.environ},
    }


def start_local_rank_group(
    context: LocalRankGroupContext,
    config: Any,
    model_paths: Optional[str],
    *,
    engine_argv: Optional[Sequence[str]] = None,
    engine_environment: Optional[Dict[str, str]] = None,
) -> ServerHandle:
    """Start this host's local rank group.

    Performs per-host startup: model setup
    and download, engine environment, output directories, process-group
    isolation, rank-zero tokenizer resolution, then construction of the
    maga server manager that owns every local rank. ``config`` is the
    parsed ``PyEnvConfigs`` and ``model_paths`` is the tokenizer location
    used as a fallback when model setup leaves ``model_args.tokenizer_path``
    empty. The tokenizer otherwise uses the same resolved path as the engine
    (gang workers pass their locally resolved path). The server
    subprocess itself is launched by the caller through the returned
    handle so the caller retains failure and process lifetime ownership.

    ``engine_argv``/``engine_environment`` are the unified launch-spec
    projection of the resolved config (gang workers always supply them):
    without ``engine_argv`` the server subprocess starts with no engine
    arguments, so per-run switches such as ``reuse_cache`` would be lost.
    Both SSH startup and gang workers supply this projection.
    """
    from rtp_llm.config.server_config_setup import setup_and_configure_server
    from rtp_llm.test.utils.maga_server_manager import MagaServerManager
    from rtp_llm.utils.fuser import fetch_remote_file_to_local

    if engine_environment is not None:
        engine_environment = _host_engine_environment(engine_environment)
    configure_perf_environment(engine_environment)
    setup_and_configure_server(config)
    engine = context.plan.resolved_config["engine"]
    os.environ["MAX_SEQ_LEN"] = str(engine["max_seq_len"])
    os.environ["MAX_BATCH_TOKENS_SIZE"] = str(engine["max_batch_tokens"])
    os.environ["CONCURRENCY_LIMIT"] = str(engine["concurrency_limit"])
    os.makedirs(context.log_dir_path, exist_ok=True)
    os.environ["TORCH_CUDA_PROFILER_DIR"] = str(context.profiler_directory)
    # for maga server files
    os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = context.log_dir_path
    os.environ["MAGA_SERVER_WORK_DIR"] = os.getcwd()

    pgrp_set = False
    try:
        os.setpgrp()
        pgrp_set = True
    except Exception as e:
        logging.info(f"setpgrp error: {e}")

    tokenizer_path = None
    if context.world_rank == 0:
        # Use the same resolved tokenizer as the engine after model setup.
        model_paths = config.model_args.tokenizer_path or model_paths
        if not model_paths:
            raise RuntimeError(
                "tokenizer_path is missing; set model_args.tokenizer_path or ckpt_path"
            )
        tokenizer_path = fetch_remote_file_to_local(
            os.path.expanduser(str(model_paths).strip())
        )

    manager_kwargs: Dict[str, Any] = {
        "port": str(context.port),
        "inherited_env_exclude": server_environment_exclusions(),
    }
    if engine_argv is not None:
        # Configuration downloads model files on this host. Explicit argv
        # must use those paths rather than overriding them with remote IDs.
        model_options = {
            "checkpoint_path": config.model_args.ckpt_path,
            "tokenizer_path": config.model_args.tokenizer_path,
            "model_type": config.model_args.model_type,
        }
        local_argv = [
            arg
            for arg in engine_argv
            if arg.split("=", 1)[0] not in {"--" + key for key in model_options}
        ]
        local_argv.extend(
            f"--{key}={value}" for key, value in model_options.items() if value
        )
        manager_kwargs["smoke_args_str"] = shlex.join(local_argv)
    if engine_environment:
        # Output paths are assigned after the initial projection; refresh them
        # so manager overrides cannot restore stale or custom directories.
        manager_kwargs["env_args"] = _host_engine_environment(engine_environment)
    server = MagaServerManager(**manager_kwargs)
    return ServerHandle(
        manager=server,
        tokenizer_path=tokenizer_path,
        pgrp_set=pgrp_set,
        log_dir=str(context.log_dir_path),
    )


def _shard_index_key(member: "WorkerInfo"):
    """Sort key: the shard index encoded in the legacy ``...partN`` name."""
    match = re.search(r"part(\d+)$", member.name)
    if match:
        return (0, int(match.group(1)), member.name)
    return (1, 0, member.name)


def build_gang_config_string(
    members: Sequence["WorkerInfo"], local_world_size: int
) -> str:
    """Build the ``GANG_CONFIG_STRING`` value for one multi-host gang.

    ``members`` are the legacy member records the server side already
    parses (the ``WorkerInfo`` records ``members_from_test_env`` returns).
    The master (the member whose name ends with ``part0``) is placed first
    and the remaining members follow their shard index, matching the
    encoding the legacy SSH launcher emits::

        name:test_part0,ip:host0,port:12333;name:test_part1,ip:host1,port:12333

    ``local_world_size`` is the per-host rank-group width; the caller owns
    the rank layout (world rank = shard index * local world size).
    """
    member_list = list(members)
    if not member_list:
        raise RuntimeError("no gang members, unexpected!")
    if not isinstance(local_world_size, int) or local_world_size < 1:
        raise RuntimeError(
            f"local_world_size must be a positive integer, got {local_world_size!r}"
        )
    masters = [member for member in member_list if member.name.endswith("part0")]
    if len(masters) != 1:
        raise RuntimeError(f"gang master should contains 1 but got {len(masters)}")
    ordered = sorted(member_list, key=_shard_index_key)
    if masters[0].name != ordered[0].name:
        raise RuntimeError(
            f"gang master should be the first one but got {ordered[0].name}"
        )
    return ";".join(
        f"name:{member.name},ip:{member.ip},port:{member.server_port}"
        for member in ordered
    )
