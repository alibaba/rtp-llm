import argparse
import logging
import os

from rtp_llm.config.sleep_mode import (
    COLLECTIVE_MEMORY_ENV,
    ENABLE_ENV,
    LEGACY_RUNTIME_CACHES_ENV,
    LEVEL_ENV,
    RUNTIME_CACHES_ENV,
    resolve_sleep_level,
    resource_release_enabled,
)
from rtp_llm.server.server_args.util import str2bool


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be a non-negative integer"
        ) from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def init_engine_group_args(parser, runtime_config):
    for name in ("enable_sleep_mode", "sleep_mode_level"):
        if not hasattr(runtime_config, name):
            raise RuntimeError(
                f"RuntimeConfig binding is missing {name}; rebuild matching C++ bindings"
            )
    ##############################################################################################################
    # Engine Configuration
    # Fields merged from EngineConfig to RuntimeConfig (warm_up, warm_up_with_loss).
    ##############################################################################################################
    engine_group = parser.add_argument_group("Engine Configuration")
    engine_group.add_argument(
        "--warm_up",
        env_name="WARM_UP",
        bind_to=(runtime_config, "warm_up"),
        type=str2bool,
        default=True,
        help="在服务启动时是否开启预热",
    )
    engine_group.add_argument(
        "--warm_up_with_loss",
        env_name="WARM_UP_WITH_LOSS",
        bind_to=(runtime_config, "warm_up_with_loss"),
        type=str2bool,
        default=False,
        help="在服务启动时是否使用 loss 路径预热",
    )
    engine_group.add_argument(
        "--model_warm_up",
        env_name="MODEL_WARM_UP",
        bind_to=(runtime_config, "model_warm_up"),
        type=str2bool,
        default=True,
        help="在服务启动时是否开启模型侧预热",
    )

    engine_group.add_argument(
        "--output_dispatcher_worker_count",
        env_name="OUTPUT_DISPATCHER_WORKER_COUNT",
        bind_to=(runtime_config, "output_dispatcher_worker_count"),
        type=_non_negative_int,
        metavar="INT",
        default=0,
        help="输出分发器并行处理的 worker 数量，必须为非负整数，0 表示串行分发",
    )
    engine_group.add_argument(
        "--enable_sleep_mode",
        "--enable-sleep-mode",
        env_name=ENABLE_ENV,
        type=str2bool,
        default=None,
        help="Legacy sleep switch; use sleep_mode_level. Conflicting values are rejected.",
    )
    engine_group.add_argument(
        "--sleep_mode_level",
        "--sleep-mode-level",
        env_name=LEVEL_ENV,
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="0 disables sleep; 1 backs weights up to host; 2 reloads weights on wake.",
    )
    engine_group.add_argument(
        "--sleep_free_runtime_caches",
        "--sleep-free-runtime-caches",
        env_name=RUNTIME_CACHES_ENV,
        type=str2bool,
        default=None,
        help="Reclaim safely rebuildable runtime caches. Defaults to sleep activation; "
        "explicit 0 disables reclaim. Captured pointers remain protected.",
    )
    engine_group.add_argument(
        "--sleep_release_collective_memory",
        "--sleep-release-collective-memory",
        env_name=COLLECTIVE_MEMORY_ENV,
        type=str2bool,
        default=None,
        help="Suspend supported NCCL memory on sleep. Defaults to sleep activation; "
        "explicit 0 disables it. Offloaded GPU memory uses pinned host memory.",
    )


def configure_sleep_args(args, runtime_config):
    """Resolve once before native configuration serialization and worker spawn."""
    level = resolve_sleep_level(args.sleep_mode_level, args.enable_sleep_mode)
    enabled = level > 0
    runtime_caches = args.sleep_free_runtime_caches
    if runtime_caches is None:
        runtime_caches = resource_release_enabled(
            RUNTIME_CACHES_ENV, default=enabled, legacy_alias=LEGACY_RUNTIME_CACHES_ENV
        )
    collective = args.sleep_release_collective_memory
    if collective is None:
        collective = enabled
    runtime_config.enable_sleep_mode = enabled
    runtime_config.sleep_mode_level = level
    os.environ.update(
        {
            LEVEL_ENV: str(level),
            ENABLE_ENV: str(int(enabled)),
            RUNTIME_CACHES_ENV: str(int(runtime_caches)),
            COLLECTIVE_MEMORY_ENV: str(int(collective)),
        }
    )
    if args.enable_sleep_mode is not None:
        logging.warning(
            "ENABLE_SLEEP_MODE is deprecated; use SLEEP_MODE_LEVEL=%d", level
        )
    logging.info(
        "Sleep configuration: level=%d enabled=%s runtime_caches=%s collective_memory=%s "
        "(resource capability/safety checks still apply)",
        level,
        enabled,
        runtime_caches,
        collective,
    )
