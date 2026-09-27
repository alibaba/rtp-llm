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


def init_engine_group_args(parser, runtime_config):
    # These fields are a required Python/C++ contract, not optional old bindings.
    # The general argument binder only logs assignment failures.
    for name in ("enable_sleep_mode", "sleep_mode_level"):
        try:
            getattr(runtime_config, name)
        except AttributeError as e:
            raise RuntimeError(
                f"RuntimeConfig binding is missing {name}; rebuild the matching C++ bindings"
            ) from e

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
        "--enable_sleep_mode",
        "--enable-sleep-mode",
        env_name=ENABLE_ENV,
        type=str2bool,
        default=None,
        help="兼容旧配置；新部署只需 sleep_mode_level。单独设为 1 等价于 level 1；"
        "与显式 level 矛盾时启动报错",
    )
    engine_group.add_argument(
        "--sleep_mode_level",
        "--sleep-mode-level",
        env_name=LEVEL_ENV,
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="统一 sleep 开关和权重策略：0=关闭（默认），1=权重备份到 pinned host，"
        "2=丢弃权重并在 wake 从原始 checkpoint 原地重载。level 在加载前固定；"
        "/sleep 请求仍只接受与启动配置一致的 1/2，0 不是一种 sleep 操作",
    )
    engine_group.add_argument(
        "--sleep_free_runtime_caches",
        "--sleep-free-runtime-caches",
        env_name=RUNTIME_CACHES_ENV,
        type=str2bool,
        default=None,
        help="sleep 时释放安全可重建的 Python runtime caches；未设置时跟随 sleep 开启，"
        "显式 0 可关闭。CUDA graph 捕获的指针和必须保留的通信资源不受此开关强制释放",
    )
    engine_group.add_argument(
        "--sleep_release_collective_memory",
        "--sleep-release-collective-memory",
        env_name=COLLECTIVE_MEMORY_ENV,
        type=str2bool,
        default=None,
        help="sleep 时通过 ncclCommSuspend/Resume 释放通信显存；未设置时跟随 sleep 开启，"
        "显式 0 可关闭。需要兼容的 NCCL API/通信器及所有 rank 同意；不支持时记录原因并跳过。"
        "释放的显存会等量占用 pinned host 内存，并增加 sleep/wake 耗时",
    )


def configure_sleep_args(args, runtime_config):
    """Resolve once, then publish identical settings to C++ and child Python."""
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
    # Keep the native/pickle contract; the boolean is derived, not another input.
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
