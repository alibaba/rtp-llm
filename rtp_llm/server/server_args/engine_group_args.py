import argparse

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
