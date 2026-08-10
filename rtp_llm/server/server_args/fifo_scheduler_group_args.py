import argparse

from rtp_llm.server.server_args.util import str2bool

MAX_CONTEXT_BATCH_COALESCING_WINDOW_MS = 60_000


def parse_context_batch_coalescing_window_ms(value):
    try:
        window_ms = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "context batch coalescing window must be an integer"
        ) from error
    if not 0 <= window_ms <= MAX_CONTEXT_BATCH_COALESCING_WINDOW_MS:
        raise argparse.ArgumentTypeError(
            "context batch coalescing window must be in "
            f"[0, {MAX_CONTEXT_BATCH_COALESCING_WINDOW_MS}]"
        )
    return window_ms


def init_fifo_scheduler_group_args(parser, fifo_scheduler_config):
    ##############################################################################################################
    # FIFO 调度器配置
    ##############################################################################################################
    fifo_scheduler_group = parser.add_argument_group("FIFO Scheduler")

    fifo_scheduler_group.add_argument(
        "--max_context_batch_size",
        env_name="MAX_CONTEXT_BATCH_SIZE",
        bind_to=[(fifo_scheduler_config, "max_context_batch_size")],
        type=int,
        default=1,
        help=(
            "最大 context batch size，影响默认调度器的凑批决策。"
            "PDFUSION 且 coalescing window 大于 0 时，调度聚合将它作为软上限；"
            "为保证进展，首个不可拆请求可超过该值。"
        ),
    )
    fifo_scheduler_group.add_argument(
        "--max_batch_tokens_size",
        env_name="MAX_BATCH_TOKENS_SIZE",
        bind_to=[(fifo_scheduler_config, "max_batch_tokens_size")],
        type=int,
        default=0,
        help="最大 batch tokens 大小。",
    )
    fifo_scheduler_group.add_argument(
        "--context_batch_coalescing_window_ms",
        env_name="CONTEXT_BATCH_COALESCING_WINDOW_MS",
        bind_to=[(fifo_scheduler_config, "context_batch_coalescing_window_ms")],
        type=parse_context_batch_coalescing_window_ms,
        default=0,
        help=(
            "PDFUSION 空闲时用于新到 context 与异步 cache load 完成 cohort 的两个独立固定凑批窗口，"
            "单位毫秒；两个截止时间均不续期。仅在 max_context_batch_size 大于 1 时启用等待和行数软上限；"
            "默认 0 不启用这些优化，纯相位调度与 cache 完成后的重新准入始终生效。"
        ),
    )
