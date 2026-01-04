from rtp_llm.server.server_args.util import str2bool


def init_fifo_scheduler_group_args(parser, fifo_scheduler_config):
    ##############################################################################################################
    # FIFO 调度器配置
    ##############################################################################################################
    fifo_scheduler_group = parser.add_argument_group("FIFO Scheduler")
    
    fifo_scheduler_group.add_argument(
        "--max_context_batch_size",
        env_name="MAX_CONTEXT_BATCH_SIZE",
        bind_to=[(fifo_scheduler_config, 'max_context_batch_size')],
        type=int,
        default=1,
        help="（设备参数）为设备参数设置的最大 context batch size，影响默认调度器的凑批决策。",
    )
    fifo_scheduler_group.add_argument(
        "--scheduler_reserve_resource_ratio",
        env_name="SCHEDULER_RESERVE_RESOURCE_RATIO",
        bind_to=[(fifo_scheduler_config, 'scheduler_reserve_resource_ratio')],
        type=int,
        default=5,
        help="默认调度器将尝试保留的 KV cache blocks 的最小百分比。这有助于应对突发请求模式，为高优先级请求预留空间，或防止系统性能颠簸。",
    )
    fifo_scheduler_group.add_argument(
        "--max_batch_tokens_size",
        env_name="MAX_BATCH_TOKENS_SIZE",
        bind_to=[(fifo_scheduler_config, 'max_batch_tokens_size')],
        type=int,
        default=0,
        help="最大 batch tokens 大小。",
    )
