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
        "--enable_fast_gen",
        env_name="ENABLE_FAST_GEN",
        bind_to=[(fifo_scheduler_config, 'enable_fast_gen')],
        type=str2bool,
        default=False,
        help="若为 True，长请求会被拆分为chunks并分步处理。这主要用于提高长序列或流式输入的处理效率，并能改善并发场景下其他请求的交互性。注意：仅在使用默认调度器时有效。",
    )

    fifo_scheduler_group.add_argument(
        "--fast_gen_context_budget",
        env_name="FAST_GEN_MAX_CONTEXT_LEN",  # 和参数名不一致
        bind_to=[(fifo_scheduler_config, 'fast_gen_context_budget')],
        type=int,
        help="当 ENABLE_FAST_GEN 启用时，拆分成的chunk的大小。注意：仅当 ENABLE_FAST_GEN 为 True 且使用默认调度器时有效。",
    )
    fifo_scheduler_group.add_argument(
        "--enable_partial_fallback",
        env_name="ENABLE_PARTIAL_FALLBACK",
        bind_to=[(fifo_scheduler_config, 'enable_partial_fallback')],
        type=str2bool,
        default=False,
        help="若为 True，则允许默认调度器在系统内存不足以满足活动请求时，从某些请求中回收一部分 KV cache blocks。这可以在高负载下提高系统利用率，但可能会影响那些资源被回收的请求的公平性。注意：在使用默认调度器时有效。",
    )
    fifo_scheduler_group.add_argument(
        "--max_batch_tokens_size",
        env_name="MAX_BATCH_TOKENS_SIZE",
        bind_to=[(fifo_scheduler_config, 'max_batch_tokens_size')],
        type=int,
        default=0,
        help="最大 batch tokens 大小。",
    )
