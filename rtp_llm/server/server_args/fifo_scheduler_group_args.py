from rtp_llm.server.server_args.util import str2bool


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
        help="（设备参数）为设备参数设置的最大 context batch size，影响默认调度器的凑批决策。",
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
        "--cp_force_single_prefill",
        env_name="CP_FORCE_SINGLE_PREFILL",
        bind_to=[(fifo_scheduler_config, "cp_force_single_prefill")],
        type=str2bool,
        default=True,
        help="CP prefill 开启时是否强制每轮只调度一个 prefill 请求。",
    )
    fifo_scheduler_group.add_argument(
        "--max_inited_kv_cache_streams",
        env_name="MAX_INITED_KV_CACHE_STREAMS",
        bind_to=[(fifo_scheduler_config, "max_inited_kv_cache_streams")],
        type=int,
        default=0,
        help="FIFO 中已初始化 KV cache block 的 stream 数上限，0 表示不限制。",
    )
