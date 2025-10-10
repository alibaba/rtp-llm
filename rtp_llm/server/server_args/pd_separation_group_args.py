from rtp_llm.server.server_args.util import str2bool


def init_pd_separation_group_args(parser):
    ##############################################################################################################
    # PD分离的配置
    ##############################################################################################################
    pd_separation_group = parser.add_argument_group("pd_separation")
    pd_separation_group.add_argument(
        "--prefill_retry_times",
        env_name="PREFILL_RETRY_TIMES",
        type=int,
        default=0,
        help="prefill部分流程的重试次数，0 表示禁用重试",
    )

    pd_separation_group.add_argument(
        "--prefill_retry_timeout_ms",
        env_name="PREFILL_RETRY_TIMEOUT_MS",
        type=int,
        default=0,
        help="prefill重试的总超时时间（毫秒）",
    )

    pd_separation_group.add_argument(
        "--prefill_max_wait_timeout_ms",
        env_name="PREFILL_MAX_WAIT_TIMEOUT_MS",
        type=int,
        default=600 * 1000,
        help="prefill的最大等待运行超时时间（毫秒）",
    )

    pd_separation_group.add_argument(
        "--load_cache_timeout_ms",
        env_name="LOAD_CACHE_TIMEOUT_MS",
        type=int,
        default=5000,
        help="KVCache远端加载超时时间（毫秒）",
    )

    pd_separation_group.add_argument(
        "--decode_retry_times",
        env_name="DECODE_RETRY_TIMES",
        type=int,
        default=100,
        help="Decode部分流程重试次数，0 表示禁用重试",
    )

    pd_separation_group.add_argument(
        "--decode_retry_timeout_ms",
        env_name="DECODE_RETRY_TIMEOUT_MS",
        type=int,
        default=100,
        help="Decode流程重试的总超时时间（毫秒）",
    )

    pd_separation_group.add_argument(
        "--rdma_connect_retry_times",
        env_name="RDMA_CONNECT_RETRY_TIMES",
        type=int,
        default=0,
        help="RDMA 连接建立的重试次数",
    )

    pd_separation_group.add_argument(
        "--decode_polling_kv_cache_step_ms",
        env_name="DECODE_POLLING_KV_CACHE_STEP_MS",
        type=int,
        default=30,
        help="轮询 KV 加载状态的间隔时间（毫秒）",
    )

    pd_separation_group.add_argument(
        "--decode_entrance",
        env_name="DECODE_ENTRANCE",
        type=str2bool,
        default=False,
        help="Decode 是否作为流量的入口点",
    )
