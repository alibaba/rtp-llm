from rtp_llm.server.server_args.util import str2bool


def init_pd_separation_group_args(parser, pd_separation_config):
    ##############################################################################################################
    # PD分离的配置
    ##############################################################################################################
    pd_separation_group = parser.add_argument_group("pd_separation")
    pd_separation_group.add_argument(
        "--prefill_retry_times",
        env_name="PREFILL_RETRY_TIMES",
        bind_to=(pd_separation_config, "prefill_retry_times"),
        type=int,
        default=0,
        help="prefill部分流程的重试次数，0 表示禁用重试",
    )

    pd_separation_group.add_argument(
        "--prefill_retry_timeout_ms",
        env_name="PREFILL_RETRY_TIMEOUT_MS",
        bind_to=(pd_separation_config, "prefill_retry_timeout_ms"),
        type=int,
        default=0,
        help="prefill重试的总超时时间（毫秒）",
    )

    pd_separation_group.add_argument(
        "--prefill_max_wait_timeout_ms",
        env_name="PREFILL_MAX_WAIT_TIMEOUT_MS",
        bind_to=(pd_separation_config, "prefill_max_wait_timeout_ms"),
        type=int,
        default=600 * 1000,
        help="prefill的最大等待运行超时时间（毫秒）",
    )

    pd_separation_group.add_argument(
        "--load_cache_timeout_ms",
        env_name="LOAD_CACHE_TIMEOUT_MS",
        bind_to=(pd_separation_config, "load_cache_timeout_ms"),
        type=int,
        default=5000,
        help="KVCache远端加载超时时间（毫秒）",
    )

    pd_separation_group.add_argument(
        "--max_rpc_timeout_ms",
        env_name="MAX_RPC_TIMEOUT_MS",
        bind_to=(pd_separation_config, "max_rpc_timeout_ms"),
        type=int,
        default=2 * 3600 * 1000,  # 2h
        help="RPC 调用最大超时（毫秒），用作 per-request 未传 timeout 时的默认值；"
        "<=0 表示不设 deadline（链路不超时）；"
        "per-request generate_config.timeout_ms 优先级更高",
    )

    pd_separation_group.add_argument(
        "--batch_dispatch_timeout_ms",
        env_name="BATCH_DISPATCH_TIMEOUT_MS",
        bind_to=(pd_separation_config, "batch_dispatch_timeout_ms"),
        type=int,
        default=60000,
        help="EnqueueBatch 跨 DP 分发超时（毫秒），防止远端 DP 卡死阻塞整个 batch",
    )

    pd_separation_group.add_argument(
        "--batch_prepare_timeout_ms",
        env_name="BATCH_PREPARE_TIMEOUT_MS",
        bind_to=(pd_separation_config, "batch_prepare_timeout_ms"),
        type=int,
        default=10000,
        help="EnqueueGroup 内部 prepareAllocateResource 超时（毫秒）",
    )

    pd_separation_group.add_argument(
        "--batch_load_timeout_ms",
        env_name="BATCH_LOAD_TIMEOUT_MS",
        bind_to=(pd_separation_config, "batch_load_timeout_ms"),
        type=int,
        default=10000,
        help="EnqueueGroup 内部 remoteLoadCacheStart 超时（毫秒）",
    )

    pd_separation_group.add_argument(
        "--prefill_enqueue_pool_size",
        env_name="PREFILL_ENQUEUE_POOL_SIZE",
        bind_to=(pd_separation_config, "prefill_enqueue_pool_size"),
        type=int,
        default=0,
        help="Prefill L1 enqueue 线程池大小，0 表示使用公式默认值",
    )

    pd_separation_group.add_argument(
        "--prefill_worker_lambda_pool_size",
        env_name="PREFILL_WORKER_LAMBDA_POOL_SIZE",
        bind_to=(pd_separation_config, "prefill_worker_lambda_pool_size"),
        type=int,
        default=0,
        help="Prefill worker lambda 线程池大小，0 表示使用公式默认值",
    )

    pd_separation_group.add_argument(
        "--prefill_slot_pool_size",
        env_name="PREFILL_SLOT_POOL_SIZE",
        bind_to=(pd_separation_config, "prefill_slot_pool_size"),
        type=int,
        default=0,
        help="Prefill slot 线程池大小，0 表示使用公式默认值",
    )

    pd_separation_group.add_argument(
        "--prefill_stop_stream_wait_timeout_ms",
        env_name="PREFILL_STOP_STREAM_WAIT_TIMEOUT_MS",
        bind_to=(pd_separation_config, "prefill_stop_stream_wait_timeout_ms"),
        type=int,
        default=2000,
        help="stopStream() 中等待 Engine Loop 调用 finish_internal() 的最大时间（毫秒）",
    )

    pd_separation_group.add_argument(
        "--decode_retry_times",
        env_name="DECODE_RETRY_TIMES",
        bind_to=(pd_separation_config, "decode_retry_times"),
        type=int,
        default=100,
        help="Decode部分流程重试次数，0 表示禁用重试",
    )

    pd_separation_group.add_argument(
        "--decode_retry_timeout_ms",
        env_name="DECODE_RETRY_TIMEOUT_MS",
        bind_to=(pd_separation_config, "decode_retry_timeout_ms"),
        type=int,
        default=100,
        help="Decode流程重试的总超时时间（毫秒）",
    )

    pd_separation_group.add_argument(
        "--decode_retry_interval_ms",
        env_name="DECODE_RETRY_INTERVAL_MS",
        bind_to=(pd_separation_config, "decode_retry_interval_ms"),
        type=int,
        default=1,
        help="Decode流程重试的区间间隔（毫秒）",
    )

    pd_separation_group.add_argument(
        "--rdma_connect_retry_times",
        env_name="RDMA_CONNECT_RETRY_TIMES",
        bind_to=(pd_separation_config, "rdma_connect_retry_times"),
        type=int,
        default=0,
        help="RDMA 连接建立的重试次数",
    )

    pd_separation_group.add_argument(
        "--decode_polling_kv_cache_step_ms",
        env_name="DECODE_POLLING_KV_CACHE_STEP_MS",
        bind_to=(pd_separation_config, "decode_polling_kv_cache_step_ms"),
        type=int,
        default=30,
        help="轮询 KV 加载状态的间隔时间（毫秒）",
    )

    pd_separation_group.add_argument(
        "--decode_entrance",
        env_name="DECODE_ENTRANCE",
        bind_to=(pd_separation_config, "decode_entrance"),
        type=str2bool,
        default=False,
        help="Decode 是否作为流量的入口点",
    )
