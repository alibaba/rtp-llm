from rtp_llm.server.server_args.util import str2bool


def init_cache_store_group_args(parser, cache_store_config):
    ##############################################################################################################
    # Cache Store 配置
    ##############################################################################################################
    cache_store_group = parser.add_argument_group("Cache Store")
    cache_store_group.add_argument(
        "--cache_store_rdma_mode",
        env_name="CACHE_STORE_RDMA_MODE",
        bind_to=(cache_store_config, "cache_store_rdma_mode"),
        type=str2bool,
        default=False,
        help="控制 cache store 是否使用 RDMA 模式。",
    )

    cache_store_group.add_argument(
        "--cache_store_thread_count",
        env_name="CACHE_STORE_THREAD_COUNT",
        bind_to=(cache_store_config, "thread_count"),
        type=int,
        default=32,
        help="为 cache store 线程池设置线程数量。",
    )

    cache_store_group.add_argument(
        "--cache_store_rdma_connect_timeout_ms",
        env_name="CACHE_STORE_RDMA_CONNECT_TIMEOUT_MS",
        bind_to=(cache_store_config, "rdma_connect_timeout_ms"),
        type=int,
        default=250,
        help="为 cache store RDMA 连接设置超时时间，单位为毫秒。",
    )

    cache_store_group.add_argument(
        "--cache_store_rdma_qp_count_per_connection",
        env_name="CACHE_STORE_RDMA_QP_COUNT_PER_CONNECTION",
        bind_to=(cache_store_config, "rdma_qp_count_per_connection"),
        type=int,
        default=2,
        help="为 cache store RDMA 连接设置每个连接的底层QP数量。",
    )

    cache_store_group.add_argument(
        "--cache_store_rdma_io_thread_count",
        env_name="CACHE_STORE_RDMA_IO_THREAD_COUNT",
        bind_to=(cache_store_config, "rdma_io_thread_count"),
        type=int,
        default=4,
        help="为 cache store RDMA 通信层设置 IO 线程数量。",
    )

    cache_store_group.add_argument(
        "--cache_store_rdma_worker_thread_count",
        env_name="CACHE_STORE_RDMA_WORKER_THREAD_COUNT",
        bind_to=(cache_store_config, "rdma_worker_thread_count"),
        type=int,
        default=2,
        help="为 cache store RDMA 通信层设置 worker 线程数量。",
    )

    cache_store_group.add_argument(
        "--cache_store_rdma_max_block_pairs_per_connection",
        env_name="CACHE_STORE_RDMA_MAX_BLOCK_PAIRS_PER_CONNECTION",
        bind_to=(cache_store_config, "rdma_max_block_pairs_per_connection"),
        type=int,
        default=0,
        help="限制单个 RDMA 连接读操作中包含的 block_pair 数量，0 表示不拆分。",
    )

    cache_store_group.add_argument(
        "--cache_store_messager_io_thread_count",
        env_name="CACHE_STORE_MESSAGER_IO_THREAD_COUNT",
        bind_to=(cache_store_config, "messager_io_thread_count"),
        type=int,
        default=2,
        help="为 cache store P2P messager 通信层设置 IO 线程数量。",
    )
    cache_store_group.add_argument(
        "--cache_store_messager_worker_thread_count",
        env_name="CACHE_STORE_MESSAGER_WORKER_THREAD_COUNT",
        bind_to=(cache_store_config, "messager_worker_thread_count"),
        type=int,
        default=32,
        help="为 cache store P2P messager 通信层设置 worker 线程数量。",
    )

    cache_store_group.add_argument(
        "--p2p_transfer_not_done_resource_hold_ms",
        env_name="P2P_TRANSFER_NOT_DONE_RESOURCE_HOLD_MS",
        bind_to=(cache_store_config, "p2p_transfer_not_done_resource_hold_ms"),
        type=int,
        default=10000,
        help=(
            "RDMA：传输超时后等待该时长再关闭相关连接（毫秒）。"
            "已派发的读操作若缺少完成回调，Decode 会将目标 KV 保留至收到停写确认或隔离满 5 分钟。"
        ),
    )
    cache_store_group.add_argument(
        "--p2p_lease_query_timeout_ms",
        env_name="P2P_LEASE_QUERY_TIMEOUT_MS",
        bind_to=(cache_store_config, "p2p_lease_query_timeout_ms"),
        type=int,
        default=20000,
        help="Decode：单轮 lease 查询的超时上限（毫秒），内部最多等待 500ms；未确认停止则继续持有 KV 并重试。",
    )
    cache_store_group.add_argument(
        "--p2p_kv_force_release_timeout_ms",
        env_name="P2P_KV_FORCE_RELEASE_TIMEOUT_MS",
        bind_to=(cache_store_config, "p2p_kv_force_release_timeout_ms"),
        type=int,
        default=300000,
        help="已废弃，仅用于兼容旧启动参数。KV 需要确认物理传输结束才会释放。",
    )

    cache_store_group.add_argument(
        "--p2p_resource_store_timeout_check_interval_ms",
        env_name="P2P_RESOURCE_STORE_TIMEOUT_CHECK_INTERVAL_MS",
        bind_to=(cache_store_config, "p2p_resource_store_timeout_check_interval_ms"),
        type=int,
        default=100,
        help="兼容旧启动配置。当前资源过期由请求截止时间驱动。",
    )
    cache_store_group.add_argument(
        "--p2p_layer_cache_buffer_store_timeout_ms",
        env_name="P2P_LAYER_CACHE_BUFFER_STORE_TIMEOUT_MS",
        bind_to=(cache_store_config, "p2p_layer_cache_buffer_store_timeout_ms"),
        type=int,
        default=100000,
        help="保留用于兼容旧配置。当前 P2P 缓冲区使用请求截止时间。",
    )
    cache_store_group.add_argument(
        "--p2p_cancel_broadcast_timeout_ms",
        env_name="P2P_CANCEL_BROADCAST_TIMEOUT_MS",
        bind_to=(cache_store_config, "p2p_cancel_broadcast_timeout_ms"),
        type=int,
        default=1000,
        help="P2P Scheduler 广播 CANCEL 时的 gRPC 超时（毫秒）。",
    )
    cache_store_group.add_argument(
        "--p2p_cancelled_keys_ttl_ms",
        env_name="P2P_CANCELLED_KEYS_TTL_MS",
        bind_to=(cache_store_config, "p2p_cancelled_keys_ttl_ms"),
        type=int,
        default=3600000,
        help="P2P cancelled key tombstone 的 TTL（毫秒），用于拒绝迟到的 prefill 资源。",
    )
    cache_store_group.add_argument(
        "--cache_store_tcp_anet_rpc_thread_num",
        env_name="CACHE_STORE_TCP_ANET_RPC_THREAD_NUM",
        bind_to=(cache_store_config, "cache_store_tcp_anet_rpc_thread_num"),
        type=int,
        default=3,
        help="P2P TCP 控制面 ANetRPCServer threadNum（与 ArpcServerWrapper.threadNum 语义一致）。",
    )
    cache_store_group.add_argument(
        "--cache_store_tcp_anet_rpc_queue_num",
        env_name="CACHE_STORE_TCP_ANET_RPC_QUEUE_NUM",
        bind_to=(cache_store_config, "cache_store_tcp_anet_rpc_queue_num"),
        type=int,
        default=100,
        help="P2P TCP 控制面 ANetRPCServer queueNum。",
    )
    cache_store_group.add_argument(
        "--cache_store_tcp_worker_queue_size",
        env_name="CACHE_STORE_TCP_WORKER_QUEUE_SIZE",
        bind_to=(cache_store_config, "cache_store_tcp_worker_queue_size"),
        type=int,
        default=500,
        help="P2P TCP/RDMA 控制面 worker 线程池（tcp_server_rpc_threadpool）队列深度，默认 500。",
    )
    cache_store_group.add_argument(
        "--rdma_transfer_worker_thread_count",
        env_name="RDMA_TRANSFER_WORKER_THREAD_COUNT",
        bind_to=(cache_store_config, "rdma_transfer_worker_thread_count"),
        type=int,
        default=16,
        help="RDMA 传输 worker 线程池（RdmaTransferServiceWorker）线程数，默认 16。",
    )
    cache_store_group.add_argument(
        "--rdma_transfer_worker_queue_size",
        env_name="RDMA_TRANSFER_WORKER_QUEUE_SIZE",
        bind_to=(cache_store_config, "rdma_transfer_worker_queue_size"),
        type=int,
        default=100,
        help="RDMA 传输 worker 线程池（RdmaTransferServiceWorker）队列深度，默认 100。",
    )
    cache_store_group.add_argument(
        "--p2p_rdma_enable_h2d_copy",
        env_name="P2P_RDMA_ENABLE_H2D_COPY",
        bind_to=(cache_store_config, "p2p_rdma_enable_h2d_copy"),
        type=str2bool,
        default=False,
        help="P2P RDMA 是否先读入 pinned host staging buffer，再执行 H2D copy。",
    )
    cache_store_group.add_argument(
        "--p2p_rdma_staging_total_bytes",
        env_name="P2P_RDMA_STAGING_TOTAL_BYTES",
        bind_to=(cache_store_config, "p2p_rdma_staging_total_bytes"),
        type=int,
        default=0,
        help="P2P RDMA H2D pinned-host staging 总字节数；开启 H2D copy 时必须至少容纳一个 LLM KV block。",
    )
    cache_store_group.add_argument(
        "--p2p_prefill_sender_thread_count",
        env_name="P2P_PREFILL_SENDER_THREAD_COUNT",
        bind_to=(cache_store_config, "p2p_prefill_sender_thread_count"),
        type=int,
        default=4,
        help="P2P Prefill 异步发送线程池的线程数，必须大于 0。",
    )
    cache_store_group.add_argument(
        "--p2p_prefill_sender_queue_size",
        env_name="P2P_PREFILL_SENDER_QUEUE_SIZE",
        bind_to=(cache_store_config, "p2p_prefill_sender_queue_size"),
        type=int,
        default=10000,
        help="P2P Prefill 异步发送线程池的队列容量，必须大于 0。",
    )

    cache_store_group.add_argument(
        "--cache_store_rdma_transfer_wait_timeout_ms",
        env_name="CACHE_STORE_RDMA_TRANSFER_WAIT_TIMEOUT_MS",
        bind_to=(cache_store_config, "rdma_transfer_wait_timeout_ms"),
        type=int,
        default=180000,
        help="兼容旧启动配置，P2P 传输路径不使用此参数。",
    )

    cache_store_group.add_argument(
        "--p2p_read_steal_before_deadline_ms",
        env_name="P2P_READ_STEAL_BEFORE_DEADLINE_MS",
        bind_to=(cache_store_config, "p2p_read_steal_before_deadline_ms"),
        type=int,
        default=250,
        help="兼容旧启动配置，P2P 传输路径不使用此参数。",
    )

    cache_store_group.add_argument(
        "--p2p_read_return_before_deadline_ms",
        env_name="P2P_READ_RETURN_BEFORE_DEADLINE_MS",
        bind_to=(cache_store_config, "p2p_read_return_before_deadline_ms"),
        type=int,
        default=100,
        help="兼容旧启动配置，P2P 传输路径不使用此参数。",
    )
