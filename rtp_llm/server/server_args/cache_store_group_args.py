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
        "--cache_store_mooncake_mode",
        env_name="CACHE_STORE_MOONCAKE_MODE",
        bind_to=(cache_store_config, "cache_store_mooncake_mode"),
        type=str2bool,
        default=False,
        help="控制 cache store 是否使用 Mooncake P2P backend。",
    )
    cache_store_group.add_argument(
        "--cache_store_mooncake_metadata_conn_string",
        env_name="CACHE_STORE_MOONCAKE_METADATA_CONN_STRING",
        bind_to=(cache_store_config, "cache_store_mooncake_metadata_conn_string"),
        type=str,
        default="",
        help="Mooncake metadata backend 连接串。",
    )
    cache_store_group.add_argument(
        "--cache_store_mooncake_local_server_name",
        env_name="CACHE_STORE_MOONCAKE_LOCAL_SERVER_NAME",
        bind_to=(cache_store_config, "cache_store_mooncake_local_server_name"),
        type=str,
        default="",
        help="Mooncake classic TransferEngine 的本地 server name。",
    )
    cache_store_group.add_argument(
        "--cache_store_mooncake_ip_or_host_name",
        env_name="CACHE_STORE_MOONCAKE_IP_OR_HOST_NAME",
        bind_to=(cache_store_config, "cache_store_mooncake_ip_or_host_name"),
        type=str,
        default="",
        help="Mooncake classic TransferEngine 对外暴露的 ip 或 host 名称。",
    )
    cache_store_group.add_argument(
        "--cache_store_mooncake_transport",
        env_name="CACHE_STORE_MOONCAKE_TRANSPORT",
        bind_to=(cache_store_config, "cache_store_mooncake_transport"),
        type=str,
        default="tcp",
        help="Mooncake classic TransferEngine 传输协议，当前支持 tcp / rdma / barex / nvlink / nvlink_intra，兼容别名 nvlink_intraNode。",
    )
    cache_store_group.add_argument(
        "--cache_store_mooncake_rpc_port",
        env_name="CACHE_STORE_MOONCAKE_RPC_PORT",
        bind_to=(cache_store_config, "cache_store_mooncake_rpc_port"),
        type=int,
        default=12345,
        help="Mooncake classic TransferEngine 的 rpc/listen 端口。",
    )
    cache_store_group.add_argument(
        "--cache_store_mooncake_control_plane_port",
        env_name="CACHE_STORE_MOONCAKE_CONTROL_PLANE_PORT",
        bind_to=(cache_store_config, "cache_store_mooncake_control_plane_port"),
        type=int,
        default=0,
        help="Mooncake prepare/finish 控制面的监听端口，0 表示沿用 cache_store_listen_port。",
    )
    cache_store_group.add_argument(
        "--cache_store_mooncake_location",
        env_name="CACHE_STORE_MOONCAKE_LOCATION",
        bind_to=(cache_store_config, "cache_store_mooncake_location"),
        type=str,
        default="*",
        help="Mooncake backend location 标识。",
    )
    cache_store_group.add_argument(
        "--cache_store_mooncake_remote_accessible",
        env_name="CACHE_STORE_MOONCAKE_REMOTE_ACCESSIBLE",
        bind_to=(cache_store_config, "cache_store_mooncake_remote_accessible"),
        type=str2bool,
        default=True,
        help="Mooncake backend 是否对远端可访问。",
    )
    cache_store_group.add_argument(
        "--cache_store_mooncake_update_metadata",
        env_name="CACHE_STORE_MOONCAKE_UPDATE_METADATA",
        bind_to=(cache_store_config, "cache_store_mooncake_update_metadata"),
        type=str2bool,
        default=True,
        help="Mooncake backend 是否在初始化时更新 metadata。",
    )

    cache_store_group.add_argument(
        "--wrr_available_ratio",
        env_name="WRR_AVAILABLE_RATIO",
        bind_to=(cache_store_config, "wrr_available_ratio"),
        type=int,
        default=80,
        help="为 WRR (Weighted Round Robin) 负载均衡器设置的可用性阈值百分比 (0-100)，数值越低越容易启用动态权重分配，但可能降低全局负载均衡准确性。",
    )

    cache_store_group.add_argument(
        "--rank_factor",
        env_name="RANK_FACTOR",
        bind_to=(cache_store_config, "rank_factor"),
        type=int,
        default=0,
        choices=[0, 1],
        help="指定 WRR 负载均衡器使用的排序因子。`0` 表示基于 KV_CACHE 使用情况排序，`1` 表示基于正在处理的请求数 (ONFLIGHT_REQUESTS) 排序。",
    )

    cache_store_group.add_argument(
        "--cache_store_thread_count",
        env_name="CACHE_STORE_THREAD_COUNT",
        bind_to=(cache_store_config, "thread_count"),
        type=int,
        default=16,
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
        default=16,
        help="为 cache store P2P messager 通信层设置 worker 线程数量。",
    )
    cache_store_group.add_argument(
        "--cache_store_rdma_transfer_wait_timeout_ms",
        env_name="CACHE_STORE_RDMA_TRANSFER_WAIT_TIMEOUT_MS",
        bind_to=(cache_store_config, "rdma_transfer_wait_timeout_ms"),
        type=int,
        default=180000,
        help="RDMA 传输完成最大等待超时时间（毫秒），默认 180 秒。",
    )

    cache_store_group.add_argument(
        "--p2p_read_steal_before_deadline_ms",
        env_name="P2P_READ_STEAL_BEFORE_DEADLINE_MS",
        bind_to=(cache_store_config, "p2p_read_steal_before_deadline_ms"),
        type=int,
        default=250,
        help="Decode read：距 deadline 小于该毫秒数时从 recv store steal，阻止新 transfer 匹配。",
    )
    cache_store_group.add_argument(
        "--p2p_read_return_before_deadline_ms",
        env_name="P2P_READ_RETURN_BEFORE_DEADLINE_MS",
        bind_to=(cache_store_config, "p2p_read_return_before_deadline_ms"),
        type=int,
        default=100,
        help="Decode read 与 Prefill send：transfer 须在 deadline 前该毫秒数内完成（与对端对齐）。",
    )
    cache_store_group.add_argument(
        "--p2p_transfer_not_done_resource_hold_ms",
        env_name="P2P_TRANSFER_NOT_DONE_RESOURCE_HOLD_MS",
        bind_to=(cache_store_config, "p2p_transfer_not_done_resource_hold_ms"),
        type=int,
        default=10000,
        help="Scheduler：TRANSFER_NOT_DONE 后延迟 done 以保留显存安全窗口（毫秒）。",
    )

    cache_store_group.add_argument(
        "--p2p_resource_store_timeout_check_interval_ms",
        env_name="P2P_RESOURCE_STORE_TIMEOUT_CHECK_INTERVAL_MS",
        bind_to=(cache_store_config, "p2p_resource_store_timeout_check_interval_ms"),
        type=int,
        default=100,
        help="P2P decode 侧资源 store 周期扫描超时资源的间隔（毫秒）。",
    )
    cache_store_group.add_argument(
        "--p2p_layer_cache_buffer_store_timeout_ms",
        env_name="P2P_LAYER_CACHE_BUFFER_STORE_TIMEOUT_MS",
        bind_to=(cache_store_config, "p2p_layer_cache_buffer_store_timeout_ms"),
        type=int,
        default=100000,
        help="P2P LayerCacheBufferStore 条目保留时长（毫秒），默认 100s。",
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
