from rtp_llm.server.server_args.util import str2bool


def init_master_group_args(parser, master_config):
    master_group = parser.add_argument_group("Master Configuration")

    master_group.add_argument(
        "--master_queue_reject_threshold",
        env_name="MASTER_QUEUE_REJECT_THRESHOLD",
        bind_to=(master_config, "master_queue_reject_threshold"),
        type=int,
        default=100000,
        help="Master queue reject threshold",
    )

    master_group.add_argument(
        "--master_default_timeout_ms",
        env_name="MASTER_DEFAULT_TIMEOUT_MS",
        bind_to=(master_config, "master_default_timeout_ms"),
        type=int,
        default=3600000,
        help="Master default timeout in milliseconds",
    )

    master_group.add_argument(
        "--master_max_connect_pool_size",
        env_name="MASTER_MAX_CONNECT_POOL_SIZE",
        bind_to=(master_config, "master_max_connect_pool_size"),
        type=int,
        default=100000,
        help="Master max connect pool size",
    )

    master_group.add_argument(
        "--master_session_timeout_s",
        env_name="MASTER_SESSION_TIMEOUT_S",
        bind_to=(master_config, "master_session_timeout_s"),
        type=float,
        default=-1,
        help="Master HTTP session total timeout in seconds. -1: auto (3600 when queue mode, 0.5 otherwise); else use this value.",
    )

    master_group.add_argument(
        "--master_client_fallback",
        env_name="MASTER_CLIENT_FALLBACK",
        bind_to=(master_config, "master_client_fallback"),
        type=str2bool,
        default=False,
        help="Enable the Master client fallback chain after FlexLB availability failure",
    )

    master_group.add_argument(
        "--master_kvcm_service_id",
        env_name="MASTER_KVCM_SERVICE_ID",
        bind_to=(master_config, "master_kvcm_service_id"),
        type=str,
        default="",
        help="KVCM bootstrap service id or local IP:port list",
    )

    master_group.add_argument(
        "--master_kvcm_bootstrap_port",
        env_name="MASTER_KVCM_BOOTSTRAP_PORT",
        bind_to=(master_config, "master_kvcm_bootstrap_port"),
        type=int,
        default=6381,
        help="KVCM GetClusterInfo bootstrap gRPC port",
    )

    master_group.add_argument(
        "--master_kvcm_instance_id",
        env_name="MASTER_KVCM_INSTANCE_ID",
        bind_to=(master_config, "master_kvcm_instance_id"),
        type=str,
        default="",
        help="Exact KVCM cache namespace/instance_id",
    )

    master_group.add_argument(
        "--master_kvcm_block_size",
        env_name="MASTER_KVCM_BLOCK_SIZE",
        bind_to=(master_config, "master_kvcm_block_size"),
        type=int,
        default=0,
        help="Token block size used for vLLM sha256_cbor hashing",
    )

    master_group.add_argument(
        "--master_kvcm_request_timeout_ms",
        env_name="MASTER_KVCM_REQUEST_TIMEOUT_MS",
        bind_to=(master_config, "master_kvcm_request_timeout_ms"),
        type=int,
        default=100,
        help="KVCM GetClusterInfo/GetHostCacheState deadline",
    )

    master_group.add_argument(
        "--master_client_fallback_worker_grpc_port_override",
        env_name="MASTER_CLIENT_FALLBACK_WORKER_GRPC_PORT_OVERRIDE",
        bind_to=(master_config, "master_client_fallback_worker_grpc_port_override"),
        type=int,
        default=0,
        help="Fallback-selected worker gRPC port; zero uses reported HTTP port + 1",
    )

    master_group.add_argument(
        "--master_client_fallback_worker_status_port",
        env_name="MASTER_CLIENT_FALLBACK_WORKER_STATUS_PORT",
        bind_to=(master_config, "master_client_fallback_worker_status_port"),
        type=int,
        default=0,
        help="WorkerStatus gRPC port; zero uses the selected worker route port",
    )

    master_group.add_argument(
        "--master_client_fallback_candidate_pool_size",
        env_name="MASTER_CLIENT_FALLBACK_CANDIDATE_POOL_SIZE",
        bind_to=(master_config, "master_client_fallback_candidate_pool_size"),
        type=int,
        default=3,
        help="Maximum WorkerStatus probes per fallback request, including KVCM hot workers",
    )

    master_group.add_argument(
        "--master_kvcm_hot_candidate_pool_size",
        env_name="MASTER_KVCM_HOT_CANDIDATE_POOL_SIZE",
        bind_to=(master_config, "master_kvcm_hot_candidate_pool_size"),
        type=int,
        default=2,
        help="Maximum cache-hit candidates retained from KVCM",
    )

    master_group.add_argument(
        "--master_client_fallback_cold_candidate_batch_size",
        env_name="MASTER_CLIENT_FALLBACK_COLD_CANDIDATE_BATCH_SIZE",
        bind_to=(
            master_config,
            "master_client_fallback_cold_candidate_batch_size",
        ),
        type=int,
        default=3,
        help="New service-discovered workers probed in each fallback round",
    )

    master_group.add_argument(
        "--master_client_fallback_worker_status_concurrency",
        env_name="MASTER_CLIENT_FALLBACK_WORKER_STATUS_CONCURRENCY",
        bind_to=(master_config, "master_client_fallback_worker_status_concurrency"),
        type=int,
        default=3,
        help="Maximum concurrent WorkerStatus RPCs across fallback requests in one process",
    )

    master_group.add_argument(
        "--master_client_fallback_worker_status_timeout_ms",
        env_name="MASTER_CLIENT_FALLBACK_WORKER_STATUS_TIMEOUT_MS",
        bind_to=(master_config, "master_client_fallback_worker_status_timeout_ms"),
        type=int,
        default=200,
        help="Per-worker WorkerStatus RPC deadline",
    )

    master_group.add_argument(
        "--master_client_fallback_prefill_queue_size_threshold",
        env_name="MASTER_CLIENT_FALLBACK_PREFILL_QUEUE_SIZE_THRESHOLD",
        bind_to=(master_config, "master_client_fallback_prefill_queue_size_threshold"),
        type=int,
        default=1024,
        help="Exclude workers whose waiting Prefill queue reaches this size",
    )

    master_group.add_argument(
        "--master_client_fallback_p2p_hit_discount",
        env_name="MASTER_CLIENT_FALLBACK_P2P_HIT_DISCOUNT",
        bind_to=(master_config, "master_client_fallback_p2p_hit_discount"),
        type=float,
        default=0.2,
        help="FlexLB-compatible discount applied to remote cache-hit blocks",
    )

    master_group.add_argument(
        "--master_client_fallback_cache_affinity_first_max_extra_work_tokens",
        env_name="MASTER_CLIENT_FALLBACK_CACHE_AFFINITY_FIRST_MAX_EXTRA_WORK_TOKENS",
        bind_to=(
            master_config,
            "master_client_fallback_cache_affinity_first_max_extra_work_tokens",
        ),
        type=int,
        default=0,
        help="Maximum extra estimated work allowed for the fallback cache leader",
    )

    master_group.add_argument(
        "--master_client_fallback_outstanding_uncached_tokens_threshold",
        env_name="MASTER_CLIENT_FALLBACK_OUTSTANDING_UNCACHED_TOKENS_THRESHOLD",
        bind_to=(
            master_config,
            "master_client_fallback_outstanding_uncached_tokens_threshold",
        ),
        type=int,
        default=0,
        help="Outstanding uncached-token guard; zero disables the guard",
    )

    master_group.add_argument(
        "--master_client_fallback_cache_affinity_first_min_hit_rate",
        env_name="MASTER_CLIENT_FALLBACK_CACHE_AFFINITY_FIRST_MIN_HIT_RATE",
        bind_to=(
            master_config,
            "master_client_fallback_cache_affinity_first_min_hit_rate",
        ),
        type=float,
        default=5.0,
        help="Minimum cache-hit percentage required to prefer the cache leader",
    )

    master_group.add_argument(
        "--master_kvcm_use_local",
        env_name="MASTER_KVCM_USE_LOCAL",
        bind_to=(master_config, "master_kvcm_use_local"),
        type=str2bool,
        default=False,
        help="Treat service id as a comma-separated local IP:port list",
    )

    master_group.add_argument(
        "--master_client_fallback_discovery_refresh_ms",
        env_name="MASTER_CLIENT_FALLBACK_DISCOVERY_REFRESH_MS",
        bind_to=(master_config, "master_client_fallback_discovery_refresh_ms"),
        type=int,
        default=1000,
        help="Background fallback service-discovery refresh interval",
    )

    master_group.add_argument(
        "--master_client_fallback_discovery_stale_ms",
        env_name="MASTER_CLIENT_FALLBACK_DISCOVERY_STALE_MS",
        bind_to=(master_config, "master_client_fallback_discovery_stale_ms"),
        type=int,
        default=5000,
        help="Last-known-good fallback discovery snapshot lifetime",
    )

    master_group.add_argument(
        "--master_client_fallback_flexlb_transport_timeout_ms",
        env_name="MASTER_CLIENT_FALLBACK_FLEXLB_TRANSPORT_TIMEOUT_MS",
        bind_to=(
            master_config,
            "master_client_fallback_flexlb_transport_timeout_ms",
        ),
        type=int,
        default=0,
        help="Per-FlexLB-attempt fallback trigger timeout; zero uses the request TTFT timeout",
    )
