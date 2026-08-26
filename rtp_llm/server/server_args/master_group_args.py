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
        "--master_kvcm_fallback_enabled",
        env_name="MASTER_KVCM_FALLBACK_ENABLED",
        bind_to=(master_config, "master_kvcm_fallback_enabled"),
        type=str2bool,
        default=False,
        help="Query KVCM after FlexLB availability failure",
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
        "--master_kvcm_worker_grpc_port_override",
        env_name="MASTER_KVCM_WORKER_GRPC_PORT_OVERRIDE",
        bind_to=(master_config, "master_kvcm_worker_grpc_port_override"),
        type=int,
        default=0,
        help="Selected worker gRPC port; zero uses reported HTTP port + 1",
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
        "--master_flexlb_transport_timeout_ms",
        env_name="MASTER_FLEXLB_TRANSPORT_TIMEOUT_MS",
        bind_to=(master_config, "master_flexlb_transport_timeout_ms"),
        type=int,
        default=0,
        help="Per FlexLB attempt timeout; zero uses the request TTFT timeout",
    )
