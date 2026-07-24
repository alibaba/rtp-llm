from .util import str2bool


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
        help="Master 调度 per-request 默认超时（毫秒），用作 per-request 未传 timeout 时的默认值；"
        "<=0 表示不设超时（链路不超时）；per-request 显式 timeout 优先级更高",
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
        "--master_connector_limit_per_host",
        env_name="MASTER_CONNECTOR_LIMIT_PER_HOST",
        bind_to=(master_config, "master_connector_limit_per_host"),
        type=int,
        default=0,
        help="Max HTTP connections per master host (0 = use default 30)",
    )

    master_group.add_argument(
        "--master_session_timeout_s",
        env_name="MASTER_SESSION_TIMEOUT_S",
        bind_to=(master_config, "master_session_timeout_s"),
        type=float,
        default=-1,
        help="Master HTTP session total timeout (seconds). "
        "<0: auto (3600 when queue mode, 0.5 otherwise); "
        "==0: 不设超时（链路不超时）; >0: 使用该值",
    )

    master_group.add_argument(
        "--master_disable_domain_fallback",
        env_name="MASTER_DISABLE_DOMAIN_FALLBACK",
        bind_to=(master_config, "disable_domain_fallback"),
        type=str2bool,
        default=False,
        help="When True, disable domain fallback routing when master is unavailable or not configured. "
        "Requests will fail with ROUTE_ERROR instead of falling back to VipServer domain routing.",
    )

    master_group.add_argument(
        "--min_remaining_deadline_ms",
        env_name="MIN_REMAINING_DEADLINE_MS",
        bind_to=(master_config, "min_remaining_deadline_ms"),
        type=int,
        default=500,
        help="Minimum remaining deadline (ms) for absolute-deadline propagation. "
        "If remaining time falls below this threshold, the stage aborts before "
        "making a gRPC call. Default 500ms.",
    )
